"""NWB adapters presenting interval and event tables as event channels."""

import json
import logging
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final

import numpy as np
import numpy.typing as npt
from pynwb import TimeSeries
from pynwb.epoch import TimeIntervals
from pynwb.event import EventsTable
from pynwb.misc import AnnotationSeries, Units
from timeseries_zarr import MAX_LABELS, EventColumn, text_width

logger = logging.getLogger(__name__)

INTERVAL_COLUMNS: Final = frozenset(
    {"start_time", "stop_time", "tags", "timeseries"}
)
"""Columns of a TimeIntervals table that are not free-form text."""

UNTAGGED: Final = "untagged"
"""Label name of the category of rows that carry no tags."""

MICROSECONDS_PER_SECOND: Final = 1_000_000.0
"""Microseconds in one second, for converting NWB seconds."""


def _json_value(value: object) -> object:
    """Return value as something json can write, with NaN and infinity as null."""
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _as_text(value: object) -> str:
    """Return a table or series string as text, decoding bytes as UTF-8."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def container_provenance(
    container: TimeSeries | TimeIntervals | EventsTable | Units,
    nwb_path: str | None,
) -> dict[str, object]:
    """Return the provenance of a channel read from one NWB object.

    It holds the neurodata type of the object and, when nwb_path is not None,
    the path of the object in the file, such as /acquisition/ElectricalSeries.
    """
    provenance: dict[str, object] = {
        "neurodata_type": str(container.neurodata_type)
    }
    if nwb_path is not None:
        provenance["nwb_path"] = nwb_path
    return provenance


def _to_microseconds(
    seconds: object, what: str, table_name: str
) -> npt.NDArray[np.int64]:
    """Return seconds as whole microseconds, raising if any is not finite."""
    values = np.asarray(seconds, dtype=np.float64)
    bad = ~np.isfinite(values)
    if bad.any():
        row = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"row {row} of {table_name} has a {what} that is not a finite number"
        )
    return np.rint(values * MICROSECONDS_PER_SECOND).astype(np.int64)


def _read_labels(
    table: TimeIntervals, rows: int, order: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.uint16] | None, list[str]]:
    """Return the sorted rows' label indices and the category names."""
    if "tags" not in table.colnames:
        return None, []
    tuples = [
        tuple(sorted(str(tag) for tag in table["tags"][row]))
        for row in range(rows)
    ]
    categories = sorted(set(tuples))
    if categories in ([], [()]):
        return None, []
    if len(categories) > MAX_LABELS:
        raise ValueError(
            f"{len(categories)} tag combinations exceed {MAX_LABELS} labels"
        )
    index = {category: i for i, category in enumerate(categories)}
    labels = np.array([index[t] for t in tuples], dtype=np.uint16)[order]
    names = [",".join(c) if c else UNTAGGED for c in categories]
    return labels, names


def _read_texts(
    table: TimeIntervals, rows: int, order: npt.NDArray[np.intp]
) -> list[str] | None:
    """Return one JSON object of the free-form columns per sorted row."""
    extra = [c for c in table.colnames if c not in INTERVAL_COLUMNS]
    if not extra or not rows:
        return None
    texts = [
        json.dumps(
            {c: _json_value(table[c][row]) for c in extra},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for row in range(rows)
    ]
    return [texts[i] for i in order]


def _read_channels(
    table: TimeIntervals,
    rows: int,
    order: npt.NDArray[np.intp],
    channel_ids_by_series: Mapping[str, Sequence[str]],
) -> list[list[str]] | None:
    """Return the channel ids each sorted row applies to, or None."""
    if "timeseries" not in table.colnames:
        return None
    unresolved = 0
    per_row: list[list[str]] = []
    for row in range(rows):
        ids: list[str] = []
        for reference in table["timeseries"][row]:
            series = reference.timeseries
            if series is None:
                continue
            if series.name in channel_ids_by_series:
                ids.extend(channel_ids_by_series[series.name])
            else:
                unresolved += 1
        per_row.append(list(dict.fromkeys(ids)))
    if unresolved:
        logger.warning(
            "%s: %d timeseries references name a series with no channel "
            "in the bundle and are ignored",
            table.name,
            unresolved,
        )
    if not any(per_row):
        return None
    return [per_row[i] for i in order]


class _EventArrays:
    """An event channel whose columns are held in memory, in time order.

    The table sources below build the arrays from their NWB table and share
    this behavior. A column the source lacks is None and is not offered.
    """

    def __init__(
        self,
        id: str,
        provenance: dict[str, object],
        reference_us: int,
        events: npt.NDArray[np.int64],
        durations: npt.NDArray[np.int64] | None = None,
        labels: npt.NDArray[np.uint16] | None = None,
        label_names: list[str] | None = None,
        texts: list[str] | None = None,
        channel_ids: list[list[str]] | None = None,
    ) -> None:
        """Hold the arrays, which must already be sorted by event time."""
        self._id = id
        self._provenance = provenance
        self._reference_us = reference_us
        self._events = events
        self._durations = durations
        self._labels = labels
        self._label_names = label_names or []
        self._texts = texts
        self._channel_ids = channel_ids

    @property
    def id(self) -> str:
        """The table's name."""
        return self._id

    @property
    def name(self) -> str:
        """The table's name, the same value as id."""
        return self._id

    @property
    def unit(self) -> str | None:
        """Always None: these events have no measurement."""
        return None

    def provenance(self) -> dict[str, object]:
        """Return the neurodata type and the path of the table in the file."""
        return dict(self._provenance)

    def start_us(self) -> int:
        """Return the earlier of the reference time and the first event."""
        if self._events.shape[0]:
            return min(self._reference_us, int(self._events[0]))
        return self._reference_us

    def num_events(self) -> int:
        """Return the number of events."""
        return int(self._events.shape[0])

    def columns(self) -> frozenset[EventColumn]:
        """Return the optional columns the table provides.

        A table with no events offers none.
        """
        if not self.num_events():
            return frozenset()
        present: set[EventColumn] = set()
        if self._durations is not None:
            present.add("durations")
        if self._labels is not None:
            present.add("labels")
        if self._texts is not None:
            present.add("text")
        if self._channel_ids is not None:
            present.add("channels")
        return frozenset(present)

    def label_names(self) -> list[str] | None:
        """Return the joined tags of each category, or None without labels."""
        return list(self._label_names) if self._labels is not None else None

    def num_labels(self) -> int:
        """Return the number of label categories, or 0 without labels."""
        return len(self._label_names)

    def max_duration_us(self) -> int | None:
        """Return the longest duration, or None without durations or events."""
        if self._durations is None or not self.num_events():
            return None
        return int(self._durations.max())

    def text_width(self) -> int:
        """Return the byte width of the longest text, or 0 without text."""
        return 0 if self._texts is None else text_width(self._texts)

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of event times in microseconds."""
        return self._events[start:stop]

    def read_durations(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of durations in microseconds.

        Raises ValueError if the table has no durations.
        """
        if self._durations is None:
            raise ValueError("the table has no durations column")
        return self._durations[start:stop]

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the [start, stop) window of label indices.

        Raises ValueError if the table has no tags.
        """
        if self._labels is None:
            raise ValueError("the table has no tags, so no labels column")
        return self._labels[start:stop]

    def read_text(self, start: int, stop: int) -> list[str]:
        """Return the [start, stop) window of texts.

        Raises ValueError if the table has no text.
        """
        if self._texts is None:
            raise ValueError("the table has no text column")
        return self._texts[start:stop]

    def read_channel_ids(self, start: int, stop: int) -> list[list[str]]:
        """Return the channel ids of each event in [start, stop).

        Raises ValueError if no row references a bundled series.
        """
        if self._channel_ids is None:
            raise ValueError("no row references a channel, so no channels")
        return self._channel_ids[start:stop]

    def waveform_mean(self) -> npt.NDArray[np.float32]:
        """Raise ValueError, because these events have no snippets."""
        raise ValueError("the table has no waveform_mean column")

    def waveform_period_us(self) -> float:
        """Raise ValueError, because these events have no snippets."""
        raise ValueError("the table has no waveform_mean column")


class NwbIntervalSource(_EventArrays):
    """An event channel backed by one NWB TimeIntervals table.

    Each row is one event placed at its start_time, with its length as the
    duration. The rows are sorted by start time, stably. The table's tags
    become label categories, its free-form columns become one JSON object of
    text per row, and its timeseries references become the channels the event
    applies to.
    """

    def __init__(
        self,
        table: TimeIntervals,
        reference_time: datetime,
        channel_ids_by_series: Mapping[str, Sequence[str]],
        *,
        nwb_path: str | None = None,
    ) -> None:
        """Bind a TimeIntervals table as an event channel.

        reference_time places the table's seconds, which count from it, in
        absolute microseconds. channel_ids_by_series maps the name of each
        series the bundle holds to the ids of that series' channels. A
        reference to a series the mapping lacks is ignored with a warning.
        nwb_path is the table's path in the file, for the provenance.
        Each distinct tag tuple, with its tags sorted, is one label category,
        in sorted order, and a row without tags is the category "untagged". A
        table with no tags in any row has no labels. Raises ValueError if a
        time is not finite, a row stops before it starts, or the tag tuples
        outnumber MAX_LABELS.
        """
        name = str(table.name)
        reference_us = round(
            reference_time.timestamp() * MICROSECONDS_PER_SECOND
        )
        starts = np.asarray(table["start_time"][:], dtype=np.float64)
        stops = np.asarray(table["stop_time"][:], dtype=np.float64)
        offsets = _to_microseconds(starts, "start_time", name)
        durations = _to_microseconds(stops - starts, "stop_time", name)
        if (durations < 0).any():
            row = int(np.flatnonzero(durations < 0)[0])
            raise ValueError(f"row {row} of {name} stops before it starts")
        order = np.argsort(starts, kind="stable")
        rows = len(starts)
        labels, label_names = _read_labels(table, rows, order)
        super().__init__(
            name,
            container_provenance(table, nwb_path),
            reference_us,
            (reference_us + offsets)[order],
            durations[order],
            labels,
            label_names,
            _read_texts(table, rows, order),
            _read_channels(table, rows, order, channel_ids_by_series),
        )


class NwbEventsTableSource(_EventArrays):
    """An event channel backed by one NWB EventsTable.

    Each row is one event at its timestamp. The duration column, when the table
    has one, gives the durations, and the annotation column, when it has one,
    gives the text as written. Other columns are not carried.
    """

    def __init__(
        self,
        table: EventsTable,
        reference_time: datetime,
        *,
        nwb_path: str | None = None,
    ) -> None:
        """Bind an EventsTable as an event channel.

        reference_time places the table's seconds, which count from it, in
        absolute microseconds. nwb_path is the table's path in the file, for
        the provenance. Columns other than timestamp, duration, and
        annotation are ignored with a warning. Raises ValueError if a
        timestamp or duration is not finite or a duration is negative.
        """
        name = str(table.name)
        reference_us = round(
            reference_time.timestamp() * MICROSECONDS_PER_SECOND
        )
        stamps = np.asarray(table["timestamp"][:], dtype=np.float64)
        offsets = _to_microseconds(stamps, "timestamp", name)
        order = np.argsort(stamps, kind="stable")
        durations = None
        if "duration" in table.colnames:
            lengths = _to_microseconds(table["duration"][:], "duration", name)
            if (lengths < 0).any():
                row = int(np.flatnonzero(lengths < 0)[0])
                raise ValueError(f"row {row} of {name} has a negative duration")
            durations = lengths[order]
        texts = None
        if "annotation" in table.colnames and len(stamps):
            notes = [_as_text(note) for note in table["annotation"][:]]
            texts = [notes[i] for i in order]
        ignored = [
            c
            for c in table.colnames
            if c not in {"timestamp", "duration", "annotation"}
        ]
        if ignored:
            logger.warning("%s: columns %s are not carried", name, ignored)
        super().__init__(
            name,
            container_provenance(table, nwb_path),
            reference_us,
            (reference_us + offsets)[order],
            durations,
            texts=texts,
        )


class NwbAnnotationSeriesSource(_EventArrays):
    """An event channel backed by one NWB AnnotationSeries.

    Each sample is one event at its timestamp, and its string is the text.
    The events have no duration. pynwb deprecates the AnnotationSeries in
    favor of the EventsTable, but files still hold it.
    """

    def __init__(
        self,
        series: AnnotationSeries,
        reference_time: datetime,
        *,
        nwb_path: str | None = None,
    ) -> None:
        """Bind an AnnotationSeries as an event channel.

        reference_time places the series' timestamps, which count from it
        in seconds, in absolute microseconds. nwb_path is the series' path in
        the file, for the provenance. Raises ValueError if the series
        has no timestamps, its data and timestamps differ in length, or a
        timestamp is not finite.
        """
        name = str(series.name)
        if series.timestamps is None:
            raise ValueError(f"{name} needs timestamps, not a sampling rate")
        stamps = np.asarray(series.timestamps[:], dtype=np.float64)
        notes = [_as_text(note) for note in series.data[:]]
        if len(notes) != len(stamps):
            raise ValueError(
                f"{name} has {len(notes)} texts but {len(stamps)} timestamps"
            )
        reference_us = round(
            reference_time.timestamp() * MICROSECONDS_PER_SECOND
        )
        offsets = _to_microseconds(stamps, "timestamp", name)
        order = np.argsort(stamps, kind="stable")
        super().__init__(
            name,
            container_provenance(series, nwb_path),
            reference_us,
            (reference_us + offsets)[order],
            texts=[notes[i] for i in order] if len(stamps) else None,
        )
