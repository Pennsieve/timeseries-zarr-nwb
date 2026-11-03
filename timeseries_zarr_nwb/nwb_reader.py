"""Concrete NWB adapters implementing the channel-source protocols."""

import logging
import math
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import datetime
from importlib.metadata import version
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
from pynwb import NWBFile, TimeSeries
from pynwb.device import Device
from pynwb.ecephys import ElectricalSeries
from pynwb.epoch import TimeIntervals
from pynwb.event import EventsTable
from pynwb.misc import AnnotationSeries, Units
from timeseries_zarr import MAX_LABELS, UNIT_TO_UV, EventColumn, RecordingMeta
from timeseries_zarr import Run as SourceRun
from timeseries_zarr import __version__ as writer_version

from timeseries_zarr_nwb.nwb_events import (
    MICROSECONDS_PER_SECOND,
    NwbAnnotationSeriesSource,
    NwbEventsTableSource,
    NwbIntervalSource,
    container_provenance,
)
from timeseries_zarr_nwb.nwb_layout import (
    GRID_TOLERANCE_S,
    Layout,
    series_layout,
    stitch,
)

logger = logging.getLogger(__name__)

MAX_SERIES_RANK = 2
"""Highest data rank a series can have and still be read as channels."""

WINDOW_BYTES: Final = 64 * 2**20
"""Most bytes of float64 samples one window of a series source holds."""


def _layout(series: TimeSeries) -> Layout:
    """Return the layout of one series, reading timestamps in windows."""
    return series_layout(series, WINDOW_BYTES // np.dtype(np.float64).itemsize)


def _start_us(layout: Layout, reference_time: datetime) -> int:
    """Return wall-clock microseconds of sample 0, rounded to whole microseconds."""
    return _time_us(layout.start_s, reference_time)


def _time_us(start_s: float, reference_time: datetime) -> int:
    """Return start_s after reference_time as microseconds since the epoch."""
    return round(
        (reference_time.timestamp() + start_s) * MICROSECONDS_PER_SECOND
    )


def _runs_us(layout: Layout, reference_time: datetime) -> list[SourceRun]:
    """Return the runs of a layout with their start times in microseconds.

    A run that starts within GRID_TOLERANCE_S of the end of the run before it
    continues that run, so a recording without a pause is one run.
    """
    period_s = 1 / layout.rate_hz
    runs: list[SourceRun] = []
    end_s = math.nan
    for run in layout.runs:
        if runs and abs(run.start_s - end_s) <= GRID_TOLERANCE_S:
            runs[-1] = SourceRun(
                runs[-1].start_us, runs[-1].num_samples + run.rows
            )
        else:
            runs.append(
                SourceRun(_time_us(run.start_s, reference_time), run.rows)
            )
        end_s = run.start_s + run.rows * period_s
    return runs


def _read_column(
    series: TimeSeries, channel_index: int, start: int, stop: int
) -> npt.NDArray[np.float64]:
    """Return one channel's [start, stop) window as float64.

    A rank-1 series is its own single channel; a rank-2 series is indexed by
    column.
    """
    data = series.data
    window = (
        data[start:stop]
        if len(data.shape) == 1
        else data[start:stop, channel_index]
    )
    return np.asarray(window, dtype=np.float64)


def _uv_factor(series: TimeSeries) -> float | None:
    """Return the factor from the series unit to microvolts.

    A unit matches a symbol of UNIT_TO_UV as written, or a spelled-out name
    in any case. None means the unit is outside the volts family.
    """
    unit = str(series.unit)
    factor = UNIT_TO_UV.get(unit)
    if factor is None:
        factor = UNIT_TO_UV.get(unit.lower())
    return factor


def _scale(
    series: TimeSeries,
    window: npt.NDArray[np.float64],
    columns: range,
    uv_factor: float | None,
) -> npt.NDArray[np.float32]:
    """Return a window of stored values as float32 in the channel unit.

    window holds the stored values of the columns, rank 1 for one column or
    shaped (rows, columns) for several. Applies the series' conversion, the
    channel_conversion of each column when the series has one, the offset,
    and then uv_factor when it is not None.
    """
    scaled = window * float(series.conversion)
    channel_conversion = getattr(series, "channel_conversion", None)
    if channel_conversion is not None:
        scaled = scaled * np.asarray(
            channel_conversion[columns.start : columns.stop], dtype=np.float64
        )
    scaled = scaled + float(series.offset)
    if uv_factor is not None:
        scaled = scaled * uv_factor
    return scaled.astype(np.float32)


def _read_samples(
    layout: Layout, channel_index: int, start: int, stop: int
) -> npt.NDArray[np.float32]:
    """Return one channel's [start, stop) recorded samples as float32.

    The samples of the runs follow one another, and a pause between runs adds
    none. Each run is scaled by its own series. A range that ends past the
    last sample is cut at the last sample.
    """
    stop = min(stop, layout.num_samples)
    samples = np.full(max(stop - start, 0), np.nan, dtype=np.float32)
    columns = range(channel_index, channel_index + 1)
    first = bisect_right(layout.runs, start, key=lambda run: run.first_sample)
    for run in layout.runs[max(first - 1, 0) :]:
        if run.first_sample >= stop:
            break
        low = max(start, run.first_sample)
        high = min(stop, run.first_sample + run.rows)
        if low >= high:
            continue
        row = run.first_row + low - run.first_sample
        column = _read_column(run.series, channel_index, row, row + high - low)
        samples[low - start : high - start] = _scale(
            run.series, column, columns, _uv_factor(run.series)
        )
    return samples


def _channel_count(series: TimeSeries) -> int:
    """Return the number of channels a rank-1 or rank-2 series holds."""
    shape = series.data.shape
    return 1 if len(shape) == 1 else int(shape[1])


class NwbContinuousSource:
    """A continuous channel backed by one column of an NWB ElectricalSeries."""

    def __init__(
        self,
        electrical_series: ElectricalSeries,
        channel_index: int,
        reference_time: datetime,
        *,
        nwb_path: str | None = None,
        layout: Layout | None = None,
    ) -> None:
        """Bind one channel of an ElectricalSeries as a continuous source.

        channel_index selects the column this source exposes; a rank-1 series
        holds one channel at index 0. reference_time is the wall-clock instant
        the series' times count from. nwb_path is the series' path in the
        file, for the provenance. layout holds the runs of the series, and
        of the later segments of a stitched recording. None lays out the
        series alone. Raises ValueError for a unit outside
        the volts family, or for timestamps series_layout rejects.
        """
        self._layout = _layout(electrical_series) if layout is None else layout
        for segment in self._layout.series:
            if _uv_factor(segment) is None:
                unit = str(segment.unit)
                raise ValueError(f"unsupported ElectricalSeries unit: {unit!r}")
        self._series = electrical_series
        self._channel_index = channel_index
        self._reference_time = reference_time
        self._nwb_path = nwb_path

    @property
    def series_names(self) -> list[str]:
        """The names of the ElectricalSeries this channel reads, in time order."""
        return [str(segment.name) for segment in self._layout.series]

    @property
    def id(self) -> str:
        """The selected electrode's table id."""
        electrodes = self._series.electrodes
        row_index = electrodes.data[self._channel_index]
        return str(electrodes.table.id[row_index])

    @property
    def name(self) -> str:
        """The electrode's channel_name column, then its label, then the id."""
        electrodes = self._series.electrodes
        row_index = electrodes.data[self._channel_index]
        table = electrodes.table
        for column in ("channel_name", "label"):
            if column in table.colnames:
                return str(table[column][row_index])
        return self.id

    @property
    def unit(self) -> str:
        """Always "uV"; read_samples normalizes every series to microvolts."""
        return "uV"

    def provenance(self) -> dict[str, object]:
        """Return the series' neurodata type and path, filtering, and device.

        filtering is the series' description of the filters applied to it.
        device is the name of the device of the electrode's group. Each one is
        left out when the file does not record it. A channel stitched from
        several series adds segments, their names in time order.
        """
        provenance = container_provenance(self._series, self._nwb_path)
        if len(self._layout.series) > 1:
            provenance["segments"] = self.series_names
        if self._series.filtering:
            provenance["filtering"] = str(self._series.filtering)
        electrodes = self._series.electrodes
        table = electrodes.table
        if "group" in table.colnames:
            group = table["group"][electrodes.data[self._channel_index]]
            if group.device is not None:
                provenance["device"] = str(group.device.name)
        return provenance

    def rate_hz(self) -> float:
        """Return the sample rate of the channel's grid in hertz."""
        return self._layout.rate_hz

    def start_us(self) -> int:
        """Return the wall-clock microseconds of sample index 0.

        The reference time plus the start of the earliest segment, rounded to
        whole microseconds. It is the start of the first run.
        """
        return _start_us(self._layout, self._reference_time)

    def runs(self) -> list[SourceRun]:
        """Return the runs of the channel in time order."""
        return _runs_us(self._layout, self._reference_time)

    def num_samples(self) -> int:
        """Return the recorded samples of the channel, which a pause does not add to."""
        return self._layout.num_samples

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) sample window as float32 microvolts.

        Applies each series' affine scaling, then converts from the series'
        own unit to microvolts. Index i counts recorded samples, so a pause
        between runs adds no index. An empty range (stop <= start) yields a
        length-0 array.
        """
        return _read_samples(self._layout, self._channel_index, start, stop)


class NwbTimeSeriesSource:
    """A continuous channel backed by one column of a plain NWB TimeSeries.

    The samples keep the series' own unit unless that unit is in the volts
    family, in which case they are normalized to microvolts the way an
    ElectricalSeries is. A rank-1 series is one channel named after the series;
    a rank-2 series is one channel per column, named series/column.
    """

    def __init__(
        self,
        series: TimeSeries,
        channel_index: int,
        reference_time: datetime,
        *,
        nwb_path: str | None = None,
        layout: Layout | None = None,
    ) -> None:
        """Bind one channel of a TimeSeries as a continuous source.

        channel_index selects the column of a rank-2 series and is 0 for a
        rank-1 series. nwb_path is the series' path in the file, for the
        provenance. layout holds the runs of the series, and None lays them
        out here. Raises ValueError for data with a
        rank above 2, or for timestamps series_layout rejects.
        """
        rank = len(series.data.shape)
        if rank > MAX_SERIES_RANK:
            raise ValueError(
                f"unsupported TimeSeries shape: {series.name} has rank {rank}, "
                f"and a channel needs rank 1 or 2"
            )
        self._series = series
        self._channel_index = channel_index
        self._reference_time = reference_time
        self._nwb_path = nwb_path
        self._layout = _layout(series) if layout is None else layout
        self._unit = str(series.unit) if _uv_factor(series) is None else "uV"

    @property
    def series_names(self) -> list[str]:
        """The name of the TimeSeries this channel is a column of, in a list."""
        return [str(segment.name) for segment in self._layout.series]

    @property
    def id(self) -> str:
        """The series name, with the column index appended for a rank-2 series."""
        if len(self._series.data.shape) == 1:
            return str(self._series.name)
        return f"{self._series.name}/{self._channel_index}"

    @property
    def name(self) -> str:
        """The same value as id."""
        return self.id

    @property
    def unit(self) -> str:
        """The series unit, or "uV" when that unit is in the volts family."""
        return self._unit

    def provenance(self) -> dict[str, object]:
        """Return the neurodata type and the path of the series in the file."""
        return container_provenance(self._series, self._nwb_path)

    def rate_hz(self) -> float:
        """Return the sample rate of the channel's grid in hertz."""
        return self._layout.rate_hz

    def start_us(self) -> int:
        """Return the wall-clock microseconds of sample index 0."""
        return _start_us(self._layout, self._reference_time)

    def runs(self) -> list[SourceRun]:
        """Return the runs of the channel in time order."""
        return _runs_us(self._layout, self._reference_time)

    def num_samples(self) -> int:
        """Return the recorded samples of the channel, which a pause does not add to."""
        return self._layout.num_samples

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the [start, stop) sample window as float32 in the channel unit.

        Applies the series' conversion and offset, then the volts-to-microvolts
        factor when the unit is in the volts family. Index i counts recorded
        samples, so a pause between runs adds no index. An empty range
        (stop <= start) yields a length-0 array.
        """
        return _read_samples(self._layout, self._channel_index, start, stop)


type ContinuousSource = NwbContinuousSource | NwbTimeSeriesSource


def _window_rows(series: TimeSeries, channel_count: int) -> int:
    """Return the rows of one window over channel_count columns of a series.

    A window holds at most WINDOW_BYTES of float64 samples and at least one
    row. When the data is chunked and one chunk fits in that budget, a window
    is a whole number of chunk rows. No chunk is then read in two windows.
    """
    row_bytes = np.dtype(np.float64).itemsize * channel_count
    rows = max(1, WINDOW_BYTES // row_bytes)
    chunks = getattr(series.data, "chunks", None)
    if chunks and int(chunks[0]) <= rows:
        rows -= rows % int(chunks[0])
    return rows


class NwbSeriesSource:
    """Consecutive columns of a laid-out series, read together window by window.

    The layout holds one series, or the segments of a stitched recording,
    which share their columns. Each window reads every column of the range
    in one access to the data. The column of a channel in the windows holds
    the same values as its read_samples.
    """

    def __init__(
        self,
        layout: Layout,
        columns: range,
        channels: Sequence[ContinuousSource],
    ) -> None:
        """Bind a range of columns of a layout and the channels they back.

        columns is a contiguous, ascending range of column indexes, and is
        range(0, 1) for a rank-1 series. channels holds the channel source of
        each column of columns, in column order.
        """
        self._layout = layout
        self._columns = columns
        self._channels = list(channels)

    def channels(self) -> Sequence[ContinuousSource]:
        """Return the channel source of each column, in column order."""
        return list(self._channels)

    def iter_windows(self) -> Iterator[npt.NDArray[np.float32]]:
        """Yield each window as a float32 array of (rows, channels).

        The array has one column for each channel, in column order. The
        windows hold recorded rows only, so a pause between runs adds none. A
        layout with no samples yields no windows.
        """
        width = len(self._columns)
        for run in self._layout.runs:
            step = _window_rows(run.series, width)
            yield from self._run_windows(
                run.series, run.first_row, run.rows, step
            )

    def _run_windows(
        self, series: TimeSeries, first_row: int, rows: int, step: int
    ) -> Iterator[npt.NDArray[np.float32]]:
        """Yield the windows of rows of a series, cut at multiples of step."""
        data = series.data
        uv_factor = _uv_factor(series)
        start, stop_row = first_row, first_row + rows
        while start < stop_row:
            stop = min((start // step + 1) * step, stop_row)
            if len(data.shape) == 1:
                window = data[start:stop]
            else:
                window = data[
                    start:stop, self._columns.start : self._columns.stop
                ]
            scaled = _scale(
                series,
                np.asarray(window, dtype=np.float64),
                self._columns,
                uv_factor,
            )
            yield scaled.reshape(scaled.shape[0], len(self._columns))
            start = stop


def _series_sources(
    layout: Layout,
    channels: Sequence[ContinuousSource],
    max_channels: int | None,
) -> Iterator[NwbSeriesSource]:
    """Yield one series source per group of consecutive columns of a layout.

    channels holds the source of every column in column order. Each group
    holds at most max_channels columns, and None puts every column in one
    group. A series without columns yields nothing.
    """
    if not channels:
        return
    size = len(channels) if max_channels is None else max_channels
    for start in range(0, len(channels), size):
        stop = min(start + size, len(channels))
        yield NwbSeriesSource(layout, range(start, stop), channels[start:stop])


class NwbUnitSource:
    """A unit (spike) channel backed by an NWB Units table.

    Flattens the table's per-cluster rows into the per-event streams the
    bundle stores: all spikes merged into one timestamp series sorted
    ascending, each event labeled with its cluster's dense index in table row
    order (not the upstream unit id). The table's waveform_mean column, when it
    has one, becomes the per-label mean snippets.
    """

    def __init__(
        self,
        units: Units,
        waveform_rate_hz: float,
        reference_time: datetime,
        *,
        nwb_path: str | None = None,
    ) -> None:
        """Bind an NWB Units table as a flattened per-event spike source.

        waveform_rate_hz is the sample rate within a waveform; the table
        carries no rate of its own. reference_time places event timestamps
        in absolute microseconds. nwb_path is the table's path in the file,
        for the provenance. Raises ValueError if the table holds more
        than MAX_LABELS units, past the uint16 label range, or if the units'
        waveform_mean snippets differ in width.
        """
        unit_count = len(units)
        if unit_count > MAX_LABELS:
            raise ValueError(f"a unit channel holds at most {MAX_LABELS} units")

        self._id = str(units.name)
        self._provenance = container_provenance(units, nwb_path)
        self._rate_hz = float(waveform_rate_hz)
        self._label_names = [
            str(units["unit_name"][i])
            if "unit_name" in units.colnames
            else str(units.id[i])
            for i in range(unit_count)
        ]

        reference_s = reference_time.timestamp()
        times: list[npt.NDArray[np.float64]] = []
        labels: list[npt.NDArray[np.uint16]] = []
        for cluster_id in range(unit_count):
            spike_times: npt.NDArray[np.float64] = np.asarray(
                units.get_unit_spike_times(cluster_id), dtype=np.float64
            )
            times.append(spike_times)
            labels.append(
                np.full(spike_times.shape[0], cluster_id, dtype=np.uint16)
            )
        all_times = np.concatenate(times) if times else np.empty(0)
        order = np.argsort(all_times, kind="stable")
        self._events = (
            ((all_times[order] + reference_s) * MICROSECONDS_PER_SECOND)
            .round()
            .astype(np.int64)
        )
        self._labels = (
            np.concatenate(labels)[order]
            if labels
            else np.empty(0, dtype=np.uint16)
        )
        self._means: npt.NDArray[np.float32] | None = None
        if unit_count and "waveform_mean" in units.colnames:
            self._means = np.stack(
                [
                    np.asarray(units["waveform_mean"][i], dtype=np.float32)
                    for i in range(unit_count)
                ]
            )

    @property
    def id(self) -> str:
        """The Units table's name."""
        return self._id

    @property
    def name(self) -> str:
        """The Units table's name, the same value as id."""
        return self._id

    @property
    def unit(self) -> str | None:
        """Always None, because spikes have no measurement.

        waveform_mean carries no unit metadata, so its amplitudes are stored
        unscaled.
        """
        return None

    def provenance(self) -> dict[str, object]:
        """Return the neurodata type and the path of the table in the file."""
        return dict(self._provenance)

    def num_events(self) -> int:
        """Return the total number of spike events across all units."""
        return int(self._events.shape[0])

    def columns(self) -> frozenset[EventColumn]:
        """Return the optional columns: labels, and waveform_mean when present."""
        if not self._label_names:
            return frozenset()
        if self._means is None:
            return frozenset({"labels"})
        return frozenset({"labels", "waveform_mean"})

    def label_names(self) -> list[str] | None:
        """Return the unit_name of each unit, or its id, in table row order."""
        return list(self._label_names)

    def num_labels(self) -> int:
        """Return the number of units."""
        return len(self._label_names)

    def max_duration_us(self) -> int | None:
        """Return None, because spikes have no duration."""
        return None

    def text_width(self) -> int:
        """Return 0, because spikes carry no text."""
        return 0

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the half-open [start, stop) window of event timestamps.

        Absolute-microsecond int64. An empty range (stop <= start) yields a
        length-0 array.
        """
        return self._events[start:stop]

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the [start, stop) window of dense unit indices as uint16."""
        return self._labels[start:stop]

    def read_durations(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Raise ValueError, because spikes have no durations column."""
        raise ValueError("a unit channel has no durations column")

    def read_text(self, start: int, stop: int) -> list[str]:
        """Raise ValueError, because spikes have no text column."""
        raise ValueError("a unit channel has no text column")

    def read_channel_ids(self, start: int, stop: int) -> list[list[str]]:
        """Raise ValueError, because spikes have no channels column."""
        raise ValueError("a unit channel has no channels column")

    def waveform_mean(self) -> npt.NDArray[np.float32]:
        """Return the mean snippet of each unit, shaped (units, samples).

        Raises ValueError if the table has no waveform_mean column.
        """
        if self._means is None:
            raise ValueError("the Units table has no waveform_mean column")
        return self._means

    def waveform_period_us(self) -> float:
        """Return the microseconds between adjacent snippet samples.

        Raises ValueError if the waveform rate is not positive.
        """
        if self._rate_hz <= 0:
            raise ValueError(
                f"waveform_rate_hz must be positive, got {self._rate_hz!r}"
            )
        return MICROSECONDS_PER_SECOND / self._rate_hz


def _iter_units_tables(nwbfile: NWBFile) -> Iterator[tuple[str, Units]]:
    """Yield the path and table of every Units table in discovery order.

    The root Units table first when present, then the Units containers of each
    processing module, modules in name order and their containers in name
    order.
    """
    if nwbfile.units is not None:
        yield "/units", nwbfile.units
    for module_name in sorted(nwbfile.processing):
        module = nwbfile.processing[module_name]
        for container_name in sorted(module.data_interfaces):
            container = module.data_interfaces[container_name]
            if isinstance(container, Units):
                yield f"/processing/{module_name}/{container_name}", container


def _holds_channel_data(series: TimeSeries) -> bool:
    """Return whether a series holds numeric samples of rank 1 or 2."""
    data = series.data
    shape = getattr(data, "shape", None)
    dtype = getattr(data, "dtype", None)
    if shape is None or dtype is None:
        array = np.asarray(data)
        shape, dtype = array.shape, array.dtype
    return 1 <= len(shape) <= MAX_SERIES_RANK and bool(
        np.issubdtype(dtype, np.number)
    )


def _iter_plain_series(nwbfile: NWBFile) -> Iterator[TimeSeries]:
    """Yield the acquisition TimeSeries that are channels but not ElectricalSeries.

    A series whose data is not numeric samples of rank 1 or 2 is skipped with a
    warning. Order is the acquisition's insertion order.
    """
    for acq in nwbfile.acquisition.values():
        if isinstance(
            acq, ElectricalSeries | AnnotationSeries
        ) or not isinstance(acq, TimeSeries):
            continue
        if not _holds_channel_data(acq):
            logger.warning(
                "skipping acquisition %s: its data is not numeric samples of "
                "rank 1 or 2",
                acq.name,
            )
            continue
        yield acq


type EventSource = (
    NwbUnitSource
    | NwbIntervalSource
    | NwbEventsTableSource
    | NwbAnnotationSeriesSource
)
"""Any source that discovery turns an NWB table or series into an event channel."""


class NwbSources(NamedTuple):
    """The sources and the recording metadata discovered in one NWB file."""

    continuous: list[NwbSeriesSource]
    events: list[EventSource]
    recording: RecordingMeta


def _require_unique_ids(
    sources: Sequence[ContinuousSource | EventSource],
) -> None:
    """Raise ValueError when two sources share a channel id."""
    seen: set[str] = set()
    for source in sources:
        if source.id in seen:
            raise ValueError(
                f"duplicate channel id {source.id!r}: every ElectricalSeries "
                "electrode id, TimeSeries name, and event table name must be "
                "distinct"
            )
        seen.add(source.id)


def _iter_interval_tables(
    nwbfile: NWBFile,
) -> Iterator[tuple[str, TimeIntervals]]:
    """Yield the path and table of every TimeIntervals table once, in order.

    The epochs, trials, and invalid times tables come first, in that order,
    then the file's other interval tables in name order.
    """
    seen: set[int] = set()
    named = [nwbfile.epochs, nwbfile.trials, nwbfile.invalid_times]
    named.extend(nwbfile.intervals[name] for name in sorted(nwbfile.intervals))
    for table in named:
        if table is not None and id(table) not in seen:
            seen.add(id(table))
            yield f"/intervals/{table.name}", table


def _iter_events_tables(
    nwbfile: NWBFile,
) -> Iterator[tuple[str, EventsTable]]:
    """Yield the path and table of the file's EventsTable tables in name order."""
    tables = getattr(nwbfile, "events", None) or {}
    for name in sorted(tables):
        yield f"/events/{name}", tables[name]


def _iter_annotation_series(
    nwbfile: NWBFile,
) -> Iterator[tuple[str, AnnotationSeries]]:
    """Yield the path and series of each acquisition AnnotationSeries in order."""
    for name, acq in nwbfile.acquisition.items():
        if isinstance(acq, AnnotationSeries):
            yield f"/acquisition/{name}", acq


def _non_empty[T](
    tables: Iterable[tuple[str, T]], size: Callable[[T], int]
) -> Iterator[tuple[str, T]]:
    """Yield the paths and tables that hold something, logging the others."""
    for path, table in tables:
        if size(table):
            yield path, table
        else:
            logger.info("skipping empty %s", path)


def _build_events(
    nwbfile: NWBFile,
    waveform_rate_hz: float | None,
    continuous: Sequence[ContinuousSource],
) -> list[EventSource]:
    """Build one event source per non-empty table or annotation series.

    waveform_rate_hz is the rate of the first ElectricalSeries, or None when
    the file has none.
    """
    reference_time = nwbfile.timestamps_reference_time
    channel_ids_by_series: dict[str, list[str]] = {}
    for source in continuous:
        for name in source.series_names:
            channel_ids_by_series.setdefault(name, []).append(source.id)

    events: list[EventSource] = []
    for path, units in _non_empty(_iter_units_tables(nwbfile), len):
        if waveform_rate_hz is None:
            raise ValueError("unit channels need an ElectricalSeries rate")
        events.append(
            NwbUnitSource(
                units, waveform_rate_hz, reference_time, nwb_path=path
            )
        )
    events.extend(
        NwbIntervalSource(
            table, reference_time, channel_ids_by_series, nwb_path=path
        )
        for path, table in _non_empty(_iter_interval_tables(nwbfile), len)
    )
    events.extend(
        NwbEventsTableSource(table, reference_time, nwb_path=path)
        for path, table in _non_empty(_iter_events_tables(nwbfile), len)
    )
    events.extend(
        NwbAnnotationSeriesSource(annotations, reference_time, nwb_path=path)
        for path, annotations in _non_empty(
            _iter_annotation_series(nwbfile), lambda a: len(a.data)
        )
    )
    return events


def _electrode_ids(series: ElectricalSeries) -> tuple[str, ...]:
    """Return the electrode table id of each column of a series, in order."""
    electrodes = series.electrodes
    return tuple(str(electrodes.table.id[row]) for row in electrodes.data[:])


def _segment_groups(
    series: Sequence[ElectricalSeries],
) -> list[list[ElectricalSeries]]:
    """Return the series grouped into recordings, in order of first member.

    Series with the same electrode ids in the same order are the segments of
    one recording. A series without electrodes is a group of its own.
    Raises ValueError if two series share an electrode id but not the whole
    list of ids.
    """
    groups: list[list[ElectricalSeries]] = []
    group_of: dict[tuple[str, ...], list[ElectricalSeries]] = {}
    owner: dict[str, tuple[str, ...]] = {}
    for es in series:
        ids = _electrode_ids(es)
        for electrode in ids:
            key = owner.setdefault(electrode, ids)
            if key != ids:
                raise ValueError(
                    f"{es.name} and {group_of[key][0].name} share electrode id "
                    f"{electrode!r} but not their electrode list; the segments "
                    "of one recording list the same electrodes in the same order"
                )
        if ids and ids in group_of:
            group_of[ids].append(es)
        else:
            group_of[ids] = [es]
            groups.append(group_of[ids])
    return groups


def build_sources_from_nwb(
    nwbfile: NWBFile,
    *,
    include_identifier: bool = False,
    max_sweep_channels: int | None = None,
) -> NwbSources:
    """Discover the channels and recording metadata to write from an NWB file.

    Every ElectricalSeries in the file's acquisition contributes one channel
    per column, in series order then column order. ElectricalSeries with the
    same electrode ids in the same order are the segments of one recording:
    they give one channel per electrode at the position of the first of them,
    with the segments in time order as the runs of the channel. Every other
    acquisition TimeSeries holding numeric samples of rank 1 or 2 follows, one
    channel per column in the same order. A series sampled by timestamps takes
    its rate from them, and each pause between them starts a run. continuous
    holds one multi-channel source per group of at most max_sweep_channels
    consecutive columns of each recording or series, and None puts every
    column in one group. The grouping leaves the channel order unchanged.
    Event sources follow, one per non-empty table or series,
    in this order: Units tables (see _iter_units_tables, waveform rate from
    the first ElectricalSeries), the epochs, trials, and invalid times
    tables, the other interval tables by name, EventsTable tables by name, and
    acquisition AnnotationSeries in insertion order. A table's timeseries
    references resolve to the ids of the channels of that series, or of the
    recording it is a segment of. Each source's provenance names the path of
    its series or table in the file. Every source counts its times from the
    file's timestamps_reference_time, which recording names as the bundle's
    reference time. recording is the identity recording_meta reads, and
    include_identifier passes to it. Raises ValueError if a non-empty Units
    table is present and no ElectricalSeries supplies the rate, if
    series_layout rejects the timestamps of a series, if stitch rejects the
    segments of a recording, if two ElectricalSeries share some electrode ids
    but not all, if an ElectricalSeries unit is outside the volts family, if
    two sources share an id, or if max_sweep_channels is below 1.
    """
    if max_sweep_channels is not None and max_sweep_channels < 1:
        raise ValueError(
            f"max_sweep_channels must be 1 or more, got {max_sweep_channels}"
        )
    reference_time = nwbfile.timestamps_reference_time
    series = [
        acq
        for acq in nwbfile.acquisition.values()
        if isinstance(acq, ElectricalSeries)
    ]
    continuous: list[NwbSeriesSource] = []
    waveform_rate_hz: float | None = None
    for group in _segment_groups(series):
        layout = stitch([_layout(es) for es in group])
        if waveform_rate_hz is None:
            waveform_rate_hz = layout.rate_hz
        first = layout.series[0]
        path = f"/acquisition/{first.name}"
        electrodes = [
            NwbContinuousSource(
                first, index, reference_time, nwb_path=path, layout=layout
            )
            for index in range(_channel_count(first))
        ]
        continuous.extend(
            _series_sources(layout, electrodes, max_sweep_channels)
        )
    for plain in _iter_plain_series(nwbfile):
        path = f"/acquisition/{plain.name}"
        layout = _layout(plain)
        columns = [
            NwbTimeSeriesSource(
                plain, index, reference_time, nwb_path=path, layout=layout
            )
            for index in range(_channel_count(plain))
        ]
        continuous.extend(_series_sources(layout, columns, max_sweep_channels))
    channels = [
        channel for source in continuous for channel in source.channels()
    ]
    events = _build_events(nwbfile, waveform_rate_hz, channels)
    _require_unique_ids([*channels, *events])
    return NwbSources(
        continuous,
        events,
        recording_meta(nwbfile, include_identifier=include_identifier),
    )


_SUBJECT_FIELDS: Final = (
    ("subject_id", "id"),
    ("species", "species"),
    ("sex", "sex"),
    ("age", "age"),
    ("description", "description"),
)
"""NWB Subject fields carried into recording.json, with their keys there."""


def _nwb_format(nwbfile: NWBFile) -> str:
    """Return "NWB" followed by the schema version of the file it was read from.

    A file that was not read from disk, or that records no version, gives
    "NWB" alone.
    """
    io = nwbfile.get_read_io()
    schema_version = None if io is None else io.nwb_version[0]
    return "NWB" if schema_version is None else f"NWB {schema_version}"


def _generated_by(nwbfile: NWBFile) -> list[str]:
    """Return each "name version" pair of the file's was_generated_by field."""
    if nwbfile.was_generated_by is None:
        return []
    return [
        " ".join(str(part) for part in entry)
        for entry in nwbfile.was_generated_by[:]
    ]


def _device_entry(device: Device) -> dict[str, object]:
    """Return the name, manufacturer, model, and model number of a device.

    The device's model, when it has one, supplies the last three. Each field
    is left out when it is unset. The serial number and the free-text
    description are not read.
    """
    model = device.model
    fields = (
        ("name", device.name),
        (
            "manufacturer",
            device.manufacturer if model is None else model.manufacturer,
        ),
        ("model", device.model_name if model is None else model.name),
        (
            "model_number",
            device.model_number if model is None else model.model_number,
        ),
    )
    return {key: str(value) for key, value in fields if value}


def _recording_source(
    nwbfile: NWBFile, *, include_identifier: bool
) -> dict[str, object]:
    """Return the source object of recording.json; see recording_meta."""
    source: dict[str, object] = {
        "converter": f"timeseries-zarr-nwb {version('timeseries-zarr-nwb')}",
        "writer": f"timeseries-zarr {writer_version}",
        "format": _nwb_format(nwbfile),
    }
    generated_by = _generated_by(nwbfile)
    if generated_by:
        source["generated_by"] = generated_by
    if nwbfile.devices:
        source["devices"] = [
            _device_entry(nwbfile.devices[name])
            for name in sorted(nwbfile.devices)
        ]
    if include_identifier:
        source["identifier"] = str(nwbfile.identifier)
    return source


def recording_meta(
    nwbfile: NWBFile, *, include_identifier: bool = False
) -> RecordingMeta:
    """Return the recording.json content an NWB file supplies.

    session_start_time is the file's session start time. reference_time is the
    file's timestamps_reference_time, the instant every source counts its times
    from. Bundle offsets then equal NWB seconds as whole microseconds. subject
    carries the id, species, sex, age, description, and date of birth (as ISO
    8601) of the file's Subject, and session carries the session id and
    description, each only when set. source names this converter, the writer,
    and the NWB format with their versions. It lists the software in the file's
    was_generated_by field under generated_by and the file's devices in name
    order under devices, each only when the file holds one. include_identifier
    adds the file identifier to source. Nothing else identifying is read: the
    experimenter, the institution, the source script, and the serial numbers of
    devices stay out of the bundle.
    """
    subject: dict[str, object] = {}
    if nwbfile.subject is not None:
        for field, key in _SUBJECT_FIELDS:
            value = getattr(nwbfile.subject, field, None)
            if value:
                subject[key] = str(value)
        born = getattr(nwbfile.subject, "date_of_birth", None)
        if born is not None:
            subject["date_of_birth"] = born.isoformat()
    session: dict[str, object] = {}
    if nwbfile.session_id:
        session["id"] = str(nwbfile.session_id)
    if nwbfile.session_description:
        session["description"] = str(nwbfile.session_description)
    return RecordingMeta(
        session_start_time=nwbfile.session_start_time,
        reference_time=nwbfile.timestamps_reference_time,
        subject=subject,
        session=session,
        source=_recording_source(
            nwbfile, include_identifier=include_identifier
        ),
    )
