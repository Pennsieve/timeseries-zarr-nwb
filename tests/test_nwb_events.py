import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from pynwb.epoch import TimeIntervals
from pynwb.event import EventsTable
from pynwb.misc import AnnotationSeries
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from pynwb.testing.mock.file import mock_NWBFile
from timeseries_zarr import EventChannelSource

from timeseries_zarr_nwb.nwb_events import (
    UNTAGGED,
    NwbAnnotationSeriesSource,
    NwbEventsTableSource,
    NwbIntervalSource,
    _as_text,
    _json_value,
)

STARTED = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
STARTED_US = round(STARTED.timestamp() * 1_000_000)
_IDS = {"es": ["LA1", "LA2", "LA3"]}


def _epochs(rows):
    """Build an epochs table from (start, stop, tags, with_series) rows.

    The first row with a series reference creates the table; rows without one
    are added through the table so their reference list stays empty.
    """
    nwb = mock_NWBFile()
    es = mock_ElectricalSeries(
        nwbfile=nwb, name="es", data=np.zeros((100, 3)), rate=1000.0
    )
    for start, stop, tags, with_series in rows:
        if with_series:
            nwb.add_epoch(
                start_time=start,
                stop_time=stop,
                tags=tags,
                timeseries=[es],
            )
        elif nwb.epochs is None:
            nwb.add_epoch(start_time=start, stop_time=stop, tags=tags)
        else:
            nwb.epochs.add_row(
                start_time=start,
                stop_time=stop,
                tags=tags,
                timeseries=[],
            )
    return nwb.epochs


def _table(rows):
    """Build a TimeIntervals table from (start, stop, tags) rows.

    A tags value of None leaves the tags column out of the table.
    """
    table = TimeIntervals(name="epochs", description="d")
    for start, stop, tags in rows:
        extra = {} if tags is None else {"tags": tags}
        table.add_row(start_time=start, stop_time=stop, **extra)
    return table


def _source(table, ids=_IDS):
    return NwbIntervalSource(table, STARTED, ids)


def test_events_are_reference_time_plus_the_start_time_sorted_stably():
    table = _table(
        [
            (2.0, 2.5, ["b"]),
            (0.5, 0.75, ["a"]),
            (2.0, 3.0, []),
        ]
    )
    src = _source(table)
    assert src.read_events(0, 3).tolist() == [
        STARTED_US + 500_000,
        STARTED_US + 2_000_000,
        STARTED_US + 2_000_000,
    ]
    assert src.read_durations(0, 3).tolist() == [250_000, 500_000, 1_000_000]


def test_event_times_are_exact_to_the_microsecond():
    table = _epochs([(0.1234567, 0.1234568, [], False)])
    src = _source(table)
    assert src.read_events(0, 1).tolist() == [STARTED_US + 123_457]
    assert src.read_durations(0, 1).tolist() == [0]


def test_events_and_durations_are_int64():
    src = _source(_epochs([(1.0, 2.0, ["a"], False)]))
    assert src.read_events(0, 1).dtype == np.int64
    assert src.read_durations(0, 1).dtype == np.int64


def test_a_point_event_has_a_zero_duration():
    src = _source(_epochs([(1.0, 1.0, ["a"], False)]))
    assert src.read_durations(0, 1).tolist() == [0]
    assert src.max_duration_us() == 0


def test_max_duration_bounds_every_duration():
    src = _source(_table([(0.0, 3.0, ["a"]), (1.0, 1.5, ["b"])]))
    assert src.max_duration_us() == 3_000_000


def test_a_row_that_stops_before_it_starts_is_rejected_by_row_and_table():
    table = _table([(1.0, 2.0, ["a"]), (5.0, 4.0, ["a"])])
    with pytest.raises(ValueError, match="row 1 of epochs stops before"):
        _source(table)


def test_tags_become_sorted_label_categories_with_joined_names():
    rows = [
        (0.0, 1.0, ["seizure"]),
        (1.0, 2.0, ["note", "artifact"]),
        (2.0, 3.0, []),
        (3.0, 4.0, ["artifact", "note"]),
        (4.0, 5.0, ["seizure"]),
    ]
    src = _source(_table(rows))
    assert src.label_names() == [UNTAGGED, "artifact,note", "seizure"]
    assert src.num_labels() == 3
    assert src.read_labels(0, 5).tolist() == [2, 1, 0, 1, 2]
    assert src.read_labels(0, 5).dtype == np.uint16


def test_a_table_without_tags_has_no_labels():
    src = _source(_table([(0.0, 1.0, None), (1.0, 2.0, None)]))
    assert "labels" not in src.columns()
    assert src.label_names() is None
    assert src.num_labels() == 0
    with pytest.raises(ValueError, match="no tags"):
        src.read_labels(0, 1)


def test_more_tag_combinations_than_labels_are_rejected(monkeypatch):
    monkeypatch.setattr("timeseries_zarr_nwb.nwb_events.MAX_LABELS", 2)
    rows = [(i, i + 1.0, [f"t{i}"]) for i in range(3)]
    with pytest.raises(ValueError, match="exceed 2 labels"):
        _source(_table(rows))


def _custom_table():
    table = TimeIntervals(name="seizures", description="marks")
    table.add_column(name="note", description="free text")
    table.add_column(name="score", description="score")
    table.add_row(
        start_time=10.0,
        stop_time=20.0,
        note="evolving 4 Hz",
        score=0.75,
        tags=["x"],
    )
    table.add_row(
        start_time=5.0, stop_time=5.0, note="", score=float("nan"), tags=["x"]
    )
    return table


def test_free_form_columns_become_compact_json_text_in_start_order():
    src = _source(_custom_table())
    texts = src.read_text(0, 2)
    assert texts == [
        '{"note":"","score":null}',
        '{"note":"evolving 4 Hz","score":0.75}',
    ]
    assert [json.loads(t) for t in texts][1]["score"] == 0.75


def test_text_width_is_the_longest_text_in_bytes():
    src = _source(_custom_table())
    assert src.text_width() == len('{"note":"evolving 4 Hz","score":0.75}')


def test_non_ascii_text_is_kept_as_utf8_and_counted_in_bytes():
    table = TimeIntervals(name="notes", description="d")
    table.add_column(name="note", description="n")
    table.add_row(
        start_time=0.0,
        stop_time=1.0,
        note="caf\N{LATIN SMALL LETTER E WITH ACUTE}",
    )
    src = _source(table)
    assert "caf\N{LATIN SMALL LETTER E WITH ACUTE}" in src.read_text(0, 1)[0]
    assert src.text_width() == len(src.read_text(0, 1)[0].encode("utf-8"))


def test_a_table_without_free_form_columns_has_no_text():
    src = _source(_epochs([(0.0, 1.0, ["a"], False)]))
    assert "text" not in src.columns()
    assert src.text_width() == 0
    with pytest.raises(ValueError, match="no text column"):
        src.read_text(0, 1)


def test_timeseries_references_become_the_ids_of_that_series_channels():
    table = _epochs([(1.0, 2.0, ["a"], True), (0.0, 0.5, ["b"], False)])
    src = _source(table)
    assert src.columns() >= {"channels"}
    assert src.read_channel_ids(0, 2) == [[], ["LA1", "LA2", "LA3"]]


def test_repeated_references_to_one_series_list_its_channels_once():
    nwb = mock_NWBFile()
    es = mock_ElectricalSeries(
        nwbfile=nwb, name="es", data=np.zeros((100, 3)), rate=1000.0
    )
    nwb.add_epoch(
        start_time=0.0, stop_time=1.0, tags=["a"], timeseries=[es, es]
    )
    src = _source(nwb.epochs)
    assert src.read_channel_ids(0, 1) == [["LA1", "LA2", "LA3"]]


def test_a_reference_to_a_series_the_bundle_lacks_is_ignored_with_a_warning(
    caplog,
):
    table = _epochs([(1.0, 2.0, ["a"], True), (0.0, 0.5, ["b"], False)])
    with caplog.at_level(logging.WARNING):
        src = _source(table, ids={"other": ["X"]})
    assert "channels" not in src.columns()
    assert "1 timeseries references" in caplog.text
    with pytest.raises(ValueError, match="no row references"):
        src.read_channel_ids(0, 1)


def test_a_table_without_a_timeseries_column_has_no_channels():
    src = _source(_custom_table())
    assert "channels" not in src.columns()


def test_columns_collect_everything_the_table_provides():
    table = _epochs([(1.0, 2.0, ["a"], True)])
    assert _source(table).columns() == {"durations", "labels", "channels"}
    assert _source(_custom_table()).columns() == {"durations", "labels", "text"}


def test_the_source_is_named_by_its_table_and_has_no_unit():
    src = _source(_epochs([(0.0, 1.0, ["a"], False)]))
    assert src.id == "epochs"
    assert src.name == "epochs"
    assert src.unit is None


def test_a_table_with_no_rows_offers_no_columns_and_no_events():
    table = TimeIntervals(name="none", description="d")
    src = _source(table)
    assert src.num_events() == 0
    assert src.columns() == frozenset()
    assert src.max_duration_us() is None
    assert src.read_events(0, 0).shape == (0,)


def test_the_source_serves_the_event_protocol_without_waveform_snippets():
    src = _source(_epochs([(0.0, 1.0, ["a"], False)]))
    assert isinstance(src, EventChannelSource)
    with pytest.raises(ValueError, match="no waveform_mean"):
        src.waveform_mean()
    with pytest.raises(ValueError, match="no waveform_mean"):
        src.waveform_period_us()


def test_reads_return_windows_of_every_column():
    rows = [(float(i), i + 0.5, [f"t{i % 2}"]) for i in range(6)]
    src = _source(_table(rows))
    assert src.num_events() == 6
    assert src.read_events(2, 4).tolist() == [
        STARTED_US + 2_000_000,
        STARTED_US + 3_000_000,
    ]
    assert src.read_durations(2, 4).tolist() == [500_000, 500_000]
    assert src.read_labels(2, 4).tolist() == [0, 1]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.arange(3), [0, 1, 2]),
        (np.float32(1.5), 1.5),
        (np.int64(7), 7),
        (np.float64("nan"), None),
        (float("inf"), None),
        (np.array([1.0, np.nan]), [1.0, None]),
        (b"ab", "ab"),
        ("text", "text"),
        (True, True),
        (None, None),
        ({"a": 1}, "{'a': 1}"),
    ],
)
def test_json_value_converts_what_json_cannot_write(value, expected):
    assert _json_value(value) == expected


class _FakeTable:
    """A minimal TimeIntervals stand-in with one row and an empty reference."""

    name = "fake"
    neurodata_type = "TimeIntervals"
    colnames = ("start_time", "stop_time", "timeseries")

    def __getitem__(self, column):
        return {
            "start_time": [1.0],
            "stop_time": [2.0],
            "timeseries": [[SimpleNamespace(timeseries=None)]],
        }[column]


def test_a_reference_without_a_series_applies_to_no_channel():
    src = NwbIntervalSource(_FakeTable(), STARTED, _IDS)
    assert "channels" not in src.columns()
    assert src.read_events(0, 1).tolist() == [STARTED_US + 1_000_000]


def _marks(rows, **kwargs):
    table = EventsTable(name="marks", description="d")
    for row in rows:
        table.add_row(**row)
    return NwbEventsTableSource(table, STARTED, **kwargs)


def test_events_table_times_are_reference_time_plus_timestamp_sorted_stably():
    src = _marks(
        [
            {"timestamp": 2.0, "annotation": "b"},
            {"timestamp": 0.5, "annotation": "a"},
            {"timestamp": 2.0, "annotation": "c"},
        ]
    )
    assert src.read_events(0, 3).tolist() == [
        STARTED_US + 500_000,
        STARTED_US + 2_000_000,
        STARTED_US + 2_000_000,
    ]
    assert src.read_text(0, 3) == ["a", "b", "c"]


def test_events_table_durations_are_microseconds_in_time_order():
    src = _marks(
        [
            {"timestamp": 2.0, "duration": 0.25},
            {"timestamp": 1.0, "duration": 1.5},
        ]
    )
    assert src.read_durations(0, 2).tolist() == [1_500_000, 250_000]
    assert src.max_duration_us() == 1_500_000
    assert src.columns() == {"durations"}


def test_events_table_annotation_text_is_kept_as_written():
    word = "caf\N{LATIN SMALL LETTER E WITH ACUTE} onset"
    src = _marks([{"timestamp": 0.0, "annotation": word}])
    assert src.read_text(0, 1) == [word]
    assert src.text_width() == len(word.encode("utf-8"))
    assert src.columns() == {"text"}


def test_events_table_with_duration_and_annotation_offers_both():
    src = _marks([{"timestamp": 1.0, "duration": 0.5, "annotation": "x"}])
    assert src.columns() == {"durations", "text"}


def test_events_table_with_only_timestamps_offers_no_optional_column():
    src = _marks([{"timestamp": 1.0}, {"timestamp": 2.0}])
    assert src.columns() == frozenset()
    assert src.max_duration_us() is None
    assert src.text_width() == 0
    with pytest.raises(ValueError, match="no durations"):
        src.read_durations(0, 1)
    with pytest.raises(ValueError, match="no text column"):
        src.read_text(0, 1)


def test_events_table_has_no_labels_or_channels_or_snippets():
    src = _marks([{"timestamp": 1.0, "annotation": "x"}])
    assert src.label_names() is None
    assert src.num_labels() == 0
    with pytest.raises(ValueError, match="no tags"):
        src.read_labels(0, 1)
    with pytest.raises(ValueError, match="no row references"):
        src.read_channel_ids(0, 1)
    with pytest.raises(ValueError, match="no waveform_mean"):
        src.waveform_mean()


def test_events_table_other_columns_are_ignored_with_a_warning(caplog):
    table = EventsTable(name="marks", description="d")
    table.add_column(name="severity", description="s")
    table.add_row(timestamp=1.0, annotation="x", severity=3)
    with caplog.at_level(logging.WARNING):
        src = NwbEventsTableSource(table, STARTED)
    assert "['severity'] are not carried" in caplog.text
    assert src.columns() == {"text"}


def test_events_table_is_named_by_its_table_and_has_no_unit():
    src = _marks([{"timestamp": 1.0}])
    assert (src.id, src.name, src.unit) == ("marks", "marks", None)
    assert isinstance(src, EventChannelSource)


def test_events_table_with_no_rows_offers_no_columns():
    table = EventsTable(name="empty", description="d")
    table.add_column(name="annotation", description="a")
    src = NwbEventsTableSource(table, STARTED)
    assert src.num_events() == 0
    assert src.columns() == frozenset()


def test_events_table_rejects_a_negative_duration():
    with pytest.raises(ValueError, match="row 1 of marks has a negative"):
        _marks(
            [
                {"timestamp": 1.0, "duration": 0.5},
                {"timestamp": 2.0, "duration": -0.5},
            ]
        )


@pytest.mark.parametrize("column", ["timestamp", "duration"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_events_table_rejects_a_value_that_is_not_finite(column, bad):
    row = {"timestamp": 1.0, "duration": 0.5, column: bad}
    with pytest.raises(
        ValueError, match=f"has a {column} that is not a finite"
    ):
        _marks([row])


@pytest.mark.parametrize("which", ["start", "stop"])
def test_interval_table_rejects_a_time_that_is_not_finite(which):
    nan = float("nan")
    rows = [(nan, 2.0, ["a"])] if which == "start" else [(1.0, nan, ["a"])]
    with pytest.raises(ValueError, match="not a finite number"):
        _source(_table(rows))


_DEPRECATED = pytest.mark.filterwarnings(
    "ignore:AnnotationSeries is deprecated:UserWarning"
)


def _annotations(data, timestamps, name="marks"):
    return AnnotationSeries(name=name, data=data, timestamps=timestamps)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"caf\xc3\xa9", "caf\N{LATIN SMALL LETTER E WITH ACUTE}"),
        ("text", "text"),
        (np.str_("numpy"), "numpy"),
        (7, "7"),
    ],
)
def test_as_text_decodes_bytes_and_stringifies_the_rest(value, expected):
    assert _as_text(value) == expected


@_DEPRECATED
def test_annotation_series_events_are_reference_time_plus_timestamps_in_order():
    series = _annotations(["b", "a", "c"], [2.0, 0.5, 2.0])
    src = NwbAnnotationSeriesSource(series, STARTED)
    assert src.read_events(0, 3).tolist() == [
        STARTED_US + 500_000,
        STARTED_US + 2_000_000,
        STARTED_US + 2_000_000,
    ]
    assert src.read_text(0, 3) == ["a", "b", "c"]


@_DEPRECATED
def test_annotation_series_offers_only_text():
    src = NwbAnnotationSeriesSource(
        _annotations(["x", "yy"], [1.0, 2.0]), STARTED
    )
    assert src.columns() == {"text"}
    assert src.text_width() == 2
    assert src.max_duration_us() is None
    with pytest.raises(ValueError, match="no durations"):
        src.read_durations(0, 1)


@_DEPRECATED
def test_annotation_series_keeps_utf8_text_and_counts_its_bytes():
    word = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    src = NwbAnnotationSeriesSource(_annotations([word], [1.0]), STARTED)
    assert src.read_text(0, 1) == [word]
    assert src.text_width() == len(word.encode("utf-8"))


@_DEPRECATED
def test_annotation_series_is_named_by_its_series_and_serves_the_protocol():
    src = NwbAnnotationSeriesSource(
        _annotations(["x"], [1.0], name="notes"), STARTED
    )
    assert (src.id, src.name, src.unit) == ("notes", "notes", None)
    assert isinstance(src, EventChannelSource)


@_DEPRECATED
def test_annotation_series_with_no_samples_offers_no_columns():
    src = NwbAnnotationSeriesSource(_annotations([], []), STARTED)
    assert src.num_events() == 0
    assert src.columns() == frozenset()


@_DEPRECATED
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_annotation_series_rejects_a_timestamp_that_is_not_finite(bad):
    with pytest.raises(ValueError, match="not a finite number"):
        NwbAnnotationSeriesSource(_annotations(["x"], [bad]), STARTED)


def test_annotation_series_without_timestamps_is_rejected():
    series = SimpleNamespace(name="rated", timestamps=None, data=["x"])
    with pytest.raises(ValueError, match="needs timestamps"):
        NwbAnnotationSeriesSource(series, STARTED)


def test_annotation_series_whose_data_and_timestamps_differ_is_rejected():
    series = SimpleNamespace(name="odd", timestamps=[1.0], data=["x", "y"])
    with pytest.raises(ValueError, match="2 texts but 1 timestamps"):
        NwbAnnotationSeriesSource(series, STARTED)


def test_annotation_series_decodes_byte_strings_from_a_file():
    series = SimpleNamespace(
        name="bytes",
        neurodata_type="AnnotationSeries",
        timestamps=[1.0],
        data=[b"onset"],
    )
    src = NwbAnnotationSeriesSource(series, STARTED)
    assert src.read_text(0, 1) == ["onset"]


def test_interval_provenance_names_the_type_and_the_path():
    table = _table([(1.0, 2.0, ["a"])])
    src = NwbIntervalSource(table, STARTED, _IDS, nwb_path="/intervals/epochs")
    assert src.provenance() == {
        "neurodata_type": "TimeIntervals",
        "nwb_path": "/intervals/epochs",
    }


def test_provenance_without_a_path_names_only_the_type():
    assert _source(_table([(1.0, 2.0, ["a"])])).provenance() == {
        "neurodata_type": "TimeIntervals"
    }


def test_events_table_provenance_names_the_type_and_the_path():
    src = _marks([{"timestamp": 1.0}], nwb_path="/events/marks")
    assert src.provenance() == {
        "neurodata_type": "EventsTable",
        "nwb_path": "/events/marks",
    }


@_DEPRECATED
def test_annotation_series_provenance_names_the_type_and_the_path():
    src = NwbAnnotationSeriesSource(
        _annotations(["x"], [1.0]), STARTED, nwb_path="/acquisition/notes"
    )
    assert src.provenance() == {
        "neurodata_type": "AnnotationSeries",
        "nwb_path": "/acquisition/notes",
    }
