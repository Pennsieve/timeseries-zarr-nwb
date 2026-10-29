import numpy as np
import pytest
from pynwb import TimeSeries

from timeseries_zarr_nwb.nwb_layout import Run, series_layout, stitch

RATE = 1000.0
"""Sample rate in hertz of the series these tests build."""

WINDOW = 1 << 20
"""Timestamps one window reads, more than any series here holds."""


def _stamped(stamps, name="stamped"):
    """Return a TimeSeries sampled at stamps, in seconds."""
    stamps = np.asarray(stamps, dtype=np.float64)
    return TimeSeries(
        name=name, data=np.zeros(len(stamps)), unit="n/a", timestamps=stamps
    )


def _rated(rows, *, rate=RATE, start=0.0, name="rated"):
    """Return a TimeSeries of rows samples at rate from start, in seconds."""
    return TimeSeries(
        name=name,
        data=np.zeros(rows),
        unit="n/a",
        rate=rate,
        starting_time=start,
    )


def _grid(*spans):
    """Return the times of the grid samples in each [first, stop) span."""
    return np.concatenate([np.arange(a, b) for a, b in spans]) / RATE


def _spans(layout):
    """Return the first row, rows, and start time of each run."""
    return [(run.first_row, run.rows, run.start_s) for run in layout.runs]


def test_a_series_with_a_rate_is_one_run_from_its_starting_time():
    series = _rated(5, start=2.5)
    layout = series_layout(series, WINDOW)
    assert layout.rate_hz == RATE
    assert layout.start_s == 2.5
    assert layout.runs == (Run(series, 0, 5, 0, 2.5),)
    assert layout.num_samples == 5
    assert layout.series == (series,)


def test_a_series_with_a_rate_and_no_rows_has_no_runs():
    layout = series_layout(_rated(0), WINDOW)
    assert layout.runs == ()
    assert layout.num_samples == 0


def test_regular_timestamps_are_one_run_at_their_rate():
    layout = series_layout(_stamped(100.25 + _grid((0, 50))), WINDOW)
    assert layout.rate_hz == pytest.approx(RATE, rel=1e-12)
    assert layout.start_s == 100.25
    assert _spans(layout) == [(0, 50, 100.25)]
    assert layout.num_samples == 50


def test_a_pause_starts_a_run_at_the_time_of_its_first_row():
    layout = series_layout(_stamped(_grid((0, 4), (10, 13))), WINDOW)
    assert _spans(layout) == [(0, 4, 0.0), (4, 3, 0.010)]
    assert [run.first_sample for run in layout.runs] == [0, 4]
    assert layout.num_samples == 7


@pytest.mark.parametrize("window_rows", [2, 3, 5, WINDOW])
def test_pauses_start_the_same_runs_for_any_window(window_rows):
    stamps = _grid((0, 1), (3, 7), (9, 10), (20, 25))
    layout = series_layout(_stamped(stamps), window_rows)
    assert [(run.first_row, run.rows) for run in layout.runs] == [
        (0, 1),
        (1, 4),
        (5, 1),
        (6, 5),
    ]
    assert [run.start_s for run in layout.runs] == pytest.approx(
        [0.0, 0.003, 0.009, 0.020]
    )
    assert layout.num_samples == 11


def test_a_long_pause_after_a_short_run_starts_a_run():
    rate = 30000.0
    samples = np.concatenate([np.arange(10**6), 10**8 + np.arange(10)])
    series = _stamped(1234.5 + samples / rate)
    for window_rows in (4096, WINDOW):
        layout = series_layout(series, window_rows)
        assert [(run.first_row, run.rows) for run in layout.runs] == [
            (0, 10**6),
            (10**6, 10),
        ]
        assert layout.rate_hz == pytest.approx(rate, rel=1e-9)


def test_a_pause_that_resumes_off_the_grid_starts_a_run_at_its_own_time():
    stamps = _grid((0, 4), (10, 13))
    stamps[4:] += 0.4 / RATE
    layout = series_layout(_stamped(stamps), WINDOW)
    assert _spans(layout) == [(0, 4, 0.0), (4, 3, pytest.approx(0.0104))]


def test_a_step_of_a_period_and_a_half_or_less_is_jitter_and_is_rejected():
    stamps = _grid((0, 100))
    stamps[50:] += 0.4 / RATE
    with pytest.raises(
        ValueError, match=r"of stamped is -?\d+\.\d+ us off the .* of its run"
    ):
        series_layout(_stamped(stamps), WINDOW)


def test_jitter_within_the_tolerance_is_accepted():
    jitter = np.random.default_rng(0).uniform(-0.2e-6, 0.2e-6, 1000)
    layout = series_layout(_stamped(_grid((0, 1000)) + jitter), WINDOW)
    assert [(run.first_row, run.rows) for run in layout.runs] == [(0, 1000)]


def test_jitter_beyond_the_tolerance_is_rejected():
    stamps = _grid((0, 1000))
    stamps[500] += 2e-6
    with pytest.raises(
        ValueError,
        match="timestamp 500 of stamped is 2.000 us off the 1000 Hz sample "
        "grid of its run; every timestamp must be within 1 us of the grid of "
        "its run",
    ):
        series_layout(_stamped(stamps), WINDOW)


@pytest.mark.parametrize("earlier", [3, 4])
def test_timestamps_that_do_not_increase_are_rejected(earlier):
    stamps = _grid((0, 10))
    stamps[5] = stamps[earlier]
    with pytest.raises(
        ValueError, match="timestamp 5 of stamped is not later than timestamp 4"
    ):
        series_layout(_stamped(stamps), WINDOW)


def test_timestamps_closer_than_half_a_period_are_rejected():
    stamps = np.insert(_grid((0, 10)), 5, 4.3 / RATE)
    with pytest.raises(ValueError, match="timestamp 5 .* less than half"):
        series_layout(_stamped(stamps), WINDOW)


def test_a_timestamp_that_is_not_finite_is_rejected():
    stamps = _grid((0, 10))
    stamps[3] = np.nan
    with pytest.raises(
        ValueError, match="timestamp 3 of stamped is not a finite"
    ):
        series_layout(_stamped(stamps), WINDOW)


def test_a_single_timestamp_is_rejected():
    with pytest.raises(ValueError, match="needs 2 or more timestamps"):
        series_layout(_stamped([1.0]), WINDOW)


def test_timestamps_that_outnumber_the_rows_are_rejected():
    series = _stamped(_grid((0, 5)))
    series.fields["data"] = np.zeros(4)
    with pytest.raises(ValueError, match="has 4 rows but 5 timestamps"):
        series_layout(series, WINDOW)


def test_stitch_of_one_layout_is_that_layout():
    layout = series_layout(_rated(5), WINDOW)
    assert stitch([layout]) == layout


def test_stitch_concatenates_the_runs_of_segments_in_time_order():
    first = _rated(4, start=0.5, name="first")
    second = _rated(3, start=0.51, name="second")
    layout = stitch([series_layout(s, WINDOW) for s in (second, first)])
    assert layout.start_s == 0.5
    assert layout.runs == (
        Run(first, 0, 4, 0, 0.5),
        Run(second, 0, 3, 4, 0.51),
    )
    assert layout.num_samples == 7
    assert layout.series == (first, second)


def test_stitch_keeps_a_segment_that_resumes_off_the_grid_of_the_earliest():
    first = _rated(4, name="first")
    second = _rated(3, start=0.0104, name="second")
    layout = stitch([series_layout(s, WINDOW) for s in (first, second)])
    assert [run.start_s for run in layout.runs] == [0.0, 0.0104]


def test_stitch_keeps_the_pauses_inside_a_segment_sampled_by_timestamps():
    first = _rated(4, name="first")
    second = _stamped(0.01 + _grid((0, 3), (5, 6)), name="second")
    layout = stitch([series_layout(s, WINDOW) for s in (first, second)])
    assert [(run.rows, run.first_sample) for run in layout.runs] == [
        (4, 0),
        (3, 4),
        (1, 7),
    ]
    assert layout.num_samples == 8


def test_stitch_rejects_a_segment_that_overlaps_the_one_before():
    layouts = [
        series_layout(_rated(10, name="first"), WINDOW),
        series_layout(_rated(3, start=0.005, name="second"), WINDOW),
    ]
    with pytest.raises(ValueError, match="second starts before first ends"):
        stitch(layouts)


def test_stitch_accepts_segments_that_abut_within_the_tolerance():
    first = _rated(10, name="first")
    second = _rated(3, start=0.0099995, name="second")
    assert stitch([series_layout(s, WINDOW) for s in (first, second)]).runs


def test_stitch_rejects_a_segment_whose_rate_drifts_off_the_grid():
    layouts = [
        series_layout(_rated(4, name="first"), WINDOW),
        series_layout(
            _rated(1000, rate=1000.01, start=0.01, name="second"), WINDOW
        ),
    ]
    with pytest.raises(
        ValueError,
        match=r"second drifts -9\.99\d us from the 1000 Hz sample grid of "
        "first within one run; the drift must stay within 1 us",
    ):
        stitch(layouts)


def test_stitch_accepts_a_rate_difference_within_the_tolerance():
    layouts = [
        series_layout(_rated(4, name="first"), WINDOW),
        series_layout(
            _rated(1000, rate=1000.0001, start=0.01, name="second"), WINDOW
        ),
    ]
    assert stitch(layouts).num_samples == 1004


def test_stitch_lists_a_segment_without_samples_and_adds_no_run():
    first = _rated(4, name="first")
    empty = _rated(0, start=0.002, name="empty")
    last = _rated(3, start=0.01, name="last")
    layout = stitch([series_layout(s, WINDOW) for s in (first, empty, last)])
    assert layout.series == (first, empty, last)
    assert layout.runs == (Run(first, 0, 4, 0, 0.0), Run(last, 0, 3, 4, 0.01))


def test_stitch_takes_the_start_of_the_earliest_segment_with_samples():
    empty = _rated(0, start=-1.0, name="empty")
    first = _rated(4, start=0.5, name="first")
    layout = stitch([series_layout(s, WINDOW) for s in (first, empty)])
    assert layout.start_s == 0.5
    assert layout.series == (empty, first)
    assert layout.runs == (Run(first, 0, 4, 0, 0.5),)
    assert layout.num_samples == 4
