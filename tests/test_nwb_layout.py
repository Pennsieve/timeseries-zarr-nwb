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
    """Return the first row, rows, and first sample of each run."""
    return [(run.first_row, run.rows, run.first_sample) for run in layout.runs]


def test_a_series_with_a_rate_is_one_run_from_sample_zero():
    series = _rated(5, start=2.5)
    layout = series_layout(series, WINDOW)
    assert layout.rate_hz == RATE
    assert layout.start_s == 2.5
    assert layout.runs == (Run(series, 0, 5, 0),)
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
    assert _spans(layout) == [(0, 50, 0)]
    assert layout.num_samples == 50


def test_a_gap_starts_a_run_at_its_grid_sample():
    layout = series_layout(_stamped(_grid((0, 4), (10, 13))), WINDOW)
    assert _spans(layout) == [(0, 4, 0), (4, 3, 10)]
    assert layout.num_samples == 13


@pytest.mark.parametrize("window_rows", [2, 3, 5, WINDOW])
def test_gaps_land_on_the_same_samples_for_any_window(window_rows):
    stamps = _grid((0, 1), (3, 7), (9, 10), (20, 25))
    layout = series_layout(_stamped(stamps), window_rows)
    assert _spans(layout) == [(0, 1, 0), (1, 4, 3), (5, 1, 9), (6, 5, 20)]
    assert layout.num_samples == 25


def test_a_long_gap_after_a_short_run_lands_on_its_grid_sample():
    rate = 30000.0
    samples = np.concatenate([np.arange(10**6), 10**8 + np.arange(10)])
    series = _stamped(1234.5 + samples / rate)
    for window_rows in (4096, WINDOW):
        layout = series_layout(series, window_rows)
        assert _spans(layout) == [(0, 10**6, 0), (10**6, 10, 10**8)]
        assert layout.rate_hz == pytest.approx(rate, rel=1e-9)


def test_a_gap_off_the_grid_is_rejected_at_its_first_row():
    stamps = _grid((0, 4), (10, 13))
    stamps[4:] += 0.4 / RATE
    with pytest.raises(
        ValueError, match="timestamp 4 of stamped is 400.000 us off its"
    ):
        series_layout(_stamped(stamps), WINDOW)


def test_jitter_within_the_tolerance_is_accepted():
    jitter = np.random.default_rng(0).uniform(-0.2e-6, 0.2e-6, 1000)
    layout = series_layout(_stamped(_grid((0, 1000)) + jitter), WINDOW)
    assert _spans(layout) == [(0, 1000, 0)]


def test_jitter_beyond_the_tolerance_is_rejected():
    stamps = _grid((0, 1000))
    stamps[500] += 2e-6
    with pytest.raises(
        ValueError, match="timestamp 500 of stamped is 2.000 us off"
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


def test_stitch_places_segments_on_the_grid_of_the_earliest():
    first = _rated(4, start=0.5, name="first")
    second = _rated(3, start=0.51, name="second")
    layout = stitch([series_layout(s, WINDOW) for s in (second, first)])
    assert layout.start_s == 0.5
    assert layout.runs == (Run(first, 0, 4, 0), Run(second, 0, 3, 10))
    assert layout.num_samples == 13
    assert layout.series == (first, second)


def test_stitch_keeps_the_gaps_inside_a_segment_sampled_by_timestamps():
    first = _rated(4, name="first")
    second = _stamped(0.01 + _grid((0, 3), (5, 6)), name="second")
    layout = stitch([series_layout(s, WINDOW) for s in (first, second)])
    assert _spans(layout) == [(0, 4, 0), (0, 3, 10), (3, 1, 15)]
    assert layout.num_samples == 16


def test_stitch_rejects_a_segment_that_overlaps_the_one_before():
    layouts = [
        series_layout(_rated(10, name="first"), WINDOW),
        series_layout(_rated(3, start=0.005, name="second"), WINDOW),
    ]
    with pytest.raises(ValueError, match="second starts before first ends"):
        stitch(layouts)


def test_stitch_rejects_a_segment_off_the_grid_of_the_earliest():
    layouts = [
        series_layout(_rated(4, name="first"), WINDOW),
        series_layout(_rated(3, start=0.0104, name="second"), WINDOW),
    ]
    with pytest.raises(
        ValueError,
        match="second is 400.000 us off the 1000 Hz sample grid of first",
    ):
        stitch(layouts)


def test_stitch_rejects_a_segment_whose_rate_drifts_off_the_grid():
    layouts = [
        series_layout(_rated(4, name="first"), WINDOW),
        series_layout(
            _rated(1000, rate=1000.01, start=0.01, name="second"), WINDOW
        ),
    ]
    with pytest.raises(ValueError, match=r"second is -9\.99\d us off"):
        stitch(layouts)


def test_stitch_accepts_a_rate_difference_within_the_tolerance():
    layouts = [
        series_layout(_rated(4, name="first"), WINDOW),
        series_layout(
            _rated(1000, rate=1000.0001, start=0.01, name="second"), WINDOW
        ),
    ]
    assert stitch(layouts).num_samples == 1010


def test_stitch_lists_a_segment_without_samples_and_places_nothing():
    first = _rated(4, name="first")
    empty = _rated(0, start=0.002, name="empty")
    last = _rated(3, start=0.01, name="last")
    layout = stitch([series_layout(s, WINDOW) for s in (first, empty, last)])
    assert layout.series == (first, empty, last)
    assert layout.runs == (Run(first, 0, 4, 0), Run(last, 0, 3, 10))


def test_stitch_takes_the_grid_of_the_earliest_segment_with_samples():
    empty = _rated(0, start=-1.0, name="empty")
    first = _rated(4, start=0.5, name="first")
    layout = stitch([series_layout(s, WINDOW) for s in (first, empty)])
    assert layout.start_s == 0.5
    assert layout.series == (empty, first)
    assert layout.runs == (Run(first, 0, 4, 0),)
    assert layout.num_samples == 4
