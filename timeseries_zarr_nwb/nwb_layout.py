"""Placement of the rows of NWB series on the uniform sample grid of a channel."""

from collections.abc import Sequence
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
from pynwb import TimeSeries

GRID_TOLERANCE_S: Final = 1e-6
"""Largest distance in seconds from a sample time to the grid of its channel."""

SEED_ROWS: Final = 4096
"""Leading timestamps whose steps give the first estimate of the sample period."""

MIN_TIMESTAMPS: Final = 2
"""Fewest timestamps that give a sample rate."""


class Run(NamedTuple):
    """Consecutive rows of one series at consecutive samples of a grid.

    Attributes:
        series: The series that holds the rows.
        first_row: Index of the first row in the data of the series.
        rows: Number of rows.
        first_sample: Grid sample of the first row.
    """

    series: TimeSeries
    first_row: int
    rows: int
    first_sample: int


class Layout(NamedTuple):
    """Where the rows of one or more series sit on one sample grid.

    Attributes:
        rate_hz: Sample rate of the grid.
        start_s: Time of grid sample 0, in seconds from the file's
            timestamps_reference_time.
        runs: The runs in grid order. Runs do not overlap, and a grid sample
            outside every run is a gap.
        num_samples: Grid samples from sample 0 to the end of the last run.
        series: Every series the layout places, in time order.
    """

    rate_hz: float
    start_s: float
    runs: tuple[Run, ...]
    num_samples: int
    series: tuple[TimeSeries, ...]


def series_layout(series: TimeSeries, window_rows: int) -> Layout:
    """Return the layout of one series sampled by a rate or by timestamps.

    A series with a rate is one run from grid sample 0. A series with
    timestamps reads them window_rows at a time. Its rate comes from the
    timestamps, and most steps between the first SEED_ROWS timestamps must be
    one sample period. A step of more than one sample period is a gap.

    Raises:
        ValueError: The series has fewer than 2 timestamps, more or fewer
            timestamps than rows, a timestamp that is not finite, timestamps
            that do not increase, two timestamps less than half a sample
            period apart, or a timestamp more than GRID_TOLERANCE_S from the
            sample grid.
    """
    if series.rate is not None:
        rows = int(series.data.shape[0])
        runs = (Run(series, 0, rows, 0),) if rows else ()
        return Layout(
            float(series.rate),
            float(series.starting_time),
            runs,
            rows,
            (series,),
        )
    return _timestamps_layout(series, max(1, window_rows))


def _read_stamps(
    series: TimeSeries, start: int, stop: int
) -> npt.NDArray[np.float64]:
    """Return timestamps [start, stop) as float64, raising if any is not finite."""
    values = np.asarray(series.timestamps[start:stop], dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        row = start + int(bad[0])
        raise ValueError(
            f"timestamp {row} of {series.name} is not a finite number"
        )
    return values


def _require_increasing(
    steps: npt.NDArray[np.float64], first_row: int, name: str
) -> None:
    """Raise ValueError at the first step between timestamps that is not positive.

    steps[i] is the step that ends at timestamp first_row + i.
    """
    bad = np.flatnonzero(steps <= 0)
    if bad.size:
        row = first_row + int(bad[0])
        raise ValueError(
            f"timestamp {row} of {name} is not later than timestamp {row - 1}"
        )


def _period(span_s: float, steps: int, seed: float, seed_steps: int) -> float:
    """Return the best estimate of the sample period from the runs seen so far.

    span_s is the time the runs span, summed over the runs, and steps is the
    number of sample periods they span. seed is the median of seed_steps
    steps between timestamps.
    """
    # The median of n steps and the span of k periods are about equally
    # precise when k is near the square root of n.
    if steps and steps * steps >= seed_steps:
        return span_s / steps
    return seed


class _Runs(NamedTuple):
    """Where the runs of a series sampled by timestamps start.

    Attributes:
        rows: The first row of each run.
        samples: The grid sample of the first row of each run.
        period_s: Sample period of the grid in seconds.
        num_samples: Grid samples from sample 0 to the end of the last run.
    """

    rows: list[int]
    samples: list[int]
    period_s: float
    num_samples: int


def _timestamps_layout(series: TimeSeries, window_rows: int) -> Layout:
    """Return the layout of a series sampled by timestamps; see series_layout."""
    name = str(series.name)
    total = len(series.timestamps)
    rows = int(series.data.shape[0])
    if total != rows:
        raise ValueError(f"{name} has {rows} rows but {total} timestamps")
    if total < MIN_TIMESTAMPS:
        raise ValueError(
            f"{name} needs {MIN_TIMESTAMPS} or more timestamps to give a "
            f"sample rate, got {total}"
        )
    first = _read_stamps(series, 0, min(SEED_ROWS, total))
    origin = float(first[0])
    first_steps = np.diff(first)
    _require_increasing(first_steps, 1, name)
    found = _find_runs(
        series,
        window_rows,
        origin,
        float(np.median(first_steps)),
        len(first_steps),
    )
    _check_grid(series, window_rows, origin, found)
    ends = [*found.rows[1:], total]
    runs = tuple(
        Run(series, first_row, end - first_row, first_sample)
        for first_row, end, first_sample in zip(
            found.rows, ends, found.samples, strict=True
        )
    )
    return Layout(
        1 / found.period_s, origin, runs, found.num_samples, (series,)
    )


def _find_runs(
    series: TimeSeries,
    window_rows: int,
    origin: float,
    seed: float,
    seed_steps: int,
) -> _Runs:
    """Return the runs of a series sampled by timestamps and its sample period.

    origin is the first timestamp. seed is the median of the first
    seed_steps steps between timestamps. The period comes from the spans
    inside runs. A misaligned gap does not change it.
    """
    name = str(series.name)
    total = len(series.timestamps)
    run_rows = [0]
    run_samples = [0]
    run_start_s = origin
    span_s = 0.0
    steps_done = 0
    previous = origin
    sample = 0
    for start in range(1, total, window_rows):
        stop = min(start + window_rows, total)
        times = _read_stamps(series, start, stop)
        steps = np.diff(times, prepend=previous)
        _require_increasing(steps, start, name)
        period = _period(
            span_s + previous - run_start_s,
            steps_done + sample - run_samples[-1],
            seed,
            seed_steps,
        )
        done = start
        for jump in np.flatnonzero(np.rint(steps / period) != 1):
            row = start + int(jump)
            sample += row - done
            before = previous if jump == 0 else float(times[jump - 1])
            span_s += before - run_start_s
            steps_done += sample - run_samples[-1]
            period = _period(span_s, steps_done, seed, seed_steps)
            step = round(float(steps[jump]) / period)
            if step < 1:
                raise ValueError(
                    f"timestamp {row} of {name} is less than half a sample "
                    f"period after timestamp {row - 1}"
                )
            sample += step
            run_rows.append(row)
            run_samples.append(sample)
            run_start_s = float(times[jump])
            done = row + 1
        sample += stop - done
        previous = float(times[-1])

    span_s += previous - run_start_s
    steps_done += sample - run_samples[-1]
    period = span_s / steps_done if steps_done else (previous - origin) / sample
    return _Runs(run_rows, run_samples, period, sample + 1)


def _check_grid(
    series: TimeSeries, window_rows: int, origin: float, found: _Runs
) -> None:
    """Raise ValueError at the first timestamp off the grid of found.

    origin is the time of grid sample 0.
    """
    total = len(series.timestamps)
    starts = np.asarray(found.rows)
    offsets = np.asarray(found.samples)
    for start in range(0, total, window_rows):
        stop = min(start + window_rows, total)
        times = _read_stamps(series, start, stop)
        indexes = np.arange(start, stop)
        run = np.searchsorted(starts, indexes, side="right") - 1
        samples = offsets[run] + indexes - starts[run]
        error = times - (origin + samples * found.period_s)
        bad = np.flatnonzero(np.abs(error) > GRID_TOLERANCE_S)
        if bad.size:
            index = int(bad[0])
            raise ValueError(
                f"timestamp {start + index} of {series.name} is "
                f"{error[index] * 1e6:.3f} us off its "
                f"{1 / found.period_s:g} Hz sample grid; every timestamp must "
                f"be within {GRID_TOLERANCE_S * 1e6:g} us of the grid"
            )


def stitch(layouts: Sequence[Layout]) -> Layout:
    """Return one layout that places every layout on one grid.

    The layouts are the segments of one recording. They are taken in order
    of start_s, and a tie keeps the given order. The grid is that of the
    earliest segment with samples. A layout without samples adds its series
    and places no sample.

    Raises:
        ValueError: A segment starts before the previous segment ends, or its
            first or last sample is more than GRID_TOLERANCE_S from the grid.
    """
    ordered = sorted(layouts, key=lambda layout: layout.start_s)
    base = next(
        (layout for layout in ordered if layout.num_samples), ordered[0]
    )
    period = 1 / base.rate_hz
    runs: list[Run] = []
    end = 0
    previous = base
    for layout in ordered:
        if not layout.num_samples:
            continue
        name = layout.series[0].name
        offset_s = layout.start_s - base.start_s
        first_sample = round(offset_s / period)
        first_error = offset_s - first_sample * period
        drift = (layout.num_samples - 1) * (1 / layout.rate_hz - period)
        error = max(first_error, first_error + drift, key=abs)
        if abs(error) > GRID_TOLERANCE_S:
            raise ValueError(
                f"{name} is {error * 1e6:.3f} us off the {base.rate_hz:g} Hz "
                f"sample grid of {base.series[0].name}; every segment must be "
                f"within {GRID_TOLERANCE_S * 1e6:g} us of the grid of the "
                "earliest segment"
            )
        if first_sample < end:
            raise ValueError(
                f"{name} starts before {previous.series[0].name} ends; "
                "segments of one recording must not overlap"
            )
        runs.extend(
            run._replace(first_sample=run.first_sample + first_sample)
            for run in layout.runs
        )
        end = first_sample + layout.num_samples
        previous = layout
    series = tuple(segment for layout in ordered for segment in layout.series)
    return Layout(base.rate_hz, base.start_s, tuple(runs), end, series)
