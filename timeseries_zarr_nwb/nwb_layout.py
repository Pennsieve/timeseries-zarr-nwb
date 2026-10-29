"""Runs of NWB series rows: spans of samples one sample period apart."""

import math
from collections.abc import Sequence
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
from pynwb import TimeSeries

GRID_TOLERANCE_S: Final = 1e-6
"""Largest distance in seconds from a sample time to the grid of its run."""

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
        first_sample: Index of the first row among the recorded samples of
            the layout. A pause between runs adds no recorded sample.
        start_s: Time of the first row in seconds from the file's
            timestamps_reference_time.
    """

    series: TimeSeries
    first_row: int
    rows: int
    first_sample: int
    start_s: float


class Layout(NamedTuple):
    """The runs of one or more series that share a sample rate.

    Attributes:
        rate_hz: Sample rate of every run.
        start_s: Time of the first row of the first run, in seconds from the
            file's timestamps_reference_time.
        runs: The runs in time order. Runs do not overlap in time.
        num_samples: Rows of every run. A pause between runs adds none.
        series: Every series the layout holds, in time order.
    """

    rate_hz: float
    start_s: float
    runs: tuple[Run, ...]
    num_samples: int
    series: tuple[TimeSeries, ...]


def series_layout(series: TimeSeries, window_rows: int) -> Layout:
    """Return the layout of one series sampled by a rate or by timestamps.

    A series with a rate is one run. A series with timestamps reads them
    window_rows at a time. Its rate comes from the timestamps, and most steps
    between the first SEED_ROWS timestamps must be one sample period. A step
    of more than one sample period starts a new run at the time of its
    timestamp.

    Raises:
        ValueError: The series has fewer than 2 timestamps, more or fewer
            timestamps than rows, a timestamp that is not finite, timestamps
            that do not increase, two timestamps less than half a sample
            period apart, or a timestamp more than GRID_TOLERANCE_S from the
            sample grid of its run.
    """
    if series.rate is not None:
        rows = int(series.data.shape[0])
        start_s = float(series.starting_time)
        runs = (Run(series, 0, rows, 0, start_s),) if rows else ()
        return Layout(float(series.rate), start_s, runs, rows, (series,))
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
        starts_s: The time of the first row of each run.
        period_s: Sample period of the runs in seconds.
    """

    rows: list[int]
    starts_s: list[float]
    period_s: float


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
    _check_grid(series, window_rows, found)
    ends = [*found.rows[1:], total]
    runs = tuple(
        Run(series, first_row, end - first_row, first_row, start_s)
        for first_row, end, start_s in zip(
            found.rows, ends, found.starts_s, strict=True
        )
    )
    return Layout(1 / found.period_s, origin, runs, total, (series,))


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
    inside runs. A pause does not change it.
    """
    name = str(series.name)
    total = len(series.timestamps)
    run_rows = [0]
    run_starts_s = [origin]
    span_s = 0.0
    steps_done = 0
    previous = origin
    for start in range(1, total, window_rows):
        stop = min(start + window_rows, total)
        times = _read_stamps(series, start, stop)
        steps = np.diff(times, prepend=previous)
        _require_increasing(steps, start, name)
        period = _period(
            span_s + previous - run_starts_s[-1],
            steps_done + start - 1 - run_rows[-1],
            seed,
            seed_steps,
        )
        for jump in np.flatnonzero(np.rint(steps / period) != 1):
            row = start + int(jump)
            before = previous if jump == 0 else float(times[jump - 1])
            span_s += before - run_starts_s[-1]
            steps_done += row - 1 - run_rows[-1]
            period = _period(span_s, steps_done, seed, seed_steps)
            if round(float(steps[jump]) / period) < 1:
                raise ValueError(
                    f"timestamp {row} of {name} is less than half a sample "
                    f"period after timestamp {row - 1}"
                )
            run_rows.append(row)
            run_starts_s.append(float(times[jump]))
        previous = float(times[-1])

    span_s += previous - run_starts_s[-1]
    steps_done += total - 1 - run_rows[-1]
    period = span_s / steps_done if steps_done else seed
    return _Runs(run_rows, run_starts_s, period)


def _check_grid(series: TimeSeries, window_rows: int, found: _Runs) -> None:
    """Raise ValueError at the first timestamp off the grid of its run.

    The grid of a run starts at the timestamp of its first row.
    """
    total = len(series.timestamps)
    firsts = np.asarray(found.rows)
    origins = np.asarray(found.starts_s)
    for start in range(0, total, window_rows):
        stop = min(start + window_rows, total)
        times = _read_stamps(series, start, stop)
        indexes = np.arange(start, stop)
        run = np.searchsorted(firsts, indexes, side="right") - 1
        error = times - (
            origins[run] + (indexes - firsts[run]) * found.period_s
        )
        bad = np.flatnonzero(np.abs(error) > GRID_TOLERANCE_S)
        if bad.size:
            index = int(bad[0])
            raise ValueError(
                f"timestamp {start + index} of {series.name} is "
                f"{error[index] * 1e6:.3f} us off the "
                f"{1 / found.period_s:g} Hz sample grid of its run. A run "
                f"holds samples one sample period apart, and jitter or drift "
                f"beyond {GRID_TOLERANCE_S * 1e6:g} us has no place in it"
            )


def stitch(layouts: Sequence[Layout]) -> Layout:
    """Return one layout that holds the runs of every layout.

    The layouts are the segments of one recording. They are taken in order
    of start_s, and a tie keeps the given order. The sample rate is that of
    the earliest segment with samples. A layout without samples adds its
    series and no run.

    Raises:
        ValueError: A segment starts more than GRID_TOLERANCE_S before the
            previous segment ends, or its sample rate differs from the rate of
            the earliest segment by enough to move the last sample of a run
            more than GRID_TOLERANCE_S off its grid.
    """
    ordered = sorted(layouts, key=lambda layout: layout.start_s)
    base = next(
        (layout for layout in ordered if layout.num_samples), ordered[0]
    )
    period = 1 / base.rate_hz
    runs: list[Run] = []
    count = 0
    end_s = -math.inf
    previous = base
    for layout in ordered:
        if not layout.num_samples:
            continue
        name = layout.series[0].name
        longest = max(run.rows for run in layout.runs)
        drift = (longest - 1) * (1 / layout.rate_hz - period)
        if abs(drift) > GRID_TOLERANCE_S:
            raise ValueError(
                f"{name} drifts {drift * 1e6:.3f} us from the "
                f"{base.rate_hz:g} Hz sample grid of {base.series[0].name} "
                f"within one run. A run must stay within "
                f"{GRID_TOLERANCE_S * 1e6:g} us of its grid, so every segment "
                "needs the sample rate of the earliest segment"
            )
        if layout.start_s < end_s - GRID_TOLERANCE_S:
            raise ValueError(
                f"{name} starts before {previous.series[0].name} ends; "
                "segments of one recording must not overlap"
            )
        runs.extend(
            run._replace(first_sample=run.first_sample + count)
            for run in layout.runs
        )
        count += layout.num_samples
        last = layout.runs[-1]
        end_s = max(end_s, last.start_s + last.rows / layout.rate_hz)
        previous = layout
    series = tuple(segment for layout in ordered for segment in layout.series)
    return Layout(base.rate_hz, base.start_s, tuple(runs), count, series)
