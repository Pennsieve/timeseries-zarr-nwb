"""Run configuration assembled from the environment and command line."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from timeseries_zarr import (
    WINDOW_BATCH_BYTES,
    WriteOpts,
    channel_writer_bytes,
    max_levels_for,
)

from timeseries_zarr_nwb.nwb_reader import WINDOW_BYTES

MAX_ZSTD_LEVEL: Final = 22
"""Highest Zstd compression level the codec accepts."""

MIN_DECIMATION_FACTOR: Final = 2
"""Smallest decimation factor the writer accepts."""

DEFAULT_SWEEP_BYTES: Final = 2 * 2**30
"""Default bytes of memory for the writers and windows of one group of channels."""

WINDOW_RESERVE_BYTES: Final = 2 * (WINDOW_BATCH_BYTES + WINDOW_BYTES)
"""Bytes the sweep reserves for the two batches of windows that the writer holds."""


@dataclass(frozen=True, slots=True)
class Config:
    """Everything one bundle run needs, resolved from env and argv.

    include_identifier adds the NWB file identifier to recording.json.
    sweep_bytes bounds the memory of the channel writers and the two batches
    while one group of channels of a series is written.
    """

    nwb_path: Path
    output_path: Path
    opts: WriteOpts
    include_identifier: bool = False
    sweep_bytes: int = DEFAULT_SWEEP_BYTES

    @property
    def sweep_channels(self) -> int:
        """The most channels whose writers fit in sweep_bytes, at least 1.

        The budget is sweep_bytes less WINDOW_RESERVE_BYTES. Each channel
        costs channel_writer_bytes under opts.
        """
        budget = self.sweep_bytes - WINDOW_RESERVE_BYTES
        return max(1, budget // channel_writer_bytes(self.opts))


def load_config(env: Mapping[str, str], argv: Sequence[str]) -> Config:
    """Resolve a Config from environment variables and command-line arguments.

    argv holds the arguments after the program name: the input NWB path and
    the output path. The writer settings come from the ZARR_WRITER_ prefixed
    variables; unset settings fall back to the WriteOpts defaults.
    ZARR_WRITER_ZSTD_LEVEL takes 0 through MAX_ZSTD_LEVEL.
    ZARR_WRITER_DECIMATION_FACTOR takes MIN_DECIMATION_FACTOR or more.
    ZARR_WRITER_MAX_LEVELS takes 0 through max_levels_for that factor, and
    0 writes raw samples only. ZARR_WRITER_MIN_BINS, ZARR_WRITER_INNER_LEN,
    ZARR_WRITER_TARGET_SHARD_BYTES, ZARR_WRITER_SWEEP_BYTES, and
    ZARR_WRITER_MAX_WORKERS take 1 or more. ZARR_WRITER_SWEEP_BYTES defaults
    to DEFAULT_SWEEP_BYTES, and ZARR_WRITER_MAX_WORKERS to the CPUs available.
    ZARR_WRITER_VAR and ZARR_WRITER_INCLUDE_IDENTIFIER take 0 or 1, and
    ZARR_WRITER_INCLUDE_IDENTIFIER defaults to 0. Raises ValueError if argv
    does not hold exactly two entries or if a setting lies outside its range.
    """
    nwb_path, output_path = _resolve_paths(argv)

    defaults = WriteOpts()
    max_workers = (
        _int_env(env, "ZARR_WRITER_MAX_WORKERS", 1, 1)
        if "ZARR_WRITER_MAX_WORKERS" in env
        else defaults.max_workers
    )
    decimation_factor = _int_env(
        env,
        "ZARR_WRITER_DECIMATION_FACTOR",
        defaults.decimation_factor,
        MIN_DECIMATION_FACTOR,
    )
    max_levels = (
        _int_env(
            env,
            "ZARR_WRITER_MAX_LEVELS",
            0,
            0,
            max_levels_for(decimation_factor),
        )
        if "ZARR_WRITER_MAX_LEVELS" in env
        else defaults.max_levels
    )
    opts = WriteOpts(
        zstd_level=_int_env(
            env,
            "ZARR_WRITER_ZSTD_LEVEL",
            defaults.zstd_level,
            0,
            MAX_ZSTD_LEVEL,
        ),
        decimation_factor=decimation_factor,
        max_levels=max_levels,
        min_bins=_int_env(env, "ZARR_WRITER_MIN_BINS", defaults.min_bins, 1),
        inner_len=_int_env(env, "ZARR_WRITER_INNER_LEN", defaults.inner_len, 1),
        target_shard_bytes=_int_env(
            env,
            "ZARR_WRITER_TARGET_SHARD_BYTES",
            defaults.target_shard_bytes,
            1,
        ),
        write_var=_flag_env(env, "ZARR_WRITER_VAR", defaults.write_var),
        max_workers=max_workers,
    )

    return Config(
        nwb_path=nwb_path,
        output_path=output_path,
        opts=opts,
        include_identifier=_flag_env(
            env, "ZARR_WRITER_INCLUDE_IDENTIFIER", default=False
        ),
        sweep_bytes=_int_env(
            env, "ZARR_WRITER_SWEEP_BYTES", DEFAULT_SWEEP_BYTES, 1
        ),
    )


def _resolve_paths(argv: Sequence[str]) -> tuple[Path, Path]:
    """Return the input NWB path and the output path from argv.

    Raises ValueError unless argv holds exactly those two entries.
    """
    positional_count = 2
    if len(argv) != positional_count:
        raise ValueError(
            "expected the input NWB path and the output path as two "
            f"arguments, got {len(argv)}"
        )
    return Path(argv[0]), Path(argv[1])


def _int_env(
    env: Mapping[str, str],
    key: str,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Return the integer env value for key, or default if it is unset.

    The value must lie in minimum through maximum inclusive. A maximum of
    None leaves it unbounded above. Raises ValueError if the value is present
    but is not an integer or lies outside that range.
    """
    raw = env.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from None
    if maximum is None and value < minimum:
        raise ValueError(f"{key} must be {minimum} or more, got {value}")
    if maximum is not None and not minimum <= value <= maximum:
        raise ValueError(f"{key} must be {minimum} to {maximum}, got {value}")
    return value


def _flag_env(env: Mapping[str, str], key: str, default: bool) -> bool:
    """Return the flag env value for key, or default if it is unset.

    A flag is "0" for off or "1" for on. Raises ValueError for any other
    value, including an empty one.
    """
    raw = env.get(key)
    if raw is None:
        return default
    if raw not in ("0", "1"):
        raise ValueError(f"{key} must be 0 or 1, got {raw!r}")
    return raw == "1"
