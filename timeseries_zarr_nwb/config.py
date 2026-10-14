"""Run configuration assembled from the environment and command line."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from timeseries_zarr.constants import MAX_LEVELS
from timeseries_zarr.types import WriteOpts

MAX_ZSTD_LEVEL: Final = 22
"""Highest Zstd compression level the codec accepts."""


@dataclass(frozen=True, slots=True)
class Config:
    """Everything one bundle run needs, resolved from env and argv."""

    nwb_path: Path
    output_path: Path
    opts: WriteOpts


def load_config(env: Mapping[str, str], argv: Sequence[str]) -> Config:
    """Resolve a Config from environment variables and command-line arguments.

    argv holds the arguments after the program name: the input NWB path and
    the output path. The writer settings come from the ZARR_WRITER_ prefixed
    variables; unset settings fall back to the WriteOpts defaults.
    ZARR_WRITER_ZSTD_LEVEL takes 0 through MAX_ZSTD_LEVEL and
    ZARR_WRITER_MAX_LEVELS takes 1 through MAX_LEVELS. ZARR_WRITER_MIN_BINS,
    ZARR_WRITER_INNER_LEN, and ZARR_WRITER_TARGET_SHARD_BYTES take 1 or more.
    ZARR_WRITER_VAR takes 0 or 1. Raises ValueError if argv does not hold
    exactly two entries or if a setting lies outside its range.
    """
    nwb_path, output_path = _resolve_paths(argv)

    defaults = WriteOpts()
    opts = WriteOpts(
        zstd_level=_int_env(
            env,
            "ZARR_WRITER_ZSTD_LEVEL",
            defaults.zstd_level,
            0,
            MAX_ZSTD_LEVEL,
        ),
        max_levels=_int_env(
            env, "ZARR_WRITER_MAX_LEVELS", defaults.max_levels, 1, MAX_LEVELS
        ),
        min_bins=_int_env(env, "ZARR_WRITER_MIN_BINS", defaults.min_bins, 1),
        inner_len=_int_env(env, "ZARR_WRITER_INNER_LEN", defaults.inner_len, 1),
        target_shard_bytes=_int_env(
            env,
            "ZARR_WRITER_TARGET_SHARD_BYTES",
            defaults.target_shard_bytes,
            1,
        ),
        write_var=_flag_env(env, "ZARR_WRITER_VAR", defaults.write_var),
    )

    return Config(nwb_path=nwb_path, output_path=output_path, opts=opts)


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
