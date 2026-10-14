"""Run configuration assembled from the environment and command line."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from timeseries_zarr.constants import MAX_LEVELS
from timeseries_zarr.types import WriteOpts


@dataclass(frozen=True, slots=True)
class Config:
    """Everything one bundle run needs, resolved from env and argv.

    staging_dir is the scratch path the bundle is built in before its atomic
    rename onto final_dir.
    """

    nwb_path: Path
    staging_dir: Path
    final_dir: Path
    opts: WriteOpts


def load_config(env: Mapping[str, str], argv: Sequence[str]) -> Config:
    """Resolve a Config from environment variables and command-line arguments.

    argv holds the arguments after the program name: the input NWB path and
    the final output directory. The writer settings and the staging directory
    come from the ZARR_WRITER_ prefixed variables; unset settings fall back to
    the WriteOpts defaults and an unset staging directory derives from
    final_dir. Raises ValueError on a bad invocation: an argv without exactly
    two entries, a non-integer setting, ZARR_WRITER_MAX_LEVELS outside 1
    through MAX_LEVELS, or ZARR_WRITER_VAR other than 0 or 1.
    """
    nwb_path, final_dir = _resolve_paths(argv)

    defaults = WriteOpts()
    max_levels = _int_env(env, "ZARR_WRITER_MAX_LEVELS", defaults.max_levels)
    if not 1 <= max_levels <= MAX_LEVELS:
        raise ValueError(
            f"ZARR_WRITER_MAX_LEVELS must be 1 to {MAX_LEVELS}, "
            f"got {max_levels}"
        )
    opts = WriteOpts(
        zstd_level=_int_env(env, "ZARR_WRITER_ZSTD_LEVEL", defaults.zstd_level),
        max_levels=max_levels,
        min_bins=_int_env(env, "ZARR_WRITER_MIN_BINS", defaults.min_bins),
        inner_len=_int_env(env, "ZARR_WRITER_INNER_LEN", defaults.inner_len),
        target_shard_bytes=_int_env(
            env, "ZARR_WRITER_TARGET_SHARD_BYTES", defaults.target_shard_bytes
        ),
        write_var=_flag_env(env, "ZARR_WRITER_VAR", defaults.write_var),
    )

    staging_raw = env.get("ZARR_WRITER_STAGING_DIR")
    staging_dir = (
        Path(staging_raw)
        if staging_raw is not None
        else final_dir.with_name(final_dir.name + ".staging")
    )

    return Config(
        nwb_path=nwb_path,
        staging_dir=staging_dir,
        final_dir=final_dir,
        opts=opts,
    )


def _resolve_paths(argv: Sequence[str]) -> tuple[Path, Path]:
    """Return the (input NWB, final output dir) paths from argv.

    Raises ValueError unless argv holds exactly those two entries.
    """
    positional_count = 2
    if len(argv) != positional_count:
        raise ValueError(
            "expected the input NWB path and the output directory as two "
            f"arguments, got {len(argv)}"
        )
    return Path(argv[0]), Path(argv[1])


def _int_env(env: Mapping[str, str], key: str, default: int) -> int:
    """Return the integer env value for key, or default if it is unset.

    Raises ValueError if the value is present but not an integer.
    """
    raw = env.get(key)
    return default if raw is None else int(raw)


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
