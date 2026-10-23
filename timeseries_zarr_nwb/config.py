"""Run configuration assembled from the environment and command line."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from timeseries_zarr import MAX_LEVELS, WriteOpts

MAX_ZSTD_LEVEL: Final = 22
"""Highest Zstd compression level the codec accepts."""

DEFAULT_SWEEP_BYTES: Final = 2 * 2**30
"""Default bytes of raw shards held while one group of channels is read."""

RAW_SAMPLE_BYTES: Final = 4
"""Bytes of one raw sample, which a bundle stores as float32."""


@dataclass(frozen=True, slots=True)
class Config:
    """Everything one bundle run needs, resolved from env and argv.

    include_identifier adds the NWB file identifier to recording.json.
    sweep_bytes bounds the raw shards held while one group of channels of a
    series is read.
    """

    nwb_path: Path
    output_path: Path
    opts: WriteOpts
    include_identifier: bool = False
    sweep_bytes: int = DEFAULT_SWEEP_BYTES

    @property
    def raw_shard_bytes(self) -> int:
        """The most bytes one raw shard of a channel holds under opts.

        A shard holds the whole inner chunks that fit in target_shard_bytes,
        and at least one.
        """
        chunk_bytes = self.opts.inner_len * RAW_SAMPLE_BYTES
        chunks = max(1, self.opts.target_shard_bytes // chunk_bytes)
        return chunks * chunk_bytes

    @property
    def sweep_channels(self) -> int:
        """The most channels whose raw shards fit in sweep_bytes, at least 1."""
        return max(1, self.sweep_bytes // self.raw_shard_bytes)


def load_config(env: Mapping[str, str], argv: Sequence[str]) -> Config:
    """Resolve a Config from environment variables and command-line arguments.

    argv holds the arguments after the program name: the input NWB path and
    the output path. The writer settings come from the ZARR_WRITER_ prefixed
    variables; unset settings fall back to the WriteOpts defaults.
    ZARR_WRITER_ZSTD_LEVEL takes 0 through MAX_ZSTD_LEVEL.
    ZARR_WRITER_MAX_LEVELS takes 0 through MAX_LEVELS, and 0 writes raw
    samples only. ZARR_WRITER_MIN_BINS, ZARR_WRITER_INNER_LEN,
    ZARR_WRITER_TARGET_SHARD_BYTES, and ZARR_WRITER_SWEEP_BYTES take 1 or
    more, and ZARR_WRITER_SWEEP_BYTES defaults to DEFAULT_SWEEP_BYTES.
    ZARR_WRITER_VAR and ZARR_WRITER_INCLUDE_IDENTIFIER take 0 or 1, and
    ZARR_WRITER_INCLUDE_IDENTIFIER defaults to 0. Raises ValueError if argv
    does not hold exactly two entries or if a setting lies outside its range.
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
            env, "ZARR_WRITER_MAX_LEVELS", defaults.max_levels, 0, MAX_LEVELS
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
