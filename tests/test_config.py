from pathlib import Path

import pytest
from timeseries_zarr import MAX_LEVELS, WriteOpts

from timeseries_zarr_nwb.config import MAX_ZSTD_LEVEL, Config, load_config

ARGV = ["in.nwb", "out_bundle"]


def test_two_positionals_populate_paths():
    cfg = load_config({}, ["in.nwb", "out_bundle"])
    assert isinstance(cfg, Config)
    assert cfg.nwb_path == Path("in.nwb")
    assert cfg.output_path == Path("out_bundle")


def test_unset_settings_use_writeopts_defaults():
    cfg = load_config({}, ["in.nwb", "out_bundle"])
    assert cfg.opts == WriteOpts()


def test_env_settings_override_defaults():
    cfg = load_config(
        {
            "ZARR_WRITER_ZSTD_LEVEL": "9",
            "ZARR_WRITER_MAX_LEVELS": "4",
            "ZARR_WRITER_MIN_BINS": "512",
            "ZARR_WRITER_INNER_LEN": "1024",
            "ZARR_WRITER_TARGET_SHARD_BYTES": "2048",
            "ZARR_WRITER_VAR": "1",
        },
        ["in.nwb", "out_bundle"],
    )
    assert cfg.opts == WriteOpts(
        zstd_level=9,
        max_levels=4,
        min_bins=512,
        inner_len=1024,
        target_shard_bytes=2048,
        write_var=True,
    )


@pytest.mark.parametrize(
    ("key", "value", "valid"),
    [
        ("ZARR_WRITER_ZSTD_LEVEL", -1, f"0 to {MAX_ZSTD_LEVEL}"),
        (
            "ZARR_WRITER_ZSTD_LEVEL",
            MAX_ZSTD_LEVEL + 1,
            f"0 to {MAX_ZSTD_LEVEL}",
        ),
        ("ZARR_WRITER_MAX_LEVELS", 0, f"1 to {MAX_LEVELS}"),
        ("ZARR_WRITER_MAX_LEVELS", MAX_LEVELS + 1, f"1 to {MAX_LEVELS}"),
        ("ZARR_WRITER_MIN_BINS", 0, "1 or more"),
        ("ZARR_WRITER_INNER_LEN", 0, "1 or more"),
        ("ZARR_WRITER_INNER_LEN", -1, "1 or more"),
        ("ZARR_WRITER_TARGET_SHARD_BYTES", 0, "1 or more"),
    ],
)
def test_a_setting_outside_its_range_raises(key, value, valid):
    with pytest.raises(
        ValueError, match=f"^{key} must be {valid}, got {value}$"
    ):
        load_config({key: str(value)}, ARGV)


@pytest.mark.parametrize(
    ("key", "value", "field"),
    [
        ("ZARR_WRITER_ZSTD_LEVEL", 0, "zstd_level"),
        ("ZARR_WRITER_ZSTD_LEVEL", MAX_ZSTD_LEVEL, "zstd_level"),
        ("ZARR_WRITER_MAX_LEVELS", 1, "max_levels"),
        ("ZARR_WRITER_MAX_LEVELS", MAX_LEVELS, "max_levels"),
        ("ZARR_WRITER_MIN_BINS", 1, "min_bins"),
        ("ZARR_WRITER_INNER_LEN", 1, "inner_len"),
        ("ZARR_WRITER_TARGET_SHARD_BYTES", 1, "target_shard_bytes"),
    ],
)
def test_accepts_a_setting_at_an_end_of_its_range(key, value, field):
    cfg = load_config({key: str(value)}, ARGV)
    assert getattr(cfg.opts, field) == value


def test_zero_positionals_raises():
    with pytest.raises(ValueError):
        load_config({}, [])


def test_one_positional_raises():
    with pytest.raises(ValueError):
        load_config({}, ["in.nwb"])


def test_three_positionals_raise():
    with pytest.raises(ValueError):
        load_config({}, ["in.nwb", "out_bundle", "extra"])


def test_zero_positionals_raise_when_input_and_output_dirs_are_set(tmp_path):
    (tmp_path / "session.nwb").write_bytes(b"")
    with pytest.raises(ValueError):
        load_config(
            {"INPUT_DIR": str(tmp_path), "OUTPUT_DIR": str(tmp_path / "out")},
            [],
        )


@pytest.mark.parametrize("value", ["high", "", "1.5"])
def test_non_integer_setting_raises(value):
    with pytest.raises(
        ValueError,
        match=f"^ZARR_WRITER_INNER_LEN must be an integer, got {value!r}$",
    ):
        load_config({"ZARR_WRITER_INNER_LEN": value}, ARGV)


def test_var_is_off_when_the_variable_is_unset():
    cfg = load_config({}, ["in.nwb", "out_bundle"])
    assert cfg.opts.write_var is False


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("0", False)])
def test_var_follows_a_zero_or_one_flag(value, expected):
    cfg = load_config({"ZARR_WRITER_VAR": value}, ["in.nwb", "out_bundle"])
    assert cfg.opts.write_var is expected


@pytest.mark.parametrize("value", ["", "true", "yes", "2", " 1", "01"])
def test_var_rejects_any_value_but_zero_or_one(value):
    with pytest.raises(ValueError, match="ZARR_WRITER_VAR must be 0 or 1"):
        load_config({"ZARR_WRITER_VAR": value}, ["in.nwb", "out_bundle"])
