from pathlib import Path

import pytest
from timeseries_zarr.constants import MAX_LEVELS
from timeseries_zarr.types import WriteOpts

from timeseries_zarr_nwb.config import Config, load_config


def test_two_positionals_populate_paths():
    cfg = load_config({}, ["in.nwb", "out_bundle"])
    assert isinstance(cfg, Config)
    assert cfg.nwb_path == Path("in.nwb")
    assert cfg.final_dir == Path("out_bundle")


def test_unset_settings_use_writeopts_defaults():
    cfg = load_config({}, ["in.nwb", "out_bundle"])
    assert cfg.opts == WriteOpts()


def test_unset_staging_derives_from_final_dir():
    cfg = load_config({}, ["in.nwb", "out_bundle"])
    assert cfg.staging_dir != cfg.final_dir
    assert cfg.staging_dir.parent == cfg.final_dir.parent


def test_env_staging_dir_honored():
    cfg = load_config(
        {"ZARR_WRITER_STAGING_DIR": "scratch/stage"},
        ["in.nwb", "out_bundle"],
    )
    assert cfg.staging_dir == Path("scratch/stage")


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


def test_max_levels_above_the_format_cap_raises():
    with pytest.raises(ValueError):
        load_config(
            {"ZARR_WRITER_MAX_LEVELS": str(MAX_LEVELS + 1)},
            ["in.nwb", "out_bundle"],
        )


def test_max_levels_below_one_raises():
    with pytest.raises(ValueError):
        load_config({"ZARR_WRITER_MAX_LEVELS": "0"}, ["in.nwb", "out_bundle"])


def test_max_levels_at_the_cap_is_accepted():
    cfg = load_config(
        {"ZARR_WRITER_MAX_LEVELS": str(MAX_LEVELS)}, ["in.nwb", "out_bundle"]
    )
    assert cfg.opts.max_levels == MAX_LEVELS


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


def test_non_integer_setting_raises():
    with pytest.raises(ValueError):
        load_config(
            {"ZARR_WRITER_ZSTD_LEVEL": "high"}, ["in.nwb", "out_bundle"]
        )


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
