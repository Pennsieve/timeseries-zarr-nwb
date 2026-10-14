import sys

import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from pynwb.testing.mock.file import mock_NWBFile

from timeseries_zarr_nwb.main import main, run


def _write_nwb(path):
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((4096, 2)), rate=1000.0)
    with NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwb)
    return path


def test_main_converts_a_file_and_publishes_the_bundle(tmp_path):
    nwb = _write_nwb(tmp_path / "session.nwb")
    out = tmp_path / "session.zarr"
    assert main([str(nwb), str(out)]) == 0
    assert (out / "zarr.json").exists()
    assert (out / "recording.json").exists()
    assert (out / "0" / "raw" / "zarr.json").exists()
    assert not (tmp_path / "session.zarr.staging").exists()


def test_main_honors_writer_settings_from_the_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ZARR_WRITER_VAR", "1")
    nwb = _write_nwb(tmp_path / "session.nwb")
    out = tmp_path / "session.zarr"
    assert main([str(nwb), str(out)]) == 0
    assert (out / "0" / "1" / "var" / "zarr.json").exists()


@pytest.mark.parametrize("argv", [[], ["only.nwb"], ["a.nwb", "b", "c"]])
def test_main_returns_2_on_a_bad_invocation(argv):
    assert main(argv) == 2


def test_main_returns_1_when_the_input_cannot_be_read(tmp_path):
    missing = tmp_path / "missing.nwb"
    assert main([str(missing), str(tmp_path / "out.zarr")]) == 1
    assert not (tmp_path / "out.zarr").exists()


def test_run_exits_with_the_main_code(tmp_path, monkeypatch):
    nwb = _write_nwb(tmp_path / "session.nwb")
    out = tmp_path / "session.zarr"
    monkeypatch.setattr(
        sys, "argv", ["timeseries-zarr-nwb", str(nwb), str(out)]
    )
    with pytest.raises(SystemExit) as exit_info:
        run()
    assert exit_info.value.code == 0
    assert (out / "zarr.json").exists()
