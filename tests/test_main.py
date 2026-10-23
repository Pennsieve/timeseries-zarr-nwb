import json
import sys
from datetime import UTC, datetime, timedelta

import h5py
import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from pynwb.testing.mock.file import mock_NWBFile

from timeseries_zarr_nwb.main import main, run


def _write_nwb(path):
    nwb = mock_NWBFile()
    mock_ElectricalSeries(
        name="es", nwbfile=nwb, data=np.zeros((4096, 2)), rate=1000.0
    )
    with NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwb)
    return path


def test_main_converts_a_file_and_writes_the_bundle(tmp_path):
    nwb = _write_nwb(tmp_path / "session.nwb")
    out = tmp_path / "session.zarr"
    assert main([str(nwb), str(out)]) == 0
    assert (out / "zarr.json").exists()
    assert (out / "recording.json").exists()
    assert (out / "0" / "raw" / "zarr.json").exists()


def test_main_replaces_a_bundle_from_an_earlier_run(tmp_path):
    nwb = _write_nwb(tmp_path / "session.nwb")
    out = tmp_path / "session.zarr"
    assert main([str(nwb), str(out)]) == 0
    (out / "9").mkdir()
    assert main([str(nwb), str(out)]) == 0
    assert (out / "0" / "raw" / "zarr.json").exists()
    assert not (out / "9").exists()


def test_main_returns_1_for_an_output_directory_holding_other_files(tmp_path):
    nwb = _write_nwb(tmp_path / "session.nwb")
    out = tmp_path / "session.zarr"
    out.mkdir()
    (out / "notes.txt").write_text("keep")
    assert main([str(nwb), str(out)]) == 1
    assert (out / "notes.txt").read_text() == "keep"
    assert not (out / "zarr.json").exists()


def test_main_honors_writer_settings_from_the_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ZARR_WRITER_VAR", "1")
    nwb = _write_nwb(tmp_path / "session.nwb")
    out = tmp_path / "session.zarr"
    assert main([str(nwb), str(out)]) == 0
    assert (out / "0" / "1" / "var" / "zarr.json").exists()


def test_main_writes_channel_provenance_and_leaves_the_identifier_out(
    tmp_path,
):
    nwb = _write_nwb(tmp_path / "session.nwb")
    out = tmp_path / "session.zarr"
    assert main([str(nwb), str(out)]) == 0
    channel = json.loads((out / "0" / "zarr.json").read_text())
    provenance = channel["attributes"]["tsz"]["provenance"]
    assert provenance["nwb_path"] == "/acquisition/es"
    with h5py.File(nwb, "r") as file:
        assert provenance["nwb_path"] in file
    recording = json.loads((out / "recording.json").read_text())
    assert recording["source"]["format"].startswith("NWB 2.")
    assert "identifier" not in recording["source"]


def test_main_places_data_from_the_reference_time_and_keeps_the_session_start(
    tmp_path,
):
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    nwb = mock_NWBFile(
        session_start_time=started,
        timestamps_reference_time=started - timedelta(seconds=30),
    )
    mock_ElectricalSeries(
        name="es", nwbfile=nwb, data=np.zeros((4096, 2)), rate=1000.0
    )
    path = tmp_path / "session.nwb"
    with NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwb)
    out = tmp_path / "session.zarr"
    assert main([str(path), str(out)]) == 0
    recording = json.loads((out / "recording.json").read_text())
    assert recording["reference_time"] == "2021-06-01T11:59:30.000000+00:00"
    assert (
        recording["session"]["start_time"] == "2021-06-01T12:00:00.000000+00:00"
    )
    channel = json.loads((out / "0" / "zarr.json").read_text())
    assert channel["attributes"]["tsz"]["offset_us"] == 0


def test_main_writes_the_identifier_when_the_environment_asks(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ZARR_WRITER_INCLUDE_IDENTIFIER", "1")
    nwb = _write_nwb(tmp_path / "session.nwb")
    out = tmp_path / "session.zarr"
    assert main([str(nwb), str(out)]) == 0
    with NWBHDF5IO(str(nwb), mode="r") as io:
        identifier = io.read().identifier
    recording = json.loads((out / "recording.json").read_text())
    assert recording["source"]["identifier"] == identifier


@pytest.mark.parametrize("argv", [[], ["only.nwb"], ["a.nwb", "b", "c"]])
def test_main_returns_2_on_a_bad_invocation(argv):
    assert main(argv) == 2


def test_main_returns_2_for_a_setting_outside_its_range(tmp_path, monkeypatch):
    monkeypatch.setenv("ZARR_WRITER_INNER_LEN", "0")
    nwb = _write_nwb(tmp_path / "session.nwb")
    out = tmp_path / "session.zarr"
    assert main([str(nwb), str(out)]) == 2
    assert not out.exists()


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
