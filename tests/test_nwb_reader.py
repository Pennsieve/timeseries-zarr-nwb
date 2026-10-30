from datetime import UTC, datetime, timedelta
from importlib.metadata import version

import numpy as np
import pytest
import zarr
from hdmf.common import DynamicTableRegion
from pynwb import NWBHDF5IO, H5DataIO, NWBFile, TimeSeries
from pynwb.base import ProcessingModule
from pynwb.device import Device, DeviceModel
from pynwb.ecephys import ElectricalSeries
from pynwb.epoch import TimeIntervals
from pynwb.event import EventsTable
from pynwb.file import Subject
from pynwb.misc import AnnotationSeries, Units
from pynwb.testing.mock.base import mock_TimeSeries
from pynwb.testing.mock.device import mock_Device
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from pynwb.testing.mock.file import mock_NWBFile
from timeseries_zarr import (
    EventChannelSource,
    MultiChannelSource,
    Run,
    WriteOpts,
    write_bundle,
)
from timeseries_zarr import __version__ as writer_version

from timeseries_zarr_nwb import nwb_reader
from timeseries_zarr_nwb.nwb_reader import (
    NwbContinuousSource,
    NwbSeriesSource,
    NwbTimeSeriesSource,
    NwbUnitSource,
    build_sources_from_nwb,
    recording_meta,
)

STARTED = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)


def _channels(continuous):
    """Return the channels of every series source, in bundle order."""
    return [channel for source in continuous for channel in source.channels()]


def _series_with_electrode_columns(extra_columns, n=3):
    """Build an ElectricalSeries whose electrodes table carries extra columns.

    extra_columns maps a column name to its per-electrode values. The returned
    series has n channels wired to those n electrodes in order.
    """
    nwb = mock_NWBFile()
    device = mock_Device(nwbfile=nwb)
    group = nwb.create_electrode_group(
        name="grp", description="d", location="loc", device=device
    )
    for column in extra_columns:
        nwb.add_electrode_column(name=column, description=column)
    for i in range(n):
        nwb.add_electrode(
            group=group,
            location="loc",
            **{col: values[i] for col, values in extra_columns.items()},
        )
    region = DynamicTableRegion(
        name="electrodes",
        data=list(range(n)),
        description="r",
        table=nwb.electrodes,
    )
    return mock_ElectricalSeries(
        nwbfile=nwb, data=np.zeros((4, n)), electrodes=region
    )


def _make_units(unit_specs, *, name="my_units", unit_names=None):
    """Build a pynwb Units table from (spike_times, waveform_mean) pairs."""
    units = Units(name=name)
    if unit_names is not None:
        units.add_column(name="unit_name", description="unit name")
    for index, (spike_times, waveform_mean) in enumerate(unit_specs):
        extra = {} if unit_names is None else {"unit_name": unit_names[index]}
        units.add_unit(
            spike_times=list(spike_times),
            waveform_mean=np.asarray(waveform_mean, dtype=float),
            **extra,
        )
    return units


# Two units whose spikes interleave in time; dense cluster ids are 0 and 1.
# Sorted events: 0.0(u0) 0.1(u1) 0.2(u0) 0.3(u1) 0.5(u0).
_WM0 = [1.0, 2.0, 3.0, 4.0]
_WM1 = [5.0, 6.0, 7.0, 8.0]
_TWO_UNITS = [([0.0, 0.2, 0.5], _WM0), ([0.1, 0.3], _WM1)]
_SORTED_TIMES = [0.0, 0.1, 0.2, 0.3, 0.5]
_SORTED_CLUSTERS = [0, 1, 0, 1, 0]
_SORTED_WAVEFORMS = [_WM0, _WM1, _WM0, _WM1, _WM0]


def test_continuous_init_stores_references(electrical_series):
    es = electrical_series()
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert src._series is es
    assert src._channel_index == 2
    assert src._reference_time == started


def test_rate_hz_from_series_rate(electrical_series):
    es = electrical_series()  # mock rate is 30000.0 Hz
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert src.rate_hz() == 30000.0


def test_num_samples_is_time_axis_length(electrical_series):
    es = electrical_series()  # mock data shape is (10, 5)
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert src.num_samples() == 10


def test_start_us_is_reference_time_plus_offset_in_microseconds(
    electrical_series,
):
    es = electrical_series()  # mock starting_time is 0.0 s
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    expected = round(started.timestamp() * 1_000_000)
    assert src.start_us() == expected


def test_id_is_a_string(electrical_series):
    es = electrical_series()
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert isinstance(src.id, str)


def test_read_samples_returns_the_channel_column(electrical_series):
    # The mock's affine scaling is identity; 1e6 converts volts to microvolts.
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    expected = (np.asarray(es.data[1:5, 2], dtype=np.float64) * 1e6).astype(
        np.float32
    )
    assert np.array_equal(src.read_samples(1, 5), expected)


def test_read_samples_dtype_is_float32(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.read_samples(0, 4).dtype == np.float32


def test_read_samples_shape_matches_window(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.read_samples(2, 7).shape == (5,)


def test_read_samples_empty_range_is_length_zero(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.read_samples(3, 3).shape == (0,)


def test_read_samples_applies_conversion_and_offset():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    es = mock_ElectricalSeries(data=data, conversion=2.0, offset=10.0)
    src = NwbContinuousSource(es, 2, STARTED)
    expected = ((data[0:4, 2] * 2.0 + 10.0) * 1e6).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)


def test_read_samples_applies_channel_conversion():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    channel_conversion = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    es = mock_ElectricalSeries(
        data=data, conversion=2.0, channel_conversion=channel_conversion
    )
    src = NwbContinuousSource(es, 2, STARTED)
    expected = ((data[0:4, 2] * 2.0 * 3.0) * 1e6).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)


def test_read_samples_without_channel_conversion_skips_it():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    es = mock_ElectricalSeries(data=data, conversion=2.0)
    assert es.channel_conversion is None
    src = NwbContinuousSource(es, 2, STARTED)
    expected = ((data[0:4, 2] * 2.0) * 1e6).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)


def test_read_samples_normalizes_millivolts_to_microvolts():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    es = mock_ElectricalSeries(data=data)
    es.fields["unit"] = "millivolts"
    src = NwbContinuousSource(es, 2, STARTED)
    expected = (data[0:4, 2] * 1e3).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)


def test_continuous_rejects_a_unit_outside_the_volts_family():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    es = mock_ElectricalSeries(data=data)
    es.fields["unit"] = "furlongs"
    with pytest.raises(ValueError, match="furlongs"):
        NwbContinuousSource(es, 2, STARTED)


def test_continuous_unit_is_microvolts(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.unit == "uV"


def test_continuous_name_from_channel_name_column():
    es = _series_with_electrode_columns({"channel_name": ["a", "b", "c"]})
    src = NwbContinuousSource(es, 1, STARTED)
    assert src.name == "b"


def test_continuous_name_falls_back_to_label_column():
    es = _series_with_electrode_columns({"label": ["x", "y", "z"]})
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.name == "z"


def test_continuous_name_prefers_channel_name_over_label():
    es = _series_with_electrode_columns(
        {"channel_name": ["a", "b", "c"], "label": ["x", "y", "z"]}
    )
    src = NwbContinuousSource(es, 0, STARTED)
    assert src.name == "a"


def test_continuous_name_falls_back_to_id_when_no_columns(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.name == src.id


def test_unit_num_events_sums_all_units_spikes():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.num_events() == 5


def test_unit_id_is_the_table_name():
    src = NwbUnitSource(
        _make_units(_TWO_UNITS, name="probeA"), 30000.0, STARTED
    )
    assert src.id == "probeA"


def test_unit_name_is_the_table_name():
    src = NwbUnitSource(
        _make_units(_TWO_UNITS, name="probeA"), 30000.0, STARTED
    )
    assert src.name == "probeA"


def test_unit_has_no_unit_because_spikes_have_no_measurement():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.unit is None


def test_unit_read_events_is_ascending_absolute_microseconds():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    base = STARTED.timestamp()
    expected = np.array(
        [round((base + t) * 1_000_000) for t in _SORTED_TIMES],
        dtype=np.int64,
    )
    events = src.read_events(0, 5)
    assert events.dtype == np.int64
    assert np.array_equal(events, expected)
    assert np.all(np.diff(events) >= 0)


def test_unit_read_events_window_is_a_subset():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert np.array_equal(src.read_events(1, 3), src.read_events(0, 5)[1:3])


def test_unit_read_events_empty_range_is_length_zero():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.read_events(2, 2).shape == (0,)


def test_unit_init_rejects_more_units_than_a_uint16_label_holds(monkeypatch):
    monkeypatch.setattr("timeseries_zarr_nwb.nwb_reader.MAX_LABELS", 3)
    specs = [([float(i)], [0.0]) for i in range(4)]
    with pytest.raises(ValueError, match="at most 3 units"):
        NwbUnitSource(_make_units(specs), 30000.0, STARTED)


def test_unit_source_serves_the_event_protocol():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert isinstance(src, EventChannelSource)


def test_unit_read_labels_are_dense_uint16_indices_in_time_order():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    labels = src.read_labels(0, 5)
    assert labels.dtype == np.uint16
    assert labels.tolist() == _SORTED_CLUSTERS
    assert src.read_labels(1, 3).tolist() == _SORTED_CLUSTERS[1:3]
    assert src.read_labels(2, 2).shape == (0,)


def test_unit_source_holds_more_than_256_units_as_labels():
    specs = [([float(i)], [0.0]) for i in range(300)]
    src = NwbUnitSource(_make_units(specs), 30000.0, STARTED)
    assert src.num_labels() == 300
    assert int(src.read_labels(0, 300).max()) == 299


def test_unit_label_names_default_to_the_unit_ids():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.label_names() == ["0", "1"]
    assert src.num_labels() == 2


def test_unit_label_names_use_the_unit_name_column_when_present():
    units = _make_units(_TWO_UNITS, unit_names=["alpha", "beta"])
    src = NwbUnitSource(units, 30000.0, STARTED)
    assert src.label_names() == ["alpha", "beta"]


def test_unit_columns_are_labels_and_waveform_mean():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.columns() == {"labels", "waveform_mean"}


def test_unit_waveform_mean_is_the_per_unit_table_in_float32():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    means = src.waveform_mean()
    assert means.dtype == np.float32
    assert means.tolist() == [_WM0, _WM1]


def test_unit_waveform_period_is_the_sample_period_in_microseconds():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.waveform_period_us() == 1_000_000 / 30000.0


@pytest.mark.parametrize("rate_hz", [0.0, -30000.0])
def test_unit_waveform_period_rejects_a_non_positive_rate(rate_hz):
    src = NwbUnitSource(_make_units(_TWO_UNITS), rate_hz, STARTED)
    with pytest.raises(ValueError, match="must be positive"):
        src.waveform_period_us()


def test_unit_without_a_waveform_mean_column_offers_only_labels():
    units = Units(name="bare")
    units.add_unit(spike_times=[0.1, 0.2])
    src = NwbUnitSource(units, 30000.0, STARTED)
    assert src.columns() == {"labels"}
    with pytest.raises(ValueError, match="no waveform_mean"):
        src.waveform_mean()


def test_unit_spikes_carry_no_durations_text_or_channels():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.max_duration_us() is None
    assert src.text_width() == 0
    for reader in (src.read_durations, src.read_text, src.read_channel_ids):
        with pytest.raises(ValueError, match="has no"):
            reader(0, 2)


def test_unit_table_without_units_has_no_events_and_no_columns():
    src = NwbUnitSource(Units(name="none"), 30000.0, STARTED)
    assert src.num_events() == 0
    assert src.num_labels() == 0
    assert src.columns() == frozenset()


def test_unit_table_whose_units_have_no_spikes_has_labels_and_no_events():
    specs = [([], _WM0), ([], _WM1)]
    src = NwbUnitSource(_make_units(specs), 30000.0, STARTED)
    assert src.num_events() == 0
    assert src.columns() == {"labels", "waveform_mean"}
    assert src.read_labels(0, 0).shape == (0,)


def test_build_yields_one_channel_per_column():
    nwb = mock_NWBFile()
    data = np.arange(50, dtype=np.float64).reshape(10, 5)
    mock_ElectricalSeries(nwbfile=nwb, data=data, rate=30000.0)
    continuous, units, _ = build_sources_from_nwb(nwb)
    assert len(continuous) == 1
    assert len(_channels(continuous)) == 5
    assert units == []
    assert np.array_equal(
        _channels(continuous)[2].read_samples(0, 10),
        (data[:, 2] * 1e6).astype(np.float32),
    )


def test_build_yields_unit_source_with_first_series_rate():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    nwb.units = _make_units(_TWO_UNITS, name="units")
    continuous, units, _ = build_sources_from_nwb(nwb)
    assert len(units) == 1
    assert units[0].waveform_period_us() == 1_000_000 / 30000.0
    assert units[0].num_events() == 5


def test_build_with_no_electrical_series_yields_no_continuous():
    nwb = mock_NWBFile()
    continuous, units, _ = build_sources_from_nwb(nwb)
    assert continuous == []
    assert units == []


def test_build_with_no_units_yields_no_unit_sources():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    _, units, _ = build_sources_from_nwb(nwb)
    assert units == []


def test_build_units_without_series_rate_raises():
    nwb = mock_NWBFile()
    nwb.units = _make_units(_TWO_UNITS, name="units")
    with pytest.raises(ValueError):
        build_sources_from_nwb(nwb)


def _add_units_to_module(nwb, module_name, units_table):
    """Attach a Units table to a processing module, creating the module if absent."""
    if module_name in nwb.processing:
        module = nwb.processing[module_name]
    else:
        module = nwb.create_processing_module(name=module_name, description="d")
    module.add(units_table)


def test_build_discovers_units_in_a_processing_module():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    _add_units_to_module(nwb, "ecephys", _make_units(_TWO_UNITS, name="sorted"))
    _, units, _ = build_sources_from_nwb(nwb)
    assert len(units) == 1
    assert units[0].id == "sorted"


def test_build_orders_root_units_before_module_units():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    nwb.units = _make_units(_TWO_UNITS, name="units")
    _add_units_to_module(nwb, "ecephys", _make_units(_TWO_UNITS, name="module"))
    _, units, _ = build_sources_from_nwb(nwb)
    assert [u.id for u in units] == ["units", "module"]


def test_build_orders_modules_by_name():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    _add_units_to_module(nwb, "zebra", _make_units(_TWO_UNITS, name="z_units"))
    _add_units_to_module(nwb, "alpha", _make_units(_TWO_UNITS, name="a_units"))
    _, units, _ = build_sources_from_nwb(nwb)
    assert [u.id for u in units] == ["a_units", "z_units"]


def test_build_orders_containers_within_a_module_by_name():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    _add_units_to_module(nwb, "ecephys", _make_units(_TWO_UNITS, name="second"))
    _add_units_to_module(nwb, "ecephys", _make_units(_TWO_UNITS, name="first"))
    _, units, _ = build_sources_from_nwb(nwb)
    assert [u.id for u in units] == ["first", "second"]


def test_build_module_units_without_series_rate_raises():
    nwb = mock_NWBFile()
    _add_units_to_module(nwb, "ecephys", _make_units(_TWO_UNITS, name="sorted"))
    with pytest.raises(ValueError):
        build_sources_from_nwb(nwb)


def _one_channel_series(data, *, rate=30000.0, **series_fields):
    """Build a rank-1 ElectricalSeries wired to one electrode, in a new file."""
    nwb = mock_NWBFile()
    device = mock_Device(nwbfile=nwb)
    group = nwb.create_electrode_group(
        name="grp", description="d", location="loc", device=device
    )
    nwb.add_electrode(group=group, location="loc")
    region = DynamicTableRegion(
        name="electrodes", data=[0], description="r", table=nwb.electrodes
    )
    series = ElectricalSeries(
        name="single",
        data=np.asarray(data),
        electrodes=region,
        rate=rate,
        starting_time=0.0,
        **series_fields,
    )
    nwb.add_acquisition(series)
    return nwb, series


def test_continuous_reads_a_rank_one_series_as_one_channel():
    nwb, es = _one_channel_series(np.arange(8, dtype=np.float64))
    src = NwbContinuousSource(es, 0, STARTED)
    assert src.num_samples() == 8
    assert np.array_equal(
        src.read_samples(2, 5), (np.arange(2, 5) * 1e6).astype(np.float32)
    )
    continuous, _, _ = build_sources_from_nwb(nwb)
    assert len(_channels(continuous)) == 1


def test_continuous_takes_the_rate_and_start_of_a_series_from_its_timestamps():
    data = np.arange(20, dtype=np.float64).reshape(10, 2)
    es = mock_ElectricalSeries(
        data=data, timestamps=2.0 + np.arange(10) / 512.0
    )
    src = NwbContinuousSource(es, 1, STARTED)
    assert src.rate_hz() == pytest.approx(512.0)
    assert src.start_us() == _STARTED_US + 2_000_000
    assert src.num_samples() == 10
    assert np.array_equal(
        src.read_samples(0, 10), (data[:, 1] * 1e6).astype(np.float32)
    )


def test_timeseries_source_keeps_a_unit_outside_the_volts_family():
    ts = mock_TimeSeries(
        name="SpO2",
        data=np.arange(10, dtype=np.float64),
        unit="%",
        rate=512.0,
        conversion=2.0,
        offset=1.0,
    )
    src = NwbTimeSeriesSource(ts, 0, STARTED)
    assert src.id == "SpO2"
    assert src.name == "SpO2"
    assert src.unit == "%"
    assert src.rate_hz() == 512.0
    assert src.num_samples() == 10
    assert src.start_us() == round(STARTED.timestamp() * 1_000_000)
    expected = (np.arange(4) * 2.0 + 1.0).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)
    assert src.read_samples(0, 4).dtype == np.float32


def test_timeseries_source_normalizes_a_volts_family_unit_to_microvolts():
    ts = mock_TimeSeries(
        name="aux", data=np.arange(10, dtype=np.float64), unit="mV", rate=512.0
    )
    src = NwbTimeSeriesSource(ts, 0, STARTED)
    assert src.unit == "uV"
    assert np.array_equal(
        src.read_samples(0, 3), (np.arange(3) * 1e3).astype(np.float32)
    )


@pytest.mark.parametrize("unit", ["V", "mV", "uV", "nV", "Volts", "MICROVOLTS"])
def test_timeseries_source_converts_a_voltage_unit_as_written(unit):
    ts = mock_TimeSeries(
        name="aux", data=np.arange(10, dtype=np.float64), unit=unit, rate=512.0
    )
    assert NwbTimeSeriesSource(ts, 0, STARTED).unit == "uV"


def test_timeseries_source_keeps_a_unit_that_matches_only_lowercased():
    ts = mock_TimeSeries(
        name="aux", data=np.arange(10, dtype=np.float64), unit="MV", rate=512.0
    )
    assert NwbTimeSeriesSource(ts, 0, STARTED).unit == "MV"


def test_continuous_source_reports_an_unsupported_unit_as_written(
    electrical_series,
):
    es = electrical_series()
    es.fields["unit"] = "MV"
    with pytest.raises(
        ValueError, match="unsupported ElectricalSeries unit: 'MV'"
    ):
        NwbContinuousSource(es, 0, STARTED)


def test_timeseries_source_names_the_columns_of_a_rank_two_series():
    data = np.arange(20, dtype=np.float64).reshape(10, 2)
    ts = mock_TimeSeries(name="resp", data=data, unit="cmH2O", rate=512.0)
    first = NwbTimeSeriesSource(ts, 0, STARTED)
    second = NwbTimeSeriesSource(ts, 1, STARTED)
    assert (first.id, second.id) == ("resp/0", "resp/1")
    assert np.array_equal(
        second.read_samples(0, 3), data[0:3, 1].astype(np.float32)
    )


def test_timeseries_source_reads_recorded_samples_across_a_pause():
    stamps = np.concatenate([np.arange(4), 10 + np.arange(3)]) / 512.0
    ts = mock_TimeSeries(
        name="gapped",
        data=np.arange(7, dtype=np.float64),
        unit="%",
        timestamps=stamps,
    )
    src = NwbTimeSeriesSource(ts, 0, STARTED)
    assert src.num_samples() == 7
    expected = np.arange(7, dtype=np.float32)
    assert np.array_equal(src.read_samples(0, 7), expected)
    assert np.array_equal(src.read_samples(2, 6), expected[2:6])
    started_us = round(STARTED.timestamp() * 1_000_000)
    assert src.runs() == [
        Run(started_us, 4),
        Run(started_us + round(10 / 512.0 * 1_000_000), 3),
    ]
    assert src.start_us() == src.runs()[0].start_us


def test_timeseries_source_rejects_data_above_rank_two():
    ts = mock_TimeSeries(
        name="video", data=np.zeros((4, 2, 2)), unit="n/a", rate=1.0
    )
    with pytest.raises(ValueError, match="rank"):
        NwbTimeSeriesSource(ts, 0, STARTED)


def test_build_yields_plain_series_columns_after_electrical_series_columns():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 2)), rate=512.0)
    mock_TimeSeries(
        name="SpO2", nwbfile=nwb, data=np.zeros(10), unit="%", rate=512.0
    )
    mock_TimeSeries(
        name="resp",
        nwbfile=nwb,
        data=np.zeros((10, 2)),
        unit="cmH2O",
        rate=512.0,
    )
    continuous, _, _ = build_sources_from_nwb(nwb)
    assert [src.id for src in _channels(continuous)] == [
        "0",
        "1",
        "SpO2",
        "resp/0",
        "resp/1",
    ]
    assert [src.unit for src in _channels(continuous)] == [
        "uV",
        "uV",
        "%",
        "cmH2O",
        "cmH2O",
    ]


def test_build_skips_a_series_that_is_not_numeric_samples(caplog):
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 2)), rate=512.0)
    nwb.add_acquisition(
        TimeSeries(name="notes", data=["a", "b"], unit="n/a", rate=1.0)
    )
    mock_TimeSeries(
        name="video",
        nwbfile=nwb,
        data=np.zeros((4, 2, 2)),
        unit="n/a",
        rate=1.0,
    )
    with caplog.at_level("WARNING"):
        continuous, _, _ = build_sources_from_nwb(nwb)
    assert [src.id for src in _channels(continuous)] == ["0", "1"]
    skipped = [record.getMessage() for record in caplog.records]
    assert any("notes" in message for message in skipped)
    assert any("video" in message for message in skipped)


def test_build_rejects_two_sources_with_one_id():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 2)), rate=512.0)
    mock_TimeSeries(
        name="0", nwbfile=nwb, data=np.zeros(10), unit="%", rate=512.0
    )
    with pytest.raises(ValueError, match="duplicate channel id '0'"):
        build_sources_from_nwb(nwb)


def test_build_reports_a_timestamp_off_the_sample_grid_of_its_run():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 2)), rate=512.0)
    stamps = np.arange(10) / 512.0
    stamps[6] += 5e-6
    mock_TimeSeries(
        name="misaligned",
        nwbfile=nwb,
        data=np.zeros(10),
        unit="%",
        timestamps=stamps,
    )
    with pytest.raises(
        ValueError, match="of misaligned is .* off the .* of its run"
    ):
        build_sources_from_nwb(nwb)


def _file(**kwargs):
    return NWBFile(
        session_description="EMU admission, day 3",
        identifier="file-identifier-123",
        session_start_time=STARTED,
        **kwargs,
    )


def test_recording_meta_session_start_time_is_the_files_session_start():
    assert recording_meta(_file()).session_start_time == STARTED


def test_recording_meta_reference_time_is_the_files_timestamps_reference():
    reference = STARTED + timedelta(seconds=10)
    meta = recording_meta(_file(timestamps_reference_time=reference))
    assert meta.reference_time == reference


def test_recording_meta_carries_every_subject_field_that_is_set():
    subject = Subject(
        subject_id="sub-0931",
        species="Homo sapiens",
        sex="F",
        age="P34Y",
        description="adult",
        date_of_birth=datetime(1990, 1, 2, tzinfo=UTC),
    )
    meta = recording_meta(_file(subject=subject))
    assert meta.subject == {
        "id": "sub-0931",
        "species": "Homo sapiens",
        "sex": "F",
        "age": "P34Y",
        "description": "adult",
        "date_of_birth": "1990-01-02T00:00:00+00:00",
    }


def test_recording_meta_omits_subject_fields_that_are_unset():
    meta = recording_meta(_file(subject=Subject(subject_id="sub-1")))
    assert meta.subject == {"id": "sub-1"}


def test_recording_meta_without_a_subject_has_an_empty_subject():
    assert recording_meta(_file()).subject == {}


def test_recording_meta_session_has_the_id_and_description_when_set():
    meta = recording_meta(_file(session_id="ses-01"))
    assert meta.session == {
        "id": "ses-01",
        "description": "EMU admission, day 3",
    }


def test_recording_meta_session_omits_an_unset_id():
    assert recording_meta(_file()).session == {
        "description": "EMU admission, day 3"
    }


def test_recording_meta_source_names_the_converter_writer_and_format():
    assert recording_meta(_file()).source == {
        "converter": f"timeseries-zarr-nwb {version('timeseries-zarr-nwb')}",
        "writer": f"timeseries-zarr {writer_version}",
        "format": "NWB",
    }


def test_recording_meta_format_carries_the_schema_version_of_a_read_file(
    tmp_path,
):
    path = tmp_path / "session.nwb"
    with NWBHDF5IO(str(path), mode="w") as io:
        io.write(_file())
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        expected = f"NWB {io.nwb_version[0]}"
        assert recording_meta(nwb).source["format"] == expected


def test_recording_meta_lists_the_software_that_generated_the_file():
    nwb = _file(
        was_generated_by=[["neuroconv", "0.6.1"], ["spikeinterface", "0.102"]]
    )
    assert recording_meta(nwb).source["generated_by"] == [
        "neuroconv 0.6.1",
        "spikeinterface 0.102",
    ]


def test_recording_meta_omits_generated_by_when_the_file_has_none():
    assert "generated_by" not in recording_meta(_file()).source


def test_recording_meta_lists_devices_in_name_order_with_their_model():
    nwb = _file()
    model = DeviceModel(name="Quantum", manufacturer="Natus", model_number="Q1")
    nwb.add_device_model(model)
    nwb.add_device(Device(name="scalp", model=model, serial_number="SN-9"))
    nwb.add_device(Device(name="probe", description="64-channel probe"))
    assert recording_meta(nwb).source["devices"] == [
        {"name": "probe"},
        {
            "name": "scalp",
            "manufacturer": "Natus",
            "model": "Quantum",
            "model_number": "Q1",
        },
    ]


def test_recording_meta_reads_the_model_fields_of_a_device_without_a_model():
    nwb = _file()
    nwb.add_device(
        Device(
            name="amp",
            manufacturer="Natus",
            model_name="Quantum",
            model_number="Q1",
            serial_number="SN-9",
        )
    )
    assert recording_meta(nwb).source["devices"] == [
        {
            "name": "amp",
            "manufacturer": "Natus",
            "model": "Quantum",
            "model_number": "Q1",
        }
    ]


def test_recording_meta_omits_devices_when_the_file_has_none():
    assert "devices" not in recording_meta(_file()).source


def test_recording_meta_adds_the_identifier_when_asked():
    meta = recording_meta(_file(), include_identifier=True)
    assert meta.source["identifier"] == "file-identifier-123"


def test_recording_meta_leaves_other_identifying_fields_out():
    meta = recording_meta(
        _file(
            session_id="ses-01",
            experimenter=["Doe, Jane"],
            institution="Example Hospital",
            source_script="# written by jdoe",
            source_script_file_name="/home/jdoe/convert.py",
            subject=Subject(subject_id="sub-1"),
            devices=[Device(name="amp", serial_number="SN-9")],
        )
    )
    text = str((meta.subject, meta.session, meta.source))
    for leaked in (
        "Doe",
        "Example Hospital",
        "jdoe",
        "SN-9",
        "file-identifier-123",
    ):
        assert leaked not in text


def _nwb_with_series(n_channels=3):
    nwb = mock_NWBFile()
    es = mock_ElectricalSeries(
        nwbfile=nwb, name="es", data=np.zeros((20, n_channels)), rate=512.0
    )
    return nwb, es


def _interval_table(name):
    table = TimeIntervals(name=name, description="d")
    table.add_row(start_time=1.0, stop_time=2.0)
    return table


def _events_table(name):
    table = EventsTable(name=name, description="d")
    table.add_row(timestamp=1.0)
    return table


@pytest.mark.filterwarnings("ignore:AnnotationSeries is deprecated:UserWarning")
def test_build_orders_event_sources_by_kind_then_name():
    nwb, es = _nwb_with_series()
    nwb.units = _make_units(_TWO_UNITS, name="units")
    nwb.add_epoch(start_time=1.0, stop_time=2.0, tags=["a"], timeseries=[es])
    nwb.add_trial(start_time=0.5, stop_time=0.6)
    nwb.add_invalid_time_interval(start_time=3.0, stop_time=4.0)
    for name in ("zeta", "alpha"):
        nwb.add_time_intervals(_interval_table(name))
    for name in ("marks", "events2"):
        nwb.add_events_table(_events_table(name))
    nwb.add_acquisition(
        AnnotationSeries(name="notes", data=["x"], timestamps=[1.0])
    )
    ids = [source.id for source in build_sources_from_nwb(nwb).events]
    assert ids == [
        "units",
        "epochs",
        "trials",
        "invalid_times",
        "alpha",
        "zeta",
        "events2",
        "marks",
        "notes",
    ]


def test_build_resolves_timeseries_references_to_the_series_channel_ids():
    nwb, es = _nwb_with_series()
    nwb.add_epoch(start_time=1.0, stop_time=2.0, tags=["a"], timeseries=[es])
    built = build_sources_from_nwb(nwb)
    ids = [source.id for source in _channels(built.continuous)]
    epochs = next(src for src in built.events if src.id == "epochs")
    assert epochs.read_channel_ids(0, 1) == [ids]
    assert "channels" in epochs.columns()


def test_build_resolves_a_reference_to_a_plain_series_by_its_name():
    nwb, _ = _nwb_with_series()
    pulse = TimeSeries(name="pulse", data=np.zeros(20), unit="bpm", rate=512.0)
    nwb.add_acquisition(pulse)
    nwb.add_epoch(start_time=1.0, stop_time=2.0, tags=["a"], timeseries=[pulse])
    epochs = next(
        src for src in build_sources_from_nwb(nwb).events if src.id == "epochs"
    )
    assert epochs.read_channel_ids(0, 1) == [["pulse"]]


def test_build_skips_tables_with_no_rows_without_needing_a_rate():
    nwb = mock_NWBFile()
    nwb.units = Units(name="units")
    nwb.add_time_intervals(TimeIntervals(name="empty", description="d"))
    nwb.add_events_table(EventsTable(name="none", description="d"))
    assert build_sources_from_nwb(nwb).events == []


@pytest.mark.filterwarnings("ignore:AnnotationSeries is deprecated:UserWarning")
def test_build_skips_an_empty_annotation_series():
    nwb = mock_NWBFile()
    nwb.add_acquisition(AnnotationSeries(name="notes", data=[], timestamps=[]))
    assert build_sources_from_nwb(nwb).events == []


@pytest.mark.filterwarnings("ignore:AnnotationSeries is deprecated:UserWarning")
def test_build_reads_an_annotation_series_as_events_not_as_a_skipped_series(
    caplog,
):
    nwb, _ = _nwb_with_series()
    nwb.add_acquisition(
        AnnotationSeries(name="notes", data=["x"], timestamps=[1.0])
    )
    with caplog.at_level("WARNING"):
        built = build_sources_from_nwb(nwb)
    assert [src.id for src in built.events] == ["notes"]
    assert "notes" not in caplog.text


def test_build_returns_the_recording_metadata_of_the_file():
    nwb, _ = _nwb_with_series()
    built = build_sources_from_nwb(nwb)
    assert built.recording.session_start_time == nwb.session_start_time
    assert built.recording.source["format"] == "NWB"


_STARTED_US = round(STARTED.timestamp() * 1_000_000)
"""STARTED in microseconds since the Unix epoch."""


def _nwb_with_an_earlier_reference_time():
    nwb = NWBFile(
        session_description="d",
        identifier="id",
        session_start_time=STARTED,
        timestamps_reference_time=STARTED - timedelta(seconds=30),
    )
    mock_ElectricalSeries(
        nwbfile=nwb,
        name="es",
        data=np.zeros((20, 1)),
        rate=512.0,
        starting_time=0.0,
    )
    nwb.add_epoch(start_time=1.0, stop_time=2.0)
    return nwb


def test_build_places_series_from_the_timestamps_reference_time():
    built = build_sources_from_nwb(_nwb_with_an_earlier_reference_time())
    channel = _channels(built.continuous)[0]
    assert channel.start_us() == _STARTED_US - 30_000_000


def test_build_places_events_from_the_timestamps_reference_time():
    built = build_sources_from_nwb(_nwb_with_an_earlier_reference_time())
    epochs = next(src for src in built.events if src.id == "epochs")
    assert epochs.read_events(0, 1).tolist() == [_STARTED_US - 29_000_000]


def test_build_keeps_the_session_start_time_when_the_reference_time_differs():
    built = build_sources_from_nwb(_nwb_with_an_earlier_reference_time())
    assert built.recording.session_start_time == STARTED


def test_build_passes_include_identifier_to_the_recording_metadata():
    nwb, _ = _nwb_with_series()
    assert "identifier" not in build_sources_from_nwb(nwb).recording.source
    built = build_sources_from_nwb(nwb, include_identifier=True)
    assert built.recording.source["identifier"] == nwb.identifier


def test_build_without_events_has_an_empty_event_list():
    nwb, _ = _nwb_with_series()
    assert build_sources_from_nwb(nwb).events == []


def test_build_rejects_an_event_table_that_shares_an_id_with_a_series():
    nwb, es = _nwb_with_series()
    nwb.add_acquisition(
        TimeSeries(name="epochs", data=np.zeros(5), unit="n/a", rate=1.0)
    )
    nwb.add_epoch(start_time=1.0, stop_time=2.0, tags=["a"], timeseries=[es])
    with pytest.raises(ValueError, match="duplicate channel id 'epochs'"):
        build_sources_from_nwb(nwb)


def test_build_unit_tables_still_need_an_electrical_series_rate():
    nwb = mock_NWBFile()
    nwb.units = _make_units(_TWO_UNITS, name="units")
    with pytest.raises(ValueError, match="need an ElectricalSeries rate"):
        build_sources_from_nwb(nwb)


def test_continuous_sources_report_the_name_of_their_series():
    nwb, es = _nwb_with_series()
    plain = TimeSeries(name="pulse", data=np.zeros(20), unit="bpm", rate=1.0)
    assert NwbContinuousSource(es, 1, STARTED).series_names == ["es"]
    assert NwbTimeSeriesSource(plain, 0, STARTED).series_names == ["pulse"]


def test_electrical_series_provenance_names_the_series_filtering_and_device(
    electrical_series,
):
    es = electrical_series(filtering="300-6000 Hz bandpass")
    src = NwbContinuousSource(
        es, 1, STARTED, nwb_path="/acquisition/ElectricalSeries"
    )
    device = es.electrodes.table["group"][es.electrodes.data[1]].device
    assert src.provenance() == {
        "neurodata_type": "ElectricalSeries",
        "nwb_path": "/acquisition/ElectricalSeries",
        "filtering": "300-6000 Hz bandpass",
        "device": device.name,
    }


def test_electrical_series_provenance_omits_unset_filtering(electrical_series):
    src = NwbContinuousSource(electrical_series(filtering=None), 0, STARTED)
    assert "filtering" not in src.provenance()


def test_time_series_provenance_names_the_type_and_the_path():
    pulse = TimeSeries(name="pulse", data=np.zeros(4), unit="bpm", rate=1.0)
    src = NwbTimeSeriesSource(pulse, 0, STARTED, nwb_path="/acquisition/pulse")
    assert src.provenance() == {
        "neurodata_type": "TimeSeries",
        "nwb_path": "/acquisition/pulse",
    }


def test_unit_provenance_names_the_type_and_the_path():
    src = NwbUnitSource(
        _make_units(_TWO_UNITS), 30000.0, STARTED, nwb_path="/units"
    )
    assert src.provenance() == {
        "neurodata_type": "Units",
        "nwb_path": "/units",
    }


@pytest.mark.filterwarnings("ignore:AnnotationSeries is deprecated:UserWarning")
def test_build_gives_every_source_the_path_of_its_object_in_the_file():
    nwb, es = _nwb_with_series(n_channels=2)
    nwb.add_acquisition(
        TimeSeries(name="pulse", data=np.zeros(20), unit="bpm", rate=512.0)
    )
    nwb.units = _make_units(_TWO_UNITS, name="units")
    module = ProcessingModule(name="ecephys", description="d")
    nwb.add_processing_module(module)
    module.add(_make_units(_TWO_UNITS, name="sorted"))
    nwb.add_epoch(start_time=1.0, stop_time=2.0, timeseries=[es])
    nwb.add_time_intervals(_interval_table("alpha"))
    nwb.add_events_table(_events_table("marks"))
    nwb.add_acquisition(
        AnnotationSeries(name="notes", data=["x"], timestamps=[1.0])
    )
    built = build_sources_from_nwb(nwb)
    paths = [
        source.provenance()["nwb_path"]
        for source in [*_channels(built.continuous), *built.events]
    ]
    assert paths == [
        "/acquisition/es",
        "/acquisition/es",
        "/acquisition/pulse",
        "/units",
        "/processing/ecephys/sorted",
        "/intervals/epochs",
        "/intervals/alpha",
        "/events/marks",
        "/acquisition/notes",
    ]


_SAMPLES = np.random.default_rng(0).integers(-3000, 3000, (1000, 5))
"""Stored int16 values of a 1000-row, 5-column series."""


def _window_blocks(source):
    """Return each channel's column of the windows of a series source, joined."""
    windows = list(source.iter_windows())
    return [
        np.concatenate([window[:, index] for window in windows])
        for index in range(len(source.channels()))
    ]


def _assert_windows_match_read_samples(source):
    channels = source.channels()
    blocks = _window_blocks(source)
    assert len(blocks) == len(channels)
    for channel, block in zip(channels, blocks, strict=True):
        expected = channel.read_samples(0, channel.num_samples())
        assert block.dtype == np.float32
        assert np.array_equal(block, expected, equal_nan=True)


@pytest.fixture
def small_windows(monkeypatch):
    """Shrink the window budget so a 1000-row series spans several windows."""
    monkeypatch.setattr(nwb_reader, "WINDOW_BYTES", 8 * 5 * 96)


def test_series_source_serves_the_multi_channel_protocol():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 3)), rate=512.0)
    (source,) = build_sources_from_nwb(nwb).continuous
    assert isinstance(source, NwbSeriesSource)
    assert isinstance(source, MultiChannelSource)


def test_series_source_channels_are_the_column_sources_in_column_order():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 3)), rate=512.0)
    (source,) = build_sources_from_nwb(nwb).continuous
    channels = source.channels()
    assert all(isinstance(c, NwbContinuousSource) for c in channels)
    assert [c._channel_index for c in channels] == [0, 1, 2]


def test_series_source_windows_match_electrical_series_read_samples(
    small_windows,
):
    nwb = mock_NWBFile()
    es = mock_ElectricalSeries(
        nwbfile=nwb,
        data=_SAMPLES.astype(np.int16),
        rate=512.0,
        conversion=0.195,
        offset=-1.5,
        channel_conversion=np.array([1.0, 0.5, 2.0, 0.25, 3.0]),
    )
    es.fields["unit"] = "millivolts"
    (source,) = build_sources_from_nwb(nwb).continuous
    assert len(list(source.iter_windows())) > 1
    _assert_windows_match_read_samples(source)


@pytest.mark.parametrize("unit", ["volts", "cmH2O"])
@pytest.mark.parametrize("shape", [(1000,), (1000, 5)])
def test_series_source_windows_match_time_series_read_samples(
    small_windows, unit, shape
):
    nwb = mock_NWBFile()
    data = _SAMPLES[:, 0] if len(shape) == 1 else _SAMPLES
    mock_TimeSeries(
        name="aux",
        nwbfile=nwb,
        data=data.astype(np.int16),
        unit=unit,
        rate=512.0,
        conversion=0.5,
        offset=1.25,
    )
    (source,) = build_sources_from_nwb(nwb).continuous
    assert len(list(source.iter_windows())) > 1
    _assert_windows_match_read_samples(source)


def test_build_groups_consecutive_columns_by_max_sweep_channels():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 5)), rate=512.0)
    grouped = build_sources_from_nwb(nwb, max_sweep_channels=2).continuous
    whole = build_sources_from_nwb(nwb).continuous
    assert [source._columns for source in grouped] == [
        range(0, 2),
        range(2, 4),
        range(4, 5),
    ]
    assert [c.id for c in _channels(grouped)] == [
        c.id for c in _channels(whole)
    ]


def test_build_rejects_a_max_sweep_channels_below_one():
    with pytest.raises(ValueError, match="max_sweep_channels must be 1"):
        build_sources_from_nwb(mock_NWBFile(), max_sweep_channels=0)


def test_build_skips_a_series_without_columns():
    nwb = mock_NWBFile()
    mock_TimeSeries(
        name="none", nwbfile=nwb, data=np.zeros((10, 0)), unit="%", rate=1.0
    )
    assert build_sources_from_nwb(nwb).continuous == []


def _write_chunked_nwb(path):
    nwb = mock_NWBFile()
    mock_ElectricalSeries(
        name="es",
        nwbfile=nwb,
        data=H5DataIO(_SAMPLES.astype(np.int16), chunks=(64, 5)),
        rate=512.0,
        conversion=0.195,
        channel_conversion=np.array([1.0, 0.5, 2.0, 0.25, 3.0]),
    )
    mock_TimeSeries(
        name="resp",
        nwbfile=nwb,
        data=_SAMPLES[:, :2].astype(np.float32),
        unit="cmH2O",
        rate=512.0,
    )
    nwb.add_epoch(start_time=0.5, stop_time=1.0)
    with NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwb)
    return path


def test_series_source_windows_are_whole_chunks_of_a_chunked_dataset(
    tmp_path, small_windows
):
    path = _write_chunked_nwb(tmp_path / "chunked.nwb")
    with NWBHDF5IO(str(path), mode="r") as io:
        es_source = build_sources_from_nwb(io.read()).continuous[0]
        assert es_source._layout.series[0].data.chunks == (64, 5)
        lengths = [len(window) for window in es_source.iter_windows()]
        _assert_windows_match_read_samples(es_source)
    assert lengths[:-1] == [64] * (len(lengths) - 1)
    assert sum(lengths) == 1000


def test_series_source_window_keeps_the_budget_below_one_chunk(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(nwb_reader, "WINDOW_BYTES", 8 * 5 * 40)
    path = _write_chunked_nwb(tmp_path / "chunked.nwb")
    with NWBHDF5IO(str(path), mode="r") as io:
        es_source = build_sources_from_nwb(io.read()).continuous[0]
        lengths = {len(window) for window in es_source.iter_windows()}
        _assert_windows_match_read_samples(es_source)
    assert lengths == {40}


def test_series_source_windows_match_a_rank_one_electrical_series(
    small_windows,
):
    data = _SAMPLES[:, 0].astype(np.int16)
    nwb, _ = _one_channel_series(
        data, conversion=0.195, channel_conversion=[2.5]
    )
    (source,) = build_sources_from_nwb(nwb).continuous
    assert len(list(source.iter_windows())) > 1
    _assert_windows_match_read_samples(source)


@pytest.mark.parametrize("columns", [1, 3])
def test_series_source_yields_each_window_as_rows_by_channels(columns):
    nwb = mock_NWBFile()
    mock_ElectricalSeries(
        nwbfile=nwb, data=_SAMPLES[:, :columns].astype(np.int16), rate=512.0
    )
    (source,) = build_sources_from_nwb(nwb).continuous
    (window,) = source.iter_windows()
    assert window.shape == (1000, columns)
    assert window.dtype == np.float32
    assert window.flags.c_contiguous


def test_series_source_without_samples_yields_no_windows(tmp_path):
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((0, 2)), rate=512.0)
    sources = build_sources_from_nwb(nwb)
    (source,) = sources.continuous
    assert list(source.iter_windows()) == []
    out = tmp_path / "empty.zarr"
    write_bundle(
        sources.continuous,
        [],
        output_path=out,
        opts=WriteOpts(),
        recording=sources.recording,
    )
    assert (out / "0" / "raw" / "zarr.json").exists()
    assert (out / "1" / "raw" / "zarr.json").exists()


def _bundle_files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_series_sources_write_the_same_bundle_as_their_channels(
    tmp_path, small_windows
):
    path = _write_chunked_nwb(tmp_path / "chunked.nwb")
    opts = WriteOpts(inner_len=64, min_bins=4)
    with NWBHDF5IO(str(path), mode="r") as io:
        sources = build_sources_from_nwb(io.read(), max_sweep_channels=2)
        assert len(sources.continuous) == 4
        for name, continuous in (
            ("multi", sources.continuous),
            ("single", _channels(sources.continuous)),
        ):
            write_bundle(
                continuous,
                sources.events,
                output_path=tmp_path / name,
                opts=opts,
                recording=sources.recording,
            )
    multi = _bundle_files(tmp_path / "multi")
    assert (tmp_path / "multi" / "6" / "zarr.json").exists()
    assert multi == _bundle_files(tmp_path / "single")


def _nwb_with_electrodes(count):
    """Return a file whose electrode table holds count electrodes of one group."""
    nwb = mock_NWBFile()
    device = mock_Device(nwbfile=nwb)
    group = nwb.create_electrode_group(
        name="grp", description="d", location="loc", device=device
    )
    for _ in range(count):
        nwb.add_electrode(group=group, location="loc")
    return nwb


def _add_segment(nwb, name, electrodes, **fields):
    """Add an ElectricalSeries over the given electrode rows to the acquisition."""
    region = DynamicTableRegion(
        name="electrodes",
        data=list(electrodes),
        description="r",
        table=nwb.electrodes,
    )
    series = ElectricalSeries(name=name, electrodes=region, **fields)
    nwb.add_acquisition(series)
    return series


def test_build_stitches_series_over_the_same_electrodes_into_one_channel_each():
    nwb = _nwb_with_electrodes(2)
    first = np.arange(8, dtype=np.float64).reshape(4, 2)
    second = 100 + np.arange(6, dtype=np.float64).reshape(3, 2)
    _add_segment(
        nwb, "first", [0, 1], data=first, rate=512.0, starting_time=1.0
    )
    _add_segment(
        nwb,
        "second",
        [0, 1],
        data=second,
        rate=512.0,
        starting_time=1.0 + 10 / 512.0,
    )
    (source,) = build_sources_from_nwb(nwb).continuous
    channels = source.channels()
    assert [channel.id for channel in channels] == ["0", "1"]
    reference_us = round(nwb.timestamps_reference_time.timestamp() * 1_000_000)
    assert channels[1].start_us() == reference_us + 1_000_000
    assert channels[1].num_samples() == 7
    expected = np.concatenate([first[:, 1], second[:, 1]]) * 1e6
    assert np.array_equal(
        channels[1].read_samples(0, 7), expected.astype(np.float32)
    )
    assert channels[1].runs() == [
        Run(reference_us + 1_000_000, 4),
        Run(reference_us + 1_000_000 + round(10 / 512.0 * 1_000_000), 3),
    ]


def test_build_places_a_recording_at_its_first_series_with_segments_in_time_order():
    nwb = _nwb_with_electrodes(4)
    data = np.zeros((4, 2))
    _add_segment(nwb, "late", [0, 1], data=data, rate=512.0, starting_time=1.0)
    _add_segment(nwb, "other", [2, 3], data=data, rate=512.0, starting_time=0.0)
    _add_segment(nwb, "early", [0, 1], data=data, rate=512.0, starting_time=0.0)
    channels = _channels(build_sources_from_nwb(nwb).continuous)
    assert [channel.id for channel in channels] == ["0", "1", "2", "3"]
    assert channels[0].series_names == ["early", "late"]
    provenance = channels[0].provenance()
    assert provenance["nwb_path"] == "/acquisition/early"
    assert provenance["segments"] == ["early", "late"]
    assert "segments" not in channels[2].provenance()


def test_stitched_windows_match_read_samples_across_the_pause(small_windows):
    nwb = _nwb_with_electrodes(5)
    _add_segment(
        nwb,
        "first",
        range(5),
        data=_SAMPLES[:600].astype(np.int16),
        rate=512.0,
        starting_time=0.0,
        conversion=0.195,
    )
    _add_segment(
        nwb,
        "second",
        range(5),
        data=_SAMPLES[600:].astype(np.int16),
        rate=512.0,
        starting_time=1100 / 512.0,
        conversion=0.5,
    )
    (source,) = build_sources_from_nwb(nwb).continuous
    channel = source.channels()[0]
    assert channel.num_samples() == 1000
    assert not np.isnan(channel.read_samples(0, 1000)).any()
    assert channel.read_samples(600, 601)[0] == np.float32(
        _SAMPLES[600, 0] * 0.5 * 1e6
    )
    assert len(list(source.iter_windows())) > 1
    _assert_windows_match_read_samples(source)


def test_build_resolves_a_reference_to_a_segment_to_the_stitched_ids():
    nwb = _nwb_with_electrodes(2)
    data = np.zeros((4, 2))
    _add_segment(nwb, "first", [0, 1], data=data, rate=512.0, starting_time=0.0)
    second = _add_segment(
        nwb, "second", [0, 1], data=data, rate=512.0, starting_time=1.0
    )
    nwb.add_epoch(start_time=1.0, stop_time=1.001, timeseries=[second])
    epochs = next(
        src for src in build_sources_from_nwb(nwb).events if src.id == "epochs"
    )
    assert epochs.read_channel_ids(0, 1) == [["0", "1"]]


def test_build_rejects_series_that_share_some_electrodes_but_not_all():
    nwb = _nwb_with_electrodes(3)
    _add_segment(nwb, "a", [0, 1], data=np.zeros((4, 2)), rate=512.0)
    _add_segment(nwb, "b", [1, 2], data=np.zeros((4, 2)), rate=512.0)
    with pytest.raises(ValueError, match="b and a share electrode id '1'"):
        build_sources_from_nwb(nwb)


def test_build_rejects_segments_that_overlap():
    nwb = _nwb_with_electrodes(2)
    data = np.zeros((4, 2))
    _add_segment(nwb, "a", [0, 1], data=data, rate=512.0, starting_time=0.0)
    _add_segment(nwb, "b", [0, 1], data=data, rate=512.0, starting_time=2 / 512)
    with pytest.raises(ValueError, match="b starts before a ends"):
        build_sources_from_nwb(nwb)


def test_build_keeps_a_segment_that_resumes_off_the_grid_of_the_earliest():
    nwb = _nwb_with_electrodes(2)
    data = np.zeros((4, 2))
    _add_segment(nwb, "a", [0, 1], data=data, rate=512.0, starting_time=0.0)
    _add_segment(
        nwb, "b", [0, 1], data=data, rate=512.0, starting_time=10.4 / 512
    )
    (source,) = build_sources_from_nwb(nwb).continuous
    channel = source.channels()[0]
    reference_us = round(nwb.timestamps_reference_time.timestamp() * 1_000_000)
    assert channel.runs() == [
        Run(reference_us, 4),
        Run(reference_us + round(10.4 / 512 * 1_000_000), 4),
    ]


def test_build_joins_abutting_segments_into_one_run():
    nwb = _nwb_with_electrodes(1)
    data = np.zeros((7, 1))
    _add_segment(nwb, "a", [0], data=data, rate=30000.0, starting_time=0.0)
    _add_segment(
        nwb, "b", [0], data=data, rate=30000.0, starting_time=7 / 30000.0
    )
    (source,) = build_sources_from_nwb(nwb).continuous
    reference_us = round(nwb.timestamps_reference_time.timestamp() * 1_000_000)
    assert source.channels()[0].runs() == [Run(reference_us, 14)]


def test_build_writes_no_runs_array_for_abutting_segments(tmp_path):
    nwb = _nwb_with_electrodes(1)
    data = np.arange(3000, dtype=np.float64).reshape(-1, 1)
    _add_segment(nwb, "a", [0], data=data, rate=512.0, starting_time=0.0)
    _add_segment(
        nwb, "b", [0], data=data[:2000], rate=512.0, starting_time=3000 / 512
    )
    (source,) = build_sources_from_nwb(nwb).continuous
    out = tmp_path / "out.zarr"
    write_bundle([source], [], output_path=out, opts=WriteOpts())
    assert not (out / "0" / "runs").exists()
    assert zarr.open_array(out / "0" / "raw", mode="r").shape == (5000,)


def test_build_rejects_a_segment_whose_rate_drifts_from_the_earliest():
    nwb = _nwb_with_electrodes(2)
    _add_segment(
        nwb, "a", [0, 1], data=np.zeros((4, 2)), rate=512.0, starting_time=0.0
    )
    _add_segment(
        nwb,
        "b",
        [0, 1],
        data=np.zeros((4000, 2)),
        rate=512.01,
        starting_time=1.0,
    )
    with pytest.raises(ValueError, match="b drifts .* us from the 512 Hz"):
        build_sources_from_nwb(nwb)


def test_build_rejects_a_segment_with_a_unit_outside_the_volts_family():
    nwb = _nwb_with_electrodes(2)
    data = np.zeros((4, 2))
    _add_segment(nwb, "a", [0, 1], data=data, rate=512.0, starting_time=0.0)
    late = _add_segment(
        nwb, "b", [0, 1], data=data, rate=512.0, starting_time=1.0
    )
    late.fields["unit"] = "furlongs"
    with pytest.raises(ValueError, match="furlongs"):
        build_sources_from_nwb(nwb)


def test_build_gives_units_the_rate_of_a_series_sampled_by_timestamps():
    nwb = _nwb_with_electrodes(2)
    _add_segment(
        nwb,
        "stamped",
        [0, 1],
        data=np.zeros((10, 2)),
        timestamps=np.arange(10) / 250.0,
    )
    nwb.units = _make_units(_TWO_UNITS, name="units")
    (units,) = build_sources_from_nwb(nwb).events
    assert units.waveform_period_us() == pytest.approx(4000.0)
