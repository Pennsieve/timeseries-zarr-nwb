from datetime import UTC, datetime

import numpy as np
import pytest
from hdmf.common import DynamicTableRegion
from pynwb import NWBFile, TimeSeries
from pynwb.ecephys import ElectricalSeries
from pynwb.epoch import TimeIntervals
from pynwb.event import EventsTable
from pynwb.file import Subject
from pynwb.misc import AnnotationSeries, Units
from pynwb.testing.mock.base import mock_TimeSeries
from pynwb.testing.mock.device import mock_Device
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from pynwb.testing.mock.file import mock_NWBFile
from timeseries_zarr import EventChannelSource

from timeseries_zarr_nwb.nwb_reader import (
    NwbContinuousSource,
    NwbTimeSeriesSource,
    NwbUnitSource,
    build_sources_from_nwb,
    recording_meta,
)

STARTED = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)


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
    assert src._session_start_time == started


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


def test_start_us_is_session_start_plus_offset_in_microseconds(
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


def test_read_samples_raises_on_unknown_unit():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    es = mock_ElectricalSeries(data=data)
    es.fields["unit"] = "furlongs"
    src = NwbContinuousSource(es, 2, STARTED)
    with pytest.raises(ValueError, match="furlongs"):
        src.read_samples(0, 4)


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


def test_unit_start_us_is_session_start_in_microseconds():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.start_us() == round(STARTED.timestamp() * 1_000_000)


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


def test_unit_start_is_the_first_spike_when_it_precedes_the_session():
    specs = [([-2.0, 0.5], _WM0)]
    src = NwbUnitSource(_make_units(specs), 30000.0, STARTED)
    assert src.start_us() == round((STARTED.timestamp() - 2.0) * 1_000_000)


def test_unit_table_without_units_has_no_events_and_no_columns():
    src = NwbUnitSource(Units(name="none"), 30000.0, STARTED)
    assert src.num_events() == 0
    assert src.num_labels() == 0
    assert src.columns() == frozenset()
    assert src.start_us() == round(STARTED.timestamp() * 1_000_000)


def test_unit_table_whose_units_have_no_spikes_has_labels_and_no_events():
    specs = [([], _WM0), ([], _WM1)]
    src = NwbUnitSource(_make_units(specs), 30000.0, STARTED)
    assert src.num_events() == 0
    assert src.columns() == {"labels", "waveform_mean"}
    assert src.read_labels(0, 0).shape == (0,)


def test_build_yields_one_continuous_source_per_channel():
    nwb = mock_NWBFile()
    data = np.arange(50, dtype=np.float64).reshape(10, 5)
    mock_ElectricalSeries(nwbfile=nwb, data=data, rate=30000.0)
    continuous, units, _ = build_sources_from_nwb(nwb)
    assert len(continuous) == 5
    assert units == []
    assert np.array_equal(
        continuous[2].read_samples(0, 10),
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


def _one_channel_series(data, *, rate=30000.0):
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
    assert len(continuous) == 1


def test_continuous_rejects_a_series_sampled_by_timestamps():
    es = mock_ElectricalSeries(
        data=np.zeros((10, 2)), timestamps=np.arange(10) / 512.0
    )
    with pytest.raises(ValueError, match="irregular sampling"):
        NwbContinuousSource(es, 0, STARTED)


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


def test_timeseries_source_names_the_columns_of_a_rank_two_series():
    data = np.arange(20, dtype=np.float64).reshape(10, 2)
    ts = mock_TimeSeries(name="resp", data=data, unit="cmH2O", rate=512.0)
    first = NwbTimeSeriesSource(ts, 0, STARTED)
    second = NwbTimeSeriesSource(ts, 1, STARTED)
    assert (first.id, second.id) == ("resp/0", "resp/1")
    assert np.array_equal(
        second.read_samples(0, 3), data[0:3, 1].astype(np.float32)
    )


def test_timeseries_source_rejects_a_series_sampled_by_timestamps():
    ts = mock_TimeSeries(
        name="irregular",
        data=np.zeros(10),
        unit="%",
        timestamps=np.arange(10) / 512.0,
    )
    with pytest.raises(ValueError, match="irregular sampling"):
        NwbTimeSeriesSource(ts, 0, STARTED)


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
    assert [src.id for src in continuous] == [
        "0",
        "1",
        "SpO2",
        "resp/0",
        "resp/1",
    ]
    assert [src.unit for src in continuous] == [
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
    assert [src.id for src in continuous] == ["0", "1"]
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


def test_build_reports_a_series_sampled_by_timestamps():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, data=np.zeros((10, 2)), rate=512.0)
    mock_TimeSeries(
        name="irregular",
        nwbfile=nwb,
        data=np.zeros(10),
        unit="%",
        timestamps=np.arange(10) / 512.0,
    )
    with pytest.raises(ValueError, match="irregular sampling.*irregular"):
        build_sources_from_nwb(nwb)


def _file(**kwargs):
    return NWBFile(
        session_description="EMU admission, day 3",
        identifier="file-identifier-123",
        session_start_time=STARTED,
        **kwargs,
    )


def test_recording_meta_start_is_the_session_start_time():
    assert recording_meta(_file()).start == STARTED


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


def test_recording_meta_source_names_the_converter_and_the_format():
    assert recording_meta(_file()).source == {
        "converter": "timeseries-zarr-nwb",
        "format": "NWB",
    }


def test_recording_meta_leaves_other_identifying_fields_out():
    meta = recording_meta(
        _file(
            session_id="ses-01",
            experimenter=["Doe, Jane"],
            institution="Example Hospital",
            subject=Subject(subject_id="sub-1"),
        )
    )
    text = str((meta.subject, meta.session, meta.source))
    for leaked in ("Doe", "Example Hospital", "file-identifier-123"):
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
    ids = [source.id for source in built.continuous]
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
    assert built.recording.start == nwb.session_start_time
    assert built.recording.source["format"] == "NWB"


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
    assert NwbContinuousSource(es, 1, STARTED).series_name == "es"
    assert NwbTimeSeriesSource(plain, 0, STARTED).series_name == "pulse"
