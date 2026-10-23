# timeseries-zarr-nwb

Converts an NWB neurophysiology recording into a TS-Zarr bundle,
a static Zarr v3 hierarchy that a browser reads over HTTP range requests to
render time series interactively across various time spans.

This package is an NWB input adapter for
[timeseries-zarr](https://github.com/Pennsieve/timeseries-zarr-py), which specifies
the format and writes the bundle.

It opens an NWB file with `pynwb`, presents its series and tables as channel sources, and runs the writer.

## Installation

```bash
pip install git+https://github.com/Pennsieve/timeseries-zarr-nwb.git
```

## Usage

```bash
timeseries-zarr-nwb recording.nwb /data/recording.zarr
```

`python -m timeseries_zarr_nwb` runs the same command. The exit code is 0 on success,
2 for a bad invocation, and 1 for a failed conversion.

The output path must be absent, empty, or hold a bundle from an earlier run, which the
converter replaces.

The converter reads every `ElectricalSeries` and every other numeric `TimeSeries` in
the acquisition as continuous channels, one per column. It reads `Units` tables,
interval tables (epochs, trials, invalid times, and any other `TimeIntervals`),
`EventsTable` tables, and `AnnotationSeries` as event channels. Voltage samples are
stored in microvolts. A series in any other unit keeps its unit.

The subject, the session, the session start time, and the reference time go into
`recording.json` at the bundle root and into no other metadata. The reference time is
the NWB `timestamps_reference_time`, and every time in the bundle counts from it. The
`source` object of `recording.json` names the converter, the writer, and the NWB schema
version. It also lists the software in the file's `was_generated_by` field and the name,
manufacturer, and model of each device. Delete that file to de-identify a bundle. Free
text in event channels can still hold identity. The NWB file identifier stays out unless
`ZARR_WRITER_INCLUDE_IDENTIFIER` is 1.

Each channel's `provenance` holds the neurodata type and the path of the NWB object it
came from, such as `/acquisition/ElectricalSeries`. An `ElectricalSeries` channel also
carries the series' `filtering` and the name of its electrode's device when the file
records them.

Writer settings come from the environment under the `ZARR_WRITER_` prefix. Every
setting is optional.

| Variable | Governs | Range | Default |
|---|---|---|---|
| `ZARR_WRITER_ZSTD_LEVEL` | Zstd compression level | 0 to 22 | 5 |
| `ZARR_WRITER_MAX_LEVELS` | most levels above the raw samples; 0 writes raw samples only | 0 to 15 | 15 |
| `ZARR_WRITER_MIN_BINS` | bin threshold for keeping a coarser level | 1 or more | 1024 |
| `ZARR_WRITER_INNER_LEN` | inner Zarr chunk length in samples | 1 or more | 8192 |
| `ZARR_WRITER_TARGET_SHARD_BYTES` | target outer shard size in bytes | 1 or more | 16 MiB |
| `ZARR_WRITER_VAR` | write the per-bin variance member | 0 or 1 | 0 |
| `ZARR_WRITER_INCLUDE_IDENTIFIER` | add the NWB file identifier to `recording.json` | 0 or 1 | 0 |

`ZARR_WRITER_VAR` sets the library's `write_var` option.
[The library README](https://github.com/Pennsieve/timeseries-zarr-py#usage) describes what
it writes and what it costs in memory.

## Development

Python 3.12, fully typed under `mypy --strict`, with a strict `ruff` ruleset. Tests in
`tests/` mirror `timeseries_zarr_nwb/` one to one.

```bash
make venv        # create the virtualenv and install this package with the library from PyPI
source venv/bin/activate

make test        # pytest
make typecheck   # mypy --strict
make lint        # ruff check --fix + ruff format (rewrites files)
make check       # ruff check + format check + mypy + pytest
make pre-commit  # install the git pre-commit hook
```

`make check` must stay green.

## Dependencies

Runtime: `timeseries-zarr`, `numpy`, `pynwb>=4`, `h5py`. Development: `pytest`,
`pytest-cov`, `pytest-mock`, `mypy`, `ruff`, `pre-commit`.

## License

Apache-2.0. See [LICENSE](LICENSE).
