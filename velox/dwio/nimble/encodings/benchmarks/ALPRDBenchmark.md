# ALP_RD benchmark

This benchmark measures explicit ALP_RD encoding on its target distributions:
shared high bits, multiple prefixes, and dictionary misses. It also includes
broad-exponent and random-bit controls. Automatic encoding selection is outside
its scope; ALP_RD is not expected to replace every other Nimble encoding.

## Build and run

From the Velox repository root, using a Release build:

```sh
cmake --build _build/release --target nimble_alprd_benchmark -j6

taskset -c 7 _build/release/velox/dwio/nimble/encodings/benchmarks/nimble_alprd_benchmark \
  --profile --rows=65536 --profile_trials=7 --profile_min_ms=50 > alprd.csv
```

Choose an available CPU instead of CPU 7. Repeat with `--rows=1024` or
`--rows=1048576` for other payload sizes. Optional filters:
`--profile_type=float`, `--profile_type=double`, and
`--profile_dataset=exceptions_1pct`. Without `--profile`, the executable runs
its eight Folly encode/construct-and-decode cases using `--rows` values.

## Synthetic workloads

Inputs use `std::mt19937_64` with seed `0xA1F0`. FLOAT and DOUBLE use their own
representation. Random low bits are retained without decimal rounding.

| Dataset | Distribution |
| --- | --- |
| `shared_prefix` | One common 16-bit high prefix and random low bits. |
| `eight_prefixes` | Eight neighboring high prefixes and random low bits. Training may choose a different split. |
| `mixed_sign` | A common magnitude prefix with random signs and low bits. |
| `exceptions_1pct` | A common prefix plus about 1% values from 256 uncommon high prefixes. |
| `exceptions_10pct` | The same construction with about 10% uncommon prefixes. |
| `duckdb_best_case` | `10 + uniform[0, 1)`, following DuckDB's ALP_RD best-case workload. |
| `wide_exponents` | Random signs and mantissas with binary exponents from -100 through 100. |
| `random_bits` | Random IEEE patterns plus signed zeros, infinities, NaNs, subnormals and maximum finite values. |

Random outlier positions avoid aliasing evenly spaced training samples. The
reported exception count measures misses in the trained dictionary and need
not equal the generator's outlier count.

## External data and paper samples

`--input_file` reads raw little-endian IEEE values. Specify one
`--profile_type` and omit `--profile_dataset`. Up to `--rows` values from the
beginning of the file form one encoding payload. The actual row count is
reported; short files are not repeated, and empty/partial-value files are
rejected. Input loading and conversion are outside timing.

The ALP artifact marks `POI-lat` and `POI-lon` as `suitable_for_cutting` in its
[dataset descriptors](https://github.com/cwida/ALP/blob/31ca0ed11c93c99d3f5b5c30e01a3e1c3832d3ce/data/include/double/alp_dataset.hpp).
Its checked-in CSV samples contain 1024 DOUBLE values each. For example:

```sh
curl -L --fail \
  https://raw.githubusercontent.com/cwida/ALP/31ca0ed11c93c99d3f5b5c30e01a3e1c3832d3ce/data/samples/poi_lat.csv \
  -o /tmp/alprd-poi-lat.csv

python3 - <<'PY'
from pathlib import Path
import struct
values = [float(v) for v in Path('/tmp/alprd-poi-lat.csv').read_text().split()]
assert len(values) == 1024
Path('/tmp/alprd-poi-lat.f64').write_bytes(struct.pack('<1024d', *values))
PY

taskset -c 7 _build/release/velox/dwio/nimble/encodings/benchmarks/nimble_alprd_benchmark \
  --profile --profile_type=double --input_file=/tmp/alprd-poi-lat.f64 \
  --rows=1024 --profile_trials=7 --profile_min_ms=50 > poi-lat.csv
```

Repeat with `poi_lon.csv` for longitude. These are artifact samples, not
full-column measurements. Source CSV SHA-256 hashes at the pinned commit:

```text
poi_lat.csv  488c7c20a857eb615ab85028360a1e29a0af02c0a96465c2d3d91e1a9a952931
poi_lon.csv  e1e65410cc1cb609f3963fb86681ffaa829aa9cc993aa1848aa3698f00a9ab72
```

## Timing contract

Operations are single-threaded and reuse warm input. ALP_RD uses Nimble's
default child selection, no secondary compression and no scratch buffer pools.
`--exact_bits` enables the existing exact-bit option for child encodings;
the default keeps byte-rounded FixedBitWidth storage. This flag does not
change ALP_RD's training heuristic or enable automatic selection.

| Operation | Timed work |
| --- | --- |
| `train` | Sampling, frequency counting and split/dictionary selection. |
| `encode` | Complete Nimble encoding: statistics, training, splitting, child selection, allocations, serialization and destruction. |
| `construct_decode` | Factory dispatch, decoder construction, child decoding, materialization and decoder destruction. |
| `reset_decode` | Reset and materialize an existing decoder. Construction, eager exception loading and destruction are outside timing. |

Generation/loading, snapshot copying, caller output allocation, validation and
reporting are outside timing. Decoder-internal work within the measured
operation is included. Both decoding lifecycles are checked bit-for-bit before
timing, including signed zero and NaN payloads; final output is checked again.

The profiler uses `std::chrono::steady_clock` and `CLOCK_THREAD_CPUTIME_ID`,
independently of Folly's timing runner. Each operation is calibrated to at least
`--profile_min_ms` CPU time, then measured for `--profile_trials` trials with
shuffled operation order. Wall/CPU medians are computed separately. Keep the
raw trials when assessing variability.

The paper's [RD encoding benchmark](https://github.com/cwida/ALP/blob/31ca0ed11c93c99d3f5b5c30e01a3e1c3832d3ce/publication/source_code/bench_speed/bench_alp_cutter_encode.cpp)
initializes parameters/buffers outside timing and measures RD encode plus two
FastLanes packing calls. Its decoding test measures unpack/reconstruction
kernels. Our `encode` has a different boundary; `reset_decode` still includes
Nimble reset/child-decoder work. Neither directly reproduces the paper's kernel
timings. The separate `train` result is diagnostic: subtracting it from `encode`
does not yield an independently measured kernel time. Comparisons must align
input, CPU/ISA, training scope and timed operations first.

## CSV output

Each operation emits `trial` and `median` records. `wall_ns` and `cpu_ns` are
per complete payload, divided by the calibrated iteration count.
`encoded_bytes` and `bits_per_value` include all root/child headers and payloads.
Dictionary size, right bit width, exception count and `exact_bits` describe the
actual encoding.

For file input, the dataset field is `input`; retain the command, input path/hash
and build flags alongside the CSV. These are codec measurements, not file I/O,
footer parsing, NULL handling, filtering or a multithreaded scan.
