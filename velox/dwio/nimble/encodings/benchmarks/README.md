# Nimble Encoding Microbenchmarks

Per-encoding microbenchmarks measuring encode and decode (materialize) throughput on 100K elements across multiple data patterns.

## Running

```bash
# Run the encoding comparison (storage + decode time table for all encodings × all patterns)
buck run fbcode//velox/dwio/nimble/encodings/benchmarks:encoding_comparison

# Run a single encoding's folly::Benchmark microbenchmarks
buck run fbcode//velox/dwio/nimble/encodings/benchmarks:trivial_benchmark

# Run with minimum iterations
buck run fbcode//velox/dwio/nimble/encodings/benchmarks:rle_benchmark -- --bm_min_iters=10

# Build all benchmarks
buck build fbcode//velox/dwio/nimble/encodings/benchmarks:
```

## Deterministic encoding runner

`nimble_encoding_runner_bin` is a machine-readable Stage-0 runner for Nimble
encoding optimization. It currently accepts Dictionary, Nullable, ALP, and
DeltaBlock task IDs.

```bash
buck2 run fbcode//velox/dwio/nimble/encodings/benchmarks:nimble_encoding_runner_bin -- \
  --mode=sample \
  --task_id=nimble.dictionary.decode_dense.v1 \
  --artifact_path=/tmp/nimble-dictionary.bin
```

`sample` writes the raw encoded artifact when `--artifact_path` is supplied
and prints one `nimble_encoding_measurement` JSON object. `verify` reads an
artifact and prints a `nimble_encoding_verification` object after validating it
against the task's deterministic corpus:

```bash
buck2 run fbcode//velox/dwio/nimble/encodings/benchmarks:nimble_encoding_runner_bin -- \
  --mode=verify \
  --task_id=nimble.dictionary.decode_dense.v1 \
  --artifact_path=/tmp/nimble-dictionary.bin
```

Decode, view, and slice comparisons must time one canonical pristine artifact
in both binaries. Export it with the reference runner, then pass it to the
candidate sample without re-encoding:

```bash
buck2 run fbcode//velox/dwio/nimble/encodings/benchmarks:nimble_encoding_runner_bin -- \
  --mode=sample \
  --task_id=nimble.dictionary.decode_dense.v1 \
  --input_artifact_path=/tmp/nimble-dictionary.bin
```

The measurement JSON contains task identity, corpus parameters, encoded byte
count, input/output/artifact digests, correctness, and raw timing samples. It
describes one linked Nimble implementation and is deliberately not an
aggregate scoring report. A trusted host-side grader must run pristine and
candidate binaries, verify their artifacts in both directions, and merge their
outputs with manifest-owned scoring metadata.

## Encoding Comparison

The `encoding_comparison` binary prints a consolidated table of storage size and decode time for every encoding across all data patterns, for both uint32 and uint64:

```bash
buck run fbcode//velox/dwio/nimble/encodings/benchmarks:encoding_comparison
```

Output columns:
- **Raw(KB)** — unencoded data size
- **Encoded(KB)** — encoded size on disk
- **Ratio** — encoded/raw (lower = better compression)
- **Decode(ns/el)** — decode time per element, averaged over 100 iterations
- **SKIP** — encoding cannot handle this data pattern (e.g., Constant on non-constant data)

## Per-Encoding Benchmark Targets

| Target | Encoding | Supported Types |
|--------|----------|-----------------|
| `encoding_comparison` | All encodings | uint32, uint64 |
| `trivial_benchmark` | Trivial | All |
| `fixed_bit_width_benchmark` | FixedBitWidth | Numeric |
| `rle_benchmark` | RLE | All |
| `dictionary_benchmark` | Dictionary | All except bool |
| `delta_benchmark` | Delta | Integer |
| `constant_benchmark` | Constant | All |
| `mainly_constant_benchmark` | MainlyConstant | All except bool |
| `varint_benchmark` | Varint | 32/64-bit numeric |
| `sparse_bool_benchmark` | SparseBool | bool |
| `for_benchmark` | FOR (Frame of Reference) | Integer |
| `frequency_partition_benchmark` | FrequencyPartition | All hashable |
| `sub_int_split_benchmark` | SubIntSplit | 32/64-bit numeric |
| `nullable_benchmark` | Nullable (wraps Trivial) | All |

## Data Patterns

Each encoding is tested against all applicable patterns to show how it performs on both ideal and adversarial inputs:

| Pattern | Description | Ideal For |
|---------|-------------|-----------|
| `Random` | Uniform random values | Baseline / worst case |
| `Narrow8bit` | Values fit in 8 bits | FixedBitWidth, Varint |
| `Constant` | All identical values | Constant |
| `MainlyConstant` | 95% dominant value, 5% random | MainlyConstant |
| `RunLength` | Runs of 10–60 repeated values | RLE |
| `Increasing` | Monotonically increasing, small deltas | Delta, FOR |
| `LowCardinality` | 64 unique values | Dictionary, FrequencyPartition |

SparseBool additionally tests sparse (5% true), dense (50% true), and all-false.
Nullable varies null density (80%, 50%, 10% non-null) across patterns.

## Adding a New Encoding Benchmark

1. Create `<Encoding>Benchmark.cpp` in this directory.
2. Include `BenchmarkUtils.h` for shared data generators and encode/decode helpers.
3. Use the macro pattern (see existing files) to generate encode+decode benchmarks for each data pattern.
4. Add a `cpp_benchmark` target in `BUCK` with `deps = _COMMON_DEPS`.
