# CuDF Benchmarks

Benchmark binaries for cast expressions and TPC-H/TPC-DS queries with cuDF GPU acceleration.

## Binaries

| Binary | Benchmark | Mode | Source |
|--------|-----------|------|--------|
| `velox_tpch_benchmark` | TPC-H (Q1-Q22) | CPU | `velox/benchmarks/tpch/` |
| `velox_cudf_tpch_benchmark` | TPC-H (Q1-Q22) | GPU | `CudfTpchBenchmark.cpp` |
| `velox_tpcds_benchmark` | TPC-DS (Q1-Q99) | CPU | `velox/benchmarks/tpcds/` |
| `velox_cudf_tpcds_benchmark` | TPC-DS (Q1-Q99) | GPU | `CudfTpcdsBenchmark.cpp` |
| `velox_cudf_cast_benchmark` | Spark numeric casts | CPU and GPU | `CudfCastBenchmark.cpp` |

CPU query binaries use HiveConnector. GPU query binaries use CudfHiveConnector and register
cuDF GPU operator replacements.

## Build

```bash
# GPU binaries (requires CUDA)
CUDA_ARCHITECTURES="native" EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_BENCHMARKS=ON" make cudf
cd _build/release
ninja velox_cudf_tpch_benchmark velox_cudf_tpcds_benchmark

# CPU-only binaries (no CUDA required)
ninja velox_tpch_benchmark velox_tpcds_benchmark
```

---

## Cast Expression Benchmark

`velox_cudf_cast_benchmark` compares native CPU Spark expressions with the
production cuDF expression evaluator in exact and experimental numeric modes.
It constructs a separate typed `spark_legacy_cast` evaluator for each GPU mode
and disables CPU fallback. It does not change the default conversion mode.

Configure with the cuDF benchmark settings above and
`-DVELOX_ENABLE_SPARK_FUNCTIONS=ON` (the default). From the repository root,
build and run:

```bash
cmake --build _build/release --parallel 8 --target velox_cudf_cast_benchmark

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
benchmark=_build/release/velox/experimental/cudf/benchmarks/velox_cudf_cast_benchmark

# Validate all inputs and both GPU modes before collecting timings.
"$benchmark" --cast_validate_only

# Five independent processes; each callback performs 100 conversion calls.
mkdir -p /tmp/cudf-cast-results
for run in 1 2 3 4 5; do
  "$benchmark" --cast_warmups=3 --cast_iterations=100 \
    --bm_min_iters=1 --bm_max_iters=2 --bm_max_trials=1 \
    --bm_json_verbose="/tmp/cudf-cast-results/run-$run.json" \
    > "/tmp/cudf-cast-results/run-$run.log" 2>&1
done
```

The cases cover supported numeric casts from the CPU benchmarks, decimal
rescaling, and REAL/DOUBLE to short and long decimals. Each uses 1,000, 65,536,
and 1,048,576 rows, with either no input NULLs or 20% NULLs. Inputs repeat
deterministic common values or rounding, nonfinite, and precision-overflow
fixtures. Java 17 decimal expectations use
`BigDecimal.valueOf((double) input).setScale(scale, HALF_UP)` followed by a
precision check. Explicit fixture expectations distinguish known CPU and numeric
mode differences from unexpected value or NULL-mask differences, which fail the
run. Validation prints difference counts and example input bits and outputs.

Folly reports elapsed time **per conversion call**, with units shown in its
output. Input generation, expression construction, input/output column transfers,
and validation are outside the timer. GPU calls include output allocation,
destruction, and stream synchronization. Production validity work, including
internal scalar/metadata transfers, remains timed. CPU calls include flattening and
`prepareForReuse`, reuse output buffers within each sample, and include the
sample's initial output allocation and final destruction. Each engine receives
untimed warmups. These are expression timings, not kernel-only measurements or
whole-query speedups.

Use `--cast_case_filter=double_rounding_and_nonfinite_rows1000_null0` to select
one input configuration, or Folly's `--bm_regex` to filter timed engine entries.
The case filter also limits validation; `--bm_regex` does not. The program prints
the CUDA device UUID and PCI address. A separate Nsight Systems run can confirm
the production `castKernel` exact and numeric specializations without adding
profiler overhead to timing runs:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --output=/tmp/cudf-cast-proof "$benchmark" --cast_validate_only \
  --cast_case_filter=double_rounding_and_nonfinite_rows1000_null0
```

---

## TPC-H Benchmark

### Data

Generate TPC-H parquet data using the [velox-testing](https://github.com/rapidsai/velox-testing) data generation tool (see [Data Generation](#data-generation) below), or the standard `dbgen` tool and convert decimal columns to float.

### Run

```bash
# CPU - all queries
./velox_tpch_benchmark --data_path=/path/to/tpch/sf100 --data_format=parquet

# GPU (CuDF) - all queries
./velox_cudf_tpch_benchmark --data_path=/path/to/tpch/sf100 --data_format=parquet
```

---

## TPC-DS Benchmark

TPC-DS plans are loaded from pre-dumped Velox plan JSON files (serialized from Presto).

### 1. Get Plan JSON Files

Clone the plans repository:

```bash
git clone https://github.com/karthikeyann/VeloxPlans.git
# Plans are at: VeloxPlans/presto/tpcds/sf100/
```

The directory contains `Q1.json`, `Q2.json`, ..., `Q99.json`.

### 2. Get TPC-DS Data

Generate TPC-DS parquet data using the [velox-testing](https://github.com/rapidsai/velox-testing) data generation tool (see [Data Generation](#data-generation) below). The data directory must have one subdirectory per table:

```
/path/to/tpcds/sf100/
  store_sales/
  customer/
  date_dim/
  item/
  ...
```

Each subdirectory contains parquet files for that table.

### 3. Run

**CPU - all queries (folly benchmark mode):**

```bash
./velox_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet
```

**CPU - single query with stats:**

```bash
./velox_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet \
  --run_query_verbose=1
```

**GPU (CuDF) - all queries:**

```bash
./velox_cudf_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet
```

**GPU (CuDF) - single query with stats:**

```bash
./velox_cudf_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet \
  --run_query_verbose=1
```

### TPC-DS Flags

These flags are shared by both CPU and GPU binaries:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | (required) | Root directory of TPC-DS table data |
| `--plan_path` | (required) | Directory containing Q*.json plan files |
| `--data_format` | `parquet` | Data file format |
| `--run_query_verbose` | `-1` | Run single query with stats (`-1` = run all) |
| `--num_drivers` | `4` | Number of parallel drivers |
| `--include_results` | `false` | Print query results |

### CuDF Flags (GPU binaries only)

These flags apply to `velox_cudf_tpch_benchmark` and `velox_cudf_tpcds_benchmark`:

| Flag | Default | Description |
|------|---------|-------------|
| `--cudf_chunk_read_limit` | `0` | Chunk read limit for cuDF parquet reader |
| `--cudf_pass_read_limit` | `0` | Pass read limit for cuDF parquet reader |
| `--cudf_gpu_batch_size_rows` | `100000` | GPU batch size in rows |
| `--velox_cudf_table_scan` | `true` | Use CuDF table scan |
| `--cudf_properties` | `""` | Path to a CudfConfig properties file (key=value per line). See `CudfConfig.h` for available keys |

---

## Data Generation

Both TPC-H and TPC-DS parquet data can be generated using the
[velox-testing](https://github.com/rapidsai/velox-testing) repository.
Full instructions are also available in the
[VeloxPlans TPC-DS README](https://github.com/karthikeyann/VeloxPlans/tree/main/presto/tpcds/sf100).

### Quick Start

```bash
# 1. Clone velox-testing
git clone https://github.com/rapidsai/velox-testing.git
cd velox-testing

# 2. Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r benchmark_data_tools/requirements.txt

# 3. Generate TPC-DS data (sf100)
python benchmark_data_tools/generate_data_files.py \
  --benchmark-type tpcds \
  --data-dir-path /path/to/tpcds/sf100/data \
  --scale-factor 100 \
  --convert-decimals-to-floats

# 4. Generate TPC-H data (sf100)
python benchmark_data_tools/generate_data_files.py \
  --benchmark-type tpch \
  --data-dir-path /path/to/tpch/sf100/data \
  --scale-factor 100 \
  --convert-decimals-to-floats
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--benchmark-type` | `tpcds` or `tpch` |
| `--data-dir-path` | Output directory for parquet files |
| `--scale-factor` | Scale factor (e.g. `1`, `10`, `100`) |
| `--convert-decimals-to-floats` | Convert decimal columns to double (recommended for Velox) |

The output directory will contain one subdirectory per table, each with `.parquet` files.
For a quick sanity check, use `--scale-factor 1` first.
