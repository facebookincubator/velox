# CuDF Benchmarks

Benchmark binaries for TPC-H and TPC-DS queries with optional CuDF GPU acceleration.

## Binaries

| Binary | Benchmark | Source |
|--------|-----------|--------|
| `velox_cudf_tpch_benchmark` | TPC-H (Q1-Q22) | `CudfTpchBenchmark.cpp` |
| `velox_cudf_tpcds_benchmark` | TPC-DS (Q1-Q99) | `CudfTpcdsBenchmark.cpp` |

Both binaries support CPU-only (Hive) and GPU-accelerated (CuDF) modes.

## Build

```bash
CUDA_ARCHITECTURES="native" EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_BENCHMARKS=ON" make cudf
cd _build/release
ninja velox_cudf_tpch_benchmark velox_cudf_tpcds_benchmark
```

---

## TPC-H Benchmark

### Data

Generate TPC-H parquet data using the [velox-testing](https://github.com/rapidsai/velox-testing) data generation tool (see [Data Generation](#data-generation) below), or the standard `dbgen` tool and convert decimal columns to float.

### Run

```bash
# CPU (Hive) - all queries
./velox_cudf_tpch_benchmark --data_path=/path/to/tpch/sf100 --data_format=parquet

# GPU (CuDF) - all queries
./velox_cudf_tpch_benchmark --data_path=/path/to/tpch/sf100 --data_format=parquet \
  --velox_cudf_table_scan=true
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

**CPU (Hive) - all queries (folly benchmark mode):**

```bash
./velox_cudf_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet
```

**CPU (Hive) - single query with stats:**

```bash
./velox_cudf_tpcds_benchmark \
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
  --data_format=parquet \
  --cudf_enabled
```

**GPU (CuDF) - single query with stats:**

```bash
./velox_cudf_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet \
  --cudf_enabled \
  --run_query_verbose=1
```

### TPC-DS Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | (required) | Root directory of TPC-DS table data |
| `--plan_path` | (required) | Directory containing Q*.json plan files |
| `--data_format` | `parquet` | Data file format |
| `--run_query_verbose` | `-1` | Run single query with stats (`-1` = run all) |
| `--num_drivers` | `4` | Number of parallel drivers |
| `--include_results` | `false` | Print query results |

### CuDF-specific Flags (with `--cudf_enabled`)

| Flag | Default | Description |
|------|---------|-------------|
| `--cudf_enabled` | `false` | Enable CuDF GPU acceleration |
| `--cudf_chunk_read_limit` | `0` | Chunk read limit for cuDF parquet reader |
| `--cudf_pass_read_limit` | `0` | Pass read limit for cuDF parquet reader |
| `--cudf_gpu_batch_size_rows` | `100000` | GPU batch size in rows |
| `--cudf_memory_resource` | `async` | RMM memory resource type |
| `--cudf_memory_percent` | `50` | Percentage of GPU memory to allocate for pool memory resource only |
| `--velox_cudf_table_scan` | `true` | Use CuDF table scan |
| `--cudf_debug_enabled` | `false` | Enable debug printing |

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
