# CuDF Benchmarks

Benchmark binaries for TPC-H and TPC-DS queries with optional CuDF GPU acceleration.

## Binaries

| Binary | Benchmark | Mode | Source |
|--------|-----------|------|--------|
| `velox_tpch_benchmark` | TPC-H (Q1-Q22) | CPU | `velox/benchmarks/tpch/` |
| `velox_cudf_tpch_benchmark` | TPC-H (Q1-Q22) | GPU | `CudfTpchBenchmark.cpp` |
| `velox_tpcds_benchmark` | TPC-DS (Q1-Q99) | CPU | `velox/benchmarks/tpcds/` |
| `velox_cudf_tpcds_benchmark` | TPC-DS (Q1-Q99) | GPU | `CudfTpcdsBenchmark.cpp` |

CPU binaries use HiveConnector. GPU binaries use CudfHiveConnector and register
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
