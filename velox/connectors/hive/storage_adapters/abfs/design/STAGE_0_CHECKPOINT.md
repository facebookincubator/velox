# Stage 0 checkpoint: standalone native ABFS `preadvAsync`

Date: 2026-07-19
Verdict: **PASS**

## Scope and base

Standalone current-main baseline only. No production transport code is
included; the scope is the Stage 0 benchmark harness/CMake, standalone
specification/report, and evidence. No branch or commit was created. The
exact base is `06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`, from a detached WSL
ext4 checkout. A downstream PR is not required and is not a gate.

## Environment and dependencies

- Ubuntu 24.04.4 WSL2; kernel `6.18.33.2-microsoft-standard-WSL2`; ext4.
- `io_uring_disabled=0`; `ulimit -n=10240`.
- CMake `3.30.4`; GCC `13.3.0`; OpenSSL `3.0.13`.
- Azure Storage Data Lake SDK `12.8.0`, installed through its package config
  under a user-owned ext4 dependency prefix.
- Folly and Fizz `v2026.01.05.00`; Node `22.23.1`; Azurite `3.36.0`.
- liburing was not measured and is not needed for the Stage 0 reference
  baseline.

## Reproduction commands and timings

All private roots are sanitized placeholders. The environment used
`PATH=<user-local-bin>:/usr/local/bin:/usr/bin:/bin`,
`CMAKE_PREFIX_PATH=<dependency-prefix>`, and `--parallel 4`.

```text
env PATH=<user-local-bin>:/usr/local/bin:/usr/bin:/bin CMAKE_PREFIX_PATH=<dependency-prefix> cmake -S <repository-root> -B <release-build> -DCMAKE_BUILD_TYPE=Release -DVELOX_ENABLE_ABFS=ON -DVELOX_ENABLE_PARQUET=ON -DVELOX_ENABLE_HIVE_CONNECTOR=ON -DVELOX_ENABLE_EXEC=ON -DVELOX_ENABLE_BENCHMARKS=ON -DVELOX_BUILD_TESTING=ON -DCMAKE_PREFIX_PATH=<dependency-prefix>
cmake --build <release-build> --target velox_abfs_parquet_benchmark --parallel 4
cmake --build <release-build> --target velox_abfs_test velox_abfs_registration_test velox_dwio_parquet_reader_test --parallel 4
env PATH=<user-local-bin>:/usr/local/bin:/usr/bin:/bin ctest --test-dir <release-build> --output-on-failure --timeout 1800 -R '^(velox_abfs_test|velox_abfs_registration_test|velox_dwio_parquet_reader_test)$'
env PATH=<user-local-bin>:/usr/local/bin:/usr/bin:/bin timeout --signal=TERM --kill-after=30s 900s <release-build>/velox/connectors/hive/storage_adapters/abfs/tests/velox_abfs_parquet_benchmark

env PATH=<user-local-bin>:/usr/local/bin:/usr/bin:/bin CMAKE_PREFIX_PATH=<dependency-prefix> cmake -S <repository-root> -B <debug-build> -G Ninja -DCMAKE_BUILD_TYPE=Debug -DVELOX_ENABLE_ABFS=ON -DVELOX_ENABLE_PARQUET=ON -DVELOX_ENABLE_HIVE_CONNECTOR=ON -DVELOX_ENABLE_EXEC=ON -DVELOX_ENABLE_BENCHMARKS=ON -DVELOX_BUILD_TESTING=ON -DCMAKE_PREFIX_PATH=<dependency-prefix>
cmake --build <debug-build> --target velox_abfs_test velox_abfs_registration_test velox_dwio_parquet_reader_test velox_abfs_parquet_benchmark --parallel 4
env PATH=<user-local-bin>:/usr/local/bin:/usr/bin:/bin ctest --test-dir <debug-build> --output-on-failure --timeout 1800 -R '^(velox_abfs_test|velox_abfs_registration_test|velox_dwio_parquet_reader_test)$'
```

Release configure took 19.35s; the three Release test targets built in
173.542s. Debug configure took 30.544s; the four Debug targets built in
3918.536s. The Release benchmark target built successfully. Non-fatal
configure warnings were CMake `CMP0167` and the existing
`VELOX_GTEST_INCUDE_DIR` parent-scope warning.

## Tests

All three tests passed in both configurations; no failures or skips were
reported. Release: Parquet 6.17s, ABFS 57.69s, registration 0.07s; total
64.16s, outer 64.49s. Debug: Parquet 29.26s, ABFS 58.02s, registration
0.07s; total 87.57s, outer 87.73s.

## Benchmark fixture and controls

Schema `id:BIGINT,payload:BIGINT`; seed `1000000007`; `id=seed+row*1000003`;
`payload=(id<<7)^(id>>3)^5a5a5a5a5a5a5a5a`. The fixture has 48,000 rows,
four row groups, and a 938,794-byte local file. Eight identical splits/copies
produce 384,000 rows. Fixture checksum:
`16009607394047081120`; result checksum modulo $2^{64}$:
`17396394710119339264`.

Exactly three sequential rounds were run for IO executor sizes 1, 2, and 8,
with eight query drivers, cold cache, preload disabled, file-handle cache
disabled, and the synchronous Azure SDK default transport over local Azurite.
Every round reported `storageReadBytes=1,199,104`, table scan input bytes
`7,327,744`, raw input `9,592,832`, and stable rows/checksum. Storage
operation count equals the derived one-download-per-current-ABFS-`preadv`
count, not an observed wire HTTP count; observed HTTP count was unavailable.

| IO threads | Wall mean / sample SD / min / max / CV (us) | Total CPU mean / SD / min / max / CV (us) | Ops | Peak threads | Peak RSS (KiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 47316.67 / 7349.47 / 41327 / 55518 / 15.53% | 45521.67 / 12470.26 / 36057 / 59652 / 27.39% | 5 / 4 / 4 | 20 | 125448 |
| 2 | 38819.33 / 2354.51 / 36367 / 41062 / 6.07% | 37137.33 / 3076.17 / 34467 / 40501 / 8.28% | 4 / 4 / 4 | 20 | 133496 |
| 8 | 34506.0 / 2364.89 / 32668 / 37174 / 6.85% | 37949.67 / 2728.03 / 35761 / 41006 / 7.19% | 5 / 3 / 4 | 27 | 140260 |

Neutral interpretation: local synchronous wall means improve with more IO
executor threads while process threads rise. This makes no native
async/fiber/io_uring claim, and WSL/Azurite is not production performance.

## C0 criteria

- Clean baseline tests: **PASS**.
- Reproducible current-main baseline without a downstream PR: **PASS**.
- Repeatable Release benchmark commands and input identities: **PASS**.
- Three sequential rounds and variance recorded: **PASS**.
- Reproduction without private paths or credentials in source, logs, or
  command lines: **PASS**.

## Evidence

All paths are repository-relative:

- CSV: `velox/connectors/hive/storage_adapters/abfs/design/results/stage0-main-06dec49a/abfs-parquet-baseline.csv`
- Summary JSON: `velox/connectors/hive/storage_adapters/abfs/design/results/stage0-main-06dec49a/abfs-parquet-baseline-summary.json`
- Empty stderr: `velox/connectors/hive/storage_adapters/abfs/design/results/stage0-main-06dec49a/abfs-parquet-baseline.stderr.txt`
- Run file: `velox/connectors/hive/storage_adapters/abfs/design/results/stage0-main-06dec49a/abfs-parquet-baseline-run.txt`
- Environment: `velox/connectors/hive/storage_adapters/abfs/design/results/stage0-main-06dec49a/environment.txt`
- Build summary: `velox/connectors/hive/storage_adapters/abfs/design/results/stage0-main-06dec49a/tests/build-summary.txt`
- Release CTest: `velox/connectors/hive/storage_adapters/abfs/design/results/stage0-main-06dec49a/tests/release-ctest.txt`
- Debug CTest: `velox/connectors/hive/storage_adapters/abfs/design/results/stage0-main-06dec49a/tests/debug-ctest.txt`
- Preflight: `velox/connectors/hive/storage_adapters/abfs/design/preflight/stage-0-preflight.txt`

## Unresolved risks

WSL/Azurite only; one-thread variance; derived request count rather than wire
count; no async behavior in C0; native Linux and real Azure validation later;
liburing/io_uring/Fizz compatibility later.

## Exact next entry: Stage 1 only

Read the specification and this report. Keep the Stage 0 harness unchanged.
Prove the Azure fiber bridge over the reference EventBase with the real
synchronous SDK, disabling retries only for the isolated spike. Use a
deterministic delayed HTTP/TLS server and establish executable C1 criteria for
pending submission, concurrent progress, bounded buffering, TLS verification,
and fiber-wait behavior. Downstream reader integration remains deferred.
