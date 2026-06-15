# CXL hash-aggregation benchmark

Measures a grouping aggregation, selected with `--query`:

```sql
-- --query=q18 (default)
SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 312
-- --query=q17
SELECT l_partkey, avg(l_quantity) FROM lineitem GROUP BY l_partkey
-- --query=zipf (synthesized input; --zipf_groups, --zipf_skew)
SELECT k, sum(v) FROM zipf_rows GROUP BY k
```

across three memory-placement strategies, to see whether building the
aggregation in DRAM and relocating it to CXL under pressure (`CxlHashAggregation`)
beats the alternatives. The query is driven Velox-local (TableScan →
aggregation → filter) so all configs are apples-to-apples in one process; there
is no Presto/Spark layer.

## Configurations

| Flag | Role | Memory |
| --- | --- | --- |
| `--config=dram` | A — no-CXL competitor | DRAM pool capped; on-disk spill |
| `--config=interleave` | B — OS striping | DRAM+CXL interleaved by `numactl` |
| `--config=cxl` | C — ours | DRAM capped; relocate overflow to CXL |

A and C share the same DRAM cap (`--dram_limit_mb`), differing only in overflow
target (disk vs CXL); the driver script sweeps the cap (`DRAM_MB_LIST`) so the
cap's effect on each is visible. (A fourth config, `--config=dram_big` —
uncapped DRAM, no spill — exists in the binary as an optional speed-ceiling
reference; run it directly if wanted.)

## Two queries: clustered vs random key arrival

The queries share the operator shape (single hash aggregation over two lineitem
columns) and differ only in how grouping keys arrive, which controls how often
relocated (CXL-resident) payload is touched again:

- **q18 — clustered (the relocation-friendly case).** dbgen emits lineitem
  ordered by orderkey, so a group's 1–7 rows arrive consecutively: one cold
  probe per group, then cache hits, then the group is never updated again.
  Payload relocated to CXL belongs to finished groups and stays cold until
  output.
- **q17 — random (the relocation-adversarial case).** `l_partkey` is
  uniform-random across rows, so each of the 200K-per-SF parts is probed ~30
  times scattered over the entire scan. Relocated payload keeps taking updates
  at CXL latency for the rest of the run, which is exactly the case a
  hotness-blind relocation policy should be stressed against.
- **zipf — skewed.** Synthesized `(k, v)` rows (no TPC-H): keys drawn
  Zipf(`--zipf_skew`) over `--zipf_groups` ranks, scattered across the key
  space, random arrival, row count = lineitem at the same SF. A few hot groups
  take most updates. Requires `--pregen` (default).

q17 has 7.5× fewer groups than q18 at the same SF (200K vs 1.5M per SF), so it
needs a larger SF to push the group table past LLC and past interesting DRAM
caps — see the tuning table below. Neither TPC-H query is skewed (the spec is
uniform); `--query=zipf` is the skewed data point.

By default (`--pregen`) the two scanned columns are materialized **once** at
startup and the trials aggregate from a `Values` node, so per-trial time
measures the aggregation rather than on-the-fly TPC-H row generation (dbgen
synthesizes all 16 lineitem columns per row and dominates a streamed scan).
The pregenerated input lives outside the capped query pool. Pass
`--nopregen` to stream from the connector instead, for scale factors whose
two-column input does not fit in memory (roughly SF >= 100; ~16 B/row).

The default scale factor for the driver script is **SF10** (~60M rows, ~15M
groups, ~1GB group table), where the 300-600MB cap sweep clears the ~256MB index floor and creates
pressure; SF1's ~80MB working set fits under any interesting cap.

B's DRAM share is controlled differently: a Velox byte cap would not change
where the kernel places pages, so the script sweeps the **placement ratio**
instead, via weighted interleave (`WEIGHTS_LIST`, e.g. `"4:1 2:1 1:1 1:2"`
dram:cxl). This needs Linux 6.9+ (`/sys/kernel/mm/mempolicy/
weighted_interleave`) and a numactl with `--weighted-interleave`; weights are
global sysfs settings and writing them requires root. Without support, B runs
only the classic 1:1 stripe.

The two sweeps line up: B at ratio `r` spends `r × footprint` bytes of DRAM
spread uniformly over every page, while C at `--dram_limit_mb ≈ r × footprint`
spends the same DRAM budget structurally (bucket array + not-yet-relocated
rows). Comparing the matched pairs isolates what semantic placement buys over
blind striping at an equal DRAM budget.

## 1. Bring the CXL device online as a NUMA node

```bash
sudo scripts/cxl_numa_setup.sh dax0.0
```

This reconfigures the CXL expander to `system-ram` and prints the resulting
CPU-less NUMA node id. The same node is used for `--membind` (config C's pool
binds itself there) and for `--interleave=0,<node>` (config B) — `membind` and
`interleave` are different `numactl` policies over the same node, not different
setups.

## 2. Build

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DVELOX_ENABLE_CXL=ON \
  -DVELOX_ENABLE_BENCHMARKS=ON <other flags> .
make velox_cxl_aggregation_benchmark
```

Linux only. Requires the libnuma **development** package — `libnuma-dev`
(Debian/Ubuntu) or `numactl-devel` (RHEL/CentOS) — for the `numa.h` header and
the `libnuma.so` symlink that CMake's `find_library(numa)` resolves; the runtime
`libnuma1` alone is not enough, and configure aborts with a `libnuma not found`
error. **Benchmark only release builds** — debug numbers are 5-20x inflated and
meaningless.

Alternatively, build via the Makefile target, which wires the flags for you:

```bash
make benchmarks-build EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_CXL=ON"
```

`VELOX_ENABLE_BENCHMARKS=ON` auto-enables `VELOX_BUILD_TEST_UTILS`, so
`velox_exec_test_lib` (the `PlanBuilder` harness the benchmark links) is built
even though this target sets `VELOX_BUILD_TESTING=OFF`.

## 3. Run

```bash
BENCH=_build/release/velox/experimental/cxl/benchmarks/velox_cxl_aggregation_benchmark \
CXL_NODE=2 DRAM_NODE=0 \
DRAM_MB_LIST="300 400 500 600" WEIGHTS_LIST="4:1 2:1 1:1 1:2" CXL_MB=4096 \
  ./run_cxl_benchmark.sh
```

This sweeps the interleave ratio over `WEIGHTS_LIST` for config B, then the
DRAM cap over `DRAM_MB_LIST` for the `dram` and `cxl` configs. Set `QUERY=q17`
(with `SF` and caps from the tuning table below) for the random-key query. Or
run a single config directly:

```bash
numactl --cpunodebind=0 --membind=0 \
  ./velox_cxl_aggregation_benchmark --config=cxl --cxl_numa_node=2 \
  --cxl_capacity_mb=4096 --dram_limit_mb=400
```

Each run reports median elapsed time plus an aggregation-operator breakdown
(CPU, wall, blocked, peak), scan IO time, spill bytes/time (config A), CXL
relocations (config C), and a result row count + checksum.

## Reading the results

The raw log interleaves results with arbitrator stack dumps (see the
`MEM_CAP_EXCEEDED` note below). For a one-line-per-leg table, pipe it through the
summarizer:

```bash
./run_cxl_benchmark.sh ... | scripts/summarize.sh
# or on a saved log:
scripts/summarize.sh results/sf30_raw.log
```

- **Correctness**: the `checksum` and `rows` lines must match across all
  configs.
- **Relocation fired (config C)**: `cxl relocations` must be `> 0`. If it is `0`,
  the DRAM cap did not trigger the arbitrator — lower `--dram_limit_mb`.
- **`MEM_CAP_EXCEEDED` stack traces are expected, not a crash (config C)**: at
  the tightest caps the log fills with full `SharedArbitrator::growCapacity`
  dumps ending in `ErrorCode: MEM_CAP_EXCEEDED`. That failed DRAM grow is
  precisely what triggers relocation to CXL — the run continues and finishes
  with a correct checksum. Treat them as noise unless the process actually exits
  non-zero or `LEG FAILED` is printed by the driver.

## Expectation, and the honest caveat

`GROUP BY l_orderkey` (q18) is high-cardinality but **near-uniform** (each
order has 1–7 lineitems) and clustered, and the current operator does
**wholesale** relocate (not cold-aware). So expect:

- **C ≫ A**: C avoids the serialize/sort/disk/merge of external aggregation;
  rows stay live and byte-addressable on CXL.
- **C ≈ B**: C wins by pinning the bucket array in DRAM; B wins by leaving some
  payload on DRAM by luck of the stripe. Roughly a wash on this query.
- **C < A1**: the DRAM ceiling is faster — C pays CXL latency on payload.

q18's clustered keys mean relocated payload is never updated again, so it is
the **best case** for relocation. `--query=q17` is the counterweight: random
key arrival keeps updating relocated payload at CXL latency for the whole run.
Expect C ≫ A to persist (spill still pays the re-aggregation CPU), but C vs B
may invert: how much C can beat B is bounded by how much placing half the table
on CXL costs interleave in the first place.

Neither TPC-H query has **access skew** (uniform by spec); `--query=zipf`
supplies it for the C-vs-B comparison.

The benchmark must use the **real** NUMA-bound CXL pool (it fails if
`--cxl_numa_node`/`--cxl_capacity_mb` are unset for `--config=cxl`); never wire
it to the unit tests' `MallocAllocator` resource, which is DRAM-speed and would
make C's numbers meaningless. `--cxl_capacity_mb` is pre-reserved by the
allocator, so size it to the device.

## Tuning note: the DRAM cap has a floor

Relocation moves row payload to CXL but the bucket array (`table_`) stays
pinned in DRAM by design, so `--dram_limit_mb` must exceed the bucket-array
size at the query's group count (tens of MB at SF1's ~1.5M groups) plus scan
working memory. The benchmark warns when no relocation fired; sweep the cap if
that happens.

The floor scales with the group count, which scales with SF (1.5M groups per
SF for q18, 200K per SF for q17). Anchoring on **~256 MB at q18 SF10** (15M
groups), pick caps just above the floor and step up from there. `CXL_MB` must
hold the relocated payload (total group table minus the DRAM cap), so size it
up with SF too:

| Query | SF  | Groups | Bucket-array floor | `DRAM_MB_LIST`                  | `CXL_MB` |
|-------|----:|-------:|-------------------:|---------------------------------|---------:|
| q18   | 10  | 15M    | ~256 MB            | `300 400 500 600` (default)     | `4096`   |
| q18   | 30  | 45M    | ~768 MB            | `1000 1500 2000 2500`           | `8192`   |
| q18   | 100 | 150M   | ~2.5 GB            | `3000 4000 5000 6000` + `--nopregen` | `16384` |
| q17   | 100 | 20M    | ~256 MB (est.)     | `300 400 500 600` + `--nopregen` | `4096`  |
| zipf  | 10  | 10M (`--zipf_groups`) | ~170 MB (est.) | `200 300 400 500` | `4096` |

For zipf, pick `--zipf_groups` so rows ≫ groups (60M rows over 10M groups at
SF10 ≈ 6 hits per group on average, far more on the hot head) — with groups ≳
rows most groups are hit once and the skew that distinguishes C from B is lost.

The q17 row is estimated, not yet measured: 20M groups land in the same
power-of-two bucket count as q18 SF10's 15M, so the floor and cap sweep should
match; read the measured peak off an uncapped `dram_big` leg and adjust. Lower
SFs are uninteresting for q17 — at SF10 its ~2M-group table fits in a large
server's LLC, and the cache hides exactly the random-access penalty the query
exists to measure.

A cap below the floor is a legitimate `LEG FAILED` for config C (the bucket
array alone will not fit), not a bug.

Building this benchmark surfaced (and led to fixing) a real operator gap:
`CxlHashAggregation::addInput` originally had no `ensureInputFits`-style
reservation handshake, so the driver-installed non-reclaimable section meant
the arbitrator could never reach `reclaim()` and a DRAM cap hard-failed the
query instead of relocating. The operator now reserves memory at a safe point
inside a reclaimable section before each input batch (mirroring
`GroupingSet::ensureInputFits`), which is what makes config C measurable;
`CxlHashAggregationTest.relocatesViaMemoryArbitratorUnderCappedPool` covers the
real arbitration path.
