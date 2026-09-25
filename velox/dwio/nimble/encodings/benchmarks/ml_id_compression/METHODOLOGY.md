# Sweep methodology

What the sweep measures, what was removed from it, and why. Each entry is
marked **measured** or **judgement** so a reader can tell which reductions rest
on data and which rest on an argument.

The drivers stay general: every reduction below is expressed as a flag with the
old behaviour as its default, and the configuration lives in the launcher
scripts. A run with no flags reproduces every earlier sweep exactly.

## Arms: 38 → 31

| Arm | Verdict | Why |
|---|---|---|
| `SIS/legacyCost`, `+view` | **judgement** | Pinned the cost model that preceded Delta, FOR, PFOR, Huffman, DeltaBlock, BlockBitPacking, SimdForBitpack and FrequencyPartition. Nothing proposes returning to it. |
| `SIS/huffOff`, `+view` | **measured** | Byte-identical to `SIS/realNested` in 26 of 26 cells. |
| `SIS/huffOff/key_derived`, `+view` | **measured** | Byte-identical to `SIS/key_derived` in 26 of 26 cells. |
| `FPE/fpe_tagtag` | **measured** | Dominated on every axis: worst on point in 39 of 42 cells, worst on bulk in 42 of 42, mid-table on size. |

Two arms were deliberately **kept** against the obvious reading:

- **`SIS/huffOn`** differs from `realNested` in 9 of 26 cells, so it is a real
  ablation rather than a duplicate. What moves is not which encoding a section
  gets — Huffman is not a candidate on this path — but where the split
  boundaries fall, because the planner costs Huffman when it scores a bit
  range.
- **`FPE/fpe_pertier`** is worst on compression in 26 of 26 cells, which makes
  it look like the next cut. It is competitive on point access (76.9ns against
  `fpe_noindex`'s 74.7ns). Cutting on size alone would have removed the wrong
  arm.

## Gather: selectivity ladder replaces the linear axis

**measured.** `linSpaced(0.05, 1.0, 8)` spends most of its cells where nothing
happens. On snowflake the run-length axis spans **60.45x** at selectivity 0.05
(1.8 to 107.0 Meps) and **1.01x** at selectivity 1.0. Small gathers carry all
the nuance; large ones coalesce into a scan.

Ladder: `0.001, 0.01, 0.05, 0.10, 0.33, 0.66, 1.00` via `--selectivity_values`.

## Gather: four run lengths instead of six

**judgement**, not measured. Keep `1, ~111, ~12417, ~131072`; drop `~10` and
`~1175`. The existing pull only contains run lengths 1 and 131072, so there is
no data on which interior points matter — this is a reasonable spacing, not a
demonstrated one. If a later sweep shows structure between 111 and 12417, this
is the reduction to revisit first.

Ladder via `--run_length_values`.

## Range: log-spaced sizes, offsets scaled to size

**measured.** Spread across offsets at fixed size:

| B | 1 | 8 | 64 | 512 | 4096 | 32768 |
|---|---|---|---|---|---|---|
| spread | 1.03x | 1.54x | 1.86x | **7.44x** | 3.06x | 1.62x |

A handful of offsets suffices at either end — a single row lands the same way
wherever it is, and a large span averages over its own placement — but the
middle needs about 32 before the median settles. A flat 32 buys nothing at the
ends; a flat 8 leaves the middle untrustworthy.

This replaces the `grid=16` triangle (136 cells, minimum slice 1/16 of the
column) with an explicit size list spanning the small-slice regime through the
whole column, via `--range_sizes` and `--range_offsets_by_size`.

## Cold cache states: kept

**measured, and a correction to an earlier reading.** Cold was reported as a
null result because `table4_cold_cache.tex` tabulated only
`SIS/realNested+view` and `openzl/auto` — the two arms that show nothing.

`FixedBitWidth/view` loses **1.84x** cold at selectivity 0.05 with long runs
(2305.2 → 1252.4 Meps) and 1.23x at full selectivity. The fast arms are
memory-bound, so the refault dominates them; SIS and OpenZL are compute-bound
and hide it. Cold matters, for the whitebox arms specifically, and it narrows
the throughput gap SIS has to close — so the table has to include the whitebox
arms or it discards a point in our favour.

## `compzl`: compression driver only

**judgement.** Sub-stream OpenZL compression changes the bytes, not the read
path, so it belongs on the driver that measures bytes. It is one iteration and
therefore cheap. It does not need to appear on any decode driver.

## Compression driver iterations

**measured, and a non-change.** The compression driver has no measurement loop
and never reads `--mlidc_iters`; it already encodes exactly once per arm.
Passing `--mlidc_iters=1` to it was always a no-op. Recorded so nobody adds a
change that does nothing.

## Encode cache

Every driver encodes its own targets, so a sweep paid each encode once per
driver. `--mlidc_encode_cache_dir` makes a target encoded once and reused by
every later driver. Off by default.

The key covers the input bytes themselves, the arm identity, a hash of the
encoder sources, and every flag not on an explicit measurement-only allowlist.
A source hash is only sound while the binary matches the sources, so the cache
disables itself, loudly, if any hashed source is newer than the running binary.
A wrong hit would make every downstream number wrong and mutually consistent,
which is the hardest kind of error to notice, so the key errs toward missing.

Known limit, reasoned rather than tested: a checkout that restores an *older*
mtime with changed content would not trip the staleness check, but is still
safe, because the content hash changes the key.

## Repeats and pinning

**measured, and a correction.** Three repeats, unpinned.

The five-repeat, `taskset -c 2` rule belongs to the laptop, where unpinned runs
are bimodal with a 3.34x spread. This box's unpinned noise floor is about 0.8%,
and the pinned baseline spreads actually measured came out at 0.3-1.4% -- the
same floor. The pinning was buying nothing, and at a 0.8% floor five repeats is
more than the medians need.

## Per-driver arm sets

**measured.** The grid reductions above do not pay for themselves on their own,
because measurements are cells multiplied by arms. At 31 arms the gather ladder
is roughly break-even against the old axis, and the range ladder is worse than
the old triangle. Cutting the arm set is what makes the grid reduction count.

- **Access drivers** (bulk, point, gather, range) take **24 arms**: the whole
  SIS family, `openzl/auto`, the `FixedBitWidth` pair, and the three
  surviving FPE index variants.
- **Encode driver** takes **14 arms**: the SIS family without `+view`, plus
  `openzl/auto`, `FixedBitWidth`, and the FPE variants.
- **Compression and compzl** take all **31**. One iteration each, so they are
  cheap, and the ablation table needs the full set.

`FixedBitWidth/view` is the best whitebox arm on every access driver: best in 20
of 42 gather cells (median 1321.7 Meps), 23 of 42 range cells (1381.8), and 12
of 14 skip cells (1309.9). The cursor variant ties on medians and takes most of
the remaining cells, so both stay -- the pair is the view-versus-cursor
comparison, not redundancy.

`Dictionary` and `RLE` are excluded from the access set deliberately: their
access numbers in the existing pull came from the defective nested-selection
configuration and would have to be re-measured before they meant anything.
`FixedBitWidth` and `Trivial` have no sub-streams and were never affected, which
is what makes the best-whitebox finding safe to build on.

The forced-transform SIS arms stay. Their decode behaviour is a claim in its own
right: `key_derived` costs 1.80x on point access on `publicbi_npi`, the
compresses-better-probes-worse case that the compression numbers alone hide.

## Resulting cell counts

Measurements = arms x grid cells x 26 dataset-orders x 2 cache states x 3
repeats. Grid cells are per dataset-order-cache-state.

| Driver | grid cells | arms | measurements |
|---|---|---|---|
| bulk | 1 | 24 | 3,744 |
| point | 1 | 24 | 3,744 |
| gather | 28 (was 48) | 24 (was 31) | 104,832 (was 232,128) |
| range | 121 (was 136) | 24 (was 31) | 452,608 (was 657,696) |
| **access total** | | | **565,032** (was 899,496) |
| compression + compzl | 1 | 31 | 1,612 (one iteration, no repeats) |

37% fewer measurements than the same configuration at 31 arms and 3 repeats,
and 62% fewer than the original 31-arm, 5-repeat, old-grid configuration
(1,499,160).

## RANGE_SIZES is tied to N and must move with it

The ladder's top entry is the whole column. At N=524288 that is the 524288 entry
in `sweep_config.sh`; at N=2000000 the top entry has to become 2000000, or the
sweep silently stops measuring the whole-column case and the largest slice
becomes a quarter of the column. The lower rungs are absolute sizes and stay put
-- the point of them is the small-slice regime, which does not scale with N.

## The FPE family is kept, and is not baseline colour

**judgement, corrected.** `FPE/fpe_noindex`, `fpe_pertier` and `fpe_elias` stay
on both the access and encode drivers. They were briefly grouped with the other
whitebox baselines and cut; that was wrong on the merits. The FPE index
variants are the paper's showcase for making a value-grouping encoding randomly
addressable, so their access numbers are a claim the text depends on rather
than context around it.

`fpe_pertier` is the specific trap: it is worst on compression in 26 of 26
cells, which has twice made it look like the obvious next cut, and it is
competitive on point access at 76.9ns against `fpe_noindex`'s 74.7ns. Cutting
the family would have removed the evidence for the section that argues for it.

`fpe_tagtag` stays dropped. It is dominated on all three axes, which is
measured and is unaffected by keeping the rest of the family.

## Encode driver: no `+view` arms

**measured.** A view is a read path, so a `+view` arm encodes the same bytes by
the same route as its cursor twin, and measuring both measured one of them
twice.

Across **546 view/cursor pairs** in the existing `enc_*` CSVs: **zero payload
byte mismatches**, encode-throughput ratio median **0.993**, and 531 of 546
pairs within ten percent. The 0.819-1.527 spread is measurement noise on a
quantity that should be identical.

**This is a measured equivalence, not an inference from reading the code, and it
is load-bearing.** If the view ever gains an encode-time component -- an index
built at encode, a second pass, anything that makes `encodeWith` do work its
cursor twin does not -- the equivalence breaks and every encode number for the
SIS family becomes an average over two different things. The `+view` arms would
have to be restored on the encode driver at that point. Anyone changing what a
view does at encode time should re-run the pair comparison before assuming this
still holds.

## The encode driver ignores the encode cache

**Structural, and enforced in code.** `MlIdEncodeBenchmark` times
`enc.factory()`, and the encode cache sits inside it. With a cache directory
set, the first iteration would encode and store and every iteration after it
would load, so `measure()`'s median would report a cache read rather than an
encode -- wrong, and wrong by a consistent factor across every arm.

The driver therefore clears `--mlidc_encode_cache_dir` at startup and says so,
rather than leaving it to whoever writes the launcher. The flag is one every
other driver wants set, so the mistake would be easy to make and silent to
suffer.

## Point access is bounded

**measured, and a concern of mine that did not survive checking.** I flagged
point as a possibly unbounded term because OpenZL costs ~6.1ms per probe.
`--mlidc_block_codec_probes` defaults to **64** and is applied in the point
driver, so an OpenZL point cell costs about 64 x 6.1ms = 0.39s, and about a
minute across the whole sweep. The 65,536 default on `--probes` applies only to
the cheap arms, where a probe is ~500ns and a full cell is ~33ms. No cap change
is needed.
