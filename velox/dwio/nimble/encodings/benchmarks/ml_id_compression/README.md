# ML ID Compression Benchmarks

Benchmark drivers for 64-bit ML ID columns, built to assess `SubIntSplitEncoding`
(SIS) against the other Nimble encodings and against OpenZL as a black-box codec.

These drivers are gated on `NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS`. Without it the
sources compile to a stub `main` that exits non-zero.

## Drivers

| Target | Measures | OpenZL |
|---|---|---|
| `nimble_ml_id_smoke_benchmark` | Encode and decode round-trip correctness, no timing | no |
| `nimble_ml_id_compression_benchmark` | Encoded size, ratio, bits per element | yes |
| `nimble_ml_id_encode_benchmark` | Encode throughput | yes |
| `nimble_ml_id_decode_bulk_benchmark` | Full-materialisation decode throughput | yes |
| `nimble_ml_id_decode_range_benchmark` | Contiguous-range decode throughput | yes |
| `nimble_ml_id_decode_point_benchmark` | Single-probe point-lookup latency | yes |
| `nimble_ml_id_decode_gather_benchmark` | Gather decode throughput over a (selectivity, run length) grid | yes |
| `nimble_ml_id_cost_model_oracle_benchmark` | SIS DP cost model against a measured oracle | no |
| `nimble_ml_id_index_oracle_benchmark` | FPE index-type sweep | no |
| `nimble_ml_id_ablation_benchmark` | Progressive encoding-set restriction for SIS sections | no |

## Datasets

Six synthetic datasets are always present: `uniform-full`, `narrow-20bit`,
`narrow-40bit`, `increasing-small-delta`, `low-cardinality-256`, `run-length`.

`narrow-40bit` appears only for the 8-byte types. A 40-bit draw has no meaning
in a 32-bit element, so for `int32`, `uint32` and `float` the suite is five
datasets rather than six.

The same six names are used whatever `--mlidc_dtype` is set to, so one
`--mlidc_datasets` selection works across types and rows line up dataset by
dataset. For `float` and `double` the generators work in the value domain
rather than the bit domain: they produce ordinary finite values with fractional
parts, because bit-casting random words into floats yields mostly NaNs and
denormals, which compress unlike any real float column.

A real column is added with `--mlidc_file`, which takes **a text file with one
value per line**, parsed as the type named by `--mlidc_dtype`. The `int64` case
is deliberately the same format as the `--file` flag of
`velox/dwio/nimble/tools/encoding_bench`, so one column dump feeds both tools and
results can be cross-checked; that tool reads int64 only, so the other types are
an extension this suite makes alone. `--mlidc_dataset_name` sets the name it
reports under. With the flag unset, behaviour is exactly the synthetic datasets,
so nothing changes for anyone without the file.

No dataset is committed. To benchmark a production column, dump it to that format
and point the flag at it.

## Element types

Every driver runs one element type per invocation, chosen by `--mlidc_dtype`.
The supported set is `int32`, `uint32`, `int64`, `uint64`, `float`, `double`.

That set is a ceiling, not a preference. It is exactly what
`SubIntSplitEncoding` documents and what its own typed-test suite covers:
SubIntSplit static_asserts that the physical type is 4 or 8 bytes, so the 8- and
16-bit types cannot be instantiated at all while it is in the encoder suite.

Float and double reach the encodings through their physical type
(`TypeTraits<float>::physicalType` is `uint32_t`), so what gets compressed is
the bit pattern. The drivers that hand samples to the SubIntSplit sampler view
the column as that physical type for the same reason: a bit-range analysis is
only meaningful over the bits the encoding actually splits.

`nimble_ml_id_smoke_benchmark` ignores the flag and sweeps every supported type,
which makes it the round-trip and compile-coverage gate for the suite.

## Compression axes

The suite separates two things that a "compression ratio" normally conflates.

**`--mlidc_substream_compression`** (default `Uncompressed`) applies a compressor
to each encoding's streams, *including the sub-streams of a nested encoding such
as SIS*. Accepts `Uncompressed`, `Zstd`, `Lz4`, `OpenZL`. Setting it to
`Uncompressed` isolates what the selected sub-encoders achieve on their own,
which is the measurement that separates the SIS split-and-encode decision from
entropy coding.

**`--mlidc_outer_compression`** (default `Uncompressed`) applies one compressor to
the whole encoded payload, on top of any sub-stream compression. This models
shipping an encoded column through a block codec. Entries gain a `+outer:<codec>`
suffix in their reported name.

Both flags parse names with `nimble::toCompressionType`, so a codec added to
Nimble becomes available here with no change to this directory.

### Why this needs its own code

`test::Encoder` cannot express these choices, which is why `SubstreamCompression.h`
exists:

- Its `TestCompressPolicy` handles only `Uncompressed` and `Zstd`. Under
  `DISABLE_META_INTERNAL_COMPRESSOR`, which this build defines, everything else
  falls through to Zstd level 3. Asking it for OpenZL would silently report Zstd
  numbers labelled OpenZL.
- Its policy classes are private members of `test::Encoder`, so they cannot be
  reused.
- Its nested path hard-codes `ManualEncodingSelectionPolicyFactory{..., std::nullopt}`,
  leaving sub-streams on the default compressor whatever the caller asked for.

`BenchEncodingSelectionPolicy` mirrors the test policy's encoding *choices*, so
layouts stay comparable, and differs only in routing the requested compressor
into nested selection.

Note that `compressionOptionsFor()` sets `compressionAcceptRatio = 1.0` and zeroes
the per-codec minimum sizes, so the requested compressor is always actually
applied. Production defaults (0.98, non-zero minimums) would leave some streams
uncompressed, which would read as a codec difference rather than a threshold
effect. These numbers are therefore a clean comparison, not a production
prediction.

## How OpenZL serves partial reads

OpenZL has no addressable interior. `OpenZLBenchTarget` serves a range or gather
the only way a block codec can: decompress the entire column, then copy out the
requested rows, with the decompression charged on every call and never cached.
That is the cost a reader actually pays, and it is the comparison the decode
drivers exist to make.

The same applies to any encoding wrapped by `--mlidc_outer_compression`, via the
`OuterCompressedTarget` decorator. One caveat: that decorator times the
decompression but discards the output, since the inner target still holds the
payload it encoded. A real reader would also rebuild the `Encoding` from the
decompressed bytes, so the penalty measured there is a **lower bound**.

### Block-addressable codec arms

`openzl/auto` compresses the whole column as one unit, which makes read cost a
function of column length. That is one way to deploy a block codec, not the only
one: a columnar format ships a codec in fixed-size blocks and decompresses only
the blocks a read overlaps. `BlockCodecTarget.h` is that addressable sibling, and
the two together separate the codec from the granularity it is shipped at.

`BlockCompressedTarget` splits the column into blocks of K elements, compresses
each independently, and serves a read of `count` elements at `begin` by
decompressing exactly the blocks in `[begin / K, (begin + count - 1) / K]` — that
is `floor((begin + count - 1) / K) - floor(begin / K) + 1` blocks, never more. A
point read costs one block whatever the column length is. Within one
`skipThenMaterialize` call the scratch block is reused, so two ranges landing in
the same block cost one decompression; nothing is cached across calls, matching
`OuterCompressedTarget`, which charges every call for its own decompression.

The arms are `zstd/block-K` and `openzl/block-K` for K in 1024, 65536 and 262144
elements — 8 KB, 512 KB and 2 MB at eight bytes per element. The Zstd arms need
no OpenZL and are added by every driver; the OpenZL arms follow `openzl/auto`.
`openzl/block-K` uses the same `select_numeric` graph as `openzl/auto`, applied
one block at a time, so the block size is the only difference between them.
The Zstd arms go through nimble's own compressor registry, so Zstd here is the
same Zstd the encodings use for their sub-streams. Zstd declines an
incompressible block, which a block of random IDs routinely is; those blocks are
stored verbatim and the block directory records which form each one is in.

`payloadSize()` includes the block directory, since a reader cannot address a
block without it: five bytes per block (a 32-bit start offset and a stored-form
byte), one terminating offset, and an eight-byte header of element count and
block size. At K = 1024 and eight-byte elements that is 0.061% of the raw column,
and it falls by 64x at K = 65536.

`BlockCompressedTarget` reports `ReadPath::kBlock`, so the iteration caps below
leave it alone: a read does not decompress the whole payload, and capping these
arms would hide the block-size effect they exist to measure. The one exception
is derived rather than declared — a column short enough to fit in a single block
reports `kWholePayload`, because an arm holding all of it in one block is a
whole-payload codec whatever K was set to. `zstd/whole` is that case on purpose.
`tests/BlockCodecTargetTest.cpp` pins both the round trip — including partial
final blocks, single-element columns and ranges spanning block boundaries — and
the "decompresses only what it overlaps" property itself.

### Keeping the sweeps bounded

Targets where every read decompresses everything report
`ReadPath::kWholePayload` from `NimbleBenchTargetBase::readPath()`. This used to
be a flag on `EncoderEntry`, which `--mlidc_outer_compression` made wrong: that
flag wraps every arm in a whole-payload codec without any of them declaring one,
so those runs went uncapped. The range and gather drivers sweep hundreds of grid
cells, and a full decompress per cell would dominate wall-clock time, so
`specFor()` in `MeasureLoop.h` caps those targets at
`--mlidc_block_codec_iters` (default 1) and drops warmup. Their timings are
correspondingly noisier than the rest, which is the intended trade: enough signal
to compare orders of magnitude, without the sweep taking hours.

The point driver needs a second cap. Its unit of work is a sweep of `--probes`
lookups inside one measured operation, so capping iterations alone still leaves one
full decompress per probe. `--mlidc_block_codec_probes` (default 64) bounds the
probe count for those entries. Per-probe cost is constant, so a small sample gives
the same `ns_per_probe`. The CSV records the probe count and iterations actually
used, not the nominal flag values.

Note that the sequential Nimble encodings are slow here too, for a different reason:
a probe resets the encoding and skips from position zero, so cost grows with the row
index. Expect to lower `--probes` for a run that includes RLE or SIS.

## View construction, and what it is amortised over

A view, or a decoded buffer, costs something to build and then serves reads
cheaply. Reporting only one of those two numbers is misleading in whichever
direction the arm happens to favour, so the harness reports both, always, and
from one run.

`NimbleBenchTargetBase` exposes the question directly: `readPath()` says how a
read reaches its rows, `buildsAccessStructure()` says whether there is a
one-time build at all, and `buildAccessStructure()` / `discardAccessStructure()`
let a driver time it. A driver asks the target rather than reading the arm's
name. That is not hypothetical tidiness — the point driver's
`emulated_point_read` column used to be written as a constant `1` for every arm
including the view arms it was meant to separate, and a previous analysis had to
reconstruct it from the encoding name. It now reads `0` exactly when the target
answers a one-row probe with one row's work.

Every measured driver therefore writes four extra columns: `read_path`,
`builds_access_structure`, `build_ns`, and `time_incl_build_ns`. `time_ns` stays
the read time with the structure already built, so both numbers are on the same
row and neither can be quoted without the other. This replaces the old
`SIS/...+view+ctor` arms, which covered two view arms out of a dozen and put the
two numbers on different rows.

Blackbox codecs are on the same curves. `openzl/auto` and `zstd/whole`
decompress everything per read, which is what a reader holding only the
compressed column pays, and it is why `openzl/auto` reads 7.5e-05 Mprobes/s on
snowflake. That is a property of the deployment rather than of the codec, so
`MaterializingTarget` wraps either one to decompress once on the first access
and serve the rest from the decoded buffer — the structure
`MaterializedEncodingView` already uses for a section whose encoding has no
view. The `openzl/auto+materialize` and `zstd/whole+materialize` arms are those,
and they encode to the same bytes as their bare twins, so the pair differs in
how a read is addressed and in nothing else.

## Resident memory as an axis

Time alone made the amortisation curves compare unlike things, and the
comparison flattered whichever arm was willing to hold the most memory.
`+materialize` decompresses the **entire** column and answers reads by indexing
a flat array. A view does not: it materialises only the sections that lack a
real view and serves the rest from the compressed representation. Both answer
the same reads, so on a time-only chart `openzl/auto+materialize` looked
unanswerable — 13.7 ms build and 35.9 ns/op on snowflake point against
`SIS/realNested+view`'s 71.7 ms and 765 ns/op — while holding roughly 8 MB of
uncompressed column to do it, against about 5 MB compressed. That is a cache
result reported as a compression result.

Decompressing both in full is not the fix; that is bulk decode, which
`nimble_ml_id_decode_bulk_benchmark` already measures. The fix is to let each
system materialise lazily at its own natural granularity and to put what it
holds on an axis of its own. Every measured driver therefore writes a
`resident_bytes` column beside `read_path` and `build_ns`.

`NimbleBenchTargetBase::residentBytes()` is **pure**, for the same reason
`readPath()` is: a target that quietly held a decoded copy of the column would
otherwise be compared on time alone against one holding only compressed bytes,
and the two would look like the same deployment. Each target answers for what
it actually keeps — the cursor and view targets report their encoded bytes plus
their own memory pool, `MaterializingTarget` adds the decoded buffer,
`BlockCompressedTarget` reports payload plus directory plus scratch,
`OpenZLBenchTarget` its frame plus scratch, and `OuterCompressedTarget` sums
the compressed bytes, the inner target and the last decompressed buffer.

Two details make those numbers honest rather than nominal.

**Per-target memory pools.** Each target takes its own
`memoryManager()->addLeafPool("mlidc_target_N")` via `makeTargetPool()` instead
of sharing `benchmarks::benchmarkPool()`. (That pool is itself a leaf, so it
cannot have children — a target sharing it would report the whole sweep's
allocations as its own.) This is what makes a view's index structures genuinely
measured rather than estimated from bits per element, since they and the buffer
a `MaterializedEncodingView` decodes a viewless section into are both
pool-backed.

**Decode caches that no pool can see.** A transformed SubIntSplit stream keeps
a decoded block across `reset()`, in plain `std::vector`s rather than
pool-backed `Vector<T>`s, so neither `payloadSize()` nor the pool reports it.
Worse, the block size is only ever assigned under a condition no existing
transform satisfies, so the "block" is the whole column: the first probe
decodes every row and every probe after it copies out of that cache. Measured
on a 1M-row column that is roughly 8 MB held against a 4.98 MB payload, and it
is why such an arm reads flat in N while its untransformed twin scales
linearly. `resident_bytes` does not count it.

The same finding gives those arms a build cost the harness used to attribute
nowhere: a whole-column decode hidden entirely by warmup. `Encoding` therefore
also answers `retainsDecodeCache()` and `dropDecodeCache()`, and the cursor
target reports `buildsAccessStructure()` true exactly when the encoding retains
a cache, builds it by forcing that first read, and discards it for real. A
transformed arm now appears on the amortisation curves as build-plus-residency,
next to the materialised blackbox arms it actually resembles, rather than as a
per-probe constant that is really a memcpy.

`resident_bytes` is sampled in `setAccessColumns()`, which every driver calls
while writing the row — that is **after** that row's reads, not after its
build. The ordering is deliberate: an arm that materialises lazily has no final
footprint until a workload has touched it, so sampling at construction would
report every lazy arm at its compressed size and erase the axis.

### Block-lazy arms

`BlockLazyTarget` is the middle regime between holding a column compressed and
holding it decoded, and it is the one a reader actually deploys: materialise at
the granularity the format is addressable at, and let the workload decide how
much ends up resident. It keeps the blocks it has decoded, indexed by block
number in a vector rather than a hash map, so a hit is an array index and the
bookkeeping stays out of the measured loop.

It wraps the **block** arms rather than `openzl/auto`, and that is not
incidental: with a whole-payload inner, decoding "one block" would decompress
the entire column and the laziness would be fictitious. The arms are
`zstd/block-65536+lazy` and `openzl/block-65536+lazy`, composed by
`withBlockLazyMaterialization` exactly as `withMaterializedAccess` composes the
full-materialise arms, so each lazy arm encodes to the same bytes as its bare
twin and differs only in what a read leaves behind.

It reports `ReadPath::kBlock` and `buildsAccessStructure() == false` with
`build_ns` of zero, because its cost genuinely is not a one-time build: it is
spread across whichever reads happen to miss, and depends on which rows a
workload touches rather than on the column. The story is told by
`resident_bytes` rising along the ops axis. `materializeAll` bypasses the cache
so that a bulk scan cannot turn the arm into full materialisation by a side
door.

Keep all three regimes when reading a chart. Compressed-only, block-lazy and
fully-materialised together are what make the frontier legible; any one of them
alone reproduces the original error in a different direction.
`tests/BlockLazyTargetTest.cpp` pins the behaviour rather than the timing: that
k scattered probes decode `min(k, blocks)` blocks and never the column, that
repeated probes in one block decode it once, that `resident_bytes` rises with
the reads and returns on discard, and that `materializeAll` leaves nothing
cached.

## Running

```bash
cmake -S . -B _build_release -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DVELOX_ENABLE_NIMBLE=ON -DNIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS=ON \
  -DVELOX_BUILD_TESTING=OFF
ninja -C _build_release nimble_ml_id_smoke_benchmark

BIN=_build_release/velox/dwio/nimble/encodings/benchmarks/ml_id_compression
$BIN/nimble_ml_id_smoke_benchmark --mlidc_file=/path/to/column.txt --mlidc_rows=10000000
```

Run correctness before quoting any number, and use a **Release** build: a Debug
build's correctness output is trustworthy but its timings are not.

Drivers write their CSV and JSON manifest into the current working directory
unless `--mlidc_output_csv` and `--mlidc_output_manifest` say otherwise, so run
them from a scratch directory.

### Shared flags

| Flag | Default | Meaning |
|---|---|---|
| `--mlidc_dtype` | int64 | Element type: `int32`, `uint32`, `int64`, `uint64`, `float`, `double` |
| `--mlidc_rows` | 100000 | Rows per dataset |
| `--mlidc_iters` | 5 | Iterations per (encoder, dataset) cell |
| `--mlidc_seed` | 42 | Seed for the synthetic generators |
| `--mlidc_file` | "" | Text file, one value per line, parsed as `--mlidc_dtype`, added as a dataset |
| `--mlidc_dataset_name` | twitter-snowflake | Name for that dataset |
| `--mlidc_substream_compression` | Uncompressed | Per-stream codec, sub-streams included |
| `--mlidc_outer_compression` | Uncompressed | Whole-payload codec |
| `--mlidc_block_codec_iters` | 1 | Iteration cap for whole-payload codecs |
| `--mlidc_block_codec_probes` | 64 | Point-probe cap for whole-payload codecs |
| `--mlidc_datasets` | "" | Comma-separated dataset names to run; empty runs all |
| `--mlidc_output_csv` | per-driver | CSV output path |
| `--mlidc_output_manifest` | mlidc_manifest.json | Run manifest path |

## Where the shared code lives

`BenchCommon.h` holds the bench targets, the encoder and dataset suites, per-target
memory pools (`makeTargetPool`) and outer compression. `BlockCodecTarget.h` holds
the block-addressable codec target and its Zstd arms, plus `BlockLazyTarget` and
the `withBlockLazyMaterialization` decorator, which live there rather than in a
file of their own because they are meaningful only over a block-addressable
inner; the OpenZL codec and arms sit in `OpenZLBenchTarget.h`
alongside the graph they share with `openzl/auto`. `ResultWriter.h` holds the CSV writer and the run manifest.
`SubstreamCompression.h` holds the encode path described above. `ElemType.h`
holds the element-type vocabulary: parsing `--mlidc_dtype`, the name reported in
the `dtype` column, and the dispatch that turns the runtime choice into the
static type each driver body is templated on.

`DriverSweep.h` holds the scaffolding every sweep driver repeats: building the
encoder and dataset suites (`makeSweepContext`), encoding one dataset with skip
handling (`makeTargetOrSkip`), preparing the cache for one measurement cell
(`makeCellCache`), and the CSV setters for the columns every driver writes.

Each driver keeps its own measurement call and its own CSV columns inline. That
is the part worth reading when opening a driver, so it deliberately did not move
into the shared header.

Each driver body is a `runBenchmark<Elem>` template; `main` parses
`--mlidc_dtype` and dispatches into it. The `dtype` CSV column is set in
`DriverSweep.h` alongside the other identity columns, so every sweep driver
reports it without repeating the call.

Two drivers do not use all of it. `MlIdEncodeBenchmark.cpp` times the encode
itself, so the factory call sits inside its measurement lambda and cannot use
`makeTargetOrSkip`. `MlIdCompressionBenchmark.cpp` and `MlIdEncodeBenchmark.cpp`
do no cache sweep, so they build their context with a fixed hot cache state.

## Interpreting results

Two regimes are worth knowing before reading a table.

At `--mlidc_substream_compression=OpenZL` the codec dominates the ratio and the
encoding choice nearly stops mattering for size. SIS's case there rests on access
performance, not bytes.

In gather, SIS's advantage over OpenZL grows with run length and shrinks as range
count rises. At very high range counts with run length 1, per-range skip overhead
can make SIS slower than decompressing the whole column. Report the grid, not a
single cell.

Scattered point lookup is SIS's weakest pattern. Without a positional index a probe
costs a traversal from position zero, which puts it orders of magnitude behind
Trivial, FixedBitWidth and the indexed FrequencyPartition variants. It still beats a
block codec, which decompresses everything per probe, but that is a low bar.
