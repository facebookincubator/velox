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

### Keeping the sweeps bounded

Entries where every read decompresses everything are marked
`EncoderEntry::wholePayloadCodec`. The range and gather drivers sweep hundreds of
grid cells, and a full decompress per cell would dominate wall-clock time, so
`specFor()` in `MeasureLoop.h` caps those entries at
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

`BenchCommon.h` holds the bench targets, the encoder and dataset suites, and
outer compression. `ResultWriter.h` holds the CSV writer and the run manifest.
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
