/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Defines the gflags shared across all ML ID compression benchmark drivers.
// Declared in BenchCommon.h; defined here in exactly one TU.

#include <gflags/gflags.h>

DEFINE_string(mlidc_output_csv, "mlidc_results.csv", "CSV output path");
DEFINE_string(
    mlidc_output_manifest,
    "mlidc_manifest.json",
    "JSON manifest sidecar path");
DEFINE_int32(mlidc_rows, 100000, "Number of rows per dataset instance");
DEFINE_int32(
    mlidc_chunk_rows,
    0,
    "Encode each column as independently encoded chunks of this many rows, as "
    "a chunking Nimble writer flushes a long stream. 0 encodes the column as "
    "one block. The writer's maximum chunk is 20 MiB raw "
    "(kChunkingWriterMaxChunkSize), 2,621,440 rows of a 64-bit type.");
DEFINE_int32(
    mlidc_iters,
    5,
    "Benchmark iterations per (encoder, dataset) pair");
DEFINE_int64(mlidc_seed, 42, "Base random seed for dataset generators");
DEFINE_string(
    mlidc_file,
    "",
    "Text file with one value per line, parsed as --mlidc_dtype, added as a "
    "real-data dataset alongside the synthetic ones. Empty disables. The int64 "
    "case is the same format as the --file flag of "
    "velox/dwio/nimble/tools/encoding_bench, so a column dump feeds both.");
DEFINE_string(
    mlidc_substream_compression,
    "Uncompressed",
    "Compressor applied to each encoding's streams, including the sub-streams "
    "of nested encodings such as SubIntSplit. One of Uncompressed, Zstd, Lz4, "
    "OpenZL. 'Uncompressed' isolates what the selected sub-encoders achieve on "
    "their own.");
DEFINE_string(
    mlidc_outer_compression,
    "Uncompressed",
    "Compressor applied once to the whole encoded payload, on top of any "
    "sub-stream compression. Same names as --mlidc_substream_compression. "
    "Models shipping an encoded column through a block codec, which costs a "
    "full decompress before any read.");
DEFINE_int32(
    mlidc_block_codec_probes,
    64,
    "Point-lookup probes used for encoders where every read decompresses the "
    "whole payload. Each probe costs one full decompress, so the default 65536 "
    "probes would take hours. Per-probe cost is constant, so a small sample "
    "gives the same ns_per_probe.");
DEFINE_bool(
    mlidc_allow_delta_block,
    false,
    "Whether SubIntSplit may cost and select DeltaBlock. False is what "
    "production ships (Encoding::Options::subIntSplitAllowDeltaBlock); pass "
    "true to measure the withdrawn configuration on any driver. It is one flag "
    "rather than a second arm because crossing it with the Huffman arms would "
    "double every SubIntSplit target for a comparison that is run once.");
DEFINE_double(
    mlidc_sis_decode_weight,
    0.0,
    "Encoding::Options::subIntSplitDecodeWeight: how much a SubIntSplit "
    "section's predicted decode cost counts against its encoded size when the "
    "split planner chooses boundaries and encodings. 0.0 is what production "
    "ships and is size-only selection, bit for bit; raise it to price decode "
    "and see which sections change encoding.");
DEFINE_int32(
    mlidc_sis_decode_access_pattern,
    0,
    "Encoding::Options::subIntSplitDecodeAccessPattern: the read shape decode "
    "is costed for when --mlidc_sis_decode_weight is non-zero. 0 bulk, "
    "1 point, 2 gather, 3 range.");
DEFINE_int32(
    mlidc_sis_decode_read_path,
    0,
    "Encoding::Options::subIntSplitDecodeReadPath: the reader decode is costed "
    "for when --mlidc_sis_decode_weight is non-zero. 0 cursor, 1 view, both "
    "with construction amortised; 2 cursor and 3 view paying construction.");
DEFINE_int32(
    mlidc_sis_admission,
    0,
    "Encoding::Options::subIntSplitAdmission: how selection decides whether "
    "SubIntSplit competes. 0 the size estimate, 1 the bit-flip profile's "
    "gradient guard, 2 the gradient guard plus the entropy guard.");
DEFINE_bool(
    mlidc_sis_admission_forces,
    false,
    "Encoding::Options::subIntSplitAdmissionForces: whether a bit-flip "
    "admission selects SubIntSplit outright instead of only letting it "
    "compete on size. Reproduces the pre-candidacy behaviour for the "
    "ablation. Inert at --mlidc_sis_admission=0.");
DEFINE_bool(
    mlidc_sis_row_frame,
    true,
    "Encoding::Options::subIntSplitRowFrame: whether SubIntSplit may subtract "
    "a fitted line or step from every value before planning its sections.");
DEFINE_string(
    mlidc_sis_upstream_features,
    "",
    "Comma-separated overrides of the upstream SubIntSplit switches; empty "
    "keeps the library defaults. Each name sets its switch on, and the same "
    "name prefixed with no- sets it off. Names, with their library default: "
    "delta (subIntSplitDeltaPreTransform, off), trim "
    "(subIntSplitTrimConstantPlanes, off), prune "
    "(subIntSplitBoundaryPruneThreshold at upstream's 0.001; off means 0.0), "
    "fold (subIntSplitFoldConstantSections, on), passthrough "
    "(subIntSplitPassThrough, on), visitorblock (subIntSplitVisitorBlockBuffer, "
    "off), huffmandeep (huffmanPriceLengthLimited, on). Example: "
    "--mlidc_sis_upstream_features=no-fold,no-passthrough,no-huffmandeep "
    "reproduces the pre-integration behaviour.");
DEFINE_bool(
    mlidc_sis_estimate_compression_guard,
    true,
    "Encoding::Options::subIntSplitEstimateCompressionGuard: whether the size "
    "estimate withholds the row frame's credit when the streams will be "
    "handed to a substream compressor.");
DEFINE_bool(
    mlidc_sis_estimate_bitflip_screen,
    true,
    "Encoding::Options::subIntSplitEstimateBitFlipScreen: whether selection "
    "may rule SubIntSplit out from the bit-flip gradient gate rather than by "
    "planning a split.");
DEFINE_double(
    mlidc_sis_max_size_regression,
    0.05,
    "Encoding::Options::subIntSplitMaxSizeRegression: the most encoded size, "
    "as a fraction, that decode weighting may give up against what size-only "
    "selection would have chosen. Inert at --mlidc_sis_decode_weight=0, where "
    "the two plans are the same plan. Pass something large to reproduce the "
    "unbounded objective, which pays an uncompressed column for speed.");
DEFINE_bool(
    mlidc_dump_encoding,
    false,
    "Print the encoding tree each encoder selected, including the bit ranges "
    "SubIntSplit split into and the encoding chosen for each section.");
DEFINE_string(
    mlidc_datasets,
    "",
    "Comma-separated dataset names to run, e.g. twitter-snowflake. Empty runs "
    "every dataset. Lets a production column be benchmarked without paying for "
    "the synthetic sweep.");
DEFINE_string(
    mlidc_encoders,
    "",
    "Comma-separated encoder names to run, matched against the same name "
    "emitted to the CSV's encoding column, e.g. SIS/key_derived+view. Empty "
    "runs every encoder. Lets one arm's decode be profiled without the "
    "encode and decode cost of the other thirty in the same run.");
DEFINE_int32(
    mlidc_block_codec_iters,
    1,
    "Iterations used for encoders where every read decompresses the whole "
    "payload (OpenZL, or any encoding under --mlidc_outer_compression). The "
    "fine-grained range and gather sweeps run hundreds of cells, and a full "
    "decompress per cell would otherwise dominate wall-clock time. Timings for "
    "these entries are correspondingly noisier.");
DEFINE_string(
    mlidc_input_order,
    "shipped",
    "Order the real-data column is presented in, before any encoding: shipped "
    "(as the file stores it), shuffled, sorted, mergeirr=k (k monotone runs "
    "interleaved at irregular rates, with nothing in the data saying which run "
    "a row came from), or mergekey=s (partitioned by section s, each partition "
    "ordered, interleaved irregularly -- the multi-writer case a Snowflake id "
    "actually arrives in). A file's own order is usually the arrival order "
    "already sorted, which is the input a reordering transform has least to do "
    "on, so measuring only that understates the layer.");
DEFINE_string(
    mlidc_encode_cache_dir,
    "",
    "Directory holding cached encoded payloads. Empty disables the cache, "
    "which is the default: a sweep opts in, nothing is cached behind anyone's "
    "back. Every driver otherwise re-encodes the same targets, so a sweep pays "
    "each encode once per driver.");
DEFINE_string(
    mlidc_dataset_name,
    "twitter-snowflake",
    "Name reported for the --mlidc_file dataset");
DEFINE_string(
    mlidc_dtype,
    "int64",
    "Element type to benchmark: int32, uint32, int64, uint64, float, double. "
    "The 8- and 16-bit types are excluded because SubIntSplitEncoding only "
    "supports 32- and 64-bit types. With --mlidc_file, the column is parsed as "
    "this type.");

DEFINE_bool(
    mlidc_sis_withdraw_frequency_partition,
    false,
    "Whether SubIntSplit's split planner may cost a bit range as "
    "FrequencyPartition. False is what production ships. Pass true to measure "
    "what withdrawing it costs in bytes and buys in decode: FrequencyPartition "
    "stores a low-cardinality section well and reads it slowly, and a size-only "
    "objective cannot see the second half of that.");

DEFINE_uint32(
    mlidc_selection_screen_rows,
    0,
    "Sets Encoding::Options::selectionScreenRows on every Nimble target: "
    "selection prices its costly candidates on this many sampled rows first "
    "and prices them on the whole stream only where the sample does not rule "
    "them out. Zero, the default, is production behaviour.");

DEFINE_double(
    mlidc_selection_screen_margin,
    1.25,
    "Sets Encoding::Options::selectionScreenMargin. Only read when "
    "--mlidc_selection_screen_rows is set.");

DEFINE_uint32(
    mlidc_sis_section_threads,
    4,
    "Threads SubIntSplit encodes its sections on, through "
    "Encoding::Options::subIntSplitSectionExecutor. Four by default; zero "
    "encodes them on the calling thread. Encode wall time is then not "
    "comparable with single-threaded codecs: compare the reported CPU time.");

DEFINE_string(
    mlidc_sis_withdraw_nested_encodings,
    "",
    "Comma-separated encoding names withdrawn from the candidate list a "
    "SubIntSplit section's encoding is chosen from, e.g. FrequencyPartition. "
    "Empty, the default, is production behaviour. This is the list that "
    "decides a section's encoding; Encoding::Options::"
    "subIntSplitAllowedEncodings gates only the split planner's cost models, "
    "so withdrawing an encoding there alone leaves the section encoded as "
    "before. Matching is by substring, so Delta withdraws DeltaBlock too.");
