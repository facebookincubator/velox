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
DEFINE_int32(mlidc_iters, 5, "Benchmark iterations per (encoder, dataset) pair");
DEFINE_int64(mlidc_seed, 42, "Base random seed for dataset generators");
DEFINE_string(
    mlidc_file,
    "",
    "Text file with one int64 per line, added as a real-data dataset alongside "
    "the synthetic ones. Empty disables. Same format as the --file flag of "
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
DEFINE_int32(
    mlidc_block_codec_iters,
    1,
    "Iterations used for encoders where every read decompresses the whole "
    "payload (OpenZL, or any encoding under --mlidc_outer_compression). The "
    "fine-grained range and gather sweeps run hundreds of cells, and a full "
    "decompress per cell would otherwise dominate wall-clock time. Timings for "
    "these entries are correspondingly noisier.");
DEFINE_string(
    mlidc_dataset_name,
    "twitter-snowflake",
    "Name reported for the --mlidc_file dataset");
