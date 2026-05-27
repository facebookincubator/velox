/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/dwio/parquet/reader/Metadata.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

namespace facebook::velox::parquet {

namespace {

// Heap-allocated payload sizes representative of real footers: long
// enough to defeat SSO so the walk pays for the heap dereferences,
// short enough to keep the working set bounded.
constexpr size_t kPathSegmentLen = 24;
constexpr size_t kStatsValueLen = 32;

thrift::ColumnChunk makeColumn(size_t pathDepth) {
  thrift::ColumnChunk column;
  std::vector<std::string> path;
  path.reserve(pathDepth);
  for (size_t i = 0; i < pathDepth; ++i) {
    path.emplace_back(kPathSegmentLen, 'p');
  }
  column.meta_data.path_in_schema = std::move(path);
  column.meta_data.encodings = {
      thrift::Encoding::PLAIN,
      thrift::Encoding::RLE,
      thrift::Encoding::RLE_DICTIONARY,
  };
  column.meta_data.__isset.statistics = true;
  column.meta_data.statistics.__set_min(std::string(kStatsValueLen, 'm'));
  column.meta_data.statistics.__set_max(std::string(kStatsValueLen, 'M'));
  column.meta_data.statistics.__set_min_value(std::string(kStatsValueLen, 'a'));
  column.meta_data.statistics.__set_max_value(std::string(kStatsValueLen, 'z'));
  column.meta_data.num_values = 1'000'000;
  column.meta_data.total_compressed_size = 1 << 20;
  column.meta_data.total_uncompressed_size = 2 << 20;
  return column;
}

thrift::FileMetaData makeFooter(size_t numColumns, size_t numRowGroups) {
  thrift::FileMetaData metadata;
  metadata.created_by = "velox-benchmark";
  metadata.num_rows = static_cast<int64_t>(numColumns) * numRowGroups;
  metadata.schema.resize(numColumns + 1);
  metadata.schema[0].name = "root";
  metadata.schema[0].num_children = static_cast<int32_t>(numColumns);
  for (size_t i = 0; i < numColumns; ++i) {
    metadata.schema[i + 1].name = "col_" + std::to_string(i);
  }
  metadata.row_groups.resize(numRowGroups);
  for (auto& rowGroup : metadata.row_groups) {
    rowGroup.columns.reserve(numColumns);
    for (size_t i = 0; i < numColumns; ++i) {
      rowGroup.columns.push_back(makeColumn(/*pathDepth=*/2));
    }
  }
  return metadata;
}

void runEstimate(size_t numColumns, size_t numRowGroups) {
  folly::BenchmarkSuspender suspender;
  auto footer = makeFooter(numColumns, numRowGroups);
  FileMetaDataPtr ptr(&footer);
  suspender.dismiss();
  auto bytes = ptr.estimateFileMetadataSize();
  folly::doNotOptimizeAway(bytes);
}

} // namespace

// Mirrors the per-call cost of estimateFileMetadataSize() across
// realistic footer shapes:
//   small  - 10 columns x  4 row groups (typical narrow file)
//   medium - 100 columns x 50 row groups
//   wide   - 6879 columns x 147 row groups (the OOM pathology)
BENCHMARK(estimateSmall) {
  runEstimate(10, 4);
}

BENCHMARK(estimateMedium) {
  runEstimate(100, 50);
}

BENCHMARK(estimateWide) {
  runEstimate(6879, 147);
}

} // namespace facebook::velox::parquet

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  folly::runBenchmarks();
  return 0;
}
