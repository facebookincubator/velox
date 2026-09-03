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

#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/caching/FileHandle.h"
#include "velox/common/config/Config.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

class MetadataOnlySplitReader final : public CudfSplitReader {
 public:
  using CudfSplitReader::CudfSplitReader;

  cudf::ast::expression const* logicalFilter() const {
    return subfieldFilter();
  }

  cudf::ast::expression const* splitFilter() const {
    return pushdownFilter();
  }

  bool hasSplitFilter() const {
    return hasSplitSpecificPushdownFilter();
  }

 protected:
  void prepareSplitInternal(
      dwio::common::RuntimeStats& /*runtimeStats*/) override {
    fileMetaDatas();
    // Metadata caching must not rebuild the filter during one preparation.
    fileMetaDatas();
  }
};

class CudfSplitReaderTest : public ::facebook::velox::cudf_velox::exec::test::
                                CudfHiveConnectorTestBase {};

TEST_F(CudfSplitReaderTest, buildsPushdownFilterForEachSplitPreparation) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})}));

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  ::facebook::velox::connector::ConnectorQueryCtx connectorQueryCtx(
      pool_.get(),
      pool_.get(),
      properties.get(),
      nullptr,
      common::PrefixSortConfig{},
      nullptr,
      nullptr,
      "query.CudfSplitReaderTest",
      "task.CudfSplitReaderTest",
      "plan.CudfSplitReaderTest",
      0,
      "");
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto split =
      CudfHiveConnectorSplitBuilder(dataFile->getPath())
          .connectorId(
              ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
          .build();

  cudf::ast::column_reference logicalFilter{0};
  cudf::ast::column_reference firstSplitFilter{0};
  cudf::ast::column_reference secondSplitFilter{0};
  MetadataOnlySplitReader reader(
      std::move(split),
      ::facebook::velox::cudf_velox::exec::test::CudfHiveConnectorTestBase::
          makeTableHandle("parquet_table", rowType),
      rowType,
      {"c0"},
      &fileHandleFactory,
      ioExecutor_.get(),
      &connectorQueryCtx,
      std::make_shared<CudfHiveConfig>(properties),
      std::make_shared<io::IoStatistics>(),
      std::make_shared<IoStats>(),
      false,
      &logicalFilter);

  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &logicalFilter);
  EXPECT_FALSE(reader.hasSplitFilter());

  size_t builderCalls = 0;
  std::vector<size_t> schemaSizes;
  reader.setPushdownFilterBuilder(
      [&](const cudf::io::parquet::FileMetaData& metadata) {
        schemaSizes.push_back(metadata.schema.size());
        return builderCalls++ == 0
            ? static_cast<cudf::ast::expression const*>(&firstSplitFilter)
            : static_cast<cudf::ast::expression const*>(&secondSplitFilter);
      });

  // Installing a builder does not change the filter until split metadata is
  // available.
  EXPECT_EQ(reader.splitFilter(), &logicalFilter);
  EXPECT_FALSE(reader.hasSplitFilter());

  dwio::common::RuntimeStats runtimeStats;
  reader.prepareSplit(runtimeStats);
  EXPECT_EQ(builderCalls, 1);
  ASSERT_EQ(schemaSizes.size(), 1);
  EXPECT_GT(schemaSizes.front(), 1);
  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &firstSplitFilter);
  EXPECT_TRUE(reader.hasSplitFilter());

  // Preparing again resets the previous split filter and rebuilds it from the
  // footer without replacing the logical filter.
  reader.prepareSplit(runtimeStats);
  EXPECT_EQ(builderCalls, 2);
  ASSERT_EQ(schemaSizes.size(), 2);
  EXPECT_GT(schemaSizes.back(), 1);
  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &secondSplitFilter);
  EXPECT_TRUE(reader.hasSplitFilter());
  EXPECT_EQ(runtimeStats.processedSplits, 2);
}

TEST_F(CudfSplitReaderTest, pinnedRangeCacheAssemblesOverlaps) {
  auto input =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(100, folly::identity)});
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, input->pool(), stream, mr);
  auto ranges = cudf::slice(cudfTable->view(), {0, 50, 50, 100}, stream);
  ASSERT_EQ(ranges.size(), 2);

  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  CudfDecodedColumnCache::ColumnKey key{
      .file = {.connectorId = "test", .filePath = "overlapping-ranges"},
      .deviceId = deviceId,
      .columnName = "c0",
      .veloxType = BIGINT()->toString(),
      .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
      .usePandasMetadata = true,
      .useArrowSchema = true,
      .allowMismatchedSchemas = false,
  };

  auto& cache = CudfDecodedColumnCache::instance();
  EXPECT_EQ(cache.pinnedBytes(), 0);
  EXPECT_EQ(CudfDecodedColumnCache::kMaxPinnedBytes, 70ULL << 30);
  ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
      key, 0, 50, ranges[0].column(0), stream, mr));
  ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
      key, 50, 100, ranges[1].column(0), stream, mr));
  EXPECT_GT(cache.pinnedBytes(), 0);
  EXPECT_LE(cache.pinnedBytes(), CudfDecodedColumnCache::kMaxPinnedBytes);

  auto coverage = cache.findColumnRanges(key, 25, 75);
  ASSERT_TRUE(coverage.has_value());
  ASSERT_EQ(coverage->size(), 2);
  EXPECT_EQ(coverage->at(0).firstRow, 25);
  EXPECT_EQ(coverage->at(0).lastRow, 50);
  EXPECT_EQ(coverage->at(1).firstRow, 50);
  EXPECT_EQ(coverage->at(1).lastRow, 75);
  for (const auto& range : *coverage) {
    cudaPointerAttributes attributes{};
    CUDF_CUDA_TRY(
        cudaPointerGetAttributes(&attributes, range.chunk->pinnedData()));
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
  }

  auto assembled = cache.materializeColumnRange(key, 25, 75, stream, mr, mr);
  ASSERT_NE(assembled, nullptr);
  ASSERT_EQ(assembled->size(), 50);
  std::vector<int64_t> actual(assembled->size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      actual.data(),
      assembled->view().data<int64_t>(),
      actual.size() * sizeof(int64_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], i + 25);
  }

  EXPECT_EQ(
      cache.materializeColumnRange(key, 25, 125, stream, mr, mr), nullptr);
  const auto pinnedBytes = cache.pinnedBytes();
  EXPECT_FALSE(cache.insertColumnRangeIfAbsent(
      key, 0, 50, ranges[0].column(0), stream, mr));
  EXPECT_EQ(cache.pinnedBytes(), pinnedBytes);
}

TEST_F(CudfSplitReaderTest, batchesDecodedColumnsAcrossFileRowGroups) {
  constexpr cudf::size_type kRowsPerRowGroup = 4;
  constexpr cudf::size_type kNumRowGroups = 3;
  auto fileRowType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});
  auto input = makeRowVector(
      {"c0", "c1", "c2"},
      {makeFlatVector<int64_t>(12, folly::identity),
       makeFlatVector<int64_t>(12, [](auto row) { return 100 + row; }),
       makeFlatVector<int64_t>(12, [](auto row) { return 200 + row; })});
  auto dataFile = common::testutil::TempFilePath::create();

  auto stream = cudf::get_default_stream();
  auto cudfTable = with_arrow::toCudfTable(
      input, input->pool(), stream, cudf::get_current_device_resource_ref());
  cudf::io::table_input_metadata metadata(cudfTable->view());
  for (size_t i = 0; i < fileRowType->size(); ++i) {
    metadata.column_metadata[i].set_name(fileRowType->nameOf(i));
  }
  auto writerOptions =
      cudf::io::parquet_writer_options::builder(
          cudf::io::sink_info{dataFile->getPath()}, cudfTable->view())
          .metadata(std::move(metadata))
          .row_group_size_rows(kRowsPerRowGroup)
          .max_page_size_rows(kRowsPerRowGroup)
          .max_page_fragment_size(kRowsPerRowGroup)
          .build();
  cudf::io::write_parquet(writerOptions, stream);
  stream.synchronize();

  auto sources =
      cudf::io::make_datasources(cudf::io::source_info{dataFile->getPath()});
  auto fileMetadata = cudf::io::read_parquet_footers(sources);
  ASSERT_EQ(fileMetadata.size(), 1);
  ASSERT_EQ(fileMetadata.front().row_groups.size(), kNumRowGroups);
  auto rowGroupOffset = [](const cudf::io::parquet::RowGroup& rowGroup) {
    if (rowGroup.file_offset.has_value()) {
      return static_cast<uint64_t>(rowGroup.file_offset.value());
    }
    if (rowGroup.columns.front().file_offset != 0) {
      return static_cast<uint64_t>(rowGroup.columns.front().file_offset);
    }
    const auto& columnMetadata = rowGroup.columns.front().meta_data;
    return static_cast<uint64_t>(
        columnMetadata.dictionary_page_offset != 0
            ? std::min(
                  columnMetadata.dictionary_page_offset,
                  columnMetadata.data_page_offset)
            : columnMetadata.data_page_offset);
  };
  const auto middleRowGroupOffset =
      rowGroupOffset(fileMetadata.front().row_groups[1]);
  sources.clear();
  fileMetadata.clear();

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kUseExperimentalCudfReader, "true"},
          {CudfHiveConfig::kExperimentalDecodedColumnCacheEnabled, "true"},
          {CudfHiveConfig::kImmutableFiles, "true"},
      });
  ::facebook::velox::connector::ConnectorQueryCtx connectorQueryCtx(
      pool_.get(),
      pool_.get(),
      properties.get(),
      nullptr,
      common::PrefixSortConfig{},
      nullptr,
      nullptr,
      "query.CudfSplitReaderTest",
      "task.CudfSplitReaderTest",
      "plan.CudfSplitReaderTest",
      0,
      "");
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto tableHandle =
      CudfHiveConnectorTestBase::makeTableHandle("parquet_table", fileRowType);

  struct ReadResult {
    uint64_t hits;
    uint64_t misses;
    uint64_t decodeCalls;
    size_t chunks;
    size_t rows;
    bool fullyCached;
  };
  auto read = [&](std::vector<std::string> names,
                  RowTypePtr outputType,
                  uint64_t start,
                  uint64_t length) {
    auto split =
        CudfHiveConnectorSplitBuilder(dataFile->getPath())
            .connectorId(
                ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
            .start(start)
            .length(length)
            .build();
    CudfSplitReader reader(
        std::move(split),
        tableHandle,
        outputType,
        names,
        &fileHandleFactory,
        ioExecutor_.get(),
        &connectorQueryCtx,
        std::make_shared<CudfHiveConfig>(properties),
        std::make_shared<io::IoStatistics>(),
        std::make_shared<IoStats>(),
        true,
        nullptr);
    dwio::common::RuntimeStats runtimeStats;
    reader.prepareSplit(runtimeStats);
    const auto fullyCached = reader.isFullyDecodedColumnCacheHit();
    size_t chunks = 0;
    size_t rows = 0;
    while (auto chunk = reader.next(0)) {
      ++chunks;
      rows += chunk.value()->num_rows();
      reader.stream().synchronize();
    }
    return ReadResult{
        reader.decodedColumnCacheHits(),
        reader.decodedColumnCacheMisses(),
        reader.decodedColumnCacheDecodeCalls(),
        chunks,
        rows,
        fullyCached};
  };

  // Warm c0 and c1 together across all three row groups. The cold read must
  // issue one multi-column, multi-row-group cuDF decode.
  auto first = read(
      {"c0", "c1"},
      ROW({"c0", "c1"}, {BIGINT(), BIGINT()}),
      0,
      std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(first.hits, 0);
  EXPECT_EQ(first.misses, 2);
  EXPECT_EQ(first.decodeCalls, 1);
  EXPECT_EQ(first.chunks, 1);
  EXPECT_EQ(first.rows, kRowsPerRowGroup * kNumRowGroups);
  EXPECT_FALSE(first.fullyCached);

  // Select only the middle row group and overlap on c1. c1 hits while c2 is
  // decoded and cached, demonstrating independent column and row-group reuse.
  auto second = read(
      {"c1", "c2"},
      ROW({"c1", "c2"}, {BIGINT(), BIGINT()}),
      middleRowGroupOffset,
      1);
  EXPECT_EQ(second.hits, 1);
  EXPECT_EQ(second.misses, 1);
  EXPECT_EQ(second.decodeCalls, 1);
  EXPECT_EQ(second.chunks, 1);
  EXPECT_EQ(second.rows, kRowsPerRowGroup);
  EXPECT_FALSE(second.fullyCached);

  // A full hit must not open the file for its footer or column data.
  ASSERT_TRUE(std::filesystem::remove(dataFile->getPath()));
  auto third = read(
      {"c1", "c2"},
      ROW({"c1", "c2"}, {BIGINT(), BIGINT()}),
      middleRowGroupOffset,
      1);
  EXPECT_EQ(third.hits, 2);
  EXPECT_EQ(third.misses, 0);
  EXPECT_EQ(third.decodeCalls, 0);
  EXPECT_EQ(third.chunks, 1);
  EXPECT_EQ(third.rows, kRowsPerRowGroup);
  EXPECT_TRUE(third.fullyCached);
}

TEST_F(CudfSplitReaderTest, cachesStatsPrunedNonContiguousRowGroups) {
  constexpr cudf::size_type kRowsPerRowGroup = 4;
  auto fileRowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto input = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>(
           12,
           [](auto row) {
             if (row < 4) {
               return static_cast<int64_t>(row);
             }
             if (row < 8) {
               return static_cast<int64_t>(100 + row);
             }
             return static_cast<int64_t>(row - 4);
           }),
       makeFlatVector<int64_t>(
           12, [](auto row) { return static_cast<int64_t>(1'000 + row); })});
  auto dataFile = common::testutil::TempFilePath::create();

  auto stream = cudf::get_default_stream();
  auto cudfTable = with_arrow::toCudfTable(
      input, input->pool(), stream, cudf::get_current_device_resource_ref());
  cudf::io::table_input_metadata metadata(cudfTable->view());
  for (size_t i = 0; i < fileRowType->size(); ++i) {
    metadata.column_metadata[i].set_name(fileRowType->nameOf(i));
  }
  auto writerOptions =
      cudf::io::parquet_writer_options::builder(
          cudf::io::sink_info{dataFile->getPath()}, cudfTable->view())
          .metadata(std::move(metadata))
          .row_group_size_rows(kRowsPerRowGroup)
          .max_page_size_rows(kRowsPerRowGroup)
          .max_page_fragment_size(kRowsPerRowGroup)
          .build();
  cudf::io::write_parquet(writerOptions, stream);
  stream.synchronize();

  auto subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add("c0", std::make_unique<common::BigintRange>(0, 9, false))
          .build();
  cudf::ast::tree filterTree;
  std::vector<std::unique_ptr<cudf::scalar>> filterScalars;
  const auto& filterExpr = createAstFromSubfieldFilters(
      subfieldFilters, filterTree, filterScalars, fileRowType);

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kUseExperimentalCudfReader, "true"},
          {CudfHiveConfig::kExperimentalDecodedColumnCacheEnabled, "true"},
          {CudfHiveConfig::kImmutableFiles, "true"},
      });
  ::facebook::velox::connector::ConnectorQueryCtx connectorQueryCtx(
      pool_.get(),
      pool_.get(),
      properties.get(),
      nullptr,
      common::PrefixSortConfig{},
      nullptr,
      nullptr,
      "query.CudfSplitReaderTest",
      "task.CudfSplitReaderTest",
      "plan.CudfSplitReaderTest",
      0,
      "");
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto tableHandle =
      CudfHiveConnectorTestBase::makeTableHandle("parquet_table", fileRowType);

  struct ReadResult {
    uint64_t hits;
    uint64_t misses;
    uint64_t decodeCalls;
    size_t chunks;
    size_t rows;
    bool fullyCached;
  };
  auto read = [&]() {
    auto split =
        CudfHiveConnectorSplitBuilder(dataFile->getPath())
            .connectorId(
                ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
            .build();
    CudfSplitReader reader(
        std::move(split),
        tableHandle,
        fileRowType,
        {"c0", "c1"},
        &fileHandleFactory,
        ioExecutor_.get(),
        &connectorQueryCtx,
        std::make_shared<CudfHiveConfig>(properties),
        std::make_shared<io::IoStatistics>(),
        std::make_shared<IoStats>(),
        true,
        &filterExpr);
    dwio::common::RuntimeStats runtimeStats;
    reader.prepareSplit(runtimeStats);
    const auto fullyCached = reader.isFullyDecodedColumnCacheHit();
    size_t chunks = 0;
    size_t rows = 0;
    while (auto chunk = reader.next(0)) {
      ++chunks;
      rows += chunk.value()->num_rows();
      reader.stream().synchronize();
    }
    return ReadResult{
        reader.decodedColumnCacheHits(),
        reader.decodedColumnCacheMisses(),
        reader.decodedColumnCacheDecodeCalls(),
        chunks,
        rows,
        fullyCached};
  };

  // Footer statistics prune the middle row group. The two surviving,
  // non-contiguous groups are decoded together and stored as two source-row
  // runs.
  auto cold = read();
  EXPECT_EQ(cold.hits, 0);
  EXPECT_EQ(cold.misses, 2);
  EXPECT_EQ(cold.decodeCalls, 1);
  EXPECT_EQ(cold.chunks, 1);
  EXPECT_EQ(cold.rows, 8);
  EXPECT_FALSE(cold.fullyCached);

  ASSERT_TRUE(std::filesystem::remove(dataFile->getPath()));
  auto hot = read();
  EXPECT_EQ(hot.hits, 2);
  EXPECT_EQ(hot.misses, 0);
  EXPECT_EQ(hot.decodeCalls, 0);
  EXPECT_EQ(hot.chunks, 1);
  EXPECT_EQ(hot.rows, 8);
  EXPECT_TRUE(hot.fullyCached);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
