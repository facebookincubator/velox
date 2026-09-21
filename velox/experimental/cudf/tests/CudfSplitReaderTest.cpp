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
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/caching/FileHandle.h"
#include "velox/common/config/Config.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/parquet/writer/Writer.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

class MetadataOnlySplitReader final : public CudfSplitReader {
 public:
  using CudfSplitReader::CudfSplitReader;

  cudf::ast::expression const* logicalFilter() const {
    return subfieldFilterAst();
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
                                CudfHiveConnectorTestBase {
 protected:
  void writeCompactDecimalParquet(
      const std::shared_ptr<common::testutil::TempFilePath>& filePath,
      const RowVectorPtr& vector) {
    auto fs = filesystems::getFileSystem(filePath->getPath(), {});
    auto writeFile = fs->openFileForWrite(
        filePath->getPath(),
        {.shouldCreateParentDirectories = true,
         .shouldThrowOnFileAlreadyExists = false});
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::move(writeFile), filePath->getPath());
    auto writerPool = rootPool_->addAggregateChild(
        "CudfSplitReaderTest.ParquetWriter");
    dwio::common::WriterOptions options;
    options.memoryPool = writerPool.get();
    auto parquetOptions = std::make_shared<parquet::ParquetWriterOptions>();
    parquetOptions->enableStoreDecimalAsInteger = true;
    options.formatSpecificOptions = std::move(parquetOptions);
    parquet::Writer writer(
        std::move(sink), options, writerPool, vector->rowType());
    writer.write(vector);
    writer.close();
  }

  cudf::data_type readPhysicalType(
      const std::shared_ptr<common::testutil::TempFilePath>& filePath,
      const RowTypePtr& rowType,
      bool preserveCompactDecimals) {
    auto properties = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{
            {CudfHiveConfig::kPreserveCompactDecimalsSession,
             preserveCompactDecimals ? "true" : "false"}});
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
    auto split = CudfHiveConnectorSplitBuilder(filePath->getPath()).build();
    CudfSplitReader reader(
        std::move(split),
        makeTableHandle("parquet_table", rowType),
        rowType,
        rowType->names(),
        &fileHandleFactory,
        ioExecutor_.get(),
        &connectorQueryCtx,
        std::make_shared<CudfHiveConfig>(properties),
        std::make_shared<io::IoStatistics>(),
        std::make_shared<IoStats>(),
        nullptr);
    dwio::common::RuntimeStats runtimeStats;
    reader.prepareSplit(runtimeStats);
    auto result = reader.next(1);
    VELOX_CHECK(result.has_value());
    return result.value()->view().column(0).type();
  }
};

TEST_F(CudfSplitReaderTest, preservesCompactDecimalsFromReader) {
  auto fileType = ROW({"c0"}, {DECIMAL(7, 2)});
  auto fileVector = makeRowVector(
      {makeNullableFlatVector<int64_t>({123, -456, std::nullopt}, DECIMAL(7, 2))});
  auto filePath = common::testutil::TempFilePath::create();
  writeCompactDecimalParquet(filePath, fileVector);
  auto decimal64Path = common::testutil::TempFilePath::create();
  writeCompactDecimalParquet(
      decimal64Path,
      makeRowVector(
          {makeFlatVector<int64_t>({123}, DECIMAL(18, 2))}));

  EXPECT_EQ(
      readPhysicalType(
          filePath, fileType, /*preserveCompactDecimals=*/false),
      (cudf::data_type{cudf::type_id::DECIMAL64, -2}));
  EXPECT_EQ(
      readPhysicalType(filePath, fileType, /*preserveCompactDecimals=*/true),
      (cudf::data_type{cudf::type_id::DECIMAL32, -2}));
  EXPECT_EQ(
      readPhysicalType(
          filePath,
          ROW({"c0"}, {DECIMAL(12, 4)}),
          /*preserveCompactDecimals=*/true),
      (cudf::data_type{cudf::type_id::DECIMAL64, -4}));
  EXPECT_EQ(
      readPhysicalType(
          decimal64Path,
          ROW({"c0"}, {DECIMAL(7, 2)}),
          /*preserveCompactDecimals=*/true),
      (cudf::data_type{cudf::type_id::DECIMAL32, -2}));
  EXPECT_EQ(
      readPhysicalType(
          decimal64Path,
          ROW({"c0"}, {DECIMAL(18, 2)}),
          /*preserveCompactDecimals=*/true),
      (cudf::data_type{cudf::type_id::DECIMAL64, -2}));
}

TEST_F(CudfSplitReaderTest, canonicalizesCompactDecimalsByPrecision) {
  auto stream = cudf::get_default_stream();
  auto makeTable = [&](cudf::type_id type, int32_t scale) {
    auto column = cudf::make_fixed_width_column(
        cudf::data_type{type, -scale},
        1,
        cudf::mask_state::UNALLOCATED,
        stream);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(column));
    return std::make_unique<cudf::table>(std::move(columns));
  };
  const std::array<TypePtr, 1> logicalType{DECIMAL(7, 2)};
  auto normalize = [&](cudf::type_id type,
                       int32_t scale,
                       bool preserveCompactDecimals) {
    return castDecimalColumnsToVeloxTypes(
        makeTable(type, scale),
        logicalType,
        /*numPrependedColumns=*/0,
        preserveCompactDecimals,
        stream,
        cudf::get_current_device_resource_ref());
  };

  EXPECT_EQ(
      normalize(cudf::type_id::DECIMAL32, 2, false)->view().column(0).type(),
      (cudf::data_type{cudf::type_id::DECIMAL64, -2}));
  EXPECT_EQ(
      normalize(cudf::type_id::DECIMAL32, 2, true)->view().column(0).type(),
      (cudf::data_type{cudf::type_id::DECIMAL32, -2}));
  EXPECT_EQ(
      normalize(cudf::type_id::DECIMAL32, 4, true)->view().column(0).type(),
      (cudf::data_type{cudf::type_id::DECIMAL32, -2}));
  EXPECT_EQ(
      normalize(cudf::type_id::DECIMAL64, 2, true)->view().column(0).type(),
      (cudf::data_type{cudf::type_id::DECIMAL32, -2}));
}

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

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
