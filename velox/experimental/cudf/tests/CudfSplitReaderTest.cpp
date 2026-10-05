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
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/caching/FileHandle.h"
#include "velox/common/config/Config.h"

#include <cudf/ast/expressions.hpp>

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
                                CudfHiveConnectorTestBase {};

TEST_F(CudfSplitReaderTest, accountsFilterScalarsDuringSourceConstruction) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  FileHandleFactory files(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  ColumnHandleMap columns{{"c0", makeColumnHandle("c0", BIGINT())}};
  for (const bool tracked : {false, true}) {
    SCOPED_TRACE(tracked);
    auto root = memory::memoryManager()->addRootPool();
    auto gpuPool = root->addLeafChild("filter-scalars");
    auto builder = ::facebook::velox::connector::ConnectorQueryCtx::Builder();
    builder.operatorPool(pool_.get())
        .connectorPool(pool_.get())
        .sessionProperties(properties.get());
    if (tracked) {
      builder.customPools({{"gpu", gpuPool.get()}});
    }
    auto context = builder.build();
    common::SubfieldFilters filters;
    filters.emplace(
        common::Subfield("c0"),
        std::make_unique<common::BigintRange>(1, 3, false));
    auto table = makeTableHandle("filtered", rowType, std::move(filters));
    const auto baseline = cudfAllocatedBytes();
    auto source = std::make_unique<CudfHiveDataSource>(
        rowType,
        table,
        columns,
        &files,
        ioExecutor_.get(),
        context.get(),
        std::make_shared<CudfHiveConfig>(properties));
    EXPECT_GT(cudfAllocatedBytes(), baseline);
    if (tracked) {
      EXPECT_GT(gpuPool->usedBytes(), 0);
    } else {
      EXPECT_EQ(gpuPool->usedBytes(), 0);
    }
    source.reset();
    EXPECT_EQ(gpuPool->usedBytes(), 0);
    EXPECT_EQ(cudfAllocatedBytes(), baseline);
  }
}

TEST_F(CudfSplitReaderTest, adoptedReaderOwnsResourcesWithoutQueryContext) {
  const auto rowType = ROW({"c0"}, {BIGINT()});
  constexpr vector_size_t kRows = 4096;
  constexpr int kGroups = 3;
  auto file = common::testutil::TempFilePath::create();
  std::vector<RowVectorPtr> inputs;
  for (int group = 0; group < kGroups; ++group) {
    inputs.push_back(makeRowVector(
        {"c0"}, {makeFlatVector<int64_t>(kRows, [group](vector_size_t i) {
          return group * kRows + i;
        })}));
  }
  writeToFile(file->getPath(), inputs);

  for (const bool buffered : {false, true}) {
    SCOPED_TRACE(buffered ? "buffered" : "direct");
    auto properties = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{
            {CudfHiveConfig::kUseBufferedInput, buffered ? "true" : "false"},
            {CudfHiveConfig::kMaxChunkReadLimit, "8192"},
            // Force later passes after the prepared source is destroyed.
            {CudfHiveConfig::kMaxPassReadLimit, "1"}});
    auto gpuRoot = memory::memoryManager()->addRootPool("adopted-reader");
    auto preparedPool = gpuRoot->addLeafChild("prepared-scan");
    auto activePool = gpuRoot->addLeafChild("active-scan");
    std::weak_ptr<memory::MemoryPool> weakPool = preparedPool;
    std::weak_ptr<memory::MemoryPool> weakActivePool = activePool;
    auto makeContext = [&](memory::MemoryPool* gpuPool) {
      return ::facebook::velox::connector::ConnectorQueryCtx::Builder()
          .operatorPool(pool_.get())
          .connectorPool(pool_.get())
          .sessionProperties(properties.get())
          .queryId("query.reader-handoff")
          .taskId("task.reader-handoff")
          .planNodeId("scan")
          .customPools({{"gpu", gpuPool}})
          .build();
    };
    auto preparedContext = makeContext(preparedPool.get());
    auto activeContext = makeContext(activePool.get());
    FileHandleFactory files(
        std::make_unique<FileHandleCache>(1000),
        std::make_unique<FileHandleGenerator>());
    ColumnHandleMap columns{{"c0", makeColumnHandle("c0", BIGINT())}};
    auto tableHandle = makeTableHandle("parquet_table", rowType);
    auto config = std::make_shared<CudfHiveConfig>(properties);
    auto makeSource = [&](const auto& context) {
      return std::make_unique<CudfHiveDataSource>(
          rowType,
          tableHandle,
          columns,
          &files,
          ioExecutor_.get(),
          context.get(),
          config);
    };
    auto prepared = makeSource(preparedContext);
    prepared->addSplit(makeCudfHiveConnectorSplit(file->getPath()));
    preparedContext.reset();
    auto active = makeSource(activeContext);
    active->setFromDataSource(std::move(prepared));
    RowVectorPtr retained;
    vector_size_t rows = 0;
    int chunks = 0;
    for (;;) {
      ContinueFuture future;
      auto result = active->next(kRows, future);
      ASSERT_TRUE(result.has_value());
      if (!*result)
        break;
      auto gpu = std::dynamic_pointer_cast<CudfVector>(*result);
      ASSERT_NE(gpu, nullptr);
      gpu->stream().sync();
      auto column = gpu->getTableView().column(0);
      std::vector<int64_t> actual(column.size());
      ASSERT_EQ(
          cudaMemcpy(
              actual.data(),
              column.data<int64_t>(),
              actual.size() * sizeof(int64_t),
              cudaMemcpyDeviceToHost),
          cudaSuccess);
      for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_EQ(actual[i], rows + i);
      rows += gpu->size();
      ++chunks;
      if (!retained)
        retained = *result;
    }
    EXPECT_EQ(rows, kGroups * kRows);
    EXPECT_GT(chunks, 1);
    EXPECT_GT(gpuRoot->peakBytes(), 0);
    EXPECT_EQ(activePool->usedBytes(), 0);
    active.reset();
    activeContext.reset();
    preparedPool.reset();
    activePool.reset();
    gpuRoot.reset();
    ASSERT_FALSE(weakPool.expired());
    EXPECT_GT(weakPool.lock()->usedBytes(), 0);
    retained.reset();
    EXPECT_TRUE(weakPool.expired());
    EXPECT_TRUE(weakActivePool.expired());
  }
}

TEST_F(CudfSplitReaderTest, buildsPushdownFilterForEachSplitPreparation) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})}));

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto connectorQueryCtx =
      ::facebook::velox::connector::ConnectorQueryCtx::Builder()
          .operatorPool(pool_.get())
          .connectorPool(pool_.get())
          .sessionProperties(properties.get())
          .queryId("query.CudfSplitReaderTest")
          .taskId("task.CudfSplitReaderTest")
          .planNodeId("plan.CudfSplitReaderTest")
          .build();
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
      connectorQueryCtx.get(),
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
  reader.setMemoryResources(*mr_, *output_mr_);
  reader.prepareSplit(runtimeStats);
  EXPECT_THROW(reader.setMemoryResources(*mr_, *output_mr_), VeloxRuntimeError);
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
