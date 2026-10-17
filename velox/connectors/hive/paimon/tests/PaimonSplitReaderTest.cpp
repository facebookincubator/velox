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

#include "velox/connectors/hive/paimon/PaimonDataSource.h"

#include "velox/common/file/LocalFile.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#ifdef VELOX_ENABLE_NIMBLE
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#endif

namespace facebook::velox::connector::hive::paimon {
namespace {

using dwio::common::FileFormat;
using RuntimeStatsMap = std::unordered_map<std::string, RuntimeMetric>;

class PaimonSplitReaderTest : public exec::test::HiveConnectorTestBase,
                              public testing::WithParamInterface<FileFormat> {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    queryCtx_ = core::QueryCtx::create();
    context_ = ConnectorQueryCtx::Builder()
                   .operatorPool(pool_.get())
                   .connectorPool(pool_.get())
                   .sessionProperties(config_.get())
                   .expressionEvaluator(
                       std::make_unique<exec::SimpleExpressionEvaluator>(
                           queryCtx_.get(), pool_.get()))
                   .queryId("PaimonSplitReaderTest")
                   .taskId("PaimonSplitReaderTest")
                   .planNodeId("scan")
                   .build();
#ifdef VELOX_ENABLE_NIMBLE
    nimble::registerSelectiveNimbleReaderFactory();
#endif
  }

  void TearDown() override {
#ifdef VELOX_ENABLE_NIMBLE
    nimble::unregisterSelectiveNimbleReaderFactory();
#endif
    context_.reset();
    queryCtx_.reset();
    HiveConnectorTestBase::TearDown();
  }

  template <typename T>
  std::unique_ptr<T> makeSource() {
    return std::make_unique<T>(
        type_,
        makeTableHandle({}, nullptr, "test", type_),
        ColumnHandleMap{{"c0", regularColumn("c0", BIGINT())}},
        &fileFactory_,
        nullptr,
        context_.get(),
        std::make_shared<PaimonConfig>(config_));
  }

  void writeFile(const std::string& path, vector_size_t rows) {
    auto input = makeRowVector(
        {makeFlatVector<int64_t>(rows, [](auto row) { return row; })});
    if (GetParam() == FileFormat::DWRF) {
      writeToFile(path, input);
    }
#ifdef VELOX_ENABLE_NIMBLE
    else {
      nimble::Writer writer(
          input->type(),
          std::make_unique<LocalWriteFile>(path, false, false),
          *pool_,
          nimble::WriterOptions{});
      writer.write(input);
      writer.close();
    }
#endif
  }

  static RowVectorPtr next(DataSource& source) {
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto result = source.next(50, future);
    VELOX_CHECK(result.has_value());
    VELOX_CHECK(!future.valid());
    if (result.value()) {
      result.value()->loadedVector();
    }
    return result.value();
  }

  static void expectStatsEqual(
      const RuntimeStatsMap& expected,
      const RuntimeStatsMap& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (const auto& [name, metric] : expected) {
      SCOPED_TRACE(name);
      const auto it = actual.find(name);
      ASSERT_NE(it, actual.end());
      EXPECT_EQ(metric.unit, it->second.unit);
      EXPECT_EQ(metric.sum, it->second.sum);
      EXPECT_EQ(metric.count, it->second.count);
      EXPECT_EQ(metric.min, it->second.min);
      EXPECT_EQ(metric.max, it->second.max);
    }
  }

  void testRuntimeStats(bool cancel) {
    const auto files = makeFilePaths(3);
    PaimonConnectorSplitBuilder builder(
        exec::test::kHiveConnectorId,
        /*snapshotId=*/1,
        PaimonTableType::kAppendOnly,
        GetParam());
    for (int i = 0; i < files.size(); ++i) {
      writeFile(files[i]->getPath(), 100 * (i + 1));
      builder.addFile(files[i]->getPath(), /*fileSize=*/0);
    }
    auto source = makeSource<PaimonDataSource>();
    source->addSplit(builder.build());

    std::vector<std::string> counters{
        "processedSplits", "footerBufferOverread"};
    if (GetParam() == FileFormat::DWRF) {
      counters.push_back("processedStrides");
      counters.push_back("numStripes");
    } else {
      counters.push_back("processedRows");
    }
    std::unordered_map<std::string, int64_t> completedTotals;
    const auto expectTotals = [&](const RuntimeStatsMap& active) {
      const auto stats = source->getRuntimeStats();
      expectStatsEqual(stats, source->getRuntimeStats());
      for (const auto& name : counters) {
        SCOPED_TRACE(name);
        const auto it = stats.find(name);
        ASSERT_NE(it, stats.end());
        const auto activeIt = active.find(name);
        const auto expected = completedTotals[name] +
            (activeIt == active.end() ? 0 : activeIt->second.sum);
        EXPECT_GT(expected, 0);
        EXPECT_EQ(it->second.sum, expected);
        EXPECT_EQ(it->second.count, 1);
        EXPECT_EQ(it->second.min, expected);
        EXPECT_EQ(it->second.max, expected);
      }
    };

    uint64_t totalRows = 0;
    for (int i = 0; i < files.size(); ++i) {
      SCOPED_TRACE(i);
      // Independent single-file reads provide the expected counters at the
      // same read position, including the active file after each transition.
      auto single = makeSource<FileDataSource>();
      auto split = makeHiveConnectorSplit(files[i]->getPath());
      split->fileFormat = GetParam();
      single->addSplit(split);
      while (auto expected = next(*single)) {
        auto actual = next(*source);
        ASSERT_TRUE(actual);
        test::assertEqualVectors(expected, actual);
        totalRows += actual->size();
        expectTotals(single->getRuntimeStats());

        if (cancel && i == 1) {
          // The first file is archived; the second still has unread rows.
          const auto stats = source->getRuntimeStats();
          source->cancel();
          expectStatsEqual(stats, source->getRuntimeStats());
          source->cancel();
          expectStatsEqual(stats, source->getRuntimeStats());
          EXPECT_EQ(source->getCompletedRows(), totalRows);
          return;
        }
      }
      const auto stats = single->getRuntimeStats();
      for (const auto& name : counters) {
        completedTotals[name] += stats.at(name).sum;
      }
    }

    // EOF destroys the last physical reader too. Its statistics must survive.
    EXPECT_EQ(next(*source), nullptr);
    EXPECT_EQ(totalRows, 600);
    EXPECT_EQ(source->getCompletedRows(), totalRows);
    expectTotals({});
    const auto stats = source->getRuntimeStats();
    source->cancel();
    expectStatsEqual(stats, source->getRuntimeStats());
  }

  const RowTypePtr type_ = ROW({{"c0", BIGINT()}});
  const std::shared_ptr<config::ConfigBase> config_ =
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{});
  FileHandleFactory fileFactory_{
      std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(0),
      std::make_unique<FileHandleGenerator>()};
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<ConnectorQueryCtx> context_;
};

TEST_P(PaimonSplitReaderTest, runtimeStatsAcrossFiles) {
  testRuntimeStats(false);
}

TEST_P(PaimonSplitReaderTest, runtimeStatsAfterCancel) {
  testRuntimeStats(true);
}

INSTANTIATE_TEST_SUITE_P(
    Dwrf,
    PaimonSplitReaderTest,
    testing::Values(FileFormat::DWRF));

#ifdef VELOX_ENABLE_NIMBLE
INSTANTIATE_TEST_SUITE_P(
    Nimble,
    PaimonSplitReaderTest,
    testing::Values(FileFormat::NIMBLE));
#endif

} // namespace
} // namespace facebook::velox::connector::hive::paimon
