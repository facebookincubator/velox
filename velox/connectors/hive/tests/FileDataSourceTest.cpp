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

#include "velox/connectors/hive/FileDataSource.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::connector::hive {
namespace {

// Models a logical reader that consumes physical rows, blocks, then emits
// buffered logical rows. This is deliberately independent of a file format.
class BufferedScanReader : public FileScanReader {
 public:
  explicit BufferedScanReader(RowVectorPtr rows) : rows_(std::move(rows)) {}

  ScanReadResult next(uint64_t /*maxRows*/, ContinueFuture& future) override {
    switch (calls_++) {
      case 0:
        physicalRows_ = 11;
        return {
            ScanReadResult::State::kData,
            RowVector::createEmpty(rows_->type(), rows_->pool()),
            11};
      case 1:
        future = promise_.getSemiFuture();
        return {ScanReadResult::State::kBlocked, nullptr};
      case 2:
        return {ScanReadResult::State::kData, rows_};
      default:
        return {ScanReadResult::State::kEnd, nullptr};
    }
  }

  void unblock() {
    promise_.setValue();
  }
  void resetFilterCaches() override {}
  void updateRuntimeStats(dwio::common::RuntimeStats& stats) const override {
    stats.processedRows += physicalRows_;
  }
  int64_t estimatedRowSize() const override {
    return 8;
  }
  bool allPrefetchIssued() const override {
    return true;
  }
  void resetSplit() override {
    rows_.reset();
  }
  void cancel() override {
    rows_.reset();
    if (!promise_.isFulfilled()) {
      promise_.setValue();
    }
    cancelled = true;
  }
  void setConnectorQueryCtx(const ConnectorQueryCtx* /*ctx*/) override {}
  const FileConnectorSplit* currentFileSplit() const override {
    return nullptr;
  }

  bool cancelled{false};

 private:
  RowVectorPtr rows_;
  ContinuePromise promise_{"BufferedScanReader"};
  int32_t calls_{0};
  int64_t physicalRows_{0};
};

class BufferedDataSource : public FileDataSource {
 public:
  using FileDataSource::FileDataSource;
  std::unique_ptr<FileScanReader> reader;

 protected:
  std::unique_ptr<FileScanReader> createScanReader() override {
    return std::move(reader);
  }
};

class FileDataSourceTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    query_ = core::QueryCtx::create();
    ctx_ = std::make_unique<ConnectorQueryCtx>(
        pool(),
        pool(),
        &session_,
        nullptr,
        common::PrefixSortConfig(),
        std::make_unique<exec::SimpleExpressionEvaluator>(query_.get(), pool()),
        nullptr,
        "query",
        "task",
        "scan",
        0,
        "");
  }

  void TearDown() override {
    ctx_.reset();
    query_.reset();
    HiveConnectorTestBase::TearDown();
  }

  std::unique_ptr<BufferedDataSource> makeSource(const RowTypePtr& outputType) {
    const auto type = ROW({"c0"}, {BIGINT()});
    auto source = std::make_unique<BufferedDataSource>(
        outputType,
        makeTableHandle({}, parseExpr("c0 % 2 = 0", type), "buffered", type),
        allRegularColumns(type),
        nullptr,
        nullptr,
        ctx_.get(),
        std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{})));
    source->reader = std::make_unique<BufferedScanReader>(
        makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3, 4})}));
    return source;
  }

  config::ConfigBase session_{{}};
  std::shared_ptr<core::QueryCtx> query_;
  std::unique_ptr<ConnectorQueryCtx> ctx_;
};

TEST_F(FileDataSourceTest, bufferedOutputBlockedAndEnd) {
  for (bool emptyProjection : {false, true}) {
    const auto type = emptyProjection ? ROW({}, {}) : ROW({"c0"}, {BIGINT()});
    auto source = makeSource(type);
    auto* reader = static_cast<BufferedScanReader*>(source->reader.get());
    // A logical split need not be a FileConnectorSplit.
    source->addSplit(
        std::make_shared<ConnectorSplit>(exec::test::kHiveConnectorId));
    ContinueFuture future;
    auto empty = source->next(10, future);
    ASSERT_TRUE(empty.has_value());
    ASSERT_NE(*empty, nullptr);
    EXPECT_EQ((*empty)->size(), 0);
    EXPECT_EQ(source->getCompletedRows(), 11);
    EXPECT_FALSE(source->next(10, future).has_value());
    ASSERT_TRUE(future.valid());
    EXPECT_FALSE(future.isReady());
    reader->unblock();
    EXPECT_TRUE(future.isReady());
    auto data = source->next(10, future);
    ASSERT_TRUE(data.has_value());
    ASSERT_NE(*data, nullptr);
    EXPECT_EQ((*data)->size(), 2);
    EXPECT_EQ((*data)->childrenSize(), type->size());
    EXPECT_EQ(source->getCompletedRows(), 11);
    if (!emptyProjection) {
      test::assertEqualVectors(
          makeRowVector({"c0"}, {makeFlatVector<int64_t>({2, 4})}), *data);
    }
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(source->getRuntimeStats().at("processedRows").sum, 11);
    }
    auto end = source->next(10, future);
    ASSERT_TRUE(end.has_value());
    EXPECT_EQ(*end, nullptr);
    EXPECT_EQ(source->getRuntimeStats().at("processedRows").sum, 11);
    source->cancel();
    EXPECT_EQ(source->getRuntimeStats().at("processedRows").sum, 11);
  }
}

TEST_F(FileDataSourceTest, cancelArchivesActiveStatisticsOnce) {
  auto source = makeSource(ROW({"c0"}, {BIGINT()}));
  auto* reader = static_cast<BufferedScanReader*>(source->reader.get());
  source->addSplit(
      std::make_shared<ConnectorSplit>(exec::test::kHiveConnectorId));
  ContinueFuture future;
  ASSERT_NE(*source->next(10, future), nullptr);
  EXPECT_EQ(source->getRuntimeStats().at("processedRows").sum, 11);
  source->cancel();
  EXPECT_TRUE(reader->cancelled);
  source->cancel();
  EXPECT_EQ(source->getRuntimeStats().at("processedRows").sum, 11);
}

TEST_F(FileDataSourceTest, reusesReleasedNestedRows) {
  auto rows = makeRowVector(
      {"outer"},
      {makeRowVector(
          {"inner"},
          {makeRowVector(
              {"n"}, {makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})})})});
  auto paths = makeFilePaths(1);
  writeToFile(paths[0]->getPath(), rows);

  for (bool retainOutput : {false, true}) {
    SCOPED_TRACE(retainOutput);
    auto source =
        ConnectorRegistry::tryGet(exec::test::kHiveConnectorId)
            ->createDataSource(
                rows->rowType(),
                makeTableHandle({}, nullptr, "reuse", rows->rowType()),
                allRegularColumns(rows->rowType()),
                ctx_.get());
    source->addSplit(makeHiveConnectorSplit(paths[0]->getPath()));
    std::weak_ptr<BaseVector> previousOuter;
    std::weak_ptr<BaseVector> previousInner;
    std::vector<RowVectorPtr> held;
    ContinueFuture future;
    for (vector_size_t offset = 0; offset < rows->size(); offset += 2) {
      auto batch = source->next(2, future);
      ASSERT_TRUE(batch.has_value());
      ASSERT_NE(*batch, nullptr);
      test::assertEqualVectors(rows->slice(offset, 2), *batch);
      const auto& outer = BaseVector::loadedVectorShared((*batch)->childAt(0));
      const auto& inner =
          BaseVector::loadedVectorShared(outer->as<RowVector>()->childAt(0));
      if (offset > 0 && !retainOutput) {
        // Weak references observe actual DWIO reuse without retaining a batch.
        EXPECT_FALSE(previousOuter.expired());
        EXPECT_FALSE(previousInner.expired());
        EXPECT_EQ(previousOuter.lock().get(), outer.get());
        EXPECT_EQ(previousInner.lock().get(), inner.get());
      }
      previousOuter = outer;
      previousInner = inner;
      if (retainOutput) {
        held.push_back(*batch);
        for (vector_size_t i = 0; i < held.size(); ++i) {
          test::assertEqualVectors(rows->slice(i * 2, 2), held[i]);
        }
      }
    }
    auto end = source->next(2, future);
    ASSERT_TRUE(end.has_value());
    EXPECT_EQ(*end, nullptr);
    source->cancel();
  }
}

} // namespace
} // namespace facebook::velox::connector::hive
