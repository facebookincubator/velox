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
#include "velox/common/file/LocalFile.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#ifdef VELOX_ENABLE_NIMBLE
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#endif

namespace facebook::velox::connector::hive {
namespace {
using exec::test::HiveConnectorTestBase;
using exec::test::kHiveConnectorId;
using exec::test::PlanBuilder;
using exec::test::TempFilePath;
using State = ScanReadResult::State;

struct Step {
  State state;
  uint64_t scanned{0};
  std::vector<int64_t> values{};
  bool blockedFuture{false};
  bool fail{false};
  bool missingOutput{false};
};

struct ReaderState {
  std::vector<Step> steps;
  size_t position{0};
  uint64_t scanned{0};
  int cancels{0};
  int destroyed{0};
  int filterResets{0};
  bool failStats{false};
  ContinuePromise promise{ContinuePromise::makeEmpty()};
  const ConnectorQueryCtx* context{nullptr};
};

class ScriptedReader final : public FileScanReader {
 public:
  ScriptedReader(std::shared_ptr<ReaderState> state, memory::MemoryPool* pool)
      : state_(std::move(state)), pool_(pool) {}

  ScriptedReader(const ScriptedReader&) = delete;
  ScriptedReader& operator=(const ScriptedReader&) = delete;
  ScriptedReader(ScriptedReader&&) = delete;
  ScriptedReader& operator=(ScriptedReader&&) = delete;

  ~ScriptedReader() override {
    ++state_->destroyed;
  }

  ScanReadResult
  next(uint64_t maxRows, VectorPtr& output, ContinueFuture& future) override {
    VELOX_CHECK_LT(state_->position, state_->steps.size());
    const auto& step = state_->steps[state_->position++];
    VELOX_CHECK(!step.fail, "Injected reader failure");
    state_->scanned += step.scanned;
    if (step.blockedFuture) {
      auto pair = makeVeloxContinuePromiseContract("FileDataSourceTest");
      state_->promise = std::move(pair.first);
      future = std::move(pair.second);
    }
    if (step.state == State::kData) {
      if (step.missingOutput) {
        output.reset();
      } else {
        VELOX_CHECK_LE(step.values.size(), maxRows);
        const auto type = ROW({{"c0", BIGINT()}});
        const auto numRows = static_cast<vector_size_t>(step.values.size());
        BaseVector::ensureWritable(
            SelectivityVector(numRows), type, pool_, output);
        output->resize(numRows);
        auto* values =
            output->as<RowVector>()->childAt(0)->asFlatVector<int64_t>();
        for (vector_size_t i = 0; i < step.values.size(); ++i) {
          values->set(i, step.values[i]);
        }
      }
    }
    return {step.state, step.scanned};
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats()
      const override {
    VELOX_CHECK(!state_->failStats, "Injected statistics failure");
    return {{"testScanned", RuntimeMetric(saturateCast(state_->scanned))}};
  }
  void resetFilterCaches() override {
    ++state_->filterResets;
  }
  int64_t estimatedRowSize() const override {
    return 8;
  }
  bool allPrefetchIssued() const override {
    return true;
  }
  void setConnectorQueryCtx(const ConnectorQueryCtx* context) override {
    state_->context = context;
  }
  void cancel() noexcept override {
    if (!cancelled_) {
      cancelled_ = true;
      ++state_->cancels;
      state_->promise = ContinuePromise::makeEmpty();
    }
  }

 private:
  const std::shared_ptr<ReaderState> state_;
  memory::MemoryPool* const pool_;
  bool cancelled_{false};
};

class TestDataSource : public FileDataSource {
 public:
  using FileDataSource::FileDataSource;
  std::shared_ptr<ReaderState> state;

 protected:
  std::unique_ptr<FileScanReader> createScanReader() override {
    VELOX_CHECK_NOT_NULL(activeSplit_);
    // Exercise a logical ConnectorSplit without a fabricated file split.
    VELOX_CHECK_NULL(dynamic_cast<FileConnectorSplit*>(activeSplit_.get()));
    return std::make_unique<ScriptedReader>(state, pool_);
  }
};

class TestConnector : public HiveConnector {
 public:
  TestConnector(
      const std::shared_ptr<const config::ConfigBase>& config,
      std::shared_ptr<ReaderState> state)
      : HiveConnector(kHiveConnectorId, config, nullptr),
        config_(std::make_shared<HiveConfig>(config)),
        state_(std::move(state)) {}

  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columns,
      ConnectorQueryCtx* context) override {
    auto source = std::make_unique<TestDataSource>(
        outputType, tableHandle, columns, nullptr, nullptr, context, config_);
    source->state = state_;
    return source;
  }

  bool supportsSplitPreload() const override {
    return false;
  }

 private:
  const std::shared_ptr<HiveConfig> config_;
  const std::shared_ptr<ReaderState> state_;
};

class FileDataSourceTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    queryCtx_ = core::QueryCtx::create();
    context_ = makeContext();
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

  std::unique_ptr<ConnectorQueryCtx> makeContext() {
    return ConnectorQueryCtx::Builder()
        .operatorPool(pool_.get())
        .connectorPool(pool_.get())
        .sessionProperties(config_.get())
        .expressionEvaluator(
            std::make_unique<exec::SimpleExpressionEvaluator>(
                queryCtx_.get(), pool_.get()))
        .queryId("FileDataSourceTest")
        .taskId("FileDataSourceTest")
        .planNodeId("scan")
        .build();
  }

  std::unique_ptr<TestDataSource> makeSource(
      const std::shared_ptr<ReaderState>& state,
      const std::string& remainingFilter = "",
      bool countOnly = false) {
    auto source = std::make_unique<TestDataSource>(
        countOnly ? ROW({}) : type_,
        makeTableHandle(
            {},
            remainingFilter.empty() ? nullptr
                                    : parseExpr(remainingFilter, type_),
            "test",
            type_),
        countOnly ? ColumnHandleMap{} : assignments_,
        &fileFactory_,
        nullptr,
        context_.get(),
        hiveConfig_);
    source->state = state;
    return source;
  }

  static std::shared_ptr<ConnectorSplit> logicalSplit() {
    return std::make_shared<ConnectorSplit>(kHiveConnectorId);
  }

  static std::shared_ptr<ReaderState> script(std::vector<Step> steps) {
    auto state = std::make_shared<ReaderState>();
    state->steps = std::move(steps);
    return state;
  }

  RowVectorPtr next(DataSource& source, uint64_t size = 10) {
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto result = source.next(size, future);
    VELOX_CHECK(result.has_value());
    VELOX_CHECK(!future.valid());
    return result.value();
  }

  static void expectStatsEqual(
      const std::unordered_map<std::string, RuntimeMetric>& expected,
      const std::unordered_map<std::string, RuntimeMetric>& actual) {
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

  void expectValues(
      const RowVectorPtr& result,
      const std::vector<int64_t>& values) {
    ASSERT_TRUE(result);
    test::assertEqualVectors(
        makeRowVector({makeFlatVector<int64_t>(values)}), result);
  }

  const RowTypePtr type_ = ROW({{"c0", BIGINT()}});
  const ColumnHandleMap assignments_{{"c0", regularColumn("c0", BIGINT())}};
  const std::shared_ptr<config::ConfigBase> config_ =
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{});
  const std::shared_ptr<HiveConfig> hiveConfig_ =
      std::make_shared<HiveConfig>(config_);
  FileHandleFactory fileFactory_{
      std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(0),
      std::make_unique<FileHandleGenerator>()};
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<ConnectorQueryCtx> context_;
};

TEST_F(FileDataSourceTest, emptyBlockedCachedAndEnd) {
  auto state = script(
      {{State::kData, 4, {}},
       {State::kBlocked, 2, {}, true},
       {State::kData, 0, {1, 2, 3}},
       {State::kEnd, 1}});
  auto source = makeSource(state, "c0 % 2 = 1");
  auto split = logicalSplit();
  std::weak_ptr<ConnectorSplit> weakSplit = split;
  source->addSplit(std::move(split));
  ASSERT_EQ(next(*source)->size(), 0);
  ASSERT_FALSE(weakSplit.expired());
  ASSERT_EQ(source->getCompletedRows(), 4);
  ContinueFuture future = ContinueFuture::makeEmpty();
  ASSERT_FALSE(source->next(10, future).has_value());
  ASSERT_TRUE(future.valid());
  ASSERT_FALSE(future.isReady());
  ASSERT_EQ(source->getCompletedRows(), 6);
  state->promise.setValue();
  std::move(future).get();
  auto cached = next(*source);
  expectValues(cached, {1, 3});
  int callbacks = 0;
  source->setScanBatchCallback([&](const core::ScanBatchEvent& event) {
    const auto& fileEvent = static_cast<const FileScanBatchEvent&>(event);
    EXPECT_TRUE(fileEvent.filePath.empty());
    EXPECT_EQ(fileEvent.partitionKeys, nullptr);
    EXPECT_EQ(fileEvent.fileFormat, dwio::common::FileFormat::UNKNOWN);
    ++callbacks;
  });
  source->fireScanBatchCallback({});
  ASSERT_EQ(callbacks, 1);
  ASSERT_EQ(source->getCompletedRows(), 6);
  ASSERT_EQ(next(*source), nullptr);
  ASSERT_EQ(source->getCompletedRows(), 7);
  ASSERT_TRUE(weakSplit.expired());
  ASSERT_EQ(state->cancels, 1);
  ASSERT_EQ(state->destroyed, 1);
  expectValues(cached, {1, 3});
}

TEST_F(FileDataSourceTest, outputOwnershipAndReuse) {
  auto state = script(
      {{State::kData, 3, {1, 2, 3}},
       {State::kData, 0, {4, 5, 6}},
       {State::kData, 0, {7, 8, 9}},
       {State::kEnd}});
  auto source = makeSource(state);
  source->addSplit(logicalSplit());
  auto first = next(*source);
  const auto* reusable = first->childAt(0).get();
  first.reset();
  auto second = next(*source);
  ASSERT_EQ(second->childAt(0).get(), reusable);
  auto third = next(*source);
  expectValues(second, {4, 5, 6});
  expectValues(third, {7, 8, 9});
  ASSERT_NE(second->childAt(0).get(), third->childAt(0).get());
  source->cancel();
  expectValues(second, {4, 5, 6});
  expectValues(third, {7, 8, 9});
}

TEST_F(FileDataSourceTest, zeroColumnProjectionWithRemainingFilter) {
  auto source = makeSource(
      script({{State::kData, 0, {1, 2, 3}}, {State::kEnd}}),
      "c0 % 2 = 1",
      true);
  source->addSplit(logicalSplit());
  auto result = next(*source);
  ASSERT_EQ(result->size(), 2);
  ASSERT_EQ(result->childrenSize(), 0);
  ASSERT_EQ(result->type()->size(), 0);
  ASSERT_EQ(source->getCompletedRows(), 0);
  ASSERT_EQ(next(*source), nullptr);
}

TEST_F(FileDataSourceTest, invalidReaderResultsReleaseResources) {
  const std::vector<std::pair<Step, std::string>> cases{
      {{State::kBlocked}, "must provide a future"},
      {{State::kData, 0, {}, false, false, true}, "without a vector"},
      {{State::kData, 0, {1}, true}, "data and a future"},
      {{State::kEnd, 0, {}, true}, "returned a future"},
      {{State::kData, 0, {}, false, true}, "Injected reader failure"}};
  for (const auto& [step, error] : cases) {
    SCOPED_TRACE(error);
    auto state = script({step});
    auto source = makeSource(state);
    source->addSplit(logicalSplit());
    // A stale, valid future must not satisfy the blocked contract.
    ContinueFuture future = folly::makeSemiFuture();
    VELOX_ASSERT_THROW(source->next(10, future), error);
    ASSERT_EQ(state->cancels, 1);
    ASSERT_EQ(state->destroyed, 1);
    source->cancel();
    ASSERT_EQ(state->cancels, 1);
  }
}

TEST_F(FileDataSourceTest, failureAfterOutputKeepsProgress) {
  auto state = script(
      {{State::kData, 3, {1, 2, 3}}, {State::kData, 0, {}, false, true}});
  auto source = makeSource(state);
  source->addSplit(logicalSplit());
  auto output = next(*source);
  VELOX_ASSERT_THROW(next(*source), "Injected reader failure");
  ASSERT_EQ(source->getCompletedRows(), 3);
  ASSERT_EQ(source->getRuntimeStats().at("testScanned").sum, 3);
  ASSERT_EQ(state->destroyed, 1);
  expectValues(output, {1, 2, 3});
}

TEST_F(FileDataSourceTest, statisticsFailureStillReleasesResources) {
  for (bool readFails : {false, true}) {
    SCOPED_TRACE(readFails);
    auto state = script({{State::kEnd, 0, {}, false, readFails}});
    state->failStats = true;
    auto source = makeSource(state);
    auto split = logicalSplit();
    std::weak_ptr<ConnectorSplit> weakSplit = split;
    source->addSplit(std::move(split));
    VELOX_ASSERT_THROW(
        next(*source),
        readFails ? "Injected reader failure" : "Injected statistics failure");
    ASSERT_TRUE(weakSplit.expired());
    ASSERT_EQ(state->cancels, 1);
    ASSERT_EQ(state->destroyed, 1);
    source->cancel();
    ASSERT_EQ(state->cancels, 1);
  }

  // Destruction must not call the potentially throwing statistics hook.
  auto state = script({});
  state->failStats = true;
  {
    auto source = makeSource(state);
    source->addSplit(logicalSplit());
  }
  ASSERT_EQ(state->cancels, 1);
  ASSERT_EQ(state->destroyed, 1);
}

TEST_F(FileDataSourceTest, statisticsSnapshotsAndCleanup) {
  auto state = script({{State::kData, 3, {1}}, {State::kEnd}});
  auto source = makeSource(state);
  ASSERT_EQ(source->estimatedRowSize(), DataSource::kUnknownRowSize);
  source->addSplit(logicalSplit());
  ASSERT_EQ(source->estimatedRowSize(), 8);
  ASSERT_TRUE(source->allPrefetchIssued());
  next(*source);
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(source->getRuntimeStats().at("testScanned").sum, 3);
  }
  ASSERT_EQ(next(*source), nullptr);
  ASSERT_EQ(source->getRuntimeStats().at("testScanned").sum, 3);
  source->cancel();
  ASSERT_EQ(source->getRuntimeStats().at("testScanned").sum, 3);
  source->state =
      script({{State::kData, 2, {4}}, {State::kBlocked, 0, {}, true}});
  source->addSplit(logicalSplit());
  next(*source);
  ContinueFuture future = ContinueFuture::makeEmpty();
  ASSERT_FALSE(source->next(10, future).has_value());
  source->cancel();
  ASSERT_TRUE(future.isReady());
  ASSERT_EQ(source->getRuntimeStats().at("testScanned").sum, 5);
  ASSERT_EQ(source->getCompletedRows(), 5);
  ASSERT_EQ(source->state->cancels, 1);
  source->cancel();
  ASSERT_EQ(source->getRuntimeStats().at("testScanned").sum, 5);
}

TEST_F(FileDataSourceTest, destructionAndSplitContract) {
  auto state = script({});
  {
    auto source = makeSource(state);
    source->addSplit(logicalSplit());
    VELOX_ASSERT_THROW(source->addSplit(logicalSplit()), "Previous split");
    source->addDynamicFilter(
        0, std::make_shared<common::BigintRange>(0, 10, false));
    ASSERT_EQ(state->filterResets, 1);
  }
  ASSERT_EQ(state->cancels, 1);
  ASSERT_EQ(state->destroyed, 1);
}

TEST_F(FileDataSourceTest, takeoverPreservesActiveStatsAndContext) {
  auto state = script({{State::kData, 4, {4}}, {State::kEnd}});
  auto target = makeSource(state);
  auto source = makeSource(state);
  source->addSplit(logicalSplit());
  next(*source);
  target->setFromDataSource(std::move(source));
  ASSERT_EQ(state->context, context_.get());
  ASSERT_EQ(state->destroyed, 0);
  ASSERT_EQ(target->getCompletedRows(), 4);
  ASSERT_EQ(target->getRuntimeStats().at("testScanned").sum, 4);
  ASSERT_EQ(next(*target), nullptr);
  ASSERT_EQ(target->getRuntimeStats().at("testScanned").sum, 4);
  ASSERT_EQ(state->destroyed, 1);
}

TEST_F(FileDataSourceTest, physicalProgressOverflow) {
  auto state = script(
      {{State::kData, std::numeric_limits<uint64_t>::max(), {}},
       {State::kData, 1, {}}});
  auto source = makeSource(state);
  source->addSplit(logicalSplit());
  next(*source);
  VELOX_ASSERT_THROW(next(*source), "Physical row count overflow");
  ASSERT_EQ(state->destroyed, 1);
}

TEST_F(FileDataSourceTest, fileAdapterReadFilterReuseAndTakeover) {
  std::vector<dwio::common::FileFormat> formats{dwio::common::FileFormat::DWRF};
#ifdef VELOX_ENABLE_NIMBLE
  formats.push_back(dwio::common::FileFormat::NIMBLE);
#endif
  std::vector<std::string> strings;
  strings.reserve(100);
  for (int i = 0; i < 100; ++i) {
    strings.push_back(fmt::format("out-of-line string value {}", i));
  }
  const auto input = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto i) { return i; }),
       makeFlatVector<std::string>(strings)});
  const auto fileType = asRowType(input->type());
  const ColumnHandleMap fileAssignments{
      {"c0", regularColumn("c0", BIGINT())},
      {"c1", regularColumn("c1", VARCHAR())}};
  for (auto format : formats) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    auto file = TempFilePath::create();
    if (format == dwio::common::FileFormat::DWRF) {
      writeToFile(file->getPath(), input);
    }
#ifdef VELOX_ENABLE_NIMBLE
    else {
      nimble::Writer writer(
          input->type(),
          std::make_unique<LocalWriteFile>(file->getPath(), false, false),
          *pool_,
          nimble::WriterOptions{});
      writer.write(input);
      writer.close();
    }
#endif
    auto makeFileSource = [&](const ConnectorQueryCtx* context,
                              const std::string& remaining = "") {
      return std::make_unique<HiveDataSource>(
          fileType,
          makeTableHandle(
              {},
              remaining.empty() ? nullptr : parseExpr(remaining, fileType),
              "test",
              fileType),
          fileAssignments,
          &fileFactory_,
          nullptr,
          context,
          hiveConfig_);
    };
    auto split = makeHiveConnectorSplit(file->getPath());
    split->fileFormat = format;
    auto preloadContext = makeContext();
    auto preloaded = makeFileSource(preloadContext.get());
    preloaded->addSplit(split);
    auto source = makeFileSource(context_.get());
    source->setFromDataSource(std::move(preloaded));
    preloadContext.reset();
    int callbacks = 0;
    source->setScanBatchCallback([&](const core::ScanBatchEvent& event) {
      const auto& fileEvent = static_cast<const FileScanBatchEvent&>(event);
      EXPECT_EQ(fileEvent.filePath, file->getPath());
      EXPECT_EQ(fileEvent.fileFormat, format);
      EXPECT_EQ(fileEvent.tableName, "test");
      ++callbacks;
    });
    auto first = next(*source, 10);
    source->fireScanBatchCallback({});
    ASSERT_EQ(callbacks, 1);
    // Load before advancing, as required by DWIO's lazy batch contract.
    first->loadedVector();
    auto second = next(*source, 10);
    second->loadedVector();
    ASSERT_EQ(source->getCompletedRows(), 20);
    const auto stats = source->getRuntimeStats();
    expectStatsEqual(stats, source->getRuntimeStats());
    source->cancel();
    expectStatsEqual(stats, source->getRuntimeStats());
    test::assertEqualVectors(input->slice(0, 10), first);
    test::assertEqualVectors(input->slice(10, 10), second);

    // An empty filtered batch is not EOF. A later batch still returns rows.
    auto filtered = makeFileSource(context_.get(), "c0 % 100 >= 10");
    filtered->addSplit(split);
    ASSERT_EQ(next(*filtered, 10)->size(), 0);
    auto rows = next(*filtered, 10);
    test::assertEqualVectors(input->slice(10, 10), rows);
    int count = rows->size();
    while (auto batch = next(*filtered, 10)) {
      count += batch->size();
    }
    ASSERT_EQ(count, 90);
    ASSERT_EQ(filtered->getCompletedRows(), 100);
    const auto finishedStats = filtered->getRuntimeStats();
    filtered->cancel();
    expectStatsEqual(finishedStats, filtered->getRuntimeStats());

    common::SubfieldFilters filters;
    filters.emplace(
        common::Subfield("c0"),
        std::make_unique<common::BigintRange>(200, 300, false));
    auto skipped = std::make_unique<HiveDataSource>(
        fileType,
        makeTableHandle(std::move(filters), nullptr, "test", fileType),
        fileAssignments,
        &fileFactory_,
        nullptr,
        context_.get(),
        hiveConfig_);
    skipped->addSplit(split);
    while (auto batch = next(*skipped, 10)) {
      ASSERT_EQ(batch->size(), 0);
    }
  }
}

TEST_F(FileDataSourceTest, tableScanBlockedProgressAndCachedOutput) {
  for (bool cancelWhileBlocked : {false, true}) {
    SCOPED_TRACE(cancelWhileBlocked);
    auto state = script(
        {{State::kData, 2, {}},
         {State::kBlocked, 3, {}, true},
         {State::kData, 0, {1, 2, 3}},
         {State::kEnd}});
    ConnectorRegistry::global().erase(kHiveConnectorId);
    ConnectorRegistry::global().insert(
        kHiveConnectorId, std::make_shared<TestConnector>(config_, state));
    auto plan = PlanBuilder(pool_.get()).tableScan(type_).planFragment();
    auto task = exec::Task::create(
        "logical-reader",
        std::move(plan),
        0,
        core::QueryCtx::create(),
        exec::Task::ExecutionMode::kSerial);
    task->addSplit("0", exec::Split(logicalSplit()));
    task->noMoreSplits("0");
    ContinueFuture future = ContinueFuture::makeEmpty();
    ASSERT_EQ(task->next(&future), nullptr);
    ASSERT_TRUE(future.valid());
    ASSERT_FALSE(future.isReady());
    const auto stats = task->taskStats();
    ASSERT_EQ(
        stats.pipelineStats.at(0).operatorStats.at(0).rawInputPositions, 5);
    if (cancelWhileBlocked) {
      task->requestCancel().wait();
    } else {
      state->promise.setValue();
      std::move(future).get();
      auto output = task->next();
      expectValues(output, {1, 2, 3});
      ASSERT_EQ(task->next(), nullptr);
    }
    ASSERT_EQ(state->cancels, 1);
    ASSERT_EQ(state->destroyed, 1);
    ASSERT_EQ(
        task->taskStats()
            .pipelineStats.at(0)
            .operatorStats.at(0)
            .runtimeStats.at("testScanned")
            .sum,
        5);
    ASSERT_EQ(
        task->taskStats()
            .pipelineStats.at(0)
            .operatorStats.at(0)
            .rawInputPositions,
        5);
  }
}

TEST_F(FileDataSourceTest, fileAdapterPrepareFailure) {
  HiveDataSource source(
      type_,
      makeTableHandle({}, nullptr, "test", type_),
      assignments_,
      &fileFactory_,
      nullptr,
      context_.get(),
      hiveConfig_);
  auto missing =
      makeHiveConnectorSplit("/no-such-directory/FileDataSourceTest");
  ASSERT_THROW(source.addSplit(missing), VeloxException);
  source.cancel();
  auto file = TempFilePath::create();
  writeToFile(
      file->getPath(), makeRowVector({makeFlatVector<int64_t>({1, 2})}));
  source.addSplit(makeHiveConnectorSplit(file->getPath()));
  expectValues(next(source), {1, 2});
  ASSERT_EQ(next(source), nullptr);
}
} // namespace
} // namespace facebook::velox::connector::hive
