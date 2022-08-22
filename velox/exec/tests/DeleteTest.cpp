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

#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

namespace {

const std::string kMockConnectorId = "mock";
const vector_size_t kMaxBatchSize = 1024;

class MockTableHandle : public connector::ConnectorTableHandle {
 public:
  MockTableHandle() : connector::ConnectorTableHandle(kMockConnectorId) {}

  std::string toString() const override {
    VELOX_NYI();
  }
};

class MockSplit : public connector::ConnectorSplit {
 public:
  MockSplit(const RowVectorPtr& data)
      : connector::ConnectorSplit(kMockConnectorId), data_(data) {}

  RowVectorPtr data() const {
    return data_;
  }

 private:
  RowVectorPtr data_;
};

std::vector<std::shared_ptr<connector::ConnectorSplit>> makeMockConnectorSplits(
    const std::vector<RowVectorPtr>& inputs) {
  std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
  splits.resize(inputs.size());
  std::transform(
      inputs.begin(), inputs.end(), splits.begin(), [](RowVectorPtr input) {
        return std::make_shared<MockSplit>(input);
      });
  return splits;
}

std::vector<exec::Split> makeSplits(
    const std::vector<std::shared_ptr<connector::ConnectorSplit>>&
        connectorSplits) {
  std::vector<exec::Split> splits;
  splits.reserve(connectorSplits.size());
  for (const auto& connectorSplit : connectorSplits) {
    splits.emplace_back(exec::Split(folly::copy(connectorSplit), -1));
  }
  return splits;
}

class MockDataSource : public connector::UpdatableDataSource {
 public:
  void addSplit(std::shared_ptr<connector::ConnectorSplit> split) override {
    auto mockSplit = std::dynamic_pointer_cast<MockSplit>(split);
    VELOX_CHECK_NOT_NULL(mockSplit);
    VELOX_CHECK_NULL(split_);
    split_ = std::move(mockSplit);
    current_ = 0;
    rowsPerBatch_ = std::min(kMaxBatchSize, split_->data()->size() - current_);
  }

  std::optional<RowVectorPtr> next(uint64_t size, ContinueFuture& future)
      override {
    if (split_) {
      const auto& data = split_->data();
      RowVectorPtr output = std::static_pointer_cast<RowVector>(
          data->slice(current_, rowsPerBatch_));
      current_ += rowsPerBatch_;
      rowsPerBatch_ = std::min(kMaxBatchSize, data->size() - current_);
      if (rowsPerBatch_ <= 0) {
        split_ = nullptr;
      }
      return output;
    }
    return nullptr;
  }

  void deleteRows(const VectorPtr& rowIds) override {}

  void addDynamicFilter(
      column_index_t /* outputChannel */,
      const std::shared_ptr<common::Filter>& /* filter */) override {
    VELOX_NYI();
  }

  uint64_t getCompletedBytes() override {
    return 0;
  }

  uint64_t getCompletedRows() override {
    return 0;
  }

  std::unordered_map<std::string, RuntimeCounter> runtimeStats() override {
    return {};
  }

 private:
  std::shared_ptr<MockSplit> split_;
  vector_size_t current_;
  vector_size_t rowsPerBatch_;
};

class MockConnector : public connector::Connector {
 public:
  MockConnector(const std::string& id, std::shared_ptr<const Config> properties)
      : connector::Connector(id, std::move(properties)) {}

  std::shared_ptr<connector::DataSource> createDataSource(
      const RowTypePtr& /* outputType */,
      const std::shared_ptr<connector::ConnectorTableHandle>& /* tableHandle */,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& /* columnHandles */,
      connector::ConnectorQueryCtx* connectorQueryCtx) override {
    return std::make_shared<MockDataSource>();
  }

  std::shared_ptr<connector::DataSink> createDataSink(
      RowTypePtr /* inputType */,
      std::shared_ptr<connector::ConnectorInsertTableHandle>
      /* connectorInsertTableHandle */,
      connector::ConnectorQueryCtx* /* connectorQueryCtx */) override {
    VELOX_NYI();
  }
};

class MockConnectorFactory : public connector::ConnectorFactory {
 public:
  static constexpr const char* kMockConnectorName = "mock";

  MockConnectorFactory() : connector::ConnectorFactory(kMockConnectorName) {}

  std::shared_ptr<connector::Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const Config> properties,
      folly::Executor* /* executor */) override {
    return std::make_shared<MockConnector>(id, std::move(properties));
  }
};

core::PlanNodeId getOnlyLeafPlanNodeId(const core::PlanNodePtr& root) {
  const auto& sources = root->sources();
  if (sources.empty()) {
    return root->id();
  }
  VELOX_CHECK_EQ(1, sources.size());
  return getOnlyLeafPlanNodeId(sources[0]);
}
} // namespace

class DeleteTest : public OperatorTestBase {
 public:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    connector::unregisterConnector(kMockConnectorId);
    connector::registerConnectorFactory(
        std::make_shared<MockConnectorFactory>());
    auto testConnector =
        connector::getConnectorFactory(MockConnectorFactory::kMockConnectorName)
            ->newConnector(kMockConnectorId, nullptr, nullptr);
    connector::registerConnector(testConnector);
  }

  static void TearDownTestCase() {
    connector::unregisterConnector(kMockConnectorId);
    OperatorTestBase::TearDownTestCase();
  }

 protected:
  void verifyDelete(
      const std::shared_ptr<const core::PlanNode>& plan,
      const int numThreads = 1) {
    verifyDelete(plan, 0, 0, numThreads);
  }

  void verifyDelete(
      const std::shared_ptr<const core::PlanNode>& plan,
      const int numSplits,
      const int rowsPerSplit,
      const int numThreads = 1) {
    verifyDelete(
        plan,
        makeSplits(makeMockConnectorSplits(
            makeInputs(rowType_, numSplits, rowsPerSplit))),
        rowsPerSplit,
        numThreads);
  }

  void verifyDelete(
      const std::shared_ptr<const core::PlanNode>& plan,
      std::vector<exec::Split>&& splits,
      const int rowsPerSplit,
      const int numThreads) {
    VELOX_CHECK_GE(
        numThreads,
        1,
        "numThreads parameter must be greater then or equal to 1");

    bool emptySplits = splits.size() == 0;
    const auto splitNodeId = getOnlyLeafPlanNodeId(plan);

    CursorParameters params;
    params.planNode = plan;
    params.maxDrivers = numThreads;

    bool noMoreSplits = false;
    auto result = readCursor(params, [&](exec::Task* task) {
      if (noMoreSplits) {
        return;
      }
      if (!emptySplits) {
        for (auto& split : splits) {
          task->addSplit(splitNodeId, std::move(split));
        }
      }
      task->noMoreSplits(splitNodeId);
      noMoreSplits = true;
    });

    // assert output layout
    auto numColumns = result.second[0]->childrenSize();
    ASSERT_EQ(numColumns, 3);

    // assert output
    ASSERT_EQ(numThreads, result.second.size());
    uint64_t actualNumDeleted = 0;
    for (auto const& output : result.second) {
      ASSERT_EQ(1, output->size());
      auto column = output->childAt(0)->asFlatVector<int64_t>();
      if (emptySplits) {
        ASSERT_EQ(column->valueAt(0), 0);
      } else {
        actualNumDeleted += column->valueAt(0);
        ASSERT_TRUE(actualNumDeleted % rowsPerSplit == 0);
      }
    }
    ASSERT_EQ(actualNumDeleted, rowsPerSplit * splits.size());
  }

  std::vector<RowVectorPtr> makeInputs(
      const RowTypePtr& rowType,
      int32_t numVectors,
      int32_t rowsPerVector) {
    std::vector<RowVectorPtr> vectors;
    VectorFuzzer::Options opts;
    opts.vectorSize = rowsPerVector;
    VectorFuzzer fuzzer(opts, pool(), folly::Random::rand32());
    for (int32_t i = 0; i < numVectors; ++i) {
      const auto& vector = fuzzer.fuzzRow(rowType);
      vectors.push_back(vector);
    }
    return vectors;
  }

  RowTypePtr rowType_{ROW({"c0", "p1"}, {BIGINT(), INTEGER()})};
};

TEST_F(DeleteTest, emptySourceSplit) {
  auto tableHandle = std::make_shared<MockTableHandle>();
  auto plan = PlanBuilder()
                  .tableScan(rowType_, tableHandle, {})
                  .project({"c0"})
                  .tableDelete("c0")
                  .planNode();
  verifyDelete(plan);
  verifyDelete(plan, 8);
}

TEST_F(DeleteTest, multiBatch) {
  auto tableHandle = std::make_shared<MockTableHandle>();
  auto plan = PlanBuilder()
                  .tableScan(rowType_, tableHandle, {})
                  .project({"c0"})
                  .tableDelete("c0")
                  .planNode();
  verifyDelete(plan, 1, kMaxBatchSize - 1);
  vector_size_t batchSize = 5 * kMaxBatchSize - 1;
  verifyDelete(plan, 1, batchSize);
}

TEST_F(DeleteTest, multiSplit) {
  auto tableHandle = std::make_shared<MockTableHandle>();
  auto plan = PlanBuilder()
                  .tableScan(rowType_, tableHandle, {})
                  .project({"c0"})
                  .tableDelete("c0")
                  .planNode();
  verifyDelete(plan, 11, kMaxBatchSize + 3, 1);
}

TEST_F(DeleteTest, multiThread) {
  auto tableHandle = std::make_shared<MockTableHandle>();
  auto plan = PlanBuilder()
                  .tableScan(rowType_, tableHandle, {})
                  .project({"c0"})
                  .tableDelete("c0")
                  .planNode();
  vector_size_t batchSize = 7 * kMaxBatchSize - 1;
  verifyDelete(plan, 1, batchSize, 32);
  verifyDelete(plan, 32, batchSize, 32);
  verifyDelete(plan, 129, batchSize, 32);
}