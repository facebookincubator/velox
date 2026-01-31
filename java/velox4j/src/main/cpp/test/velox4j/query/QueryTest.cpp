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
#include <gtest/gtest.h>
#include <velox/connectors/fuzzer/tests/FuzzerConnectorTestBase.h>
#include <velox/exec/tests/utils/HiveConnectorTestBase.h>
#include <velox/exec/tests/utils/PlanBuilder.h>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "velox4j/conf/Config.h"
#include "velox4j/iterator/UpIterator.h"
#include "velox4j/memory/AllocationListener.h"
#include "velox4j/memory/MemoryManager.h"
#include "velox4j/query/Query.h"
#include "velox4j/query/QueryExecutor.h"
#include "velox4j/test/Init.h"

namespace facebook::velox4j::test {
using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class QueryTest : public testing::Test, public velox::test::VectorTestBase {
 protected:
  const std::string kFuzzerConnectorId = "test-fuzzer";

  static void SetUpTestCase() {
    testingEnsureInitializedForSpark();
  }

  void SetUp() override {
    Test::SetUp();
    connector::registerConnectorFactory(
        std::make_shared<connector::fuzzer::FuzzerConnectorFactory>());
    std::shared_ptr<const config::ConfigBase> config;
    auto fuzzerConnector =
        connector::getConnectorFactory(
            connector::fuzzer::FuzzerConnectorFactory::kFuzzerConnectorName)
            ->newConnector(kFuzzerConnectorId, config);
    registerConnector(fuzzerConnector);
  }

  void TearDown() override {
    connector::unregisterConnector(kFuzzerConnectorId);
    connector::unregisterConnectorFactory(
        connector::fuzzer::FuzzerConnectorFactory::kFuzzerConnectorName);
    Test::TearDown();
  }

  QueryTest() {
    memoryManager_ =
        std::make_shared<MemoryManager>(AllocationListener::noop());
  }

  std::vector<RowVectorPtr> collect(UpIterator& itr) {
    std::vector<RowVectorPtr> out{};
    while (true) {
      const UpIterator::State state = itr.advance();
      switch (state) {
        case UpIterator::State::AVAILABLE: {
          const auto rv = itr.get();
          out.push_back(std::dynamic_pointer_cast<RowVector>(
              RowVector::loadedVectorShared(rv)));
          break;
        }
        case UpIterator::State::BLOCKED:
          continue;
        case UpIterator::State::FINISHED:
          goto OUT;
      }
    }
  OUT:
    return out;
  }

  exec::Split makeFuzzerSplit(size_t numRows) const {
    return exec::Split(
        std::make_shared<connector::fuzzer::FuzzerConnectorSplit>(
            kFuzzerConnectorId, numRows));
  }

  std::vector<exec::Split> makeFuzzerSplits(
      size_t rowsPerSplit,
      size_t numSplits) const {
    std::vector<exec::Split> splits;
    splits.reserve(numSplits);

    for (size_t i = 0; i < numSplits; ++i) {
      splits.emplace_back(makeFuzzerSplit(rowsPerSplit));
    }
    return splits;
  }

  std::shared_ptr<connector::fuzzer::FuzzerTableHandle> makeFuzzerTableHandle(
      size_t fuzzerSeed = 0) const {
    return std::make_shared<connector::fuzzer::FuzzerTableHandle>(
        kFuzzerConnectorId, VectorFuzzer::Options{}, fuzzerSeed);
  }

  std::shared_ptr<MemoryManager> memoryManager_;
};

TEST_F(QueryTest, fuzzer) {
  const size_t rowsPerSplit = 100;
  const size_t numSplits = 10;

  auto type = VectorFuzzer({}, pool()).randRowType();

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(type)
                  .tableHandle(makeFuzzerTableHandle())
                  .endTableScan()
                  .planNode();

  auto query = std::make_shared<const Query>(
      plan,
      std::make_shared<const ConfigArray>(
          std::vector<std::pair<std::string, std::string>>({})),
      std::make_shared<const ConnectorConfigArray>(
          std::vector<
              std::pair<std::string, std::shared_ptr<const ConfigArray>>>({})));
  auto executor = std::make_shared<QueryExecutor>(memoryManager_.get(), query);
  auto serialTask = executor->execute();

  const auto fuzzerSplits = makeFuzzerSplits(rowsPerSplit, numSplits);
  for (const auto& s : fuzzerSplits) {
    serialTask->addSplit(plan->id(), s.groupId, s.connectorSplit);
  }
  serialTask->noMoreSplits(plan->id());

  auto vectors = collect(*serialTask);

  size_t actualRows = 0;
  for (const auto& v : vectors) {
    actualRows += v->size();
  }
  ASSERT_EQ(actualRows, rowsPerSplit * numSplits);
}
} // namespace facebook::velox4j::test
