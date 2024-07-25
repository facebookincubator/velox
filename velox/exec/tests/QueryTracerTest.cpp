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

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>

#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/ArbitratorTestUtil.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/trace/QueryTraceDataReader.h"
#include "velox/exec/trace/QueryTraceMetadataReader.h"
#include "velox/exec/trace/QueryTraceRestore.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::exec::test {
class QueryTracerTest : public HiveConnectorTestBase {
 public:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
    HiveConnectorTestBase::SetUpTestCase();
  }

 protected:
  void SetUp() override {
    if (!isRegisteredVectorSerde()) {
      serializer::presto::PrestoVectorSerde::registerVectorSerde();
    }
    filesystems::registerLocalFileSystem();
    HiveConnectorTestBase::SetUp();
  }

  static VectorFuzzer::Options getFuzzerOptions() {
    return VectorFuzzer::Options{
        .vectorSize = 16,
        .nullRatio = 0.2,
        .stringLength = 1024,
        .stringVariableLength = false,
        .allowLazyVector = false,
    };
  }

  QueryTracerTest() : vectorFuzzer_{getFuzzerOptions(), pool_.get()} {
    filesystems::registerLocalFileSystem();
  }

  RowTypePtr generateTypes(size_t numColumns) {
    std::vector<std::string> names;
    names.reserve(numColumns);
    std::vector<TypePtr> types;
    types.reserve(numColumns);
    for (auto i = 0; i < numColumns; ++i) {
      names.push_back(fmt::format("c{}", i));
      types.push_back(vectorFuzzer_.randType((2)));
    }
    return ROW(std::move(names), std::move(types));
    ;
  }

  VectorFuzzer vectorFuzzer_;
};

TEST_F(QueryTracerTest, plan) {
  const auto rowType = generateTypes(5);
  std::vector<RowVectorPtr> rows;
  constexpr auto numBatch = 1;
  rows.reserve(numBatch);
  for (auto i = 0; i < numBatch; ++i) {
    rows.push_back(vectorFuzzer_.fuzzInputRow(rowType));
  }

  const auto outputDir = TempDirectoryPath::create();
  const auto planNode =
      PlanBuilder().values(rows).tableWrite(outputDir->getPath()).planNode();
  std::shared_ptr<Task> task;
  AssertQueryBuilder(planNode)
      .maxDrivers(1)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, outputDir->getPath())
      .config(core::QueryConfig::kQueryTraceNodes, "1")
      .copyResults(pool(), task);

  const auto dataPath =
      fmt::format("{}/{}", outputDir->getPath(), "1/0/0/data");
  const auto reader = QueryTraceDataReader(dataPath);

  RowVectorPtr actual;
  reader.read(actual);
  RowVectorPtr tmp;
  while (reader.read(tmp)) {
    actual->append(tmp.get());
  }

  RowVectorPtr expected = rows[0];
  for (int i = 1; i < numBatch; ++i) {
    expected->append(rows[i].get());
  }

  const auto sz = actual->size();
  ASSERT_EQ(sz, expected->size());
  for (auto i = 0; i < sz; ++i) {
    actual->compare(expected.get(), i, i, {.nullsFirst = true});
  }

  const auto metaPath = fmt::format("{}/metadata", outputDir->getPath());
  const auto tracer = QueryTraceMetadataReader(metaPath);
  std::unordered_map<std::string, std::string> actualQueryConfigs;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      actualConnectorConfigs;
  core::PlanNodePtr queryPlan = nullptr;
  tracer.read(actualQueryConfigs, actualConnectorConfigs, queryPlan);
  auto targetPlanNode = findPlanNodeById(queryPlan, "1");

  auto restoredPlanNode =
      PlanBuilder()
          .traceScan(dataPath)
          .addNode(addTableWriter(
              std::dynamic_pointer_cast<const core::TableWriteNode>(
                  targetPlanNode)))
          .planNode();
  AssertQueryBuilder(restoredPlanNode)
      .maxDrivers(1)
      .configs(actualQueryConfigs)
      .connectorSessionProperties(actualConnectorConfigs)
      .config(core::QueryConfig::kQueryTraceEnabled, false)
      .copyResults(pool(), task);
}
} // namespace facebook::velox::exec::test
