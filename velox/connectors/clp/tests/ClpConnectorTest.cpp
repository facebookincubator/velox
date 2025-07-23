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

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/clp/ClpColumnHandle.h"
#include "velox/connectors/clp/ClpConnector.h"
#include "velox/connectors/clp/ClpConnectorSplit.h"
#include "velox/connectors/clp/ClpTableHandle.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"

namespace {

using namespace facebook::velox;
using namespace facebook::velox::connector::clp;

using facebook::velox::exec::test::PlanBuilder;

// Epoch seconds and nanoseconds for the timestamp "2025-04-30T08:50:05Z"
constexpr int64_t kTestTimestampSeconds{1746003005};
constexpr uint64_t kTestTimestampNanoseconds{0ULL};

class ClpConnectorTest : public exec::test::OperatorTestBase {
 public:
  const std::string kClpConnectorId = "test-clp";

  void SetUp() override {
    OperatorTestBase::SetUp();
    connector::registerConnectorFactory(
        std::make_shared<connector::clp::ClpConnectorFactory>());
    auto clpConnector =
        connector::getConnectorFactory(
            connector::clp::ClpConnectorFactory::kClpConnectorName)
            ->newConnector(
                kClpConnectorId,
                std::make_shared<config::ConfigBase>(
                    std::unordered_map<std::string, std::string>{
                        {"clp.split-source", "local"}}));
    connector::registerConnector(clpConnector);
  }

  void TearDown() override {
    connector::unregisterConnector(kClpConnectorId);
    connector::unregisterConnectorFactory(
        connector::clp::ClpConnectorFactory::kClpConnectorName);
    OperatorTestBase::TearDown();
  }

  exec::Split makeClpSplit(
      const std::string& splitPath,
      std::shared_ptr<std::string> kqlQuery) {
    return exec::Split(std::make_shared<ClpConnectorSplit>(
        kClpConnectorId, splitPath, kqlQuery));
  }

  RowVectorPtr getResults(
      const core::PlanNodePtr& planNode,
      std::vector<exec::Split>&& splits) {
    return exec::test::AssertQueryBuilder(planNode)
        .splits(std::move(splits))
        .copyResults(pool());
  }

  static std::string getExampleFilePath(const std::string& filePath) {
    std::string current_path = fs::current_path().string();
    return current_path + "/examples/" + filePath;
  }
};

TEST_F(ClpConnectorTest, test1NoPushdown) {
  const std::shared_ptr<std::string> kqlQuery = nullptr;
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(
                      ROW({"requestId", "userId", "method"},
                          {VARCHAR(), VARCHAR(), VARCHAR()}))
                  .tableHandle(std::make_shared<ClpTableHandle>(
                      kClpConnectorId, "test_1"))
                  .assignments({
                      {"requestId",
                       std::make_shared<ClpColumnHandle>(
                           "requestId", "requestId", VARCHAR(), true)},
                      {"userId",
                       std::make_shared<ClpColumnHandle>(
                           "userId", "userId", VARCHAR(), true)},
                      {"method",
                       std::make_shared<ClpColumnHandle>(
                           "method", "method", VARCHAR(), true)},
                  })
                  .endTableScan()
                  .filter("method = 'GET'")
                  .planNode();

  auto output = getResults(
      plan, {makeClpSplit(getExampleFilePath("test_1.clps"), kqlQuery)});
  auto expected = makeRowVector(
      {// requestId
       makeFlatVector<StringView>(
           {"req-100", "req-105", "req-107", "req-109", "req-102"}),
       // userId
       makeNullableFlatVector<StringView>(
           {"user201", "user204", "user202", "user203", std::nullopt}),
       // method
       makeFlatVector<StringView>({
           "GET",
           "GET",
           "GET",
           "GET",
           "GET",
       })});
  test::assertEqualVectors(expected, output);
}

TEST_F(ClpConnectorTest, test1Pushdown) {
  auto kqlQuery =
      std::make_shared<std::string>("method: \"POST\" AND status: 200");
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(
                      ROW({"requestId", "userId", "path"},
                          {VARCHAR(), VARCHAR(), VARCHAR()}))
                  .tableHandle(std::make_shared<ClpTableHandle>(
                      kClpConnectorId, "test_1"))
                  .assignments({
                      {"requestId",
                       std::make_shared<ClpColumnHandle>(
                           "requestId", "requestId", VARCHAR(), true)},
                      {"userId",
                       std::make_shared<ClpColumnHandle>(
                           "userId", "userId", VARCHAR(), true)},
                      {"path",
                       std::make_shared<ClpColumnHandle>(
                           "path", "path", VARCHAR(), true)},
                  })
                  .endTableScan()
                  .planNode();

  auto output = getResults(
      plan, {makeClpSplit(getExampleFilePath("test_1.clps"), kqlQuery)});
  auto expected =
      makeRowVector({// requestId
                     makeFlatVector<StringView>({"req-106"}),
                     // userId
                     makeNullableFlatVector<StringView>({std::nullopt}),
                     // path
                     makeFlatVector<StringView>({"/auth/login"})});
  test::assertEqualVectors(expected, output);
}

TEST_F(ClpConnectorTest, test2NoPushdown) {
  const std::shared_ptr<std::string> kqlQuery = nullptr;
  auto plan =
      PlanBuilder(pool_.get())
          .startTableScan()
          .outputType(
              ROW({"timestamp", "event"},
                  {TIMESTAMP(),
                   ROW({"type", "subtype", "severity"},
                       {VARCHAR(), VARCHAR(), VARCHAR()})}))
          .tableHandle(
              std::make_shared<ClpTableHandle>(kClpConnectorId, "test_2"))
          .assignments(
              {{"timestamp",
                std::make_shared<ClpColumnHandle>(
                    "timestamp", "timestamp", TIMESTAMP(), true)},
               {"event",
                std::make_shared<ClpColumnHandle>(
                    "event",
                    "event",
                    ROW({"type", "subtype", "severity"},
                        {VARCHAR(), VARCHAR(), VARCHAR()}),
                    true)}})
          .endTableScan()
          .filter(
              "event.severity IN ('WARNING', 'ERROR') AND "
              "((event.type = 'network' AND event.subtype = 'connection') OR "
              "(event.type = 'storage' AND event.subtype LIKE 'disk_usage%'))")
          .planNode();

  auto output = getResults(
      plan, {makeClpSplit(getExampleFilePath("test_2.clps"), kqlQuery)});
  auto expected =
      makeRowVector({// timestamp
                     makeFlatVector<Timestamp>({Timestamp(
                         kTestTimestampSeconds, kTestTimestampNanoseconds)}),
                     // event
                     makeRowVector({
                         // event.type
                         makeFlatVector<StringView>({"storage"}),
                         // event.subtype
                         makeFlatVector<StringView>({"disk_usage"}),
                         // event.severity
                         makeFlatVector<StringView>({"WARNING"}),
                     })});
  test::assertEqualVectors(expected, output);
}

TEST_F(ClpConnectorTest, test2Pushdown) {
  auto kqlQuery = std::make_shared<std::string>(
      "(event.severity: \"WARNING\" OR event.severity: \"ERROR\") AND "
      "((event.type: \"network\" AND event.subtype: \"connection\") OR "
      "(event.type: \"storage\" AND event.subtype: \"disk*\"))");
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(
                      ROW({"timestamp", "event"},
                          {TIMESTAMP(),
                           ROW({"type", "subtype", "severity"},
                               {VARCHAR(), VARCHAR(), VARCHAR()})}))
                  .tableHandle(std::make_shared<ClpTableHandle>(
                      kClpConnectorId, "test_2"))
                  .assignments(
                      {{"timestamp",
                        std::make_shared<ClpColumnHandle>(
                            "timestamp", "timestamp", TIMESTAMP(), true)},
                       {"event",
                        std::make_shared<ClpColumnHandle>(
                            "event",
                            "event",
                            ROW({"type", "subtype", "severity"},
                                {VARCHAR(), VARCHAR(), VARCHAR()}),
                            true)}})
                  .endTableScan()
                  .planNode();

  auto output = getResults(
      plan, {makeClpSplit(getExampleFilePath("test_2.clps"), kqlQuery)});
  auto expected =
      makeRowVector({// timestamp
                     makeFlatVector<Timestamp>({Timestamp(
                         kTestTimestampSeconds, kTestTimestampNanoseconds)}),
                     // event
                     makeRowVector({
                         // event.type
                         makeFlatVector<StringView>({"storage"}),
                         // event.subtype
                         makeFlatVector<StringView>({"disk_usage"}),
                         // event.severity
                         makeFlatVector<StringView>({"WARNING"}),
                     })});
  test::assertEqualVectors(expected, output);
}

TEST_F(ClpConnectorTest, test2Hybrid) {
  auto kqlQuery = std::make_shared<std::string>(
      "((event.type: \"network\" AND event.subtype: \"connection\") OR "
      "(event.type: \"storage\" AND event.subtype: \"disk*\"))");
  auto plan =
      PlanBuilder(pool_.get())
          .startTableScan()
          .outputType(
              ROW({"timestamp", "event"},
                  {TIMESTAMP(),
                   ROW({"type", "subtype", "severity", "tags"},
                       {VARCHAR(), VARCHAR(), VARCHAR(), ARRAY(VARCHAR())})}))
          .tableHandle(
              std::make_shared<ClpTableHandle>(kClpConnectorId, "test_2"))
          .assignments(
              {{"timestamp",
                std::make_shared<ClpColumnHandle>(
                    "timestamp", "timestamp", TIMESTAMP(), true)},
               {"event",
                std::make_shared<ClpColumnHandle>(
                    "event",
                    "event",
                    ROW({"type", "subtype", "severity", "tags"},
                        {VARCHAR(), VARCHAR(), VARCHAR(), ARRAY(VARCHAR())}),
                    true)}})
          .endTableScan()
          .filter("upper(event.severity) IN ('WARNING', 'ERROR')")
          .planNode();

  auto output = getResults(
      plan, {makeClpSplit(getExampleFilePath("test_2.clps"), kqlQuery)});
  auto expected = makeRowVector(
      {// timestamp
       makeFlatVector<Timestamp>(
           {Timestamp(kTestTimestampSeconds, kTestTimestampNanoseconds)}),
       // event
       makeRowVector({// event.type
                      makeFlatVector<StringView>({"storage"}),
                      // event.subtype
                      makeFlatVector<StringView>({"disk_usage"}),
                      // event.severity
                      makeFlatVector<StringView>({"WARNING"}),
                      // event.tags
                      makeArrayVector<StringView>(
                          {{"\"filesystem\"", "\"monitoring\""}})})

      });
  test::assertEqualVectors(expected, output);
}

TEST_F(ClpConnectorTest, test3TimestampMarshalling) {
  const std::shared_ptr<std::string> kqlQuery = nullptr;
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(ROW({"timestamp"}, {TIMESTAMP()}))
                  .tableHandle(std::make_shared<ClpTableHandle>(
                      kClpConnectorId, "test_3"))
                  .assignments(
                      {{"timestamp",
                        std::make_shared<ClpColumnHandle>(
                            "timestamp", "timestamp", TIMESTAMP(), true)}})
                  .endTableScan()
                  .planNode();

  auto output = getResults(
      plan, {makeClpSplit(getExampleFilePath("test_3.clps"), kqlQuery)});
  auto expected = makeRowVector({
      // timestamp
      makeFlatVector<Timestamp>(
          {Timestamp(kTestTimestampSeconds, kTestTimestampNanoseconds),
           Timestamp(kTestTimestampSeconds, kTestTimestampNanoseconds),
           Timestamp(kTestTimestampSeconds, kTestTimestampNanoseconds),
           Timestamp(kTestTimestampSeconds, kTestTimestampNanoseconds)}),
  });
  test::assertEqualVectors(expected, output);
}

} // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
