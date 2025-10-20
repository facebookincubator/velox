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
#include <folly/json.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/fuzzer/LocalRunnerService.h"
#include "velox/exec/fuzzer/if/gen-cpp2/LocalRunnerService.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::runner;
using namespace facebook::velox::test;

namespace facebook::velox::fuzzer::test {
class LocalRunnerServiceTest : public functions::test::FunctionBaseTest {
 protected:
  void SetUp() override {
    Type::registerSerDe();
    core::PlanNode::registerSerDe();
    core::ITypedExpr::registerSerDe();
    functions::prestosql::registerAllScalarFunctions();
    functions::prestosql::registerInternalFunctions();

    createTestData();
  }

  void createTestData() {
    // Create test vectors for different data types
    auto rowType = ROW({
        {"bool_col", BOOLEAN()},
        {"int_col", INTEGER()},
        {"bigint_col", BIGINT()},
        {"double_col", DOUBLE()},
        {"varchar_col", VARCHAR()},
        {"timestamp_col", TIMESTAMP()},
        {"array_col", ARRAY(ARRAY(INTEGER()))},
    });

    testRowVector_ = makeRowVector(
        {"bool_col",
         "int_col",
         "bigint_col",
         "double_col",
         "varchar_col",
         "timestamp_col",
         "array_col"},
        {
            makeFlatVector<bool>(
                10,
                [](auto row) { return row % 2 == 0; },
                [](auto row) { return row % 3 == 0; }),
            makeFlatVector<int32_t>(
                10,
                [](auto row) { return row; },
                [](auto row) { return row % 3 == 0; }),
            makeFlatVector<int64_t>(
                10,
                [](auto row) { return row; },
                [](auto row) { return row % 3 == 0; }),
            makeFlatVector<double>(
                10,
                [](auto row) { return row * 1.1; },
                [](auto row) { return row % 3 == 0; }),
            makeFlatVector<std::string>(
                10,
                [](auto row) { return fmt::format("str_{}", row); },
                [](auto row) { return row % 3 == 0; }),
            makeFlatVector<facebook::velox::Timestamp>(
                10,
                [](auto row) { return facebook::velox::Timestamp(row, 0); },
                [](auto row) { return row % 3 == 0; }),
            makeNestedArrayVectorFromJson<int64_t>(
                {"[[1, 2]]",
                 "[[3]]",
                 "[[4]]",
                 "[[5, 6]]",
                 "[[7]]",
                 "[[8]]",
                 "[[9]]",
                 "[[10]]",
                 "[[11]]",
                 "[[12]]"}),
        });

    testRowVectorWrapped_ = makeRowVector(
        {"bool_col",
         "int_col",
         "bigint_col",
         "double_col",
         "varchar_col",
         "timestamp_col",
         "array_col"},
        {
            makeFlatVector<bool>(
                5,
                [](auto row) { return row % 2 == 0; },
                [](auto row) { return row % 3 == 0; }),
            wrapInDictionary(
                makeIndices(5, [](auto row) { return (row * 17 + 3) % 10; }),
                5,
                makeFlatVector<int32_t>(
                    10,
                    [](auto row) { return row; },
                    [](auto row) { return row % 3 == 0; })),
            BaseVector::wrapInConstant(
                5,
                0,
                makeFlatVector<int64_t>(
                    10,
                    [](auto row) { return row; },
                    [](auto row) { return row % 3 == 0; })),
            makeFlatVector<double>(
                5,
                [](auto row) { return row * 1.1; },
                [](auto row) { return row % 3 == 0; }),
            makeFlatVector<std::string>(
                5,
                [](auto row) { return fmt::format("str_{}", row); },
                [](auto row) { return row % 3 == 0; }),
            makeFlatVector<facebook::velox::Timestamp>(
                5,
                [](auto row) { return facebook::velox::Timestamp(row, 0); },
                [](auto row) { return row % 3 == 0; }),
            makeNestedArrayVectorFromJson<int64_t>(
                {"[[1, 2]]",
                 "[[3]]",
                 "[[4]]",
                 "[[5, 6]]",
                 "[[7]]",
                 "[[8]]",
                 "[[9]]",
                 "[[10]]",
                 "[[11]]",
                 "[[12]]"}),
        });
  }

  RowVectorPtr testRowVector_;
  RowVectorPtr testRowVectorWrapped_;
};

TEST_F(LocalRunnerServiceTest, ConvertToBatches) {
  auto queryCtx = core::QueryCtx::create();
  core::ExecCtx execCtx(rootPool_.get(), queryCtx.get());
  exec::EvalCtx evalCtx(&execCtx);
  auto result =
      facebook::velox::runner::convertToBatches({testRowVector_}, evalCtx);

  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].columnNames()->size(), 7);
  ASSERT_EQ(result[0].columnTypes()->size(), 7);
  ASSERT_EQ(result[0].numRows(), 10);
  ASSERT_EQ(result[0].columns()->size(), 7);
  ASSERT_EQ(result[0].columns()[0].rows()->size(), 10);
  ASSERT_EQ(result[0].columns()[1].rows()->size(), 10);
  ASSERT_EQ(result[0].columns()[2].rows()->size(), 10);
  ASSERT_EQ(result[0].columns()[3].rows()->size(), 10);
  ASSERT_EQ(result[0].columns()[4].rows()->size(), 10);
  ASSERT_EQ(result[0].columns()[5].rows()->size(), 10);
}

TEST_F(LocalRunnerServiceTest, ConvertToBatchesWrapped) {
  auto queryCtx = core::QueryCtx::create();
  core::ExecCtx execCtx(rootPool_.get(), queryCtx.get());
  exec::EvalCtx evalCtx(&execCtx);
  auto result = facebook::velox::runner::convertToBatches(
      {testRowVectorWrapped_}, evalCtx);

  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].columnNames()->size(), 7);
  ASSERT_EQ(result[0].columnTypes()->size(), 7);
  ASSERT_EQ(result[0].numRows(), 5);
  ASSERT_EQ(result[0].columns()->size(), 7);
  ASSERT_EQ(result[0].columns()[0].rows()->size(), 5);
  ASSERT_EQ(result[0].columns()[1].rows()->size(), 5);
  ASSERT_EQ(result[0].columns()[2].rows()->size(), 5);
  ASSERT_EQ(result[0].columns()[3].rows()->size(), 5);
  ASSERT_EQ(result[0].columns()[4].rows()->size(), 5);
  ASSERT_EQ(result[0].columns()[5].rows()->size(), 5);

  auto kWrappedConstantIndex = 2;
  ASSERT_EQ(
      result[0].columns()[kWrappedConstantIndex].rows()[0],
      result[0].columns()[kWrappedConstantIndex].rows()[1]);
  ASSERT_EQ(
      result[0].columns()[kWrappedConstantIndex].rows()[0],
      result[0].columns()[kWrappedConstantIndex].rows()[2]);
  ASSERT_EQ(
      result[0].columns()[kWrappedConstantIndex].rows()[0],
      result[0].columns()[kWrappedConstantIndex].rows()[3]);
  ASSERT_EQ(
      result[0].columns()[kWrappedConstantIndex].rows()[0],
      result[0].columns()[kWrappedConstantIndex].rows()[4]);

  auto reference =
      facebook::velox::runner::convertToBatches({testRowVector_}, evalCtx);
  auto kWrappedDictionaryIndex = 1;
  ASSERT_EQ(
      result[0].columns()[kWrappedDictionaryIndex].rows()[0],
      reference[0].columns()[kWrappedDictionaryIndex].rows()[3]);
  ASSERT_EQ(
      result[0].columns()[kWrappedDictionaryIndex].rows()[1],
      reference[0].columns()[kWrappedDictionaryIndex].rows()[0]);
  ASSERT_EQ(
      result[0].columns()[kWrappedDictionaryIndex].rows()[2],
      reference[0].columns()[kWrappedDictionaryIndex].rows()[7]);
  ASSERT_EQ(
      result[0].columns()[kWrappedDictionaryIndex].rows()[3],
      reference[0].columns()[kWrappedDictionaryIndex].rows()[4]);
  ASSERT_EQ(
      result[0].columns()[kWrappedDictionaryIndex].rows()[4],
      reference[0].columns()[kWrappedDictionaryIndex].rows()[1]);
  ASSERT_NE(
      result[0].columns()[kWrappedDictionaryIndex].rows()[0],
      reference[0].columns()[kWrappedDictionaryIndex].rows()[1]);
}

TEST_F(LocalRunnerServiceTest, ServiceHandlerMockRequestIntegration) {
  LocalRunnerServiceHandler handler;

  auto request = std::make_unique<ExecutePlanRequest>();
  // Serialized plan for the following:
  // expressions: (p0:DOUBLE, plus(null,0.1646418017335236))
  request->serializedPlan() =
      R"({"names":["p0","p1"],"id":"project","name":"ProjectNode","sources":[{"name":"ProjectNode","id":"transform","projections":[{"name":"FieldAccessTypedExpr","type":{"name":"Type","type":"BIGINT"},"inputs":[{"name":"InputTypedExpr","type":{"type":"ROW","name":"Type","names":["row_number"],"cTypes":[{"name":"Type","type":"BIGINT"}]}}],"fieldName":"row_number"}],"names":["row_number"],"sources":[{"name":"ValuesNode","id":"efb6650a_8541_4214_82dd_9792a4965380","data":"AAAAAF4AAAB7ImNUeXBlcyI6W3sidHlwZSI6IkJJR0lOVCIsIm5hbWUiOiJUeXBlIn1dLCJuYW1lcyI6WyJyb3dfbnVtYmVyIl0sInR5cGUiOiJST1ciLCJuYW1lIjoiVHlwZSJ9AQAAAAABAAAAAQAAAAAfAAAAeyJ0eXBlIjoiQklHSU5UIiwibmFtZSI6IlR5cGUifQEAAAAAAQgAAAAAAAAAAAAAAA==","parallelizable":false,"repeatTimes":1}]}],"projections":[{"name":"CallTypedExpr","type":{"name":"Type","type":"DOUBLE"},"functionName":"plus","inputs":[{"name":"ConstantTypedExpr","type":{"name":"Type","type":"DOUBLE"},"valueVector":"AQAAAB8AAAB7InR5cGUiOiJET1VCTEUiLCJuYW1lIjoiVHlwZSJ9AQAAAAE="},{"name":"ConstantTypedExpr","type":{"name":"Type","type":"DOUBLE"},"valueVector":"AQAAAB8AAAB7InR5cGUiOiJET1VCTEUiLCJuYW1lIjoiVHlwZSJ9AQAAAAABAAAAifsSxT8="}]},{"name":"FieldAccessTypedExpr","type":{"name":"Type","type":"BIGINT"},"fieldName":"row_number"}]})";
  request->queryId() = "query1";

  ExecutePlanResponse response;
  handler.execute(response, std::move(request));

  EXPECT_TRUE(*response.success());
  EXPECT_EQ(response.results()->size(), 1);

  const auto& batch = (*response.results()).front();
  EXPECT_EQ(batch.columnNames()->size(), 2);
  EXPECT_EQ((*batch.columnNames())[0], "p0");
  EXPECT_EQ(batch.columnTypes()->size(), 2);
  EXPECT_EQ((*batch.columnTypes())[0], "DOUBLE");
  EXPECT_EQ(batch.numRows(), 1);
  EXPECT_EQ(batch.columns()->size(), 2);

  const auto& column = (*batch.columns())[0];
  EXPECT_EQ(column.rows()->size(), 1);
  const auto& row = (*column.rows())[0];
  EXPECT_TRUE(*row.isNull());
  EXPECT_FALSE(row.scalarValue().has_value());
}

TEST_F(LocalRunnerServiceTest, ServiceHandlerMockRequestIntegrationFailure) {
  LocalRunnerServiceHandler handler;

  auto request = std::make_unique<ExecutePlanRequest>();
  // Serialized plan for the following:
  // expressions: (p0:TINYINT, divide(89,"c0")
  // Will encounter divide by zero error.
  request->serializedPlan() =
      R"({"projections":[{"inputs":[{"valueVector":"AQAAACAAAAB7InR5cGUiOiJUSU5ZSU5UIiwibmFtZSI6IlR5cGUifQEAAAAAAVk=","type":{"type":"TINYINT","name":"Type"},"name":"ConstantTypedExpr"},{"fieldName":"c0","type":{"type":"TINYINT","name":"Type"},"name":"FieldAccessTypedExpr"}],"functionName":"divide","type":{"type":"TINYINT","name":"Type"},"name":"CallTypedExpr"},{"fieldName":"row_number","type":{"type":"BIGINT","name":"Type"},"name":"FieldAccessTypedExpr"}],"sources":[{"projections":[{"inputs":[{"type":{"cTypes":[{"type":"TINYINT","name":"Type"},{"type":"BIGINT","name":"Type"}],"names":["c0","row_number"],"type":"ROW","name":"Type"},"name":"InputTypedExpr"}],"fieldName":"c0","type":{"type":"TINYINT","name":"Type"},"name":"FieldAccessTypedExpr"},{"inputs":[{"type":{"cTypes":[{"type":"TINYINT","name":"Type"},{"type":"BIGINT","name":"Type"}],"names":["c0","row_number"],"type":"ROW","name":"Type"},"name":"InputTypedExpr"}],"fieldName":"row_number","type":{"type":"BIGINT","name":"Type"},"name":"FieldAccessTypedExpr"}],"sources":[{"parallelizable":false,"repeatTimes":1,"data":"AAAAAIQAAAB7ImNUeXBlcyI6W3sidHlwZSI6IlRJTllJTlQiLCJuYW1lIjoiVHlwZSJ9LHsidHlwZSI6IkJJR0lOVCIsIm5hbWUiOiJUeXBlIn1dLCJuYW1lcyI6WyJjMCIsInJvd19udW1iZXIiXSwidHlwZSI6IlJPVyIsIm5hbWUiOiJUeXBlIn0KAAAAAAIAAAABAgAAACAAAAB7InR5cGUiOiJUSU5ZSU5UIiwibmFtZSI6IlR5cGUifQoAAAAAKAAAAAMAAAACAAAABgAAAAAAAAABAAAACAAAAAUAAAAAAAAACAAAAAUAAAACAAAAIAAAAHsidHlwZSI6IlRJTllJTlQiLCJuYW1lIjoiVHlwZSJ9CgAAAAECAAAA9/8oAAAACQAAAAQAAAAJAAAAAAAAAAYAAAAHAAAABAAAAAYAAAAAAAAAAAAAAAIAAAAgAAAAeyJ0eXBlIjoiVElOWUlOVCIsIm5hbWUiOiJUeXBlIn0KAAAAAQIAAAD7oigAAAAJAAAAAQAAAAkAAAAHAAAAAAAAAAUAAAAEAAAAAwAAAAEAAAAAAAAAAAAAACAAAAB7InR5cGUiOiJUSU5ZSU5UIiwibmFtZSI6IlR5cGUifQoAAAAAAQoAAABTOkYvJBw5ZUAAAQAAAAAfAAAAeyJ0eXBlIjoiQklHSU5UIiwibmFtZSI6IlR5cGUifQoAAAAAAVAAAAAAAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAAQAAAAAAAAABQAAAAAAAAAGAAAAAAAAAAcAAAAAAAAACAAAAAAAAAAJAAAAAAAAAA==","id":"d69f11dc_1f0e_40ae_8c5d_2cde4b784a12","name":"ValuesNode"}],"names":["c0","row_number"],"id":"transform","name":"ProjectNode"}],"names":["p0","p1"],"id":"project","name":"ProjectNode"})";
  request->queryId() = "query1";

  ExecutePlanResponse response;
  handler.execute(response, std::move(request));

  ASSERT_TRUE(response.errorMessage().has_value());
  auto errorMsg = response.errorMessage().value();
  EXPECT_NE(errorMsg.find("Error Source: USER"), std::string::npos);
  EXPECT_NE(errorMsg.find("Error Code: ARITHMETIC_ERROR"), std::string::npos);
  EXPECT_NE(errorMsg.find("Reason: division by zero"), std::string::npos);

  EXPECT_FALSE(*response.success());
  EXPECT_EQ(response.results()->size(), 0);
}

} // namespace facebook::velox::fuzzer::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
