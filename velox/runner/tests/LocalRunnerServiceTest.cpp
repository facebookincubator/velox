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
#include <memory>

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/runner/LocalRunnerService.h"
#include "velox/runner/if/gen-cpp2/LocalRunnerService.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::runner;
using namespace facebook::velox::test;

class LocalRunnerServiceTest : public functions::test::FunctionBaseTest {
 protected:
  void SetUp() override {
    createTestData();
  }

  void createTestData() {
    // Create test vectors for different data types
    auto rowType = ROW(
        {{"bool_col", BOOLEAN()},
         {"int_col", INTEGER()},
         {"bigint_col", BIGINT()},
         {"double_col", DOUBLE()},
         {"varchar_col", VARCHAR()},
         {"timestamp_col", TIMESTAMP()}});

    testRowVector_ = makeRowVector(
        {"bool_col",
         "int_col",
         "bigint_col",
         "double_col",
         "varchar_col",
         "timestamp_col"},
        {makeFlatVector<bool>(
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
             [](auto row) { return row % 3 == 0; })});

    testRowVectorWrapped_ = makeRowVector(
        {"bool_col",
         "int_col",
         "bigint_col",
         "double_col",
         "varchar_col",
         "timestamp_col"},
        {makeFlatVector<bool>(
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
             [](auto row) { return row % 3 == 0; })});
  }

  RowVectorPtr testRowVector_;
  RowVectorPtr testRowVectorWrapped_;
};

// Simple test handler for testing
class TestLocalRunnerServiceHandler : public LocalRunnerServiceSvIf {
 public:
  void execute(
      ExecutePlanResponse& response,
      std::unique_ptr<ExecutePlanRequest> request) override {
    // Simple test implementation
    response.success() = true;
    response.output() = "Test execution completed";

    // Create a simple result batch
    Batch batch;
    batch.columnNames()->push_back("test_col");
    batch.columnTypes()->push_back("INTEGER");
    batch.numRows() = 1;

    // Create a column with one row
    Column column;
    Value value;
    value.isNull() = false;

    ScalarValue scalarValue;
    scalarValue.integerValue_ref() = 42;
    value.scalarValue_ref() = scalarValue;

    column.rows()->push_back(value);

    batch.columns()->push_back(column);
    response.results()->push_back(batch);
  }
};

TEST_F(LocalRunnerServiceTest, ConvertToBatches) {
  auto queryCtx = core::QueryCtx::create();
  core::ExecCtx execCtx(rootPool_.get(), queryCtx.get());
  exec::EvalCtx evalCtx(&execCtx);
  auto result =
      facebook::velox::runner::convertToBatches({testRowVector_}, evalCtx);

  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].columnNames()->size(), 6);
  ASSERT_EQ(result[0].columnTypes()->size(), 6);
  ASSERT_EQ(result[0].numRows(), 10);
  ASSERT_EQ(result[0].columns()->size(), 6);
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
  ASSERT_EQ(result[0].columnNames()->size(), 6);
  ASSERT_EQ(result[0].columnTypes()->size(), 6);
  ASSERT_EQ(result[0].numRows(), 5);
  ASSERT_EQ(result[0].columns()->size(), 6);
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

// Test service handler with valid request
TEST_F(LocalRunnerServiceTest, ServiceHandlerValidRequest) {
  // Create a simple plan for testing
  auto planBuilder = exec::test::PlanBuilder();
  auto plan = planBuilder
                  .values({makeRowVector(
                      {makeFlatVector<int32_t>({1, 2, 3}),
                       makeFlatVector<std::string>({"a", "b", "c"})})})
                  .planNode();

  // Serialize the plan
  auto serializedPlan = folly::toJson(plan->serialize());

  // Create request
  auto request = std::make_unique<ExecutePlanRequest>();
  request->serializedPlan() = serializedPlan;
  request->queryId() = "test_query_123";
  request->numWorkers() = 2;
  request->numDrivers() = 1;

  EXPECT_EQ(*request->queryId(), "test_query_123");
  EXPECT_EQ(*request->numWorkers(), 2);
  EXPECT_EQ(*request->numDrivers(), 1);
  EXPECT_FALSE(request->serializedPlan()->empty());
}

// Test service handler with mock request - Integration test
TEST_F(LocalRunnerServiceTest, ServiceHandlerMockRequestIntegration) {
  // Create a test handler
  TestLocalRunnerServiceHandler handler;

  // Create a mock request
  auto request = std::make_unique<ExecutePlanRequest>();
  request->serializedPlan() =
      R"({"planNodeId": "test", "planNodeType": "ValuesNode"})";
  request->queryId() = "integration_test_query";
  request->numWorkers() = 1;
  request->numDrivers() = 1;

  // Execute the request
  ExecutePlanResponse response;
  handler.execute(response, std::move(request));

  // Verify the response
  EXPECT_TRUE(*response.success());
  EXPECT_EQ(*response.output(), "Test execution completed");
  EXPECT_EQ(response.results()->size(), 1);

  // Verify the result batch structure
  const auto& batch = (*response.results())[0];
  EXPECT_EQ(batch.columnNames()->size(), 1);
  EXPECT_EQ((*batch.columnNames())[0], "test_col");
  EXPECT_EQ(batch.columnTypes()->size(), 1);
  EXPECT_EQ((*batch.columnTypes())[0], "INTEGER");
  EXPECT_EQ(batch.numRows(), 1);
  EXPECT_EQ(batch.columns()->size(), 1);

  // Verify the column data
  const auto& column = (*batch.columns())[0];
  EXPECT_EQ(column.rows()->size(), 1);
  const auto& row = (*column.rows())[0];
  EXPECT_FALSE(*row.isNull());
  EXPECT_TRUE(row.scalarValue_ref().has_value());
  EXPECT_EQ(*row.scalarValue_ref()->integerValue_ref(), 42);
}

// Test ScalarValue union handling
TEST_F(LocalRunnerServiceTest, ScalarValueTypes) {
  // Test different scalar value types
  ScalarValue boolVal;
  boolVal.boolValue_ref() = true;
  EXPECT_TRUE(*boolVal.boolValue_ref());

  ScalarValue intVal;
  intVal.integerValue_ref() = 123;
  EXPECT_EQ(*intVal.integerValue_ref(), 123);

  ScalarValue strVal;
  strVal.varcharValue_ref() = "test_string";
  EXPECT_EQ(*strVal.varcharValue_ref(), "test_string");

  ScalarValue doubleVal;
  doubleVal.doubleValue_ref() = 3.14;
  EXPECT_DOUBLE_EQ(*doubleVal.doubleValue_ref(), 3.14);

  ScalarValue timestampVal;
  facebook::velox::runner::Timestamp ts;
  ts.seconds_ref() = 1234567890;
  ts.nanos_ref() = 123456789;
  timestampVal.timestampValue_ref() = ts;
  EXPECT_EQ(*timestampVal.timestampValue_ref()->seconds_ref(), 1234567890);
  EXPECT_EQ(*timestampVal.timestampValue_ref()->nanos_ref(), 123456789);
}

// Test ColumnRow null handling
TEST_F(LocalRunnerServiceTest, ColumnRowNullHandling) {
  // Test null row
  Value nullRow;
  nullRow.isNull() = true;
  EXPECT_TRUE(*nullRow.isNull());
  EXPECT_FALSE(nullRow.scalarValue_ref().has_value());

  // Test non-null row
  Value nonNullRow;
  nonNullRow.isNull() = false;

  ScalarValue scalarValue;
  scalarValue.integerValue_ref() = 42;
  nonNullRow.scalarValue_ref() = scalarValue;

  EXPECT_FALSE(*nonNullRow.isNull());
  EXPECT_TRUE(nonNullRow.scalarValue_ref().has_value());
  EXPECT_EQ(*nonNullRow.scalarValue_ref()->integerValue_ref(), 42);
}

// Test Batch structure
TEST_F(LocalRunnerServiceTest, BatchStructure) {
  Batch batch;

  // Add column metadata
  batch.columnNames()->push_back("id");
  batch.columnNames()->push_back("name");
  batch.columnTypes()->push_back("INTEGER");
  batch.columnTypes()->push_back("VARCHAR");
  batch.numRows() = 1;

  // Create columns
  batch.columns()->resize(2);

  // ID column
  Column& idColumn = (*batch.columns())[0];
  Value idRow;
  idRow.isNull() = false;

  ScalarValue idScalar;
  idScalar.integerValue_ref() = 1;
  idRow.scalarValue_ref() = idScalar;

  idColumn.rows()->push_back(idRow);

  // Name column
  Column& nameColumn = (*batch.columns())[1];
  Value nameRow;
  nameRow.isNull() = false;

  ScalarValue nameScalar;
  nameScalar.varcharValue_ref() = "John";
  nameRow.scalarValue_ref() = nameScalar;

  nameColumn.rows()->push_back(nameRow);

  // Verify structure
  EXPECT_EQ(batch.columnNames()->size(), 2);
  EXPECT_EQ(batch.columnTypes()->size(), 2);
  EXPECT_EQ(batch.columns()->size(), 2);
  EXPECT_EQ(batch.numRows(), 1);
  EXPECT_EQ((*batch.columns())[0].rows()->size(), 1);
  EXPECT_EQ((*batch.columns())[1].rows()->size(), 1);

  EXPECT_EQ((*batch.columnNames())[0], "id");
  EXPECT_EQ((*batch.columnNames())[1], "name");
  EXPECT_EQ((*batch.columnTypes())[0], "INTEGER");
  EXPECT_EQ((*batch.columnTypes())[1], "VARCHAR");
}

// Test response structure
TEST_F(LocalRunnerServiceTest, ResponseStructure) {
  ExecutePlanResponse response;

  // Test successful response
  response.success() = true;
  response.output() = "Query executed successfully";

  Batch batch;
  batch.columnNames()->push_back("col1");
  batch.columnTypes()->push_back("INTEGER");
  batch.numRows() = 1;

  // Create column
  Column column;
  Value row;
  row.isNull() = false;

  ScalarValue scalarValue;
  scalarValue.integerValue_ref() = 42;
  row.scalarValue_ref() = scalarValue;

  column.rows()->push_back(row);
  batch.columns()->push_back(column);

  response.results()->push_back(batch);

  EXPECT_TRUE(*response.success());
  EXPECT_EQ(*response.output(), "Query executed successfully");
  EXPECT_EQ(response.results()->size(), 1);
  EXPECT_EQ((*response.results())[0].numRows(), 1);
  EXPECT_EQ((*response.results())[0].columns()->size(), 1);
  EXPECT_EQ((*(*response.results())[0].columns())[0].rows()->size(), 1);

  // Test error response
  ExecutePlanResponse errorResponse;
  errorResponse.success() = false;
  errorResponse.errorMessage() = "Plan deserialization failed";

  EXPECT_FALSE(*errorResponse.success());
  EXPECT_EQ(*errorResponse.errorMessage(), "Plan deserialization failed");
}

// Test using testRowVector_ for data validation
TEST_F(LocalRunnerServiceTest, TestRowVectorContent) {
  // Ensure testRowVector_ is initialized
  ASSERT_NE(testRowVector_, nullptr);
  EXPECT_EQ(testRowVector_->size(), 10);
  EXPECT_EQ(testRowVector_->childrenSize(), 6);

  // Check bool_col values and nulls
  auto boolVector = testRowVector_->childAt(0)->asFlatVector<bool>();
  for (vector_size_t i = 0; i < boolVector->size(); ++i) {
    if (i % 3 == 0) {
      EXPECT_TRUE(boolVector->isNullAt(i));
    } else {
      EXPECT_EQ(boolVector->valueAt(i), i % 2 == 0);
    }
  }

  // Check int_col values and nulls
  auto intVector = testRowVector_->childAt(1)->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < intVector->size(); ++i) {
    if (i % 3 == 0) {
      EXPECT_TRUE(intVector->isNullAt(i));
    } else {
      EXPECT_EQ(intVector->valueAt(i), i);
    }
  }

  // Check varchar_col values and nulls
  auto strVector = testRowVector_->childAt(4)->asFlatVector<StringView>();
  for (vector_size_t i = 0; i < strVector->size(); ++i) {
    if (i % 3 == 0) {
      EXPECT_TRUE(strVector->isNullAt(i));
    } else {
      EXPECT_EQ(strVector->valueAt(i).str(), fmt::format("str_{}", i));
    }
  }
}
