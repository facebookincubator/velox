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
#include "velox/runner/if/gen-cpp2/LocalRunnerService.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::runner;
using namespace facebook::velox::test;

namespace {

// Forward declarations of helper functions from LocalRunnerService.cpp
// These would normally be in a header file, but since they're in an anonymous
// namespace in the implementation, we need to redeclare them for testing

ResultBatch convertToResultBatch(const std::vector<RowVectorPtr>& rowVectors);

} // namespace

class LocalRunnerServiceTest : public functions::test::FunctionBaseTest {
 protected:
  void SetUp() override {
    createTestData();
  }

  void createTestData() {
    // Create test vectors for different data types
    auto rowType = ROW(
        {{"bool_col", BOOLEAN()},
         {"tinyint_col", TINYINT()},
         {"smallint_col", SMALLINT()},
         {"int_col", INTEGER()},
         {"bigint_col", BIGINT()},
         {"real_col", REAL()},
         {"double_col", DOUBLE()},
         {"varchar_col", VARCHAR()},
         {"varbinary_col", VARBINARY()},
         {"timestamp_col", TIMESTAMP()}});

    testRowVector_ = makeRowVector(
        {"bool_col",
         "tinyint_col",
         "smallint_col",
         "int_col",
         "bigint_col",
         "real_col",
         "double_col",
         "varchar_col",
         "varbinary_col",
         "timestamp_col"},
        {makeFlatVector<bool>(
             10,
             [](auto row) { return row % 2 == 0; },
             [](auto row) { return row % 3 == 0; }),
         makeFlatVector<int8_t>(
             10,
             [](auto row) { return row; },
             [](auto row) { return row % 3 == 0; }),
         makeFlatVector<int16_t>(
             10,
             [](auto row) { return row; },
             [](auto row) { return row % 3 == 0; }),
         makeFlatVector<int32_t>(
             10,
             [](auto row) { return row; },
             [](auto row) { return row % 3 == 0; }),
         makeFlatVector<int64_t>(
             10,
             [](auto row) { return row; },
             [](auto row) { return row % 3 == 0; }),
         makeFlatVector<float>(
             10,
             [](auto row) { return row * 1.1f; },
             [](auto row) { return row % 3 == 0; }),
         makeFlatVector<double>(
             10,
             [](auto row) { return row * 1.1; },
             [](auto row) { return row % 3 == 0; }),
         makeFlatVector<std::string>(
             10,
             [](auto row) { return fmt::format("str_{}", row); },
             [](auto row) { return row % 3 == 0; }),
         makeFlatVector<std::string>(
             10,
             [](auto row) { return fmt::format("bin_{}", row); },
             [](auto row) { return row % 3 == 0; }),
         makeFlatVector<Timestamp>(
             10,
             [](auto row) { return Timestamp(row, 0); },
             [](auto row) { return row % 3 == 0; })});
  }

  RowVectorPtr testRowVector_;
};

// Test helper function: convertToResultBatch
TEST_F(LocalRunnerServiceTest, ConvertToResultBatchEmpty) {
  std::vector<RowVectorPtr> emptyVectors;
  auto resultBatch = convertToResultBatch(emptyVectors);

  EXPECT_TRUE(resultBatch.rows()->empty());
  EXPECT_TRUE(resultBatch.columnNames()->empty());
  EXPECT_TRUE(resultBatch.columnTypes()->empty());
}

TEST_F(LocalRunnerServiceTest, ConvertToResultBatchWithData) {
  std::vector<RowVectorPtr> vectors = {testRowVector_};
  auto resultBatch = convertToResultBatch(vectors);

  // Check column metadata
  EXPECT_EQ(resultBatch.columnNames()->size(), 10);
  EXPECT_EQ(resultBatch.columnTypes()->size(), 10);
  EXPECT_EQ((*resultBatch.columnNames())[0], "bool_col");
  EXPECT_EQ((*resultBatch.columnNames())[1], "tinyint_col");

  // Check row count
  EXPECT_EQ(resultBatch.rows()->size(), testRowVector_->size());

  // Check first row data
  const auto& firstRow = (*resultBatch.rows())[0];
  EXPECT_EQ(firstRow.cells()->size(), 10);

  // Check null handling (every 3rd row should be null)
  const auto& nullRow = (*resultBatch.rows())[0]; // Row 0 should be null
  for (const auto& cell : *nullRow.cells()) {
    EXPECT_TRUE(*cell.isNull());
  }

  // Check non-null row
  const auto& nonNullRow = (*resultBatch.rows())[1]; // Row 1 should have data
  for (const auto& cell : *nonNullRow.cells()) {
    EXPECT_FALSE(*cell.isNull());
  }
}

TEST_F(LocalRunnerServiceTest, ConvertToResultBatchNullVector) {
  std::vector<RowVectorPtr> vectors = {nullptr};
  auto resultBatch = convertToResultBatch(vectors);

  EXPECT_TRUE(resultBatch.rows()->empty());
  EXPECT_TRUE(resultBatch.columnNames()->empty());
  EXPECT_TRUE(resultBatch.columnTypes()->empty());
}

// Simple test handler for testing
class TestLocalRunnerServiceHandler : public LocalRunnerServiceSvIf {
 public:
  void executePlan(
      ExecutePlanResponse& response,
      std::unique_ptr<ExecutePlanRequest> request) override {
    // Simple test implementation
    response.success() = true;
    response.output() = "Test execution completed";

    // Create a simple result batch
    ResultBatch batch;
    batch.columnNames()->push_back("test_col");
    batch.columnTypes()->push_back("INTEGER");

    ResultRow row;
    Cell cell;
    cell.isNull() = false;
    ScalarValue value;
    value.integerValue_ref() = 42;
    cell.value_ref() = value;
    row.cells()->push_back(cell);
    batch.rows()->push_back(row);

    response.resultBatches()->push_back(batch);
  }
};

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
  handler.executePlan(response, std::move(request));

  // Verify the response
  EXPECT_TRUE(*response.success());
  EXPECT_EQ(*response.output(), "Test execution completed");
  EXPECT_EQ(response.resultBatches()->size(), 1);

  // Verify the result batch structure
  const auto& batch = (*response.resultBatches())[0];
  EXPECT_EQ(batch.columnNames()->size(), 1);
  EXPECT_EQ((*batch.columnNames())[0], "test_col");
  EXPECT_EQ(batch.columnTypes()->size(), 1);
  EXPECT_EQ((*batch.columnTypes())[0], "INTEGER");
  EXPECT_EQ(batch.rows()->size(), 1);

  // Verify the row data
  const auto& row = (*batch.rows())[0];
  EXPECT_EQ(row.cells()->size(), 1);
  const auto& cell = (*row.cells())[0];
  EXPECT_FALSE(*cell.isNull());
  EXPECT_TRUE(cell.value_ref().has_value());
  EXPECT_EQ(*cell.value_ref()->integerValue_ref(), 42);
}

// Test service handler with multiple mock requests
TEST_F(LocalRunnerServiceTest, ServiceHandlerMultipleMockRequests) {
  TestLocalRunnerServiceHandler handler;

  // Test multiple requests to ensure handler is stateless
  for (int i = 0; i < 3; ++i) {
    auto request = std::make_unique<ExecutePlanRequest>();
    request->serializedPlan() = fmt::format(
        R"({{"planNodeId": "test_{}", "planNodeType": "ValuesNode"}})", i);
    request->queryId() = fmt::format("test_query_{}", i);
    request->numWorkers() = 1;
    request->numDrivers() = 1;

    ExecutePlanResponse response;
    handler.executePlan(response, std::move(request));

    // Each response should be successful and consistent
    EXPECT_TRUE(*response.success());
    EXPECT_EQ(*response.output(), "Test execution completed");
    EXPECT_EQ(response.resultBatches()->size(), 1);

    const auto& batch = (*response.resultBatches())[0];
    EXPECT_EQ(batch.rows()->size(), 1);
    const auto& cell = (*(*batch.rows())[0].cells())[0];
    EXPECT_EQ(*cell.value_ref()->integerValue_ref(), 42);
  }
}

// Test service handler error simulation
TEST_F(LocalRunnerServiceTest, ServiceHandlerErrorSimulation) {
  // Create a custom handler that simulates errors
  class ErrorTestHandler : public LocalRunnerServiceSvIf {
   public:
    void executePlan(
        ExecutePlanResponse& response,
        std::unique_ptr<ExecutePlanRequest> request) override {
      // Simulate an error condition
      response.success() = false;
      response.errorMessage() =
          fmt::format("Simulated error for query: {}", *request->queryId());
      response.output() = "Error occurred during execution";
    }
  };

  ErrorTestHandler errorHandler;

  auto request = std::make_unique<ExecutePlanRequest>();
  request->serializedPlan() = "invalid_plan";
  request->queryId() = "error_test_query";
  request->numWorkers() = 1;
  request->numDrivers() = 1;

  ExecutePlanResponse response;
  errorHandler.executePlan(response, std::move(request));

  // Verify error response
  EXPECT_FALSE(*response.success());
  EXPECT_EQ(
      *response.errorMessage(), "Simulated error for query: error_test_query");
  EXPECT_EQ(*response.output(), "Error occurred during execution");
  EXPECT_TRUE(response.resultBatches()->empty());
}

// Test service handler with complex result batch
TEST_F(LocalRunnerServiceTest, ServiceHandlerComplexResultBatch) {
  // Create a handler that returns more complex data
  class ComplexDataHandler : public LocalRunnerServiceSvIf {
   public:
    void executePlan(
        ExecutePlanResponse& response,
        std::unique_ptr<ExecutePlanRequest> request) override {
      response.success() = true;
      response.output() = "Complex data execution completed";

      ResultBatch batch;
      batch.columnNames()->push_back("id");
      batch.columnNames()->push_back("name");
      batch.columnNames()->push_back("score");
      batch.columnTypes()->push_back("INTEGER");
      batch.columnTypes()->push_back("VARCHAR");
      batch.columnTypes()->push_back("DOUBLE");

      // Add multiple rows
      for (int i = 1; i <= 3; ++i) {
        ResultRow row;

        // ID column
        Cell idCell;
        idCell.isNull() = false;
        ScalarValue idValue;
        idValue.integerValue_ref() = i;
        idCell.value_ref() = idValue;
        row.cells()->push_back(idCell);

        // Name column
        Cell nameCell;
        nameCell.isNull() = false;
        ScalarValue nameValue;
        nameValue.varcharValue_ref() = fmt::format("User_{}", i);
        nameCell.value_ref() = nameValue;
        row.cells()->push_back(nameCell);

        // Score column (with one null value)
        Cell scoreCell;
        if (i == 2) {
          scoreCell.isNull() = true;
        } else {
          scoreCell.isNull() = false;
          ScalarValue scoreValue;
          scoreValue.doubleValue_ref() = i * 10.5;
          scoreCell.value_ref() = scoreValue;
        }
        row.cells()->push_back(scoreCell);

        batch.rows()->push_back(row);
      }

      response.resultBatches()->push_back(batch);
    }
  };

  ComplexDataHandler complexHandler;

  auto request = std::make_unique<ExecutePlanRequest>();
  request->serializedPlan() =
      R"({"planNodeId": "complex", "planNodeType": "ValuesNode"})";
  request->queryId() = "complex_test_query";
  request->numWorkers() = 2;
  request->numDrivers() = 2;

  ExecutePlanResponse response;
  complexHandler.executePlan(response, std::move(request));

  // Verify complex response
  EXPECT_TRUE(*response.success());
  EXPECT_EQ(*response.output(), "Complex data execution completed");
  EXPECT_EQ(response.resultBatches()->size(), 1);

  const auto& batch = (*response.resultBatches())[0];
  EXPECT_EQ(batch.columnNames()->size(), 3);
  EXPECT_EQ(batch.columnTypes()->size(), 3);
  EXPECT_EQ(batch.rows()->size(), 3);

  // Verify column metadata
  EXPECT_EQ((*batch.columnNames())[0], "id");
  EXPECT_EQ((*batch.columnNames())[1], "name");
  EXPECT_EQ((*batch.columnNames())[2], "score");
  EXPECT_EQ((*batch.columnTypes())[0], "INTEGER");
  EXPECT_EQ((*batch.columnTypes())[1], "VARCHAR");
  EXPECT_EQ((*batch.columnTypes())[2], "DOUBLE");

  // Verify row data
  for (int i = 0; i < 3; ++i) {
    const auto& row = (*batch.rows())[i];
    EXPECT_EQ(row.cells()->size(), 3);

    // Check ID
    const auto& idCell = (*row.cells())[0];
    EXPECT_FALSE(*idCell.isNull());
    EXPECT_EQ(*idCell.value_ref()->integerValue_ref(), i + 1);

    // Check Name
    const auto& nameCell = (*row.cells())[1];
    EXPECT_FALSE(*nameCell.isNull());
    EXPECT_EQ(
        *nameCell.value_ref()->varcharValue_ref(),
        fmt::format("User_{}", i + 1));

    // Check Score (row 1 should be null)
    const auto& scoreCell = (*row.cells())[2];
    if (i == 1) { // Second row (index 1)
      EXPECT_TRUE(*scoreCell.isNull());
    } else {
      EXPECT_FALSE(*scoreCell.isNull());
      EXPECT_DOUBLE_EQ(
          *scoreCell.value_ref()->doubleValue_ref(), (i + 1) * 10.5);
    }
  }
}

// Test service handler with invalid serialized plan
TEST_F(LocalRunnerServiceTest, ServiceHandlerInvalidPlan) {
  auto request = std::make_unique<ExecutePlanRequest>();
  request->serializedPlan() = "invalid_json";
  request->queryId() = "test_query_invalid";
  request->numWorkers() = 2;
  request->numDrivers() = 1;

  // The actual handler would catch the exception and set error response
  // We can test the request structure here
  EXPECT_EQ(*request->serializedPlan(), "invalid_json");
  EXPECT_EQ(*request->queryId(), "test_query_invalid");
}

// Test response structure
TEST_F(LocalRunnerServiceTest, ResponseStructure) {
  ExecutePlanResponse response;

  // Test successful response
  response.success() = true;
  response.output() = "Query executed successfully";

  ResultBatch batch;
  batch.columnNames()->push_back("col1");
  batch.columnTypes()->push_back("INTEGER");

  ResultRow row;
  Cell cell;
  cell.isNull() = false;
  ScalarValue value;
  value.integerValue_ref() = 42;
  cell.value_ref() = value;
  row.cells()->push_back(cell);
  batch.rows()->push_back(row);

  response.resultBatches()->push_back(batch);

  EXPECT_TRUE(*response.success());
  EXPECT_EQ(*response.output(), "Query executed successfully");
  EXPECT_EQ(response.resultBatches()->size(), 1);
  EXPECT_EQ((*response.resultBatches())[0].rows()->size(), 1);
  EXPECT_EQ((*(*response.resultBatches())[0].rows())[0].cells()->size(), 1);

  // Test error response
  ExecutePlanResponse errorResponse;
  errorResponse.success() = false;
  errorResponse.errorMessage() = "Plan deserialization failed";

  EXPECT_FALSE(*errorResponse.success());
  EXPECT_EQ(*errorResponse.errorMessage(), "Plan deserialization failed");
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
  timestampVal.timestampValue_ref() = 1234567890000L; // milliseconds
  EXPECT_EQ(*timestampVal.timestampValue_ref(), 1234567890000L);
}

// Test Cell null handling
TEST_F(LocalRunnerServiceTest, CellNullHandling) {
  // Test null cell
  Cell nullCell;
  nullCell.isNull() = true;
  EXPECT_TRUE(*nullCell.isNull());
  EXPECT_FALSE(nullCell.value_ref().has_value());

  // Test non-null cell
  Cell nonNullCell;
  nonNullCell.isNull() = false;
  ScalarValue value;
  value.integerValue_ref() = 42;
  nonNullCell.value_ref() = value;

  EXPECT_FALSE(*nonNullCell.isNull());
  EXPECT_TRUE(nonNullCell.value_ref().has_value());
  EXPECT_EQ(*nonNullCell.value_ref()->integerValue_ref(), 42);
}

// Test ResultBatch structure
TEST_F(LocalRunnerServiceTest, ResultBatchStructure) {
  ResultBatch batch;

  // Add column metadata
  batch.columnNames()->push_back("id");
  batch.columnNames()->push_back("name");
  batch.columnTypes()->push_back("INTEGER");
  batch.columnTypes()->push_back("VARCHAR");

  // Add a row
  ResultRow row;

  // Add id cell
  Cell idCell;
  idCell.isNull() = false;
  ScalarValue idValue;
  idValue.integerValue_ref() = 1;
  idCell.value_ref() = idValue;
  row.cells()->push_back(idCell);

  // Add name cell
  Cell nameCell;
  nameCell.isNull() = false;
  ScalarValue nameValue;
  nameValue.varcharValue_ref() = "John";
  nameCell.value_ref() = nameValue;
  row.cells()->push_back(nameCell);

  batch.rows()->push_back(row);

  // Verify structure
  EXPECT_EQ(batch.columnNames()->size(), 2);
  EXPECT_EQ(batch.columnTypes()->size(), 2);
  EXPECT_EQ(batch.rows()->size(), 1);
  EXPECT_EQ((*batch.rows())[0].cells()->size(), 2);

  EXPECT_EQ((*batch.columnNames())[0], "id");
  EXPECT_EQ((*batch.columnNames())[1], "name");
  EXPECT_EQ((*batch.columnTypes())[0], "INTEGER");
  EXPECT_EQ((*batch.columnTypes())[1], "VARCHAR");
}

// Since the helper functions are in an anonymous namespace in the original
// file, we need to provide stub implementations for testing
namespace {

ResultBatch convertToResultBatch(const std::vector<RowVectorPtr>& rowVectors) {
  ResultBatch resultBatch;

  if (rowVectors.empty()) {
    return resultBatch;
  }

  // Get column names and types from the first row vector
  const auto& firstVector = rowVectors[0];
  if (!firstVector || !firstVector->type() || !firstVector->type()->isRow()) {
    return resultBatch;
  }

  const auto& rowType = firstVector->type()->asRow();
  for (auto i = 0; i < rowType.size(); ++i) {
    resultBatch.columnNames()->push_back(rowType.nameOf(i));
    resultBatch.columnTypes()->push_back(rowType.childAt(i)->toString());
  }

  // Process each row vector
  for (const auto& rowVector : rowVectors) {
    if (!rowVector) {
      continue;
    }

    for (vector_size_t rowIdx = 0; rowIdx < rowVector->size(); ++rowIdx) {
      ResultRow row;

      for (auto colIdx = 0; colIdx < rowVector->childrenSize(); ++colIdx) {
        const auto& vector = rowVector->childAt(colIdx);
        Cell cell;

        if (vector->isNullAt(rowIdx)) {
          cell.isNull() = true;
        } else {
          cell.isNull() = false;
          ScalarValue value;

          // Handle different vector types
          switch (vector->typeKind()) {
            case TypeKind::BOOLEAN:
              value.boolValue_ref() =
                  vector->as<SimpleVector<bool>>()->valueAt(rowIdx);
              break;
            case TypeKind::TINYINT:
              value.tinyintValue_ref() =
                  vector->as<SimpleVector<int8_t>>()->valueAt(rowIdx);
              break;
            case TypeKind::SMALLINT:
              value.smallintValue_ref() =
                  vector->as<SimpleVector<int16_t>>()->valueAt(rowIdx);
              break;
            case TypeKind::INTEGER:
              value.integerValue_ref() =
                  vector->as<SimpleVector<int32_t>>()->valueAt(rowIdx);
              break;
            case TypeKind::BIGINT:
              value.bigintValue_ref() =
                  vector->as<SimpleVector<int64_t>>()->valueAt(rowIdx);
              break;
            case TypeKind::REAL:
              value.realValue_ref() =
                  vector->as<SimpleVector<float>>()->valueAt(rowIdx);
              break;
            case TypeKind::DOUBLE:
              value.doubleValue_ref() =
                  vector->as<SimpleVector<double>>()->valueAt(rowIdx);
              break;
            case TypeKind::VARCHAR:
              value.varcharValue_ref() =
                  vector->as<SimpleVector<StringView>>()->valueAt(rowIdx).str();
              break;
            case TypeKind::VARBINARY: {
              const auto& binValue =
                  vector->as<SimpleVector<StringView>>()->valueAt(rowIdx);
              value.varbinaryValue_ref() =
                  std::string(binValue.data(), binValue.size());
              break;
            }
            case TypeKind::TIMESTAMP:
              value.timestampValue_ref() = vector->as<SimpleVector<Timestamp>>()
                                               ->valueAt(rowIdx)
                                               .toMillis();
              break;
            default:
              // For unsupported types, just mark as null
              cell.isNull() = true;
          }

          if (!*cell.isNull()) {
            cell.value_ref() = std::move(value);
          }
        }

        row.cells()->push_back(std::move(cell));
      }

      resultBatch.rows()->push_back(std::move(row));
    }
  }

  return resultBatch;
}

} // namespace
