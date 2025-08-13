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

#include "velox/exec/fuzzer/VeloxLocalQueryRunner.h"

#include <cpr/cpr.h> // @manual
#include <fmt/format.h>
#include <folly/Uri.h>
#include <folly/json.h>
#include <thrift/lib/cpp2/async/RocketClientChannel.h>
#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/prestosql/types/BingTileType.h"
#include "velox/functions/prestosql/types/GeometryType.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/functions/prestosql/types/JsonType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/runner/if/gen-cpp2/LocalRunnerService.h"
#include "velox/vector/VectorSaver.h"

using namespace facebook::velox::runner;

namespace facebook::velox::exec::test {

VeloxLocalQueryRunner::VeloxLocalQueryRunner(
    memory::MemoryPool* aggregatePool,
    std::string serviceUri,
    std::chrono::milliseconds timeout)
    : ReferenceQueryRunner(aggregatePool),
      serviceUri_(std::move(serviceUri)),
      timeout_(timeout) {
  pool_ = aggregatePool->addLeafChild("leaf");

  folly::Uri uri(serviceUri_);
  thriftHost_ = uri.host();
  thriftPort_ = uri.port();
}

const std::vector<TypePtr>& VeloxLocalQueryRunner::supportedScalarTypes()
    const {
  static const std::vector<TypePtr> kScalarTypes{
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
      VARBINARY(),
      TIMESTAMP(),
      DATE(),
  };
  return kScalarTypes;
}

const std::unordered_map<std::string, DataSpec>&
VeloxLocalQueryRunner::aggregationFunctionDataSpecs() const {
  static const std::unordered_map<std::string, DataSpec>
      kAggregationFunctionDataSpecs{};
  return kAggregationFunctionDataSpecs;
}

std::optional<std::string> VeloxLocalQueryRunner::toSql(
    const core::PlanNodePtr& /*plan*/) {
  // We don't need to convert to SQL for VeloxLocalQueryRunner
  // as we're sending the serialized plan directly
  return std::nullopt;
}

bool VeloxLocalQueryRunner::isConstantExprSupported(
    const core::TypedExprPtr& /*expr*/) {
  // Since we're using Velox directly, we support all constant expressions
  return true;
}

bool VeloxLocalQueryRunner::isSupported(
    const exec::FunctionSignature& /*signature*/) {
  // Since we're using Velox directly, we support all function signatures
  return true;
}

std::pair<
    std::optional<std::multiset<std::vector<velox::variant>>>,
    ReferenceQueryErrorCode>
VeloxLocalQueryRunner::execute(const core::PlanNodePtr& plan) {
  std::pair<
      std::optional<std::vector<velox::RowVectorPtr>>,
      ReferenceQueryErrorCode>
      result = executeAndReturnVector(plan);
  if (result.first) {
    return std::make_pair(
        exec::test::materialize(*result.first), result.second);
  }
  return std::make_pair(std::nullopt, result.second);
}

std::pair<
    std::optional<std::vector<velox::RowVectorPtr>>,
    ReferenceQueryErrorCode>
VeloxLocalQueryRunner::executeAndReturnVector(const core::PlanNodePtr& plan) {
  try {
    std::string serializedPlan = serializePlan(plan);
    auto results = executeSerializedPlan(serializedPlan);
    return std::make_pair(
        std::move(results), ReferenceQueryErrorCode::kSuccess);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error executing query: " << e.what();
    return std::make_pair(
        std::nullopt, ReferenceQueryErrorCode::kReferenceQueryFail);
  }
}

std::vector<RowVectorPtr> VeloxLocalQueryRunner::execute(
    const std::string& /*sql*/) {
  VELOX_FAIL("VeloxLocalQueryRunner does not support SQL execution");
}

std::vector<RowVectorPtr> VeloxLocalQueryRunner::execute(
    const std::string& /*sql*/,
    const std::string& /*sessionProperty*/) {
  VELOX_FAIL("VeloxLocalQueryRunner does not support SQL execution");
}

std::string VeloxLocalQueryRunner::serializePlan(
    const core::PlanNodePtr& plan) {
  // Serialize the plan to JSON
  folly::dynamic serializedPlan = plan->serialize();
  return folly::toJson(serializedPlan);
}

// Convert Thrift ResultBatch to Velox RowVector
std::vector<RowVectorPtr> convertToRowVectors(
    const std::vector<ResultBatch>& resultBatches,
    memory::MemoryPool* pool) {
  std::vector<RowVectorPtr> queryResults;

  for (const auto& batch : resultBatches) {
    if (batch.rows()->empty() || batch.columnNames()->empty() ||
        batch.columnTypes()->empty()) {
      continue;
    }

    // Create a row type from column names and types
    std::vector<std::string> names = *batch.columnNames();
    std::vector<TypePtr> types;

    for (const auto& typeStr : *batch.columnTypes()) {
      TypePtr type;

      // Parse the type string to create the appropriate TypePtr
      if (typeStr == "BOOLEAN") {
        type = BOOLEAN();
      } else if (typeStr == "TINYINT") {
        type = TINYINT();
      } else if (typeStr == "SMALLINT") {
        type = SMALLINT();
      } else if (typeStr == "INTEGER") {
        type = INTEGER();
      } else if (typeStr == "BIGINT") {
        type = BIGINT();
      } else if (typeStr == "REAL") {
        type = REAL();
      } else if (typeStr == "DOUBLE") {
        type = DOUBLE();
      } else if (typeStr.find("VARCHAR") == 0) {
        type = VARCHAR();
      } else if (typeStr.find("VARBINARY") == 0) {
        type = VARBINARY();
      } else if (typeStr == "TIMESTAMP") {
        type = TIMESTAMP();
      } else if (typeStr == "DATE") {
        type = DATE();
      } else if (typeStr == "BINGTILE") {
        type = BINGTILE();
      } else if (typeStr == "INTERVAL YEAR TO MONTH") {
        type = INTERVAL_YEAR_MONTH();
      } else if (typeStr == "TIMESTAMP WITH TIME ZONE") {
        type = TIMESTAMP_WITH_TIME_ZONE();
      } else if (typeStr == "HYPERLOGLOG") {
        type = HYPERLOGLOG();
      } else if (typeStr == "UNKNOWN") {
        type = UNKNOWN();
      } else if (typeStr == "GEOMETRY") {
        type = GEOMETRY();
      } else if (typeStr == "JSON") {
        type = JSON();
      } else {
        VELOX_FAIL(fmt::format("Unsupported type: {}", typeStr));
      }

      types.push_back(type);
    }

    auto rowType = ROW(std::move(names), std::move(types));

    // Create vectors for each column
    std::vector<VectorPtr> columns;
    for (size_t colIdx = 0; colIdx < rowType->size(); colIdx++) {
      const auto& type = rowType->childAt(colIdx);
      auto vector = BaseVector::create(type, batch.rows()->size(), pool);
      columns.push_back(vector);
    }

    // Populate the vectors with data
    for (size_t rowIdx = 0; rowIdx < batch.rows()->size(); rowIdx++) {
      const auto& row = batch.rows()[rowIdx];

      for (size_t colIdx = 0;
           colIdx < std::min(row.cells()->size(), columns.size());
           colIdx++) {
        const auto& cell = row.cells()[colIdx];
        auto& vector = columns[colIdx];

        if (*cell.isNull()) {
          vector->setNull(rowIdx, true);
        } else {
          const auto& value = *cell.value();

          switch (vector->typeKind()) {
            case TypeKind::BOOLEAN:
              if (value.boolValue_ref().has_value()) {
                vector->as<FlatVector<bool>>()->set(
                    rowIdx, *value.boolValue_ref());
              }
              break;
            case TypeKind::TINYINT:
              if (value.tinyintValue_ref().has_value()) {
                vector->as<FlatVector<int8_t>>()->set(
                    rowIdx, *value.tinyintValue_ref());
              }
              break;
            case TypeKind::SMALLINT:
              if (value.smallintValue_ref().has_value()) {
                vector->as<FlatVector<int16_t>>()->set(
                    rowIdx, *value.smallintValue_ref());
              }
              break;
            case TypeKind::INTEGER:
              if (value.integerValue_ref().has_value()) {
                vector->as<FlatVector<int32_t>>()->set(
                    rowIdx, *value.integerValue_ref());
              }
              break;
            case TypeKind::BIGINT:
              if (value.bigintValue_ref().has_value()) {
                vector->as<FlatVector<int64_t>>()->set(
                    rowIdx, *value.bigintValue_ref());
              }
              break;
            case TypeKind::REAL:
              if (value.realValue_ref().has_value()) {
                vector->as<FlatVector<float>>()->set(
                    rowIdx, *value.realValue_ref());
              }
              break;
            case TypeKind::DOUBLE:
              if (value.doubleValue_ref().has_value()) {
                vector->as<FlatVector<double>>()->set(
                    rowIdx, *value.doubleValue_ref());
              }
              break;
            case TypeKind::VARCHAR:
              if (value.varcharValue_ref().has_value()) {
                vector->as<FlatVector<StringView>>()->set(
                    rowIdx, StringView(value.varcharValue_ref()->c_str()));
              }
              break;
            case TypeKind::VARBINARY:
              if (value.varbinaryValue_ref().has_value()) {
                const auto& binValue = *value.varbinaryValue_ref();
                vector->as<FlatVector<StringView>>()->set(
                    rowIdx, StringView(binValue.data(), binValue.size()));
              }
              break;
            case TypeKind::TIMESTAMP:
              if (value.timestampValue_ref().has_value()) {
                vector->as<FlatVector<Timestamp>>()->set(
                    rowIdx, Timestamp::fromMillis(*value.timestampValue_ref()));
              }
              break;
            default:
              VELOX_FAIL(
                  fmt::format("Unsupported type: {}", vector->typeKind()));
          }
        }
      }
    }

    // Create a row vector from the columns
    auto rowVector = std::make_shared<RowVector>(
        pool, rowType, nullptr, batch.rows()->size(), std::move(columns));
    queryResults.push_back(rowVector);
  }

  return queryResults;
}

// Create a Thrift client
std::shared_ptr<LocalRunnerServiceAsyncClient> createThriftClient(
    const std::string& host,
    int port,
    std::chrono::milliseconds timeout,
    folly::EventBase* evb) {
  folly::SocketAddress addr(host, port);
  auto socket = folly::AsyncSocket::newSocket(evb, addr, timeout.count());
  auto channel =
      apache::thrift::RocketClientChannel::newChannel(std::move(socket));
  return std::make_shared<LocalRunnerServiceAsyncClient>(std::move(channel));
}

std::vector<RowVectorPtr> VeloxLocalQueryRunner::executeSerializedPlan(
    const std::string& serializedPlan) {
  std::string queryId = fmt::format("velox_local_query_runner_{}", rand());

  auto client =
      createThriftClient(thriftHost_, thriftPort_, timeout_, &eventBase_);

  // Create the request
  ExecutePlanRequest request;
  request.serializedPlan() = serializedPlan;
  request.queryId() = queryId;
  request.numWorkers() = 4; // Default value
  request.numDrivers() = 2; // Default value

  // Send the request
  ExecutePlanResponse response;
  try {
    client->sync_executePlan(response, request);
  } catch (const std::exception& e) {
    VELOX_FAIL("Thrift request failed: {}", e.what());
  }

  // Check for errors
  if (!*response.success()) {
    VELOX_FAIL(
        "Query execution failed: {}",
        response.errorMessage().has_value() ? *response.errorMessage()
                                            : "Unknown error");
  }

  // Convert the results to RowVectorPtr
  return convertToRowVectors(*response.resultBatches(), pool_.get());
}

} // namespace facebook::velox::exec::test
