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
#include "velox/runner/if/gen-cpp2/LocalRunnerService.h"
#include "velox/vector/VectorSaver.h"

using namespace facebook::velox::runner;

namespace facebook::velox::exec::test {

namespace {
// Helper class to handle HTTP responses
class ServerResponse {
 public:
  explicit ServerResponse(const std::string& response)
      : response_(folly::parseJson(response)) {}

  void throwIfFailed() {
    if (response_.getDefault("status", "").asString() != "success") {
      VELOX_FAIL(
          "Query failed: {}",
          response_.getDefault("message", "Unknown error").asString());
    }
  }

  folly::dynamic& results() {
    return response_["results"];
  }

 private:
  folly::dynamic response_;
};
} // namespace

VeloxLocalQueryRunner::VeloxLocalQueryRunner(
    memory::MemoryPool* aggregatePool,
    std::string serviceUri,
    std::chrono::milliseconds timeout)
    : ReferenceQueryRunner(aggregatePool),
      serviceUri_(std::move(serviceUri)),
      timeout_(timeout) {
  // eventBaseThread_.start("VeloxLocalQueryRunner");
  pool_ = aggregatePool->addLeafChild("leaf");

  // Parse the URI to determine if it's HTTP or Thrift
  try {
    folly::Uri uri(serviceUri_);
    if (uri.scheme() == "thrift") {
      // Extract host and port for Thrift
      useThrift_ = true;
      thriftHost_ = uri.host() == "" ? "127.0.0.1" : uri.host();
      thriftPort_ = uri.port() > 0 ? uri.port() : 9091; // Default Thrift port
      LOG(INFO) << thriftHost_ << ":" << thriftPort_;
    } else {
      // Use HTTP
      useThrift_ = false;
    }
  } catch (const std::exception& e) {
    // If URI parsing fails, default to HTTP
    useThrift_ = false;
    LOG(WARNING) << "Failed to parse service URI: " << e.what()
                 << ". Defaulting to HTTP.";
  }
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
      } else {
        // Default to VARCHAR for unsupported types
        type = VARCHAR();
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
            /*case TypeKind::DATE:
              if (value.dateValue_ref().has_value()) {
                vector->as<FlatVector<Date>>()->set(
                    rowIdx, Date(*value.dateValue_ref()));
              }
              break;*/
            default:
              // For unsupported types, set to null
              vector->setNull(rowIdx, true);
              break;
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

  if (useThrift_) {
    LOG(INFO) << "Using Thrift protocol for query execution";
    // Use Thrift protocol
    // auto evb = eventBaseThread_.getEventBase();
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
      client->sync_executePlan(response, request); // sync_executePlan
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
  } else {
    LOG(INFO) << "Using HTTP protocol for query execution";
    // Use HTTP protocol (existing implementation)
    // Prepare the request body
    folly::dynamic requestBody = folly::dynamic::object;
    requestBody["serialized_plan"] = serializedPlan;
    requestBody["query_id"] = queryId;

    // Send the request to the LocalRunnerService
    cpr::Url url{serviceUri_};
    cpr::Body body{folly::toJson(requestBody)};
    cpr::Header header{{"Content-Type", "application/json"}};
    cpr::Timeout timeout{timeout_};
    cpr::Response response = cpr::Post(url, body, header, timeout);

    // Check for HTTP errors
    VELOX_CHECK_EQ(
        response.status_code,
        200,
        "HTTP request failed: {} {}",
        response.status_code,
        response.error.message);

    // Parse the response
    ServerResponse serverResponse(response.text);
    serverResponse.throwIfFailed();

    // Convert the results to RowVectorPtr
    std::vector<RowVectorPtr> queryResults;
    folly::dynamic& results = serverResponse.results();

    // If there are no results, return an empty vector
    if (results.empty()) {
      return queryResults;
    }

    // Process each batch of results
    for (const auto& batch : results) {
      if (batch.empty()) {
        continue;
      }

      // Extract column names and types from the first row
      std::vector<std::string> names;
      std::vector<TypePtr> types;
      for (const auto& item : batch[0].items()) {
        names.push_back(item.first.asString());
        const auto& value = item.second;

        // Determine the type based on the value
        if (value.isNull()) {
          types.push_back(VARCHAR()); // Default to VARCHAR for null values
        } else if (value.isBool()) {
          types.push_back(BOOLEAN());
        } else if (value.isInt()) {
          types.push_back(BIGINT());
        } else if (value.isDouble()) {
          types.push_back(DOUBLE());
        } else if (value.isString()) {
          types.push_back(VARCHAR());
        } else {
          // For complex types, use VARCHAR as a fallback
          types.push_back(VARCHAR());
        }
      }

      // Create a row type
      auto rowType = ROW(std::move(names), std::move(types));

      // Create vectors for each column
      std::vector<VectorPtr> columns;
      for (size_t colIdx = 0; colIdx < rowType->size(); colIdx++) {
        const auto& type = rowType->childAt(colIdx);
        auto vector = BaseVector::create(type, batch.size(), pool_.get());
        columns.push_back(vector);
      }

      // Populate the vectors with data
      for (size_t rowIdx = 0; rowIdx < batch.size(); rowIdx++) {
        const auto& row = batch[rowIdx];
        for (size_t colIdx = 0; colIdx < rowType->size(); colIdx++) {
          const auto& name = rowType->nameOf(colIdx);
          const auto& value = row[name];
          auto& vector = columns[colIdx];

          if (value.isNull()) {
            vector->setNull(rowIdx, true);
          } else {
            switch (vector->typeKind()) {
              case TypeKind::BOOLEAN:
                vector->as<FlatVector<bool>>()->set(rowIdx, value.asBool());
                break;
              case TypeKind::BIGINT:
                vector->as<FlatVector<int64_t>>()->set(rowIdx, value.asInt());
                break;
              case TypeKind::DOUBLE:
                vector->as<FlatVector<double>>()->set(rowIdx, value.asDouble());
                break;
              case TypeKind::VARCHAR:
                vector->as<FlatVector<StringView>>()->set(
                    rowIdx, StringView(value.asString().c_str()));
                break;
              default:
                // For unsupported types, set to null
                vector->setNull(rowIdx, true);
            }
          }
        }
      }

      // Create a row vector from the columns
      auto rowVector = std::make_shared<RowVector>(
          pool_.get(), rowType, nullptr, batch.size(), std::move(columns));
      queryResults.push_back(rowVector);
    }

    return queryResults;
  }
}

} // namespace facebook::velox::exec::test
