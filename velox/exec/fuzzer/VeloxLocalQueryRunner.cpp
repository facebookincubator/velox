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
#include <folly/json.h>
#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/vector/VectorSaver.h"

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
  eventBaseThread_.start("VeloxLocalQueryRunner");
  pool_ = aggregatePool->addLeafChild("leaf");
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

std::vector<RowVectorPtr> VeloxLocalQueryRunner::executeSerializedPlan(
    const std::string& serializedPlan) {
  // Prepare the request body
  folly::dynamic requestBody = folly::dynamic::object;
  requestBody["serialized_plan"] = serializedPlan;
  requestBody["query_id"] = fmt::format("velox_local_query_runner_{}", rand());

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

} // namespace facebook::velox::exec::test
