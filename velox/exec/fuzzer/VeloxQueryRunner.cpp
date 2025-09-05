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

#include "velox/exec/fuzzer/VeloxQueryRunner.h"

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
#include "velox/type/parser/TypeParser.h"

using namespace facebook::velox::runner;

namespace facebook::velox::exec::test {

namespace {
// Forward declaration for recursive conversion
void setValue(
    VectorPtr vector,
    vector_size_t index,
    const Value& value,
    memory::MemoryPool* pool);

// Shared function to convert scalar value from thrift to Velox vector element
void setScalarValue(
    VectorPtr vector,
    vector_size_t index,
    const ScalarValue& scalarValue) {
  switch (vector->typeKind()) {
    case TypeKind::BOOLEAN:
      if (scalarValue.boolValue_ref().has_value()) {
        vector->as<FlatVector<bool>>()->set(
            index, *scalarValue.boolValue_ref());
      }
      break;
    case TypeKind::TINYINT:
      if (scalarValue.tinyintValue_ref().has_value()) {
        vector->as<FlatVector<int8_t>>()->set(
            index, *scalarValue.tinyintValue_ref());
      }
      break;
    case TypeKind::SMALLINT:
      if (scalarValue.smallintValue_ref().has_value()) {
        vector->as<FlatVector<int16_t>>()->set(
            index, *scalarValue.smallintValue_ref());
      }
      break;
    case TypeKind::INTEGER:
      if (scalarValue.integerValue_ref().has_value()) {
        vector->as<FlatVector<int32_t>>()->set(
            index, *scalarValue.integerValue_ref());
      }
      break;
    case TypeKind::BIGINT:
      if (scalarValue.bigintValue_ref().has_value()) {
        vector->as<FlatVector<int64_t>>()->set(
            index, *scalarValue.bigintValue_ref());
      }
      break;
    case TypeKind::REAL:
      if (scalarValue.realValue_ref().has_value()) {
        vector->as<FlatVector<float>>()->set(
            index, *scalarValue.realValue_ref());
      }
      break;
    case TypeKind::DOUBLE:
      if (scalarValue.doubleValue_ref().has_value()) {
        vector->as<FlatVector<double>>()->set(
            index, *scalarValue.doubleValue_ref());
      }
      break;
    case TypeKind::VARCHAR:
      if (scalarValue.varcharValue_ref().has_value()) {
        vector->as<FlatVector<StringView>>()->set(
            index, StringView(scalarValue.varcharValue_ref()->c_str()));
      }
      break;
    case TypeKind::VARBINARY:
      if (scalarValue.varbinaryValue_ref().has_value()) {
        const auto& binValue = *scalarValue.varbinaryValue_ref();
        vector->as<FlatVector<StringView>>()->set(
            index, StringView(binValue.data(), binValue.size()));
      }
      break;
    case TypeKind::TIMESTAMP:
      if (scalarValue.timestampValue_ref().has_value()) {
        const auto& ts = *scalarValue.timestampValue_ref();
        vector->as<FlatVector<facebook::velox::Timestamp>>()->set(
            index,
            facebook::velox::Timestamp(
                ts.seconds_ref().value(), ts.nanos_ref().value()));
      }
      break;
    default:
      VELOX_FAIL(
          fmt::format("Unsupported scalar type: {}", vector->typeKind()));
  }
}

// Convert complex value from thrift to Velox vector element
void setComplexValue(
    VectorPtr vector,
    vector_size_t index,
    const ComplexValue& complexValue,
    memory::MemoryPool* pool) {
  switch (vector->typeKind()) {
    case TypeKind::ARRAY: {
      if (complexValue.arrayValue_ref().has_value()) {
        auto arrayVector = vector->as<ArrayVector>();
        const auto& arrayValue = *complexValue.arrayValue_ref();
        const auto& elements = *arrayValue.values();

        // Set array offset and size
        arrayVector->setOffsetAndSize(
            index, arrayVector->elements()->size(), elements.size());

        // Add elements to the elements vector
        auto elementsVector = arrayVector->elements();
        auto currentSize = elementsVector->size();
        elementsVector->resize(currentSize + elements.size());

        for (auto i = 0; i < elements.size(); ++i) {
          setValue(elementsVector, currentSize + i, elements[i], pool);
        }
      }
      break;
    }
    case TypeKind::MAP: {
      if (complexValue.mapValue_ref().has_value()) {
        auto mapVector = vector->as<MapVector>();
        const auto& mapValue = *complexValue.mapValue_ref();
        const auto& mapEntries = *mapValue.values();

        // Set map offset and size
        mapVector->setOffsetAndSize(
            index, mapVector->mapKeys()->size(), mapEntries.size());

        // Add keys and values to the respective vectors
        auto keysVector = mapVector->mapKeys();
        auto valuesVector = mapVector->mapValues();
        auto currentSize = keysVector->size();
        keysVector->resize(currentSize + mapEntries.size());
        valuesVector->resize(currentSize + mapEntries.size());

        auto entryIdx = 0;
        for (const auto& [key, value] : mapEntries) {
          setValue(keysVector, currentSize + entryIdx, key, pool);
          setValue(valuesVector, currentSize + entryIdx, value, pool);
          ++entryIdx;
        }
      }
      break;
    }
    case TypeKind::ROW: {
      if (complexValue.rowValue_ref().has_value()) {
        auto rowVector = vector->as<RowVector>();
        const auto& rowValue = *complexValue.rowValue_ref();
        const auto& fieldValues = *rowValue.fieldValues();

        // Set values for each field
        for (auto i = 0;
             i < fieldValues.size() && i < rowVector->childrenSize();
             ++i) {
          auto childVector = rowVector->childAt(i);
          setValue(childVector, index, fieldValues[i], pool);
        }
      }
      break;
    }
    default:
      VELOX_FAIL(
          fmt::format("Unsupported complex type: {}", vector->typeKind()));
  }
}

// Recursive function to set any value (scalar or complex)
void setValue(
    VectorPtr vector,
    vector_size_t index,
    const Value& value,
    memory::MemoryPool* pool) {
  if (value.isNull().has_value() && value.isNull().value()) {
    vector->setNull(index, true);
  } else if (value.scalarValue_ref().has_value()) {
    setScalarValue(vector, index, *value.scalarValue_ref());
  } else {
    setComplexValue(vector, index, *value.complexValue_ref(), pool);
  }
}

// Convert Column as defined by thrift protocol to Velox Vector.
VectorPtr deserializeColumn(
    const TypePtr type,
    const Column& column,
    memory::MemoryPool* pool) {
  auto vector = BaseVector::create(type, (*column.columnRows()).size(), pool);

  // Populate the vector with data from the column
  for (auto i = 0; i < (*column.columnRows()).size(); ++i) {
    const auto& columnRow = (*column.columnRows())[i];

    if (columnRow.isNull().value()) {
      vector->setNull(i, true);
    } else if (columnRow.value_ref().has_value()) {
      const auto& value = *columnRow.value_ref();
      setValue(vector, i, value, pool);
    }
  }

  return vector;
}

// Iterate through the columns of ResultBatch and create RowVectors.
std::vector<RowVectorPtr> convertToRowVectors(
    const std::vector<ResultBatch>& resultBatches,
    memory::MemoryPool* pool) {
  std::vector<RowVectorPtr> queryResults;

  for (const auto& batch : resultBatches) {
    if (batch.columns()->empty() || batch.columnNames()->empty() ||
        batch.columnTypes()->empty()) {
      continue;
    }

    std::vector<TypePtr> types;
    std::vector<VectorPtr> columns;
    vector_size_t size = 0;

    for (auto i = 0; i < (*batch.columns()).size(); i++) {
      types.push_back(parseType((*batch.columnTypes())[i]));
      columns.push_back(
          deserializeColumn(types.back(), (*batch.columns())[i], pool));
      size = std::max(size, columns.back()->size());
    }

    auto rowVector = std::make_shared<RowVector>(
        pool,
        ROW(*batch.columnNames(), std::move(types)),
        nullptr,
        size,
        std::move(columns));
    queryResults.push_back(rowVector);
  }

  return queryResults;
}

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
} // namespace

VeloxQueryRunner::VeloxQueryRunner(
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

const std::vector<TypePtr>& VeloxQueryRunner::supportedScalarTypes() const {
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
  };
  return kScalarTypes;
}

const std::unordered_map<std::string, DataSpec>&
VeloxQueryRunner::aggregationFunctionDataSpecs() const {
  static const std::unordered_map<std::string, DataSpec>
      kAggregationFunctionDataSpecs{};
  return kAggregationFunctionDataSpecs;
}

std::optional<std::string> VeloxQueryRunner::toSql(
    const core::PlanNodePtr& /*plan*/) {
  // We don't need to convert to SQL for VeloxQueryRunner
  // as we're sending the serialized plan directly
  return std::nullopt;
}

bool VeloxQueryRunner::isConstantExprSupported(
    const core::TypedExprPtr& /*expr*/) {
  // Since we're using Velox directly, we support all constant expressions
  return true;
}

bool VeloxQueryRunner::isSupported(
    const exec::FunctionSignature& /*signature*/) {
  // Since we're using Velox directly, we support all function signatures
  return true;
}

std::vector<RowVectorPtr> VeloxQueryRunner::execute(
    const std::string& /*sql*/) {
  VELOX_FAIL("VeloxQueryRunner does not support SQL execution");
}

std::vector<RowVectorPtr> VeloxQueryRunner::execute(
    const std::string& /*sql*/,
    const std::string& /*sessionProperty*/) {
  VELOX_FAIL("VeloxQueryRunner does not support SQL execution");
}

std::pair<
    std::optional<std::multiset<std::vector<velox::variant>>>,
    ReferenceQueryErrorCode>
VeloxQueryRunner::execute(const core::PlanNodePtr& plan) {
  auto serializedPlan = serializePlan(plan);
  auto queryId = fmt::format("velox_local_query_runner_{}", rand());

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

  // Handle the response
  if (*response.success()) {
    LOG(INFO) << "Reference eval succeeded.";
    return std::make_pair(
        exec::test::materialize(
            convertToRowVectors(*response.resultBatches(), pool_.get())),
        ReferenceQueryErrorCode::kSuccess);
    ;
  } else {
    LOG(INFO) << "Reference eval failed.";
    return std::make_pair(
        std::nullopt, ReferenceQueryErrorCode::kReferenceQueryFail);
  }
}

std::string VeloxQueryRunner::serializePlan(const core::PlanNodePtr& plan) {
  // Serialize the plan to JSON
  folly::dynamic serializedPlan = plan->serialize();
  return folly::toJson(serializedPlan);
}

} // namespace facebook::velox::exec::test
