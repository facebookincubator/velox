/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <fmt/format.h>
#include <folly/Uri.h>
#include <folly/json.h>
#include <re2/re2.h>
#include <thrift/lib/cpp2/async/RocketClientChannel.h>
#include "velox/core/PlanNode.h"
#include "velox/exec/fuzzer/if/gen-cpp2/LocalRunnerService.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/prestosql/types/BingTileType.h"
#include "velox/functions/prestosql/types/GeometryType.h"
#include "velox/functions/prestosql/types/IPAddressType.h"
#include "velox/functions/prestosql/types/IPPrefixType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/functions/prestosql/types/UuidType.h"
#include "velox/functions/prestosql/types/parser/TypeParser.h"
#include "velox/serializers/PrestoSerializer.h"

using namespace facebook::velox::runner;

namespace facebook::velox::exec::test {

namespace {

RowTypePtr parseBatchRowType(Batch batch) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;

  for (const auto& name : *batch.columnNames()) {
    names.push_back(name);
  }

  // Clean up type strings and format according to TypeParser::parseType
  // expectation. Input types are serialized by type()->asRow() for Thrift
  // struct by LocalRunnerService in the format of SOME_COMPLEX_TYPE<TYPE_1,
  // TYPE_2, etc.> (note the '<' and '>'). And in the case of complex type ROW,
  // types follow column name and a semicolon: as an example, ROW<f0:TYPE_1,
  // f1:TYPE_2, etc.>. We need to change these particular character choices to
  // paranthesis (in the case of the angled brackets) and spaces (in the case of
  // the semicolon). As an example,
  //  MAP<VARCHAR,TIMESTAMP> -> MAP(VARCHAR, TIMESTAMP)
  //  ROW<f0:TYPE_1, f1:TYPE_2, etc.> -> ROW(f0 TYPE_1, f1 TYPE_2, etc.)
  // as expected by TypeParser::parseType. Without this cleanup, parse will
  // crash and fuzzer will fail.
  for (const auto& typeString : *batch.columnTypes()) {
    auto parsedTypeString = typeString;
    std::replace(parsedTypeString.begin(), parsedTypeString.end(), '<', '(');
    std::replace(parsedTypeString.begin(), parsedTypeString.end(), '>', ')');
    std::replace(parsedTypeString.begin(), parsedTypeString.end(), ':', ' ');
    types.push_back(functions::prestosql::parseType(parsedTypeString));
  }

  return ROW(std::move(names), std::move(types));
}

std::vector<RowVectorPtr> deserializeBatches(
    const std::vector<Batch>& resultBatches,
    memory::MemoryPool* pool) {
  std::vector<RowVectorPtr> queryResults;

  auto serde = std::make_unique<serializer::presto::PrestoVectorSerde>();
  serializer::presto::PrestoVectorSerde::PrestoOptions options;

  for (const auto& batch : resultBatches) {
    VELOX_CHECK(
        apache::thrift::is_non_optional_field_set_manually_or_by_serializer(
            batch.serializedData()));
    VELOX_CHECK(!batch.serializedData()->empty());

    // Deserialize binary data.
    const auto& serializedData = *batch.serializedData();
    ByteRange byteRange{
        reinterpret_cast<uint8_t*>(const_cast<char*>(serializedData.data())),
        static_cast<int32_t>(serializedData.length()),
        0};
    auto byteStream = std::make_unique<BufferInputStream>(
        std::vector<ByteRange>{{byteRange}});

    RowVectorPtr rowVector;
    serde->deserialize(
        byteStream.get(),
        pool,
        parseBatchRowType(batch),
        &rowVector,
        0,
        &options);

    VELOX_CHECK_NOT_NULL(rowVector);
    queryResults.push_back(rowVector);
  }

  return queryResults;
}

std::shared_ptr<apache::thrift::Client<LocalRunnerService>> createThriftClient(
    const std::string& host,
    int port,
    std::chrono::milliseconds timeout,
    folly::EventBase* evb) {
  folly::SocketAddress addr(host, port);
  auto socket = folly::AsyncSocket::newSocket(evb, addr, timeout.count());
  auto channel =
      apache::thrift::RocketClientChannel::newChannel(std::move(socket));
  return std::make_shared<apache::thrift::Client<LocalRunnerService>>(
      std::move(channel));
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
      TIMESTAMP(),
      TIMESTAMP_WITH_TIME_ZONE(),
      IPADDRESS(),
      UUID(),
      // https://github.com/facebookincubator/velox/issues/15379 (IPPREFIX)
      // https://github.com/facebookincubator/velox/issues/15380 (Non-orderable
      // custom types such as HYPERLOGLOG, JSON, BINGTILE, GEOMETRY, etc.)
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
  VELOX_FAIL("VeloxQueryRunner does not support SQL conversion");
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
    client->sync_execute(response, request);
  } catch (const std::exception& e) {
    VELOX_FAIL("Thrift request failed: {}", e.what());
  }

  // Handle the response
  if (*response.success()) {
    LOG(INFO) << "Reference eval succeeded.";
    return std::make_pair(
        exec::test::materialize(
            deserializeBatches(*response.results(), pool_.get())),
        ReferenceQueryErrorCode::kSuccess);
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
