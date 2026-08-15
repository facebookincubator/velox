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

#include "velox/exec/fuzzer/if/gen-cpp2/LocalRunnerService.h"
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <folly/json.h>
#include <folly/portability/GFlags.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrift/lib/cpp2/server/ThriftServer.h>
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/exec/fuzzer/LocalRunnerService.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/serializers/PrestoSerializer.h"

using namespace facebook::velox;
using namespace facebook::velox::runner;

namespace facebook::velox::runner {
namespace {

class StdoutCapture {
 public:
  StdoutCapture() {
    oldCoutBuf_ = std::cout.rdbuf();
    std::cout.rdbuf(buffer_.rdbuf());
  }
  ~StdoutCapture() {
    std::cout.rdbuf(oldCoutBuf_);
  }
  std::string str() {
    return buffer_.str();
  }

 private:
  std::stringstream buffer_;
  std::streambuf* oldCoutBuf_;
};

std::pair<RowVectorPtr, std::string> execute(
    const std::string& serializedPlan,
    const std::string& queryId,
    std::shared_ptr<memory::MemoryPool> pool) {
  StdoutCapture stdoutCapture;

  core::PlanNodePtr plan;
  try {
    folly::dynamic planJson = folly::parseJson(serializedPlan);
    VLOG(1) << "Deserializing plan:\n" << serializedPlan;
    plan = core::PlanNode::deserialize<core::PlanNode>(planJson, pool.get());
  } catch (const std::exception& e) {
    throw std::runtime_error(
        fmt::format("Failed to deserialize plan: {}", e.what()));
  }
  VLOG(1) << "Deserialized plan:\n" << plan->toString(true, true);

  try {
    exec::test::AssertQueryBuilder queryBuilder(plan);
    queryBuilder.config("session_timezone", "America/Los_Angeles");

    std::shared_ptr<exec::Task> task;
    auto results = queryBuilder.copyResults(pool.get(), task);

    return {results, stdoutCapture.str()};
  } catch (const std::exception& e) {
    throw std::runtime_error(
        fmt::format("Error executing query: {}", e.what()));
  }
}

} // namespace

std::string serializeBatch(
    const RowVectorPtr& rowVector,
    memory::MemoryPool* pool) {
  std::ostringstream out;

  OStreamOutputStream outputStream(&out);

  auto serde = std::make_unique<serializer::presto::PrestoVectorSerde>();
  serializer::presto::PrestoVectorSerde::PrestoOptions options;

  auto serializer = serde->createBatchSerializer(pool, &options);
  serializer->serialize(rowVector, &outputStream);

  return out.str();
}

std::vector<Batch> convertToBatches(
    const std::vector<RowVectorPtr>& rowVectors,
    memory::MemoryPool* pool) {
  std::vector<Batch> results;

  if (rowVectors.empty()) {
    return results;
  }

  auto leafPool = pool->addLeafChild("batchSerialization");

  for (const auto& rowVector : rowVectors) {
    Batch result;
    const auto& rowType = rowVector->type()->asRow();

    for (auto i = 0; i < rowType.size(); ++i) {
      result.columnNames()->push_back(rowType.nameOf(i));
      result.columnTypes()->push_back(rowType.childAt(i)->toString());
    }

    std::string serializedData = serializeBatch(rowVector, leafPool.get());
    result.serializedData() = std::move(serializedData);

    results.push_back(std::move(result));
  }

  return results;
}

void LocalRunnerServiceHandler::execute(
    ExecutePlanResponse& response,
    std::unique_ptr<ExecutePlanRequest> request) {
  VLOG(1) << "Received executePlan request";

  auto rootPool = memory::memoryManager()->addRootPool();
  auto pool = rootPool->addLeafChild("localRunnerHandler");

  RowVectorPtr results;
  std::string output;

  try {
    VLOG(1) << "Executing plan in service handler";
    std::tie(results, output) =
        ::execute(*request->serializedPlan(), *request->queryId(), pool);

    VLOG(1) << fmt::format(
        "Result:\nresult rowVector: {}\nstdout: {}",
        results->toString(true),
        output);
  } catch (const std::exception& e) {
    VLOG(1) << "Exception executing plan: " << e.what();
    response.success() = false;
    response.errorMessage() = e.what();
    return;
  }

  VLOG(1) << "Converting results to Thrift response";
  auto resultBatches = convertToBatches({results}, rootPool.get());
  response.results() = std::move(resultBatches);
  response.output() = output;
  response.success() = true;
  VLOG(1) << "Response sent";
}

} // namespace facebook::velox::runner
