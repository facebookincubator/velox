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

#include "velox/runner/if/gen-cpp2/LocalRunnerService.h"
#include <folly/init/Init.h>
#include <folly/json.h>
#include <folly/portability/GFlags.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrift/lib/cpp2/server/ThriftServer.h>
#include <memory>
#include <string>
#include "axiom/runner/LocalRunner.h"
#include "velox/expression/EvalCtx.h"
#include "velox/functions/facebook/prestosql/Register.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/runner/tests/LocalRunnerService.h"

using namespace facebook::velox;
using namespace facebook::velox::runner;

DEFINE_int32(
    port,
    9091,
    "LocalRunnerService port number to be used in conjunction with ExpressionFuzzerTest flag local_runner_port.");

namespace {

std::vector<core::TableScanNodePtr> collectScans(
    const core::PlanNodePtr& node) {
  std::vector<core::TableScanNodePtr> scans;

  if (auto tableScan =
          std::dynamic_pointer_cast<const core::TableScanNode>(node)) {
    scans.push_back(tableScan);
  }

  for (const auto& source : node->sources()) {
    auto sourceScans = collectScans(source);
    scans.insert(scans.end(), sourceScans.begin(), sourceScans.end());
  }

  return scans;
}

std::shared_ptr<memory::MemoryPool> makeRootPool(const std::string& queryId) {
  static std::atomic_uint64_t poolId{0};
  return memory::memoryManager()->addRootPool(
      fmt::format("{}_{}", queryId, poolId++));
}

std::vector<RowVectorPtr> readCursor(
    std::shared_ptr<facebook::axiom::runner::LocalRunner>& runner,
    memory::MemoryPool* pool) {
  std::vector<RowVectorPtr> result;
  while (auto rows = runner->next()) {
    if (auto rowVector = std::dynamic_pointer_cast<RowVector>(
            BaseVector::copy(*rows, pool))) {
      result.push_back(rowVector);
    }
  }
  return result;
}

facebook::axiom::runner::MultiFragmentPlanPtr createSingleFragmentPlan(
    const core::PlanNodePtr& plan,
    const std::string& queryId) {
  facebook::axiom::runner::MultiFragmentPlan::Options options = {
      .queryId = queryId, .numWorkers = 1, .numDrivers = 1};

  facebook::axiom::runner::ExecutableFragment fragment{queryId};
  fragment.width = 1;
  fragment.fragment = core::PlanFragment{plan};
  fragment.scans = collectScans(plan);

  return std::make_shared<facebook::axiom::runner::MultiFragmentPlan>(
      std::vector<facebook::axiom::runner::ExecutableFragment>{fragment},
      std::move(options));
}

std::shared_ptr<facebook::axiom::runner::SimpleSplitSourceFactory>
createEmptySplitSourceFactory() {
  folly::F14FastMap<
      core::PlanNodeId,
      std::vector<std::shared_ptr<connector::ConnectorSplit>>>
      nodeSplitMap;
  return std::make_shared<facebook::axiom::runner::SimpleSplitSourceFactory>(
      std::move(nodeSplitMap));
}

std::pair<std::vector<RowVectorPtr>, std::string> execute(
    const std::string& serializedPlan,
    const std::string& queryId,
    std::shared_ptr<memory::MemoryPool> pool,
    std::shared_ptr<facebook::velox::core::QueryCtx> context) {
  StdoutCapture stdoutCapture;

  core::PlanNodePtr plan;
  try {
    folly::dynamic planJson = folly::parseJson(serializedPlan);
    plan = core::PlanNode::deserialize<core::PlanNode>(planJson, pool.get());
  } catch (const std::exception& e) {
    throw std::runtime_error(
        fmt::format("Failed to deserialize plan: {}", e.what()));
  }
  VLOG(1) << "Deserialized plan:\n" << plan->toString(true, true);

  auto multiFragmentPlan = createSingleFragmentPlan(plan, queryId);

  auto localRunner = std::make_shared<facebook::axiom::runner::LocalRunner>(
      multiFragmentPlan, context, createEmptySplitSourceFactory());
  localRunner->setSingleThreadedExecution();

  std::vector<RowVectorPtr> results;
  try {
    results = readCursor(localRunner, pool.get());
    localRunner->waitForCompletion(500'000);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        fmt::format("Error executing query: {}", e.what()));
  }

  return {results, stdoutCapture.str()};
}
} // namespace

class LocalRunnerServiceHandler : public LocalRunnerServiceSvIf {
 public:
  void execute(
      ExecutePlanResponse& response,
      std::unique_ptr<ExecutePlanRequest> request) override {
    VLOG(1) << "Received executePlan request";

    std::shared_ptr<memory::MemoryPool> rootPool =
        makeRootPool(*request->queryId());
    std::shared_ptr<memory::MemoryPool> pool =
        memory::memoryManager()->addLeafPool();
    auto context = core::QueryCtx::create();

    std::vector<RowVectorPtr> results;
    std::string output;

    try {
      VLOG(1) << "Executing plan in service handler";

      auto [executionResults, capturedOutput] = ::execute(
          *request->serializedPlan(), *request->queryId(), pool, context);
      results = std::move(executionResults);
      output = std::move(capturedOutput);

      std::ostringstream result;
      result << "Result:";
      for (const auto& rowVector : results) {
        result << "\nresult rowVector: " << rowVector->toString(true);
      }
      result << "\nstdout: " << output;
      VLOG(1) << result.str();
    } catch (const std::exception& e) {
      VLOG(1) << "Exception executing plan: " << e.what();
      response.success() = false;
      response.errorMessage() = e.what();
      return;
    }

    core::ExecCtx execCtx(pool.get(), context.get());
    exec::EvalCtx evalCtx(&execCtx);

    VLOG(1) << "Converting results to Thrift response";
    auto resultBatches = convertToBatches(results, evalCtx);
    for (auto& batch : resultBatches) {
      response.results()->push_back(std::move(batch));
    }
    response.output() = output;
    response.success() = true;
    VLOG(1) << "Response sent";
  }
};

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  memory::initializeMemoryManager(memory::MemoryManager::Options{});
  Type::registerSerDe();
  core::PlanNode::registerSerDe();
  core::ITypedExpr::registerSerDe();
  functions::prestosql::registerAllScalarFunctions();
  functions::prestosql::registerAllScalarFacebookOnlyFunctions("");
  functions::prestosql::registerInternalFunctions();

  std::shared_ptr<apache::thrift::ThriftServer> thriftServer =
      std::make_shared<apache::thrift::ThriftServer>();
  thriftServer->setPort(FLAGS_port);
  thriftServer->setInterface(std::make_shared<LocalRunnerServiceHandler>());
  thriftServer->setNumIOWorkerThreads(1);
  thriftServer->setNumCPUWorkerThreads(1);

  VLOG(1) << "Starting LocalRunnerService";
  thriftServer->serve();

  return 0;
}
