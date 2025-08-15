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
#include <folly/io/async/EventBase.h>
#include <folly/io/async/EventBaseManager.h>
#include <folly/json.h>
#include <folly/portability/GFlags.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrift/lib/cpp2/server/ThriftServer.h>
#include <memory>
#include <string>
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/BingTileType.h"
#include "velox/functions/prestosql/types/GeometryType.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/functions/prestosql/types/JsonType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/runner/LocalRunner.h"

using namespace facebook::velox;
using namespace facebook::velox::runner;
// using namespace proxygen;

DEFINE_int32(port, 9091, "");
DEFINE_int32(num_workers, 4, "Number of workers to use for query execution");
DEFINE_int32(num_drivers, 2, "Number of drivers per worker");

// Capture stdout to a string
class StdoutCapture {
 public:
  StdoutCapture() {
    oldCoutBuf_ = std::cout.rdbuf();
    std::cout.rdbuf(buffer_.rdbuf());
  }

  ~StdoutCapture() {
    std::cout.rdbuf(oldCoutBuf_);
  }

  std::string str() const {
    return buffer_.str();
  }

 private:
  std::stringstream buffer_;
  std::streambuf* oldCoutBuf_;
};

// Helper functions for LocalRunnerService
namespace {

// Convert Velox RowVector to Thrift ResultBatch
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
              VELOX_FAIL(fmt::format("Unsupported type: {}", vector->type()));
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

std::shared_ptr<memory::MemoryPool> makeRootPool(const std::string& queryId) {
  static std::atomic_uint64_t poolId{0};
  return memory::memoryManager()->addRootPool(
      fmt::format("{}_{}", queryId, poolId++));
}

std::vector<RowVectorPtr> readCursor(
    std::shared_ptr<runner::LocalRunner>& runner,
    memory::MemoryPool* pool) {
  // We'll check the result after tasks are deleted, so copy the result
  // vectors to 'pool' that has longer lifetime.
  std::vector<RowVectorPtr> result;
  while (auto rows = runner->next()) {
    if (auto rowVector =
            std::dynamic_pointer_cast<RowVector>(BaseVector::copy(*rows, pool)))
      result.push_back(rowVector);
  }
  return result;
}

std::shared_ptr<core::QueryCtx> makeQueryCtx(
    const std::string& queryId,
    memory::MemoryPool* rootPool,
    folly::Executor* executor) {
  std::unordered_map<std::string, std::string> config;
  std::unordered_map<std::string, std::string> hiveConfig;
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs;
  connectorConfigs["test-hive"] =
      std::make_shared<config::ConfigBase>(std::move(hiveConfig));

  return core::QueryCtx::create(
      executor,
      core::QueryConfig(config),
      std::move(connectorConfigs),
      cache::AsyncDataCache::getInstance());
}

// Create a MultiFragmentPlan from a single PlanNode
runner::MultiFragmentPlanPtr createSingleFragmentPlan(
    const core::PlanNodePtr& plan,
    const std::string& queryId,
    int32_t numWorkers,
    int32_t numDrivers) {
  runner::MultiFragmentPlan::Options options = {
      .queryId = queryId, .numWorkers = numWorkers, .numDrivers = numDrivers};

  // Create a single fragment with the given plan
  runner::ExecutableFragment fragment{queryId};
  fragment.width = 1; // Single task
  fragment.fragment = core::PlanFragment{plan};

  // Get all table scan nodes from the plan
  std::vector<core::TableScanNodePtr> scans;
  std::function<void(const core::PlanNodePtr&)> collectScans =
      [&scans, &collectScans](const core::PlanNodePtr& node) {
        if (auto tableScan =
                std::dynamic_pointer_cast<const core::TableScanNode>(node)) {
          scans.push_back(tableScan);
        }
        for (const auto& source : node->sources()) {
          collectScans(source);
        }
      };
  collectScans(plan);
  fragment.scans = std::move(scans);

  return std::make_shared<runner::MultiFragmentPlan>(
      std::vector<runner::ExecutableFragment>{fragment}, std::move(options));
}

// Create a SimpleSplitSourceFactory with empty splits
std::shared_ptr<runner::SimpleSplitSourceFactory>
createEmptySplitSourceFactory() {
  std::unordered_map<
      core::PlanNodeId,
      std::vector<std::shared_ptr<connector::ConnectorSplit>>>
      nodeSplitMap;
  return std::make_shared<runner::SimpleSplitSourceFactory>(
      std::move(nodeSplitMap));
}

// Execute a plan and return the results
std::pair<std::vector<RowVectorPtr>, std::string> executePlan(
    const std::string& serializedPlan,
    const std::string& queryId,
    int32_t numWorkers,
    int32_t numDrivers,
    std::shared_ptr<folly::CPUThreadPoolExecutor> executor,
    std::shared_ptr<memory::MemoryPool> rootPool,
    std::shared_ptr<memory::MemoryPool> pool) {
  // Capture stdout
  StdoutCapture stdoutCapture;

  // Deserialize the plan
  core::PlanNodePtr plan;
  try {
    folly::dynamic planJson = folly::parseJson(serializedPlan);
    plan = core::PlanNode::deserialize<core::PlanNode>(planJson, pool.get());
  } catch (const std::exception& e) {
    throw std::runtime_error(
        fmt::format("Failed to deserialize plan: {}", e.what()));
  }
  LOG(INFO) << "Deserialized plan:\n" << plan->toString(true, true);

  // Create a MultiFragmentPlan from the deserialized plan
  auto multiFragmentPlan =
      createSingleFragmentPlan(plan, queryId, numWorkers, numDrivers);

  // Create a LocalRunner
  executor = std::make_shared<folly::CPUThreadPoolExecutor>(4);
  auto localRunner = std::make_shared<runner::LocalRunner>(
      multiFragmentPlan,
      makeQueryCtx(queryId, rootPool.get(), executor.get()),
      createEmptySplitSourceFactory());

  // Run the query and collect results
  std::vector<RowVectorPtr> results;
  try {
    results = readCursor(localRunner, pool.get());
    localRunner->waitForCompletion(500'000); // 500ms timeout
  } catch (const std::exception& e) {
    throw std::runtime_error(
        fmt::format("Error executing query: {}", e.what()));
  }

  return {results, stdoutCapture.str()};
}
} // namespace

// Implementation of the Thrift service
class LocalRunnerServiceHandler : public LocalRunnerServiceSvIf {
 public:
  void executePlan(
      ExecutePlanResponse& response,
      std::unique_ptr<ExecutePlanRequest> request) override {
    LOG(INFO) << "Received executePlan request";

    // Create memory pools
    std::shared_ptr<memory::MemoryPool> rootPool =
        makeRootPool(*request->queryId());
    std::shared_ptr<memory::MemoryPool> pool =
        memory::memoryManager()->addLeafPool("output");

    // Execute the plan
    std::vector<RowVectorPtr> results;
    std::string output;
    try {
      LOG(INFO) << "Executing plan in service handler";
      auto [executionResults, capturedOutput] = ::executePlan(
          *request->serializedPlan(),
          *request->queryId(),
          *request->numWorkers(),
          *request->numDrivers(),
          executor_,
          rootPool,
          pool);
      results = std::move(executionResults);
      output = std::move(capturedOutput);

      // Debug logging, can be removed when service is finalized.
      std::ostringstream result;
      result << "Result:";
      for (const auto& rowVector : results) {
        result << "\nresult rowVector: " << rowVector->toString();
      }
      result << "\nstdout: " << output;
      LOG(INFO) << result.str();
    } catch (const std::exception& e) {
      LOG(INFO) << "Exception executing plan: " << e.what();
      response.success() = false;
      response.errorMessage() = e.what();
      return;
    }

    // Convert results to Thrift response
    LOG(INFO) << "Converting results to Thrift response";
    response.resultBatches()->push_back(convertToResultBatch(results));
    response.output() = output;
    response.success() = true;
    LOG(INFO) << "Response sent";
  }

 private:
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
};

int main(int argc, char** argv) {
  // Initialize gflags and glog
  folly::Init init(&argc, &argv);

  // Initialize memory manager
  memory::initializeMemoryManager(memory::MemoryManager::Options{});

  // Register file systems, connectors and functions
  filesystems::registerLocalFileSystem();
  connector::registerConnectorFactory(
      std::make_shared<connector::hive::HiveConnectorFactory>());
  dwrf::registerDwrfWriterFactory();
  Type::registerSerDe();
  core::PlanNode::registerSerDe();
  core::ITypedExpr::registerSerDe();
  functions::prestosql::registerAllScalarFunctions();

  // Create the Thrift server
  std::shared_ptr<apache::thrift::ThriftServer> thriftServer =
      std::make_shared<apache::thrift::ThriftServer>();
  thriftServer->setPort(FLAGS_port);
  thriftServer->setInterface(std::make_shared<LocalRunnerServiceHandler>());
  thriftServer->setNumIOWorkerThreads(4);
  thriftServer->setNumCPUWorkerThreads(4);

  // Start the Thrift server (this blocks)
  LOG(INFO) << "Starting LocalRunnerService";
  thriftServer->serve();

  return 0;
}
