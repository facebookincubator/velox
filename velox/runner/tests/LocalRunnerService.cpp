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
#include "velox/functions/facebook/prestosql/Register.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::runner;

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

// Forward declaration
void convertValue(VectorPtr vector, Value& value, vector_size_t rowIdx);

ScalarValue handleScalarType(VectorPtr vector, vector_size_t rowIdx) {
  ScalarValue scalar;

  switch (vector->typeKind()) {
    case TypeKind::BOOLEAN:
      scalar.boolValue_ref() =
          vector->as<SimpleVector<bool>>()->valueAt(rowIdx);
      break;
    case TypeKind::TINYINT:
      scalar.tinyintValue_ref() =
          vector->as<SimpleVector<int8_t>>()->valueAt(rowIdx);
      break;
    case TypeKind::SMALLINT:
      scalar.smallintValue_ref() =
          vector->as<SimpleVector<int16_t>>()->valueAt(rowIdx);
      break;
    case TypeKind::INTEGER:
      scalar.integerValue_ref() =
          vector->as<SimpleVector<int32_t>>()->valueAt(rowIdx);
      break;
    case TypeKind::BIGINT:
      scalar.bigintValue_ref() =
          vector->as<SimpleVector<int64_t>>()->valueAt(rowIdx);
      break;
    case TypeKind::REAL:
      scalar.realValue_ref() =
          vector->as<SimpleVector<float>>()->valueAt(rowIdx);
      break;
    case TypeKind::DOUBLE:
      scalar.doubleValue_ref() =
          vector->as<SimpleVector<double>>()->valueAt(rowIdx);
      break;
    case TypeKind::VARCHAR:
      scalar.varcharValue_ref() =
          vector->as<SimpleVector<StringView>>()->valueAt(rowIdx).str();
      break;
    case TypeKind::VARBINARY: {
      const auto& binValue =
          vector->as<SimpleVector<StringView>>()->valueAt(rowIdx);
      scalar.varbinaryValue_ref() =
          std::string(binValue.data(), binValue.size());
      break;
    }
    case TypeKind::TIMESTAMP: {
      const auto& ts =
          vector->as<SimpleVector<facebook::velox::Timestamp>>()->valueAt(
              rowIdx);
      facebook::velox::runner::Timestamp timestampValue;
      timestampValue.seconds_ref() = ts.getSeconds();
      timestampValue.nanos_ref() = ts.getNanos();
      scalar.timestampValue_ref() = std::move(timestampValue);
      break;
    }
    case TypeKind::HUGEINT: {
      const auto& hugeint =
          vector->as<SimpleVector<int128_t>>()->valueAt(rowIdx);
      facebook::velox::runner::i128 hugeintValue;
      hugeintValue.msb_ref() = static_cast<int64_t>(hugeint >> 64);
      hugeintValue.lsb_ref() =
          static_cast<int64_t>(hugeint & 0xFFFFFFFFFFFFFFFFULL);
      scalar.hugeintValue_ref() = std::move(hugeintValue);
      break;
    }
    default:
      VELOX_FAIL(fmt::format("Unsupported scalar type: {}", vector->type()));
  }

  return scalar;
}

ComplexValue handleComplexType(VectorPtr vector, vector_size_t rowIdx) {
  ComplexValue complex;

  switch (vector->typeKind()) {
    case TypeKind::ARRAY: {
      auto arrayVector = vector->as<ArrayVector>();
      auto elements = arrayVector->elements();
      auto offset = arrayVector->offsetAt(rowIdx);
      auto size = arrayVector->sizeAt(rowIdx);

      facebook::velox::runner::Array arrayValue;

      for (auto i = 0; i < size; ++i) {
        auto elementIdx = offset + i;

        Value elementValue;
        if (elements->isNullAt(elementIdx)) {
          elementValue.isNull() = true;
        } else {
          convertValue(elements, elementValue, elementIdx);
        }
        arrayValue.values()->push_back(std::move(elementValue));
      }

      complex.arrayValue_ref() = std::move(arrayValue);
      break;
    }
    case TypeKind::MAP: {
      auto mapVector = vector->as<MapVector>();
      auto keys = mapVector->mapKeys();
      auto values = mapVector->mapValues();
      auto offset = mapVector->offsetAt(rowIdx);
      auto size = mapVector->sizeAt(rowIdx);

      facebook::velox::runner::Map mapValue;

      for (auto i = 0; i < size; ++i) {
        Value keyValue, valueValue;

        // Key is always non-null
        convertValue(keys, keyValue, offset + i);
        if (values->isNullAt(offset + i)) {
          valueValue.isNull() = true;
        } else {
          convertValue(values, valueValue, offset + i);
        }
        (*mapValue.values())[std::move(keyValue)] = std::move(valueValue);
      }

      complex.mapValue_ref() = std::move(mapValue);
      break;
    }
    case TypeKind::ROW: {
      auto rowVector = vector->as<RowVector>();
      facebook::velox::runner::Row rowValue;

      const auto& rowType = rowVector->type()->asRow();
      for (auto i = 0; i < rowType.size(); ++i) {
        rowValue.filedNames()->push_back(rowType.nameOf(i));

        auto childVector = rowVector->childAt(i);

        Value fieldValue;
        if (childVector->isNullAt(rowIdx)) {
          fieldValue.isNull() = true;
        } else {
          convertValue(childVector, fieldValue, rowIdx);
        }
        rowValue.fieldValues()->push_back(std::move(fieldValue));
      }

      complex.rowValue_ref() = std::move(rowValue);
      break;
    }
    default:
      VELOX_FAIL(fmt::format("Unsupported complex type: {}", vector->type()));
  }

  return complex;
}

// Recursive function to convert any value (scalar or complex)
void convertValue(VectorPtr vector, Value& value, vector_size_t rowIdx) {
  if (vector->isNullAt(rowIdx)) {
    value.isNull() = true;
  } else {
    value.isNull() = false;
    switch (vector->typeKind()) {
      case TypeKind::BOOLEAN:
      case TypeKind::TINYINT:
      case TypeKind::SMALLINT:
      case TypeKind::INTEGER:
      case TypeKind::BIGINT:
      case TypeKind::REAL:
      case TypeKind::DOUBLE:
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
      case TypeKind::TIMESTAMP:
      case TypeKind::HUGEINT:
        value.scalarValue_ref() = handleScalarType(vector, rowIdx);
        break;
      case TypeKind::ARRAY:
      case TypeKind::MAP:
      case TypeKind::ROW:
        value.complexValue_ref() = handleComplexType(vector, rowIdx);
        break;
      default:
        VELOX_FAIL(fmt::format("Unsupported type: {}", vector->type()));
    }
  }
}

std::vector<Value> serializeVector(VectorPtr vector, vector_size_t size) {
  std::vector<Value> rows;
  for (vector_size_t rowIdx = 0; rowIdx < size; ++rowIdx) {
    Value value;
    convertValue(vector, value, rowIdx);
    rows.push_back(value);
  }
  return rows;
}

// Convert Velox RowVectors to Thrift ResultBatches
std::vector<ResultBatch> convertToResultBatches(
    const std::vector<RowVectorPtr>& rowVectors) {
  std::vector<ResultBatch> resultBatches;

  if (rowVectors.empty()) {
    return resultBatches;
  }

  for (const auto& rowVector : rowVectors) {
    ResultBatch resultBatch;
    const auto& rowType = rowVector->type()->asRow();

    for (auto i = 0; i < rowType.size(); ++i) {
      resultBatch.columnNames()->push_back(rowType.nameOf(i));
      resultBatch.columnTypes()->push_back(rowType.childAt(i)->toString());
    }

    resultBatch.numRows() = rowVector->size();

    const auto numColumns = rowVector->childrenSize();
    resultBatch.columns()->resize(numColumns);

    for (auto colIdx = 0; colIdx < numColumns; ++colIdx) {
      (*resultBatch.columns())[colIdx].rows() =
          serializeVector(rowVector->childAt(colIdx), rowVector->size());
    }

    resultBatches.push_back(std::move(resultBatch));
  }

  return resultBatches;
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
  std::unordered_map<std::string, std::string> hiveConfig;
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs;
  connectorConfigs["test-hive"] =
      std::make_shared<config::ConfigBase>(std::move(hiveConfig));

  return core::QueryCtx::create(
      executor, core::QueryConfig({}), std::move(connectorConfigs));
}

facebook::axiom::runner::MultiFragmentPlanPtr createSingleFragmentPlan(
    const core::PlanNodePtr& plan,
    const std::string& queryId,
    int32_t numWorkers,
    int32_t numDrivers) {
  facebook::axiom::runner::MultiFragmentPlan::Options options = {
      .queryId = queryId, .numWorkers = numWorkers, .numDrivers = numDrivers};

  facebook::axiom::runner::ExecutableFragment fragment{queryId};
  fragment.width = 1;
  fragment.fragment = core::PlanFragment{plan};

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

  return std::make_shared<facebook::axiom::runner::MultiFragmentPlan>(
      std::vector<facebook::axiom::runner::ExecutableFragment>{fragment},
      std::move(options));
}

std::shared_ptr<facebook::axiom::runner::SimpleSplitSourceFactory>
createEmptySplitSourceFactory() {
  std::unordered_map<
      core::PlanNodeId,
      std::vector<std::shared_ptr<connector::ConnectorSplit>>>
      nodeSplitMap;
  return std::make_shared<facebook::axiom::runner::SimpleSplitSourceFactory>(
      std::move(nodeSplitMap));
}

std::pair<std::vector<RowVectorPtr>, std::string> executePlan(
    const std::string& serializedPlan,
    const std::string& queryId,
    int32_t numWorkers,
    int32_t numDrivers,
    std::shared_ptr<folly::CPUThreadPoolExecutor> executor,
    std::shared_ptr<memory::MemoryPool> rootPool,
    std::shared_ptr<memory::MemoryPool> pool) {
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

  auto multiFragmentPlan =
      createSingleFragmentPlan(plan, queryId, numWorkers, numDrivers);

  executor = std::make_shared<folly::CPUThreadPoolExecutor>(4);
  auto localRunner = std::make_shared<facebook::axiom::runner::LocalRunner>(
      multiFragmentPlan,
      makeQueryCtx(queryId, rootPool.get(), executor.get()),
      createEmptySplitSourceFactory());

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
  void executePlan(
      ExecutePlanResponse& response,
      std::unique_ptr<ExecutePlanRequest> request) override {
    VLOG(1) << "Received executePlan request";

    std::shared_ptr<memory::MemoryPool> rootPool =
        makeRootPool(*request->queryId());
    std::shared_ptr<memory::MemoryPool> pool =
        memory::memoryManager()->addLeafPool("output");

    std::vector<RowVectorPtr> results;
    std::string output;
    try {
      VLOG(1) << "Executing plan in service handler";
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

    VLOG(1) << "Converting results to Thrift response";
    auto resultBatches = convertToResultBatches(results);
    for (auto& batch : resultBatches) {
      response.resultBatches()->push_back(std::move(batch));
    }
    response.output() = output;
    response.success() = true;
    VLOG(1) << "Response sent";
  }

 private:
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
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
  thriftServer->setNumIOWorkerThreads(4);
  thriftServer->setNumCPUWorkerThreads(4);

  VLOG(1) << "Starting LocalRunnerService";
  thriftServer->serve();

  return 0;
}
