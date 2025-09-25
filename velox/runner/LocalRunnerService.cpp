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

#include <memory>
#include <string>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <folly/json.h>
#include <folly/portability/GFlags.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrift/lib/cpp2/server/ThriftServer.h>

#include "axiom/runner/LocalRunner.h"
#include "velox/expression/EvalCtx.h"
#include "velox/functions/facebook/prestosql/Register.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/runner/LocalRunnerService.h"
#include "velox/runner/if/gen-cpp2/LocalRunnerService.h"

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

// StdoutCapture implementation
StdoutCapture::StdoutCapture() {
  oldCoutBuf_ = std::cout.rdbuf();
  std::cout.rdbuf(buffer_.rdbuf());
}

StdoutCapture::~StdoutCapture() {
  std::cout.rdbuf(oldCoutBuf_);
}

std::string StdoutCapture::str() const {
  return buffer_.str();
}

// Function implementations
namespace facebook::velox::runner {

ScalarValue getScalarValue(VectorPtr vector, vector_size_t rowIdx) {
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

ComplexValue
getComplexValue(VectorPtr vector, vector_size_t rowIdx, exec::EvalCtx evalCtx) {
  ComplexValue complex;

  exec::LocalDecodedVector decoder(
      evalCtx, *vector, SelectivityVector(vector->size()));
  auto& decoded = *decoder.get();
  rowIdx = decoded.index(rowIdx);

  switch (vector->typeKind()) {
    case TypeKind::ARRAY: {
      auto arrayVector = decoded.base()->as<ArrayVector>();
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
          convertValue(elements, elementValue, elementIdx, evalCtx);
        }
        arrayValue.values()->push_back(std::move(elementValue));
      }

      complex.arrayValue_ref() = std::move(arrayValue);
      break;
    }
    case TypeKind::MAP: {
      auto mapVector = decoded.base()->as<MapVector>();
      auto keys = mapVector->mapKeys();
      auto values = mapVector->mapValues();
      auto offset = mapVector->offsetAt(rowIdx);
      auto size = mapVector->sizeAt(rowIdx);

      facebook::velox::runner::Map mapValue;

      for (auto i = 0; i < size; ++i) {
        Value keyValue, valueValue;

        VELOX_CHECK(!(keys->isNullAt(offset + i)), "Map key cannot be null");
        convertValue(keys, keyValue, offset + i, evalCtx);
        if (values->isNullAt(offset + i)) {
          valueValue.isNull() = true;
        } else {
          convertValue(values, valueValue, offset + i, evalCtx);
        }
        (*mapValue.values())[std::move(keyValue)] = std::move(valueValue);
      }

      complex.mapValue_ref() = std::move(mapValue);
      break;
    }
    case TypeKind::ROW: {
      auto rowVector = decoded.base()->as<RowVector>();
      facebook::velox::runner::Row rowValue;

      for (auto i = 0; i < rowVector->childrenSize(); ++i) {
        auto childVector = rowVector->childAt(i);

        Value fieldValue;
        if (childVector->isNullAt(rowIdx)) {
          fieldValue.isNull() = true;
        } else {
          convertValue(childVector, fieldValue, rowIdx, evalCtx);
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
void convertValue(
    VectorPtr vector,
    Value& value,
    vector_size_t rowIdx,
    exec::EvalCtx evalCtx) {
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
        value.scalarValue_ref() = getScalarValue(vector, rowIdx);
        break;
      case TypeKind::ARRAY:
      case TypeKind::MAP:
      case TypeKind::ROW:
        value.complexValue_ref() = getComplexValue(vector, rowIdx, evalCtx);
        break;
      default:
        VELOX_FAIL(fmt::format("Unsupported type: {}", vector->type()));
    }
  }
}

std::vector<Value>
convertVector(VectorPtr vector, vector_size_t size, exec::EvalCtx evalCtx) {
  std::vector<Value> rows;
  for (vector_size_t rowIdx = 0; rowIdx < size; ++rowIdx) {
    Value value;
    convertValue(vector, value, rowIdx, evalCtx);
    rows.push_back(value);
  }
  return rows;
}

// Convert Velox RowVectors to Thrift Batches
std::vector<Batch> convertToBatches(
    const std::vector<RowVectorPtr>& rowVectors,
    exec::EvalCtx evalCtx) {
  std::vector<Batch> results;

  if (rowVectors.empty()) {
    return results;
  }

  for (const auto& rowVector : rowVectors) {
    Batch result;
    const auto& rowType = rowVector->type()->asRow();

    for (auto i = 0; i < rowType.size(); ++i) {
      result.columnNames()->push_back(rowType.nameOf(i));
      result.columnTypes()->push_back(rowType.childAt(i)->toString());
    }

    result.numRows() = rowVector->size();

    const auto numColumns = rowVector->childrenSize();
    result.columns()->resize(numColumns);

    for (auto colIdx = 0; colIdx < numColumns; ++colIdx) {
      (*result.columns())[colIdx].rows() =
          convertVector(rowVector->childAt(colIdx), rowVector->size(), evalCtx);
    }

    results.push_back(std::move(result));
  }

  return results;
}

} // namespace facebook::velox::runner

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

    auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(1);
    auto context = core::QueryCtx::create(executor.get());

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
