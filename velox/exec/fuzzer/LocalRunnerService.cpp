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

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <folly/json.h>
#include <folly/portability/GFlags.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrift/lib/cpp2/server/ThriftServer.h>

#include "velox/core/QueryCtx.h"
#include "velox/exec/fuzzer/LocalRunnerService.h"
#include "velox/exec/fuzzer/if/gen-cpp2/LocalRunnerService.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/expression/EvalCtx.h"

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

    std::shared_ptr<exec::Task> task;
    auto results = queryBuilder.copyResults(pool.get(), task);

    return {results, stdoutCapture.str()};
  } catch (const std::exception& e) {
    throw std::runtime_error(
        fmt::format("Error executing query: {}", e.what()));
  }
}

} // namespace

ScalarValue getScalarValue(const VectorPtr& vector, vector_size_t rowIdx) {
  ScalarValue scalar;

  switch (vector->typeKind()) {
    case TypeKind::BOOLEAN:
      scalar.boolValue() = vector->as<SimpleVector<bool>>()->valueAt(rowIdx);
      break;
    case TypeKind::TINYINT:
      scalar.tinyintValue() =
          vector->as<SimpleVector<int8_t>>()->valueAt(rowIdx);
      break;
    case TypeKind::SMALLINT:
      scalar.smallintValue() =
          vector->as<SimpleVector<int16_t>>()->valueAt(rowIdx);
      break;
    case TypeKind::INTEGER:
      scalar.integerValue() =
          vector->as<SimpleVector<int32_t>>()->valueAt(rowIdx);
      break;
    case TypeKind::BIGINT:
      scalar.bigintValue() =
          vector->as<SimpleVector<int64_t>>()->valueAt(rowIdx);
      break;
    case TypeKind::REAL:
      scalar.realValue() = vector->as<SimpleVector<float>>()->valueAt(rowIdx);
      break;
    case TypeKind::DOUBLE:
      scalar.doubleValue() =
          vector->as<SimpleVector<double>>()->valueAt(rowIdx);
      break;
    case TypeKind::VARCHAR:
      scalar.varcharValue() =
          vector->as<SimpleVector<StringView>>()->valueAt(rowIdx).str();
      break;
    case TypeKind::VARBINARY: {
      const auto& binValue =
          vector->as<SimpleVector<StringView>>()->valueAt(rowIdx);
      scalar.varbinaryValue() = std::string(binValue.data(), binValue.size());
      break;
    }
    case TypeKind::TIMESTAMP: {
      const auto& ts =
          vector->as<SimpleVector<facebook::velox::Timestamp>>()->valueAt(
              rowIdx);
      facebook::velox::runner::Timestamp timestampValue;
      timestampValue.seconds() = ts.getSeconds();
      timestampValue.nanos() = ts.getNanos();
      scalar.timestampValue() = std::move(timestampValue);
      break;
    }
    case TypeKind::HUGEINT: {
      const auto& hugeint =
          vector->as<SimpleVector<int128_t>>()->valueAt(rowIdx);
      facebook::velox::runner::i128 hugeintValue;
      hugeintValue.msb() = static_cast<int64_t>(hugeint >> 64);
      hugeintValue.lsb() =
          static_cast<int64_t>(hugeint & 0xFFFFFFFFFFFFFFFFULL);
      scalar.hugeintValue() = std::move(hugeintValue);
      break;
    }
    default:
      VELOX_FAIL(fmt::format("Unsupported scalar type: {}", vector->type()));
  }

  return scalar;
}

ComplexValue getComplexValue(
    const VectorPtr& vector,
    vector_size_t rowIdx,
    const exec::EvalCtx& evalCtx) {
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
          elementValue = convertValue(elements, elementIdx, evalCtx);
        }
        arrayValue.values()->push_back(std::move(elementValue));
      }

      complex.arrayValue() = std::move(arrayValue);
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
        keyValue = convertValue(keys, offset + i, evalCtx);
        if (values->isNullAt(offset + i)) {
          valueValue.isNull() = true;
        } else {
          valueValue = convertValue(values, offset + i, evalCtx);
        }
        (*mapValue.values())[std::move(keyValue)] = std::move(valueValue);
      }

      complex.mapValue() = std::move(mapValue);
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
          fieldValue = convertValue(childVector, rowIdx, evalCtx);
        }
        rowValue.fieldValues()->push_back(std::move(fieldValue));
      }

      complex.rowValue() = std::move(rowValue);
      break;
    }
    default:
      VELOX_FAIL(fmt::format("Unsupported complex type: {}", vector->type()));
  }

  return complex;
}

Value convertValue(
    const VectorPtr& vector,
    vector_size_t rowIdx,
    const exec::EvalCtx& evalCtx) {
  Value value;
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
        value.scalarValue() = getScalarValue(vector, rowIdx);
        break;
      case TypeKind::ARRAY:
      case TypeKind::MAP:
      case TypeKind::ROW:
        value.complexValue() = getComplexValue(vector, rowIdx, evalCtx);
        break;
      default:
        VELOX_FAIL(fmt::format("Unsupported type: {}", vector->type()));
    }
  }
  return value;
}

std::vector<Value> convertVector(
    const VectorPtr& vector,
    vector_size_t size,
    const exec::EvalCtx& evalCtx) {
  std::vector<Value> rows;
  for (vector_size_t rowIdx = 0; rowIdx < size; ++rowIdx) {
    Value value = convertValue(vector, rowIdx, evalCtx);
    rows.push_back(value);
  }
  return rows;
}

std::vector<Batch> convertToBatches(
    const std::vector<RowVectorPtr>& rowVectors,
    const exec::EvalCtx& evalCtx) {
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

void LocalRunnerServiceHandler::execute(
    ExecutePlanResponse& response,
    std::unique_ptr<ExecutePlanRequest> request) {
  VLOG(1) << "Received executePlan request";

  std::shared_ptr<memory::MemoryPool> pool =
      memory::memoryManager()->addLeafPool();

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

  auto queryCtx = core::QueryCtx::create();
  core::ExecCtx execCtx(pool.get(), queryCtx.get());
  exec::EvalCtx evalCtx(&execCtx);

  VLOG(1) << "Converting results to Thrift response";
  auto resultBatches = convertToBatches({results}, evalCtx);
  response.results() = std::move(resultBatches);
  response.output() = output;
  response.success() = true;
  VLOG(1) << "Response sent";
}

} // namespace facebook::velox::runner
