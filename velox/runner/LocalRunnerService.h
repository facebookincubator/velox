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

#pragma once

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
#include "velox/runner/if/gen-cpp2/LocalRunnerService.h"

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
namespace facebook::velox::runner {

// Forward declaration
void convertValue(
    VectorPtr vector,
    Value& value,
    vector_size_t rowIdx,
    exec::EvalCtx evalCtx);

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

ComplexValue handleComplexType(
    VectorPtr vector,
    vector_size_t rowIdx,
    exec::EvalCtx evalCtx) {
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
          convertValue(elements, elementValue, elementIdx, evalCtx);
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
      auto rowVector = vector->as<RowVector>();
      facebook::velox::runner::Row rowValue;

      const auto& rowType = rowVector->type()->asRow();
      for (auto i = 0; i < rowType.size(); ++i) {
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
        value.scalarValue_ref() = handleScalarType(vector, rowIdx);
        break;
      case TypeKind::ARRAY:
      case TypeKind::MAP:
      case TypeKind::ROW:
        value.complexValue_ref() = handleComplexType(vector, rowIdx, evalCtx);
        break;
      default:
        VELOX_FAIL(fmt::format("Unsupported type: {}", vector->type()));
    }
  }
}

std::vector<Value>
serializeVector(VectorPtr vector, vector_size_t size, exec::EvalCtx evalCtx) {
  exec::LocalDecodedVector mapDecoder(
      evalCtx, *vector, SelectivityVector(vector->size()));
  auto& decoded = *mapDecoder.get();
  VectorPtr base;
  if (!decoded.isIdentityMapping()) {
    base = std::shared_ptr<BaseVector>(
        const_cast<BaseVector*>(decoded.base()), [](BaseVector*) {});
  }

  std::vector<Value> rows;
  for (vector_size_t rowIdx = 0; rowIdx < size; ++rowIdx) {
    Value value;
    if (decoded.isIdentityMapping()) {
      convertValue(vector, value, rowIdx, evalCtx);
    } else {
      convertValue(base, value, decoded.index(rowIdx), evalCtx);
    }
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
      (*result.columns())[colIdx].rows() = serializeVector(
          rowVector->childAt(colIdx), rowVector->size(), evalCtx);
    }

    results.push_back(std::move(result));
  }

  return results;
}

} // namespace facebook::velox::runner
