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

#include <iostream>
#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/vector/ConstantVector.h"

namespace facebook::velox::functions {
namespace {

template <typename T>
int64_t toInt64(T value);

template <>
int64_t toInt64(int64_t value) {
  return value;
}

template <>
int64_t toInt64(Date value) {
  return value.days();
}

template <typename T>
T add(T value, int64_t steps);

template <>
int64_t add(int64_t value, int64_t steps) {
  return value + steps;
}

template <>
Date add(Date value, int64_t steps) {
  return Date(value.days() + steps);
}

// See documentation at https://prestodb.io/docs/current/functions/array.html
template <typename T>
class SequenceFunction : public exec::VectorFunction {
 public:
  static constexpr int32_t kMaxResultEntries = 10'000;

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto startVector = decodedArgs.at(0);
    auto stopVector = decodedArgs.at(1);
    DecodedVector* stepVector = nullptr;
    if (args.size() == 3) {
      stepVector = decodedArgs.at(2);
    }

    const auto numRows = rows.end();
    auto pool = context.pool();
    vector_size_t numElements = 0;

    BufferPtr sizes = allocateSizes(numRows, pool);
    BufferPtr offsets = allocateOffsets(numRows, pool);
    auto rawSizes = sizes->asMutable<vector_size_t>();
    auto rawOffsets = offsets->asMutable<vector_size_t>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      auto start = toInt64(startVector->valueAt<T>(row));
      auto stop = toInt64(stopVector->valueAt<T>(row));
      const int64_t step = (stepVector == nullptr)
          ? (stop >= start ? 1 : -1)
          : stepVector->valueAt<int64_t>(row);
      rawSizes[row] = checkArguments(start, stop, step);
      numElements += rawSizes[row];
    });

    VectorPtr elements =
        BaseVector::create(outputType->childAt(0), numElements, pool);
    auto rawElements = elements->asFlatVector<T>()->mutableRawValues();

    vector_size_t elementsOffset = 0;
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto sequenceCount = rawSizes[row];
      if (sequenceCount) {
        rawOffsets[row] = elementsOffset;
        writeToElements(
            rawElements + elementsOffset,
            sequenceCount,
            startVector,
            stopVector,
            stepVector,
            row);
        elementsOffset += rawSizes[row];
      }
    });
    context.moveOrCopyResult(
        std::make_shared<ArrayVector>(
            pool, outputType, nullptr, numRows, offsets, sizes, elements),
        rows,
        result);
  }

 private:
  static vector_size_t
  checkArguments(int64_t start, int64_t stop, int64_t step) {
    VELOX_USER_CHECK_NE(step, 0, "step must not be zero");
    VELOX_USER_CHECK(
        step > 0 ? stop >= start : stop <= start,
        "sequence stop value should be greater than or equal to start value if "
        "step is greater than zero otherwise stop should be less than or equal to start")
    auto sequenceCount = (stop - start) / step + 1;
    VELOX_USER_CHECK_LE(
        sequenceCount,
        kMaxResultEntries,
        "result of sequence function must not have more than 10000 entries");
    return sequenceCount;
  }

  static void writeToElements(
      T* elements,
      vector_size_t sequenceCount,
      DecodedVector* startVector,
      DecodedVector* stopVector,
      DecodedVector* stepVector,
      vector_size_t row) {
    auto start = startVector->valueAt<T>(row);
    auto stop = stopVector->valueAt<T>(row);
    const int64_t step = (stepVector == nullptr)
        ? (toInt64(stop) >= toInt64(start) ? 1 : -1)
        : toInt64(stepVector->valueAt<T>(row));
    for (auto i = 0; i < sequenceCount; ++i) {
      elements[i] = add(start, step * i);
    }
  }
};

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  signatures = {
      exec::FunctionSignatureBuilder()
          .returnType("array(bigint)")
          .argumentType("bigint")
          .argumentType("bigint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(bigint)")
          .argumentType("bigint")
          .argumentType("bigint")
          .argumentType("bigint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(date)")
          .argumentType("date")
          .argumentType("date")
          .build()};
  return signatures;
}

std::shared_ptr<exec::VectorFunction> create(
    const std::string& /* name */,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  switch (inputArgs[0].type->kind()) {
    case TypeKind::BIGINT:
      return std::make_shared<SequenceFunction<int64_t>>();
    case TypeKind::DATE:
      return std::make_shared<SequenceFunction<Date>>();
    default:
      VELOX_UNREACHABLE();
  }
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(udf_sequence, signatures(), create);
} // namespace facebook::velox::functions
