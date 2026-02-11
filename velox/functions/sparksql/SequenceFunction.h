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

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions::sparksql {

/// Spark-compatible sequence function for integer types (tinyint, smallint,
/// integer, bigint). Generates an array of values from start to stop
/// (inclusive) with an optional step. Step defaults to 1 if start <= stop,
/// or -1 if start > stop.
template <typename T>
class SparkSequenceFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto* startVector = decodedArgs.at(0);
    auto* stopVector = decodedArgs.at(1);
    DecodedVector* stepVector = nullptr;
    if (args.size() == 3) {
      stepVector = decodedArgs.at(2);
    }

    const auto numRows = rows.end();
    auto* pool = context.pool();
    size_t numElements = 0;

    BufferPtr sizes = allocateSizes(numRows, pool);
    BufferPtr offsets = allocateOffsets(numRows, pool);
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    auto* rawOffsets = offsets->asMutable<vector_size_t>();

    const auto maxElements = context.execCtx()
                                 ->queryCtx()
                                 ->queryConfig()
                                 .maxElementsSizeInRepeatAndSequence();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      rawSizes[row] =
          checkArguments(startVector, stopVector, stepVector, row, maxElements);
      numElements += rawSizes[row];
    });

    VELOX_USER_CHECK_LE(
        numElements,
        std::numeric_limits<vector_size_t>::max(),
        "SEQUENCE result too large: {} elements exceeds maximum {}",
        numElements,
        std::numeric_limits<vector_size_t>::max());

    VectorPtr elements =
        BaseVector::create(outputType->childAt(0), numElements, pool);
    auto* rawElements = elements->asFlatVector<T>()->mutableRawValues();

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
  static vector_size_t checkArguments(
      DecodedVector* startVector,
      DecodedVector* stopVector,
      DecodedVector* stepVector,
      vector_size_t row,
      int32_t maxElements) {
    auto start = static_cast<int64_t>(startVector->valueAt<T>(row));
    auto stop = static_cast<int64_t>(stopVector->valueAt<T>(row));
    auto step = getStep(start, stop, stepVector, row);
    VELOX_USER_CHECK_NE(step, 0, "step must not be zero");
    VELOX_USER_CHECK(
        step > 0 ? stop >= start : stop <= start,
        "sequence stop value should be greater than or equal to start value if "
        "step is greater than zero otherwise stop should be less than or equal to start");

    auto sequenceCount =
        (static_cast<int128_t>(stop) - static_cast<int128_t>(start)) / step + 1;
    VELOX_USER_CHECK_LE(
        sequenceCount,
        maxElements,
        "result of sequence function must not have more than {} entries",
        maxElements);
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
    auto step = getStep(
        static_cast<int64_t>(start),
        static_cast<int64_t>(stop),
        stepVector,
        row);
    for (auto i = 0; i < sequenceCount; ++i) {
      elements[i] = static_cast<T>(static_cast<int64_t>(start) + step * i);
    }
  }

  static int64_t getStep(
      int64_t start,
      int64_t stop,
      DecodedVector* stepVector,
      vector_size_t row) {
    if (!stepVector) {
      return (stop >= start) ? 1 : -1;
    }
    return static_cast<int64_t>(stepVector->valueAt<T>(row));
  }
};

/// Returns signatures for all integer sequence overloads plus date (2-arg).
std::vector<std::shared_ptr<exec::FunctionSignature>> sparkSequenceSignatures();

/// Factory that creates the correct SparkSequenceFunction<T> based on input
/// types. Falls back to Presto's SequenceFunction for date types.
std::shared_ptr<exec::VectorFunction> makeSparkSequence(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

} // namespace facebook::velox::functions::sparksql
