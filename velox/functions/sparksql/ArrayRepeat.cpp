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
#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions {
namespace {

class RepeatFunction : public exec::VectorFunction {
 public:
  bool isDefaultNullBehavior() const override {
    // repeat(null, n) returns an array of n nulls.
    return false;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VectorPtr localResult;
    if (args[1]->isConstantEncoding()) {
      localResult = applyConstant(rows, args, outputType, context);
      // Invalid constant count.
      if (localResult == nullptr) {
        return;
      }
    } else {
      localResult = applyNonConstant(rows, args, outputType, context);
    }
    context.moveOrCopyResult(localResult, rows, result);
  }

 private:
  VectorPtr applyConstant(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context) const {
    const auto numRows = rows.end();
    auto pool = context.pool();

    auto* constantCount = args[1]->as<ConstantVector<int32_t>>();
    if (constantCount->isNullAt(0)) {
      // If count is a null constant, the result should be all nulls.
      return BaseVector::createNullConstant(outputType, numRows, pool);
    }

    auto count = constantCount->valueAt(0);
    if (count < 0) {
      // Negative count is like zero count, and empty array will be returned.
      count = 0;
    } else if (count > kMaxCount) {
      // The max limit is aligned with Spark.
      auto pex = std::make_exception_ptr(std::invalid_argument(fmt::format(
          "({} vs. {}) Count argument of array_repeat function must be less than or equal to (MAX_INT32 - 15)",
          count,
          kMaxCount)));
      context.setErrors(rows, pex);
      return nullptr;
    }

    const auto totalCount = count * numRows;

    // Allocate new vectors for indices, lengths and offsets.
    BufferPtr indices = allocateIndices(totalCount, pool);
    BufferPtr sizes = allocateSizes(numRows, pool);
    BufferPtr offsets = allocateOffsets(numRows, pool);
    auto rawIndices = indices->asMutable<vector_size_t>();
    auto rawSizes = sizes->asMutable<vector_size_t>();
    auto rawOffsets = offsets->asMutable<vector_size_t>();

    vector_size_t offset = 0;
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      rawSizes[row] = count;
      rawOffsets[row] = offset;
      std::fill(rawIndices + offset, rawIndices + offset + count, row);
      offset += count;
    });

    return std::make_shared<ArrayVector>(
        pool,
        outputType,
        nullptr,
        numRows,
        offsets,
        sizes,
        BaseVector::wrapInDictionary(nullptr, indices, totalCount, args[0]));
  }

  VectorPtr applyNonConstant(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context) const {
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto countDecoded = decodedArgs.at(1);
    int32_t totalCount = 0;
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      if (countDecoded->isNullAt(row)) {
        return;
      }
      auto count = countDecoded->valueAt<int32_t>(row);
      if (count < 0) {
        // Negative count is like zero count, and empty array will be returned.
        count = 0;
      } else {
        // The max limit is aligned with Spark.
        VELOX_USER_CHECK_LE(
            count,
            kMaxCount,
            "Count argument of array_repeat function must be less than or equal to (MAX_INT32 - 15)");
      }
      totalCount += count;
    });

    const auto numRows = rows.end();
    auto pool = context.pool();

    // Allocate new vector for nulls if necessary.
    BufferPtr nulls;
    uint64_t* rawNulls = nullptr;
    if (countDecoded->mayHaveNulls()) {
      nulls = allocateNulls(numRows, pool);
      rawNulls = nulls->asMutable<uint64_t>();
    }

    // Allocate new vectors for indices, lengths and offsets.
    BufferPtr indices = allocateIndices(totalCount, pool);
    BufferPtr sizes = allocateSizes(numRows, pool);
    BufferPtr offsets = allocateOffsets(numRows, pool);
    auto rawIndices = indices->asMutable<vector_size_t>();
    auto rawSizes = sizes->asMutable<vector_size_t>();
    auto rawOffsets = offsets->asMutable<vector_size_t>();

    exec::LocalSelectivityVector remainingRows(context, rows);
    context.deselectErrors(*remainingRows);
    vector_size_t offset = 0;
    context.applyToSelectedNoThrow(*remainingRows, [&](auto row) {
      if (rawNulls && countDecoded->isNullAt(row)) {
        // Returns null if count argument is null.
        // e.g. repeat('a', null) -> null
        bits::setNull(rawNulls, row);
        return;
      }
      auto count = countDecoded->valueAt<int32_t>(row);
      count = count < 0 ? 0 : count;
      rawSizes[row] = count;
      rawOffsets[row] = offset;
      std::fill(rawIndices + offset, rawIndices + offset + count, row);
      offset += count;
    });

    return std::make_shared<ArrayVector>(
        pool,
        outputType,
        nulls,
        numRows,
        offsets,
        sizes,
        BaseVector::wrapInDictionary(nullptr, indices, totalCount, args[0]));
  }

  static constexpr int32_t kMaxCount = std::numeric_limits<int32_t>::max() - 15;
};

static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  // T, integer -> array(T)
  return {exec::FunctionSignatureBuilder()
              .typeVariable("T")
              .returnType("array(T)")
              .argumentType("T")
              .argumentType("integer")
              .build()};
}
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_repeat,
    signatures(),
    std::make_unique<RepeatFunction>());

} // namespace facebook::velox::functions
