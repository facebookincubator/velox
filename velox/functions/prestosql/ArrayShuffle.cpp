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
#include <numeric>
#include <random>
#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions {
namespace {
// See documentation at
// https://prestodb.io/docs/current/functions/array.html#shuffle
//
// This function will shuffle identical arrays independently, i.e. even when
// the input has duplicate rows represented using constant and dictionary
// encoding, the output is flat and likely yields different values.
//
// E.g.1: constant encoding
// Input: ConstantVector(base=ArrayVector[{1,2,3}], length=3, index=0)
// Possible Output: ArrayVector[{1,3,2},{2,3,1},{3,2,1}]
//
// E.g.2: dict encoding
// Input: DictionaryVector(
//   dictionaryValues=ArrayVector[{1,2,3},{4,5},{1,2,3}],
//   dictionaryIndices=[1,2,0])
// Possible Output: ArrayVector[{5,4},{2,1,3},{1,3,2}]
//
class ArrayShuffleFunction : public exec::VectorFunction {
 public:
  bool isDeterministic() const override {
    return false;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);

    // This is a non-deterministic function, which violates the guarantee on a
    // deterministic single-arg function that the expression evaluation will
    // peel off encodings, and we will only see flat or constant inputs. Hence,
    // we need to use DecodedVector to handle ALL encodings.
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedArg = decodedArgs.at(0);
    auto arrayVector = decodedArg->base()->as<ArrayVector>();
    auto elementsVector = arrayVector->elements();

    vector_size_t numElements = 0;
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto size = arrayVector->sizeAt(decodedArg->index(row));
      numElements += size;
    });

    // Allocate new buffer to hold shuffled indices.
    BufferPtr shuffledIndices = allocateIndices(numElements, context.pool());
    BufferPtr offsets = allocateOffsets(rows.size(), context.pool());
    BufferPtr sizes = allocateSizes(rows.size(), context.pool());

    vector_size_t* rawIndices = shuffledIndices->asMutable<vector_size_t>();
    vector_size_t* rawOffsets = offsets->asMutable<vector_size_t>();
    vector_size_t* rawSizes = sizes->asMutable<vector_size_t>();

    vector_size_t newOffset = 0;
    std::mt19937 randGen(std::random_device{}());
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      vector_size_t arrayRow = decodedArg->index(row);
      vector_size_t size = arrayVector->sizeAt(arrayRow);
      vector_size_t offset = arrayVector->offsetAt(arrayRow);

      std::iota(rawIndices + newOffset, rawIndices + newOffset + size, offset);
      std::shuffle(
          rawIndices + newOffset, rawIndices + newOffset + size, randGen);

      rawSizes[row] = size;
      rawOffsets[row] = newOffset;
      newOffset += size;
    });

    auto resultElements = BaseVector::wrapInDictionary(
        nullptr, shuffledIndices, numElements, elementsVector);
    auto localResult = std::make_shared<ArrayVector>(
        context.pool(),
        arrayVector->type(),
        nullptr,
        rows.size(),
        std::move(offsets),
        std::move(sizes),
        std::move(resultElements));

    context.moveOrCopyResult(localResult, rows, result);
  }
};

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  return {// array(T) -> array(T)
          exec::FunctionSignatureBuilder()
              .typeVariable("T")
              .returnType("array(T)")
              .argumentType("array(T)")
              .build()};
}

} // namespace

// Register function.
VELOX_DECLARE_VECTOR_FUNCTION(
    udf_array_shuffle,
    signatures(),
    std::make_unique<ArrayShuffleFunction>());
} // namespace facebook::velox::functions
