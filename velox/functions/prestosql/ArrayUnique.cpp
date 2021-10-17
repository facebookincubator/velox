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

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"

#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"

namespace facebook::velox::functions {
namespace {

// See documentation at https://prestodb.io/docs/current/functions/array.html
class ArrayUniqueFunction : public exec::VectorFunction {
 public:
  explicit ArrayUniqueFunction(const std::string& name) : name_(name) {}

  // Validate number of parameters and types.
  void validateType(const std::vector<exec::VectorFunctionArg>& inputArgs) {
    VELOX_USER_CHECK_EQ(
        inputArgs.size(),
        1,
        "{} only takes argument of type array(varchar | bigint)",
        name_);

    auto arrayType = inputArgs.front().type;
    VELOX_USER_CHECK_EQ(
        arrayType->kind(),
        TypeKind::ARRAY,
        "{} requires arguments of type ARRAY",
        name_);
  }

 protected:
  // Validate element type inside the vector
  virtual void validateElementType(const TypeKind& kind) const {}

  const std::string& name_;
};

/// This class implements the array_distinct query function.
///
/// Along with the set, we maintain a `hasNull` flag that indicates whether
/// null is present in the array.
///
/// Zero element copy:
///
/// In order to prevent copies of array elements, the function reuses the
/// internal elements() vector from the original ArrayVector.
///
/// First a new vector is created containing the indices of the elements
/// which will be present in the output, and wrapped into a DictionaryVector.
/// Next the `lengths` and `offsets` vectors that control where output arrays
/// start and end are wrapped into the output ArrayVector.
template <typename T>
class ArrayDistinctFunction : public ArrayUniqueFunction {
 public:
  ArrayDistinctFunction() : ArrayUniqueFunction("array_distinct") {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      exec::Expr* caller,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    // Acquire the array elements vector.
    auto arrayVector = args.front()->as<ArrayVector>();
    auto elementsVector = arrayVector->elements();
    auto elementsRows =
        toElementRows(elementsVector->size(), rows, arrayVector);
    exec::LocalDecodedVector elements(context, *elementsVector, elementsRows);

    vector_size_t elementsCount = elementsRows.size();
    vector_size_t rowCount = arrayVector->size();

    // Allocate new vectors for indices, length and offsets.
    memory::MemoryPool* pool = context->pool();
    BufferPtr newIndices =
        AlignedBuffer::allocate<vector_size_t>(elementsCount, pool);
    BufferPtr newLengths =
        AlignedBuffer::allocate<vector_size_t>(rowCount, pool);
    BufferPtr newOffsets =
        AlignedBuffer::allocate<vector_size_t>(rowCount, pool);

    // Pointers and cursors to the raw data.
    auto* rawNewIndices = newIndices->asMutable<vector_size_t>();
    auto* rawSizes = newLengths->asMutable<vector_size_t>();
    auto* rawOffsets = newOffsets->asMutable<vector_size_t>();
    vector_size_t indicesCursor = 0;

    // Process the rows: store unique values in the hash table.
    folly::F14FastSet<T> uniqueSet;

    rows.applyToSelected([&](vector_size_t row) {
      auto size = arrayVector->sizeAt(row);
      auto offset = arrayVector->offsetAt(row);

      *rawOffsets = indicesCursor;
      bool hasNulls = false;
      for (vector_size_t i = offset; i < offset + size; ++i) {
        if (elements->isNullAt(i)) {
          if (!hasNulls) {
            hasNulls = true;
            rawNewIndices[indicesCursor++] = i;
          }
        } else {
          auto value = elements->valueAt<T>(i);

          if (uniqueSet.insert(value).second) {
            rawNewIndices[indicesCursor++] = i;
          }
        }
      }

      uniqueSet.clear();
      *rawSizes = indicesCursor - *rawOffsets;
      ++rawSizes;
      ++rawOffsets;
    });

    // Prepare and return result set.
    auto newElements =
        BaseVector::transpose(newIndices, std::move(elementsVector));
    auto resultArray = std::make_shared<ArrayVector>(
        pool,
        caller->type(),
        nullptr,
        rowCount,
        std::move(newOffsets),
        std::move(newLengths),
        std::move(newElements),
        0);
    context->moveOrCopyResult(resultArray, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // array(T) -> array(T)
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("array(T)")
                .argumentType("array(T)")
                .build()};
  }
};

/// This class implements the array_dupes query function.
///
/// Along with the hash map, we maintain a `hasNull` flag that indicates
/// whether null is present in the array.
template <typename T>
class ArrayDupesFunction : public ArrayUniqueFunction {
 public:
  ArrayDupesFunction() : ArrayUniqueFunction("array_dupes") {}

  // Execute function.
  // TODO: in java version, it will return an output array in a SORTED order
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      exec::Expr* caller,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    // Acquire the array elements vector.
    auto arrayVector = args.front()->as<ArrayVector>();
    auto elementsVector = arrayVector->elements();
    validateElementType(elementsVector->typeKind());

    auto elementsRows =
        toElementRows(elementsVector->size(), rows, arrayVector);
    exec::LocalDecodedVector elements(context, *elementsVector, elementsRows);

    vector_size_t elementsCount = elementsRows.size();
    vector_size_t rowCount = arrayVector->size();

    // Allocate new vectors for indices, length and offsets.
    memory::MemoryPool* pool = context->pool();
    BufferPtr newIndices =
        AlignedBuffer::allocate<vector_size_t>(elementsCount, pool);
    BufferPtr newLengths =
        AlignedBuffer::allocate<vector_size_t>(rowCount, pool);
    BufferPtr newOffsets =
        AlignedBuffer::allocate<vector_size_t>(rowCount, pool);

    // Pointers and cursors to the raw data.
    vector_size_t indicesCursor = 0;
    auto* rawNewIndices = newIndices->asMutable<vector_size_t>();
    auto* rawSizes = newLengths->asMutable<vector_size_t>();
    auto* rawOffsets = newOffsets->asMutable<vector_size_t>();

    // Process the rows: use a hashmap to store unique values and
    // whether it occurred once or more than once.
    folly::F14FastMap<T, bool> uniqueMap;

    rows.applyToSelected([&](vector_size_t row) {
      auto size = arrayVector->sizeAt(row);
      auto offset = arrayVector->offsetAt(row);

      *rawOffsets = indicesCursor;
      vector_size_t numOfNulls = 0;
      for (vector_size_t i = offset; i < offset + size; ++i) {
        if (elements->isNullAt(i)) {
          numOfNulls++;
          if (numOfNulls == 2) {
            rawNewIndices[indicesCursor++] = i;
          }
        } else {
          auto value = elements->valueAt<T>(i);
          if (uniqueMap.contains(value)) {
            if (uniqueMap[value]) {
              rawNewIndices[indicesCursor++] = i;
              uniqueMap[value] = false;
            }
          } else {
            uniqueMap[value] = true;
          }
        }
      }

      uniqueMap.clear();
      *rawSizes = indicesCursor - *rawOffsets;
      ++rawSizes;
      ++rawOffsets;
    });

    auto newElements =
        BaseVector::transpose(newIndices, std::move(elementsVector));

    // Prepare and return result set.
    auto resultArray = std::make_shared<ArrayVector>(
        pool,
        caller->type(),
        nullptr,
        rowCount,
        std::move(newOffsets),
        std::move(newLengths),
        std::move(newElements),
        0);
    context->moveOrCopyResult(resultArray, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // array(T) -> array(T)
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("array(T)")
                .argumentType("array(T)")
                .build()};
  }

 protected:
  void validateElementType(const TypeKind& kind) const override {
    VELOX_USER_CHECK(
        kind == TypeKind::TINYINT || kind == TypeKind::SMALLINT ||
            kind == TypeKind::INTEGER || kind == TypeKind::BIGINT ||
            kind == TypeKind::VARCHAR,
        "{} only takes argument of type array(varchar) or array(bigint)",
        ArrayUniqueFunction::name_);
  }
};

// Create function template based on type.
template <template <typename> class F, TypeKind kind>
std::shared_ptr<exec::VectorFunction> createTyped(
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_CHECK_EQ(inputArgs.size(), 1);

  auto elementType = inputArgs.front().type->childAt(0);
  using T = typename TypeTraits<kind>::NativeType;

  static_assert(
      std::is_base_of_v<functions::ArrayUniqueFunction, F<T>>,
      "Expect a subclass of ArrayUniqueFunction");

  if constexpr (std::is_same_v<ArrayDistinctFunction<T>, F<T>>) {
    auto functionPtr = std::make_shared<ArrayDistinctFunction<T>>();
    functionPtr->validateType(inputArgs);
    return functionPtr;
  } else if constexpr (std::is_same_v<ArrayDupesFunction<T>, F<T>>) {
    auto functionPtr = std::make_shared<ArrayDupesFunction<T>>();
    functionPtr->validateType(inputArgs);
    return functionPtr;
  }
}

// Create function.
template <template <typename> class F>
std::shared_ptr<exec::VectorFunction> create(
    const std::string& /* name */,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  auto elementType = inputArgs.front().type->childAt(0);

  return VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
      createTyped, F, elementType->kind(), inputArgs);
}
} // namespace

// Register function.
VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_array_distinct,
    ArrayDistinctFunction<void>::signatures(),
    create<ArrayDistinctFunction>);

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_array_dupes,
    ArrayDupesFunction<void>::signatures(),
    create<ArrayDupesFunction>);

} // namespace facebook::velox::functions
