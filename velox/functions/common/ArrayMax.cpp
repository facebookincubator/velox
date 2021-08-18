/*
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
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions {
namespace {

template <TypeKind kind>
VectorPtr applyTyped(
    const SelectivityVector& rows,
    DecodedVector& arrayDecoded,
    DecodedVector& elementsDecoded,
    exec::EvalCtx* context) {
  auto pool = context->pool();
  using T = typename TypeTraits<kind>::NativeType;

  auto baseArray = arrayDecoded.base()->as<ArrayVector>();
  auto inIndices = arrayDecoded.indices();
  auto rawSizes = baseArray->rawSizes();
  auto rawOffsets = baseArray->rawOffsets();

  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(rows.size(), pool);
  auto rawIndices = indices->asMutable<vector_size_t>();

  // Create nulls for lazy initialization.
  BufferPtr nulls(nullptr);
  uint64_t* rawNulls = nullptr;

  auto processNull = [&](vector_size_t row) {
    if (nulls == nullptr) {
      nulls = AlignedBuffer::allocate<bool>(rows.size(), pool, bits::kNotNull);
      rawNulls = nulls->asMutable<uint64_t>();
    }
    bits::setNull(rawNulls, row, true);
  };

  if (elementsDecoded.isIdentityMapping() && !elementsDecoded.mayHaveNulls()) {
    auto rawElements = elementsDecoded.values<T>();

    rows.applyToSelected([&](auto row) {
      auto size = rawSizes[inIndices[row]];
      auto offset = rawOffsets[inIndices[row]];
      if (size == 0) {
        processNull(row);
      } else {
        auto maxElementIndex = offset;
        for (auto i = 1; i < size; i++) {
          if (rawElements[offset + i] > rawElements[maxElementIndex]) {
            maxElementIndex = offset + i;
          }
        }
        rawIndices[row] = maxElementIndex;
      }
    });
  } else {
    rows.applyToSelected([&](auto row) {
      auto size = rawSizes[inIndices[row]];
      if (size == 0) {
        processNull(row);
      } else {
        auto offset = rawOffsets[inIndices[row]];
        auto maxElementIndex = offset;
        for (auto i = 0; i < size; i++) {
          if (elementsDecoded.isNullAt(offset + i)) {
            // If a NULL value is encountered, max is always NULL
            processNull(row);
            break;
          } else if (
              elementsDecoded.valueAt<T>(offset + i) >
              elementsDecoded.valueAt<T>(maxElementIndex)) {
            maxElementIndex = offset + i;
          }
        }
        rawIndices[row] = maxElementIndex;
      }
    });
  }

  return BaseVector::wrapInDictionary(
      nulls, indices, rows.size(), baseArray->elements());
}

template <>
VectorPtr applyTyped<TypeKind::BOOLEAN>(
    const SelectivityVector& rows,
    DecodedVector& arrayDecoded,
    DecodedVector& elementsDecoded,
    exec::EvalCtx* context) {
  auto pool = context->pool();
  using T = typename TypeTraits<TypeKind::BOOLEAN>::NativeType;

  auto baseArray = arrayDecoded.base()->as<ArrayVector>();
  auto inIndices = arrayDecoded.indices();
  auto rawSizes = baseArray->rawSizes();
  auto rawOffsets = baseArray->rawOffsets();

  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(rows.size(), pool);
  auto rawIndices = indices->asMutable<vector_size_t>();

  // Create nulls for lazy initialization.
  BufferPtr nulls(nullptr);
  uint64_t* rawNulls = nullptr;

  auto processNull = [&](vector_size_t row) {
    if (nulls == nullptr) {
      nulls = AlignedBuffer::allocate<bool>(rows.size(), pool, bits::kNotNull);
      rawNulls = nulls->asMutable<uint64_t>();
    }
    bits::setNull(rawNulls, row, true);
  };

  rows.applyToSelected([&](auto row) {
    auto size = rawSizes[inIndices[row]];
    if (size == 0) {
      processNull(row);
    } else {
      auto offset = rawOffsets[inIndices[row]];
      bool mayHaveNulls = elementsDecoded.mayHaveNulls();
      bool foundTrue = false;
      auto maxElementIndex = offset;
      for (auto i = 0; i < size; i++) {
        if (elementsDecoded.isNullAt(offset + i)) {
          // If a NULL value is encountered, max is always NULL
          processNull(row);
          break;
        } else if (
            !foundTrue && elementsDecoded.valueAt<T>(offset + i) == true) {
          // check for true only if we did not find it yet.
          maxElementIndex = offset + i;
          foundTrue = true;
          // if there are no Nulls, break
          if (!mayHaveNulls) {
            break;
          }
        }
      }
      rawIndices[row] = maxElementIndex;
    }
  });

  return BaseVector::wrapInDictionary(
      nulls, indices, rows.size(), baseArray->elements());
}

class ArrayMaxFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      exec::Expr* /*caller*/,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    VELOX_CHECK_EQ(args.size(), 1);
    const auto& arrayVector = args[0];
    VELOX_CHECK(arrayVector->type()->isArray());

    exec::LocalDecodedVector arrayHolder(context, *arrayVector, rows);
    auto decodedArray = arrayHolder.get();
    auto baseArray = decodedArray->base()->as<ArrayVector>();
    auto elementsVector = baseArray->elements();
    auto elementsRows = toElementRows(elementsVector->size(), rows, baseArray);
    exec::LocalDecodedVector elementsHolder(
        context, *elementsVector, elementsRows);
    auto decodedElements = elementsHolder.get();

    auto localResult = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        applyTyped,
        elementsVector->typeKind(),
        rows,
        *decodedArray,
        *decodedElements,
        context);

    context->moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // array(T) -> T
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("T")
                .argumentType("array(T)")
                .build()};
  }
};

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_array_max,
    ArrayMaxFunction::signatures(),
    std::make_unique<ArrayMaxFunction>());

} // namespace facebook::velox::functions
