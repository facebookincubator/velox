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

template <typename T>
class Min {
 public:
  bool operator()(const T& arg1, const T& arg2) const {
    return arg1 < arg2;
  }
  bool operator()() const {
    return false;
  }
};

template <typename T>
class Max {
 public:
  bool operator()(const T& arg1, const T& arg2) const {
    return arg1 > arg2;
  }
  bool operator()() const {
    return true;
  }
};

template <TypeKind kind, template <typename> class F>
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
        auto minElementIndex = offset;
        for (auto i = 1; i < size; i++) {
          if (F<T>()(rawElements[offset + i], rawElements[minElementIndex])) {
            minElementIndex = offset + i;
          }
        }
        rawIndices[row] = minElementIndex;
      }
    });
  } else {
    rows.applyToSelected([&](auto row) {
      auto size = rawSizes[inIndices[row]];
      if (size == 0) {
        processNull(row);
      } else {
        auto offset = rawOffsets[inIndices[row]];
        auto minElementIndex = offset;
        for (auto i = 0; i < size; i++) {
          if (elementsDecoded.isNullAt(offset + i)) {
            // If a NULL value is encountered, min is always NULL
            processNull(row);
            break;
          } else if (F<T>()(
                         elementsDecoded.valueAt<T>(offset + i),
                         elementsDecoded.valueAt<T>(minElementIndex))) {
            minElementIndex = offset + i;
          }
        }
        rawIndices[row] = minElementIndex;
      }
    });
  }

  return BaseVector::wrapInDictionary(
      nulls, indices, rows.size(), baseArray->elements());
}

template <template <typename> class F>
VectorPtr applyBoolean(
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
  bool mayHaveNulls = elementsDecoded.mayHaveNulls();
  rows.applyToSelected([&](auto row) {
    auto size = rawSizes[inIndices[row]];
    if (size == 0) {
      processNull(row);
    } else {
      auto offset = rawOffsets[inIndices[row]];
      bool foundMinMax = false;
      auto minElementIndex = offset;
      for (auto i = 0; i < size; i++) {
        if (elementsDecoded.isNullAt(offset + i)) {
          // If a NULL value is encountered, min is always NULL
          processNull(row);
          break;
        } else if (
            !foundMinMax &&
            elementsDecoded.valueAt<T>(offset + i) == F<T>()()) {
          // check for Min or Max only if we did not find it yet.
          minElementIndex = offset + i;
          foundMinMax = true;
          // if there are no Nulls, break
          if (!mayHaveNulls) {
            break;
          }
        }
      }
      rawIndices[row] = minElementIndex;
    }
  });

  return BaseVector::wrapInDictionary(
      nulls, indices, rows.size(), baseArray->elements());
}

template <template <typename> class F>
class ArrayMinMaxFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      exec::Expr* caller,
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
    VectorPtr localResult;
    auto typeKind = elementsVector->typeKind();
    switch (typeKind) {
      case TypeKind::BOOLEAN: {
        localResult =
            applyBoolean<F>(rows, *decodedArray, *decodedElements, context);
        break;
      }
      case ::facebook::velox::TypeKind::INTEGER: {
        localResult = applyTyped<TypeKind::INTEGER, F>(
            rows, *decodedArray, *decodedElements, context);
        break;
      }
      case ::facebook::velox::TypeKind::TINYINT: {
        localResult = applyTyped<TypeKind::TINYINT, F>(
            rows, *decodedArray, *decodedElements, context);
        break;
      }
      case ::facebook::velox::TypeKind::SMALLINT: {
        localResult = applyTyped<TypeKind::SMALLINT, F>(
            rows, *decodedArray, *decodedElements, context);
        break;
      }
      case ::facebook::velox::TypeKind::BIGINT: {
        localResult = applyTyped<TypeKind::BIGINT, F>(
            rows, *decodedArray, *decodedElements, context);
        break;
      }
      case ::facebook::velox::TypeKind::REAL: {
        localResult = applyTyped<TypeKind::REAL, F>(
            rows, *decodedArray, *decodedElements, context);
        break;
      }
      case ::facebook::velox::TypeKind::DOUBLE: {
        localResult = applyTyped<TypeKind::DOUBLE, F>(
            rows, *decodedArray, *decodedElements, context);
        break;
      }
      case ::facebook::velox::TypeKind::VARCHAR: {
        localResult = applyTyped<TypeKind::VARCHAR, F>(
            rows, *decodedArray, *decodedElements, context);
        break;
      }
      case ::facebook::velox::TypeKind::VARBINARY: {
        localResult = applyTyped<TypeKind::VARBINARY, F>(
            rows, *decodedArray, *decodedElements, context);
        break;
      }
      case ::facebook::velox::TypeKind::TIMESTAMP: {
        localResult = applyTyped<TypeKind::TIMESTAMP, F>(
            rows, *decodedArray, *decodedElements, context);
        break;
      }
      default:
        VELOX_FAIL("not a scalar type! kind: {}", mapTypeKindToName(typeKind));
    }

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
    udf_array_min,
    ArrayMinMaxFunction<Min>::signatures(),
    std::make_unique<ArrayMinMaxFunction<Min>>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_array_max,
    ArrayMinMaxFunction<Max>::signatures(),
    std::make_unique<ArrayMinMaxFunction<Max>>());

} // namespace facebook::velox::functions
