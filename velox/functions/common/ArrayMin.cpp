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
  auto indices = arrayDecoded.indices();
  auto rawSizes = baseArray->rawSizes();
  auto rawOffsets = baseArray->rawOffsets();

  BufferPtr outIndices =
      AlignedBuffer::allocate<vector_size_t>(rows.size(), pool);
  auto rawOutIndices = outIndices->asMutable<vector_size_t>();

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
  constexpr bool isBoolType = std::is_same_v<bool, T>;

  if (!isBoolType && elementsDecoded.isIdentityMapping() &&
      !elementsDecoded.mayHaveNulls()) {
    auto rawElements = elementsDecoded.values<T>();

    rows.applyToSelected([&](auto row) {
      auto size = rawSizes[indices[row]];
      auto offset = rawOffsets[indices[row]];
      if (size == 0) {
        processNull(row);
      } else {
        auto min_element_index = offset;
        for (auto i = 1; i < size; i++) {
          if (rawElements[offset + i] < rawElements[min_element_index]) {
            min_element_index = offset + i;
          }
        }
        rawOutIndices[row] = min_element_index;
      }
    });
  } else {
    rows.applyToSelected([&](auto row) {
      auto size = rawSizes[indices[row]];
      if (size == 0) {
        processNull(row);
      } else {
        auto offset = rawOffsets[indices[row]];
        auto min_element_index = offset;
        for (auto i = 0; i < size; i++) {
          if (elementsDecoded.isNullAt(offset + i)) {
            // If a NULL value is encountered, min is always NULL
            processNull(row);
            break;
          } else if (
              elementsDecoded.valueAt<T>(offset + i) <
              elementsDecoded.valueAt<T>(min_element_index)) {
            min_element_index = offset + i;
          }
        }
        rawOutIndices[row] = min_element_index;
      }
    });
  }

  return BaseVector::wrapInDictionary(
      nulls, outIndices, rows.size(), baseArray->elements());
}

class ArrayMinFunction : public exec::VectorFunction {
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
    auto elements = arrayHolder.get()->base()->as<ArrayVector>()->elements();

    exec::LocalSelectivityVector nestedRows(context, elements->size());
    nestedRows.get()->setAll();

    exec::LocalDecodedVector elementsHolder(
        context, *elements, *nestedRows.get());

    VectorPtr localResult;
    localResult = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        applyTyped,
        arrayVector->type()->asArray().elementType()->kind(),
        rows,
        *arrayHolder.get(),
        *elementsHolder.get(),
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
    udf_array_min,
    ArrayMinFunction::signatures(),
    std::make_unique<ArrayMinFunction>());

} // namespace facebook::velox::functions
