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
#include <velox/common/base/VeloxException.h>

#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/vector/NullsBuilder.h"

namespace facebook::velox::functions {

template <typename T, typename Func>
VectorPtr fastMinMax(const VectorPtr& in, const Func& func) {
  const auto numRows = in->size();
  auto result = std::static_pointer_cast<FlatVector<T>>(
      BaseVector::create(in->type()->childAt(0), numRows, in->pool()));
  auto rawResults = result->mutableRawValues();

  auto arrayVector = in->as<ArrayVector>();
  auto rawOffsets = arrayVector->rawOffsets();
  auto rawSizes = arrayVector->rawSizes();
  auto rawElements = arrayVector->elements()->as<FlatVector<T>>()->rawValues();
  for (auto row = 0; row < numRows; ++row) {
    const auto start = rawOffsets[row];
    const auto end = start + rawSizes[row];
    if (start == end) {
      result->setNull(row, true); // NULL
    } else {
      rawResults[row] = *func(rawElements + start, rawElements + end);
    }
  }

  return result;
}

template <typename T>
VectorPtr fastMin(const VectorPtr& in) {
  return fastMinMax<T, decltype(std::min_element<const T*>)>(
      in, std::min_element<const T*>);
}

template <typename T>
VectorPtr fastMax(const VectorPtr& in) {
  return fastMinMax<T, decltype(std::max_element<const T*>)>(
      in, std::max_element<const T*>);
}

template <typename T>
struct ArrayMinSimpleFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool callNullFree(
      TInput& out,
      const null_free_arg_type<Array<TInput>>& array) {
    if (array.size() == 0) {
      return false; // NULL
    }

    auto min = INT32_MAX;
    for (auto i = 0; i < array.size(); i++) {
      auto v = array[i];
      if (v < min) {
        min = v;
      }
    }
    out = min;
    return true;
  }
};

template <typename T>
struct ArrayMinSimpleFunctionIterator {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool call(
      TInput& out,
      const arg_type<Array<TInput>>& array) {
    const auto size = array.size();
    if (size == 0) {
      return false; // NULL
    }

    auto min = INT32_MAX;
    if (array.mayHaveNulls()) {
      for (const auto& item : array) {
        if (!item.has_value()) {
          return false; // NULL
        }
        auto v = item.value();
        if (v < min) {
          min = v;
        }
      }
    } else {
      for (const auto& item : array) {
        auto v = item.value();
        if (v < min) {
          min = v;
        }
      }
    }

    out = min;
    return true;
  }
};

// Returns the minimum value in an array ignoring nulls.
// The point of this is to exercise SkipNullsIterator.
template <typename T>
struct ArrayMinSimpleFunctionSkipNullIterator {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool call(
      TInput& out,
      const arg_type<Array<TInput>>& array) {
    const auto size = array.size();
    if (size == 0) {
      return false; // NULL
    }

    bool hasValue = false;
    auto min = INT32_MAX;
    for (const auto& item : array.skipNulls()) {
      hasValue = true;
      if (item < min) {
        min = item;
      }
    }

    if (!hasValue) {
      return false;
    }

    out = min;
    return true;
  }
};

template <template <typename> class F, TypeKind kind>
VectorPtr applyTyped(
    const SelectivityVector& rows,
    const ArrayVector& arrayVector,
    DecodedVector& elementsDecoded,
    exec::EvalCtx& context) {
  auto pool = context.pool();
  using T = typename TypeTraits<kind>::NativeType;

  auto rawSizes = arrayVector.rawSizes();
  auto rawOffsets = arrayVector.rawOffsets();

  BufferPtr indices = allocateIndices(rows.size(), pool);
  auto rawIndices = indices->asMutable<vector_size_t>();

  // Create nulls for lazy initialization.
  NullsBuilder nullsBuilder(rows.size(), pool);

  if (elementsDecoded.isIdentityMapping() && !elementsDecoded.mayHaveNulls()) {
    if constexpr (std::is_same_v<bool, T>) {
      auto rawElements = elementsDecoded.data<uint64_t>();
      rows.applyToSelected([&](auto row) {
        auto size = rawSizes[row];
        if (size == 0) {
          nullsBuilder.setNull(row);
        } else {
          auto offset = rawOffsets[row];
          auto elementIndex = offset;
          for (auto i = offset + 1; i < offset + size; i++) {
            if (F<T>()(
                    bits::isBitSet(rawElements, i),
                    bits::isBitSet(rawElements, elementIndex))) {
              elementIndex = i;
            }
          }
          rawIndices[row] = elementIndex;
        }
      });
    } else {
      auto rawElements = elementsDecoded.data<T>();
      rows.applyToSelected([&](auto row) {
        auto size = rawSizes[row];
        if (size == 0) {
          nullsBuilder.setNull(row);
        } else {
          auto offset = rawOffsets[row];
          auto elementIndex = offset;
          for (auto i = offset + 1; i < offset + size; i++) {
            if (F<T>()(rawElements[i], rawElements[elementIndex])) {
              elementIndex = i;
            }
          }
          rawIndices[row] = elementIndex;
        }
      });
    }
  } else {
    rows.applyToSelected([&](auto row) {
      auto size = rawSizes[row];
      if (size == 0) {
        nullsBuilder.setNull(row);
      } else {
        auto offset = rawOffsets[row];
        auto elementIndex = offset;
        for (auto i = offset; i < offset + size; i++) {
          if (elementsDecoded.isNullAt(i)) {
            // If a NULL value is encountered, min/max are always NULL
            nullsBuilder.setNull(row);
            break;
          } else if (F<T>()(
                         elementsDecoded.valueAt<T>(i),
                         elementsDecoded.valueAt<T>(elementIndex))) {
            elementIndex = i;
          }
        }
        rawIndices[row] = elementIndex;
      }
    });
  }

  return BaseVector::wrapInDictionary(
      nullsBuilder.build(), indices, rows.size(), arrayVector.elements());
}
// Decoder-based unoptimized implementation of min/max used to compare
// performance of simple function min/max.
template <template <typename> class F>
class ArrayMinMaxFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);
    auto arrayVector = args[0]->asUnchecked<ArrayVector>();

    auto elementsVector = arrayVector->elements();
    exec::LocalSelectivityVector elementsRows(context, elementsVector->size());
    exec::LocalDecodedVector elementsHolder(
        context, *elementsVector, *elementsRows.get());
    auto localResult = VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
        applyTyped,
        F,
        elementsVector->typeKind(),
        rows,
        *arrayVector,
        *elementsHolder.get(),
        context);
    context.moveOrCopyResult(localResult, rows, result);
  }
};

inline std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  static const std::vector<std::string> kSupportedTypeNames = {
      "boolean",
      "tinyint",
      "smallint",
      "integer",
      "bigint",
      "real",
      "double",
      "varchar",
      "timestamp",
  };

  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  signatures.reserve(kSupportedTypeNames.size());
  for (const auto& typeName : kSupportedTypeNames) {
    signatures.emplace_back(
        exec::FunctionSignatureBuilder()
            .returnType(typeName)
            .argumentType(fmt::format("array({})", typeName))
            .build());
  }
  return signatures;
}

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_array_min_basic,
    functions::signatures(),
    std::make_unique<ArrayMinMaxFunction<std::less>>());

} // namespace facebook::velox::functions
