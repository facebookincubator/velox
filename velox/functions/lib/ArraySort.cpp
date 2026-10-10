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

#include <folly/container/F14Set.h>

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/functions/lib/ArraySort.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/type/FloatingPointUtil.h"

namespace facebook::velox::functions {
namespace {

struct SortElementsOptions {
  bool ascending;
  // Position of top-level null elements or sort keys.
  bool nullsFirst;
  // Position of nulls nested inside complex elements or sort keys.
  bool nestedNullsFirst;
  bool throwOnNestedNull;
  // If true, elements that compare equal keep their original relative order.
  bool stable;
  // If true, throw a user error when an array with at least two elements has
  // a null element or sort key.
  bool rejectNulls;
};

// Returns indices that sort the elements of each array in 'rows' by the
// corresponding values in 'inputElements'. Only arrays in 'rowsToSort' are
// sorted and only their values in 'inputElements' are read. The elements of
// other non-null arrays in 'rows' keep their original order. 'rowsToSort'
// must be a subset of 'rows'.
BufferPtr sortElements(
    const SelectivityVector& rows,
    const SelectivityVector& rowsToSort,
    const ArrayVector& inputArray,
    const BaseVector& inputElements,
    const SortElementsOptions& options,
    exec::EvalCtx& context) {
  const SelectivityVector inputElementRows =
      toElementRows(inputElements.size(), rowsToSort, &inputArray);
  exec::LocalDecodedVector decodedElements(
      context, inputElements, inputElementRows);
  const auto* baseElementsVector = decodedElements->base();

  // Allocate new vectors for indices.
  BufferPtr indices =
      allocateIndices(inputArray.elements()->size(), context.pool());
  vector_size_t* rawIndices = indices->asMutable<vector_size_t>();

  CompareFlags flags{
      .nullsFirst = options.nestedNullsFirst, .ascending = options.ascending};
  if (options.throwOnNestedNull) {
    flags.nullHandlingMode =
        CompareFlags::NullHandlingMode::kNullAsIndeterminate;
  }

  auto decodedIndices = decodedElements->indices();
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    // Offsets and sizes of null arrays are undefined.
    if (inputArray.isNullAt(row)) {
      return;
    }

    const auto size = inputArray.sizeAt(row);
    const auto offset = inputArray.offsetAt(row);

    for (auto i = offset; i < offset + size; ++i) {
      rawIndices[i] = i;
    }

    if (size < 2 || !rowsToSort.isValid(row)) {
      return;
    }

    if (options.rejectNulls) {
      for (auto i = offset; i < offset + size; ++i) {
        if (decodedElements->isNullAt(i)) {
          VELOX_USER_FAIL(
              "array_sort comparator does not support NULL sort keys");
        }
      }
    }

    auto compare = [&](vector_size_t a, vector_size_t b) {
      if (a == b) {
        return 0;
      }
      bool aNull = decodedElements->isNullAt(a);
      bool bNull = decodedElements->isNullAt(b);

      if (aNull && bNull) {
        return 0;
      }
      if (aNull) {
        return options.nullsFirst ? -1 : 1;
      }
      if (bNull) {
        return options.nullsFirst ? 1 : -1;
      }

      std::optional<int32_t> result = baseElementsVector->compare(
          baseElementsVector, decodedIndices[a], decodedIndices[b], flags);

      if (!result.has_value()) {
        VELOX_USER_FAIL("Ordering nulls is not supported");
      }

      return result.value();
    };
    // Break ties by original position to make the sort stable without
    // auxiliary storage.
    std::sort(
        rawIndices + offset,
        rawIndices + offset + size,
        [&](vector_size_t a, vector_size_t b) {
          const auto result = compare(a, b);
          return result < 0 || (options.stable && result == 0 && a < b);
        });
  });

  return indices;
}

void applyComplexType(
    const SelectivityVector& rows,
    ArrayVector* inputArray,
    const SortElementsOptions& options,
    exec::EvalCtx& context,
    VectorPtr& resultElements) {
  auto inputElements = inputArray->elements();
  auto indices =
      sortElements(rows, rows, *inputArray, *inputElements, options, context);
  resultElements = BaseVector::transpose(indices, std::move(inputElements));
}

template <typename T>
inline void swapWithNull(
    FlatVector<T>* vector,
    vector_size_t index,
    vector_size_t nullIndex) {
  // Values are already present in vector stringBuffers. Don't create additional
  // copy.
  if constexpr (std::is_same_v<T, StringView>) {
    vector->setNoCopy(nullIndex, vector->valueAt(index));
  } else {
    vector->set(nullIndex, vector->valueAt(index));
  }
  vector->setNull(index, true);
}

// Moves the nulls in [offset, offset + size) to the beginning or end of the
// range while preserving the relative order of non-null values. Returns the
// number of nulls.
template <typename T>
vector_size_t moveNullsStable(
    FlatVector<T>* vector,
    vector_size_t offset,
    vector_size_t size,
    bool nullsFirst) {
  T* rawValues = vector->mutableRawValues();
  vector_size_t numNulls = 0;
  if (nullsFirst) {
    auto write = offset + size;
    for (auto i = offset + size - 1; i >= offset; --i) {
      if (vector->isNullAt(i)) {
        ++numNulls;
      } else {
        rawValues[--write] = rawValues[i];
      }
    }
  } else {
    auto write = offset;
    for (auto i = offset; i < offset + size; ++i) {
      if (vector->isNullAt(i)) {
        ++numNulls;
      } else {
        rawValues[write++] = rawValues[i];
      }
    }
  }

  if (numNulls > 0) {
    const auto nullsBegin = nullsFirst ? offset : offset + size - numNulls;
    for (auto i = offset; i < offset + size; ++i) {
      vector->setNull(i, i >= nullsBegin && i < nullsBegin + numNulls);
    }
  }
  return numNulls;
}

template <TypeKind kind>
void applyScalarType(
    const SelectivityVector& rows,
    const ArrayVector* inputArray,
    bool ascending,
    bool nullsFirst,
    exec::EvalCtx& context,
    VectorPtr& resultElements,
    bool stable) {
  using T = typename TypeTraits<kind>::NativeType;

  // Copy array elements to new vector.
  const VectorPtr& inputElements = inputArray->elements();
  VELOX_DCHECK(kind == inputElements->typeKind());
  const SelectivityVector inputElementRows =
      toElementRows(inputElements->size(), rows, inputArray);
  const vector_size_t elementsCount = inputElementRows.size();

  // TODO: consider to use dictionary wrapping to avoid the direct sorting on
  // the scalar values as we do for complex data type if this runs slow in
  // practice.
  resultElements =
      BaseVector::create(inputElements->type(), elementsCount, context.pool());
  resultElements->copy(
      inputElements.get(), inputElementRows, /*toSourceRow=*/nullptr);

  auto flatResults = resultElements->asFlatVector<T>();

  auto processRow = [&](vector_size_t row) {
    const auto size = inputArray->sizeAt(row);
    const auto offset = inputArray->offsetAt(row);
    if (size == 0) {
      return;
    }
    vector_size_t numNulls = 0;
    if constexpr (kind == TypeKind::REAL || kind == TypeKind::DOUBLE) {
      if (stable) {
        numNulls = moveNullsStable<T>(flatResults, offset, size, nullsFirst);
        const auto startRow = offset + (nullsFirst ? numNulls : 0);
        const auto endRow = startRow + size - numNulls;
        T* resultRawValues = flatResults->mutableRawValues();
        if (ascending) {
          std::stable_sort(
              resultRawValues + startRow,
              resultRawValues + endRow,
              util::floating_point::NaNAwareLessThan<T>());
        } else {
          std::stable_sort(
              resultRawValues + startRow,
              resultRawValues + endRow,
              util::floating_point::NaNAwareGreaterThan<T>());
        }
        return;
      }
    }

    if (nullsFirst) {
      // Move nulls to beginning of array.
      for (vector_size_t i = 0; i < size; ++i) {
        if (flatResults->isNullAt(offset + i)) {
          swapWithNull<T>(flatResults, offset + numNulls, offset + i);
          ++numNulls;
        }
      }
    } else {
      // Move nulls to end of array.
      for (vector_size_t i = size - 1; i >= 0; --i) {
        if (flatResults->isNullAt(offset + i)) {
          swapWithNull<T>(
              flatResults, offset + size - numNulls - 1, offset + i);
          ++numNulls;
        }
      }
    }
    // Exclude null values while sorting.
    const auto startRow = offset + (nullsFirst ? numNulls : 0);
    const auto endRow = startRow + size - numNulls;

    if constexpr (kind == TypeKind::BOOLEAN) {
      uint64_t* rawBits = flatResults->template mutableRawValues<uint64_t>();
      const auto numOneBits = bits::countBits(rawBits, startRow, endRow);

      if (ascending) {
        const auto endZeroRow = endRow - numOneBits;
        bits::fillBits(rawBits, startRow, endZeroRow, false);
        bits::fillBits(rawBits, endZeroRow, endRow, true);
      } else {
        bits::fillBits(rawBits, startRow, startRow + numOneBits, true);
        bits::fillBits(rawBits, startRow + numOneBits, endRow, false);
      }
    } else if constexpr (kind == TypeKind::REAL || kind == TypeKind::DOUBLE) {
      T* resultRawValues = flatResults->mutableRawValues();
      if (ascending) {
        std::sort(
            resultRawValues + startRow,
            resultRawValues + endRow,
            util::floating_point::NaNAwareLessThan<T>());
      } else {
        std::sort(
            resultRawValues + startRow,
            resultRawValues + endRow,
            util::floating_point::NaNAwareGreaterThan<T>());
      }
    } else {
      T* resultRawValues = flatResults->mutableRawValues();
      if (ascending) {
        std::sort(resultRawValues + startRow, resultRawValues + endRow);
      } else {
        std::sort(
            resultRawValues + startRow,
            resultRawValues + endRow,
            std::greater<T>());
      }
    }
  };
  rows.applyToSelected(processRow);
}

// See documentation at https://prestodb.io/docs/current/functions/array.html
template <TypeKind Kind>
class ArraySortFunction : public exec::VectorFunction {
 public:
  /// This class implements the array_sort query function. Takes an array as
  /// input and sorts it in ascending order and null elements will be placed at
  /// the end of the returned array.
  ///
  /// Along with the set, we maintain a `hasNull` flag that indicates whether
  /// null is present in the array.
  ///
  /// Zero element copy for complex data type:
  ///
  /// In order to prevent copies of array elements with complex data type, the
  /// function reuses the internal elements() vector from the original
  /// ArrayVector. A new vector is created containing the indices of the sorted
  /// elements in the output, and wrapped into a DictionaryVector. The 'lengths'
  /// and 'offsets' vectors that control where output arrays start and end
  /// remain the same in the output ArrayVector.

  explicit ArraySortFunction(const ArraySortOptions& options)
      : options_{options} {}

  // Execute function.
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    auto& arg = args[0];

    VectorPtr localResult;

    // Input can be constant or flat.
    if constexpr (Kind == TypeKind::UNKNOWN) {
      // All elements are NULL. Hence, sorting doesn't change anything.
      localResult = arg;
    } else if (arg->isConstantEncoding()) {
      auto* constantArray = arg->as<ConstantVector<ComplexType>>();
      const auto& flatArray = constantArray->valueVector();
      const auto flatIndex = constantArray->index();

      exec::LocalSingleRow singleRow(context, flatIndex);
      localResult = applyFlat(*singleRow, flatArray, context);
      localResult =
          BaseVector::wrapInConstant(rows.end(), flatIndex, localResult);
    } else {
      localResult = applyFlat(rows, arg, context);
    }

    context.moveOrCopyResult(localResult, rows, result);
  }

 private:
  VectorPtr applyFlat(
      const SelectivityVector& rows,
      const VectorPtr& arg,
      exec::EvalCtx& context) const {
    // Acquire the array elements vector.
    auto inputArray = arg->as<ArrayVector>();
    VectorPtr resultElements;

    constexpr bool kIsFloatingPoint =
        Kind == TypeKind::REAL || Kind == TypeKind::DOUBLE;
    // The in-place scalar sort supports stable sorting only for floating-point
    // types. Other stable sorts, e.g. of types with custom comparison, use the
    // index-based path.
    if constexpr (velox::TypeTraits<Kind>::isPrimitiveType) {
      if (!options_.stable || kIsFloatingPoint) {
        VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            applyScalarType,
            Kind,
            rows,
            inputArray,
            options_.ascending,
            options_.nullsFirst,
            context,
            resultElements,
            options_.stable);
      } else {
        applyComplexType(
            rows, inputArray, sortElementsOptions(), context, resultElements);
      }
    } else {
      applyComplexType(
          rows, inputArray, sortElementsOptions(), context, resultElements);
    }

    return std::make_shared<ArrayVector>(
        context.pool(),
        inputArray->type(),
        inputArray->nulls(),
        rows.end(),
        inputArray->offsets(),
        inputArray->sizes(),
        resultElements,
        inputArray->getNullCount());
  }

  SortElementsOptions sortElementsOptions() const {
    return {
        .ascending = options_.ascending,
        .nullsFirst = options_.nullsFirst,
        .nestedNullsFirst = options_.nestedNullsFirst,
        .throwOnNestedNull = options_.throwOnNestedNull,
        .stable = options_.stable,
        .rejectNulls = false};
  }

  const ArraySortOptions options_;
};

class ArraySortLambdaFunction : public exec::VectorFunction {
 public:
  explicit ArraySortLambdaFunction(const ArraySortLambdaOptions& options)
      : options_{options} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // Flatten input array.
    exec::LocalDecodedVector arrayDecoder(context, *args[0], rows);
    auto& decodedArray = *arrayDecoder.get();

    auto flatArray = flattenArray(rows, args[0], decodedArray);

    auto numElements = flatArray->elements()->size();
    const std::vector<VectorPtr> lambdaArgs = {flatArray->elements()};
    SelectivityVector rowsToSort{rows};
    if (options_.skipLambdaForTrivialArrays) {
      rows.applyToSelected([&](vector_size_t row) {
        if (flatArray->isNullAt(row) || flatArray->sizeAt(row) < 2) {
          rowsToSort.setValid(row, false);
        }
      });
      rowsToSort.updateBounds();

      if (!rowsToSort.hasSelections()) {
        context.moveOrCopyResult(flatArray, rows, result);
        return;
      }
    }

    SelectivityVector validRowsInReusedResult =
        toElementRows<ArrayVector>(numElements, rowsToSort, flatArray.get());

    VectorPtr newElements;
    // Compute sorting keys.
    applyLambdaToElements<ArrayVector>(
        args[1],
        rowsToSort,
        numElements,
        flatArray,
        lambdaArgs,
        validRowsInReusedResult,
        context,
        newElements);

    // Sort 'newElements'. Only arrays in 'rowsToSort' have sort keys. Nulls
    // nested inside sort keys are ordered as the smallest values.
    auto indices = sortElements(
        rows,
        rowsToSort,
        *flatArray,
        *newElements,
        {.ascending = options_.ascending,
         .nullsFirst = false,
         .nestedNullsFirst = options_.ascending,
         .throwOnNestedNull = options_.throwOnNestedNull,
         .stable = true,
         .rejectNulls = options_.rejectNullSortKeys},
        context);
    auto sortedElements = BaseVector::wrapInDictionary(
        nullptr,
        indices,
        indices->size() / sizeof(vector_size_t),
        flatArray->elements());

    // Set nulls for rows not present in 'rows'.
    BufferPtr newNulls = addNullsForUnselectedRows(flatArray, rows);

    VectorPtr localResult = std::make_shared<ArrayVector>(
        flatArray->pool(),
        flatArray->type(),
        std::move(newNulls),
        rows.end(),
        flatArray->offsets(),
        flatArray->sizes(),
        sortedElements);
    context.moveOrCopyResult(localResult, rows, result);
  }

 private:
  const ArraySortLambdaOptions options_;
};

// Create function template based on type.
template <TypeKind kind>
std::shared_ptr<exec::VectorFunction> createTyped(
    const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
    const ArraySortOptions& options) {
  return std::make_shared<ArraySortFunction<kind>>(options);
}

// Define function signature.
std::vector<std::shared_ptr<exec::FunctionSignature>> signatures(
    bool withComparator) {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures = {
      // array(T) -> array(T)
      exec::FunctionSignatureBuilder()
          .orderableTypeVariable("T")
          .returnType("array(T)")
          .argumentType("array(T)")
          .build(),
      // array(T), function(T,U) -> array(T)
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .orderableTypeVariable("U")
          .returnType("array(T)")
          .argumentType("array(T)")
          .constantArgumentType("function(T,U)")
          .build(),
  };

  if (withComparator) {
    signatures.push_back(
        // array(T), function(T,T,integer) -> array(T)
        exec::FunctionSignatureBuilder()
            .typeVariable("T")
            .returnType("array(T)")
            .argumentType("array(T)")
            .constantArgumentType("function(T,T,integer)")
            .build());
  }
  return signatures;
}

std::vector<std::shared_ptr<exec::FunctionSignature>>
internalCanonicalizeSignatures() {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures = {
      // array(T) -> array(T)
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("array(T)")
          .argumentType("array(T)")
          .build()};
  return signatures;
}

std::shared_ptr<exec::VectorFunction> makeArraySortAscNoThrowOnNestedNull(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  return makeArraySort(
      name,
      inputArgs,
      config,
      {.ascending = true,
       .nullsFirst = false,
       .nestedNullsFirst = false,
       .throwOnNestedNull = false});
}

core::CallTypedExprPtr asArraySortCall(
    const std::string& prefix,
    const core::TypedExprPtr& expr) {
  if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr)) {
    if (call->name() == prefix + "array_sort") {
      return call;
    }
  }
  return nullptr;
}

// Returns true if values of 'type' can compare equal while remaining
// distinguishable, e.g. -0.0 and 0.0, so that sort stability is observable.
bool hasDistinguishableEqualValues(const TypePtr& type) {
  if (type->providesCustomComparison() || type->kind() == TypeKind::REAL ||
      type->kind() == TypeKind::DOUBLE) {
    return true;
  }

  for (auto i = 0; i < type->size(); ++i) {
    if (hasDistinguishableEqualValues(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

} // namespace

std::shared_ptr<exec::VectorFunction> makeArraySortLambdaFunction(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config,
    bool ascending,
    bool throwOnNestedNull) {
  return makeArraySortLambdaFunction(
      name,
      inputArgs,
      config,
      {.ascending = ascending, .throwOnNestedNull = throwOnNestedNull});
}

std::shared_ptr<exec::VectorFunction> makeArraySortLambdaFunction(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/,
    const ArraySortLambdaOptions& options) {
  VELOX_CHECK_EQ(inputArgs.size(), 2);
  return std::make_shared<ArraySortLambdaFunction>(options);
}

std::shared_ptr<exec::VectorFunction> makeArraySort(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config,
    bool ascending,
    bool nullsFirst,
    bool throwOnNestedNull) {
  return makeArraySort(
      name,
      inputArgs,
      config,
      {.ascending = ascending,
       .nullsFirst = nullsFirst,
       .nestedNullsFirst = nullsFirst,
       .throwOnNestedNull = throwOnNestedNull});
}

std::shared_ptr<exec::VectorFunction> makeArraySort(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/,
    const ArraySortOptions& options) {
  const auto elementType = inputArgs.front().type->childAt(0);
  if (elementType->isUnknown()) {
    return createTyped<TypeKind::UNKNOWN>(inputArgs, options);
  }

  // Stability is only observable when equal values are distinguishable. Other
  // types keep the faster unstable sort.
  auto effectiveOptions = options;
  effectiveOptions.stable =
      options.stable && hasDistinguishableEqualValues(elementType);

  return VELOX_DYNAMIC_TYPE_DISPATCH(
      createTyped, elementType->kind(), inputArgs, effectiveOptions);
}

std::vector<std::shared_ptr<exec::FunctionSignature>> arraySortSignatures(
    bool withComparator) {
  return signatures(withComparator);
}

core::TypedExprPtr rewriteArraySortCall(
    const std::string& prefix,
    const core::TypedExprPtr& expr,
    const std::shared_ptr<SimpleComparisonChecker> checker) {
  return rewriteArraySortCall(prefix, expr, checker, ArraySortRewriteOptions{});
}

core::TypedExprPtr rewriteArraySortCall(
    const std::string& prefix,
    const core::TypedExprPtr& expr,
    const std::shared_ptr<SimpleComparisonChecker> checker,
    const ArraySortRewriteOptions& options) {
  auto call = asArraySortCall(prefix, expr);
  if (call == nullptr || call->inputs().size() != 2) {
    return nullptr;
  }
  auto lambda =
      dynamic_cast<const core::LambdaTypedExpr*>(call->inputs()[1].get());
  VELOX_CHECK_NOT_NULL(lambda);
  // Extract 'transform' from the comparison lambda:
  //  (x, y) -> if(func(x) < func(y),...) ===> x -> func(x).
  if (lambda->signature()->size() != 2) {
    return nullptr;
  }
  static constexpr const char* kNotSupported =
      "array_sort with comparator lambda that cannot be rewritten "
      "into a transform is not supported: {}";

  if (auto comparison = checker->isSimpleComparison(
          prefix, *lambda, options.supportsArbitraryComparatorResults)) {
    std::string name;
    if (options.rejectNullSortKeys) {
      name = comparison->isLessThen
          ? prefix + "$internal$array_sort_comparator"
          : prefix + "$internal$array_sort_comparator_desc";
    } else {
      name = comparison->isLessThen ? prefix + "array_sort"
                                    : prefix + "array_sort_desc";
    }

    if (!comparison->expr->type()->isOrderable()) {
      VELOX_USER_FAIL(kNotSupported, lambda->toString());
    }

    auto rewritten = std::make_shared<core::CallTypedExpr>(
        call->type(),
        name,
        call->inputs()[0],
        std::make_shared<core::LambdaTypedExpr>(
            ROW({lambda->signature()->nameOf(0)},
                {lambda->signature()->childAt(0)}),
            comparison->expr));

    return rewritten;
  }

  VELOX_USER_FAIL(kNotSupported, lambda->toString());
}

// An internal function to canonicalize an array to allow for comparisons. Used
// in AggregationFuzzerTest. Details in
// https://github.com/facebookincubator/velox/issues/6999.
VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_$internal$canonicalize,
    internalCanonicalizeSignatures(),
    makeArraySortAscNoThrowOnNestedNull);

} // namespace facebook::velox::functions
