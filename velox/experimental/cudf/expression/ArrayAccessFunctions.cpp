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

#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ArrayAccessFunctions.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

int64_t readConstantIntegralValue(const velox::exec::ConstantExpr& expr) {
  switch (expr.type()->kind()) {
    case TypeKind::TINYINT:
      return expr.value()->as<SimpleVector<int8_t>>()->valueAt(0);
    case TypeKind::SMALLINT:
      return expr.value()->as<SimpleVector<int16_t>>()->valueAt(0);
    case TypeKind::INTEGER:
      return expr.value()->as<SimpleVector<int32_t>>()->valueAt(0);
    case TypeKind::BIGINT:
      return expr.value()->as<SimpleVector<int64_t>>()->valueAt(0);
    default:
      VELOX_UNSUPPORTED(
          "Unsupported array access index type {}", expr.type()->toString());
  }
}

cudf::data_type indexDataType(cudf::type_id id) {
  return cudf::data_type{id};
}

template <typename T>
cudf::data_type indexDataType() {
  if constexpr (std::is_same_v<T, int32_t>) {
    return indexDataType(cudf::type_id::INT32);
  } else {
    return indexDataType(cudf::type_id::INT64);
  }
}

std::unique_ptr<cudf::column> extractNullIndex(
    cudf::lists_column_view const& listsView,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto nullIndex = cudf::numeric_scalar<cudf::size_type>(0, false, stream, mr);
  auto nullIndices =
      cudf::make_column_from_scalar(nullIndex, listsView.size(), stream, mr);
  return cudf::lists::extract_list_element(
      listsView, nullIndices->view(), stream, mr);
}

std::optional<int64_t> normalizeConstantIndex(
    int64_t index,
    const ArrayAccessPolicy& policy) {
  if (policy.indexStartsAtOne && index == 0) {
    VELOX_USER_FAIL("SQL array indices start at 1. Got 0.");
  }

  if (index < 0) {
    if (!policy.allowNegativeIndices) {
      if (policy.nullOnNegativeIndices) {
        return std::nullopt;
      }
      VELOX_USER_FAIL(
          "Array subscript index cannot be negative, Index: {}", index);
    }

    if (policy.nullOnNegativeIndices) {
      return std::nullopt;
    }

    return index;
  }

  return policy.indexStartsAtOne ? index - 1 : index;
}

std::unique_ptr<cudf::column> sanitizeBoolMask(
    cudf::column_view mask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Comparisons against null inputs produce nulls. Treat them as false before
  // combining masks or reducing them to decide whether to throw.
  auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream, mr);
  return cudf::replace_nulls(mask, falseScalar, stream, mr);
}

std::unique_ptr<cudf::column> combineBoolMasks(
    cudf::column_view lhs,
    cudf::column_view rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // These are BOOL8 value columns, not validity bitmasks. cudf::bitmask_or
  // would combine null masks instead of row-wise boolean values.
  return cudf::binary_operation(
      lhs,
      rhs,
      cudf::binary_operator::BITWISE_OR,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
}

std::unique_ptr<cudf::column> mergeBoolMasks(
    std::unique_ptr<cudf::column> lhs,
    std::unique_ptr<cudf::column> rhs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (lhs == nullptr) {
    return rhs;
  }
  if (rhs == nullptr) {
    return lhs;
  }

  return combineBoolMasks(lhs->view(), rhs->view(), stream, mr);
}

bool maskHasTrue(
    cudf::column_view mask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (mask.is_empty()) {
    return false;
  }

  auto sanitized = sanitizeBoolMask(mask, stream, mr);
  auto anyAgg = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto anyScalar = cudf::reduce(
      sanitized->view(),
      *anyAgg,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto const& boolScalar =
      static_cast<cudf::numeric_scalar<bool> const&>(*anyScalar);
  return boolScalar.is_valid(stream) && boolScalar.value(stream);
}

std::unique_ptr<cudf::column> applyNullMask(
    cudf::column_view col,
    cudf::column_view nullMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto nullScalar =
      cudf::make_default_constructed_scalar(col.type(), stream, mr);
  nullScalar->set_valid_async(false, stream);
  return cudf::copy_if_else(*nullScalar, col, nullMask, stream, mr);
}

std::unique_ptr<cudf::column> castSizes(
    cudf::column_view const& sizesView,
    cudf::data_type indexType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return indexType.id() == cudf::type_id::INT32
      ? std::make_unique<cudf::column>(sizesView, stream)
      : cudf::cast(sizesView, indexType, stream, mr);
}

template <typename IndexType>
std::unique_ptr<cudf::column> outOfBoundsMask(
    cudf::column_view const& normalized,
    cudf::column_view const& sizes,
    const ArrayAccessPolicy& policy,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto zero = cudf::numeric_scalar<IndexType>(0, true, stream, mr);
  std::unique_ptr<cudf::column> lowerBound;
  std::unique_ptr<cudf::column> belowLower;

  if (policy.allowNegativeIndices && !policy.nullOnNegativeIndices) {
    lowerBound = cudf::binary_operation(
        zero,
        sizes,
        cudf::binary_operator::SUB,
        indexDataType<IndexType>(),
        stream,
        mr);
    belowLower = cudf::binary_operation(
        normalized,
        lowerBound->view(),
        cudf::binary_operator::LESS,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
  } else {
    belowLower = cudf::binary_operation(
        normalized,
        zero,
        cudf::binary_operator::LESS,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
  }

  auto aboveUpper = cudf::binary_operation(
      normalized,
      sizes,
      cudf::binary_operator::GREATER_EQUAL,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto boundsMask = combineBoolMasks(
      sanitizeBoolMask(belowLower->view(), stream, mr)->view(),
      sanitizeBoolMask(aboveUpper->view(), stream, mr)->view(),
      stream,
      mr);

  if constexpr (std::is_same_v<IndexType, int64_t>) {
    // cuDF list extraction expects indices as cudf::size_type, so int64
    // indices must stay within the representable cudf::size_type range before
    // the cast near the end of normalizeAndValidateIndicesTyped.
    auto minSizeTypeIndex = cudf::numeric_scalar<int64_t>(
        std::numeric_limits<cudf::size_type>::min(), true, stream, mr);
    auto maxSizeTypeIndex = cudf::numeric_scalar<int64_t>(
        std::numeric_limits<cudf::size_type>::max(), true, stream, mr);
    auto belowSizeTypeMin = sanitizeBoolMask(
        cudf::binary_operation(
            normalized,
            minSizeTypeIndex,
            cudf::binary_operator::LESS,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            mr)
            ->view(),
        stream,
        mr);
    auto aboveSizeTypeMax = sanitizeBoolMask(
        cudf::binary_operation(
            normalized,
            maxSizeTypeIndex,
            cudf::binary_operator::GREATER,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            mr)
            ->view(),
        stream,
        mr);
    auto sizeTypeRangeMask = combineBoolMasks(
        belowSizeTypeMin->view(), aboveSizeTypeMax->view(), stream, mr);
    boundsMask = combineBoolMasks(
        boundsMask->view(), sizeTypeRangeMask->view(), stream, mr);
  }

  return boundsMask;
}

template <typename IndexType>
std::unique_ptr<cudf::column> normalizeAndValidateIndicesTyped(
    cudf::column_view const& rawIndexView,
    cudf::column_view const& sizesView,
    const ArrayAccessPolicy& policy,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto sizes = castSizes(sizesView, rawIndexView.type(), stream, mr);
  auto zero = cudf::numeric_scalar<IndexType>(0, true, stream, mr);
  auto one = cudf::numeric_scalar<IndexType>(1, true, stream, mr);

  auto rawLessZero = sanitizeBoolMask(
      cudf::binary_operation(
          rawIndexView,
          zero,
          cudf::binary_operator::LESS,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          mr)
          ->view(),
      stream,
      mr);

  if (policy.indexStartsAtOne) {
    auto zeroMask = sanitizeBoolMask(
        cudf::binary_operation(
            rawIndexView,
            zero,
            cudf::binary_operator::EQUAL,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            mr)
            ->view(),
        stream,
        mr);
    if (maskHasTrue(zeroMask->view(), stream, mr)) {
      VELOX_USER_FAIL("SQL array indices start at 1. Got 0.");
    }
  }

  if (!policy.allowNegativeIndices && !policy.nullOnNegativeIndices &&
      maskHasTrue(rawLessZero->view(), stream, mr)) {
    VELOX_USER_FAIL("Array subscript index cannot be negative");
  }

  std::unique_ptr<cudf::column> positiveNormalizedColumn;
  auto positiveNormalized = rawIndexView;
  if (policy.indexStartsAtOne) {
    // Preserve negative indices until the policy-specific branch below. This
    // avoids turning -1 into -2 while converting positive one-based indices to
    // zero-based indices.
    auto oneBasedSafeForSub =
        cudf::copy_if_else(one, rawIndexView, rawLessZero->view(), stream, mr);
    positiveNormalizedColumn = cudf::binary_operation(
        oneBasedSafeForSub->view(),
        one,
        cudf::binary_operator::SUB,
        indexDataType<IndexType>(),
        stream,
        mr);
    positiveNormalized = positiveNormalizedColumn->view();
  }

  std::unique_ptr<cudf::column> normalized;
  if (policy.allowNegativeIndices && !policy.nullOnNegativeIndices) {
    normalized = cudf::copy_if_else(
        rawIndexView, positiveNormalized, rawLessZero->view(), stream, mr);
  } else if (positiveNormalizedColumn != nullptr) {
    normalized = std::move(positiveNormalizedColumn);
  } else {
    normalized = std::make_unique<cudf::column>(positiveNormalized, stream);
  }

  std::unique_ptr<cudf::column> invalidMask;
  if (policy.nullOnNegativeIndices) {
    // Spark get returns null, rather than throwing, for negative indices.
    invalidMask = std::make_unique<cudf::column>(rawLessZero->view(), stream);
  }

  if (!policy.allowOutOfBound || std::is_same_v<IndexType, int64_t>) {
    auto boundsMask = outOfBoundsMask<IndexType>(
        normalized->view(), sizes->view(), policy, stream, mr);
    if (!policy.allowOutOfBound &&
        maskHasTrue(boundsMask->view(), stream, mr)) {
      VELOX_USER_FAIL("Array subscript index out of bounds");
    }

    if (policy.allowOutOfBound) {
      // Presto element_at and Spark get return null, rather than throwing, for
      // out-of-bounds indices.
      invalidMask = mergeBoolMasks(
          std::move(invalidMask), std::move(boundsMask), stream, mr);
    }
  }

  if (invalidMask != nullptr) {
    normalized =
        applyNullMask(normalized->view(), invalidMask->view(), stream, mr);
  }

  if constexpr (std::is_same_v<IndexType, int32_t>) {
    return normalized;
  } else {
    return cudf::cast(
        normalized->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
  }
}

std::unique_ptr<cudf::column> normalizeAndValidateIndices(
    cudf::column_view const& rawIndexView,
    cudf::column_view const& sizesView,
    const ArrayAccessPolicy& policy,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  switch (rawIndexView.type().id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16: {
      auto widened = cudf::cast(
          rawIndexView, cudf::data_type{cudf::type_id::INT32}, stream, mr);
      return normalizeAndValidateIndicesTyped<int32_t>(
          widened->view(), sizesView, policy, stream, mr);
    }
    case cudf::type_id::INT32:
      return normalizeAndValidateIndicesTyped<int32_t>(
          rawIndexView, sizesView, policy, stream, mr);
    case cudf::type_id::INT64:
      return normalizeAndValidateIndicesTyped<int64_t>(
          rawIndexView, sizesView, policy, stream, mr);
    default:
      VELOX_UNSUPPORTED("Unsupported array access index type");
  }
}

void validateConstantIndex(
    int64_t originalIndex,
    cudf::size_type normalizedIndex,
    cudf::column_view const& sizesView,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto indexScalar =
      cudf::numeric_scalar<cudf::size_type>(normalizedIndex, true, stream, mr);
  auto outOfBounds = sanitizeBoolMask(
      cudf::binary_operation(
          sizesView,
          indexScalar,
          cudf::binary_operator::LESS_EQUAL,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          mr)
          ->view(),
      stream,
      mr);
  if (maskHasTrue(outOfBounds->view(), stream, mr)) {
    VELOX_USER_FAIL(
        "Array subscript index out of bounds, Index: {}", originalIndex);
  }
}

std::unique_ptr<cudf::column> makeRepeatedArrayColumn(
    const velox::VectorPtr& arrayVector,
    cudf::size_type size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // CudfFunction::eval receives columns only for non literal inputs. For a
  // constant array literal like subscript(array[1, 1, 0], groupid + 1), the
  // array side is not present in inputColumns. Convert only the single Velox
  // array value to cuDF, then repeat it on GPU to match the number of index
  // rows.
  auto literalArray =
      BaseVector::create(arrayVector->type(), 1, arrayVector->pool());
  SelectivityVector singleRow(1);
  std::vector<vector_size_t> sourceRows(1, 0);
  literalArray->copy(arrayVector.get(), singleRow, sourceRows.data());

  auto rowVector = std::make_shared<RowVector>(
      arrayVector->pool(),
      ROW({"literal_array"}, {arrayVector->type()}),
      BufferPtr(nullptr),
      1,
      std::vector<VectorPtr>{literalArray});

  auto table =
      with_arrow::toCudfTable(rowVector, arrayVector->pool(), stream, mr);
  auto columns = table->release();
  VELOX_CHECK_EQ(columns.size(), 1);

  auto repeatedTable =
      cudf::repeat(cudf::table_view{{columns[0]->view()}}, size, stream, mr);
  auto repeatedColumns = repeatedTable->release();
  VELOX_CHECK_EQ(repeatedColumns.size(), 1);
  return std::move(repeatedColumns[0]);
}

// Shared array-access machinery for ARRAY index access.
//
// Presto element_at, Presto subscript, and Spark get have the same cuDF list
// extraction shape, but differ in index origin, negative-index handling, and
// out-of-bounds behavior. ArrayAccessPolicy captures those semantic differences
// so validation and extraction can stay in one implementation.
class ArrayAccessFunction : public CudfFunction {
 public:
  ArrayAccessFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      ArrayAccessPolicy policy)
      : policy_(policy) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "array access expects exactly 2 inputs");

    // inputs()[0] may be a constant array literal, e.g.
    //   subscript({1, 1, 0}, plus(groupid, 1))
    // which arises in Q70's ROLLUP bitmask projection. Cache the Velox vector
    // so we can convert it to a cuDF lists column at eval time.
    if (auto constArrayExpr =
            std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0])) {
      constantArrayVector_ = constArrayExpr->value();
    }

    // Literal indices are not passed as input columns during evaluation. Cache
    // non-null literals here and use the scalar cuDF extraction overload below.
    // Null literal indices are still literal indices, but native Velox default
    // null behavior returns null for them instead of calling the vector
    // function, so leave constantIndex_ empty and produce an all-null result.
    if (auto indexExpr =
            std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1])) {
      indexIsLiteral_ = true;
      if (!indexExpr->value()->isNullAt(0)) {
        constantIndex_ = readConstantIntegralValue(*indexExpr);
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    // Case 1: constant array, variable index.
    //
    // Example from TPC-DS Q70:
    //   subscript({1, 1, 0}, plus(groupid, 1))
    //
    // The array is stored in constantArrayVector_, so inputColumns contains
    // only the computed index column. Materialize a repeated lists column with
    // one list per index row, then use the same index normalization and cuDF
    // extraction path as the non-constant case.
    if (constantArrayVector_ != nullptr) {
      VELOX_CHECK_EQ(
          inputColumns.size(),
          1,
          "constant-array subscript expects exactly 1 input column (the index)");
      auto rawIndexView = asView(inputColumns[0]);
      auto listsColumn = makeRepeatedArrayColumn(
          constantArrayVector_, rawIndexView.size(), stream, mr);
      auto listsView = cudf::lists_column_view(listsColumn->view());
      auto sizes = cudf::lists::count_elements(listsView, stream, mr);
      auto zeroBased = normalizeAndValidateIndices(
          rawIndexView, sizes->view(), policy_, stream, mr);
      return cudf::lists::extract_list_element(
          listsView, zeroBased->view(), stream, mr);
    }

    // The remaining cases have a non-literal array input, so inputColumns[0]
    // is always the cuDF lists column.
    VELOX_CHECK_GE(
        inputColumns.size(), 1, "array access requires a non-literal input");

    auto listsView = cudf::lists_column_view(asView(inputColumns[0]));
    // Element counts are needed to validate out-of-bounds accesses and to
    // normalize negative element_at indices before calling cuDF extraction.
    auto sizes = cudf::lists::count_elements(listsView, stream, mr);

    if (indexIsLiteral_) {
      // Case 2: variable array, literal index.
      //
      // Literal index arguments are excluded from inputColumns. The array
      // column is inputColumns[0], and constantIndex_ drives the scalar-index
      // cuDF extraction overload.
      VELOX_CHECK_EQ(
          inputColumns.size(),
          1,
          "literal array access expects exactly 1 input column");
      return extractLiteralIndex(listsView, sizes->view(), stream, mr);
    }

    // Case 3: variable array, variable index.
    //
    // Both arguments are non-literal. inputColumns[0] is the lists column and
    // inputColumns[1] is the per row raw index column.
    VELOX_CHECK_EQ(
        inputColumns.size(),
        2,
        "non-constant array access expects exactly 2 input columns");

    auto rawIndexView = asView(inputColumns[1]);
    auto zeroBased = normalizeAndValidateIndices(
        rawIndexView, sizes->view(), policy_, stream, mr);
    return cudf::lists::extract_list_element(
        listsView, zeroBased->view(), stream, mr);
  }

 private:
  ColumnOrView extractLiteralIndex(
      cudf::lists_column_view const& listsView,
      cudf::column_view const& sizesView,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const {
    if (!constantIndex_.has_value()) {
      // A null literal index returns one null result per input array row.
      return extractNullIndex(listsView, stream, mr);
    }

    auto normalizedIndex = normalizeConstantIndex(*constantIndex_, policy_);
    if (!normalizedIndex.has_value()) {
      // Spark get returns null for negative indices. Reuse the same all-null
      // result shape as a null literal index.
      return extractNullIndex(listsView, stream, mr);
    }

    if (*normalizedIndex < std::numeric_limits<cudf::size_type>::min() ||
        *normalizedIndex > std::numeric_limits<cudf::size_type>::max()) {
      if (policy_.allowOutOfBound) {
        return extractNullIndex(listsView, stream, mr);
      }
      VELOX_USER_FAIL(
          "Array subscript index out of bounds, Index: {}", *constantIndex_);
    }

    const auto scalarIndex = static_cast<cudf::size_type>(*normalizedIndex);
    if (!policy_.allowOutOfBound) {
      validateConstantIndex(
          *constantIndex_, scalarIndex, sizesView, stream, mr);
    }

    return cudf::lists::extract_list_element(
        listsView, scalarIndex, stream, mr);
  }

  const ArrayAccessPolicy policy_;
  bool indexIsLiteral_{false};
  std::optional<int64_t> constantIndex_;
  // Non null when inputs()[0] is a ConstantExpr holding an ArrayVector.
  velox::VectorPtr constantArrayVector_;
};

} // namespace

std::shared_ptr<CudfFunction> makeArrayAccessFunction(
    const std::shared_ptr<velox::exec::Expr>& expr,
    ArrayAccessPolicy policy) {
  return std::make_shared<ArrayAccessFunction>(expr, policy);
}

} // namespace facebook::velox::cudf_velox
