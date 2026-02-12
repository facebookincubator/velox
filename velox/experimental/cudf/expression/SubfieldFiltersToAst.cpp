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
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"

#include "velox/common/base/Exceptions.h"
#include "velox/type/DecimalUtil.h"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/types.hpp>

#include <limits>

namespace facebook::velox::cudf_velox {
namespace {
std::pair<int128_t, int128_t> getInt128BoundsForType(const TypePtr& type) {
  if (type->isDecimal()) {
    const auto [precision, _] = getDecimalPrecisionScale(*type);
    const auto maxAbs = DecimalUtil::kPowersOfTen[precision] - 1;
    return {-maxAbs, maxAbs};
  }
  return {
      std::numeric_limits<int128_t>::min(),
      std::numeric_limits<int128_t>::max()};
}

template <
    typename RangeT,
    typename ScalarT,
    typename = std::enable_if_t<
        std::is_base_of_v<facebook::velox::common::AbstractRange, RangeT>>>
const cudf::ast::expression& createRangeExpr(
    const facebook::velox::common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const cudf::ast::expression& columnRef,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  auto* range = dynamic_cast<const RangeT*>(&filter);
  VELOX_CHECK_NOT_NULL(range, "Filter is not the expected range type");

  const bool lowerUnbounded = range->lowerUnbounded();
  const bool upperUnbounded = range->upperUnbounded();

  const cudf::ast::expression* lowerExpr = nullptr;
  const cudf::ast::expression* upperExpr = nullptr;

  auto addLiteral = [&](auto value) -> const cudf::ast::expression& {
    scalars.emplace_back(std::make_unique<ScalarT>(value, true, stream, mr));
    return tree.push(
        cudf::ast::literal{*static_cast<ScalarT*>(scalars.back().get())});
  };

  // If RangeT is BytesValues and it's a single value, return a simple equality
  // expression. This is an early return for the single-value IN-list filter on
  // bytes.
  if constexpr (std::is_same_v<RangeT, facebook::velox::common::BytesRange>) {
    if (range->isSingleValue()) {
      // Only one value in the IN-list, so just compare for equality.
      auto singleValue = range->lower();
      const auto& literal = addLiteral(singleValue);
      return tree.push(Operation{Op::EQUAL, columnRef, literal});
    }
  }

  if (!lowerUnbounded) {
    auto lowerValue = range->lower();
    const auto& lowerLiteral = addLiteral(lowerValue);

    auto lowerOp = range->lowerExclusive() ? Op::GREATER : Op::GREATER_EQUAL;
    lowerExpr = &tree.push(Operation{lowerOp, columnRef, lowerLiteral});
  }

  if (!upperUnbounded) {
    auto upperValue = range->upper();
    const auto& upperLiteral = addLiteral(upperValue);

    auto upperOp = range->upperExclusive() ? Op::LESS : Op::LESS_EQUAL;
    upperExpr = &tree.push(Operation{upperOp, columnRef, upperLiteral});
  }

  if (lowerExpr && upperExpr) {
    return tree.push(Operation{Op::NULL_LOGICAL_AND, *lowerExpr, *upperExpr});
  } else if (lowerExpr) {
    return *lowerExpr;
  } else if (upperExpr) {
    return *upperExpr;
  }

  // Both bounds unbounded => Pass-through filter (everything).
  return tree.push(Operation{Op::EQUAL, columnRef, columnRef});
}

template <TypeKind Kind>
std::reference_wrapper<const cudf::ast::expression> buildBigintRangeExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const cudf::ast::expression& columnRef,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    const TypePtr& columnTypePtr) {
  using NativeT = typename TypeTraits<Kind>::NativeType;

  if constexpr (std::is_integral_v<NativeT>) {
    using Op = cudf::ast::ast_operator;
    using Operation = cudf::ast::operation;

    auto* bigintRange = static_cast<const common::BigintRange*>(&filter);

    const auto lower = bigintRange->lower();
    const auto upper = bigintRange->upper();

    const bool skipLowerBound =
        lower <= static_cast<int64_t>(std::numeric_limits<NativeT>::min());
    const bool skipUpperBound =
        upper >= static_cast<int64_t>(std::numeric_limits<NativeT>::max());

    auto addLiteral = [&](int64_t value) -> const cudf::ast::expression& {
      variant veloxVariant = static_cast<NativeT>(value);
      const auto& literal =
          makeScalarAndLiteral<Kind>(columnTypePtr, veloxVariant, scalars);
      return tree.push(literal);
    };

    if (bigintRange->isSingleValue()) {
      // Equal comparison: column = value. This value is the same as the
      // lower/upper bound.
      if (skipLowerBound || skipUpperBound) {
        // If the singular value of this filter lies outside the range of the
        // column's NativeT type then we want to be always false
        return tree.push(Operation{Op::NOT_EQUAL, columnRef, columnRef});
      } else {
        auto const& literal = addLiteral(lower);
        return tree.push(Operation{Op::EQUAL, columnRef, literal});
      }
    } else {
      // Range comparison: column >= lower AND column <= upper

      const cudf::ast::expression* lowerExpr = nullptr;
      if (!skipLowerBound) {
        auto const& lowerLiteral = addLiteral(lower);
        lowerExpr =
            &tree.push(Operation{Op::GREATER_EQUAL, columnRef, lowerLiteral});
      }

      const cudf::ast::expression* upperExpr = nullptr;
      if (!skipUpperBound) {
        auto const& upperLiteral = addLiteral(upper);
        upperExpr =
            &tree.push(Operation{Op::LESS_EQUAL, columnRef, upperLiteral});
      }

      if (lowerExpr && upperExpr) {
        auto const& result =
            tree.push(Operation{Op::NULL_LOGICAL_AND, *lowerExpr, *upperExpr});
        return result;
      } else if (lowerExpr) {
        return *lowerExpr;
      } else if (upperExpr) {
        return *upperExpr;
      }

      // If neither lower nor upper bound expressions were created, it means
      // the filter covers the entire range of the type, so it's a no-op
      return tree.push(Operation{Op::EQUAL, columnRef, columnRef});
    }
  } else {
    VELOX_FAIL("Unsupported type for buildBigintRangeExpr: {}", Kind);
  }
}

std::reference_wrapper<const cudf::ast::expression> buildHugeintRangeExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const cudf::ast::expression& columnRef,
    const TypePtr& columnTypePtr) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  auto* hugeintRange = static_cast<const common::HugeintRange*>(&filter);
  const auto lower = hugeintRange->lower();
  const auto upper = hugeintRange->upper();

  const auto [minVal, maxVal] = getInt128BoundsForType(columnTypePtr);
  const bool skipLowerBound = lower <= minVal;
  const bool skipUpperBound = upper >= maxVal;

  auto addLiteral = [&](int128_t value) -> const cudf::ast::expression& {
    variant veloxVariant = value;
    const auto& literal = makeScalarAndLiteral<TypeKind::HUGEINT>(
        columnTypePtr, veloxVariant, scalars);
    return tree.push(literal);
  };

  if (lower == upper) {
    if (skipLowerBound || skipUpperBound) {
      return tree.push(Operation{Op::NOT_EQUAL, columnRef, columnRef});
    }
    auto const& literal = addLiteral(lower);
    return tree.push(Operation{Op::EQUAL, columnRef, literal});
  }

  const cudf::ast::expression* lowerExpr = nullptr;
  if (!skipLowerBound) {
    auto const& lowerLiteral = addLiteral(lower);
    lowerExpr = &tree.push(Operation{Op::GREATER_EQUAL, columnRef, lowerLiteral});
  }

  const cudf::ast::expression* upperExpr = nullptr;
  if (!skipUpperBound) {
    auto const& upperLiteral = addLiteral(upper);
    upperExpr = &tree.push(Operation{Op::LESS_EQUAL, columnRef, upperLiteral});
  }

  if (lowerExpr && upperExpr) {
    return tree.push(Operation{Op::NULL_LOGICAL_AND, *lowerExpr, *upperExpr});
  } else if (lowerExpr) {
    return *lowerExpr;
  } else if (upperExpr) {
    return *upperExpr;
  }

  // No bounds => pass-through filter.
  return tree.push(Operation{Op::EQUAL, columnRef, columnRef});
}

template <TypeKind Kind, typename FilterT, typename ValueT>
const cudf::ast::expression& buildHashInListExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    const cudf::ast::expression& columnRef,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const TypePtr& columnTypePtr,
    bool isNegated = false) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  auto* valuesFilter = dynamic_cast<const FilterT*>(&filter);
  VELOX_CHECK_NOT_NULL(valuesFilter, "Filter is not a hash-table list filter");
  auto const& values = valuesFilter->values();
  VELOX_CHECK(!values.empty(), "Empty List filter not supported");

  std::vector<const cudf::ast::expression*> exprVec;
  for (const auto& value : values) {
    variant veloxVariant = static_cast<ValueT>(value);
    auto const& literal = tree.push(makeScalarAndLiteral<Kind>(
        columnTypePtr, veloxVariant, scalars));
    auto const& equalExpr = tree.push(
        Operation{isNegated ? Op::NOT_EQUAL : Op::EQUAL, columnRef, literal});
    exprVec.push_back(&equalExpr);
  }

  const cudf::ast::expression* result = exprVec[0];
  for (size_t i = 1; i < exprVec.size(); ++i) {
    result = &tree.push(Operation{
        isNegated ? Op::NULL_LOGICAL_AND : Op::NULL_LOGICAL_OR,
        *result,
        *exprVec[i]});
  }

  return *result;
}

template <typename T>
auto createFloatingPointRangeExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const cudf::ast::expression& columnRef,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) -> const cudf::ast::expression& {
  return createRangeExpr<
      facebook::velox::common::FloatingPointRange<T>,
      cudf::numeric_scalar<T>>(filter, tree, scalars, columnRef, stream, mr);
};

const cudf::ast::expression& createBytesRangeExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const cudf::ast::expression& columnRef,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return createRangeExpr<
      facebook::velox::common::BytesRange,
      cudf::string_scalar>(filter, tree, scalars, columnRef, stream, mr);
}

template <typename FilterT, typename ScalarT>
const cudf::ast::expression& buildInListExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    const cudf::ast::expression& columnRef,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    bool isNegated,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  auto* valuesFilter = dynamic_cast<const FilterT*>(&filter);
  VELOX_CHECK_NOT_NULL(valuesFilter, "Filter is not a List filter");
  auto const& values = valuesFilter->values();
  if (values.empty()) {
    VELOX_FAIL("Empty List filter not supported");
  }

  std::vector<const cudf::ast::expression*> exprVec;
  for (const auto& value : values) {
    scalars.emplace_back(std::make_unique<ScalarT>(value, true, stream, mr));
    auto const& literal = tree.push(
        cudf::ast::literal{*static_cast<ScalarT*>(scalars.back().get())});
    auto const& equalExpr = tree.push(
        Operation{isNegated ? Op::NOT_EQUAL : Op::EQUAL, columnRef, literal});
    exprVec.push_back(&equalExpr);
  }

  const cudf::ast::expression* result = exprVec[0];
  for (size_t i = 1; i < exprVec.size(); ++i) {
    if (isNegated) {
      result =
          &tree.push(Operation{Op::NULL_LOGICAL_AND, *result, *exprVec[i]});
    } else {
      result = &tree.push(Operation{Op::NULL_LOGICAL_OR, *result, *exprVec[i]});
    }
  }
  return *result;
}

// Build an IN-list expression for integer columns where the filter values are
// provided as int64_t but the column may be any integral type. Values outside
// the target type's range are ignored. If all values are out of range, this
// returns a constant false expression (col != col).
template <TypeKind Kind>
std::reference_wrapper<const cudf::ast::expression> buildIntegerInListExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const cudf::ast::expression& columnRef,
    rmm::cuda_stream_view /*stream*/,
    rmm::device_async_resource_ref /*mr*/,
    const TypePtr& columnTypePtr) {
  using NativeT = typename TypeTraits<Kind>::NativeType;

  if constexpr (std::is_integral_v<NativeT>) {
    using Op = cudf::ast::ast_operator;
    using Operation = cudf::ast::operation;

    auto* valuesFilter =
        static_cast<const common::BigintValuesUsingBitmask*>(&filter);
    const auto& values = valuesFilter->values();

    std::vector<const cudf::ast::expression*> exprVec;
    exprVec.reserve(values.size());

    for (const int64_t value : values) {
      if (value < static_cast<int64_t>(std::numeric_limits<NativeT>::min()) ||
          value > static_cast<int64_t>(std::numeric_limits<NativeT>::max())) {
        // Skip values that cannot be represented in the column type.
        continue;
      }

      variant veloxVariant = static_cast<NativeT>(value);
      const auto& literal =
          makeScalarAndLiteral<Kind>(columnTypePtr, veloxVariant, scalars);
      auto const& cudfLiteral = tree.push(literal);
      auto const& equalExpr =
          tree.push(Operation{Op::EQUAL, columnRef, cudfLiteral});
      exprVec.push_back(&equalExpr);
    }

    if (exprVec.empty()) {
      // No representable values -> always false
      auto const& alwaysFalse =
          tree.push(Operation{Op::NOT_EQUAL, columnRef, columnRef});
      return std::ref(alwaysFalse);
    }

    const cudf::ast::expression* result = exprVec[0];
    for (size_t i = 1; i < exprVec.size(); ++i) {
      result = &tree.push(Operation{Op::NULL_LOGICAL_OR, *result, *exprVec[i]});
    }
    return std::ref(*result);
  } else {
    VELOX_FAIL("Unsupported type for buildIntegerInListExpr: {}", Kind);
  }
}

} // namespace

// Convert subfield filters to cudf AST
cudf::ast::expression const& createAstFromSubfieldFilter(
    const common::Subfield& subfield,
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema) {
  // First, create column reference from subfield
  // For now, only support simple field references
  if (subfield.path().empty() ||
      subfield.path()[0]->kind() != common::SubfieldKind::kNestedField) {
    VELOX_FAIL(
        "Only simple field references are supported in subfield filters");
  }

  auto nestedField = static_cast<const common::Subfield::NestedField*>(
      subfield.path()[0].get());
  const std::string& fieldName = nestedField->name();

  if (!inputRowSchema->containsChild(fieldName)) {
    VELOX_FAIL("Field '{}' not found in input schema", fieldName);
  }

  auto columnIndex = inputRowSchema->getChildIdx(fieldName);
  auto const& columnRef = tree.push(cudf::ast::column_reference(columnIndex));

  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  switch (filter.kind()) {
    case common::FilterKind::kBigintRange: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      auto result = VELOX_DYNAMIC_TYPE_DISPATCH(
          buildBigintRangeExpr,
          columnType->kind(),
          filter,
          tree,
          scalars,
          columnRef,
          stream,
          mr,
          columnType);
      return result.get();
    }

    case common::FilterKind::kHugeintRange: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      auto const& expr = buildHugeintRangeExpr(
          filter, tree, scalars, columnRef, columnType);
      return expr.get();
    }

    case common::FilterKind::kBigintValuesUsingHashTable: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      return buildHashInListExpr<
          TypeKind::BIGINT,
          common::BigintValuesUsingHashTable,
          int64_t>(filter, tree, columnRef, scalars, columnType);
    }

    case common::FilterKind::kBigintValuesUsingBitmask: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      // Dispatch by the column's integer kind and cast filter values to it.
      auto result = VELOX_DYNAMIC_TYPE_DISPATCH(
          buildIntegerInListExpr,
          columnType->kind(),
          filter,
          tree,
          scalars,
          columnRef,
          stream,
          mr,
          columnType);
      return result.get();
    }

    case common::FilterKind::kHugeintValuesUsingHashTable: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      return buildHashInListExpr<
          TypeKind::HUGEINT,
          common::HugeintValuesUsingHashTable,
          int128_t>(filter, tree, columnRef, scalars, columnType);
    }

    case common::FilterKind::kBytesValues: {
      return buildInListExpr<common::BytesValues, cudf::string_scalar>(
          filter, tree, columnRef, scalars, false, stream, mr);
    }

    case common::FilterKind::kNegatedBytesValues: {
      return buildInListExpr<common::NegatedBytesValues, cudf::string_scalar>(
          filter, tree, columnRef, scalars, true, stream, mr);
    }

    case common::FilterKind::kDoubleRange: {
      return createFloatingPointRangeExpr<double>(
          filter, tree, scalars, columnRef, stream, mr);
    }

    case common::FilterKind::kFloatRange: {
      return createFloatingPointRangeExpr<float>(
          filter, tree, scalars, columnRef, stream, mr);
    }

    case common::FilterKind::kBytesRange: {
      return createBytesRangeExpr(filter, tree, scalars, columnRef, stream, mr);
    }

    case common::FilterKind::kBoolValue: {
      auto* boolValue = static_cast<const common::BoolValue*>(&filter);
      auto matchesTrue = boolValue->testBool(true);
      scalars.emplace_back(
          std::make_unique<cudf::numeric_scalar<bool>>(
              matchesTrue, true, stream, mr));
      auto const& matchesBoolExpr = tree.push(
          cudf::ast::literal{
              *static_cast<cudf::numeric_scalar<bool>*>(scalars.back().get())});
      return tree.push(Operation{Op::EQUAL, columnRef, matchesBoolExpr});
    }

    case common::FilterKind::kIsNull: {
      return tree.push(Operation{Op::IS_NULL, columnRef});
    }

    case common::FilterKind::kIsNotNull: {
      // For IsNotNull, we can use NOT(IS_NULL)
      auto const& nullCheck = tree.push(Operation{Op::IS_NULL, columnRef});
      return tree.push(Operation{Op::NOT, nullCheck});
    }

    default:
      VELOX_NYI(
          "Filter type {} not yet supported for subfield filter conversion",
          static_cast<int>(filter.kind()));
  }
}

// Create a combined AST from a set of subfield filters by chaining them with
// logical ANDs. The returned expression is owned by the provided 'tree'.
cudf::ast::expression const& createAstFromSubfieldFilters(
    const common::SubfieldFilters& subfieldFilters,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  std::vector<const cudf::ast::expression*> exprRefs;

  // Build individual filter expressions.
  for (const auto& [subfield, filterPtr] : subfieldFilters) {
    if (!filterPtr) {
      continue;
    }
    auto const& expr = createAstFromSubfieldFilter(
        subfield, *filterPtr, tree, scalars, inputRowSchema);
    exprRefs.push_back(&expr);
  }

  VELOX_CHECK_GT(exprRefs.size(), 0, "No subfield filters provided");

  if (exprRefs.size() == 1) {
    return *exprRefs[0];
  }

  // Combine expressions with NULL_LOGICAL_AND.
  const cudf::ast::expression* result = exprRefs[0];
  for (size_t i = 1; i < exprRefs.size(); ++i) {
    auto const& andExpr =
        tree.push(Operation{Op::NULL_LOGICAL_AND, *result, *exprRefs[i]});
    result = &andExpr;
  }

  return *result;
}
} // namespace facebook::velox::cudf_velox
