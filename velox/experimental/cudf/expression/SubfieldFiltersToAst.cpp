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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"

#include "velox/common/base/Exceptions.h"
#include "velox/type/DecimalUtil.h"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/types.hpp>

#include <algorithm>
#include <limits>

namespace facebook::velox::cudf_velox {
namespace {
std::optional<SubfieldFilterDecimalType> subfieldDecimalType(
    const TypePtr& type,
    const std::string& fieldName,
    const SubfieldFilterDecimalTypes* decimalTypes) {
  if (!type->isDecimal() || decimalTypes == nullptr) {
    return std::nullopt;
  }

  if (const auto it = decimalTypes->find(fieldName);
      it != decimalTypes->end()) {
    VELOX_CHECK(
        it->second.type == cudf::type_id::INT32 ||
            it->second.type == cudf::type_id::INT64 ||
            it->second.type == cudf::type_id::DECIMAL32 ||
            it->second.type == cudf::type_id::DECIMAL64 ||
            it->second.type == cudf::type_id::DECIMAL128,
        "Invalid cuDF storage type for decimal field '{}'",
        fieldName);
    return it->second;
  }
  return std::nullopt;
}

struct RescaledDecimal {
  int128_t floor;
  int128_t ceil;
  bool exact;
  bool overflow;
};

RescaledDecimal
rescaleDecimal(int128_t value, int32_t fromScale, int32_t toScale) {
  if (fromScale == toScale) {
    return {value, value, true, false};
  }
  const auto scaleDifference = std::abs(fromScale - toScale);
  VELOX_CHECK_LE(scaleDifference, LongDecimalType::kMaxPrecision);
  const auto factor = DecimalUtil::kPowersOfTen[scaleDifference];
  if (fromScale < toScale) {
    int128_t scaled;
    if (__builtin_mul_overflow(value, factor, &scaled)) {
      return {value, value, true, true};
    }
    return {scaled, scaled, true, false};
  }

  const auto quotient = value / factor;
  const auto remainder = value % factor;
  if (remainder == 0) {
    return {quotient, quotient, true, false};
  }
  return {
      quotient - (value < 0 ? 1 : 0),
      quotient + (value > 0 ? 1 : 0),
      false,
      false};
}

RescaledDecimal rescaleDecimal(
    int128_t value,
    const TypePtr& type,
    std::optional<SubfieldFilterDecimalType> decimalType) {
  if (!decimalType) {
    return {value, value, true, false};
  }
  const auto [_, logicalScale] = getDecimalPrecisionScale(*type);
  return rescaleDecimal(value, logicalScale, decimalType->scale);
}

std::pair<int128_t, int128_t> getInt128BoundsForType(
    const TypePtr& type,
    std::optional<SubfieldFilterDecimalType> decimalType = std::nullopt) {
  if (type->isDecimal()) {
    const auto [precision, logicalScale] = getDecimalPrecisionScale(*type);
    const auto logicalMax = DecimalUtil::kPowersOfTen[precision] - 1;
    int128_t min = -logicalMax;
    int128_t max = logicalMax;
    if (decimalType) {
      const auto fileMax =
          rescaleDecimal(logicalMax, logicalScale, decimalType->scale);
      if (fileMax.overflow) {
        min = std::numeric_limits<int128_t>::min();
        max = std::numeric_limits<int128_t>::max();
      } else {
        min = -fileMax.floor;
        max = fileMax.floor;
      }
      if (decimalType->type == cudf::type_id::INT32 ||
          decimalType->type == cudf::type_id::DECIMAL32) {
        min = std::max<int128_t>(min, std::numeric_limits<int32_t>::min());
        max = std::min<int128_t>(max, std::numeric_limits<int32_t>::max());
      } else if (
          decimalType->type == cudf::type_id::INT64 ||
          decimalType->type == cudf::type_id::DECIMAL64) {
        min = std::max<int128_t>(min, std::numeric_limits<int64_t>::min());
        max = std::min<int128_t>(max, std::numeric_limits<int64_t>::max());
      }
    }
    return {min, max};
  }
  return {
      std::numeric_limits<int128_t>::min(),
      std::numeric_limits<int128_t>::max()};
}

bool decimalValueIsRepresentable(
    int128_t value,
    const TypePtr& type,
    std::optional<SubfieldFilterDecimalType> decimalType) {
  const auto [min, max] = getInt128BoundsForType(type, decimalType);
  return value >= min && value <= max;
}

const cudf::ast::expression& buildEqualityExpr(
    cudf::ast::tree& tree,
    const cudf::ast::expression& columnRef,
    const cudf::ast::expression& literal,
    bool isDecimal) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  if (!isDecimal) {
    return tree.push(Operation{Op::EQUAL, columnRef, literal});
  }

  // cuDF's Parquet Bloom-filter path cannot probe fixed-point literals. Keep
  // the equivalent row and statistics predicate while avoiding that optional
  // optimization.
  auto const& lower =
      tree.push(Operation{Op::GREATER_EQUAL, columnRef, literal});
  auto const& upper = tree.push(Operation{Op::LESS_EQUAL, columnRef, literal});
  return tree.push(Operation{Op::NULL_LOGICAL_AND, lower, upper});
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
    cuda::stream_ref stream,
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
    stream.sync();
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

template <TypeKind Kind, typename FilterT>
std::reference_wrapper<const cudf::ast::expression> buildIntegerRangeExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const cudf::ast::expression& columnRef,
    const TypePtr& columnTypePtr,
    std::optional<SubfieldFilterDecimalType> decimalType) {
  using NativeT = typename TypeTraits<Kind>::NativeType;

  if constexpr (
      (std::is_same_v<FilterT, common::BigintRange> &&
       std::is_integral_v<NativeT>) ||
      std::is_same_v<FilterT, common::HugeintRange>) {
    using Op = cudf::ast::ast_operator;
    using Operation = cudf::ast::operation;

    auto* rangeFilter = static_cast<const FilterT*>(&filter);
    const auto lower = static_cast<int128_t>(rangeFilter->lower());
    const auto upper = static_cast<int128_t>(rangeFilter->upper());
    const auto rescaledLower =
        rescaleDecimal(lower, columnTypePtr, decimalType);
    const auto rescaledUpper =
        rescaleDecimal(upper, columnTypePtr, decimalType);
    const auto [minBound, maxBound] = [&]() {
      if (columnTypePtr->isDecimal()) {
        return getInt128BoundsForType(columnTypePtr, decimalType);
      } else if constexpr (std::is_same_v<FilterT, common::HugeintRange>) {
        return getInt128BoundsForType(columnTypePtr);
      } else {
        return std::pair<int128_t, int128_t>{
            std::numeric_limits<NativeT>::min(),
            std::numeric_limits<NativeT>::max()};
      }
    }();

    const bool lowerAboveMax =
        rescaledLower.overflow ? lower > 0 : rescaledLower.ceil > maxBound;
    const bool upperBelowMin =
        rescaledUpper.overflow ? upper < 0 : rescaledUpper.floor < minBound;
    if (lowerAboveMax || upperBelowMin ||
        (!rescaledLower.overflow && !rescaledUpper.overflow &&
         rescaledLower.ceil > rescaledUpper.floor)) {
      return tree.push(Operation{Op::NOT_EQUAL, columnRef, columnRef});
    }

    const bool skipLowerBound =
        rescaledLower.overflow || rescaledLower.ceil <= minBound;
    const bool skipUpperBound =
        rescaledUpper.overflow || rescaledUpper.floor >= maxBound;

    auto addLiteral = [&](int128_t value) -> const cudf::ast::expression& {
      variant veloxVariant = static_cast<NativeT>(value);
      const auto& literal = makeScalarAndLiteral<Kind>(
          columnTypePtr,
          veloxVariant,
          scalars,
          decimalType ? std::optional{decimalType->type} : std::nullopt,
          decimalType ? std::optional{decimalType->scale} : std::nullopt);
      return tree.push(literal);
    };

    if (lower == upper) {
      if (!rescaledLower.exact || rescaledLower.overflow ||
          rescaledLower.floor < minBound || rescaledLower.floor > maxBound) {
        return tree.push(Operation{Op::NOT_EQUAL, columnRef, columnRef});
      }
      auto const& literal = addLiteral(rescaledLower.floor);
      return buildEqualityExpr(
          tree,
          columnRef,
          literal,
          columnTypePtr->isDecimal() &&
              (!decimalType || decimalType->isDecimal));
    }

    // Range comparison: column >= lower AND column <= upper.
    const cudf::ast::expression* lowerExpr = nullptr;
    if (!skipLowerBound) {
      auto const& lowerLiteral = addLiteral(rescaledLower.ceil);
      lowerExpr =
          &tree.push(Operation{Op::GREATER_EQUAL, columnRef, lowerLiteral});
    }
    const cudf::ast::expression* upperExpr = nullptr;
    if (!skipUpperBound) {
      auto const& upperLiteral = addLiteral(rescaledUpper.floor);
      upperExpr =
          &tree.push(Operation{Op::LESS_EQUAL, columnRef, upperLiteral});
    }
    if (lowerExpr && upperExpr) {
      return tree.push(Operation{Op::NULL_LOGICAL_AND, *lowerExpr, *upperExpr});
    } else if (lowerExpr) {
      return *lowerExpr;
    } else if (upperExpr) {
      return *upperExpr;
    }

    // If neither lower nor upper bound expressions were created, it means
    // the filter covers the entire range of the type, so it's a no-op.
    return tree.push(Operation{Op::EQUAL, columnRef, columnRef});
  } else {
    VELOX_FAIL("Unsupported type for buildRangeExpr: {}", Kind);
  }
}

template <TypeKind Kind>
std::reference_wrapper<const cudf::ast::expression> buildBigintRangeExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const cudf::ast::expression& columnRef,
    const TypePtr& columnTypePtr,
    std::optional<SubfieldFilterDecimalType> decimalType) {
  return buildIntegerRangeExpr<Kind, common::BigintRange>(
      filter, tree, scalars, columnRef, columnTypePtr, decimalType);
}

std::reference_wrapper<const cudf::ast::expression> buildHugeintRangeExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const cudf::ast::expression& columnRef,
    const TypePtr& columnTypePtr,
    std::optional<SubfieldFilterDecimalType> decimalType) {
  return buildIntegerRangeExpr<TypeKind::HUGEINT, common::HugeintRange>(
      filter, tree, scalars, columnRef, columnTypePtr, decimalType);
}

template <TypeKind Kind, typename FilterT, typename ValueT>
const cudf::ast::expression& buildValuesListExpr(
    const common::Filter& filter,
    cudf::ast::tree& tree,
    const cudf::ast::expression& columnRef,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const TypePtr& columnTypePtr,
    std::optional<SubfieldFilterDecimalType> decimalType,
    bool isNegated = false) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  auto* valuesFilter = dynamic_cast<const FilterT*>(&filter);
  VELOX_CHECK_NOT_NULL(valuesFilter, "Filter is not a values list filter");
  auto const& values = valuesFilter->values();
  VELOX_CHECK(!values.empty(), "Empty List filter not supported");

  std::vector<const cudf::ast::expression*> exprVec;
  for (const auto& value : values) {
    auto convertedValue = value;
    if constexpr (!std::is_same_v<ValueT, StringView>) {
      const auto rescaled = rescaleDecimal(
          static_cast<int128_t>(value), columnTypePtr, decimalType);
      if (!rescaled.exact || rescaled.overflow ||
          !decimalValueIsRepresentable(
              rescaled.floor, columnTypePtr, decimalType)) {
        continue;
      }
      convertedValue = static_cast<ValueT>(rescaled.floor);
    }
    variant veloxVariant = static_cast<ValueT>(convertedValue);
    auto const& literal = tree.push(
        makeScalarAndLiteral<Kind>(
            columnTypePtr,
            veloxVariant,
            scalars,
            decimalType ? std::optional{decimalType->type} : std::nullopt,
            decimalType ? std::optional{decimalType->scale} : std::nullopt));
    if (isNegated) {
      auto const& notEqualExpr =
          tree.push(Operation{Op::NOT_EQUAL, columnRef, literal});
      exprVec.push_back(&notEqualExpr);
    } else {
      exprVec.push_back(&buildEqualityExpr(
          tree,
          columnRef,
          literal,
          columnTypePtr->isDecimal() &&
              (!decimalType || decimalType->isDecimal)));
    }
  }

  if (exprVec.empty()) {
    return tree.push(Operation{Op::NOT_EQUAL, columnRef, columnRef});
  }

  const cudf::ast::expression* result = exprVec[0];
  for (size_t i = 1; i < exprVec.size(); ++i) {
    result = &tree.push(
        Operation{
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
    cuda::stream_ref stream,
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
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return createRangeExpr<
      facebook::velox::common::BytesRange,
      cudf::string_scalar>(filter, tree, scalars, columnRef, stream, mr);
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
    cuda::stream_ref /*stream*/,
    rmm::device_async_resource_ref /*mr*/,
    const TypePtr& columnTypePtr,
    std::optional<SubfieldFilterDecimalType> decimalType) {
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
      const auto rescaled = rescaleDecimal(value, columnTypePtr, decimalType);
      if (!rescaled.exact || rescaled.overflow ||
          !decimalValueIsRepresentable(
              rescaled.floor, columnTypePtr, decimalType)) {
        continue;
      }

      variant veloxVariant = static_cast<NativeT>(rescaled.floor);
      const auto& literal = makeScalarAndLiteral<Kind>(
          columnTypePtr,
          veloxVariant,
          scalars,
          decimalType ? std::optional{decimalType->type} : std::nullopt,
          decimalType ? std::optional{decimalType->scale} : std::nullopt);
      auto const& cudfLiteral = tree.push(literal);
      exprVec.push_back(&buildEqualityExpr(
          tree,
          columnRef,
          cudfLiteral,
          columnTypePtr->isDecimal() &&
              (!decimalType || decimalType->isDecimal)));
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

// MultiRange child null policies are ignored. Flatten nested MultiRanges and
// omit IsNull leaves before building the non-null value predicate.
void collectValueFilters(
    const common::Filter& filter,
    std::vector<const common::Filter*>& filters) {
  if (filter.kind() == common::FilterKind::kIsNull) {
    return;
  }
  if (filter.kind() != common::FilterKind::kMultiRange) {
    filters.push_back(&filter);
    return;
  }
  const auto& children =
      static_cast<const common::MultiRange&>(filter).filters();
  VELOX_CHECK(!children.empty(), "MultiRange filter must not be empty");
  for (const auto& child : children) {
    collectValueFilters(*child, filters);
  }
}

cudf::ast::expression const& createAlwaysFalseExpr(
    const cudf::ast::column_reference& columnRef,
    cudf::ast::tree& tree) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  auto const& isNull = tree.push(Operation{Op::IS_NULL, columnRef});
  auto const& isNotNull = tree.push(Operation{Op::NOT, isNull});
  return tree.push(Operation{Op::NULL_LOGICAL_AND, isNull, isNotNull});
}

cudf::ast::expression const& createAstFromSubfieldFilterImpl(
    const common::Subfield& subfield,
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    const SubfieldFilterDecimalTypes* decimalTypes);

cudf::ast::expression const& createAstFromFiltersOr(
    const common::Subfield& subfield,
    const std::vector<const common::Filter*>& filters,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    const SubfieldFilterDecimalTypes* decimalTypes) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  VELOX_CHECK(!filters.empty(), "Filters must not be empty");

  std::vector<const cudf::ast::expression*> exprRefs;
  exprRefs.reserve(filters.size());
  for (const auto* subFilter : filters) {
    auto const& subExpr = createAstFromSubfieldFilterImpl(
        subfield, *subFilter, tree, scalars, inputRowSchema, decimalTypes);
    exprRefs.push_back(&subExpr);
  }

  const cudf::ast::expression* result = exprRefs[0];
  for (size_t i = 1; i < exprRefs.size(); ++i) {
    result = &tree.push(Operation{Op::NULL_LOGICAL_OR, *result, *exprRefs[i]});
  }
  return *result;
}

cudf::ast::expression const& createAstFromSubfieldFilterImpl(
    const common::Subfield& subfield,
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    const SubfieldFilterDecimalTypes* decimalTypes) {
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

  auto stream = getDefaultStreamForCurrentThread();
  auto mr = get_temp_mr();

  switch (filter.kind()) {
    case common::FilterKind::kBigintRange: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      const auto decimalType =
          subfieldDecimalType(columnType, fieldName, decimalTypes);
      auto result = VELOX_DYNAMIC_TYPE_DISPATCH(
          buildBigintRangeExpr,
          columnType->kind(),
          filter,
          tree,
          scalars,
          columnRef,
          columnType,
          decimalType);
      return result.get();
    }

    case common::FilterKind::kHugeintRange: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      const auto decimalType =
          subfieldDecimalType(columnType, fieldName, decimalTypes);
      auto const& expr = buildHugeintRangeExpr(
          filter, tree, scalars, columnRef, columnType, decimalType);
      return expr.get();
    }

    case common::FilterKind::kBigintValuesUsingHashTable: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      const auto decimalType =
          subfieldDecimalType(columnType, fieldName, decimalTypes);
      return buildValuesListExpr<
          TypeKind::BIGINT,
          common::BigintValuesUsingHashTable,
          int64_t>(filter, tree, columnRef, scalars, columnType, decimalType);
    }

    case common::FilterKind::kBigintValuesUsingBitmask: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      const auto decimalType =
          subfieldDecimalType(columnType, fieldName, decimalTypes);
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
          columnType,
          decimalType);
      return result.get();
    }

    case common::FilterKind::kHugeintValuesUsingHashTable: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      const auto decimalType =
          subfieldDecimalType(columnType, fieldName, decimalTypes);
      return buildValuesListExpr<
          TypeKind::HUGEINT,
          common::HugeintValuesUsingHashTable,
          int128_t>(filter, tree, columnRef, scalars, columnType, decimalType);
    }

    case common::FilterKind::kBytesValues: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      return buildValuesListExpr<
          TypeKind::VARCHAR,
          common::BytesValues,
          StringView>(
          filter, tree, columnRef, scalars, columnType, std::nullopt);
    }

    case common::FilterKind::kNegatedBytesValues: {
      auto const& columnType = inputRowSchema->childAt(columnIndex);
      return buildValuesListExpr<
          TypeKind::VARCHAR,
          common::NegatedBytesValues,
          StringView>(
          filter, tree, columnRef, scalars, columnType, std::nullopt, true);
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
      stream.sync();
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

    case common::FilterKind::kBigintMultiRange: {
      auto* multiRange = static_cast<const common::BigintMultiRange*>(&filter);
      std::vector<const common::Filter*> ranges;
      ranges.reserve(multiRange->ranges().size());
      for (const auto& range : multiRange->ranges()) {
        ranges.push_back(range.get());
      }
      return createAstFromFiltersOr(
          subfield, ranges, tree, scalars, inputRowSchema, decimalTypes);
    }

    case common::FilterKind::kMultiRange: {
      std::vector<const common::Filter*> valueFilters;
      collectValueFilters(filter, valueFilters);
      if (valueFilters.empty()) {
        return createAlwaysFalseExpr(columnRef, tree);
      }
      return createAstFromFiltersOr(
          subfield, valueFilters, tree, scalars, inputRowSchema, decimalTypes);
    }

    case common::FilterKind::kNegatedBigintRange: {
      auto* negRange = static_cast<const common::NegatedBigintRange*>(&filter);
      const auto rejectedLower = negRange->lower();
      const auto rejectedUpper = negRange->upper();

      auto const& columnType = inputRowSchema->childAt(columnIndex);
      const auto decimalType =
          subfieldDecimalType(columnType, fieldName, decimalTypes);

      // Build the inner range: column >= lower AND column <= upper.
      // Then negate it: NOT(column >= lower AND column <= upper).
      // This expresses "column is outside [lower, upper]".
      common::BigintRange innerRange(
          rejectedLower, rejectedUpper, !filter.testNull());
      auto innerResult = VELOX_DYNAMIC_TYPE_DISPATCH(
          buildBigintRangeExpr,
          columnType->kind(),
          innerRange,
          tree,
          scalars,
          columnRef,
          columnType,
          decimalType);
      return tree.push(Operation{Op::NOT, innerResult.get()});
    }

    default:
      VELOX_NYI(
          "Filter type {} not yet supported for subfield filter conversion",
          static_cast<int>(filter.kind()));
  }
}

} // namespace

bool hasDecimalSubfieldFilter(
    const common::SubfieldFilters& subfieldFilters,
    const RowTypePtr& inputRowSchema) {
  for (const auto& [subfield, _] : subfieldFilters) {
    if (subfield.path().empty() ||
        subfield.path()[0]->kind() != common::SubfieldKind::kNestedField) {
      continue;
    }
    const auto* field = static_cast<const common::Subfield::NestedField*>(
        subfield.path()[0].get());
    if (inputRowSchema->containsChild(field->name()) &&
        inputRowSchema->findChild(field->name())->isDecimal()) {
      return true;
    }
  }
  return false;
}

// Convert subfield filters to cudf AST
cudf::ast::expression const& createAstFromSubfieldFilter(
    const common::Subfield& subfield,
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    const SubfieldFilterDecimalTypes* decimalTypes) {
  auto const& expression = createAstFromSubfieldFilterImpl(
      subfield, filter, tree, scalars, inputRowSchema, decimalTypes);
  if (filter.testNull() && filter.kind() != common::FilterKind::kIsNull &&
      filter.kind() != common::FilterKind::kIsNotNull) {
    using Op = cudf::ast::ast_operator;
    using Operation = cudf::ast::operation;

    auto const& columnRef = tree.push(
        cudf::ast::column_reference(
            inputRowSchema->getChildIdx(subfield.toString())));
    auto const& isNull = tree.push(Operation{Op::IS_NULL, columnRef});
    return tree.push(Operation{Op::NULL_LOGICAL_OR, isNull, expression});
  }
  return expression;
}

// Create a combined AST from a set of subfield filters by chaining them with
// logical ANDs. The returned expression is owned by the provided 'tree'.
cudf::ast::expression const& createAstFromSubfieldFilters(
    const common::SubfieldFilters& subfieldFilters,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    const SubfieldFilterDecimalTypes* decimalTypes) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;

  std::vector<const cudf::ast::expression*> exprRefs;

  // Build individual filter expressions.
  for (const auto& [subfield, filterPtr] : subfieldFilters) {
    if (!filterPtr) {
      continue;
    }
    auto const& expr = createAstFromSubfieldFilter(
        subfield, *filterPtr, tree, scalars, inputRowSchema, decimalTypes);
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
