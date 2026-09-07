/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/index/IndexFilter.h"

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble {

using namespace velox;
using namespace velox::common;

namespace {

// Dispatch macro for integer types (TINYINT, SMALLINT, INTEGER, BIGINT).
#define INTEGER_TYPE_DISPATCH(FUNC, typeKind, ...)                    \
  [&]() {                                                             \
    switch (typeKind) {                                               \
      case TypeKind::TINYINT:                                         \
        return FUNC<TypeKind::TINYINT>(__VA_ARGS__);                  \
      case TypeKind::SMALLINT:                                        \
        return FUNC<TypeKind::SMALLINT>(__VA_ARGS__);                 \
      case TypeKind::INTEGER:                                         \
        return FUNC<TypeKind::INTEGER>(__VA_ARGS__);                  \
      case TypeKind::BIGINT:                                          \
        return FUNC<TypeKind::BIGINT>(__VA_ARGS__);                   \
      default:                                                        \
        NIMBLE_UNREACHABLE("Unsupported integer type: {}", typeKind); \
    }                                                                 \
  }()

template <TypeKind Kind>
std::pair<VectorPtr, VectorPtr> createIntegerBoundVectors(
    const TypePtr& columnType,
    int64_t lower,
    int64_t upper,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<Kind>::NativeType;
  auto lowerVector = BaseVector::create<FlatVector<T>>(columnType, 1, pool);
  auto upperVector = BaseVector::create<FlatVector<T>>(columnType, 1, pool);
  lowerVector->set(0, static_cast<T>(lower));
  upperVector->set(0, static_cast<T>(upper));
  return {std::move(lowerVector), std::move(upperVector)};
}

// Information extracted from a filter for building index bounds.
struct FilterBoundInfo {
  // Whether the filter is convertible to index bounds.
  bool convertible{false};

  // Whether this is a point lookup (lower == upper, both inclusive).
  // Point lookups allow continuing to the next index column.
  // Range scans must be the terminal column.
  bool pointLookup{false};
};

// Check if a filter is convertible and whether it's a point lookup.
// For VARCHAR types (kBytesRange, kBytesValues), descending order is not
// supported yet.
FilterBoundInfo getFilterBoundInfo(
    const Filter& filter,
    const SortOrder& sortOrder) {
  FilterBoundInfo info;
  const bool isDesc = !sortOrder.ascending;

  switch (filter.kind()) {
    case FilterKind::kBigintRange: {
      if (filter.nullAllowed()) {
        return info;
      }
      const auto& range = *filter.as<BigintRange>();
      info.convertible = true;
      info.pointLookup = range.isSingleValue();
      break;
    }

    case FilterKind::kDoubleRange: {
      if (filter.nullAllowed()) {
        return info;
      }
      const auto& range = *filter.as<DoubleRange>();
      if (range.lowerUnbounded() && range.upperUnbounded()) {
        return info; // Not convertible
      }
      info.convertible = true;
      // Point lookup: both bounds exist, equal, and both inclusive.
      info.pointLookup = !range.lowerUnbounded() && !range.upperUnbounded() &&
          range.lower() == range.upper() && !range.lowerExclusive() &&
          !range.upperExclusive();
      break;
    }

    case FilterKind::kFloatRange: {
      if (filter.nullAllowed()) {
        return info;
      }
      const auto& range = *filter.as<FloatRange>();
      if (range.lowerUnbounded() && range.upperUnbounded()) {
        return info; // Not convertible
      }
      info.convertible = true;
      info.pointLookup = !range.lowerUnbounded() && !range.upperUnbounded() &&
          range.lower() == range.upper() && !range.lowerExclusive() &&
          !range.upperExclusive();
      break;
    }

    case FilterKind::kBytesRange: {
      if (filter.nullAllowed()) {
        return info;
      }
      // Descending order not supported for VARCHAR yet.
      if (isDesc) {
        return info;
      }
      const auto& range = *filter.as<BytesRange>();
      if (range.isLowerUnbounded() && range.isUpperUnbounded()) {
        return info; // Not convertible
      }
      info.convertible = true;
      info.pointLookup = range.isSingleValue();
      break;
    }

    case FilterKind::kBytesValues: {
      if (filter.nullAllowed()) {
        return info;
      }
      // Descending order not supported for VARCHAR yet.
      if (isDesc) {
        return info;
      }
      const auto& values = filter.as<BytesValues>()->values();
      // Only support single value for point lookup.
      if (values.size() != 1) {
        return info; // Not convertible
      }
      info.convertible = true;
      info.pointLookup = true;
      break;
    }

    case FilterKind::kBoolValue: {
      if (filter.nullAllowed()) {
        return info;
      }
      info.convertible = true;
      info.pointLookup = true;
      break;
    }

    case FilterKind::kTimestampRange: {
      if (filter.nullAllowed()) {
        return info;
      }
      const auto& range = *filter.as<TimestampRange>();
      info.convertible = true;
      info.pointLookup = range.isSingleValue();
      break;
    }

    default:
      // Unsupported filter type.
      break;
  }

  return info;
}

// Set a value in a FlatVector at index 0.
template <typename T>
void setValue(FlatVector<T>* vector, T value) {
  vector->set(0, value);
}

// Create bound vectors for a single column based on filter type.
// Returns pair of (lowerVector, upperVector) and (lowerInclusive,
// upperInclusive).
// For DESC columns, lower and upper bounds are swapped to maintain correct
// range semantics in a descending index.
void addColumnBound(
    const Filter& filter,
    const std::string& columnName,
    const TypePtr& columnType,
    const SortOrder& sortOrder,
    std::vector<std::string>& columnNames,
    std::vector<TypePtr>& columnTypes,
    std::vector<VectorPtr>& lowerVectors,
    std::vector<VectorPtr>& upperVectors,
    bool& lowerInclusive,
    bool& upperInclusive,
    memory::MemoryPool* pool) {
  columnNames.push_back(columnName);
  columnTypes.push_back(columnType);

  const bool isDesc = !sortOrder.ascending;

  // Helper to add vectors considering sort order.
  // For DESC columns, we swap lower and upper bounds.
  const auto addVectors = [&](VectorPtr lower, VectorPtr upper) {
    if (isDesc) {
      lowerVectors.push_back(std::move(upper));
      upperVectors.push_back(std::move(lower));
    } else {
      lowerVectors.push_back(std::move(lower));
      upperVectors.push_back(std::move(upper));
    }
  };

  switch (filter.kind()) {
    case FilterKind::kBigintRange: {
      const auto& range = *filter.as<BigintRange>();
      auto [lowerVector, upperVector] = INTEGER_TYPE_DISPATCH(
          createIntegerBoundVectors,
          columnType->kind(),
          columnType,
          range.lower(),
          range.upper(),
          pool);
      addVectors(std::move(lowerVector), std::move(upperVector));
      break;
    }

    case FilterKind::kDoubleRange: {
      const auto& range = *filter.as<DoubleRange>();
      auto lowerVector =
          BaseVector::create<FlatVector<double>>(columnType, 1, pool);
      auto upperVector =
          BaseVector::create<FlatVector<double>>(columnType, 1, pool);
      bool localLowerInclusive = true;
      bool localUpperInclusive = true;
      if (!range.lowerUnbounded()) {
        lowerVector->set(0, range.lower());
        localLowerInclusive = !range.lowerExclusive();
      } else {
        lowerVector->setNull(0, true);
      }
      if (!range.upperUnbounded()) {
        upperVector->set(0, range.upper());
        localUpperInclusive = !range.upperExclusive();
      } else {
        upperVector->setNull(0, true);
      }
      // For DESC, swap the inclusivity as well.
      if (isDesc) {
        lowerInclusive &= localUpperInclusive;
        upperInclusive &= localLowerInclusive;
      } else {
        lowerInclusive &= localLowerInclusive;
        upperInclusive &= localUpperInclusive;
      }
      addVectors(std::move(lowerVector), std::move(upperVector));
      break;
    }

    case FilterKind::kFloatRange: {
      const auto& range = *filter.as<FloatRange>();
      auto lowerVector =
          BaseVector::create<FlatVector<float>>(columnType, 1, pool);
      auto upperVector =
          BaseVector::create<FlatVector<float>>(columnType, 1, pool);
      bool localLowerInclusive = true;
      bool localUpperInclusive = true;
      if (!range.lowerUnbounded()) {
        lowerVector->set(0, static_cast<float>(range.lower()));
        localLowerInclusive = !range.lowerExclusive();
      } else {
        lowerVector->setNull(0, true);
      }
      if (!range.upperUnbounded()) {
        upperVector->set(0, static_cast<float>(range.upper()));
        localUpperInclusive = !range.upperExclusive();
      } else {
        upperVector->setNull(0, true);
      }
      if (isDesc) {
        lowerInclusive &= localUpperInclusive;
        upperInclusive &= localLowerInclusive;
      } else {
        lowerInclusive &= localLowerInclusive;
        upperInclusive &= localUpperInclusive;
      }
      addVectors(std::move(lowerVector), std::move(upperVector));
      break;
    }

    case FilterKind::kBytesRange: {
      const auto& range = *filter.as<BytesRange>();
      auto lowerVector =
          BaseVector::create<FlatVector<StringView>>(columnType, 1, pool);
      auto upperVector =
          BaseVector::create<FlatVector<StringView>>(columnType, 1, pool);
      bool localLowerInclusive = true;
      bool localUpperInclusive = true;
      if (!range.isLowerUnbounded()) {
        lowerVector->set(0, StringView(range.lower()));
        localLowerInclusive = !range.isLowerExclusive();
      } else {
        lowerVector->setNull(0, true);
      }
      if (!range.isUpperUnbounded()) {
        upperVector->set(0, StringView(range.upper()));
        localUpperInclusive = !range.isUpperExclusive();
      } else {
        upperVector->setNull(0, true);
      }
      if (isDesc) {
        lowerInclusive &= localUpperInclusive;
        upperInclusive &= localLowerInclusive;
      } else {
        lowerInclusive &= localLowerInclusive;
        upperInclusive &= localUpperInclusive;
      }
      addVectors(std::move(lowerVector), std::move(upperVector));
      break;
    }

    case FilterKind::kBytesValues: {
      const auto& values = filter.as<BytesValues>()->values();
      const auto& value = *values.begin();
      auto lowerVector =
          BaseVector::create<FlatVector<StringView>>(columnType, 1, pool);
      auto upperVector =
          BaseVector::create<FlatVector<StringView>>(columnType, 1, pool);
      lowerVector->set(0, StringView(value));
      upperVector->set(0, StringView(value));
      // Point lookup - no need to swap for DESC.
      addVectors(std::move(lowerVector), std::move(upperVector));
      break;
    }

    case FilterKind::kBoolValue: {
      const auto& boolFilter = *filter.as<BoolValue>();
      const bool value = boolFilter.testBool(true);
      auto lowerVector =
          BaseVector::create<FlatVector<bool>>(columnType, 1, pool);
      auto upperVector =
          BaseVector::create<FlatVector<bool>>(columnType, 1, pool);
      lowerVector->set(0, value);
      upperVector->set(0, value);
      // Point lookup - no need to swap for DESC.
      addVectors(std::move(lowerVector), std::move(upperVector));
      break;
    }

    case FilterKind::kTimestampRange: {
      const auto& range = *filter.as<TimestampRange>();
      auto lowerVector =
          BaseVector::create<FlatVector<Timestamp>>(columnType, 1, pool);
      auto upperVector =
          BaseVector::create<FlatVector<Timestamp>>(columnType, 1, pool);
      lowerVector->set(0, range.lower());
      upperVector->set(0, range.upper());
      addVectors(std::move(lowerVector), std::move(upperVector));
      break;
    }

    default:
      NIMBLE_UNREACHABLE("Unsupported filter kind: {}", filter.kind());
  }
}

} // namespace

std::optional<FilterToIndexBoundsResult> convertFilterToIndexBounds(
    const std::vector<std::string>& indexColumns,
    const std::vector<SortOrder>& sortOrders,
    const RowTypePtr& rowType,
    ScanSpec& scanSpec,
    memory::MemoryPool* pool) {
  NIMBLE_CHECK(!indexColumns.empty(), "Index columns cannot be empty");
  NIMBLE_CHECK_EQ(
      indexColumns.size(),
      sortOrders.size(),
      "Index columns size must match sort orders size");

  std::vector<std::string> boundColumnNames;
  std::vector<TypePtr> boundColumnTypes;
  std::vector<VectorPtr> lowerVectors;
  std::vector<VectorPtr> upperVectors;
  bool lowerInclusive{true};
  bool upperInclusive{true};

  for (size_t i = 0; i < indexColumns.size(); ++i) {
    const auto& columnName = indexColumns[i];
    const auto& sortOrder = sortOrders[i];

    // Find the ScanSpec child for this column.
    auto* childSpec = scanSpec.childByName(columnName);
    if (childSpec == nullptr) {
      break;
    }

    const auto* filter = childSpec->filter();
    if (filter == nullptr) {
      break;
    }
    NIMBLE_CHECK(
        childSpec->hasFilter(),
        "Filter on index column '{}' is unexpectedly disabled",
        columnName);

    // Check if the filter is convertible.
    const auto boundInfo = getFilterBoundInfo(*filter, sortOrder);
    if (!boundInfo.convertible) {
      break;
    }

    // Get column type.
    const auto columnIdx = rowType->getChildIdxIfExists(columnName);
    NIMBLE_CHECK(
        columnIdx.has_value(),
        "Index column '{}' not found in row type",
        columnName);
    const auto& columnType = rowType->childAt(columnIdx.value());

    // Add bounds for this column.
    addColumnBound(
        *filter,
        columnName,
        columnType,
        sortOrder,
        boundColumnNames,
        boundColumnTypes,
        lowerVectors,
        upperVectors,
        lowerInclusive,
        upperInclusive,
        pool);

    // If this is a range scan (not point lookup), we must stop here.
    // Cannot add more columns after a range scan.
    if (!boundInfo.pointLookup) {
      break;
    }
  }

  // If no columns were added, return empty result.
  if (boundColumnNames.empty()) {
    return std::nullopt;
  }

  // Save filters before removing them from scanSpec. This allows callers to
  // restore the filters when needed (e.g., when the ScanSpec is shared across
  // multiple split readers).
  std::vector<std::pair<std::string, std::shared_ptr<common::Filter>>>
      removedFilters;
  for (const auto& columnName : boundColumnNames) {
    auto* childSpec = scanSpec.childByName(columnName);
    if (childSpec != nullptr && childSpec->filter() != nullptr) {
      removedFilters.emplace_back(columnName, childSpec->filter()->clone());
      childSpec->setFilter(nullptr);
    }
  }

  // Build the IndexBounds.
  serializer::IndexBounds bounds;
  bounds.indexColumns = std::move(boundColumnNames);

  // Create RowVectors for lower and upper bounds.
  auto boundRowType =
      ROW(std::vector<std::string>(bounds.indexColumns),
          std::move(boundColumnTypes));

  auto lowerRowVector = std::make_shared<RowVector>(
      pool, boundRowType, nullptr, 1, std::move(lowerVectors));

  auto upperRowVector = std::make_shared<RowVector>(
      pool, boundRowType, nullptr, 1, std::move(upperVectors));

  bounds.lowerBound =
      serializer::IndexBound{std::move(lowerRowVector), lowerInclusive};
  bounds.upperBound =
      serializer::IndexBound{std::move(upperRowVector), upperInclusive};

  return FilterToIndexBoundsResult{
      std::move(bounds), std::move(removedFilters)};
}

} // namespace facebook::nimble
