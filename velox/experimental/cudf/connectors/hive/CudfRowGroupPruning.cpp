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

#include "velox/experimental/cudf/connectors/hive/CudfRowGroupPruning.h"

#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/DecimalArithmetic.h"
#include "velox/type/HugeInt.h"
#include "velox/type/Subfield.h"
#include "velox/type/Timestamp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

namespace pq = cudf::io::parquet;
using StatBytes = std::optional<std::vector<uint8_t>>;

// The deprecated min/max fields order values as the signed physical type, so
// they stand in for a missing min_value/max_value only where that is the value
// order: signed integers, floating points, booleans and timestamps. Strings
// and byte-array decimals never use them; old writers filled them in unsigned
// byte order.
const StatBytes& minBytes(const pq::Statistics& stats, bool allowDeprecated) {
  return stats.min_value.has_value() || !allowDeprecated ? stats.min_value
                                                         : stats.min;
}

const StatBytes& maxBytes(const pq::Statistics& stats, bool allowDeprecated) {
  return stats.max_value.has_value() || !allowDeprecated ? stats.max_value
                                                         : stats.max;
}

bool hasStatistics(const pq::Statistics& stats) {
  return stats.min_value.has_value() || stats.max_value.has_value() ||
      stats.min.has_value() || stats.max.has_value() ||
      stats.null_count.has_value();
}

bool hasOnlyDeprecatedBounds(const pq::Statistics& stats) {
  return !stats.min_value.has_value() && !stats.max_value.has_value() &&
      (stats.min.has_value() || stats.max.has_value());
}

template <typename T>
std::optional<T> load(const StatBytes& bytes) {
  if (!bytes.has_value() || bytes->size() != sizeof(T)) {
    return std::nullopt;
  }
  T value;
  std::memcpy(&value, bytes->data(), sizeof(T));
  return value;
}

// INT32 and INT64 statistics, widened to int64.
std::optional<int64_t> loadInteger(const StatBytes& bytes) {
  if (const auto value = load<int64_t>(bytes)) {
    return value;
  }
  if (const auto value = load<int32_t>(bytes)) {
    return static_cast<int64_t>(*value);
  }
  return std::nullopt;
}

// Decimal statistics hold the unscaled value as INT32, INT64 or a big-endian
// two's complement byte array of up to 16 bytes.
std::optional<int128_t> loadDecimal(
    const StatBytes& bytes,
    pq::Type physicalType) {
  switch (physicalType) {
    case pq::Type::INT32:
    case pq::Type::INT64: {
      const auto value = loadInteger(bytes);
      return value ? std::optional<int128_t>(*value) : std::nullopt;
    }
    case pq::Type::BYTE_ARRAY:
    case pq::Type::FIXED_LEN_BYTE_ARRAY: {
      if (!bytes.has_value() || bytes->empty() ||
          bytes->size() > sizeof(int128_t)) {
        return std::nullopt;
      }
      uint128_t value = 0;
      for (const auto byte : *bytes) {
        value = (value << 8) | byte;
      }
      // Sign-extend from the top bit of the first byte.
      const auto shift =
          static_cast<int>(8 * (sizeof(int128_t) - bytes->size()));
      return static_cast<int128_t>(value << shift) >> shift;
    }
    default:
      return std::nullopt;
  }
}

std::optional<std::string> toString(const StatBytes& bytes) {
  if (!bytes.has_value()) {
    return std::nullopt;
  }
  return std::string(
      reinterpret_cast<const char*>(bytes->data()), bytes->size());
}

bool isUnsignedInteger(const pq::SchemaElement& element) {
  if (element.logical_type.has_value() &&
      element.logical_type->type == pq::LogicalType::INTEGER &&
      element.logical_type->int_type.has_value()) {
    return !element.logical_type->int_type->isSigned;
  }
  return element.converted_type == pq::ConvertedType::UINT_8 ||
      element.converted_type == pq::ConvertedType::UINT_16 ||
      element.converted_type == pq::ConvertedType::UINT_32 ||
      element.converted_type == pq::ConvertedType::UINT_64;
}

// Whether the column is a decimal stored at 'scale'. 'decimal_scale' is the
// footer's scale field, which cuDF reads the values by as well.
bool isDecimalAtScale(const pq::SchemaElement& element, int32_t scale) {
  const bool decimal =
      (element.logical_type.has_value() &&
       element.logical_type->type == pq::LogicalType::DECIMAL) ||
      element.converted_type == pq::ConvertedType::DECIMAL;
  return decimal && element.decimal_scale == scale;
}

// Converts an INT64 statistic by the column's timestamp unit; nullopt when the
// column declares none.
std::optional<Timestamp> toTimestamp(
    std::optional<int64_t> value,
    const pq::SchemaElement& element) {
  if (!value.has_value()) {
    return std::nullopt;
  }
  if (element.logical_type.has_value() &&
      element.logical_type->type == pq::LogicalType::TIMESTAMP &&
      element.logical_type->timestamp_type.has_value()) {
    switch (element.logical_type->timestamp_type->unit.type) {
      case pq::TimeUnit::MILLIS:
        return Timestamp::fromMillis(*value);
      case pq::TimeUnit::MICROS:
        return Timestamp::fromMicros(*value);
      case pq::TimeUnit::NANOS:
        return Timestamp::fromNanos(*value);
      default:
        return std::nullopt;
    }
  }
  if (element.converted_type == pq::ConvertedType::TIMESTAMP_MILLIS) {
    return Timestamp::fromMillis(*value);
  }
  if (element.converted_type == pq::ConvertedType::TIMESTAMP_MICROS) {
    return Timestamp::fromMicros(*value);
  }
  return std::nullopt;
}

// Builds the column statistics common::testFilter() tests for this chunk, as
// the CPU Parquet reader builds them (buildColumnStatisticsFromThrift in
// velox/dwio/parquet/reader/Metadata.cpp), except that they report
// 'totalRows' values and no null: nullsDecide() settles the nulls before the
// bounds are tested. Returns nullptr for a column stored in a form the test
// does not cover; decimals, floating points and booleans have their own tests.
std::unique_ptr<dwio::common::ColumnStatistics> buildStatistics(
    const pq::SchemaElement& element,
    const pq::ColumnChunkMetaData& chunk,
    const Type& type,
    uint64_t totalRows) {
  const auto& stats = chunk.statistics;
  const std::optional<uint64_t> valueCount = totalRows;
  const std::optional<bool> hasNull = false;
  switch (type.kind()) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      if ((chunk.type != pq::Type::INT32 && chunk.type != pq::Type::INT64) ||
          isUnsignedInteger(element)) {
        return nullptr;
      }
      return std::make_unique<dwio::common::IntegerColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          loadInteger(minBytes(stats, true)),
          loadInteger(maxBytes(stats, true)),
          std::nullopt);
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      if (chunk.type != pq::Type::BYTE_ARRAY &&
          chunk.type != pq::Type::FIXED_LEN_BYTE_ARRAY) {
        return nullptr;
      }
      return std::make_unique<dwio::common::StringColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          toString(minBytes(stats, false)),
          toString(maxBytes(stats, false)),
          std::nullopt);
    case TypeKind::TIMESTAMP: {
      if (chunk.type != pq::Type::INT64) {
        return nullptr;
      }
      // An INT64 column without a timestamp unit cannot be read as one.
      if (!toTimestamp(int64_t{0}, element).has_value()) {
        return nullptr;
      }
      return std::make_unique<dwio::common::TimestampColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          toTimestamp(loadInteger(minBytes(stats, true)), element),
          toTimestamp(loadInteger(maxBytes(stats, true)), element));
    }
    default:
      return nullptr;
  }
}

// The null handling of common::testFilter(): the decision when the null count
// settles it, else nullopt. Nulls are then absent or fail the filter, so the
// bounds are tested without a null flag; a range test would hand one to the
// children of an OR, whose own null flags cuDF's AST ignores.
std::optional<bool> nullsDecide(
    const common::Filter& filter,
    std::optional<uint64_t> valueCount,
    uint64_t totalRows) {
  bool mayHaveNull = true;
  if (valueCount.has_value()) {
    if (*valueCount == 0) {
      return filter.testNull();
    }
    mayHaveNull = *valueCount < totalRows;
  }
  if (!mayHaveNull && filter.kind() == common::FilterKind::kIsNull) {
    return false;
  }
  if (mayHaveNull && filter.testNull()) {
    return true;
  }
  return std::nullopt;
}

// Bounds of a decimal chunk. nullopt for a byte-array chunk with only the
// deprecated fields, which cuDF's pass decodes and the host leaves to it.
struct DecimalBounds {
  std::optional<int128_t> min;
  std::optional<int128_t> max;
};

std::optional<DecimalBounds> decimalBounds(
    const pq::ColumnChunkMetaData& chunk) {
  const bool integerStorage =
      chunk.type == pq::Type::INT32 || chunk.type == pq::Type::INT64;
  if (!integerStorage && hasOnlyDeprecatedBounds(chunk.statistics)) {
    return std::nullopt;
  }
  return DecimalBounds{
      loadDecimal(minBytes(chunk.statistics, integerStorage), chunk.type),
      loadDecimal(maxBytes(chunk.statistics, integerStorage), chunk.type)};
}

// The bounds of a short decimal chunk as int64, or nullopt when a bound lies
// outside the short decimal range.
std::optional<std::pair<std::optional<int64_t>, std::optional<int64_t>>>
shortDecimalBounds(const DecimalBounds& bounds) {
  if ((bounds.min.has_value() &&
       *bounds.min < DecimalArithmetic::kShortDecimalMin) ||
      (bounds.max.has_value() &&
       *bounds.max > DecimalArithmetic::kShortDecimalMax)) {
    return std::nullopt;
  }
  const auto narrow = [](const std::optional<int128_t>& value) {
    return value ? std::optional<int64_t>(static_cast<int64_t>(*value))
                 : std::nullopt;
  };
  return std::pair{narrow(bounds.min), narrow(bounds.max)};
}

// The sorted values of an IN list. BigintValuesUsingBitmask and BytesValues
// answer a range test from the span of their values alone, so a list such as
// (2, 8) keeps a row group of 4..6, and HugeintValuesUsingHashTable has no
// range test; cuDF's statistics pass tests each value, and so does
// testInListFilter() with these.
std::optional<InListValues> inListValues(const common::Filter& filter) {
  InListValues values;
  switch (filter.kind()) {
    case common::FilterKind::kBigintValuesUsingBitmask:
      values.int64s =
          static_cast<const common::BigintValuesUsingBitmask&>(filter).values();
      std::ranges::sort(values.int64s);
      return values;
    case common::FilterKind::kHugeintValuesUsingHashTable: {
      const auto& set =
          static_cast<const common::HugeintValuesUsingHashTable&>(filter)
              .values();
      values.int128s.assign(set.begin(), set.end());
      std::ranges::sort(values.int128s);
      return values;
    }
    case common::FilterKind::kBytesValues: {
      const auto& set =
          static_cast<const common::BytesValues&>(filter).values();
      values.strings.assign(set.begin(), set.end());
      std::ranges::sort(values.strings);
      return values;
    }
    default:
      return std::nullopt;
  }
}

// Whether a value of 'sorted' lies in [min, max]; a missing bound is open.
template <typename T>
std::optional<bool> anyValueInRange(
    const std::vector<T>& sorted,
    const std::optional<T>& min,
    const std::optional<T>& max) {
  // A list of another type than the column's is left to cuDF.
  if (sorted.empty()) {
    return std::nullopt;
  }
  const auto it =
      min.has_value() ? std::ranges::lower_bound(sorted, *min) : sorted.begin();
  return it != sorted.end() && (!max.has_value() || *it <= *max);
}

// Filters on BIGINT columns, which a short decimal column carries too, all
// implement testInt64Range().
bool hasInt64RangeTest(common::FilterKind kind) {
  switch (kind) {
    case common::FilterKind::kAlwaysFalse:
    case common::FilterKind::kBigintRange:
    case common::FilterKind::kBigintValuesUsingHashTable:
    case common::FilterKind::kNegatedBigintRange:
    case common::FilterKind::kNegatedBigintValuesUsingHashTable:
    case common::FilterKind::kNegatedBigintValuesUsingBitmask:
    case common::FilterKind::kBigintMultiRange:
    case common::FilterKind::kBigintValuesUsingBloomFilter:
      return true;
    default:
      return false;
  }
}

// Whether 'kind' implements the range test common::testFilter() calls for a
// column of 'typeKind'. Another filter, such as an OR of ranges on an integer
// column, throws there, while cuDF's AST takes it.
bool hasRangeTest(common::FilterKind kind, TypeKind typeKind) {
  switch (typeKind) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      return hasInt64RangeTest(kind);
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return kind == common::FilterKind::kAlwaysFalse ||
          kind == common::FilterKind::kBytesRange ||
          kind == common::FilterKind::kNegatedBytesRange ||
          kind == common::FilterKind::kNegatedBytesValues ||
          kind == common::FilterKind::kMultiRange;
    case TypeKind::TIMESTAMP:
      return kind == common::FilterKind::kAlwaysFalse ||
          kind == common::FilterKind::kTimestampRange ||
          kind == common::FilterKind::kMultiRange;
    default:
      return false;
  }
}

// Of the filters a long decimal column carries, only the range implements
// testInt128Range().
bool hasInt128RangeTest(common::FilterKind kind) {
  return kind == common::FilterKind::kAlwaysFalse ||
      kind == common::FilterKind::kHugeintRange;
}

using ColumnFilter = HostRowGroupPruningFilters::Column;

// Whether an OR of filters holds an IN list, which the OR's range test answers
// from the span of the list's values while cuDF's pass tests each value.
bool hasInListInOr(const common::Filter& filter) {
  if (filter.kind() != common::FilterKind::kMultiRange) {
    return false;
  }
  for (const auto& child :
       static_cast<const common::MultiRange&>(filter).filters()) {
    if (child->kind() == common::FilterKind::kBytesValues ||
        hasInListInOr(*child)) {
      return true;
    }
  }
  return false;
}

// Bounds of a REAL or DOUBLE chunk; a float widens exactly.
struct FloatingPointBounds {
  std::optional<double> min;
  std::optional<double> max;
};

// Returns nullopt when the column is stored as another physical type or a
// bound is a NaN, which some writers emit and cuDF's pass compares like any
// other value.
std::optional<FloatingPointBounds> floatingPointBounds(
    const pq::ColumnChunkMetaData& chunk,
    TypeKind kind) {
  const auto& stats = chunk.statistics;
  FloatingPointBounds bounds;
  if (kind == TypeKind::REAL) {
    if (chunk.type != pq::Type::FLOAT) {
      return std::nullopt;
    }
    const auto widen = [](std::optional<float> value) {
      return value ? std::optional<double>(*value) : std::nullopt;
    };
    bounds = {
        widen(load<float>(minBytes(stats, true))),
        widen(load<float>(maxBytes(stats, true)))};
  } else {
    if (chunk.type != pq::Type::DOUBLE) {
      return std::nullopt;
    }
    bounds = {
        load<double>(minBytes(stats, true)),
        load<double>(maxBytes(stats, true))};
  }
  if ((bounds.min.has_value() && std::isnan(*bounds.min)) ||
      (bounds.max.has_value() && std::isnan(*bounds.max))) {
    return std::nullopt;
  }
  return bounds;
}

// Whether a value of the chunk can pass 'range', as cuDF's pass tests it:
// col > v as vmax > v, col >= v as vmax >= v, col < v as vmin < v and
// col <= v as vmin <= v. FloatingPointRange::testDoubleRange() tests the
// inclusive bounds only.
template <typename T>
bool rangeMayPass(
    const common::FloatingPointRange<T>& range,
    const FloatingPointBounds& bounds) {
  if (!range.lowerUnbounded() && bounds.max.has_value()) {
    const auto lower = range.lower();
    if (*bounds.max < lower ||
        (range.lowerExclusive() && *bounds.max == lower)) {
      return false;
    }
  }
  if (!range.upperUnbounded() && bounds.min.has_value()) {
    const auto upper = range.upper();
    if (*bounds.min > upper ||
        (range.upperExclusive() && *bounds.min == upper)) {
      return false;
    }
  }
  return true;
}

// Whether a value of the chunk can pass a REAL or DOUBLE filter: a range, or
// an OR of ranges whose IS NULL leaves pass no value, as cuDF's AST holds them.
// nullopt for another filter kind.
std::optional<bool> floatingPointMayPass(
    const common::Filter& filter,
    const FloatingPointBounds& bounds) {
  switch (filter.kind()) {
    case common::FilterKind::kDoubleRange:
      return rangeMayPass(
          static_cast<const common::DoubleRange&>(filter), bounds);
    case common::FilterKind::kFloatRange:
      return rangeMayPass(
          static_cast<const common::FloatRange&>(filter), bounds);
    case common::FilterKind::kMultiRange:
      for (const auto& child :
           static_cast<const common::MultiRange&>(filter).filters()) {
        if (child->kind() == common::FilterKind::kIsNull) {
          continue;
        }
        const auto childMayPass = floatingPointMayPass(*child, bounds);
        if (!childMayPass.has_value()) {
          return std::nullopt;
        }
        if (*childMayPass) {
          return true;
        }
      }
      return false;
    default:
      return std::nullopt;
  }
}

// Tests a REAL or DOUBLE filter against the chunk's bounds, in place of
// common::testFilter(), whose range test ignores exclusive bounds. Returns
// nullopt when the column is stored as another physical type, a bound is a
// NaN or the filter is not a range or an OR of ranges.
std::optional<bool> testFloatingPointFilter(
    const ColumnFilter& columnFilter,
    const pq::ColumnChunkMetaData& chunk,
    std::optional<uint64_t> valueCount,
    uint64_t totalRows) {
  const auto& filter = *columnFilter.filter;
  if (const auto decision = nullsDecide(filter, valueCount, totalRows)) {
    return decision;
  }
  const auto bounds = floatingPointBounds(chunk, columnFilter.type->kind());
  if (!bounds.has_value()) {
    return std::nullopt;
  }
  return floatingPointMayPass(filter, *bounds);
}

// Tests a BOOLEAN filter against the chunk's bounds: a row group whose maximum
// is false holds no true value, and one whose minimum is true no false value.
// common::testFilter() needs the true and false counts, which take a null
// count the footer may lack, so this repeats its null handling and compares
// the bounds as cuDF's pass does. Returns nullopt when the column is stored as
// another physical type or the filter is not a value test.
std::optional<bool> testBooleanFilter(
    const ColumnFilter& columnFilter,
    const pq::ColumnChunkMetaData& chunk,
    std::optional<uint64_t> valueCount,
    uint64_t totalRows) {
  const auto& filter = *columnFilter.filter;
  if (chunk.type != pq::Type::BOOLEAN ||
      filter.kind() != common::FilterKind::kBoolValue) {
    return std::nullopt;
  }
  if (const auto decision = nullsDecide(filter, valueCount, totalRows)) {
    return decision;
  }
  const auto min = load<uint8_t>(minBytes(chunk.statistics, true));
  const auto max = load<uint8_t>(maxBytes(chunk.statistics, true));
  if (max.has_value() && *max == 0) {
    return filter.testBool(false);
  }
  if (min.has_value() && *min != 0) {
    return filter.testBool(true);
  }
  return true;
}

// Tests a decimal filter against the chunk's bounds. common::testFilter() has
// no value test for decimals because the CPU reader's metadata does not decode
// their bounds, so this repeats its null handling and compares the unscaled
// values. Returns nullopt when the file stores the column at another scale,
// the bounds cannot be decoded or the filter has no range test.
std::optional<bool> testDecimalFilter(
    const ColumnFilter& columnFilter,
    const pq::SchemaElement& element,
    const pq::ColumnChunkMetaData& chunk,
    std::optional<uint64_t> valueCount,
    uint64_t totalRows) {
  const auto& filter = *columnFilter.filter;
  const auto& type = *columnFilter.type;
  const auto [precision, scale] = getDecimalPrecisionScale(type);
  if (!isDecimalAtScale(element, scale)) {
    return std::nullopt;
  }
  const bool shortDecimal = type.isShortDecimal();
  if (shortDecimal ? !hasInt64RangeTest(filter.kind())
                   : !hasInt128RangeTest(filter.kind())) {
    return std::nullopt;
  }

  if (const auto decision = nullsDecide(filter, valueCount, totalRows)) {
    return decision;
  }
  const auto bounds = decimalBounds(chunk);
  if (!bounds.has_value()) {
    return std::nullopt;
  }
  if (!bounds->min.has_value() && !bounds->max.has_value()) {
    return true;
  }
  if (shortDecimal) {
    const auto narrowed = shortDecimalBounds(*bounds);
    if (!narrowed.has_value()) {
      return std::nullopt;
    }
    return filter.testInt64Range(
        narrowed->first.value_or(std::numeric_limits<int64_t>::min()),
        narrowed->second.value_or(std::numeric_limits<int64_t>::max()),
        false);
  }
  return filter.testInt128Range(
      bounds->min.value_or(DecimalArithmetic::kLongDecimalMin),
      bounds->max.value_or(DecimalArithmetic::kLongDecimalMax),
      false);
}

// Tests an IN list value by value against the chunk's bounds. Returns nullopt
// for a column stored in a form the test does not cover.
std::optional<bool> testInListFilter(
    const ColumnFilter& columnFilter,
    const pq::SchemaElement& element,
    const pq::ColumnChunkMetaData& chunk,
    std::optional<uint64_t> valueCount,
    uint64_t totalRows) {
  const auto& filter = *columnFilter.filter;
  const auto& type = *columnFilter.type;
  const auto& values = *columnFilter.inList;
  const auto& stats = chunk.statistics;
  if (const auto decision = nullsDecide(filter, valueCount, totalRows)) {
    return decision;
  }
  if (type.isDecimal()) {
    const auto [precision, scale] = getDecimalPrecisionScale(type);
    if (!isDecimalAtScale(element, scale)) {
      return std::nullopt;
    }
    const auto bounds = decimalBounds(chunk);
    if (!bounds.has_value()) {
      return std::nullopt;
    }
    if (!type.isShortDecimal()) {
      return anyValueInRange(values.int128s, bounds->min, bounds->max);
    }
    const auto narrowed = shortDecimalBounds(*bounds);
    if (!narrowed.has_value()) {
      return std::nullopt;
    }
    return anyValueInRange(values.int64s, narrowed->first, narrowed->second);
  }
  switch (type.kind()) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      if ((chunk.type != pq::Type::INT32 && chunk.type != pq::Type::INT64) ||
          isUnsignedInteger(element)) {
        return std::nullopt;
      }
      return anyValueInRange(
          values.int64s,
          loadInteger(minBytes(stats, true)),
          loadInteger(maxBytes(stats, true)));
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      if (chunk.type != pq::Type::BYTE_ARRAY &&
          chunk.type != pq::Type::FIXED_LEN_BYTE_ARRAY) {
        return std::nullopt;
      }
      return anyValueInRange(
          values.strings,
          toString(minBytes(stats, false)),
          toString(maxBytes(stats, false)));
    default:
      return std::nullopt;
  }
}

// Whether the row group may hold a row passing 'columnFilter', or nullopt
// when neither the column's storage nor the filter is one the host test covers.
std::optional<bool> testRowGroup(
    const ColumnFilter& columnFilter,
    const pq::SchemaElement& element,
    const pq::ColumnChunkMetaData& chunk,
    int64_t numRows) {
  const auto& stats = chunk.statistics;
  std::optional<uint64_t> nullCount;
  if (stats.null_count.has_value() && *stats.null_count >= 0) {
    nullCount = static_cast<uint64_t>(*stats.null_count);
  }
  const auto totalRows = static_cast<uint64_t>(std::max<int64_t>(numRows, 0));
  std::optional<uint64_t> valueCount;
  if (nullCount.has_value() && *nullCount <= totalRows) {
    valueCount = totalRows - *nullCount;
  }
  std::optional<bool> hasNull;
  if (nullCount.has_value()) {
    hasNull = *nullCount > 0;
  }

  const auto* filter = columnFilter.filter;
  const auto kind = filter->kind();
  if (kind == common::FilterKind::kIsNull ||
      kind == common::FilterKind::kIsNotNull) {
    // Decided by the null count, whatever the column's type.
    dwio::common::ColumnStatistics nullStats(
        valueCount, hasNull, std::nullopt, std::nullopt);
    return common::testFilter(filter, &nullStats, totalRows, columnFilter.type);
  }
  // A chunk with only one of its bounds, which no common writer emits, is
  // left to cuDF's pass, whatever the type: common::testFilter() fills the
  // missing side with a limit of its own, which does not always decide as
  // cuDF's unknown bound does.
  const bool allowDeprecated = chunk.type != pq::Type::BYTE_ARRAY &&
      chunk.type != pq::Type::FIXED_LEN_BYTE_ARRAY;
  if (minBytes(stats, allowDeprecated).has_value() !=
      maxBytes(stats, allowDeprecated).has_value()) {
    return std::nullopt;
  }
  if (columnFilter.inList.has_value()) {
    return testInListFilter(
        columnFilter, element, chunk, valueCount, totalRows);
  }
  if (columnFilter.type->isDecimal()) {
    return testDecimalFilter(
        columnFilter, element, chunk, valueCount, totalRows);
  }
  switch (columnFilter.type->kind()) {
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
      return testFloatingPointFilter(
          columnFilter, chunk, valueCount, totalRows);
    case TypeKind::BOOLEAN:
      return testBooleanFilter(columnFilter, chunk, valueCount, totalRows);
    default:
      break;
  }
  if (!hasRangeTest(kind, columnFilter.type->kind()) ||
      hasInListInOr(*filter)) {
    return std::nullopt;
  }
  const auto columnStats =
      buildStatistics(element, chunk, *columnFilter.type, totalRows);
  if (columnStats == nullptr) {
    return std::nullopt;
  }
  if (const auto decision = nullsDecide(*filter, valueCount, totalRows)) {
    return decision;
  }
  // testFilter() has no VARBINARY case; its string test compares bytes, which
  // is the VARBINARY order too.
  static const TypePtr kVarchar = VARCHAR();
  return common::testFilter(
      filter,
      columnStats.get(),
      totalRows,
      columnFilter.type->kind() == TypeKind::VARBINARY ? kVarchar
                                                       : columnFilter.type);
}

// Where a filtered column sits in one file: its schema element and the index
// of its chunk within every row group, or -1 when the file does not hold it as
// a top-level primitive column.
struct FileColumn {
  int schemaIdx{-1};
  int chunkIdx{-1};
};

std::string fileColumnName(const std::string& name, bool caseInsensitive) {
  return caseInsensitive
      ? ::facebook::velox::functions::stringImpl::utf8StrToLowerCopy(name)
      : name;
}

FileColumn findFileColumn(
    const pq::FileMetaData& file,
    const std::string& name,
    bool caseInsensitive) {
  FileColumn column;
  for (const auto childIdx : file.schema.front().children_idx) {
    const auto& element = file.schema[childIdx];
    // A repeated leaf is a one-level list, whose null count counts its empty
    // lists; neither reader prunes on it.
    if (element.num_children == 0 &&
        element.repetition_type != pq::FieldRepetitionType::REPEATED &&
        fileColumnName(element.name, caseInsensitive) == name) {
      column.schemaIdx = childIdx;
      break;
    }
  }
  if (column.schemaIdx < 0 || file.row_groups.empty()) {
    return column;
  }
  // Chunks come in the same order in every row group.
  const auto& columns = file.row_groups.front().columns;
  const auto it = std::ranges::find_if(columns, [&](const auto& chunk) {
    const auto& path = chunk.meta_data.path_in_schema;
    return path.size() == 1 && fileColumnName(path[0], caseInsensitive) == name;
  });
  if (it != columns.end()) {
    column.chunkIdx = static_cast<int>(it - columns.begin());
  }
  return column;
}

} // namespace

HostRowGroupPruningFilters prepareRowGroupPruningFilters(
    const common::SubfieldFilters& filters,
    const RowType& tableType) {
  HostRowGroupPruningFilters prepared;
  for (const auto& [subfield, filter] : filters) {
    if (filter == nullptr ||
        filter->kind() == common::FilterKind::kAlwaysTrue) {
      continue;
    }
    const auto& path = subfield.path();
    const auto* field = path.size() == 1 &&
            path[0]->kind() == common::SubfieldKind::kNestedField
        ? static_cast<const common::Subfield::NestedField*>(path[0].get())
        : nullptr;
    const auto childIdx = field != nullptr
        ? tableType.getChildIdxIfExists(field->name())
        : std::nullopt;
    if (!childIdx.has_value()) {
      // Not a top-level column of the table: left to cuDF.
      prepared.complete = false;
      continue;
    }
    const auto& type = tableType.childAt(*childIdx);
    // Statistics describe primitive columns; neither reader prunes on a
    // nested one.
    if (!type->isPrimitiveType()) {
      continue;
    }
    prepared.columns.push_back(
        {field->name(), type, filter.get(), inListValues(*filter)});
  }
  return prepared;
}

HostRowGroupPruning pruneRowGroupsWithStatistics(
    const std::vector<pq::FileMetaData>& files,
    const std::vector<std::vector<cudf::size_type>>& rowGroupIndices,
    const HostRowGroupPruningFilters& filters,
    bool caseInsensitiveColumnNames) {
  VELOX_CHECK_EQ(
      files.size(),
      rowGroupIndices.size(),
      "Row group indices must come one list per file");
  HostRowGroupPruning result;
  result.rowGroupIndices = rowGroupIndices;
  result.complete = filters.complete;
  const auto& columnFilters = filters.columns;
  if (columnFilters.empty()) {
    return result;
  }

  std::vector<FileColumn> fileColumns(columnFilters.size());
  for (size_t src = 0; src < files.size(); ++src) {
    const auto& file = files[src];
    VELOX_CHECK(!file.schema.empty(), "Parquet footer schema is empty");
    for (size_t i = 0; i < columnFilters.size(); ++i) {
      fileColumns[i] = findFileColumn(
          file, columnFilters[i].name, caseInsensitiveColumnNames);
    }

    std::vector<cudf::size_type> kept;
    kept.reserve(rowGroupIndices[src].size());
    for (const auto rowGroupIdx : rowGroupIndices[src]) {
      VELOX_CHECK(
          rowGroupIdx >= 0 &&
              static_cast<size_t>(rowGroupIdx) < file.row_groups.size(),
          "Row group index {} out of range",
          rowGroupIdx);
      const auto& rowGroup = file.row_groups[rowGroupIdx];
      // An empty row group holds no row to pass.
      bool keep = rowGroup.num_rows > 0;
      for (size_t i = 0; i < columnFilters.size() && keep; ++i) {
        const auto& fileColumn = fileColumns[i];
        // A column the file lacks has no statistics to test; neither path
        // prunes on it.
        if (fileColumn.chunkIdx < 0 ||
            static_cast<size_t>(fileColumn.chunkIdx) >=
                rowGroup.columns.size()) {
          continue;
        }
        const auto& chunk = rowGroup.columns[fileColumn.chunkIdx].meta_data;
        if (chunk.path_in_schema !=
                file.row_groups.front()
                    .columns[fileColumn.chunkIdx]
                    .meta_data.path_in_schema ||
            !hasStatistics(chunk.statistics)) {
          continue;
        }
        const auto decision = testRowGroup(
            columnFilters[i],
            file.schema[fileColumn.schemaIdx],
            chunk,
            rowGroup.num_rows);
        if (!decision.has_value()) {
          result.complete = false;
          continue;
        }
        keep = *decision;
      }
      if (keep) {
        kept.push_back(rowGroupIdx);
      }
    }
    result.rowGroupIndices[src] = std::move(kept);
  }
  return result;
}

} // namespace facebook::velox::cudf_velox::connector::hive
