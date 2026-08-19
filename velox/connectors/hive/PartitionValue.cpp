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

#include "velox/connectors/hive/PartitionValue.h"

#include <string>
#include <type_traits>

#include <folly/Conv.h>

#include "velox/common/base/Exceptions.h"
#include "velox/type/Conversions.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::connector::hive {
namespace {

template <typename T>
std::string formatDecimal(T value, const Type& type) {
  const auto [precision, scale] = getDecimalPrecisionScale(type);
  const auto maxSize = DecimalUtil::maxStringViewSize(precision, scale);
  std::string buffer(maxSize, '\0');
  buffer.resize(
      DecimalUtil::castToString(value, scale, maxSize, buffer.data()));
  return buffer;
}

std::string timestampToString(Timestamp value) {
  static const TimestampToStringOptions kOptions{
      .precision = TimestampPrecision::kMilliseconds,
      .skipTrailingZeros = true,
      .dateTimeSeparator = ' ',
  };
  auto result = value.toString(kOptions);
  // Presto's java.sql.Timestamp.toString() always keeps one decimal place, so
  // a whole second is named "...:56.0" and must be matched as such.
  if (result.find_last_of('.') == std::string::npos) {
    result += ".0";
  }
  return result;
}

template <TypeKind kind>
std::string toStringImpl(
    const Variant& value,
    const Type& type,
    PartitionValue::TimestampMode timestampMode,
    PartitionValue::DateMode dateMode) {
  using NativeType = typename TypeTraits<kind>::NativeType;

  if (type.isDate()) {
    const auto days = value.value<TypeKind::INTEGER>();
    return dateMode == PartitionValue::DateMode::kDaysSinceEpoch
        ? fmt::to_string(days)
        : DateType::toIso8601(days);
  }

  if constexpr (
      std::is_same_v<NativeType, int64_t> ||
      std::is_same_v<NativeType, int128_t>) {
    if (type.isDecimal()) {
      return formatDecimal(value.value<kind>(), type);
    }
  }

  if constexpr (std::is_same_v<NativeType, StringView>) {
    return value.value<kind>();
  } else if constexpr (kind == TypeKind::TIMESTAMP) {
    auto timestamp = value.value<TypeKind::TIMESTAMP>();
    if (type.equivalent(*TIMESTAMP()) &&
        timestampMode == PartitionValue::TimestampMode::kLocalTime) {
      timestamp.toTimezone(Timestamp::defaultTimezone());
    }
    return timestampToString(timestamp);
  } else {
    return fmt::to_string(value.value<kind>());
  }
}

template <TypeKind kind>
Variant fromStringImpl(
    std::string_view value,
    const Type& type,
    PartitionValue::TimestampMode timestampMode,
    PartitionValue::DateMode dateMode) {
  using NativeType = typename TypeTraits<kind>::NativeType;

  if (type.isDate()) {
    const auto days = dateMode == PartitionValue::DateMode::kDaysSinceEpoch
        ? folly::to<int32_t>(value)
        : DATE()->toDays(value);
    return Variant(days);
  }

  if constexpr (
      std::is_same_v<NativeType, int64_t> ||
      std::is_same_v<NativeType, int128_t>) {
    if (type.isDecimal()) {
      NativeType decimalValue{0};
      const auto [precision, scale] = getDecimalPrecisionScale(type);
      const auto status = DecimalUtil::castFromString(
          StringView(value), precision, scale, decimalValue);
      VELOX_USER_CHECK(status.ok(), "{}", status.message());
      return Variant::create<kind>(decimalValue);
    }
  }

  if constexpr (std::is_same_v<NativeType, StringView>) {
    return Variant::create<kind>(std::string(value));
  } else {
    auto converted = util::Converter<kind>::tryCast(value).thenOrThrow(
        folly::identity,
        [&](const Status& status) { VELOX_USER_FAIL("{}", status.message()); });
    if constexpr (kind == TypeKind::TIMESTAMP) {
      if (type.equivalent(*TIMESTAMP()) &&
          timestampMode == PartitionValue::TimestampMode::kLocalTime) {
        converted.toGMT(Timestamp::defaultTimezone());
      }
    }
    return Variant::create<kind>(converted);
  }
}

} // namespace

// static
std::string PartitionValue::toString(
    const Variant& value,
    const Type& type,
    TimestampMode timestampMode,
    DateMode dateMode) {
  VELOX_USER_CHECK(!value.isNull(), "A null partition value has no string");
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      toStringImpl, type.kind(), value, type, timestampMode, dateMode);
}

// static
Variant PartitionValue::fromString(
    std::string_view value,
    const Type& type,
    TimestampMode timestampMode,
    DateMode dateMode) {
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      fromStringImpl, type.kind(), value, type, timestampMode, dateMode);
}

} // namespace facebook::velox::connector::hive
