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

template <TypeKind kind>
Variant fromStringImpl(
    std::string_view value,
    const Type& type,
    PartitionValue::TimestampMode timestampMode,
    PartitionValue::DateMode dateMode) {
  using NativeType = typename TypeTraits<kind>::NativeType;

  if (type.isDate()) {
    if (dateMode == PartitionValue::DateMode::kDaysSinceEpoch) {
      auto days = folly::tryTo<int32_t>(value);
      VELOX_USER_CHECK(
          days.hasValue(),
          "Failed to parse DATE value '{}' as days since epoch",
          value);
      return Variant(days.value());
    }
    return Variant(DATE()->toDays(value));
  }

  if constexpr (kind == TypeKind::TIMESTAMP) {
    if (timestampMode == PartitionValue::TimestampMode::kMicrosSinceEpoch) {
      auto micros = folly::tryTo<int64_t>(value);
      VELOX_USER_CHECK(
          micros.hasValue(),
          "Failed to parse TIMESTAMP value '{}' as microseconds since epoch",
          value);
      return Variant(Timestamp::fromMicros(micros.value()));
    }
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
Variant PartitionValue::fromString(
    std::string_view value,
    const Type& type,
    TimestampMode timestampMode,
    DateMode dateMode) {
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      fromStringImpl, type.kind(), value, type, timestampMode, dateMode);
}

} // namespace facebook::velox::connector::hive
