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
    PartitionValue::DateMode dateMode,
    const tz::TimeZone* timezone) {
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
      // A TIMESTAMP_UTC value is always UTC and is never shifted.
      if (type.equivalent(*TIMESTAMP())) {
        if (timezone != nullptr) {
          converted.toGMT(*timezone);
        } else if (timestampMode == PartitionValue::TimestampMode::kLocalTime) {
          converted.toGMT(Timestamp::defaultTimezone());
        }
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
    DateMode dateMode,
    const tz::TimeZone* timezone) {
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      fromStringImpl,
      type.kind(),
      value,
      type,
      timestampMode,
      dateMode,
      timezone);
}

} // namespace facebook::velox::connector::hive
