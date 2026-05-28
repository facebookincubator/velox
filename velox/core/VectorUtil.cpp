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

#include "velox/core/VectorUtil.h"
#include <cstddef>
#include <cstdint>
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/Timestamp.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"
#include "velox/vector/ConstantVector.h"

namespace facebook::velox::core {
namespace {

// Helper function to handle TimestampWithTimeZone type conversion (BIGINT kind)
std::optional<VectorPtr> handleTimestampWithTimeZoneTypeConversion(
    const TypePtr& type,
    const std::string& value,
    memory::MemoryPool* pool) {
  auto timestampResult = util::fromTimestampWithTimezoneString(
      StringView(value), util::TimestampParseMode::kSparkCast);
  if (timestampResult.hasError()) {
    // Fall through to normal BIGINT handling
    return std::nullopt;
  }
  
  auto parsed = std::move(timestampResult).value();
  Timestamp timestamp = parsed.timestamp;
  TimeZoneKey timeZoneKey = 0; // UTC_KEY

  // Pack with UTC timezone key
  int64_t packedValue = pack(timestamp, timeZoneKey);
  return std::make_shared<ConstantVector<int64_t>>(
      pool, 1, false, type, std::move(packedValue));
}

template <TypeKind kind>
VectorPtr newConstantFromStringImpl(
    const TypePtr& type,
    const std::optional<std::string>& value,
    memory::MemoryPool* pool,
    bool isLocalTimestamp,
    bool isDaysSinceEpoch,
    const tz::TimeZone* timezone) {
  using T = typename TypeTraits<kind>::NativeType;
  if (!value.has_value()) {
    return std::make_shared<ConstantVector<T>>(pool, 1, true, type, T());
  }

  if (type->isDate()) {
    int32_t days = 0;
    // For Iceberg, the date partition values are already in daysSinceEpoch
    // form.
    if (isDaysSinceEpoch) {
      days = folly::to<int32_t>(value.value());
    } else {
      days = DATE()->toDays(value.value());
    }
    return std::make_shared<ConstantVector<int32_t>>(
        pool, 1, false, type, std::move(days));
  }

  if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int128_t>) {
    if (type->isDecimal()) {
      T decimalValue = 0;
      auto [precision, scale] = getDecimalPrecisionScale(*type);
      auto status = DecimalUtil::castFromString(
          StringView(value.value()), precision, scale, decimalValue);
      if (!status.ok()) {
        VELOX_USER_FAIL(status.message());
      }
      return std::make_shared<ConstantVector<T>>(
          pool, 1, false, type, std::move(decimalValue));
    } else if (isTimestampWithTimeZoneType(type)) {
      auto result = handleTimestampWithTimeZoneTypeConversion(
          type, value.value(), pool);
      if (result.has_value()) {
        return result.value();
      }
      // Fall through to normal BIGINT handling if conversion failed
    }
  }

  if constexpr (std::is_same_v<T, StringView>) {
    return std::make_shared<ConstantVector<StringView>>(
        pool, 1, false, type, StringView(value.value()));
  } else {
    auto copy = velox::util::Converter<kind>::tryCast(value.value())
                    .thenOrThrow(folly::identity, [&](const Status& status) {
                      VELOX_USER_FAIL("{}", status.message());
                    });
    if constexpr (kind == TypeKind::TIMESTAMP) {
      VELOX_DCHECK(type->equivalent(*TIMESTAMP()));
      if (timezone) {
        copy.toGMT(*timezone);
      } else if (isLocalTimestamp) {
        copy.toGMT(Timestamp::defaultTimezone());
      }
    }
    return std::make_shared<ConstantVector<T>>(
        pool, 1, false, type, std::move(copy));
  }
}

} // namespace

VectorPtr newConstantFromString(
    const TypePtr& type,
    const std::optional<std::string>& value,
    memory::MemoryPool* pool,
    bool isLocalTimestamp,
    bool isDaysSinceEpoch,
    const tz::TimeZone* timezone) {
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
      newConstantFromStringImpl,
      type->kind(),
      type,
      value,
      pool,
      isLocalTimestamp,
      isDaysSinceEpoch,
      timezone);
}

} // namespace facebook::velox::core
