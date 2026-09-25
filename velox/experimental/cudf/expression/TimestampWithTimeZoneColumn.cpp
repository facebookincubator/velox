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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/expression/TimestampWithTimeZoneColumn.h"
#include "velox/experimental/cudf/expression/TimezoneConversion.h"

#include "velox/common/base/Exceptions.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <limits>

namespace facebook::velox::cudf_velox {
namespace {

constexpr cudf::type_id kInt64 = cudf::type_id::INT64;
constexpr cudf::type_id kBool8 = cudf::type_id::BOOL8;
constexpr cudf::type_id kTsMillis = cudf::type_id::TIMESTAMP_MILLISECONDS;

// Mirrors the CPU pack() range check: throws if any non-null millis value falls
// outside [kMinMillisUtc, kMaxMillisUtc].
void checkMillisInRange(
    const cudf::column_view& millis,
    cuda::stream_ref stream) {
  if (millis.size() == 0 || millis.null_count() == millis.size()) {
    return;
  }
  auto minScalar = cudf::reduce(
      millis,
      *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{kInt64},
      stream,
      get_temp_mr());
  auto maxScalar = cudf::reduce(
      millis,
      *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{kInt64},
      stream,
      get_temp_mr());
  const auto lo = static_cast<cudf::numeric_scalar<int64_t>*>(minScalar.get())
                      ->value(stream);
  const auto hi = static_cast<cudf::numeric_scalar<int64_t>*>(maxScalar.get())
                      ->value(stream);
  VELOX_USER_CHECK(
      lo >= kMinMillisUtc && hi <= kMaxMillisUtc,
      "TimestampWithTimeZone overflow: [{}, {}] ms",
      lo,
      hi);
}

} // namespace

std::unique_ptr<cudf::column> tswtzZoneKey(
    const cudf::column_view& packed,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return cudf::binary_operation(
      packed,
      cudf::numeric_scalar<int64_t>(kTimezoneMask, true, stream, get_temp_mr()),
      cudf::binary_operator::BITWISE_AND,
      cudf::data_type{kInt64},
      stream,
      mr);
}

std::vector<int16_t> tswtzDistinctZoneKeys(
    const cudf::column_view& perRowZoneKey,
    cuda::stream_ref stream) {
  auto unique = cudf::distinct(
      cudf::table_view{{perRowZoneKey}},
      {0},
      cudf::duplicate_keep_option::KEEP_ANY,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      get_temp_mr());
  auto validOnly = cudf::drop_nulls(unique->view(), {0}, stream, get_temp_mr());
  auto uniqueKeys = validOnly->view().column(0);
  auto hostKeys = cudf::detail::make_std_vector<int64_t>(
      cudf::device_span<int64_t const>{
          uniqueKeys.data<int64_t>(), static_cast<size_t>(uniqueKeys.size())},
      stream);

  std::vector<int16_t> keys;
  keys.reserve(uniqueKeys.size());
  for (const auto key : hostKeys) {
    keys.push_back(static_cast<int16_t>(key));
  }
  return keys;
}

std::unique_ptr<cudf::column> tswtzClearZoneKey(
    const cudf::column_view& packed,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return cudf::binary_operation(
      packed,
      cudf::numeric_scalar<int64_t>(
          ~static_cast<int64_t>(kTimezoneMask), true, stream, get_temp_mr()),
      cudf::binary_operator::BITWISE_AND,
      cudf::data_type{kInt64},
      stream,
      mr);
}

std::unique_ptr<cudf::column> tswtzUtcInstant(
    const cudf::column_view& packed,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto millis = cudf::binary_operation(
      packed,
      cudf::numeric_scalar<int64_t>(kMillisShift, true, stream, get_temp_mr()),
      cudf::binary_operator::SHIFT_RIGHT,
      cudf::data_type{kInt64},
      stream,
      get_temp_mr());
  return std::make_unique<cudf::column>(
      cudf::bit_cast(millis->view(), cudf::data_type{kTsMillis}), stream, mr);
}

std::unique_ptr<cudf::column> tswtzOffsetSeconds(
    const cudf::column_view& packed,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto utcInstant = tswtzUtcInstant(packed, stream, get_temp_mr());
  auto perRowKey = tswtzZoneKey(packed, stream, get_temp_mr());
  auto keys = tswtzDistinctZoneKeys(perRowKey->view(), stream);

  // Start all-null; fill each zone's rows. A null key matches no real key, so
  // its rows keep the null default (CPU propagates null).
  auto result = cudf::make_numeric_column(
      cudf::data_type{kInt64},
      packed.size(),
      cudf::mask_state::ALL_NULL,
      stream,
      mr);
  for (const int16_t zoneKey : keys) {
    auto offsetDuration = utcOffsetSeconds(
        utcInstant->view(),
        tz::getTimeZoneName(zoneKey),
        stream,
        get_temp_mr());
    auto offsetSeconds = std::make_unique<cudf::column>(
        cudf::bit_cast(offsetDuration->view(), cudf::data_type{kInt64}),
        stream,
        get_temp_mr());
    auto isThisZone = cudf::binary_operation(
        perRowKey->view(),
        cudf::numeric_scalar<int64_t>(zoneKey, true, stream, get_temp_mr()),
        cudf::binary_operator::EQUAL,
        cudf::data_type{kBool8},
        stream,
        get_temp_mr());
    result = cudf::copy_if_else(
        offsetSeconds->view(), result->view(), isThisZone->view(), stream, mr);
  }
  return result;
}

std::unique_ptr<cudf::column> tswtzLocalWallClock(
    const cudf::column_view& packed,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto utcInstant = tswtzUtcInstant(packed, stream, get_temp_mr());
  auto millis = cudf::bit_cast(utcInstant->view(), cudf::data_type{kInt64});
  auto offsetSeconds = tswtzOffsetSeconds(packed, stream, get_temp_mr());
  auto offsetMillis = cudf::binary_operation(
      offsetSeconds->view(),
      cudf::numeric_scalar<int64_t>(1'000, true, stream, get_temp_mr()),
      cudf::binary_operator::MUL,
      cudf::data_type{kInt64},
      stream,
      get_temp_mr());
  auto localMillis = cudf::binary_operation(
      millis,
      offsetMillis->view(),
      cudf::binary_operator::ADD,
      cudf::data_type{kInt64},
      stream,
      get_temp_mr());
  return std::make_unique<cudf::column>(
      cudf::bit_cast(localMillis->view(), cudf::data_type{kTsMillis}),
      stream,
      mr);
}

std::unique_ptr<cudf::column> tswtzLocalToUtc(
    const cudf::column_view& localMillisTs,
    const cudf::column_view& perRowZoneKey,
    const std::vector<int16_t>& distinctKeys,
    bool correctForward,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto result = cudf::make_timestamp_column(
      cudf::data_type{kTsMillis},
      localMillisTs.size(),
      cudf::mask_state::ALL_NULL,
      stream,
      mr);
  auto nullTs = cudf::make_timestamp_column(
      cudf::data_type{kTsMillis},
      localMillisTs.size(),
      cudf::mask_state::ALL_NULL,
      stream,
      get_temp_mr());
  for (const int16_t zoneKey : distinctKeys) {
    auto isThisZone = cudf::binary_operation(
        perRowZoneKey,
        cudf::numeric_scalar<int64_t>(zoneKey, true, stream, get_temp_mr()),
        cudf::binary_operator::EQUAL,
        cudf::data_type{kBool8},
        stream,
        get_temp_mr());
    // Mask out other zones' rows (null) so this zone's gap check ignores them.
    auto maskedLocal = cudf::copy_if_else(
        localMillisTs,
        nullTs->view(),
        isThisZone->view(),
        stream,
        get_temp_mr());
    const auto zoneName = tz::getTimeZoneName(zoneKey);
    // correctForward is wired to toUtcTimestampCorrecting in Phase 4
    // (date_add(TSWTZ)); Phase 2 (date_trunc) only uses the throwing path.
    VELOX_CHECK(
        !correctForward, "gap-correcting local-to-UTC is not yet implemented");
    auto utc =
        toUtcTimestamp(maskedLocal->view(), zoneName, stream, get_temp_mr());
    result = cudf::copy_if_else(
        utc->view(), result->view(), isThisZone->view(), stream, mr);
  }
  return result;
}

std::unique_ptr<cudf::column> tswtzPack(
    const cudf::column_view& utcInstant,
    const cudf::column_view& perRowZoneKey,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  // Normalize to a millisecond instant, then bit-cast to raw int64 millis.
  std::unique_ptr<cudf::column> millisTs;
  cudf::column_view millisView;
  if (utcInstant.type().id() == kTsMillis) {
    millisView = utcInstant;
  } else {
    millisTs = cudf::cast(
        utcInstant, cudf::data_type{kTsMillis}, stream, get_temp_mr());
    millisView = millisTs->view();
  }
  auto millis = std::make_unique<cudf::column>(
      cudf::bit_cast(millisView, cudf::data_type{kInt64}),
      stream,
      get_temp_mr());
  checkMillisInRange(millis->view(), stream);

  auto shifted = cudf::binary_operation(
      millis->view(),
      cudf::numeric_scalar<int64_t>(kMillisShift, true, stream, get_temp_mr()),
      cudf::binary_operator::SHIFT_LEFT,
      cudf::data_type{kInt64},
      stream,
      get_temp_mr());
  auto maskedKey = cudf::binary_operation(
      perRowZoneKey,
      cudf::numeric_scalar<int64_t>(kTimezoneMask, true, stream, get_temp_mr()),
      cudf::binary_operator::BITWISE_AND,
      cudf::data_type{kInt64},
      stream,
      get_temp_mr());
  return cudf::binary_operation(
      shifted->view(),
      maskedKey->view(),
      cudf::binary_operator::BITWISE_OR,
      cudf::data_type{kInt64},
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox
