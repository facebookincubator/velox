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

#include "velox/experimental/cudf/expression/TimestampWithTimeZoneColumn.h"
#include "velox/experimental/cudf/expression/TimezoneConversion.h"

#include "velox/common/base/Exceptions.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <limits>

namespace facebook::velox::cudf_velox {
namespace {

constexpr cudf::type_id kInt64 = cudf::type_id::INT64;
constexpr cudf::type_id kBool8 = cudf::type_id::BOOL8;
constexpr cudf::type_id kTsMillis = cudf::type_id::TIMESTAMP_MILLISECONDS;

cudf::data_type int64Type() {
  return cudf::data_type{kInt64};
}

cudf::numeric_scalar<int64_t> int64Scalar(
    int64_t value,
    rmm::cuda_stream_view stream) {
  return cudf::numeric_scalar<int64_t>(value, true, stream);
}

// Reinterprets an 8-byte-wide column (timestamp/duration/int64) as another
// 8-byte type without copying.
cudf::column_view bitcastColumn(
    const cudf::column_view& view,
    cudf::type_id id) {
  return cudf::column_view{
      cudf::data_type{id},
      view.size(),
      view.head<int64_t>(),
      view.null_mask(),
      view.null_count(),
      view.offset()};
}

// Mirrors the CPU pack() range check: throws if any non-null millis value falls
// outside [kMinMillisUtc, kMaxMillisUtc].
void checkMillisInRange(
    const cudf::column_view& millis,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (millis.size() == 0 || millis.null_count() == millis.size()) {
    return;
  }
  auto minScalar = cudf::reduce(
      millis,
      *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
      int64Type(),
      stream,
      mr);
  auto maxScalar = cudf::reduce(
      millis,
      *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
      int64Type(),
      stream,
      mr);
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
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::binary_operation(
      packed,
      int64Scalar(kTimezoneMask, stream),
      cudf::binary_operator::BITWISE_AND,
      int64Type(),
      stream,
      mr);
}

std::vector<int16_t> tswtzDistinctZoneKeys(
    const cudf::column_view& perRowZoneKey,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto unique = cudf::distinct(
      cudf::table_view{{perRowZoneKey}},
      {0},
      cudf::duplicate_keep_option::KEEP_ANY,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      mr);
  auto uniqueKeys = unique->view().column(0);
  auto uniqueValid = cudf::is_valid(uniqueKeys, stream, mr);
  std::vector<int64_t> hostKeys(uniqueKeys.size());
  std::vector<int8_t> hostValid(uniqueKeys.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      hostKeys.data(),
      uniqueKeys.data<int64_t>(),
      hostKeys.size() * sizeof(int64_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      hostValid.data(),
      uniqueValid->view().data<int8_t>(),
      hostValid.size() * sizeof(int8_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  std::vector<int16_t> keys;
  keys.reserve(uniqueKeys.size());
  for (cudf::size_type i = 0; i < uniqueKeys.size(); ++i) {
    if (hostValid[i]) { // Skip the null zone key.
      keys.push_back(static_cast<int16_t>(hostKeys[i]));
    }
  }
  return keys;
}

std::unique_ptr<cudf::column> tswtzUtcInstant(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto millis = cudf::binary_operation(
      packed,
      int64Scalar(kMillisShift, stream),
      cudf::binary_operator::SHIFT_RIGHT,
      int64Type(),
      stream,
      mr);
  return std::make_unique<cudf::column>(
      bitcastColumn(millis->view(), kTsMillis), stream, mr);
}

std::unique_ptr<cudf::column> tswtzOffsetSeconds(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto utcInstant = tswtzUtcInstant(packed, stream, mr);
  auto perRowKey = tswtzZoneKey(packed, stream, mr);
  auto keys = tswtzDistinctZoneKeys(perRowKey->view(), stream, mr);

  // Start all-null; fill each zone's rows. A null key matches no real key, so
  // its rows keep the null default (CPU propagates null).
  auto result = cudf::make_numeric_column(
      int64Type(), packed.size(), cudf::mask_state::ALL_NULL, stream, mr);
  for (const int16_t zoneKey : keys) {
    auto offsetDuration = utcOffsetSeconds(
        utcInstant->view(), tz::getTimeZoneName(zoneKey), stream, mr);
    auto offsetSeconds = std::make_unique<cudf::column>(
        bitcastColumn(offsetDuration->view(), kInt64), stream, mr);
    auto isThisZone = cudf::binary_operation(
        perRowKey->view(),
        int64Scalar(zoneKey, stream),
        cudf::binary_operator::EQUAL,
        cudf::data_type{kBool8},
        stream,
        mr);
    result = cudf::copy_if_else(
        offsetSeconds->view(), result->view(), isThisZone->view(), stream, mr);
  }
  return result;
}

std::unique_ptr<cudf::column> tswtzLocalWallClock(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto utcInstant = tswtzUtcInstant(packed, stream, mr);
  auto millis = bitcastColumn(utcInstant->view(), kInt64);
  auto offsetSeconds = tswtzOffsetSeconds(packed, stream, mr);
  auto offsetMillis = cudf::binary_operation(
      offsetSeconds->view(),
      int64Scalar(1'000, stream),
      cudf::binary_operator::MUL,
      int64Type(),
      stream,
      mr);
  auto localMillis = cudf::binary_operation(
      millis,
      offsetMillis->view(),
      cudf::binary_operator::ADD,
      int64Type(),
      stream,
      mr);
  return std::make_unique<cudf::column>(
      bitcastColumn(localMillis->view(), kTsMillis), stream, mr);
}

std::unique_ptr<cudf::column> tswtzLocalToUtc(
    const cudf::column_view& localMillisTs,
    const cudf::column_view& perRowZoneKey,
    const std::vector<int16_t>& distinctKeys,
    bool correctForward,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto result = cudf::make_timestamp_column(
      cudf::data_type{kTsMillis},
      localMillisTs.size(),
      cudf::mask_state::ALL_NULL,
      stream,
      mr);
  for (const int16_t zoneKey : distinctKeys) {
    auto isThisZone = cudf::binary_operation(
        perRowZoneKey,
        int64Scalar(zoneKey, stream),
        cudf::binary_operator::EQUAL,
        cudf::data_type{kBool8},
        stream,
        mr);
    // Mask out other zones' rows (null) so this zone's gap check ignores them.
    auto nullTs = cudf::make_timestamp_column(
        cudf::data_type{kTsMillis},
        localMillisTs.size(),
        cudf::mask_state::ALL_NULL,
        stream,
        mr);
    auto maskedLocal = cudf::copy_if_else(
        localMillisTs, nullTs->view(), isThisZone->view(), stream, mr);
    const auto zoneName = tz::getTimeZoneName(zoneKey);
    // correctForward is wired to toUtcTimestampCorrecting in Phase 4
    // (date_add(TSWTZ)); Phase 2 (date_trunc) only uses the throwing path.
    VELOX_CHECK(
        !correctForward, "gap-correcting local-to-UTC is not yet implemented");
    auto utc = toUtcTimestamp(maskedLocal->view(), zoneName, stream, mr);
    result = cudf::copy_if_else(
        utc->view(), result->view(), isThisZone->view(), stream, mr);
  }
  return result;
}

std::unique_ptr<cudf::column> tswtzPack(
    const cudf::column_view& utcInstant,
    const cudf::column_view& perRowZoneKey,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Normalize to a millisecond instant, then bit-cast to raw int64 millis.
  std::unique_ptr<cudf::column> millisTs;
  cudf::column_view millisView;
  if (utcInstant.type().id() == kTsMillis) {
    millisView = utcInstant;
  } else {
    millisTs = cudf::cast(utcInstant, cudf::data_type{kTsMillis}, stream, mr);
    millisView = millisTs->view();
  }
  auto millis = std::make_unique<cudf::column>(
      bitcastColumn(millisView, kInt64), stream, mr);
  checkMillisInRange(millis->view(), stream, mr);

  auto shifted = cudf::binary_operation(
      millis->view(),
      int64Scalar(kMillisShift, stream),
      cudf::binary_operator::SHIFT_LEFT,
      int64Type(),
      stream,
      mr);
  auto maskedKey = cudf::binary_operation(
      perRowZoneKey,
      int64Scalar(kTimezoneMask, stream),
      cudf::binary_operator::BITWISE_AND,
      int64Type(),
      stream,
      mr);
  return cudf::binary_operation(
      shifted->view(),
      maskedKey->view(),
      cudf::binary_operator::BITWISE_OR,
      int64Type(),
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox
