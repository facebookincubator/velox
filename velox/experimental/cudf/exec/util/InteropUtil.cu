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

#include "velox/experimental/cudf/exec/util/InteropUtil.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <optional>
#include <string_view>

namespace facebook::velox::cudf_velox::util {

// DeviceTimestamp matches Velox Timestamp layout for device code
// Avoids including velox/type/Timestamp.h which may conflict with folly
struct DeviceTimestamp {
  int64_t seconds_;
  uint64_t nanos_;

  __device__ int64_t getSeconds() const {
    return seconds_;
  }
  __device__ uint64_t getNanos() const {
    return nanos_;
  }
};

namespace {
constexpr int64_t kNanosPerSecond = 10'000'000'00LL;
constexpr int64_t kMicrosPerSecond = 1'000'000LL;
constexpr int64_t kMillisPerSecond = 1'000LL;
} // namespace

void convertTimestamps(
    const void* dTimestamps,
    const cudf::bitmask_type* dMask,
    int64_t* dOutput,
    cudf::size_type numRows,
    cudf::type_id timestampUnit,
    std::optional<std::string_view> timestampTimeZone,
    rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(
      !timestampTimeZone.has_value() || timestampTimeZone->empty() ||
          *timestampTimeZone == "UTC",
      "Direct GPU timestamp conversion supports only UTC or empty timezone. Got: " +
          std::string(timestampTimeZone.value_or("")));

  // Use reinterpret_cast to access Timestamp data with matching layout
  auto dTimestampsTyped = reinterpret_cast<const DeviceTimestamp*>(dTimestamps);

  // Specialize the device lambda for each timestamp unit to avoid per-row
  // branching
  switch (timestampUnit) {
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      thrust::transform(
          rmm::exec_policy_nosync(stream),
          cuda::counting_iterator<cudf::size_type>(0),
          cuda::counting_iterator<cudf::size_type>(numRows),
          thrust::counting_iterator<cudf::size_type>(numRows),
          dOutput,
          [dTimestampsTyped, dMask] __device__(auto idx) -> int64_t {
            if (dMask && !cudf::bit_is_set(dMask, idx)) {
              return int64_t{};
            }
            auto const& ts = dTimestampsTyped[idx];
            return ts.getSeconds() * kNanosPerSecond +
                static_cast<int64_t>(ts.getNanos());
          });
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      thrust::transform(
          rmm::exec_policy_nosync(stream),
          thrust::counting_iterator<cudf::size_type>(0),
          thrust::counting_iterator<cudf::size_type>(numRows),
          dOutput,
          [dTimestampsTyped, dMask] __device__(auto idx) -> int64_t {
            if (dMask && !cudf::bit_is_set(dMask, idx)) {
              return int64_t{};
            }
            auto const& ts = dTimestampsTyped[idx];
            return ts.getSeconds() * kMicrosPerSecond +
                static_cast<int64_t>(ts.getNanos() / 1000);
          });
      break;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      thrust::transform(
          rmm::exec_policy_nosync(stream),
          thrust::counting_iterator<cudf::size_type>(0),
          thrust::counting_iterator<cudf::size_type>(numRows),
          dOutput,
          [dTimestampsTyped, dMask] __device__(auto idx) -> int64_t {
            if (dMask && !cudf::bit_is_set(dMask, idx)) {
              return int64_t{};
            }
            auto const& ts = dTimestampsTyped[idx];
            return ts.getSeconds() * kMillisPerSecond +
                static_cast<int64_t>(ts.getNanos() / 1000000);
          });
      break;
    case cudf::type_id::TIMESTAMP_SECONDS:
      thrust::transform(
          rmm::exec_policy_nosync(stream),
          thrust::counting_iterator<cudf::size_type>(0),
          thrust::counting_iterator<cudf::size_type>(numRows),
          dOutput,
          [dTimestampsTyped, dMask] __device__(auto idx) -> int64_t {
            if (dMask && !cudf::bit_is_set(dMask, idx)) {
              return int64_t{};
            }
            auto const& ts = dTimestampsTyped[idx];
            return ts.getSeconds();
          });
      break;
    default:
      CUDF_FAIL("Unsupported timestamp unit for direct GPU conversion");
  }
}

std::unique_ptr<cudf::column> computeOffsetsFromSizes(
    const int32_t* dSizes,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Allocate offsets buffer (numRows + 1)
  auto dOffsets = rmm::device_uvector<int32_t>(numRows + 1, stream, mr);

  // Set first offset to 0
  CUDF_CUDA_TRY(
      cudaMemsetAsync(dOffsets.data(), 0, sizeof(int32_t), stream.value()));

  // Use Thrust inclusive_scan to compute prefix sum: offsets[i+1] =
  // sum(sizes[0..i])
  thrust::inclusive_scan(
      rmm::exec_policy_nosync(stream),
      dSizes,
      dSizes + numRows,
      dOffsets.data() + 1);

  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      numRows + 1,
      dOffsets.release(),
      rmm::device_buffer{0, stream, mr},
      0);
}

} // namespace facebook::velox::cudf_velox::util

// Made with Bob
