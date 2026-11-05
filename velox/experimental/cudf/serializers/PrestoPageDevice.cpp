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

#include "velox/experimental/cudf/serializers/PrestoPageDevice.h"
#include "velox/experimental/cudf/serializers/PrestoPageDeviceKernels.h"

#include "velox/common/base/Exceptions.h"

#include <cstdint>
#include <limits>

// The contracts of the device primitives. They live here rather than in the
// .cu because Velox's exception headers do not compile under nvcc.
namespace facebook::velox::cudf_velox::detail {
namespace {

// The offsetalator reads any type other than INT32 as int64 and checks the
// type in debug builds only.
void checkOffsetsType(const StringOffsets& offsets) {
  const auto id = offsets.offsetsType.id();
  VELOX_CHECK(
      id == cudf::type_id::INT32 || id == cudf::type_id::INT64,
      "StringOffsets::offsetsType must be INT32 or INT64, got {}",
      static_cast<int32_t>(id));
}

} // namespace

std::vector<int64_t> prestoPageBoundaries(
    int32_t numRows,
    int64_t fixedBitsPerRow,
    cudf::host_span<const StringOffsets> strings,
    int64_t pageBytes,
    int64_t maxRows,
    cuda::stream_ref stream) {
  VELOX_CHECK_GT(pageBytes, 0, "pageBytes must be positive");
  VELOX_CHECK_LE(
      pageBytes,
      std::numeric_limits<int64_t>::max() / 8,
      "pageBytes must fit in bits");
  VELOX_CHECK_GT(maxRows, 0, "maxRows must be positive");
  VELOX_CHECK_GE(fixedBitsPerRow, 0, "fixedBitsPerRow must not be negative");
  for (const auto& offsets : strings) {
    checkOffsetsType(offsets);
  }
  return device::prestoPageBoundaries(
      numRows, fixedBitsPerRow, strings, pageBytes, maxRows, stream);
}

bool fillPrestoStringOffsets(
    cudf::host_span<const StringOffsetsTask> tasks,
    cuda::stream_ref stream) {
  for (const auto& task : tasks) {
    checkOffsetsType(task.offsets);
  }
  return device::fillPrestoStringOffsets(tasks, stream);
}

std::vector<int64_t> gatherStringOffsets(
    const StringOffsets& offsets,
    cudf::host_span<const int64_t> rows,
    cuda::stream_ref stream) {
  checkOffsetsType(offsets);
  return device::gatherStringOffsets(offsets, rows, stream);
}

bool timestampsToMillis(
    const int64_t* src,
    int32_t numRows,
    cudf::type_id timestampType,
    int64_t* dst,
    cuda::stream_ref stream) {
  // Ticks of the unit to milliseconds: one of mul, div is 1.
  int64_t mul = 1;
  int64_t div = 1;
  switch (timestampType) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      mul = 1'000;
      break;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      div = 1'000;
      break;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      div = 1'000'000;
      break;
    default:
      VELOX_FAIL(
          "Not a cuDF timestamp type with 64-bit ticks: {}",
          static_cast<int32_t>(timestampType));
  }
  VELOX_CHECK_EQ(
      reinterpret_cast<uintptr_t>(src) % sizeof(int64_t),
      0,
      "'src' must be 8-byte aligned");
  VELOX_CHECK_EQ(
      reinterpret_cast<uintptr_t>(dst) % sizeof(int64_t),
      0,
      "'dst' must be 8-byte aligned");
  return device::timestampsToMillis(src, numRows, mul, div, dst, stream);
}

void decimalsToJava(
    const __int128_t* src,
    int32_t numRows,
    __int128_t* dst,
    cuda::stream_ref stream) {
  VELOX_CHECK_EQ(
      reinterpret_cast<uintptr_t>(src) % sizeof(__int128_t),
      0,
      "'src' must be 16-byte aligned");
  VELOX_CHECK_EQ(
      reinterpret_cast<uintptr_t>(dst) % sizeof(__int128_t),
      0,
      "'dst' must be 16-byte aligned");
  device::decimalsToJava(src, numRows, dst, stream);
}

} // namespace facebook::velox::cudf_velox::detail
