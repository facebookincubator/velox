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
#include "velox/experimental/cudf/exec/PrestoPageDevice.h"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <cstdint>

namespace facebook::velox::cudf_velox::detail {
namespace {

constexpr int kBlockSize = 256;
constexpr size_t kMaxGridY = 65535;

__device__ __forceinline__ bool isValidBit(
    const cudf::bitmask_type* mask,
    int64_t bit) {
  return (mask[bit >> 5] >> (bit & 31)) & 1u;
}

__device__ __forceinline__ int64_t
offsetAt(const StringOffsets& offsets, int64_t index) {
  return offsets.offsetsInt64
      ? static_cast<const int64_t*>(offsets.offsets)[index]
      : static_cast<int64_t>(
            static_cast<const int32_t*>(offsets.offsets)[index]);
}

__device__ __forceinline__ void storeInt32Unaligned(uint8_t* dst, int32_t v) {
  const auto u = static_cast<uint32_t>(v);
  dst[0] = static_cast<uint8_t>(u);
  dst[1] = static_cast<uint8_t>(u >> 8);
  dst[2] = static_cast<uint8_t>(u >> 16);
  dst[3] = static_cast<uint8_t>(u >> 24);
}

// Serialized bits of one row: the fixed part plus 8 bits per string byte.
struct RowBits {
  int64_t fixedBits;
  const StringOffsets* strings;
  int32_t numStrings;

  __device__ int64_t operator()(int32_t row) const {
    int64_t stringBytes = 0;
    for (int32_t s = 0; s < numStrings; ++s) {
      const auto index = strings[s].firstIndex + row;
      stringBytes +=
          offsetAt(strings[s], index + 1) - offsetAt(strings[s], index);
    }
    return fixedBits + 8 * stringBytes;
  }
};

// Whether row i starts a page: its last bit falls into a later span of
// pageBits than the last bit of row i - 1.
struct StartsPage {
  const int64_t* cumulativeBits;
  int64_t pageBits;

  __device__ bool operator()(int32_t i) const {
    return (cumulativeBits[i] - 1) / pageBits !=
        (cumulativeBits[i - 1] - 1) / pageBits;
  }
};

// One block row (blockIdx.y) per task, grid-stride over the task's bytes.
__global__ void nullBytesKernel(
    const NullBytesTask* tasks,
    int32_t taskOffset) {
  const NullBytesTask t = tasks[taskOffset + blockIdx.y];
  const int32_t numBytes = (t.numRows + 7) >> 3;
  for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < numBytes;
       j += gridDim.x * blockDim.x) {
    uint8_t out = 0;
    const int32_t row0 = j << 3;
    const int32_t count = min(8, t.numRows - row0);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (k < count && !isValidBit(t.mask, t.firstBit + row0 + k)) {
        out |= static_cast<uint8_t>(0x80u >> k);
      }
    }
    t.dst[j] = out;
  }
}

__global__ void stringOffsetsKernel(
    const StringOffsetsTask* tasks,
    int32_t taskOffset) {
  const StringOffsetsTask t = tasks[taskOffset + blockIdx.y];
  const int64_t first = t.offsets.firstIndex + t.firstRow;
  const int64_t start = offsetAt(t.offsets, first);
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < t.numRows;
       i += gridDim.x * blockDim.x) {
    const int64_t end = offsetAt(t.offsets, first + 1 + i);
    storeInt32Unaligned(
        t.dst + 4 * static_cast<int64_t>(i), static_cast<int32_t>(end - start));
  }
}

__global__ void gatherOffsetsKernel(
    StringOffsets offsets,
    const int64_t* rows,
    int32_t n,
    int64_t* out) {
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    out[i] = offsetAt(offsets, offsets.firstIndex + rows[i]);
  }
}

// Exactly one of mul, div is 1. Flags a product outside int64.
__global__ void toMillisKernel(
    const int64_t* src,
    int32_t n,
    int64_t mul,
    int64_t div,
    int64_t* dst,
    int32_t* overflow) {
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const int64_t v = src[i];
    if (mul != 1) {
      if (v > INT64_MAX / mul || v < INT64_MIN / mul) {
        *overflow = 1;
        dst[i] = 0;
      } else {
        dst[i] = v * mul;
      }
    } else if (div == 1) {
      dst[i] = v;
    } else {
      // Floor division, as Timestamp::toMillis() rounds.
      int64_t q = v / div;
      if (v - q * div != 0 && v < 0) {
        --q;
      }
      dst[i] = q;
    }
  }
}

struct ValidPredicate {
  const cudf::bitmask_type* mask;
  int64_t firstBit;

  __device__ bool operator()(int32_t i) const {
    return isValidBit(mask, firstBit + i);
  }
};

struct ToJavaDecimal {
  __device__ __int128_t operator()(__int128_t v) const {
    if (v >= 0) {
      return v;
    }
    const auto magnitude =
        static_cast<unsigned __int128>(0) - static_cast<unsigned __int128>(v);
    return static_cast<__int128_t>(
        magnitude | (static_cast<unsigned __int128>(1) << 127));
  }
};

struct alignas(16) Bytes16 {
  uint64_t lo;
  uint64_t hi;
};

template <typename T>
int32_t compactValidTyped(
    const T* src,
    const cudf::bitmask_type* mask,
    int64_t firstBit,
    int32_t numRows,
    T* dst,
    cuda::stream_ref stream) {
  auto end = thrust::copy_if(
      rmm::exec_policy(stream),
      src,
      src + numRows,
      thrust::counting_iterator<int32_t>(0),
      dst,
      ValidPredicate{mask, firstBit});
  return static_cast<int32_t>(end - dst);
}

uint32_t gridXFor(int64_t maxItems) {
  const int64_t blocks = (maxItems + kBlockSize - 1) / kBlockSize;
  return static_cast<uint32_t>(std::clamp<int64_t>(blocks, 1, 128));
}

} // namespace

std::vector<int64_t> prestoPageBoundaries(
    int32_t numRows,
    int64_t fixedBitsPerRow,
    std::span<const StringOffsets> strings,
    int64_t pageBytes,
    int64_t maxRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<int64_t> bounds{0};
  if (numRows == 0) {
    return bounds;
  }
  const int64_t pageBits = pageBytes * 8;
  if (strings.empty()) {
    // Every row costs the same, so the pages are equal spans of rows.
    const int64_t rowsPerPage = std::clamp<int64_t>(
        fixedBitsPerRow > 0 ? pageBits / fixedBitsPerRow : maxRows, 1, maxRows);
    for (int64_t row = rowsPerPage; row < numRows; row += rowsPerPage) {
      bounds.push_back(row);
    }
    bounds.push_back(numRows);
    return bounds;
  }

  auto deviceStrings = cudf::detail::make_device_uvector(
      cudf::host_span<StringOffsets const>{strings.data(), strings.size()},
      stream,
      mr);
  rmm::device_uvector<int64_t> cumulativeBits(numRows, stream, mr);
  thrust::transform_inclusive_scan(
      rmm::exec_policy(stream),
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(numRows),
      cumulativeBits.begin(),
      RowBits{
          fixedBitsPerRow,
          deviceStrings.data(),
          static_cast<int32_t>(strings.size())},
      cuda::std::plus<int64_t>{});
  // Page k holds the rows whose last bit lies in (k * pageBits, (k + 1) *
  // pageBits], so a page exceeds pageBits by less than one row and a row
  // larger than a page forms one page.
  rmm::device_uvector<int32_t> cuts(numRows, stream, mr);
  auto cutsEnd = thrust::copy_if(
      rmm::exec_policy(stream),
      thrust::counting_iterator<int32_t>(1),
      thrust::counting_iterator<int32_t>(numRows),
      cuts.begin(),
      StartsPage{cumulativeBits.data(), pageBits});
  const auto hostCuts = cudf::detail::make_std_vector(
      cudf::device_span<int32_t const>{
          cuts.data(), static_cast<size_t>(cutsEnd - cuts.begin())},
      stream);
  bounds.insert(bounds.end(), hostCuts.begin(), hostCuts.end());
  bounds.push_back(numRows);

  if (maxRows < numRows) {
    std::vector<int64_t> capped{0};
    for (size_t i = 1; i < bounds.size(); ++i) {
      while (bounds[i] - capped.back() > maxRows) {
        capped.push_back(capped.back() + maxRows);
      }
      capped.push_back(bounds[i]);
    }
    bounds = std::move(capped);
  }
  return bounds;
}

void fillPrestoNullBytes(
    std::span<const NullBytesTask> tasks,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (tasks.empty()) {
    return;
  }
  auto deviceTasks = cudf::detail::make_device_uvector(
      cudf::host_span<NullBytesTask const>{tasks.data(), tasks.size()},
      stream,
      mr);
  int64_t maxBytes = 0;
  for (const auto& t : tasks) {
    maxBytes = std::max<int64_t>(maxBytes, (t.numRows + 7) / 8);
  }
  const auto gridX = gridXFor(maxBytes);
  for (size_t offset = 0; offset < tasks.size(); offset += kMaxGridY) {
    const auto count =
        static_cast<uint32_t>(std::min(kMaxGridY, tasks.size() - offset));
    nullBytesKernel<<<dim3(gridX, count), kBlockSize, 0, stream.get()>>>(
        deviceTasks.data(), static_cast<int32_t>(offset));
    CUDF_CHECK_CUDA(stream.get());
  }
}

void fillPrestoStringOffsets(
    std::span<const StringOffsetsTask> tasks,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (tasks.empty()) {
    return;
  }
  auto deviceTasks = cudf::detail::make_device_uvector(
      cudf::host_span<StringOffsetsTask const>{tasks.data(), tasks.size()},
      stream,
      mr);
  int64_t maxRows = 0;
  for (const auto& t : tasks) {
    maxRows = std::max<int64_t>(maxRows, t.numRows);
  }
  const auto gridX = gridXFor(maxRows);
  for (size_t offset = 0; offset < tasks.size(); offset += kMaxGridY) {
    const auto count =
        static_cast<uint32_t>(std::min(kMaxGridY, tasks.size() - offset));
    stringOffsetsKernel<<<dim3(gridX, count), kBlockSize, 0, stream.get()>>>(
        deviceTasks.data(), static_cast<int32_t>(offset));
    CUDF_CHECK_CUDA(stream.get());
  }
}

void batchedDeviceCopy(
    std::span<const CopyTask> tasks,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<const uint8_t*> srcs;
  std::vector<uint8_t*> dsts;
  std::vector<size_t> sizes;
  srcs.reserve(tasks.size());
  dsts.reserve(tasks.size());
  sizes.reserve(tasks.size());
  for (const auto& t : tasks) {
    if (t.size == 0) {
      continue;
    }
    srcs.push_back(t.src);
    dsts.push_back(t.dst);
    sizes.push_back(t.size);
  }
  if (srcs.empty()) {
    return;
  }
  auto deviceSrcs = cudf::detail::make_device_uvector(
      cudf::host_span<const uint8_t* const>{srcs.data(), srcs.size()},
      stream,
      mr);
  auto deviceDsts = cudf::detail::make_device_uvector(
      cudf::host_span<uint8_t* const>{dsts.data(), dsts.size()}, stream, mr);
  auto deviceSizes = cudf::detail::make_device_uvector(
      cudf::host_span<size_t const>{sizes.data(), sizes.size()}, stream, mr);
  size_t tempBytes = 0;
  CUDF_CUDA_TRY(
      cub::DeviceMemcpy::Batched(
          nullptr,
          tempBytes,
          deviceSrcs.data(),
          deviceDsts.data(),
          deviceSizes.data(),
          srcs.size(),
          stream.get()));
  rmm::device_buffer temp(tempBytes, stream, mr);
  CUDF_CUDA_TRY(
      cub::DeviceMemcpy::Batched(
          temp.data(),
          tempBytes,
          deviceSrcs.data(),
          deviceDsts.data(),
          deviceSizes.data(),
          srcs.size(),
          stream.get()));
  CUDF_CHECK_CUDA(stream.get());
}

std::vector<int64_t> gatherStringOffsets(
    const StringOffsets& offsets,
    std::span<const int64_t> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (rows.empty()) {
    return {};
  }
  auto deviceRows = cudf::detail::make_device_uvector(
      cudf::host_span<int64_t const>{rows.data(), rows.size()}, stream, mr);
  rmm::device_uvector<int64_t> out(rows.size(), stream, mr);
  gatherOffsetsKernel<<<gridXFor(rows.size()), kBlockSize, 0, stream.get()>>>(
      offsets,
      deviceRows.data(),
      static_cast<int32_t>(rows.size()),
      out.data());
  CUDF_CHECK_CUDA(stream.get());
  return cudf::detail::make_std_vector(
      cudf::device_span<int64_t const>{out.data(), out.size()}, stream);
}

bool timestampsToMillis(
    const int64_t* src,
    int32_t numRows,
    int64_t mul,
    int64_t div,
    int64_t* dst,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (numRows == 0) {
    return true;
  }
  // The initial value is copied asynchronously, so it outlives the kernel.
  const int32_t zero = 0;
  rmm::device_scalar<int32_t> overflow(zero, stream, mr);
  toMillisKernel<<<gridXFor(numRows), kBlockSize, 0, stream.get()>>>(
      src, numRows, mul, div, dst, overflow.data());
  CUDF_CHECK_CUDA(stream.get());
  return overflow.value(stream) == 0;
}

void decimalsToJava(
    const __int128_t* src,
    int32_t numRows,
    __int128_t* dst,
    cuda::stream_ref stream) {
  if (numRows == 0) {
    return;
  }
  thrust::transform(
      rmm::exec_policy(stream), src, src + numRows, dst, ToJavaDecimal{});
}

int32_t compactValid(
    const uint8_t* src,
    int32_t width,
    const cudf::bitmask_type* mask,
    int64_t firstBit,
    int32_t numRows,
    uint8_t* dst,
    cuda::stream_ref stream) {
  if (numRows == 0) {
    return 0;
  }
  switch (width) {
    case 1:
      return compactValidTyped<uint8_t>(
          src, mask, firstBit, numRows, dst, stream);
    case 2:
      return compactValidTyped<uint16_t>(
          reinterpret_cast<const uint16_t*>(src),
          mask,
          firstBit,
          numRows,
          reinterpret_cast<uint16_t*>(dst),
          stream);
    case 4:
      return compactValidTyped<uint32_t>(
          reinterpret_cast<const uint32_t*>(src),
          mask,
          firstBit,
          numRows,
          reinterpret_cast<uint32_t*>(dst),
          stream);
    case 8:
      return compactValidTyped<uint64_t>(
          reinterpret_cast<const uint64_t*>(src),
          mask,
          firstBit,
          numRows,
          reinterpret_cast<uint64_t*>(dst),
          stream);
    case 16:
      return compactValidTyped<Bytes16>(
          reinterpret_cast<const Bytes16*>(src),
          mask,
          firstBit,
          numRows,
          reinterpret_cast<Bytes16*>(dst),
          stream);
    default:
      CUDF_FAIL("Unsupported fixed width for Presto page compaction");
  }
}

} // namespace facebook::velox::cudf_velox::detail
