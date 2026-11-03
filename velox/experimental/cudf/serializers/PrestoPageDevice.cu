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
#include "velox/experimental/cudf/serializers/PrestoPageDevice.h"

#include <cudf/detail/offsets_iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/iterator>
#include <cuda/numeric>
#include <cuda/std/cstring>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace facebook::velox::cudf_velox::detail {
namespace {

constexpr int kBlockSize = 256;
constexpr size_t kMaxGridY = 65535;

// Offsets indices fit cudf::size_type, as the bits of a cuDF column do.
__device__ __forceinline__ int64_t
offsetAt(const StringOffsets& offsets, int64_t index) {
  return cudf::detail::input_offsetalator(
      offsets.offsets,
      offsets.offsetsType)[static_cast<cudf::size_type>(index)];
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

// Receives the inclusive scan of RowBits row by row and marks the rows that
// start a page, those whose last bit falls into a later span of pageBits than
// the last bit of the row before; keeps the scan's last value as 'total'.
struct MarkPageStarts {
  RowBits rowBits;
  int64_t pageBits;
  int32_t numRows;
  bool* starts;
  int64_t* total;

  __device__ void operator()(int32_t i, int64_t cumulative) const {
    const int64_t previous = cumulative - rowBits(i);
    starts[i] =
        i > 0 && (cumulative - 1) / pageBits != (previous - 1) / pageBits;
    if (i == numRows - 1) {
      *total = cumulative;
    }
  }
};

// One block row (blockIdx.y) per task, grid-stride over the task's bytes.
__global__ void nullBytesKernel(
    const NullBytesTask* tasks,
    int32_t taskOffset) {
  const NullBytesTask t = tasks[taskOffset + blockIdx.y];
  const int32_t numBytes = cudf::util::div_rounding_up_safe(t.numRows, 8);
  for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < numBytes;
       j += gridDim.x * blockDim.x) {
    uint8_t out = 0;
    const int32_t row0 = j << 3;
    const int32_t count = min(8, t.numRows - row0);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (k < count &&
          !cudf::bit_is_set(
              t.mask, static_cast<cudf::size_type>(t.firstBit + row0 + k))) {
        out |= static_cast<uint8_t>(0x80u >> k);
      }
    }
    t.dst[j] = out;
  }
}

// Flags a range whose string bytes exceed what an int32 offset holds.
__global__ void stringOffsetsKernel(
    const StringOffsetsTask* tasks,
    int32_t taskOffset,
    int32_t* overflow) {
  const StringOffsetsTask t = tasks[taskOffset + blockIdx.y];
  const int64_t first = t.offsets.firstIndex + t.firstRow;
  const int64_t start = offsetAt(t.offsets, first);
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < t.numRows;
       i += gridDim.x * blockDim.x) {
    const int64_t bytes = offsetAt(t.offsets, first + 1 + i) - start;
    if (bytes > INT32_MAX) {
      *overflow = 1;
    }
    // The destination follows a null bitmap, so it is not int32 aligned.
    const auto offset = static_cast<int32_t>(bytes);
    cuda::std::memcpy(
        t.dst + 4 * static_cast<int64_t>(i), &offset, sizeof(offset));
  }
}

struct GatherOffset {
  StringOffsets offsets;

  __device__ int64_t operator()(int64_t row) const {
    return offsetAt(offsets, offsets.firstIndex + row);
  }
};

// Exactly one of mul, div is 1. Flags a product outside int64.
struct ToMillis {
  int64_t mul;
  int64_t div;
  int32_t* overflow;

  __device__ int64_t operator()(int64_t v) const {
    if (mul != 1) {
      const auto product = cuda::mul_overflow(v, mul);
      if (product.overflow) {
        *overflow = 1;
        return 0;
      }
      return product.value;
    }
    // Floor division, as Timestamp::toMillis() rounds; div == 1 passes v.
    const int64_t q = v / div;
    return (v % div != 0 && v < 0) ? q - 1 : q;
  }
};

// The three views of a CopyTask that cub::DeviceMemcpy::Batched reads.
struct CopySource {
  __device__ const uint8_t* operator()(const CopyTask& t) const {
    return t.src;
  }
};

struct CopyDestination {
  __device__ uint8_t* operator()(const CopyTask& t) const {
    return t.dst;
  }
};

struct CopySize {
  __device__ size_t operator()(const CopyTask& t) const {
    return t.size;
  }
};

struct ValidPredicate {
  const cudf::bitmask_type* mask;
  int64_t firstBit;

  __device__ bool operator()(int32_t i) const {
    return cudf::bit_is_set(mask, static_cast<cudf::size_type>(firstBit + i));
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
      rmm::exec_policy_nosync(stream),
      src,
      src + numRows,
      thrust::counting_iterator<int32_t>(0),
      dst,
      ValidPredicate{mask, firstBit});
  return static_cast<int32_t>(end - dst);
}

uint32_t gridXFor(int64_t maxItems) {
  const int64_t blocks =
      cudf::util::div_rounding_up_safe<int64_t>(maxItems, kBlockSize);
  return static_cast<uint32_t>(std::clamp<int64_t>(blocks, 1, 128));
}

void expectAligned(const void* pointer, int64_t alignment, const char* name) {
  CUDF_EXPECTS(
      reinterpret_cast<uintptr_t>(pointer) % alignment == 0,
      std::string(name) + " must be aligned to the element width");
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
  CUDF_EXPECTS(pageBytes > 0, "pageBytes must be positive");
  CUDF_EXPECTS(
      pageBytes <= std::numeric_limits<int64_t>::max() / 8,
      "pageBytes must fit in bits");
  CUDF_EXPECTS(maxRows > 0, "maxRows must be positive");
  CUDF_EXPECTS(fixedBitsPerRow >= 0, "fixedBitsPerRow must not be negative");
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

  // The scan of the row bits is consumed as it is produced: one flag per row
  // and the last value, not the 8 bytes per row of the scan itself. The
  // buffers are allocated before the upload of 'strings', which the function
  // reads back from the device below, so that nothing can throw while the
  // stream still has the caller's span to read.
  rmm::device_uvector<bool> starts(numRows, stream, mr);
  rmm::device_scalar<int64_t> total(stream, mr);
  auto deviceStrings = cudf::detail::make_device_uvector_async(
      cudf::host_span<StringOffsets const>{strings.data(), strings.size()},
      stream,
      mr);
  const RowBits rowBits{
      fixedBitsPerRow,
      deviceStrings.data(),
      static_cast<int32_t>(strings.size())};
  thrust::transform_inclusive_scan(
      rmm::exec_policy_nosync(stream, mr),
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(numRows),
      cuda::make_tabulate_output_iterator(
          MarkPageStarts{
              rowBits, pageBits, numRows, starts.data(), total.data()}),
      rowBits,
      cuda::std::plus<int64_t>{});
  // Page k holds the rows whose last bit lies in (k * pageBits, (k + 1) *
  // pageBits], so a page exceeds pageBits by less than one row and a row
  // larger than a page forms one page. The scan never decreases, since no row
  // costs less than nothing, so the page of its last value bounds the number
  // of cuts, and they take that much space rather than one slot per row.
  const int64_t totalBits = total.value(stream);
  const int64_t maxCuts = std::min<int64_t>(
      totalBits > 0 ? (totalBits - 1) / pageBits : 0,
      static_cast<int64_t>(numRows) - 1);
  rmm::device_uvector<int32_t> cuts(maxCuts, stream, mr);
  auto cutsEnd = thrust::copy_if(
      rmm::exec_policy_nosync(stream, mr),
      thrust::counting_iterator<int32_t>(1),
      thrust::counting_iterator<int32_t>(numRows),
      starts.begin() + 1,
      cuts.begin(),
      cuda::std::identity{});
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
  // Nothing below waits for the stream and the caller may free 'tasks' on
  // return, so the upload waits.
  auto deviceTasks = cudf::detail::make_device_uvector(
      cudf::host_span<NullBytesTask const>{tasks.data(), tasks.size()},
      stream,
      mr);
  int64_t maxBytes = 0;
  for (const auto& t : tasks) {
    maxBytes = std::max<int64_t>(
        maxBytes, cudf::util::div_rounding_up_safe<int64_t>(t.numRows, 8));
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

bool fillPrestoStringOffsets(
    std::span<const StringOffsetsTask> tasks,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (tasks.empty()) {
    return true;
  }
  // The flag is cleared on the device, and allocated before the upload of
  // 'tasks', which the read of the flag below waits for.
  rmm::device_scalar<int32_t> overflow(stream, mr);
  overflow.set_value_to_zero_async(stream);
  auto deviceTasks = cudf::detail::make_device_uvector_async(
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
        deviceTasks.data(), static_cast<int32_t>(offset), overflow.data());
    CUDF_CHECK_CUDA(stream.get());
  }
  return overflow.value(stream) == 0;
}

void batchedDeviceCopy(
    std::span<const CopyTask> tasks,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (tasks.empty()) {
    return;
  }
  // One upload for the three arrays cub reads, which it takes from the tasks
  // in place; cub skips a task of zero bytes. The upload waits for the
  // stream, since the caller may free 'tasks' on return.
  auto deviceTasks = cudf::detail::make_device_uvector(
      cudf::host_span<CopyTask const>{tasks.data(), tasks.size()}, stream, mr);
  const auto srcs =
      thrust::make_transform_iterator(deviceTasks.data(), CopySource{});
  const auto dsts =
      thrust::make_transform_iterator(deviceTasks.data(), CopyDestination{});
  const auto sizes =
      thrust::make_transform_iterator(deviceTasks.data(), CopySize{});
  const auto count = static_cast<int64_t>(tasks.size());
  // cudf::detail::batched_memcpy_async would take its scratch from the
  // current device resource; the caller's 'mr' is used here instead.
  size_t tempBytes = 0;
  CUDF_CUDA_TRY(
      cub::DeviceMemcpy::Batched(
          nullptr, tempBytes, srcs, dsts, sizes, count, stream.get()));
  rmm::device_buffer temp(tempBytes, stream, mr);
  CUDF_CUDA_TRY(
      cub::DeviceMemcpy::Batched(
          temp.data(), tempBytes, srcs, dsts, sizes, count, stream.get()));
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
  // The output is allocated before the upload of 'rows', which the read of
  // the result below waits for.
  rmm::device_uvector<int64_t> out(rows.size(), stream, mr);
  auto deviceRows = cudf::detail::make_device_uvector_async(
      cudf::host_span<int64_t const>{rows.data(), rows.size()}, stream, mr);
  thrust::transform(
      rmm::exec_policy_nosync(stream, mr),
      deviceRows.begin(),
      deviceRows.end(),
      out.begin(),
      GatherOffset{offsets});
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
  CUDF_EXPECTS(
      mul >= 1 && div >= 1 && (mul == 1 || div == 1),
      "mul and div must be positive and one of them 1");
  expectAligned(src, sizeof(int64_t), "src");
  expectAligned(dst, sizeof(int64_t), "dst");
  if (numRows == 0) {
    return true;
  }
  rmm::device_scalar<int32_t> overflow(stream, mr);
  overflow.set_value_to_zero_async(stream);
  thrust::transform(
      rmm::exec_policy_nosync(stream, mr),
      src,
      src + numRows,
      dst,
      ToMillis{mul, div, overflow.data()});
  return overflow.value(stream) == 0;
}

void decimalsToJava(
    const __int128_t* src,
    int32_t numRows,
    __int128_t* dst,
    cuda::stream_ref stream) {
  expectAligned(src, sizeof(__int128_t), "src");
  expectAligned(dst, sizeof(__int128_t), "dst");
  if (numRows == 0) {
    return;
  }
  thrust::transform(
      rmm::exec_policy_nosync(stream),
      src,
      src + numRows,
      dst,
      ToJavaDecimal{});
}

int32_t compactValid(
    const uint8_t* src,
    int32_t width,
    const cudf::bitmask_type* mask,
    int64_t firstBit,
    int32_t numRows,
    uint8_t* dst,
    cuda::stream_ref stream) {
  CUDF_EXPECTS(
      width == 1 || width == 2 || width == 4 || width == 8 || width == 16,
      "Unsupported fixed width for Presto page compaction");
  expectAligned(src, width, "src");
  expectAligned(dst, width, "dst");
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
    default:
      return compactValidTyped<Bytes16>(
          reinterpret_cast<const Bytes16*>(src),
          mask,
          firstBit,
          numRows,
          reinterpret_cast<Bytes16*>(dst),
          stream);
  }
}

} // namespace facebook::velox::cudf_velox::detail
