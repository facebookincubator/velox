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
#include "velox/experimental/cudf/serializers/PrestoPageDeviceKernels.h"

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
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace facebook::velox::cudf_velox::detail {
namespace {

constexpr int kBlockSize = 256;
constexpr size_t kMaxGridY = 65535;

// Serialized bits of one row: the fixed part plus 8 bits per string byte.
struct RowBits {
  int64_t fixedBits;
  const StringOffsets* strings;
  int32_t numStrings;

  __device__ int64_t operator()(int32_t row) const {
    int64_t stringBytes = 0;
    for (int32_t s = 0; s < numStrings; ++s) {
      const cudf::detail::input_offsetalator offsets(
          strings[s].offsets,
          strings[s].offsetsType,
          static_cast<cudf::size_type>(strings[s].firstIndex));
      stringBytes += offsets[row + 1] - offsets[row];
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

// The 32 mask bits from 'begin' + 32 * 'word' on, as
// cudf::detail::get_mask_offset_word reads them; its header needs
// --extended-lambda, which the cuDF targets do not enable. Reads past 'end'
// only within the word that holds it.
__device__ __forceinline__ uint32_t maskWord(
    const cudf::bitmask_type* mask,
    int32_t word,
    cudf::size_type begin,
    cudf::size_type end) {
  const auto index = word + cudf::word_index(begin);
  const auto next = cudf::word_index(end - 1) > index ? mask[index + 1]
                                                      : cudf::bitmask_type{0};
  return __funnelshift_r(mask[index], next, begin);
}

// One block row (blockIdx.y) per task, grid-stride over 32-row words. A mask
// word becomes four Presto bytes: inverted, since Presto marks the nulls,
// reversed within each byte, since Presto is MSB first, and cut at the range's
// last byte.
__global__ void nullBytesKernel(
    const NullBytesTask* tasks,
    int32_t taskOffset) {
  const NullBytesTask t = tasks[taskOffset + blockIdx.y];
  const auto begin = static_cast<cudf::size_type>(t.firstBit);
  const auto end = static_cast<cudf::size_type>(t.firstBit + t.numRows);
  const int32_t numBytes = cudf::util::div_rounding_up_safe(t.numRows, 8);
  const int32_t numWords = cudf::util::div_rounding_up_safe(t.numRows, 32);
  for (int32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < numWords;
       w += gridDim.x * blockDim.x) {
    const int32_t rows = min(32, t.numRows - 32 * w);
    const uint32_t inRange = rows == 32 ? ~0u : (1u << rows) - 1;
    const uint32_t nulls = ~maskWord(t.mask, w, begin, end) & inRange;
    const uint32_t out = __byte_perm(__brev(nulls), 0, 0x0123);
    cuda::std::memcpy(t.dst + 4 * w, &out, min(4, numBytes - 4 * w));
  }
}

// Flags a range whose string bytes exceed what an int32 offset holds.
__global__ void stringOffsetsKernel(
    const StringOffsetsTask* tasks,
    int32_t taskOffset,
    int32_t* overflow) {
  const StringOffsetsTask t = tasks[taskOffset + blockIdx.y];
  const cudf::detail::input_offsetalator offsets(
      t.offsets.offsets,
      t.offsets.offsetsType,
      static_cast<cudf::size_type>(t.offsets.firstIndex + t.firstRow));
  const int64_t start = offsets[0];
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < t.numRows;
       i += gridDim.x * blockDim.x) {
    const int64_t bytes = offsets[i + 1] - start;
    if (bytes > INT32_MAX) {
      *overflow = 1;
    }
    // The destination follows a null bitmap, so it is not int32 aligned.
    const auto offset = static_cast<int32_t>(bytes);
    cuda::std::memcpy(
        t.dst + 4 * static_cast<int64_t>(i), &offset, sizeof(offset));
  }
}

// One of mul, div is 1. Flags a product outside int64.
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

uint32_t gridXFor(int64_t maxItems) {
  const int64_t blocks =
      cudf::util::div_rounding_up_safe<int64_t>(maxItems, kBlockSize);
  return static_cast<uint32_t>(std::clamp<int64_t>(blocks, 1, 128));
}

} // namespace

std::vector<int64_t> device::prestoPageBoundaries(
    int32_t numRows,
    int64_t fixedBitsPerRow,
    std::span<const StringOffsets> strings,
    int64_t pageBytes,
    int64_t maxRows,
    cuda::stream_ref stream) {
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
  // upload of 'strings' waits for the stream: the scan allocates its scratch
  // and can throw before the stream has read the caller's span.
  auto mr = get_temp_mr();
  auto deviceStrings = cudf::detail::make_device_uvector(
      cudf::host_span<StringOffsets const>{strings.data(), strings.size()},
      stream,
      mr);
  rmm::device_uvector<bool> starts(numRows, stream, mr);
  rmm::device_scalar<int64_t> total(stream, mr);
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
    cuda::stream_ref stream) {
  if (tasks.empty()) {
    return;
  }
  // Nothing below waits for the stream and the caller may free 'tasks' on
  // return, so the upload waits.
  auto deviceTasks = cudf::detail::make_device_uvector(
      cudf::host_span<NullBytesTask const>{tasks.data(), tasks.size()},
      stream,
      get_temp_mr());
  int64_t maxWords = 0;
  for (const auto& t : tasks) {
    maxWords = std::max<int64_t>(
        maxWords, cudf::util::div_rounding_up_safe<int64_t>(t.numRows, 32));
  }
  const auto gridX = gridXFor(maxWords);
  for (size_t offset = 0; offset < tasks.size(); offset += kMaxGridY) {
    const auto count =
        static_cast<uint32_t>(std::min(kMaxGridY, tasks.size() - offset));
    nullBytesKernel<<<dim3(gridX, count), kBlockSize, 0, stream.get()>>>(
        deviceTasks.data(), static_cast<int32_t>(offset));
    CUDF_CHECK_CUDA(stream.get());
  }
}

bool device::fillPrestoStringOffsets(
    std::span<const StringOffsetsTask> tasks,
    cuda::stream_ref stream) {
  if (tasks.empty()) {
    return true;
  }
  // The flag is cleared on the device, and allocated before the upload of
  // 'tasks', which the read of the flag below waits for.
  auto mr = get_temp_mr();
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
    cuda::stream_ref stream) {
  if (tasks.empty()) {
    return;
  }
  // One upload for the three arrays cub reads, which it takes from the tasks
  // in place; cub skips a task of zero bytes. The upload waits for the
  // stream, since the caller may free 'tasks' on return.
  auto mr = get_temp_mr();
  auto deviceTasks = cudf::detail::make_device_uvector(
      cudf::host_span<CopyTask const>{tasks.data(), tasks.size()}, stream, mr);
  const auto srcs =
      thrust::make_transform_iterator(deviceTasks.data(), CopySource{});
  const auto dsts =
      thrust::make_transform_iterator(deviceTasks.data(), CopyDestination{});
  const auto sizes =
      thrust::make_transform_iterator(deviceTasks.data(), CopySize{});
  const auto count = static_cast<int64_t>(tasks.size());
  // cudf::detail::batched_memcpy_async takes its scratch from
  // cudf::get_current_device_resource_ref(), which CudfNoDefaults.h rejects
  // in this tree, so cub is called directly with the same scratch resource.
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

std::vector<int64_t> device::gatherStringOffsets(
    const StringOffsets& offsets,
    std::span<const int64_t> rows,
    cuda::stream_ref stream) {
  if (rows.empty()) {
    return {};
  }
  // The output is allocated before the upload of 'rows', which the read of
  // the result below waits for.
  auto mr = get_temp_mr();
  rmm::device_uvector<int64_t> out(rows.size(), stream, mr);
  auto deviceRows = cudf::detail::make_device_uvector_async(
      cudf::host_span<int64_t const>{rows.data(), rows.size()}, stream, mr);
  thrust::gather(
      rmm::exec_policy_nosync(stream, mr),
      deviceRows.begin(),
      deviceRows.end(),
      cudf::detail::input_offsetalator(
          offsets.offsets,
          offsets.offsetsType,
          static_cast<cudf::size_type>(offsets.firstIndex)),
      out.begin());
  return cudf::detail::make_std_vector(
      cudf::device_span<int64_t const>{out.data(), out.size()}, stream);
}

bool device::timestampsToMillis(
    const int64_t* src,
    int32_t numRows,
    int64_t mul,
    int64_t div,
    int64_t* dst,
    cuda::stream_ref stream) {
  if (numRows == 0) {
    return true;
  }
  auto mr = get_temp_mr();
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

void device::decimalsToJava(
    const __int128_t* src,
    int32_t numRows,
    __int128_t* dst,
    cuda::stream_ref stream) {
  if (numRows == 0) {
    return;
  }
  thrust::transform(
      rmm::exec_policy_nosync(stream, get_temp_mr()),
      src,
      src + numRows,
      dst,
      ToJavaDecimal{});
}

} // namespace facebook::velox::cudf_velox::detail
