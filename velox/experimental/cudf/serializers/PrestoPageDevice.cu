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
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/memory_resource>
#include <cuda/numeric>
#include <cuda/std/cstring>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <thrust/gather.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstddef>
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
    uint32_t word,
    cudf::size_type begin,
    cudf::size_type end) {
  const auto index = word + cudf::word_index(begin);
  const auto last = static_cast<uint32_t>(cudf::word_index(end - 1));
  const auto next = last > index ? mask[index + 1] : cudf::bitmask_type{0};
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
  // The indices stay unsigned, as the kernel indices they derive from.
  const auto numRows = static_cast<uint32_t>(t.numRows);
  const auto numBytes = cudf::detail::div_rounding_up_safe(numRows, 8u);
  const auto numWords = cudf::detail::div_rounding_up_safe(numRows, 32u);
  for (auto w = blockIdx.x * blockDim.x + threadIdx.x; w < numWords;
       w += gridDim.x * blockDim.x) {
    const auto rows = min(32u, numRows - 32 * w);
    const uint32_t inRange = rows == 32 ? ~0u : (1u << rows) - 1;
    const uint32_t nulls = ~maskWord(t.mask, w, begin, end) & inRange;
    const uint32_t out = __byte_perm(__brev(nulls), 0, 0x0123);
    cuda::std::memcpy(t.dst + 4 * w, &out, min(4u, numBytes - 4 * w));
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
  const auto numRows = static_cast<uint32_t>(t.numRows);
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < numRows;
       i += gridDim.x * blockDim.x) {
    const int64_t bytes = offsets[i + 1] - start;
    if (bytes > INT32_MAX) {
      *overflow = 1;
    }
    // The destination follows a null bitmap, so it is not int32 aligned.
    const auto offset = static_cast<int32_t>(bytes);
    cuda::std::memcpy(t.dst + int64_t{4} * i, &offset, sizeof(offset));
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
      cudf::detail::div_rounding_up_safe<int64_t>(maxItems, kBlockSize);
  return static_cast<uint32_t>(std::clamp<int64_t>(blocks, 1, 128));
}

// A device copy of 'values' for a function that does not wait for the stream:
// pinned memory, which a device-accessible span marks, is read in stream
// order, as the header says; pageable memory is read before this returns,
// since the caller may free it.
template <typename T>
cuda::device_buffer<T> toDevice(
    cudf::host_span<const T> values,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto result = cudf::detail::make_device_buffer_async(
      values, stream, cudf::memory_resources{mr, mr});
  if (!values.is_device_accessible()) {
    cudf::detail::sync_stream(stream);
  }
  return result;
}

// cub's execution environment: the stream and the scratch resource. The prop
// and env template arguments are spelled out because nvcc 12.9 does not
// deduce them here.
using StreamProp =
    cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>;
using MrProp = cuda::std::execution::
    prop<cuda::mr::get_memory_resource_t, rmm::device_async_resource_ref>;
using CubEnv = cuda::std::execution::env<StreamProp, MrProp>;

CubEnv cubEnv(cuda::stream_ref stream, rmm::device_async_resource_ref mr) {
  return CubEnv{
      StreamProp{cuda::get_stream_t{}, stream},
      MrProp{cuda::mr::get_memory_resource_t{}, mr}};
}

} // namespace

std::vector<int64_t> device::prestoPageBoundaries(
    int32_t numRows,
    int64_t fixedBitsPerRow,
    cudf::host_span<const StringOffsets> strings,
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
  // and the last value, not the 8 bytes per row of the scan itself. Its
  // scratch is allocated before the upload of 'strings', as hash_join.cu
  // allocates cub's, so that no allocation can fail while the stream still
  // has to read the caller's span; only the scan launch sits between the
  // upload and the read of 'total', which waits for it.
  auto mr = get_temp_mr();
  cuda::device_buffer<StringOffsets> deviceStrings(
      stream, mr, strings.size(), cuda::no_init);
  cuda::device_buffer<bool> starts(stream, mr, numRows, cuda::no_init);
  rmm::device_scalar<int64_t> total(stream, mr);
  const RowBits rowBits{
      fixedBitsPerRow,
      deviceStrings.data(),
      static_cast<int32_t>(strings.size())};
  const auto rowBitsOf = cuda::make_transform_iterator(
      cuda::counting_iterator<int32_t>(0), rowBits);
  const auto pageStarts = cuda::make_tabulate_output_iterator(
      MarkPageStarts{rowBits, pageBits, numRows, starts.data(), total.data()});
  size_t scanBytes = 0;
  CUDF_CUDA_TRY(
      cub::DeviceScan::InclusiveScan(
          nullptr,
          scanBytes,
          rowBitsOf,
          pageStarts,
          cuda::std::plus<int64_t>{},
          numRows,
          stream.get()));
  cuda::device_buffer<std::byte> scanScratch(
      stream, mr, scanBytes, cuda::no_init);
  cudf::detail::cuda_memcpy_async<StringOffsets>(
      cudf::device_span<StringOffsets>{
          deviceStrings.data(), deviceStrings.size()},
      strings,
      stream);
  CUDF_CUDA_TRY(
      cub::DeviceScan::InclusiveScan(
          scanScratch.data(),
          scanBytes,
          rowBitsOf,
          pageStarts,
          cuda::std::plus<int64_t>{},
          numRows,
          stream.get()));
  // Page k holds the rows whose last bit lies in (k * pageBits, (k + 1) *
  // pageBits], so a page exceeds pageBits by less than one row and a row
  // larger than a page forms one page. The scan never decreases, since no row
  // costs less than nothing, so the page of its last value bounds the number
  // of cuts, and they take that much space rather than one slot per row.
  const int64_t totalBits = total.value(stream);
  const int64_t maxCuts = std::min<int64_t>(
      totalBits > 0 ? (totalBits - 1) / pageBits : 0,
      static_cast<int64_t>(numRows) - 1);
  // The cut count lands in front of the cuts, so one copy brings both back:
  // cub leaves the count on the device, where thrust::copy_if would wait for
  // it. A single page has no cuts to fetch.
  if (maxCuts > 0) {
    cuda::device_buffer<int32_t> cuts(stream, mr, maxCuts + 1, cuda::no_init);
    CUDF_CUDA_TRY(
        cub::DeviceSelect::Flagged(
            cuda::counting_iterator<int32_t>(1),
            starts.data() + 1,
            cuts.data() + 1,
            cuts.data(),
            numRows - 1,
            cubEnv(stream, mr)));
    const auto hostCuts = cudf::detail::make_std_vector(
        cudf::device_span<int32_t const>{cuts.data(), cuts.size()}, stream);
    bounds.insert(
        bounds.end(), hostCuts.begin() + 1, hostCuts.begin() + 1 + hostCuts[0]);
  }
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
    cudf::host_span<const NullBytesTask> tasks,
    cuda::stream_ref stream) {
  if (tasks.empty()) {
    return;
  }
  // Nothing below waits for the stream, so a pageable 'tasks' is read before
  // the upload returns.
  auto deviceTasks = toDevice(tasks, stream, get_temp_mr());
  int64_t maxWords = 0;
  for (const auto& t : tasks) {
    maxWords = std::max<int64_t>(
        maxWords, cudf::detail::div_rounding_up_safe<int64_t>(t.numRows, 32));
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
    cudf::host_span<const StringOffsetsTask> tasks,
    cuda::stream_ref stream) {
  if (tasks.empty()) {
    return true;
  }
  // The flag is cleared on the device, and allocated before the upload of
  // 'tasks', which the read of the flag below waits for.
  auto mr = get_temp_mr();
  rmm::device_scalar<int32_t> overflow(stream, mr);
  overflow.set_value_to_zero_async(stream);
  auto deviceTasks = cudf::detail::make_device_buffer_async(
      tasks, stream, cudf::memory_resources{mr, mr});
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
    cudf::host_span<const CopyTask> tasks,
    cuda::stream_ref stream) {
  if (tasks.empty()) {
    return;
  }
  // One upload for the three arrays cub reads, which it takes from the tasks
  // in place; cub skips a task of zero bytes. Nothing below waits for the
  // stream, so a pageable 'tasks' is read before the upload returns.
  auto mr = get_temp_mr();
  auto deviceTasks = toDevice(tasks, stream, mr);
  const auto srcs = cuda::make_transform_iterator(
      deviceTasks.data(), cuda::std::mem_fn(&CopyTask::src));
  const auto dsts = cuda::make_transform_iterator(
      deviceTasks.data(), cuda::std::mem_fn(&CopyTask::dst));
  const auto sizes = cuda::make_transform_iterator(
      deviceTasks.data(), cuda::std::mem_fn(&CopyTask::size));
  const auto count = static_cast<int64_t>(tasks.size());
  // cudf::detail::batched_memcpy_async takes its scratch from
  // cudf::get_current_device_resource_ref(), which CudfNoDefaults.h rejects
  // in this tree, so cub is called directly with the same scratch resource.
  CUDF_CUDA_TRY(
      cub::DeviceMemcpy::Batched(srcs, dsts, sizes, count, cubEnv(stream, mr)));
  CUDF_CHECK_CUDA(stream.get());
}

std::vector<int64_t> device::gatherStringOffsets(
    const StringOffsets& offsets,
    cudf::host_span<const int64_t> rows,
    cuda::stream_ref stream) {
  if (rows.empty()) {
    return {};
  }
  // The output is allocated before the upload of 'rows', which the read of
  // the result below waits for.
  auto mr = get_temp_mr();
  cuda::device_buffer<int64_t> out(stream, mr, rows.size(), cuda::no_init);
  auto deviceRows = cudf::detail::make_device_buffer_async(
      rows, stream, cudf::memory_resources{mr, mr});
  thrust::gather(
      rmm::exec_policy_nosync(stream, mr),
      deviceRows.data(),
      deviceRows.data() + deviceRows.size(),
      cudf::detail::input_offsetalator(
          offsets.offsets,
          offsets.offsetsType,
          static_cast<cudf::size_type>(offsets.firstIndex)),
      out.data());
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
