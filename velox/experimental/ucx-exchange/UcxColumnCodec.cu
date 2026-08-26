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
#include "velox/experimental/ucx-exchange/UcxColumnCodec.h"
#include "velox/experimental/ucx-exchange/UcxCompression.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/utils/StackDeviceMemory.h"

namespace facebook::velox::ucx_exchange {
namespace {

constexpr int kProbBits = 10;
// Racecar default: correctness is gated end-to-end by result validation;
// frame checksums (two extra full passes) are a debug knob only.
constexpr bool kUseChecksum = false;
constexpr std::size_t kMinTypedElems = 4096;
// The inverse dictionary is negligible only for large exchange regions. The
// full-data validation pass must also amortize its launch/readback cost.
constexpr std::size_t kMinDictPforElems = 8u << 20;
// General freq-PFOR pays a fixed 16-bit histogram/rank-build cost. Regions
// below this floor are better served by the established FOR/delta paths.
constexpr std::size_t kMinFreqPforElems = 1u << 20;
constexpr uint32_t kFreqPforCodeLimit = 1u << 16;
constexpr uint32_t kFreqPforHistogramReplicas = 64;
constexpr uint32_t kFreqPforMedianSample = 1u << 18;
constexpr double kFreqPforMinTop256Mass = 0.50;
// A general dictionary candidate must buy at least this much payload space
// over the established candidate to cover its gather/patch decode work.
constexpr double kFreqPforSelectionRatio = 0.90;
// Three uint16 dictionary entries fit in a positive int64 descriptor word.
constexpr std::size_t kDictionaryCodesPerWord = 3;

// DietGPU kernels use word loads; every device pointer handed to them must
// be 16-byte aligned. Planes use an aligned stride; wire segments are placed
// at 16-byte boundaries (true sizes travel in descriptors, walkers round).
inline std::size_t roundUp16(std::size_t v) {
  return (v + 15) & ~static_cast<std::size_t>(15);
}

void recordRegionStats(
    PackedCompressResult::Stats& stats,
    const EncodedRegion& region,
    std::size_t candidateBytes) {
  PackedCompressResult::RegionStats* target{nullptr};
  switch (region.codec) {
    case RegionCodec::kRaw:
      target = &stats.raw;
      break;
    case RegionCodec::kByteRans:
      target = &stats.byteRans;
      break;
    case RegionCodec::kFor:
      target = &stats.frameOfReference;
      break;
    case RegionCodec::kDeltaFor:
      target = &stats.deltaFrameOfReference;
      break;
    case RegionCodec::kDictPfor:
      target = &stats.dictionaryPfor;
      break;
    case RegionCodec::kFreqPfor:
      target = &stats.frequencyPfor;
      break;
    case RegionCodec::kDeltaFreqPfor:
      target = &stats.deltaFrequencyPfor;
      break;
    default:
      throw std::runtime_error(
          "ucx-exchange column codec: unknown region codec in telemetry");
  }
  ++target->regions;
  target->inputBytes += static_cast<std::size_t>(region.rawBytes);
  target->candidateBytes += roundUp16(candidateBytes);
}

inline uint32_t alignedStride(uint32_t n) {
  return (n + 15u) & ~15u;
}
constexpr std::size_t kMinResidualBytes = 1u << 16;

#define UCX_CUDA_CHECK(expr)                                 \
  do {                                                       \
    cudaError_t err = (expr);                                \
    if (err != cudaSuccess) {                                \
      throw std::runtime_error(                              \
          fmt::format(                                       \
              "CUDA error in ucx-exchange column codec: {}", \
              cudaGetErrorString(err)));                     \
    }                                                        \
  } while (0)

// Per-call DietGPU scratch arena backed by rmm (stream-ordered): safe under
// concurrent encode/decode on different streams, no shared state.
constexpr std::size_t kPlaneArenaBytes = 512u << 20;

struct PlaneArena {
  rmm::device_buffer buffer;
  dietgpu::StackDeviceMemory stack;
  explicit PlaneArena(rmm::cuda_stream_view stream)
      : buffer(kPlaneArenaBytes, stream),
        stack(
            [] {
              int device = 0;
              cudaGetDevice(&device);
              return device;
            }(),
            buffer.data(),
            kPlaneArenaBytes) {}
};

// Subtracts base and splits into w byte planes (SoA, plane-major).
template <typename T>
__global__ void subSplitKernel(
    const T* values,
    int64_t base,
    uint8_t* planes,
    uint32_t n,
    uint32_t stride,
    int width) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  uint64_t adjusted =
      static_cast<uint64_t>(static_cast<int64_t>(values[i]) - base);
  for (int k = 0; k < width; ++k) {
    planes[static_cast<uint64_t>(k) * stride + i] =
        static_cast<uint8_t>((adjusted >> (8 * k)) & 0xff);
  }
}

template <typename T>
__global__ void gatherEvenSampleKernel(
    const T* __restrict__ values,
    T* __restrict__ sample,
    uint32_t n,
    uint32_t sampleSize) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sampleSize) {
    return;
  }
  const uint64_t source =
      static_cast<uint64_t>(i) * static_cast<uint64_t>(n) / sampleSize;
  sample[i] = values[source];
}

// Exact 16-bit-window histogram with striped replicas. Exceptions are counted
// once per warp rather than contending on a single counter for every row.
// This pass does not materialize a uint16 code array. The remap pass rereads T
// directly, saving one full write and read pair.
template <typename T>
__global__ void frequencyHistogramKernel(
    const T* __restrict__ values,
    int64_t base,
    uint32_t* __restrict__ replicatedHistogram,
    uint32_t* __restrict__ exceptionCount,
    uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t adjusted = 0;
  bool exception = false;
  if (i < n) {
    adjusted = static_cast<uint64_t>(static_cast<int64_t>(values[i])) -
        static_cast<uint64_t>(base);
    exception = adjusted >= kFreqPforCodeLimit;
    if (!exception) {
      const uint32_t replica = blockIdx.x % kFreqPforHistogramReplicas;
      atomicAdd(
          &replicatedHistogram
              [static_cast<uint64_t>(replica) * kFreqPforCodeLimit +
               static_cast<uint16_t>(adjusted)],
          1u);
    }
  }
  const uint32_t active = __activemask();
  const uint32_t exceptions = __ballot_sync(active, exception);
  const uint32_t lane = threadIdx.x & 31u;
  if (exceptions != 0 && lane == static_cast<uint32_t>(__ffs(exceptions) - 1)) {
    atomicAdd(exceptionCount, static_cast<uint32_t>(__popc(exceptions)));
  }
}

__global__ void reduceFrequencyHistogramKernel(
    const uint32_t* __restrict__ replicatedHistogram,
    uint32_t* __restrict__ histogram) {
  const uint32_t code = blockIdx.x * blockDim.x + threadIdx.x;
  if (code >= kFreqPforCodeLimit) {
    return;
  }
  uint32_t count = 0;
#pragma unroll 4
  for (uint32_t replica = 0; replica < kFreqPforHistogramReplicas; ++replica) {
    count += replicatedHistogram
        [static_cast<uint64_t>(replica) * kFreqPforCodeLimit + code];
  }
  histogram[code] = count;
}

// Merge-free frequency ranking. Counts are capped only for ordering; every
// present code still receives a unique compact rank.
__global__ void frequencyCountHistogramKernel(
    const uint32_t* __restrict__ histogram,
    uint32_t* __restrict__ countHistogram,
    uint32_t* __restrict__ nonzeroFlags) {
  const uint32_t code = blockIdx.x * blockDim.x + threadIdx.x;
  if (code >= kFreqPforCodeLimit) {
    return;
  }
  const uint32_t count = histogram[code];
  nonzeroFlags[code] = count != 0 ? 1u : 0u;
  if (count != 0) {
    atomicAdd(&countHistogram[min(count, kFreqPforCodeLimit - 1)], 1u);
  }
}

__global__ void frequencyRankOffsetsKernel(
    const uint32_t* __restrict__ countPrefix,
    const uint32_t* __restrict__ nonzeroPrefix,
    uint32_t* __restrict__ offsets) {
  const uint32_t count = blockIdx.x * blockDim.x + threadIdx.x;
  if (count >= kFreqPforCodeLimit) {
    return;
  }
  offsets[count] = nonzeroPrefix[kFreqPforCodeLimit - 1] - countPrefix[count];
}

__global__ void assignFrequencyRanksKernel(
    const uint32_t* __restrict__ histogram,
    const uint32_t* __restrict__ nonzeroFlags,
    const uint32_t* __restrict__ nonzeroPrefix,
    uint32_t* __restrict__ offsets,
    uint16_t* __restrict__ ranks,
    uint16_t* __restrict__ inverseRanks) {
  const uint32_t code = blockIdx.x * blockDim.x + threadIdx.x;
  if (code >= kFreqPforCodeLimit) {
    return;
  }
  const uint32_t count = histogram[code];
  uint32_t rank;
  if (count != 0) {
    rank = atomicAdd(&offsets[min(count, kFreqPforCodeLimit - 1)], 1u);
  } else {
    const uint32_t nonzero = nonzeroPrefix[kFreqPforCodeLimit - 1];
    const uint32_t priorNonzero = nonzeroPrefix[code] - nonzeroFlags[code];
    rank = nonzero + code - priorNonzero;
  }
  ranks[code] = static_cast<uint16_t>(rank);
  inverseRanks[rank] = static_cast<uint16_t>(code);
}

// Population covered by the 256 most frequent ranks. This is the property
// that makes the second rank plane mostly zero; unlike a generic entropy
// proxy it directly predicts the byte-plane transformation's payoff.
__global__ void frequencyTop256MassKernel(
    const uint32_t* __restrict__ histogram,
    const uint16_t* __restrict__ inverseRanks,
    uint32_t* __restrict__ topMass) {
  __shared__ uint32_t counts[256];
  const uint32_t rank = threadIdx.x;
  counts[rank] = histogram[inverseRanks[rank]];
  __syncthreads();
  for (uint32_t width = 128; width > 0; width >>= 1) {
    if (rank < width) {
      counts[rank] += counts[rank + width];
    }
    __syncthreads();
  }
  if (rank == 0) {
    *topMass = counts[0];
  }
}

// Remaps directly from T to rank planes and compacts exceptions with one
// global atomic reservation per warp. Position and value use the same slot.
template <typename T>
__global__ void frequencyRemapSplitKernel(
    const T* __restrict__ values,
    int64_t base,
    const uint16_t* __restrict__ ranks,
    uint8_t* __restrict__ planes,
    uint32_t stride,
    int rankWidth,
    uint32_t* __restrict__ exceptionPositions,
    T* __restrict__ exceptionValues,
    uint32_t* __restrict__ exceptionWriteCount,
    uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t adjusted = 0;
  bool exception = false;
  uint16_t rank = 0;
  if (i < n) {
    adjusted = static_cast<uint64_t>(static_cast<int64_t>(values[i])) -
        static_cast<uint64_t>(base);
    exception = adjusted >= kFreqPforCodeLimit;
    if (!exception) {
      rank = ranks[static_cast<uint16_t>(adjusted)];
    }
    planes[i] = static_cast<uint8_t>(rank);
    if (rankWidth == 2) {
      planes[static_cast<uint64_t>(stride) + i] =
          static_cast<uint8_t>(rank >> 8);
    }
  }

  const uint32_t active = __activemask();
  const uint32_t exceptions = __ballot_sync(active, exception);
  if (exceptions == 0) {
    return;
  }
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t leader = static_cast<uint32_t>(__ffs(exceptions) - 1);
  uint32_t warpBase = 0;
  if (lane == leader) {
    warpBase = atomicAdd(
        exceptionWriteCount, static_cast<uint32_t>(__popc(exceptions)));
  }
  warpBase = __shfl_sync(active, warpBase, leader);
  if (exception) {
    const uint32_t lowerLanes = lane == 0 ? 0u : ((1u << lane) - 1u);
    const uint32_t slot = warpBase + __popc(exceptions & lowerLanes);
    exceptionPositions[slot] = i;
    exceptionValues[slot] = values[i];
  }
}

template <typename T>
__global__ void frequencyPforDecodeKernel(
    const uint8_t* __restrict__ planes,
    uint32_t stride,
    int rankWidth,
    const uint16_t* __restrict__ dictionary,
    uint32_t dictionarySize,
    int64_t base,
    T* __restrict__ out,
    uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  uint32_t rank = planes[i];
  if (rankWidth == 2) {
    rank |= static_cast<uint32_t>(planes[static_cast<uint64_t>(stride) + i])
        << 8;
  }
  const uint16_t code = rank < dictionarySize ? dictionary[rank] : 0;
  out[i] = static_cast<T>(base + static_cast<int64_t>(code));
}

template <typename T>
__global__ void patchFrequencyExceptionsKernel(
    const uint32_t* __restrict__ positions,
    const T* __restrict__ values,
    T* __restrict__ out,
    uint32_t count) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[positions[i]] = values[i];
  }
}

// Maps fixed-point whole-unit values to one byte and validates the complete
// region in the same coalesced pass. Quantum is compile-time so nvcc lowers
// division/remainder by decimal powers of ten to multiply/shift sequences.
template <typename T, uint32_t Quantum>
__global__ void remapDecimalLatticeKernel(
    const T* __restrict__ values,
    int64_t base,
    uint32_t rankCount,
    uint32_t* __restrict__ invalidValue,
    uint64_t* __restrict__ maxZigzag,
    uint8_t* __restrict__ ranks,
    uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t zigzag = 0;
  bool valid = true;
  if (i < n) {
    const uint64_t difference =
        static_cast<uint64_t>(static_cast<int64_t>(values[i])) -
        static_cast<uint64_t>(base);
    const uint32_t rank = static_cast<uint32_t>(difference / Quantum);
    valid = difference % Quantum == 0 && rank < rankCount;
    ranks[i] = valid ? static_cast<uint8_t>(rank) : uint8_t{0};
    if (i != 0) {
      const int64_t delta =
          static_cast<int64_t>(values[i]) - static_cast<int64_t>(values[i - 1]);
      zigzag = (static_cast<uint64_t>(delta) << 1) ^
          static_cast<uint64_t>(delta >> 63);
    }
  }

  // One global atomic per block avoids a second full delta materialization and
  // reduction for regions that ultimately use dictionary-PFOR.
  __shared__ uint64_t blockMax[256];
  __shared__ uint32_t blockInvalid;
  if (threadIdx.x == 0) {
    blockInvalid = 0;
  }
  blockMax[threadIdx.x] = zigzag;
  __syncthreads();
  if (i < n && !valid) {
    atomicExch(&blockInvalid, 1u);
  }
  for (uint32_t width = blockDim.x / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      blockMax[threadIdx.x] =
          max(blockMax[threadIdx.x], blockMax[threadIdx.x + width]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(
        reinterpret_cast<unsigned long long*>(maxZigzag),
        static_cast<unsigned long long>(blockMax[0]));
    if (blockInvalid != 0) {
      atomicExch(invalidValue, 1u);
    }
  }
}

// Merges w byte planes and adds base.
template <typename T>
__global__ void recombAddKernel(
    const uint8_t* planes,
    int64_t base,
    T* out,
    uint32_t n,
    uint32_t stride,
    int width) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  uint64_t adjusted = 0;
  for (int k = 0; k < width; ++k) {
    adjusted |=
        static_cast<uint64_t>(planes[static_cast<uint64_t>(k) * stride + i])
        << (8 * k);
  }
  out[i] = static_cast<T>(static_cast<int64_t>(adjusted) + base);
}

template <typename T>
__global__ void dictionaryPforDecodeKernel(
    const uint8_t* __restrict__ ranks,
    const uint16_t* __restrict__ dictionary,
    uint32_t dictionarySize,
    int64_t base,
    T* __restrict__ out,
    uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const uint32_t rank = ranks[i];
  // Malformed frames cannot read past the descriptor dictionary. With frame
  // checksums disabled, substituting base lets end-to-end validation detect
  // corruption without adding a device-to-host synchronization to decode.
  const uint16_t code = rank < dictionarySize ? dictionary[rank] : 0;
  out[i] = static_cast<T>(base + static_cast<int64_t>(code));
}

// zigzag(v[i] - v[i-1]); element 0 gets 0 (first value travels in the
// descriptor).
template <typename T>
__global__ void zigzagDeltaKernel(const T* values, int64_t* out, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  int64_t delta = (i == 0)
      ? 0
      : static_cast<int64_t>(values[i]) - static_cast<int64_t>(values[i - 1]);
  out[i] = static_cast<int64_t>(
      (static_cast<uint64_t>(delta) << 1) ^ static_cast<uint64_t>(delta >> 63));
}

// Un-zigzags in place (int64 zigzag values -> signed deltas).
__global__ void unZigzagKernel(int64_t* values, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  uint64_t z = static_cast<uint64_t>(values[i]);
  values[i] = static_cast<int64_t>((z >> 1) ^ (~(z & 1) + 1));
}

// Adds `first` to every prefix-summed delta and narrows to T.
template <typename T>
__global__ void
finalizeDeltaKernel(const int64_t* summed, int64_t first, T* out, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = static_cast<T>(summed[i] + first);
}

// Pinned host staging for tiny D2H readbacks (pageable D2H pays ~50-100us
// staging latency per copy; pinned is ~5us). One slot set per thread.
struct PinnedStage {
  int64_t* values; // min, max, delta max, and first-value staging slots
  uint32_t* sizes; // up to 64 plane/segment sizes
  PinnedStage() {
    UCX_CUDA_CHECK(cudaHostAlloc(
        reinterpret_cast<void**>(&values),
        4 * sizeof(int64_t),
        cudaHostAllocDefault));
    UCX_CUDA_CHECK(cudaHostAlloc(
        reinterpret_cast<void**>(&sizes),
        64 * sizeof(uint32_t),
        cudaHostAllocDefault));
  }
};
PinnedStage& pinnedStage() {
  static thread_local PinnedStage stage;
  return stage;
}

int planesForRange(uint64_t range) {
  int width = 1;
  while ((range >> (8 * width)) != 0 && width < 8) {
    ++width;
  }
  return width;
}

struct TypedRegion {
  std::size_t offset; // byte offset in blob
  std::size_t elems;
  int width; // element width, 4 or 8
  int32_t typeId{-1};
  int32_t scale{0}; // cuDF fixed-point exponent; zero for non-decimals
};

// Recursively collects fixed-width numeric data-buffer regions.
void collectTypedRegions(
    const cudf::column_view& col,
    const uint8_t* blobBase,
    std::vector<TypedRegion>& out) {
  const auto type = col.type();
  if (col.size() >= static_cast<cudf::size_type>(kMinTypedElems) &&
      cudf::is_fixed_width(type) && col.head<uint8_t>() != nullptr) {
    const int width = cudf::size_of(type);
    if (width == 4 || width == 8) {
      const uint8_t* data =
          col.head<uint8_t>() + static_cast<std::size_t>(col.offset()) * width;
      out.push_back(
          TypedRegion{
              static_cast<std::size_t>(data - blobBase),
              static_cast<std::size_t>(col.size()),
              width,
              static_cast<int32_t>(type.id()),
              cudf::is_fixed_point(type) ? type.scale() : 0});
    }
  }
  for (auto childIndex = 0; childIndex < col.num_children(); ++childIndex) {
    collectTypedRegions(col.child(childIndex), blobBase, out);
  }
}

// Encodes `w` byte planes of n bytes each (plane-major in `planes`) with one
// strided rANS batch; returns compacted output + per-plane sizes.
std::pair<rmm::device_buffer, std::vector<int64_t>> encodePlanes(
    const uint8_t* planes,
    uint32_t n,
    uint32_t stride,
    int width,
    PlaneArena& arena,
    rmm::cuda_stream_view stream) {
  const uint32_t maxComp = alignedStride(dietgpu::getMaxCompressedSize(n));
  rmm::device_buffer scratch(static_cast<std::size_t>(width) * maxComp, stream);
  rmm::device_buffer sizesDev(width * sizeof(uint32_t), stream);
  dietgpu::ansEncodeBatchStride(
      arena.stack,
      dietgpu::ANSCodecConfig(kProbBits, kUseChecksum),
      width,
      planes,
      n,
      stride,
      nullptr,
      scratch.data(),
      maxComp,
      static_cast<uint32_t*>(sizesDev.data()),
      stream.value());
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      pinnedStage().sizes,
      sizesDev.data(),
      width * sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  std::vector<uint32_t> sizes(pinnedStage().sizes, pinnedStage().sizes + width);

  std::size_t total = 0;
  for (auto s : sizes) {
    total += roundUp16(s);
  }
  rmm::device_buffer out(total, stream);
  std::size_t off = 0;
  for (int k = 0; k < width; ++k) {
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(out.data()) + off,
        static_cast<const uint8_t*>(scratch.data()) +
            static_cast<std::size_t>(k) * maxComp,
        sizes[k],
        cudaMemcpyDeviceToDevice,
        stream.value()));
    off += roundUp16(sizes[k]);
  }
  // The packed-chunk caller performs one final stream synchronization before
  // handing its result to UCX. Keeping these copies queued lets the next
  // region's transforms launch without another host-side gap.
  return {std::move(out), std::vector<int64_t>(sizes.begin(), sizes.end())};
}

template <typename T>
int64_t frequencyPforBase(
    const T* values,
    uint32_t n,
    int64_t minimum,
    uint64_t range,
    rmm::cuda_stream_view stream) {
  if (range < kFreqPforCodeLimit) {
    return minimum;
  }

  // A device-side evenly spaced sample keeps the selection cost bounded and
  // avoids moving megabytes through the host merely to obtain a robust center.
  const uint32_t sampleSize = std::min(n, kFreqPforMedianSample);
  rmm::device_buffer sampleIn(sampleSize * sizeof(T), stream);
  rmm::device_buffer sampleOut(sampleSize * sizeof(T), stream);
  constexpr int kThreads = 256;
  const int blocks = (sampleSize + kThreads - 1) / kThreads;
  gatherEvenSampleKernel<T><<<blocks, kThreads, 0, stream.value()>>>(
      values, static_cast<T*>(sampleIn.data()), n, sampleSize);
  UCX_CUDA_CHECK(cudaGetLastError());

  std::size_t sortTempBytes = 0;
  UCX_CUDA_CHECK(
      cub::DeviceRadixSort::SortKeys(
          nullptr,
          sortTempBytes,
          static_cast<T*>(sampleIn.data()),
          static_cast<T*>(sampleOut.data()),
          sampleSize,
          0,
          sizeof(T) * 8,
          stream.value()));
  rmm::device_buffer sortTemp(sortTempBytes, stream);
  UCX_CUDA_CHECK(
      cub::DeviceRadixSort::SortKeys(
          sortTemp.data(),
          sortTempBytes,
          static_cast<T*>(sampleIn.data()),
          static_cast<T*>(sampleOut.data()),
          sampleSize,
          0,
          sizeof(T) * 8,
          stream.value()));
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      pinnedStage().values,
      static_cast<const T*>(sampleOut.data()) + sampleSize / 2,
      sizeof(T),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  const int64_t median =
      static_cast<int64_t>(*reinterpret_cast<T*>(pinnedStage().values));
  if (median < std::numeric_limits<int64_t>::min() +
          static_cast<int64_t>(kFreqPforCodeLimit / 2)) {
    return std::numeric_limits<int64_t>::min();
  }
  return median - static_cast<int64_t>(kFreqPforCodeLimit / 2);
}

struct FrequencyPforEncoding {
  bool used{false};
  rmm::device_buffer payload;
  std::vector<int64_t> segSizes;
  uint32_t dictionarySize{0};
  uint32_t exceptionCount{0};
};

// General adaptive-plane frequency-PFOR: exact replicated histogram,
// merge-free counting-sort ranks, compact inverse dictionary, rANS rank
// planes, and typed exception patching.
template <typename T>
FrequencyPforEncoding tryEncodeFrequencyPfor(
    const T* values,
    uint32_t n,
    int64_t base,
    PlaneArena& arena,
    rmm::cuda_stream_view stream) {
  constexpr int kThreads = 256;
  const int blocks = (n + kThreads - 1) / kThreads;
  const int codeBlocks = (kFreqPforCodeLimit + kThreads - 1) / kThreads;

  rmm::device_buffer replicatedHistogram(
      static_cast<std::size_t>(kFreqPforHistogramReplicas) *
          kFreqPforCodeLimit * sizeof(uint32_t),
      stream);
  rmm::device_buffer histogram(kFreqPforCodeLimit * sizeof(uint32_t), stream);
  rmm::device_buffer exceptionCountDevice(sizeof(uint32_t), stream);
  UCX_CUDA_CHECK(cudaMemsetAsync(
      replicatedHistogram.data(),
      0,
      replicatedHistogram.size(),
      stream.value()));
  UCX_CUDA_CHECK(cudaMemsetAsync(
      exceptionCountDevice.data(), 0, sizeof(uint32_t), stream.value()));
  frequencyHistogramKernel<T><<<blocks, kThreads, 0, stream.value()>>>(
      values,
      base,
      static_cast<uint32_t*>(replicatedHistogram.data()),
      static_cast<uint32_t*>(exceptionCountDevice.data()),
      n);
  reduceFrequencyHistogramKernel<<<codeBlocks, kThreads, 0, stream.value()>>>(
      static_cast<const uint32_t*>(replicatedHistogram.data()),
      static_cast<uint32_t*>(histogram.data()));

  rmm::device_buffer countHistogram(
      kFreqPforCodeLimit * sizeof(uint32_t), stream);
  rmm::device_buffer countPrefix(kFreqPforCodeLimit * sizeof(uint32_t), stream);
  rmm::device_buffer offsets(kFreqPforCodeLimit * sizeof(uint32_t), stream);
  rmm::device_buffer nonzeroFlags(
      kFreqPforCodeLimit * sizeof(uint32_t), stream);
  rmm::device_buffer nonzeroPrefix(
      kFreqPforCodeLimit * sizeof(uint32_t), stream);
  rmm::device_buffer ranks(kFreqPforCodeLimit * sizeof(uint16_t), stream);
  rmm::device_buffer inverseRanks(
      kFreqPforCodeLimit * sizeof(uint16_t), stream);
  rmm::device_buffer topMassDevice(sizeof(uint32_t), stream);
  UCX_CUDA_CHECK(cudaMemsetAsync(
      countHistogram.data(), 0, countHistogram.size(), stream.value()));
  frequencyCountHistogramKernel<<<codeBlocks, kThreads, 0, stream.value()>>>(
      static_cast<const uint32_t*>(histogram.data()),
      static_cast<uint32_t*>(countHistogram.data()),
      static_cast<uint32_t*>(nonzeroFlags.data()));

  std::size_t countScanBytes = 0;
  std::size_t nonzeroScanBytes = 0;
  UCX_CUDA_CHECK(
      cub::DeviceScan::InclusiveSum(
          nullptr,
          countScanBytes,
          static_cast<const uint32_t*>(countHistogram.data()),
          static_cast<uint32_t*>(countPrefix.data()),
          kFreqPforCodeLimit,
          stream.value()));
  UCX_CUDA_CHECK(
      cub::DeviceScan::InclusiveSum(
          nullptr,
          nonzeroScanBytes,
          static_cast<const uint32_t*>(nonzeroFlags.data()),
          static_cast<uint32_t*>(nonzeroPrefix.data()),
          kFreqPforCodeLimit,
          stream.value()));
  rmm::device_buffer scanTemp(
      std::max(countScanBytes, nonzeroScanBytes), stream);
  UCX_CUDA_CHECK(
      cub::DeviceScan::InclusiveSum(
          scanTemp.data(),
          countScanBytes,
          static_cast<const uint32_t*>(countHistogram.data()),
          static_cast<uint32_t*>(countPrefix.data()),
          kFreqPforCodeLimit,
          stream.value()));
  UCX_CUDA_CHECK(
      cub::DeviceScan::InclusiveSum(
          scanTemp.data(),
          nonzeroScanBytes,
          static_cast<const uint32_t*>(nonzeroFlags.data()),
          static_cast<uint32_t*>(nonzeroPrefix.data()),
          kFreqPforCodeLimit,
          stream.value()));
  frequencyRankOffsetsKernel<<<codeBlocks, kThreads, 0, stream.value()>>>(
      static_cast<const uint32_t*>(countPrefix.data()),
      static_cast<const uint32_t*>(nonzeroPrefix.data()),
      static_cast<uint32_t*>(offsets.data()));
  assignFrequencyRanksKernel<<<codeBlocks, kThreads, 0, stream.value()>>>(
      static_cast<const uint32_t*>(histogram.data()),
      static_cast<const uint32_t*>(nonzeroFlags.data()),
      static_cast<const uint32_t*>(nonzeroPrefix.data()),
      static_cast<uint32_t*>(offsets.data()),
      static_cast<uint16_t*>(ranks.data()),
      static_cast<uint16_t*>(inverseRanks.data()));
  frequencyTop256MassKernel<<<1, 256, 0, stream.value()>>>(
      static_cast<const uint32_t*>(histogram.data()),
      static_cast<const uint16_t*>(inverseRanks.data()),
      static_cast<uint32_t*>(topMassDevice.data()));
  UCX_CUDA_CHECK(cudaGetLastError());

  // Both counters are produced on a non-blocking stream. Read them together
  // only after all rank-build work is queued to avoid observing stale values.
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      pinnedStage().sizes + 61,
      topMassDevice.data(),
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      pinnedStage().sizes + 62,
      exceptionCountDevice.data(),
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      pinnedStage().sizes + 63,
      static_cast<const uint32_t*>(nonzeroPrefix.data()) + kFreqPforCodeLimit -
          1,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  const uint32_t topMass = pinnedStage().sizes[61];
  const uint32_t exceptionCount = pinnedStage().sizes[62];
  const uint32_t dictionarySize = pinnedStage().sizes[63];
  if (dictionarySize == 0 || dictionarySize > kFreqPforCodeLimit) {
    return {};
  }
  const uint32_t inWindow = n - exceptionCount;
  if (dictionarySize > 256 &&
      static_cast<double>(topMass) <
          kFreqPforMinTop256Mass * static_cast<double>(inWindow)) {
    return {};
  }

  const std::size_t dictionaryBytes =
      static_cast<std::size_t>(dictionarySize) * sizeof(uint16_t);
  const std::size_t positionBytes =
      static_cast<std::size_t>(exceptionCount) * sizeof(uint32_t);
  const std::size_t exceptionValueBytes =
      static_cast<std::size_t>(exceptionCount) * sizeof(T);
  const std::size_t unavoidableBytes = roundUp16(dictionaryBytes) +
      roundUp16(positionBytes) + roundUp16(exceptionValueBytes);
  const std::size_t rawBytes = static_cast<std::size_t>(n) * sizeof(T);
  if (unavoidableBytes >= rawBytes * kFreqPforSelectionRatio) {
    return {};
  }

  const int rankWidth = dictionarySize <= 256 ? 1 : 2;
  const uint32_t stride = alignedStride(n);
  rmm::device_buffer planes(
      static_cast<std::size_t>(rankWidth) * stride, stream);
  rmm::device_buffer exceptionPositions(positionBytes, stream);
  rmm::device_buffer exceptionValues(exceptionValueBytes, stream);
  rmm::device_buffer exceptionWriteCount(sizeof(uint32_t), stream);
  UCX_CUDA_CHECK(cudaMemsetAsync(
      exceptionWriteCount.data(), 0, sizeof(uint32_t), stream.value()));
  frequencyRemapSplitKernel<T><<<blocks, kThreads, 0, stream.value()>>>(
      values,
      base,
      static_cast<const uint16_t*>(ranks.data()),
      static_cast<uint8_t*>(planes.data()),
      stride,
      rankWidth,
      static_cast<uint32_t*>(exceptionPositions.data()),
      static_cast<T*>(exceptionValues.data()),
      static_cast<uint32_t*>(exceptionWriteCount.data()),
      n);
  UCX_CUDA_CHECK(cudaGetLastError());
  auto [rankPayload, segSizes] = encodePlanes(
      static_cast<const uint8_t*>(planes.data()),
      n,
      stride,
      rankWidth,
      arena,
      stream);

  const std::size_t rankBytes = rankPayload.size();
  const std::size_t totalBytes = rankBytes + roundUp16(dictionaryBytes) +
      roundUp16(positionBytes) + roundUp16(exceptionValueBytes);
  rmm::device_buffer payload(totalBytes, stream);
  std::size_t offset = 0;
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      static_cast<uint8_t*>(payload.data()) + offset,
      rankPayload.data(),
      rankBytes,
      cudaMemcpyDeviceToDevice,
      stream.value()));
  offset += rankBytes;
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      static_cast<uint8_t*>(payload.data()) + offset,
      inverseRanks.data(),
      dictionaryBytes,
      cudaMemcpyDeviceToDevice,
      stream.value()));
  offset += roundUp16(dictionaryBytes);
  if (exceptionCount != 0) {
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(payload.data()) + offset,
        exceptionPositions.data(),
        positionBytes,
        cudaMemcpyDeviceToDevice,
        stream.value()));
    offset += roundUp16(positionBytes);
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(payload.data()) + offset,
        exceptionValues.data(),
        exceptionValueBytes,
        cudaMemcpyDeviceToDevice,
        stream.value()));
  }

  FrequencyPforEncoding out;
  out.used = true;
  out.payload = std::move(payload);
  out.segSizes = std::move(segSizes);
  out.dictionarySize = dictionarySize;
  out.exceptionCount = exceptionCount;
  return out;
}

struct DictionaryPforEncoding {
  bool used{false};
  rmm::device_buffer payload;
  std::vector<uint16_t> dictionary;
};

uint32_t decimalQuantumForScale(int32_t scale) {
  switch (scale) {
    case -1:
      return 10;
    case -2:
      return 100;
    case -3:
      return 1000;
    case -4:
      return 10000;
    default:
      return 0;
  }
}

template <typename T>
void launchDecimalLatticeRemap(
    const T* values,
    uint32_t n,
    int64_t base,
    int32_t scale,
    uint32_t rankCount,
    uint32_t* invalidValue,
    uint64_t* maxZigzag,
    uint8_t* ranks,
    rmm::cuda_stream_view stream) {
  constexpr int kThreads = 256;
  const int blocks =
      static_cast<int>((static_cast<uint64_t>(n) + kThreads - 1) / kThreads);
  switch (scale) {
    case -1:
      remapDecimalLatticeKernel<T, 10><<<blocks, kThreads, 0, stream.value()>>>(
          values, base, rankCount, invalidValue, maxZigzag, ranks, n);
      break;
    case -2:
      remapDecimalLatticeKernel<T, 100>
          <<<blocks, kThreads, 0, stream.value()>>>(
              values, base, rankCount, invalidValue, maxZigzag, ranks, n);
      break;
    case -3:
      remapDecimalLatticeKernel<T, 1000>
          <<<blocks, kThreads, 0, stream.value()>>>(
              values, base, rankCount, invalidValue, maxZigzag, ranks, n);
      break;
    case -4:
      remapDecimalLatticeKernel<T, 10000>
          <<<blocks, kThreads, 0, stream.value()>>>(
              values, base, rankCount, invalidValue, maxZigzag, ranks, n);
      break;
    default:
      throw std::invalid_argument("unsupported decimal lattice scale");
  }
}

// cuDF stores fixed-point values as integer representations. Whole-unit
// values at scale -s lie on a 10^s lattice. Cheap host-side range checks gate
// one coalesced GPU pass that both emits byte ranks and validates every value.
// Any off-lattice value fails closed to the established FOR/delta selector.
template <typename T>
DictionaryPforEncoding tryEncodeDecimalLattice(
    const T* values,
    uint32_t n,
    int64_t base,
    uint64_t range,
    int32_t scale,
    int minDeltaWidth,
    rmm::cuda_stream_view stream) {
  const uint32_t quantum = decimalQuantumForScale(scale);
  if (quantum == 0 || range % quantum != 0) {
    return {};
  }
  const uint64_t rankCount64 = range / quantum + 1;
  if (rankCount64 == 0 || rankCount64 > 256) {
    return {};
  }

  const uint32_t rankCount = static_cast<uint32_t>(rankCount64);
  const uint32_t stride = alignedStride(n);
  rmm::device_buffer ranks(stride, stream);
  // Only the wire-alignment tail is not written by the mapping kernel.
  if (stride > n) {
    UCX_CUDA_CHECK(cudaMemsetAsync(
        static_cast<uint8_t*>(ranks.data()) + n,
        0,
        stride - n,
        stream.value()));
  }
  rmm::device_buffer invalidValue(sizeof(uint32_t), stream);
  rmm::device_buffer maxZigzag(sizeof(uint64_t), stream);
  UCX_CUDA_CHECK(cudaMemsetAsync(
      invalidValue.data(), 0, invalidValue.size(), stream.value()));
  UCX_CUDA_CHECK(
      cudaMemsetAsync(maxZigzag.data(), 0, maxZigzag.size(), stream.value()));
  launchDecimalLatticeRemap(
      values,
      n,
      base,
      scale,
      rankCount,
      static_cast<uint32_t*>(invalidValue.data()),
      static_cast<uint64_t*>(maxZigzag.data()),
      static_cast<uint8_t*>(ranks.data()),
      stream);
  UCX_CUDA_CHECK(cudaGetLastError());
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      pinnedStage().sizes + 63,
      invalidValue.data(),
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      pinnedStage().values,
      maxZigzag.data(),
      sizeof(uint64_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  if (pinnedStage().sizes[63] != 0 ||
      planesForRange(*reinterpret_cast<const uint64_t*>(pinnedStage().values)) <
          minDeltaWidth) {
    return {};
  }

  DictionaryPforEncoding out;
  out.dictionary.resize(rankCount);
  for (uint32_t rank = 0; rank < rankCount; ++rank) {
    out.dictionary[rank] = static_cast<uint16_t>(rank * quantum);
  }
  // A byte rank is already an 8x representation of DECIMAL64. A second rANS
  // pass was measured slower end-to-end despite a better ratio, so ranks are
  // sent directly and decoded with one bandwidth-bound lookup kernel.
  out.payload = std::move(ranks);
  out.used = true;
  return out;
}

// Decodes `segSizes.size()` planes of n bytes each into plane-major scratch.
rmm::device_buffer decodePlanes(
    const uint8_t* src,
    const std::vector<int64_t>& segSizes,
    uint32_t n,
    PlaneArena& arena,
    rmm::cuda_stream_view stream) {
  const auto width = segSizes.size();
  const uint32_t stride = alignedStride(n);
  rmm::device_buffer planes(static_cast<std::size_t>(width) * stride, stream);
  std::vector<const void*> inPtrs(width);
  std::vector<void*> outPtrs(width);
  std::vector<uint32_t> outCaps(width, n);
  std::size_t off = 0;
  for (std::size_t k = 0; k < width; ++k) {
    inPtrs[k] = src + off;
    off += roundUp16(segSizes[k]);
    outPtrs[k] = static_cast<uint8_t*>(planes.data()) + k * stride;
  }
  auto status = dietgpu::ansDecodeBatchPointer(
      arena.stack,
      dietgpu::ANSCodecConfig(kProbBits, kUseChecksum),
      width,
      inPtrs.data(),
      outPtrs.data(),
      outCaps.data(),
      nullptr,
      nullptr,
      stream.value());
  if (status.error != dietgpu::ANSDecodeError::None) {
    throw std::runtime_error("ucx-exchange column codec: plane decode failed");
  }
  // The shared per-chunk arena remains alive until decompressPacked performs
  // one final synchronization. Successive regions use the same stream, so
  // scratch reuse remains ordered without a host wait here.
  return planes;
}

template <typename T>
void encodeTypedRegion(
    const uint8_t* blobBase,
    const TypedRegion& region,
    rmm::cuda_stream_view stream,
    PlaneArena& arena,
    std::vector<EncodedRegion>& regions,
    std::vector<rmm::device_buffer>& payloads,
    bool forOnly = false,
    bool enableAdvancedCodecs = false,
    std::size_t advancedCodecMinBytes = 0) {
  const T* values = reinterpret_cast<const T*>(blobBase + region.offset);
  const uint32_t n = region.elems;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  // Value min/max via cub on rmm temp: stream-ordered, no cudaMalloc device
  // sync (thrust's internal temp allocation stalls against ALL streams).
  rmm::device_buffer minMaxOut(2 * sizeof(T), stream);
  T* minOut = static_cast<T*>(minMaxOut.data());
  T* maxOut = minOut + 1;
  std::size_t minTempBytes = 0;
  std::size_t maxTempBytes = 0;
  UCX_CUDA_CHECK(
      cub::DeviceReduce::Min(
          nullptr, minTempBytes, values, minOut, n, stream.value()));
  UCX_CUDA_CHECK(
      cub::DeviceReduce::Max(
          nullptr, maxTempBytes, values, maxOut, n, stream.value()));
  rmm::device_buffer temp(std::max(minTempBytes, maxTempBytes), stream);
  UCX_CUDA_CHECK(
      cub::DeviceReduce::Min(
          temp.data(), minTempBytes, values, minOut, n, stream.value()));
  UCX_CUDA_CHECK(
      cub::DeviceReduce::Max(
          temp.data(), maxTempBytes, values, maxOut, n, stream.value()));

  // Keep each typed result in its own 64-bit pinned slot. A contiguous
  // 2*sizeof(T) copy places int32 max at byte offset 4, while values + 1 is
  // byte offset 8, causing stale max reads and invalid codec-width choices.
  auto& stage = pinnedStage();
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      stage.values, minOut, sizeof(T), cudaMemcpyDeviceToHost, stream.value()));
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      stage.values + 1,
      maxOut,
      sizeof(T),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  const int64_t base =
      static_cast<int64_t>(*reinterpret_cast<T*>(stage.values));
  const uint64_t forRange = static_cast<uint64_t>(
      static_cast<int64_t>(*reinterpret_cast<T*>(stage.values + 1)) - base);
  const int forWidth = planesForRange(forRange);

  EncodedRegion out;
  out.blobOffset = region.offset;
  out.rawBytes = static_cast<int64_t>(region.elems) * region.width;
  out.elemWidth = region.width;

  const bool enableAdvanced = enableAdvancedCodecs &&
      static_cast<std::size_t>(out.rawBytes) >= advancedCodecMinBytes;
  FrequencyPforEncoding frequency;
  int64_t frequencyBase = base;
  if (!forOnly && enableAdvanced && n >= kMinFreqPforElems && forWidth >= 2) {
    frequencyBase = frequencyPforBase(values, n, base, forRange, stream);
    frequency = tryEncodeFrequencyPfor(values, n, frequencyBase, arena, stream);
  }

  // Retain the older raw-rank decimal lattice as a low-overhead fallback and
  // regression point. General freq-PFOR wins only when its full GPU-wire
  // payload (including dictionary/exceptions) is materially smaller.
  if (!forOnly && enableAdvanced &&
      region.typeId == static_cast<int32_t>(cudf::type_id::DECIMAL64) &&
      n >= kMinDictPforElems && forWidth == 2) {
    // The lattice codec's payload is exactly one aligned byte per row. If the
    // already-valid general candidate clears the selection margin against
    // that theoretical payload, a full lattice validation pass cannot change
    // the decision and is pure overhead.
    if (frequency.used &&
        frequency.payload.size() <=
            alignedStride(n) * kFreqPforSelectionRatio) {
      out.codec = RegionCodec::kFreqPfor;
      out.base = frequencyBase;
      out.segSizes = std::move(frequency.segSizes);
      out.dictionarySize = frequency.dictionarySize;
      out.exceptionCount = frequency.exceptionCount;
      payloads.push_back(std::move(frequency.payload));
      regions.push_back(std::move(out));
      return;
    }
    auto dense = tryEncodeDecimalLattice(
        values, n, base, forRange, region.scale, forWidth, stream);
    if (dense.used) {
      if (frequency.used &&
          frequency.payload.size() <=
              dense.payload.size() * kFreqPforSelectionRatio) {
        out.codec = RegionCodec::kFreqPfor;
        out.base = frequencyBase;
        out.segSizes = std::move(frequency.segSizes);
        out.dictionarySize = frequency.dictionarySize;
        out.exceptionCount = frequency.exceptionCount;
        payloads.push_back(std::move(frequency.payload));
        regions.push_back(std::move(out));
        return;
      }
      out.codec = RegionCodec::kDictPfor;
      out.base = base;
      out.dictionary = std::move(dense.dictionary);
      payloads.push_back(std::move(dense.payload));
      regions.push_back(std::move(out));
      return;
    }
  }

  int deltaWidth = 9; // sentinel: > any forWidth when forOnly
  int64_t firstValue = 0;
  int64_t deltaFrequencyBase = 0;
  FrequencyPforEncoding deltaFrequency;
  rmm::device_buffer deltas(
      forOnly ? 0 : static_cast<std::size_t>(n) * 8, stream);
  auto* deltasPtr = static_cast<int64_t*>(deltas.data());
  if (!forOnly) {
    // Zigzag deltas + their max (min is >= 0 by construction).
    zigzagDeltaKernel<T>
        <<<blocks, threads, 0, stream.value()>>>(values, deltasPtr, n);
    rmm::device_buffer deltaMaxDev(sizeof(int64_t), stream);
    std::size_t dTempBytes = 0;
    cub::DeviceReduce::Max(
        nullptr,
        dTempBytes,
        deltasPtr,
        static_cast<int64_t*>(deltaMaxDev.data()),
        n,
        stream.value());
    rmm::device_buffer dTemp(dTempBytes, stream);
    cub::DeviceReduce::Max(
        dTemp.data(),
        dTempBytes,
        deltasPtr,
        static_cast<int64_t*>(deltaMaxDev.data()),
        n,
        stream.value());
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        stage.values + 2,
        deltaMaxDev.data(),
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        stage.values + 3,
        values,
        sizeof(T),
        cudaMemcpyDeviceToHost,
        stream.value()));
    UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
    const int64_t deltaMax = stage.values[2];
    firstValue =
        static_cast<int64_t>(*reinterpret_cast<const T*>(stage.values + 3));
    deltaWidth = planesForRange(static_cast<uint64_t>(deltaMax));
    if (enableAdvanced && n >= kMinFreqPforElems && deltaWidth >= 2) {
      deltaFrequencyBase =
          frequencyPforBase(deltasPtr, n, int64_t{0}, deltaMax, stream);
      deltaFrequency = tryEncodeFrequencyPfor(
          deltasPtr, n, deltaFrequencyBase, arena, stream);
    }
  }

  const uint32_t stride = alignedStride(n);
  rmm::device_buffer planes(
      static_cast<std::size_t>(std::min(forWidth, deltaWidth)) * stride,
      stream);
  rmm::device_buffer selectedPayload;
  if (deltaWidth < forWidth) {
    out.codec = RegionCodec::kDeltaFor;
    out.base = 0;
    out.first = firstValue;
    subSplitKernel<int64_t><<<blocks, threads, 0, stream.value()>>>(
        deltasPtr,
        0,
        static_cast<uint8_t*>(planes.data()),
        n,
        stride,
        deltaWidth);
    auto [payload, sizes] = encodePlanes(
        static_cast<const uint8_t*>(planes.data()),
        n,
        stride,
        deltaWidth,
        arena,
        stream);
    out.segSizes = std::move(sizes);
    selectedPayload = std::move(payload);
  } else {
    out.codec = RegionCodec::kFor;
    out.base = base;
    subSplitKernel<T><<<blocks, threads, 0, stream.value()>>>(
        values,
        base,
        static_cast<uint8_t*>(planes.data()),
        n,
        stride,
        forWidth);
    auto [payload, sizes] = encodePlanes(
        static_cast<const uint8_t*>(planes.data()),
        n,
        stride,
        forWidth,
        arena,
        stream);
    out.segSizes = std::move(sizes);
    selectedPayload = std::move(payload);
  }

  if (frequency.used &&
      frequency.payload.size() <=
          selectedPayload.size() * kFreqPforSelectionRatio) {
    out.codec = RegionCodec::kFreqPfor;
    out.base = frequencyBase;
    out.first = 0;
    out.segSizes = std::move(frequency.segSizes);
    out.dictionarySize = frequency.dictionarySize;
    out.exceptionCount = frequency.exceptionCount;
    selectedPayload = std::move(frequency.payload);
  }
  if (deltaFrequency.used &&
      deltaFrequency.payload.size() <=
          selectedPayload.size() * kFreqPforSelectionRatio) {
    out.codec = RegionCodec::kDeltaFreqPfor;
    out.base = deltaFrequencyBase;
    out.first = firstValue;
    out.segSizes = std::move(deltaFrequency.segSizes);
    out.dictionarySize = deltaFrequency.dictionarySize;
    out.exceptionCount = deltaFrequency.exceptionCount;
    selectedPayload = std::move(deltaFrequency.payload);
  }

  payloads.push_back(std::move(selectedPayload));
  regions.push_back(std::move(out));
}

template <typename T>
void decodeTypedRegion(
    const uint8_t* src,
    const EncodedRegion& region,
    uint8_t* blobBase,
    PlaneArena* arena,
    rmm::cuda_stream_view stream) {
  const uint32_t n = region.rawBytes / region.elemWidth;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const uint32_t stride = alignedStride(n);
  T* out = reinterpret_cast<T*>(blobBase + region.blobOffset);

  if (region.codec == RegionCodec::kDictPfor) {
    if (!region.segSizes.empty() || region.dictionary.empty() ||
        region.dictionary.size() > 256) {
      throw std::runtime_error(
          "ucx-exchange column codec: invalid dictionary-PFOR descriptor");
    }
    rmm::device_buffer dictionaryDevice(
        region.dictionary.data(),
        region.dictionary.size() * sizeof(uint16_t),
        stream);
    dictionaryPforDecodeKernel<T><<<blocks, threads, 0, stream.value()>>>(
        src,
        static_cast<const uint16_t*>(dictionaryDevice.data()),
        static_cast<uint32_t>(region.dictionary.size()),
        region.base,
        out,
        n);
    UCX_CUDA_CHECK(cudaGetLastError());
    return;
  }

  if ((region.codec == RegionCodec::kFreqPfor ||
       region.codec == RegionCodec::kDeltaFreqPfor) &&
      ((region.segSizes.size() != 1 && region.segSizes.size() != 2) ||
       region.dictionarySize == 0 ||
       region.dictionarySize > kFreqPforCodeLimit ||
       region.exceptionCount > n ||
       (region.segSizes.size() == 1 && region.dictionarySize > 256))) {
    throw std::runtime_error(
        "ucx-exchange column codec: invalid frequency-PFOR descriptor");
  }

  if (arena == nullptr) {
    throw std::runtime_error(
        "ucx-exchange column codec: missing rANS decode arena");
  }
  auto planes = decodePlanes(src, region.segSizes, n, *arena, stream);

  if (region.codec == RegionCodec::kDeltaFreqPfor) {
    std::size_t rankBytes = 0;
    for (const auto size : region.segSizes) {
      rankBytes += roundUp16(size);
    }
    const auto* dictionary = reinterpret_cast<const uint16_t*>(src + rankBytes);
    const std::size_t dictionaryBytes =
        static_cast<std::size_t>(region.dictionarySize) * sizeof(uint16_t);
    const auto* exceptionPositions = reinterpret_cast<const uint32_t*>(
        src + rankBytes + roundUp16(dictionaryBytes));
    const std::size_t positionBytes =
        static_cast<std::size_t>(region.exceptionCount) * sizeof(uint32_t);
    const auto* exceptionValues = reinterpret_cast<const int64_t*>(
        reinterpret_cast<const uint8_t*>(exceptionPositions) +
        roundUp16(positionBytes));

    rmm::device_buffer deltas(static_cast<std::size_t>(n) * 8, stream);
    auto* deltasPtr = static_cast<int64_t*>(deltas.data());
    frequencyPforDecodeKernel<int64_t><<<blocks, threads, 0, stream.value()>>>(
        static_cast<const uint8_t*>(planes.data()),
        stride,
        region.segSizes.size(),
        dictionary,
        region.dictionarySize,
        region.base,
        deltasPtr,
        n);
    if (region.exceptionCount != 0) {
      const int exceptionBlocks =
          (region.exceptionCount + threads - 1) / threads;
      patchFrequencyExceptionsKernel<int64_t>
          <<<exceptionBlocks, threads, 0, stream.value()>>>(
              exceptionPositions,
              exceptionValues,
              deltasPtr,
              region.exceptionCount);
    }
    UCX_CUDA_CHECK(cudaGetLastError());
    unZigzagKernel<<<blocks, threads, 0, stream.value()>>>(deltasPtr, n);
    std::size_t tempBytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, tempBytes, deltasPtr, deltasPtr, n, stream.value());
    rmm::device_buffer temp(tempBytes, stream);
    cub::DeviceScan::InclusiveSum(
        temp.data(), tempBytes, deltasPtr, deltasPtr, n, stream.value());
    finalizeDeltaKernel<T><<<blocks, threads, 0, stream.value()>>>(
        deltasPtr, region.first, out, n);
    UCX_CUDA_CHECK(cudaGetLastError());
    return;
  }

  if (region.codec == RegionCodec::kFreqPfor) {
    std::size_t rankBytes = 0;
    for (const auto size : region.segSizes) {
      rankBytes += roundUp16(size);
    }
    const auto* dictionary = reinterpret_cast<const uint16_t*>(src + rankBytes);
    const std::size_t dictionaryBytes =
        static_cast<std::size_t>(region.dictionarySize) * sizeof(uint16_t);
    const auto* exceptionPositions = reinterpret_cast<const uint32_t*>(
        src + rankBytes + roundUp16(dictionaryBytes));
    const std::size_t positionBytes =
        static_cast<std::size_t>(region.exceptionCount) * sizeof(uint32_t);
    const auto* exceptionValues = reinterpret_cast<const T*>(
        reinterpret_cast<const uint8_t*>(exceptionPositions) +
        roundUp16(positionBytes));
    frequencyPforDecodeKernel<T><<<blocks, threads, 0, stream.value()>>>(
        static_cast<const uint8_t*>(planes.data()),
        stride,
        region.segSizes.size(),
        dictionary,
        region.dictionarySize,
        region.base,
        out,
        n);
    if (region.exceptionCount != 0) {
      const int exceptionBlocks =
          (region.exceptionCount + threads - 1) / threads;
      patchFrequencyExceptionsKernel<T>
          <<<exceptionBlocks, threads, 0, stream.value()>>>(
              exceptionPositions, exceptionValues, out, region.exceptionCount);
    }
    UCX_CUDA_CHECK(cudaGetLastError());
    return;
  }

  if (region.codec == RegionCodec::kFor) {
    recombAddKernel<T><<<blocks, threads, 0, stream.value()>>>(
        static_cast<const uint8_t*>(planes.data()),
        region.base,
        out,
        n,
        stride,
        region.segSizes.size());
    return;
  }
  // kDeltaFor: planes -> zigzag deltas -> signed deltas -> prefix sum ->
  // +first.
  rmm::device_buffer deltas(static_cast<std::size_t>(n) * 8, stream);
  auto* deltasPtr = static_cast<int64_t*>(deltas.data());
  recombAddKernel<int64_t><<<blocks, threads, 0, stream.value()>>>(
      static_cast<const uint8_t*>(planes.data()),
      0,
      deltasPtr,
      n,
      stride,
      region.segSizes.size());
  unZigzagKernel<<<blocks, threads, 0, stream.value()>>>(deltasPtr, n);
  std::size_t tempBytes = 0;
  cub::DeviceScan::InclusiveSum(
      nullptr, tempBytes, deltasPtr, deltasPtr, n, stream.value());
  rmm::device_buffer temp(tempBytes, stream);
  cub::DeviceScan::InclusiveSum(
      temp.data(), tempBytes, deltasPtr, deltasPtr, n, stream.value());
  finalizeDeltaKernel<T>
      <<<blocks, threads, 0, stream.value()>>>(deltasPtr, region.first, out, n);
}

} // namespace

PackedCompressResult compressPackedFor(
    const void* gpuData,
    std::size_t size,
    rmm::cuda_stream_view stream) {
  PackedCompressResult result;
  result.stats.inputBytes = size;
  const std::size_t typedBytes = size & ~static_cast<std::size_t>(7);
  if (typedBytes < (16u << 20)) { // floor: small chunks ship raw
    return result;
  }
  result.stats.attempted = true;
  std::vector<rmm::device_buffer> payloads;
  PlaneArena arena(stream);
  TypedRegion whole{0, typedBytes / 8, 8};
  encodeTypedRegion<int64_t>(
      static_cast<const uint8_t*>(gpuData),
      whole,
      stream,
      arena,
      result.regions,
      payloads,
      /*forOnly=*/true);
  if (typedBytes < size) {
    EncodedRegion tail;
    tail.blobOffset = typedBytes;
    tail.rawBytes = size - typedBytes;
    tail.codec = RegionCodec::kRaw;
    payloads.emplace_back();
    result.regions.push_back(std::move(tail));
  }
  std::size_t total = 0;
  for (std::size_t i = 0; i < result.regions.size(); ++i) {
    const auto candidateBytes = result.regions[i].codec == RegionCodec::kRaw
        ? static_cast<std::size_t>(result.regions[i].rawBytes)
        : payloads[i].size();
    total += roundUp16(candidateBytes);
    recordRegionStats(result.stats, result.regions[i], candidateBytes);
  }
  result.stats.candidateBytes = total;
  if (static_cast<double>(total) > 0.98 * size) {
    result.regions.clear();
    return result;
  }
  result.data = rmm::device_buffer(total, stream);
  std::size_t off = 0;
  for (std::size_t i = 0; i < result.regions.size(); ++i) {
    const bool raw = result.regions[i].codec == RegionCodec::kRaw;
    const auto bytes = raw
        ? static_cast<std::size_t>(result.regions[i].rawBytes)
        : payloads[i].size();
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(result.data.data()) + off,
        raw ? static_cast<const uint8_t*>(gpuData) +
                result.regions[i].blobOffset
            : static_cast<const uint8_t*>(payloads[i].data()),
        bytes,
        cudaMemcpyDeviceToDevice,
        stream.value()));
    off += roundUp16(bytes);
  }
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  result.used = true;
  return result;
}

PackedCompressResult compressPacked(
    const uint8_t* metadata,
    const void* gpuData,
    std::size_t size,
    rmm::cuda_stream_view stream,
    double minGain,
    bool enableAdvancedCodecs,
    std::size_t advancedCodecMinBytes) {
  PackedCompressResult result;
  result.stats.attempted = true;
  result.stats.inputBytes = size;
  const auto* blobBase = static_cast<const uint8_t*>(gpuData);

  std::vector<TypedRegion> typed;
  auto view = cudf::unpack(metadata, blobBase);
  for (const auto& col : view) {
    collectTypedRegions(col, blobBase, typed);
  }
  std::sort(typed.begin(), typed.end(), [](const auto& a, const auto& b) {
    return a.offset < b.offset;
  });

  std::vector<EncodedRegion> regions;
  std::vector<rmm::device_buffer> payloads;
  auto arena = typed.empty() ? nullptr : std::make_unique<PlaneArena>(stream);
  std::size_t cursor = 0;

  auto addResidual = [&](std::size_t offset, std::size_t bytes) {
    if (bytes == 0) {
      return;
    }
    EncodedRegion region;
    region.blobOffset = offset;
    region.rawBytes = bytes;
    if (bytes >= kMinResidualBytes) {
      ++result.stats.residualRansAttempts;
      result.stats.residualRansInputBytes += bytes;
      // Stage into a fresh (aligned) buffer: gap offsets inside the blob are
      // arbitrary and dietgpu requires aligned input pointers.
      rmm::device_buffer staged(bytes, stream);
      UCX_CUDA_CHECK(cudaMemcpyAsync(
          staged.data(),
          blobBase + offset,
          bytes,
          cudaMemcpyDeviceToDevice,
          stream.value()));
      auto compressed = compressBlob(staged.data(), bytes, stream, minGain, 1);
      result.stats.residualRansCandidateBytes +=
          compressed.stats.candidateBytes;
      if (compressed.used) {
        ++result.stats.residualRansAccepted;
        region.codec = RegionCodec::kByteRans;
        region.segSizes.assign(
            compressed.segSizes.begin(), compressed.segSizes.end());
        payloads.push_back(std::move(compressed.data));
        regions.push_back(std::move(region));
        return;
      }
    }
    region.codec = RegionCodec::kRaw;
    payloads.emplace_back(); // raw regions copy from the original blob
    regions.push_back(std::move(region));
  };

  for (const auto& region : typed) {
    const std::size_t regionBytes = region.elems * region.width;
    if (region.offset < cursor) {
      continue; // overlap safety; leave to residual coverage of earlier pass
    }
    addResidual(cursor, region.offset - cursor);
    if (region.width == 8) {
      encodeTypedRegion<int64_t>(
          blobBase,
          region,
          stream,
          *arena,
          regions,
          payloads,
          /*forOnly=*/false,
          enableAdvancedCodecs,
          advancedCodecMinBytes);
    } else {
      encodeTypedRegion<int32_t>(
          blobBase,
          region,
          stream,
          *arena,
          regions,
          payloads,
          /*forOnly=*/false,
          enableAdvancedCodecs,
          advancedCodecMinBytes);
    }
    cursor = region.offset + regionBytes;
  }
  addResidual(cursor, size - cursor);

  std::size_t total = 0;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto candidateBytes = regions[i].codec == RegionCodec::kRaw
        ? static_cast<std::size_t>(regions[i].rawBytes)
        : payloads[i].size();
    total += roundUp16(candidateBytes);
    recordRegionStats(result.stats, regions[i], candidateBytes);
  }
  result.stats.candidateBytes = total;
  if (static_cast<double>(total) > (1.0 - minGain) * size) {
    return result;
  }

  result.data = rmm::device_buffer(total, stream);
  std::size_t off = 0;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const bool raw = regions[i].codec == RegionCodec::kRaw;
    const auto bytes = raw ? static_cast<std::size_t>(regions[i].rawBytes)
                           : payloads[i].size();
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(result.data.data()) + off,
        raw ? blobBase + regions[i].blobOffset
            : static_cast<const uint8_t*>(payloads[i].data()),
        bytes,
        cudaMemcpyDeviceToDevice,
        stream.value()));
    off += roundUp16(bytes);
  }
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  result.regions = std::move(regions);
  result.used = true;
  return result;
}

rmm::device_buffer decompressPacked(
    const void* src,
    const std::vector<EncodedRegion>& regions,
    std::size_t uncompressedBytes,
    rmm::cuda_stream_view stream) {
  rmm::device_buffer blob(uncompressedBytes, stream);
  auto* blobBase = static_cast<uint8_t*>(blob.data());
  const auto* wire = static_cast<const uint8_t*>(src);
  std::unique_ptr<PlaneArena> arena;
  std::size_t off = 0;

  for (const auto& region : regions) {
    std::size_t encodedBytes = 0; // true wire footprint before padding
    switch (region.codec) {
      case RegionCodec::kRaw:
        encodedBytes = region.rawBytes;
        UCX_CUDA_CHECK(cudaMemcpyAsync(
            blobBase + region.blobOffset,
            wire + off,
            encodedBytes,
            cudaMemcpyDeviceToDevice,
            stream.value()));
        break;
      case RegionCodec::kByteRans: {
        for (auto s : region.segSizes) {
          encodedBytes += roundUp16(s);
        }
        std::vector<uint32_t> segSizes(
            region.segSizes.begin(), region.segSizes.end());
        auto decoded =
            decompressBlob(wire + off, segSizes, region.rawBytes, stream);
        UCX_CUDA_CHECK(cudaMemcpyAsync(
            blobBase + region.blobOffset,
            decoded.data(),
            region.rawBytes,
            cudaMemcpyDeviceToDevice,
            stream.value()));
        break;
      }
      case RegionCodec::kDictPfor: {
        encodedBytes = alignedStride(
            static_cast<uint32_t>(region.rawBytes / region.elemWidth));
        if (region.elemWidth == 8) {
          decodeTypedRegion<int64_t>(
              wire + off, region, blobBase, nullptr, stream);
        } else {
          decodeTypedRegion<int32_t>(
              wire + off, region, blobBase, nullptr, stream);
        }
        break;
      }
      case RegionCodec::kDeltaFreqPfor:
      case RegionCodec::kFreqPfor: {
        for (auto s : region.segSizes) {
          encodedBytes += roundUp16(s);
        }
        encodedBytes += roundUp16(
            static_cast<std::size_t>(region.dictionarySize) * sizeof(uint16_t));
        encodedBytes += roundUp16(
            static_cast<std::size_t>(region.exceptionCount) * sizeof(uint32_t));
        const std::size_t exceptionWidth =
            region.codec == RegionCodec::kDeltaFreqPfor
            ? sizeof(int64_t)
            : static_cast<std::size_t>(region.elemWidth);
        encodedBytes += roundUp16(
            static_cast<std::size_t>(region.exceptionCount) * exceptionWidth);
        if (!arena) {
          arena = std::make_unique<PlaneArena>(stream);
        }
        if (region.elemWidth == 8) {
          decodeTypedRegion<int64_t>(
              wire + off, region, blobBase, arena.get(), stream);
        } else {
          decodeTypedRegion<int32_t>(
              wire + off, region, blobBase, arena.get(), stream);
        }
        break;
      }
      case RegionCodec::kFor:
      case RegionCodec::kDeltaFor: {
        for (auto s : region.segSizes) {
          encodedBytes += roundUp16(s);
        }
        if (!arena) {
          arena = std::make_unique<PlaneArena>(stream);
        }
        if (region.elemWidth == 8) {
          decodeTypedRegion<int64_t>(
              wire + off, region, blobBase, arena.get(), stream);
        } else {
          decodeTypedRegion<int32_t>(
              wire + off, region, blobBase, arena.get(), stream);
        }
        break;
      }
    }
    off += roundUp16(encodedBytes);
  }
  // Settle every queued region once, both for downstream visibility and
  // before the shared DietGPU arena is released.
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  return blob;
}

void serializeRegions(
    const PackedCompressResult& result,
    std::size_t uncompressedBytes,
    std::vector<int64_t>& out) {
  out.clear();
  out.push_back(kPerColumnMagic);
  out.push_back(static_cast<int64_t>(uncompressedBytes));
  out.push_back(static_cast<int64_t>(result.regions.size()));
  for (const auto& region : result.regions) {
    out.push_back(region.blobOffset);
    out.push_back(region.rawBytes);
    out.push_back(static_cast<int64_t>(region.codec));
    out.push_back(region.elemWidth);
    out.push_back(region.base);
    out.push_back(region.first);
    out.push_back(static_cast<int64_t>(region.segSizes.size()));
    for (auto s : region.segSizes) {
      out.push_back(s);
    }
    if (region.codec == RegionCodec::kDictPfor) {
      out.push_back(static_cast<int64_t>(region.dictionary.size()));
      for (std::size_t first = 0; first < region.dictionary.size();
           first += kDictionaryCodesPerWord) {
        uint64_t packed = 0;
        for (std::size_t lane = 0; lane < kDictionaryCodesPerWord &&
             first + lane < region.dictionary.size();
             ++lane) {
          packed |= static_cast<uint64_t>(region.dictionary[first + lane])
              << (16 * lane);
        }
        out.push_back(static_cast<int64_t>(packed));
      }
    } else if (
        region.codec == RegionCodec::kFreqPfor ||
        region.codec == RegionCodec::kDeltaFreqPfor) {
      out.push_back(static_cast<int64_t>(region.dictionarySize));
      out.push_back(static_cast<int64_t>(region.exceptionCount));
    }
  }
}

bool deserializeRegions(
    const std::vector<int64_t>& in,
    std::vector<EncodedRegion>& regions,
    std::size_t& uncompressedBytes) {
  if (in.size() < 3 || in[0] != kPerColumnMagic || in[1] < 0 || in[2] < 0) {
    return false;
  }
  uncompressedBytes = static_cast<std::size_t>(in[1]);
  const auto numRegions = static_cast<std::size_t>(in[2]);
  std::size_t pos = 3;
  regions.clear();
  regions.reserve(numRegions);
  for (std::size_t r = 0; r < numRegions; ++r) {
    if (pos + 7 > in.size()) {
      return false;
    }
    EncodedRegion region;
    region.blobOffset = in[pos++];
    region.rawBytes = in[pos++];
    const auto codecValue = in[pos++];
    if (codecValue < static_cast<int64_t>(RegionCodec::kRaw) ||
        codecValue > static_cast<int64_t>(RegionCodec::kDeltaFreqPfor)) {
      return false;
    }
    region.codec = static_cast<RegionCodec>(codecValue);
    region.elemWidth = static_cast<int32_t>(in[pos++]);
    region.base = in[pos++];
    region.first = in[pos++];
    const auto numSegsValue = in[pos++];
    if (region.blobOffset < 0 || region.rawBytes < 0 || numSegsValue < 0) {
      return false;
    }
    const auto numSegs = static_cast<std::size_t>(numSegsValue);
    if (numSegs > in.size() - pos) {
      return false;
    }
    region.segSizes.assign(in.begin() + pos, in.begin() + pos + numSegs);
    pos += numSegs;
    if (std::any_of(
            region.segSizes.begin(), region.segSizes.end(), [](int64_t size) {
              return size < 0;
            })) {
      return false;
    }
    if (region.codec == RegionCodec::kDictPfor) {
      if (numSegs != 0 || pos >= in.size() || in[pos] <= 0 || in[pos] > 256 ||
          (region.elemWidth != 4 && region.elemWidth != 8)) {
        return false;
      }
      const auto dictionarySize = static_cast<std::size_t>(in[pos++]);
      const auto dictionaryWords =
          (dictionarySize + kDictionaryCodesPerWord - 1) /
          kDictionaryCodesPerWord;
      if (dictionaryWords > in.size() - pos) {
        return false;
      }
      region.dictionary.resize(dictionarySize);
      for (std::size_t index = 0; index < dictionarySize; ++index) {
        const uint64_t packed =
            static_cast<uint64_t>(in[pos + index / kDictionaryCodesPerWord]);
        region.dictionary[index] = static_cast<uint16_t>(
            (packed >> (16 * (index % kDictionaryCodesPerWord))) & 0xffffu);
      }
      pos += dictionaryWords;
    } else if (
        region.codec == RegionCodec::kFreqPfor ||
        region.codec == RegionCodec::kDeltaFreqPfor) {
      if (pos + 2 > in.size() || (numSegs != 1 && numSegs != 2) ||
          in[pos] <= 0 || in[pos] > kFreqPforCodeLimit || in[pos + 1] < 0 ||
          (region.elemWidth != 4 && region.elemWidth != 8) ||
          region.rawBytes % region.elemWidth != 0 ||
          static_cast<uint64_t>(in[pos + 1]) >
              static_cast<uint64_t>(region.rawBytes / region.elemWidth) ||
          (numSegs == 1 && in[pos] > 256)) {
        return false;
      }
      region.dictionarySize = static_cast<uint32_t>(in[pos++]);
      region.exceptionCount = static_cast<uint32_t>(in[pos++]);
    }
    regions.push_back(std::move(region));
  }
  return pos == in.size();
}

} // namespace facebook::velox::ucx_exchange
