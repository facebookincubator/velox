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
#include "velox/experimental/ucx-exchange/UcxCompression.h"

#include <cuda_runtime.h>

#include <stdexcept>

#include <fmt/format.h>
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/utils/StackDeviceMemory.h"

namespace facebook::velox::ucx_exchange {
namespace {

constexpr int kProbBits = 10;
constexpr bool kUseChecksum = false;
// DietGPU's temporary memory grows with the total uncompressed bytes in a
// batch. A complete packed string region can contain several GiB of chars;
// submitting every 32 MiB wire segment at once both defeats the fixed arena
// and has triggered an illegal address in DietGPU's large-batch path. The
// wire format is already independently segmented, so cap only the number of
// segments passed to each codec invocation. Five 32 MiB inputs require about
// 226 MiB of DietGPU scratch and fit the arena's roughly 255 MiB usable
// capacity; six request 271,266,611 bytes and fall back to synchronizing
// cudaMalloc after allocator bookkeeping reduces the nominal 256 MiB arena.
// This preserves all on-wire boundaries.
constexpr uint32_t kMaxSegmentsPerDietGpuBatch = 5;

inline std::size_t roundUp16(std::size_t v) {
  return (v + 15) & ~static_cast<std::size_t>(15);
}

#define UCX_CUDA_CHECK(expr)                                \
  do {                                                      \
    cudaError_t err = (expr);                               \
    if (err != cudaSuccess) {                               \
      throw std::runtime_error(                             \
          fmt::format(                                      \
              "CUDA error in ucx-exchange compression: {}", \
              cudaGetErrorString(err)));                    \
    }                                                       \
  } while (0)

// Per-call DietGPU scratch arena backed by rmm (stream-ordered alloc/free):
// no shared state between concurrent codec calls on different streams.
constexpr std::size_t kArenaBytes = 256u << 20;

struct CallArena {
  rmm::device_buffer buffer;
  dietgpu::StackDeviceMemory stack;
  CallArena(rmm::cuda_stream_view stream, int device)
      : buffer(kArenaBytes, stream),
        stack(device, buffer.data(), kArenaBytes) {}
};

int currentDevice() {
  int device = 0;
  UCX_CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

std::vector<std::pair<const uint8_t*, uint32_t>> segments(
    const void* src,
    std::size_t size) {
  std::vector<std::pair<const uint8_t*, uint32_t>> segs;
  const auto* base = static_cast<const uint8_t*>(src);
  for (std::size_t off = 0; off < size; off += kCompressSegmentBytes) {
    segs.emplace_back(
        base + off,
        static_cast<uint32_t>(std::min(kCompressSegmentBytes, size - off)));
  }
  return segs;
}

} // namespace

CompressResult compressBlob(
    const void* src,
    std::size_t size,
    rmm::cuda_stream_view stream,
    double minGain,
    std::size_t minBytes) {
  CompressResult result;
  result.stats.inputBytes = size;
  if (size < minBytes) {
    return result;
  }
  result.stats.attempted = true;
  const auto segs = segments(src, size);
  const uint32_t numSegs = segs.size();
  CallArena arena(stream, currentDevice());

  // Encode bounded groups of wire segments. Compact each group before
  // reusing scratch so peak temporary memory does not scale with a multi-GiB
  // string buffer.
  const uint32_t maxCompSeg =
      dietgpu::getMaxCompressedSize(kCompressSegmentBytes);
  result.segSizes.resize(numSegs);
  std::size_t paddedTotal = 0;
  std::vector<rmm::device_buffer> batches;
  batches.reserve(
      (numSegs + kMaxSegmentsPerDietGpuBatch - 1) /
      kMaxSegmentsPerDietGpuBatch);
  for (uint32_t first = 0; first < numSegs;
       first += kMaxSegmentsPerDietGpuBatch) {
    const uint32_t count =
        std::min(kMaxSegmentsPerDietGpuBatch, numSegs - first);
    rmm::device_buffer scratch(
        static_cast<std::size_t>(count) * maxCompSeg, stream);
    rmm::device_buffer outSizesDev(count * sizeof(uint32_t), stream);
    std::vector<const void*> inPtrs(count);
    std::vector<uint32_t> inSizes(count);
    std::vector<void*> outPtrs(count);
    for (uint32_t local = 0; local < count; ++local) {
      const uint32_t index = first + local;
      inPtrs[local] = segs[index].first;
      inSizes[local] = segs[index].second;
      outPtrs[local] = static_cast<uint8_t*>(scratch.data()) +
          static_cast<std::size_t>(local) * maxCompSeg;
    }

    dietgpu::ansEncodeBatchPointer(
        arena.stack,
        dietgpu::ANSCodecConfig(kProbBits, kUseChecksum),
        count,
        inPtrs.data(),
        inSizes.data(),
        /*histogram_dev=*/nullptr,
        outPtrs.data(),
        static_cast<uint32_t*>(outSizesDev.data()),
        stream.value());
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        result.segSizes.data() + first,
        outSizesDev.data(),
        count * sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));

    std::size_t batchBytes = 0;
    for (uint32_t local = 0; local < count; ++local) {
      const auto compressed = result.segSizes[first + local];
      paddedTotal += roundUp16(compressed);
      batchBytes += roundUp16(compressed);
    }
    rmm::device_buffer batch(batchBytes, stream);
    std::size_t batchOffset = 0;
    for (uint32_t local = 0; local < count; ++local) {
      const auto compressed = result.segSizes[first + local];
      UCX_CUDA_CHECK(cudaMemcpyAsync(
          static_cast<uint8_t*>(batch.data()) + batchOffset,
          outPtrs[local],
          compressed,
          cudaMemcpyDeviceToDevice,
          stream.value()));
      batchOffset += roundUp16(compressed);
    }
    UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
    batches.push_back(std::move(batch));
  }
  result.stats.candidateBytes = paddedTotal;
  if (paddedTotal == 0 ||
      static_cast<double>(paddedTotal) > (1.0 - minGain) * size) {
    return result; // did not pay; send uncompressed
  }

  // Join the already compacted batches. Each segment remains 16-byte aligned
  // as required by DietGPU decode.
  result.data = rmm::device_buffer(paddedTotal, stream);
  std::size_t off = 0;
  for (const auto& batch : batches) {
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(result.data.data()) + off,
        batch.data(),
        batch.size(),
        cudaMemcpyDeviceToDevice,
        stream.value()));
    off += batch.size();
  }
  // The compaction copies above are asynchronous and the consumer (UCXX
  // tagSend) is not stream-aware: settle the buffer before handing it out.
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  result.used = true;
  return result;
}

rmm::device_buffer decompressBlob(
    const void* src,
    const std::vector<uint32_t>& segSizes,
    std::size_t uncompressedBytes,
    rmm::cuda_stream_view stream) {
  const uint32_t numSegs = segSizes.size();
  if (numSegs == 0) {
    throw std::runtime_error("decompressBlob: empty segment list");
  }
  CallArena arena(stream, currentDevice());
  rmm::device_buffer out(uncompressedBytes, stream);

  std::size_t inOff = 0;
  std::size_t outOff = 0;
  for (uint32_t first = 0; first < numSegs;
       first += kMaxSegmentsPerDietGpuBatch) {
    const uint32_t count =
        std::min(kMaxSegmentsPerDietGpuBatch, numSegs - first);
    std::vector<const void*> inPtrs(count);
    std::vector<void*> outPtrs(count);
    std::vector<uint32_t> outCaps(count);
    for (uint32_t local = 0; local < count; ++local) {
      const uint32_t index = first + local;
      inPtrs[local] = static_cast<const uint8_t*>(src) + inOff;
      inOff += roundUp16(segSizes[index]);
      outPtrs[local] = static_cast<uint8_t*>(out.data()) + outOff;
      const auto cap = static_cast<uint32_t>(
          std::min(kCompressSegmentBytes, uncompressedBytes - outOff));
      outCaps[local] = cap;
      outOff += cap;
    }
    auto status = dietgpu::ansDecodeBatchPointer(
        arena.stack,
        dietgpu::ANSCodecConfig(kProbBits, kUseChecksum),
        count,
        inPtrs.data(),
        outPtrs.data(),
        outCaps.data(),
        /*outSuccess_dev=*/nullptr,
        /*outSize_dev=*/nullptr,
        stream.value());
    if (status.error != dietgpu::ANSDecodeError::None) {
      throw std::runtime_error(
          "ucx-exchange rANS decode failed (checksum or corrupt frame)");
    }
  }
  if (outOff != uncompressedBytes) {
    throw std::runtime_error("decompressBlob: segment/output mismatch");
  }
  // Settle before the arena (stream-ordered) frees and the caller consumes.
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  return out;
}

} // namespace facebook::velox::ucx_exchange
