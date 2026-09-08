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
#include "velox/experimental/ucx-exchange/UcxFloat64Codec.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/utils/StackDeviceMemory.h"

namespace facebook::velox::ucx_exchange {
namespace {

constexpr int kProbBits = 10;
constexpr bool kUseChecksum = false;
constexpr uint32_t kFloat64Bytes = sizeof(double);
constexpr std::size_t kArenaBytes = 256u << 20;

inline std::size_t roundUp16(std::size_t value) {
  return (value + 15) & ~static_cast<std::size_t>(15);
}

inline uint32_t alignedStride(uint32_t values) {
  return (values + 15u) & ~15u;
}

#define UCX_FLOAT64_CUDA_CHECK(expr)                                           \
  do {                                                                         \
    cudaError_t error = (expr);                                                \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error(                                                \
          fmt::format(                                                         \
              "CUDA error in UCX FP64 codec: {}", cudaGetErrorString(error))); \
    }                                                                          \
  } while (0)

int currentDevice() {
  int device = 0;
  UCX_FLOAT64_CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

struct CallArena {
  rmm::device_buffer buffer;
  dietgpu::StackDeviceMemory stack;

  explicit CallArena(rmm::cuda_stream_view stream)
      : buffer(kArenaBytes, stream),
        stack(currentDevice(), buffer.data(), kArenaBytes) {}
};

__global__ void splitFloat64Kernel(
    const uint64_t* __restrict__ input,
    uint8_t* __restrict__ planes,
    uint32_t numValues,
    uint32_t stride) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numValues) {
    return;
  }

  const uint64_t bits = input[index];
  const uint64_t rotated = (bits << 1) | (bits >> 63);
#pragma unroll
  for (uint32_t plane = 0; plane < kFloat64Bytes; ++plane) {
    const uint32_t shift = 8 * (kFloat64Bytes - plane - 1);
    planes[static_cast<std::size_t>(plane) * stride + index] =
        static_cast<uint8_t>(rotated >> shift);
  }
}

__global__ void joinFloat64Kernel(
    const uint8_t* __restrict__ planes,
    uint64_t* __restrict__ output,
    uint32_t numValues,
    uint32_t stride) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numValues) {
    return;
  }

  uint64_t rotated = 0;
#pragma unroll
  for (uint32_t plane = 0; plane < kFloat64Bytes; ++plane) {
    rotated = (rotated << 8) |
        planes[static_cast<std::size_t>(plane) * stride + index];
  }
  output[index] = (rotated >> 1) | (rotated << 63);
}

void validatePlaneCount(uint32_t exponentPlanes) {
  if (exponentPlanes < 1 || exponentPlanes > 2) {
    throw std::invalid_argument(
        "UCX FP64 codec requires one or two exponent planes");
  }
}

} // namespace

Float64CompressResult compressFloat64WithScratch(
    const double* input,
    uint32_t numValues,
    uint32_t exponentPlanes,
    rmm::cuda_stream_view stream,
    dietgpu::StackDeviceMemory& scratch) {
  validatePlaneCount(exponentPlanes);

  Float64CompressResult result;
  result.exponentPlanes = exponentPlanes;
  result.numValues = numValues;
  result.inputBytes = static_cast<std::size_t>(numValues) * sizeof(double);
  if (numValues == 0) {
    result.data = rmm::device_buffer(0, stream);
    return result;
  }
  if (input == nullptr) {
    throw std::invalid_argument("UCX FP64 codec received a null input");
  }

  const uint32_t stride = alignedStride(numValues);
  auto planes = scratch.alloc<uint8_t>(
      stream.value(), static_cast<std::size_t>(kFloat64Bytes) * stride);
  UCX_FLOAT64_CUDA_CHECK(cudaMemsetAsync(
      planes.data(),
      0,
      static_cast<std::size_t>(kFloat64Bytes) * stride,
      stream.value()));

  constexpr uint32_t kThreads = 256;
  const uint32_t blocks = (numValues + kThreads - 1) / kThreads;
  splitFloat64Kernel<<<blocks, kThreads, 0, stream.value()>>>(
      reinterpret_cast<const uint64_t*>(input),
      static_cast<uint8_t*>(planes.data()),
      numValues,
      stride);
  UCX_FLOAT64_CUDA_CHECK(cudaGetLastError());

  const uint32_t maxCompressed = static_cast<uint32_t>(
      roundUp16(dietgpu::getMaxCompressedSize(numValues)));
  auto compressedScratch = scratch.alloc<uint8_t>(
      stream.value(), static_cast<std::size_t>(exponentPlanes) * maxCompressed);
  auto sizesDevice = scratch.alloc<uint32_t>(stream.value(), exponentPlanes);
  dietgpu::ansEncodeBatchStride(
      scratch,
      dietgpu::ANSCodecConfig(kProbBits, kUseChecksum),
      exponentPlanes,
      planes.data(),
      numValues,
      stride,
      /*histogram_dev=*/nullptr,
      compressedScratch.data(),
      maxCompressed,
      static_cast<uint32_t*>(sizesDevice.data()),
      stream.value());

  result.exponentSegmentSizes.resize(exponentPlanes);
  UCX_FLOAT64_CUDA_CHECK(cudaMemcpyAsync(
      result.exponentSegmentSizes.data(),
      sizesDevice.data(),
      exponentPlanes * sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_FLOAT64_CUDA_CHECK(cudaStreamSynchronize(stream.value()));

  std::size_t candidateBytes = 0;
  for (const auto size : result.exponentSegmentSizes) {
    candidateBytes += roundUp16(size);
  }
  candidateBytes +=
      static_cast<std::size_t>(kFloat64Bytes - exponentPlanes) * stride;
  result.candidateBytes = candidateBytes;
  result.data = rmm::device_buffer(candidateBytes, stream);
  UCX_FLOAT64_CUDA_CHECK(cudaMemsetAsync(
      result.data.data(), 0, result.data.size(), stream.value()));

  std::size_t outputOffset = 0;
  for (uint32_t plane = 0; plane < exponentPlanes; ++plane) {
    const auto size = result.exponentSegmentSizes[plane];
    UCX_FLOAT64_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(result.data.data()) + outputOffset,
        static_cast<const uint8_t*>(compressedScratch.data()) +
            static_cast<std::size_t>(plane) * maxCompressed,
        size,
        cudaMemcpyDeviceToDevice,
        stream.value()));
    outputOffset += roundUp16(size);
  }
  const std::size_t rawPlaneBytes =
      static_cast<std::size_t>(kFloat64Bytes - exponentPlanes) * stride;
  UCX_FLOAT64_CUDA_CHECK(cudaMemcpyAsync(
      static_cast<uint8_t*>(result.data.data()) + outputOffset,
      static_cast<const uint8_t*>(planes.data()) +
          static_cast<std::size_t>(exponentPlanes) * stride,
      rawPlaneBytes,
      cudaMemcpyDeviceToDevice,
      stream.value()));
  UCX_FLOAT64_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  return result;
}

Float64CompressResult compressFloat64(
    const double* input,
    uint32_t numValues,
    uint32_t exponentPlanes,
    rmm::cuda_stream_view stream) {
  validatePlaneCount(exponentPlanes);
  if (numValues == 0) {
    Float64CompressResult result;
    result.exponentPlanes = exponentPlanes;
    result.data = rmm::device_buffer(0, stream);
    return result;
  }
  CallArena arena(stream);
  return compressFloat64WithScratch(
      input, numValues, exponentPlanes, stream, arena.stack);
}

void decompressFloat64PayloadIntoWithScratch(
    const void* data,
    std::size_t candidateBytes,
    const std::vector<uint32_t>& exponentSegmentSizes,
    uint32_t exponentPlanes,
    uint32_t numValues,
    double* output,
    rmm::cuda_stream_view stream,
    dietgpu::StackDeviceMemory& scratch) {
  validatePlaneCount(exponentPlanes);
  if (exponentSegmentSizes.size() != exponentPlanes) {
    throw std::invalid_argument(
        "UCX FP64 codec exponent segment metadata is inconsistent");
  }
  if (numValues == 0) {
    return;
  }
  if (data == nullptr || output == nullptr) {
    throw std::invalid_argument("UCX FP64 codec payload or output is null");
  }

  const uint32_t stride = alignedStride(numValues);
  std::size_t expectedBytes =
      static_cast<std::size_t>(kFloat64Bytes - exponentPlanes) * stride;
  for (const auto segmentSize : exponentSegmentSizes) {
    expectedBytes += roundUp16(segmentSize);
  }
  if (candidateBytes != expectedBytes) {
    throw std::invalid_argument(
        "UCX FP64 codec payload size does not match its metadata");
  }
  auto planes = scratch.alloc<uint8_t>(
      stream.value(), static_cast<std::size_t>(kFloat64Bytes) * stride);
  UCX_FLOAT64_CUDA_CHECK(cudaMemsetAsync(
      planes.data(),
      0,
      static_cast<std::size_t>(kFloat64Bytes) * stride,
      stream.value()));

  std::vector<const void*> inputPointers(exponentPlanes);
  std::vector<void*> outputPointers(exponentPlanes);
  std::vector<uint32_t> outputCapacities(exponentPlanes, numValues);
  std::size_t inputOffset = 0;
  for (uint32_t plane = 0; plane < exponentPlanes; ++plane) {
    inputPointers[plane] = static_cast<const uint8_t*>(data) + inputOffset;
    outputPointers[plane] = static_cast<uint8_t*>(planes.data()) +
        static_cast<std::size_t>(plane) * stride;
    inputOffset += roundUp16(exponentSegmentSizes[plane]);
  }

  const auto status = dietgpu::ansDecodeBatchPointer(
      scratch,
      dietgpu::ANSCodecConfig(kProbBits, kUseChecksum),
      exponentPlanes,
      inputPointers.data(),
      outputPointers.data(),
      outputCapacities.data(),
      /*outSuccess_dev=*/nullptr,
      /*outSize_dev=*/nullptr,
      stream.value());
  if (status.error != dietgpu::ANSDecodeError::None) {
    throw std::runtime_error("UCX FP64 exponent-plane decode failed");
  }

  const std::size_t rawPlaneBytes =
      static_cast<std::size_t>(kFloat64Bytes - exponentPlanes) * stride;
  UCX_FLOAT64_CUDA_CHECK(cudaMemcpyAsync(
      static_cast<uint8_t*>(planes.data()) +
          static_cast<std::size_t>(exponentPlanes) * stride,
      static_cast<const uint8_t*>(data) + inputOffset,
      rawPlaneBytes,
      cudaMemcpyDeviceToDevice,
      stream.value()));
  constexpr uint32_t kThreads = 256;
  const uint32_t blocks = (numValues + kThreads - 1) / kThreads;
  joinFloat64Kernel<<<blocks, kThreads, 0, stream.value()>>>(
      static_cast<const uint8_t*>(planes.data()),
      reinterpret_cast<uint64_t*>(output),
      numValues,
      stride);
  UCX_FLOAT64_CUDA_CHECK(cudaGetLastError());
}

void decompressFloat64PayloadInto(
    const void* data,
    std::size_t candidateBytes,
    const std::vector<uint32_t>& exponentSegmentSizes,
    uint32_t exponentPlanes,
    uint32_t numValues,
    double* output,
    rmm::cuda_stream_view stream) {
  validatePlaneCount(exponentPlanes);
  if (exponentSegmentSizes.size() != exponentPlanes) {
    throw std::invalid_argument(
        "UCX FP64 codec exponent segment metadata is inconsistent");
  }
  if (numValues == 0) {
    return;
  }
  CallArena arena(stream);
  decompressFloat64PayloadIntoWithScratch(
      data,
      candidateBytes,
      exponentSegmentSizes,
      exponentPlanes,
      numValues,
      output,
      stream,
      arena.stack);
}

rmm::device_buffer decompressFloat64Payload(
    const void* data,
    std::size_t candidateBytes,
    const std::vector<uint32_t>& exponentSegmentSizes,
    uint32_t exponentPlanes,
    uint32_t numValues,
    rmm::cuda_stream_view stream) {
  rmm::device_buffer output(
      static_cast<std::size_t>(numValues) * sizeof(double), stream);
  decompressFloat64PayloadInto(
      data,
      candidateBytes,
      exponentSegmentSizes,
      exponentPlanes,
      numValues,
      static_cast<double*>(output.data()),
      stream);
  UCX_FLOAT64_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  return output;
}

rmm::device_buffer decompressFloat64(
    const Float64CompressResult& compressed,
    rmm::cuda_stream_view stream) {
  if (compressed.inputBytes !=
      static_cast<std::size_t>(compressed.numValues) * sizeof(double)) {
    throw std::invalid_argument(
        "UCX FP64 codec input size does not match its value count");
  }
  return decompressFloat64Payload(
      compressed.data.data(),
      compressed.candidateBytes,
      compressed.exponentSegmentSizes,
      compressed.exponentPlanes,
      compressed.numValues,
      stream);
}

} // namespace facebook::velox::ucx_exchange
