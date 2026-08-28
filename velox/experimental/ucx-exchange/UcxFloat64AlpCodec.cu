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
#include "velox/experimental/ucx-exchange/UcxFloat64AlpCodec.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

#include <fmt/format.h>

namespace facebook::velox::ucx_exchange {
namespace {

constexpr uint32_t kNumExponents = 19;
// Optimize the exchange codec for inexpensive decisions. These seven pairs
// preserve the floating-point evaluation order that reconstructs common
// decimal values with zero through six fractional digits bit-for-bit. Values
// outside that set remain lossless through exceptions and naturally fall back
// to another codec when the resulting representation does not pay.
constexpr uint32_t kProbePairCount = 7;
constexpr uint32_t kProbeValues = 1u << 10;
constexpr uint32_t kThreads = 256;
constexpr int64_t kExceptionSentinel = LLONG_MIN;
__device__ __constant__ double kDevicePowers[kNumExponents] = {
    1.0,
    1e1,
    1e2,
    1e3,
    1e4,
    1e5,
    1e6,
    1e7,
    1e8,
    1e9,
    1e10,
    1e11,
    1e12,
    1e13,
    1e14,
    1e15,
    1e16,
    1e17,
    1e18};
__device__ __constant__ double kDeviceInversePowers[kNumExponents] = {
    1.0,
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-10,
    1e-11,
    1e-12,
    1e-13,
    1e-14,
    1e-15,
    1e-16,
    1e-17,
    1e-18};
__device__ __constant__ uint32_t
    kProbeExponents[kProbePairCount] = {14, 14, 14, 14, 14, 14, 14};
__device__ __constant__ uint32_t
    kProbeFactors[kProbePairCount] = {14, 13, 12, 11, 10, 9, 8};
constexpr double kEncodingLimit = 9.223372036854774784e18;

constexpr std::size_t roundUp16(std::size_t value) {
  return (value + 15) & ~static_cast<std::size_t>(15);
}

inline std::size_t packedBytes(uint32_t numValues, uint32_t bitWidth) {
  const std::size_t groups = (static_cast<std::size_t>(numValues) + 31) / 32;
  return groups * bitWidth * sizeof(uint32_t);
}

__host__ __device__ uint32_t requiredBits(uint64_t range) {
  uint32_t width = 0;
  while (range != 0) {
    ++width;
    range >>= 1;
  }
  return width;
}

#define UCX_ALP_CUDA_CHECK(expr)                      \
  do {                                                \
    cudaError_t error = (expr);                       \
    if (error != cudaSuccess) {                       \
      throw std::runtime_error(                       \
          fmt::format(                                \
              "CUDA error in UCX FP64 ALP codec: {}", \
              cudaGetErrorString(error)));            \
    }                                                 \
  } while (0)

struct ProbeResult {
  int64_t minimum;
  int64_t maximum;
  uint32_t exceptions;
  uint32_t valid;
};

struct FullResult {
  int64_t minimum;
  int64_t maximum;
  uint32_t exceptions;
};

struct ParameterSelection {
  uint32_t exponent;
  uint32_t factor;
};

struct CodecStats {
  ParameterSelection selection;
  ProbeResult sample;
  FullResult full;
  int64_t planeBase;
  uint32_t planeBitWidth;
  uint32_t planeWidth;
};

struct PlaneSelectionState {
  CodecStats stats;
  uint32_t exceptionCount;
};

constexpr std::size_t kPlaneStateOffset =
    roundUp16(kProbePairCount * sizeof(ProbeResult));
constexpr std::size_t kPlaneWorkspaceBytes =
    kPlaneStateOffset + sizeof(PlaneSelectionState);

__device__ bool encodeExactly(
    double value,
    uint32_t exponent,
    uint32_t factor,
    int64_t& encoded) {
  if (!isfinite(value)) {
    return false;
  }
  const double scaled =
      value * kDevicePowers[exponent] * kDeviceInversePowers[factor];
  // Match ALP's safe FP64-to-int64 range. LLONG_MIN remains reserved as
  // the exception marker.
  if (!isfinite(scaled) || scaled < -kEncodingLimit ||
      scaled > kEncodingLimit) {
    return false;
  }
  encoded = __double2ll_rn(scaled);
  if (encoded == kExceptionSentinel) {
    return false;
  }
  const double decoded = static_cast<double>(encoded) * kDevicePowers[factor] *
      kDeviceInversePowers[exponent];
  return __double_as_longlong(decoded) == __double_as_longlong(value);
}

__global__ void probeKernel(
    const double* __restrict__ input,
    uint32_t numValues,
    uint32_t sampleCount,
    ProbeResult* __restrict__ results) {
  const uint32_t pair = blockIdx.x;
  const uint32_t exponent = kProbeExponents[pair];
  const uint32_t factor = kProbeFactors[pair];
  int64_t localMinimum = LLONG_MAX;
  int64_t localMaximum = LLONG_MIN;
  uint32_t localExceptions = 0;
  uint32_t localValid = 0;

  for (uint32_t sample = threadIdx.x; sample < sampleCount;
       sample += blockDim.x) {
    const uint32_t index = static_cast<uint32_t>(
        static_cast<uint64_t>(sample) * numValues / sampleCount);
    int64_t encoded = 0;
    if (encodeExactly(input[index], exponent, factor, encoded)) {
      localMinimum = min(localMinimum, encoded);
      localMaximum = max(localMaximum, encoded);
      ++localValid;
    } else {
      ++localExceptions;
    }
  }

  __shared__ int64_t minima[kThreads];
  __shared__ int64_t maxima[kThreads];
  __shared__ uint32_t exceptions[kThreads];
  __shared__ uint32_t valid[kThreads];
  minima[threadIdx.x] = localMinimum;
  maxima[threadIdx.x] = localMaximum;
  exceptions[threadIdx.x] = localExceptions;
  valid[threadIdx.x] = localValid;
  __syncthreads();

  for (uint32_t width = blockDim.x / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      minima[threadIdx.x] =
          min(minima[threadIdx.x], minima[threadIdx.x + width]);
      maxima[threadIdx.x] =
          max(maxima[threadIdx.x], maxima[threadIdx.x + width]);
      exceptions[threadIdx.x] += exceptions[threadIdx.x + width];
      valid[threadIdx.x] += valid[threadIdx.x + width];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    results[pair] = ProbeResult{minima[0], maxima[0], exceptions[0], valid[0]};
  }
}

__global__ void selectParametersKernel(
    const ProbeResult* __restrict__ probes,
    uint32_t sampleCount,
    CodecStats* __restrict__ stats,
    uint32_t* __restrict__ exceptionCount) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  uint64_t bestScore = ULLONG_MAX;
  uint32_t bestExponent = 0;
  uint32_t bestFactor = 0;
  uint32_t bestPair = 0;
  for (uint32_t pair = 0; pair < kProbePairCount; ++pair) {
    const uint32_t exponent = kProbeExponents[pair];
    const uint32_t factor = kProbeFactors[pair];
    const auto probe = probes[pair];
    uint32_t bitWidth = 0;
    if (probe.valid != 0) {
      const uint64_t range = static_cast<uint64_t>(probe.maximum) -
          static_cast<uint64_t>(probe.minimum);
      bitWidth = requiredBits(range);
    }
    // Score in bit-samples. Every exception requires a 32-bit position and
    // the original 64-bit value.
    const uint64_t score = static_cast<uint64_t>(bitWidth) * sampleCount +
        static_cast<uint64_t>(probe.exceptions) * 96;
    if (score < bestScore ||
        (score == bestScore &&
         (exponent > bestExponent ||
          (exponent == bestExponent && factor > bestFactor)))) {
      bestScore = score;
      bestExponent = exponent;
      bestFactor = factor;
      bestPair = pair;
    }
  }

  const auto selected = probes[bestPair];
  stats->selection = ParameterSelection{bestExponent, bestFactor};
  stats->sample = selected;
  stats->full = FullResult{LLONG_MAX, LLONG_MIN, 0};
  if (selected.valid == 0) {
    stats->planeBase = 0;
    stats->planeBitWidth = 0;
  } else {
    stats->planeBase = selected.minimum;
    const uint64_t range = static_cast<uint64_t>(selected.maximum) -
        static_cast<uint64_t>(selected.minimum);
    stats->planeBitWidth = requiredBits(range);
  }
  stats->planeWidth = (stats->planeBitWidth + 7) / 8;
  if (exceptionCount != nullptr) {
    *exceptionCount = 0;
  }
}

__device__ void atomicMinSigned(int64_t* address, int64_t value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (static_cast<int64_t>(old) > value) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, static_cast<unsigned long long>(value));
    if (old == assumed) {
      break;
    }
  }
}

__device__ void atomicMaxSigned(int64_t* address, int64_t value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (static_cast<int64_t>(old) < value) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, static_cast<unsigned long long>(value));
    if (old == assumed) {
      break;
    }
  }
}

__global__ void transformKernel(
    const double* __restrict__ input,
    int64_t* __restrict__ encodedValues,
    uint32_t numValues,
    CodecStats* __restrict__ stats) {
  const uint32_t exponent = stats->selection.exponent;
  const uint32_t factor = stats->selection.factor;
  auto* result = &stats->full;
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t encoded = 0;
  const bool inRange = index < numValues;
  const bool valid =
      inRange && encodeExactly(input[index], exponent, factor, encoded);
  if (inRange && encodedValues != nullptr) {
    encodedValues[index] = valid ? encoded : kExceptionSentinel;
  }

  int64_t localMinimum = valid ? encoded : LLONG_MAX;
  int64_t localMaximum = valid ? encoded : LLONG_MIN;
  uint32_t localExceptions = inRange && !valid ? 1u : 0u;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t mask = __activemask();
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    localMinimum =
        min(localMinimum, __shfl_down_sync(mask, localMinimum, offset));
    localMaximum =
        max(localMaximum, __shfl_down_sync(mask, localMaximum, offset));
    localExceptions += __shfl_down_sync(mask, localExceptions, offset);
  }

  __shared__ int64_t warpMinima[kThreads / 32];
  __shared__ int64_t warpMaxima[kThreads / 32];
  __shared__ uint32_t warpExceptions[kThreads / 32];
  if (lane == 0) {
    warpMinima[warp] = localMinimum;
    warpMaxima[warp] = localMaximum;
    warpExceptions[warp] = localExceptions;
  }
  __syncthreads();

  if (warp == 0) {
    localMinimum = lane < kThreads / 32 ? warpMinima[lane] : LLONG_MAX;
    localMaximum = lane < kThreads / 32 ? warpMaxima[lane] : LLONG_MIN;
    localExceptions = lane < kThreads / 32 ? warpExceptions[lane] : 0;
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      localMinimum =
          min(localMinimum, __shfl_down_sync(mask, localMinimum, offset));
      localMaximum =
          max(localMaximum, __shfl_down_sync(mask, localMaximum, offset));
      localExceptions += __shfl_down_sync(mask, localExceptions, offset);
    }
    if (lane == 0) {
      if (localMinimum != LLONG_MAX) {
        atomicMinSigned(&result->minimum, localMinimum);
        atomicMaxSigned(&result->maximum, localMaximum);
      }
      atomicAdd(&result->exceptions, localExceptions);
    }
  }
}

__global__ void packWarpBitPlanesKernel(
    const int64_t* __restrict__ encodedValues,
    uint32_t numValues,
    int64_t base,
    uint32_t bitWidth,
    uint32_t* __restrict__ packed) {
  const uint32_t globalThread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t group = globalThread >> 5;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t groups = (numValues + 31) / 32;
  if (group >= groups) {
    return;
  }
  const uint32_t index = group * 32 + lane;
  uint64_t adjusted = 0;
  if (index < numValues && encodedValues[index] != kExceptionSentinel) {
    adjusted = static_cast<uint64_t>(encodedValues[index]) -
        static_cast<uint64_t>(base);
  }

  const uint32_t mask = __activemask();
  for (uint32_t bit = 0; bit < bitWidth; ++bit) {
    const uint32_t word = __ballot_sync(mask, (adjusted >> bit) & 1u);
    if (lane == 0) {
      packed[static_cast<std::size_t>(group) * bitWidth + bit] = word;
    }
  }
}

__global__ void compactExceptionsKernel(
    const uint64_t* __restrict__ inputBits,
    const int64_t* __restrict__ encodedValues,
    uint32_t numValues,
    uint32_t* __restrict__ positions,
    uint64_t* __restrict__ values,
    uint32_t* __restrict__ writeCount) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const bool exception =
      index < numValues && encodedValues[index] == kExceptionSentinel;
  const uint32_t mask = __activemask();
  const uint32_t exceptions = __ballot_sync(mask, exception);
  if (exceptions == 0) {
    return;
  }

  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t leader = static_cast<uint32_t>(__ffs(exceptions) - 1);
  uint32_t warpOffset = 0;
  if (lane == leader) {
    warpOffset =
        atomicAdd(writeCount, static_cast<uint32_t>(__popc(exceptions)));
  }
  warpOffset = __shfl_sync(mask, warpOffset, leader);
  if (exception) {
    const uint32_t lowerLanes = lane == 0 ? 0u : ((1u << lane) - 1u);
    const uint32_t slot = warpOffset + __popc(exceptions & lowerLanes);
    positions[slot] = index;
    values[slot] = inputBits[index];
  }
}

__global__ void splitAndCompactKernel(
    const double* __restrict__ input,
    uint32_t numValues,
    const CodecStats* __restrict__ selectedStats,
    uint32_t exponent,
    uint32_t factor,
    int64_t base,
    uint8_t* __restrict__ planes,
    uint32_t planeStride,
    uint32_t planeWidth,
    uint32_t* __restrict__ positions,
    uint64_t* __restrict__ exceptionValues,
    uint32_t* __restrict__ writeCount) {
  if (selectedStats != nullptr) {
    exponent = selectedStats->selection.exponent;
    factor = selectedStats->selection.factor;
    base = selectedStats->planeBase;
    planeWidth = selectedStats->planeWidth;
  }
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t encoded = 0;
  const bool inRange = index < numValues;
  const bool exact =
      inRange && encodeExactly(input[index], exponent, factor, encoded);
  const uint64_t adjusted =
      static_cast<uint64_t>(encoded) - static_cast<uint64_t>(base);
  const uint64_t maxAdjusted = planeWidth == sizeof(int64_t)
      ? ULLONG_MAX
      : (planeWidth == 0 ? 0 : (1ULL << (8 * planeWidth)) - 1);
  const bool representable = exact && adjusted <= maxAdjusted;
  const bool exception = inRange && !representable;
  if (inRange && representable && planes != nullptr) {
    for (uint32_t plane = 0; plane < planeWidth; ++plane) {
      planes[static_cast<uint64_t>(plane) * planeStride + index] =
          static_cast<uint8_t>((adjusted >> (8 * plane)) & 0xff);
    }
  } else if (inRange && planes != nullptr) {
    for (uint32_t plane = 0; plane < planeWidth; ++plane) {
      planes[static_cast<uint64_t>(plane) * planeStride + index] = 0;
    }
  }

  const uint32_t mask = __activemask();
  const uint32_t exceptions = __ballot_sync(mask, exception);
  if (exceptions == 0) {
    return;
  }
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t leader = static_cast<uint32_t>(__ffs(exceptions) - 1);
  if (positions == nullptr) {
    if (lane == leader) {
      atomicAdd(writeCount, static_cast<uint32_t>(__popc(exceptions)));
    }
    return;
  }

  uint32_t warpOffset = 0;
  if (lane == leader) {
    warpOffset =
        atomicAdd(writeCount, static_cast<uint32_t>(__popc(exceptions)));
  }
  warpOffset = __shfl_sync(mask, warpOffset, leader);
  if (exception) {
    const uint32_t lowerLanes = lane == 0 ? 0u : ((1u << lane) - 1u);
    const uint32_t slot = warpOffset + __popc(exceptions & lowerLanes);
    positions[slot] = index;
    exceptionValues[slot] = __double_as_longlong(input[index]);
  }
}

__global__ void decodeKernel(
    const uint32_t* __restrict__ packed,
    uint64_t* __restrict__ outputBits,
    uint32_t numValues,
    int64_t base,
    uint32_t bitWidth,
    uint32_t exponent,
    uint32_t factor) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numValues) {
    return;
  }
  const uint32_t group = index >> 5;
  const uint32_t lane = index & 31u;
  uint64_t adjusted = 0;
  for (uint32_t bit = 0; bit < bitWidth; ++bit) {
    const uint32_t word =
        packed[static_cast<std::size_t>(group) * bitWidth + bit];
    adjusted |= static_cast<uint64_t>((word >> lane) & 1u) << bit;
  }
  const int64_t encoded =
      static_cast<int64_t>(static_cast<uint64_t>(base) + adjusted);
  const double decoded = factor == 0
      ? static_cast<double>(encoded) * kDeviceInversePowers[exponent]
      : static_cast<double>(encoded) * kDevicePowers[factor] *
          kDeviceInversePowers[exponent];
  outputBits[index] = static_cast<uint64_t>(__double_as_longlong(decoded));
}

__global__ void decodeIntegersKernel(
    const int64_t* __restrict__ encodedValues,
    uint64_t* __restrict__ outputBits,
    uint32_t numValues,
    uint32_t exponent,
    uint32_t factor) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numValues) {
    return;
  }
  const int64_t encoded = encodedValues[index];
  const double decoded = factor == 0
      ? static_cast<double>(encoded) * kDeviceInversePowers[exponent]
      : static_cast<double>(encoded) * kDevicePowers[factor] *
          kDeviceInversePowers[exponent];
  outputBits[index] = static_cast<uint64_t>(__double_as_longlong(decoded));
}

template <uint32_t kPlaneWidth>
__global__ void decodeRawPlanesKernel(
    const uint8_t* __restrict__ planes,
    uint64_t* __restrict__ outputBits,
    uint32_t numValues,
    uint32_t planeStride,
    int64_t base,
    uint32_t exponent,
    uint32_t factor) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numValues) {
    return;
  }
  uint64_t adjusted = 0;
#pragma unroll
  for (uint32_t plane = 0; plane < kPlaneWidth; ++plane) {
    adjusted |= static_cast<uint64_t>(
                    planes[static_cast<uint64_t>(plane) * planeStride + index])
        << (8 * plane);
  }
  const int64_t encoded =
      static_cast<int64_t>(static_cast<uint64_t>(base) + adjusted);
  const double decoded = factor == 0
      ? static_cast<double>(encoded) * kDeviceInversePowers[exponent]
      : static_cast<double>(encoded) * kDevicePowers[factor] *
          kDeviceInversePowers[exponent];
  outputBits[index] = static_cast<uint64_t>(__double_as_longlong(decoded));
}

__global__ void patchExceptionsKernel(
    const uint32_t* __restrict__ positions,
    const uint64_t* __restrict__ values,
    uint32_t exceptionCount,
    uint64_t* __restrict__ outputBits) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < exceptionCount) {
    outputBits[positions[index]] = values[index];
  }
}

void selectFloat64AlpParameters(
    const double* input,
    uint32_t numValues,
    void* workspace,
    rmm::cuda_stream_view stream) {
  const uint32_t sampleCount = std::min(numValues, kProbeValues);
  auto* probeResults = static_cast<ProbeResult*>(workspace);
  auto* state = reinterpret_cast<PlaneSelectionState*>(
      static_cast<uint8_t*>(workspace) + kPlaneStateOffset);
  probeKernel<<<kProbePairCount, kThreads, 0, stream.value()>>>(
      input, numValues, sampleCount, probeResults);
  UCX_ALP_CUDA_CHECK(cudaGetLastError());
  selectParametersKernel<<<1, 1, 0, stream.value()>>>(
      probeResults, sampleCount, &state->stats, &state->exceptionCount);
  UCX_ALP_CUDA_CHECK(cudaGetLastError());
}

CodecStats analyzeFloat64Alp(
    const double* input,
    int64_t* encodedValues,
    uint32_t numValues,
    rmm::cuda_stream_view stream) {
  const uint32_t sampleCount = std::min(numValues, kProbeValues);
  rmm::device_buffer probeResults(
      kProbePairCount * sizeof(ProbeResult), stream);
  rmm::device_buffer statsDevice(sizeof(CodecStats), stream);
  probeKernel<<<kProbePairCount, kThreads, 0, stream.value()>>>(
      input,
      numValues,
      sampleCount,
      static_cast<ProbeResult*>(probeResults.data()));
  UCX_ALP_CUDA_CHECK(cudaGetLastError());
  selectParametersKernel<<<1, 1, 0, stream.value()>>>(
      static_cast<const ProbeResult*>(probeResults.data()),
      sampleCount,
      static_cast<CodecStats*>(statsDevice.data()),
      nullptr);
  UCX_ALP_CUDA_CHECK(cudaGetLastError());

  const uint32_t blocks = (numValues + kThreads - 1) / kThreads;
  transformKernel<<<blocks, kThreads, 0, stream.value()>>>(
      input,
      encodedValues,
      numValues,
      static_cast<CodecStats*>(statsDevice.data()));
  UCX_ALP_CUDA_CHECK(cudaGetLastError());

  CodecStats stats;
  UCX_ALP_CUDA_CHECK(cudaMemcpyAsync(
      &stats,
      statsDevice.data(),
      sizeof(stats),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_ALP_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  return stats;
}

} // namespace

Float64AlpTransformResult transformFloat64Alp(
    const double* input,
    uint32_t numValues,
    rmm::cuda_stream_view stream) {
  Float64AlpTransformResult result;
  result.numValues = numValues;
  result.inputBytes = static_cast<std::size_t>(numValues) * sizeof(double);
  if (numValues == 0) {
    return result;
  }
  if (input == nullptr) {
    throw std::invalid_argument("UCX FP64 ALP codec received a null input");
  }

  result.encodedValues = rmm::device_buffer(
      static_cast<std::size_t>(numValues) * sizeof(int64_t), stream);
  const auto stats = analyzeFloat64Alp(
      input,
      static_cast<int64_t*>(result.encodedValues.data()),
      numValues,
      stream);
  const uint32_t blocks = (numValues + kThreads - 1) / kThreads;
  result.exponentIndex = stats.selection.exponent;
  result.factorIndex = stats.selection.factor;
  result.exceptionCount = stats.full.exceptions;
  if (stats.full.exceptions == numValues) {
    result.base = 0;
    result.bitWidth = 0;
  } else {
    result.base = stats.full.minimum;
    const uint64_t range = static_cast<uint64_t>(stats.full.maximum) -
        static_cast<uint64_t>(stats.full.minimum);
    result.bitWidth = requiredBits(range);
  }

  const std::size_t positionsWireSize = roundUp16(
      static_cast<std::size_t>(result.exceptionCount) * sizeof(uint32_t));
  const std::size_t valuesWireSize = roundUp16(
      static_cast<std::size_t>(result.exceptionCount) * sizeof(uint64_t));
  result.exceptionBytes = positionsWireSize + valuesWireSize;
  result.exceptionData = rmm::device_buffer(result.exceptionBytes, stream);
  if (result.exceptionBytes != 0) {
    UCX_ALP_CUDA_CHECK(cudaMemsetAsync(
        result.exceptionData.data(),
        0,
        result.exceptionData.size(),
        stream.value()));
    auto* positions = static_cast<uint32_t*>(result.exceptionData.data());
    auto* values = reinterpret_cast<uint64_t*>(
        static_cast<uint8_t*>(result.exceptionData.data()) + positionsWireSize);
    rmm::device_buffer writeCount(sizeof(uint32_t), stream);
    UCX_ALP_CUDA_CHECK(cudaMemsetAsync(
        writeCount.data(), 0, writeCount.size(), stream.value()));
    compactExceptionsKernel<<<blocks, kThreads, 0, stream.value()>>>(
        reinterpret_cast<const uint64_t*>(input),
        static_cast<const int64_t*>(result.encodedValues.data()),
        numValues,
        positions,
        values,
        static_cast<uint32_t*>(writeCount.data()));
    UCX_ALP_CUDA_CHECK(cudaGetLastError());
  }
  return result;
}

Float64AlpPlaneResult encodeFloat64AlpPlanes(
    const double* input,
    uint32_t numValues,
    uint32_t planeStride,
    rmm::cuda_stream_view stream) {
  Float64AlpPlaneResult result;
  result.numValues = numValues;
  result.planeStride = planeStride;
  result.inputBytes = static_cast<std::size_t>(numValues) * sizeof(double);
  if (numValues == 0) {
    return result;
  }
  if (input == nullptr || planeStride < numValues) {
    throw std::invalid_argument("UCX FP64 ALP plane input is invalid");
  }

  const std::size_t maxPlaneBytes =
      static_cast<std::size_t>(sizeof(int64_t)) * planeStride;
  result.planes =
      rmm::device_buffer(maxPlaneBytes + kPlaneWorkspaceBytes, stream);
  auto* workspace = static_cast<uint8_t*>(result.planes.data()) + maxPlaneBytes;
  selectFloat64AlpParameters(input, numValues, workspace, stream);
  auto* state =
      reinterpret_cast<PlaneSelectionState*>(workspace + kPlaneStateOffset);
  const uint32_t blocks = (numValues + kThreads - 1) / kThreads;
  splitAndCompactKernel<<<blocks, kThreads, 0, stream.value()>>>(
      input,
      numValues,
      &state->stats,
      0,
      0,
      0,
      static_cast<uint8_t*>(result.planes.data()),
      result.planeStride,
      0,
      nullptr,
      nullptr,
      &state->exceptionCount);
  UCX_ALP_CUDA_CHECK(cudaGetLastError());

  PlaneSelectionState hostState;
  UCX_ALP_CUDA_CHECK(cudaMemcpyAsync(
      &hostState,
      state,
      sizeof(hostState),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_ALP_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  result.exponentIndex = hostState.stats.selection.exponent;
  result.factorIndex = hostState.stats.selection.factor;
  result.base = hostState.stats.planeBase;
  result.bitWidth = hostState.stats.planeBitWidth;
  result.planeWidth = hostState.stats.planeWidth;
  result.exceptionCount = hostState.exceptionCount;
  result.planes.resize(
      static_cast<std::size_t>(result.planeWidth) * planeStride, stream);
  return result;
}

void finalizeFloat64AlpExceptions(
    const double* input,
    uint32_t exceptionCount,
    Float64AlpPlaneResult& result,
    rmm::cuda_stream_view stream) {
  const std::size_t maxPlaneBytes =
      static_cast<std::size_t>(sizeof(int64_t)) * result.planeStride;
  if (input == nullptr || exceptionCount > result.numValues ||
      result.planes.capacity() < maxPlaneBytes + kPlaneWorkspaceBytes) {
    throw std::invalid_argument("UCX FP64 ALP exception state is invalid");
  }
  result.exceptionCount = exceptionCount;
  const std::size_t positionsWireSize =
      roundUp16(static_cast<std::size_t>(exceptionCount) * sizeof(uint32_t));
  const std::size_t valuesWireSize =
      roundUp16(static_cast<std::size_t>(exceptionCount) * sizeof(uint64_t));
  result.exceptionBytes = positionsWireSize + valuesWireSize;
  result.exceptionData = rmm::device_buffer(result.exceptionBytes, stream);
  if (exceptionCount == 0) {
    return;
  }

  auto* positions = static_cast<uint32_t*>(result.exceptionData.data());
  auto* exceptionValues = reinterpret_cast<uint64_t*>(
      static_cast<uint8_t*>(result.exceptionData.data()) + positionsWireSize);
  auto* state = reinterpret_cast<PlaneSelectionState*>(
      static_cast<uint8_t*>(result.planes.data()) + maxPlaneBytes +
      kPlaneStateOffset);
  UCX_ALP_CUDA_CHECK(cudaMemsetAsync(
      &state->exceptionCount,
      0,
      sizeof(state->exceptionCount),
      stream.value()));
  const uint32_t blocks = (result.numValues + kThreads - 1) / kThreads;
  splitAndCompactKernel<<<blocks, kThreads, 0, stream.value()>>>(
      input,
      result.numValues,
      nullptr,
      result.exponentIndex,
      result.factorIndex,
      result.base,
      nullptr,
      result.planeStride,
      result.planeWidth,
      positions,
      exceptionValues,
      &state->exceptionCount);
  UCX_ALP_CUDA_CHECK(cudaGetLastError());
}

Float64AlpCompressResult compressFloat64Alp(
    const double* input,
    uint32_t numValues,
    rmm::cuda_stream_view stream,
    double minGain) {
  if (minGain < 0.0 || minGain >= 1.0) {
    throw std::invalid_argument("UCX FP64 ALP minGain must be in [0, 1)");
  }

  auto transformed = transformFloat64Alp(input, numValues, stream);
  Float64AlpCompressResult result;
  result.numValues = transformed.numValues;
  result.exceptionCount = transformed.exceptionCount;
  result.exponentIndex = transformed.exponentIndex;
  result.factorIndex = transformed.factorIndex;
  result.bitWidth = transformed.bitWidth;
  result.base = transformed.base;
  result.inputBytes = transformed.inputBytes;
  if (numValues == 0) {
    result.used = true;
    return result;
  }

  const std::size_t packedSize = packedBytes(numValues, result.bitWidth);
  const std::size_t packedWireSize = roundUp16(packedSize);
  result.candidateBytes = packedWireSize + transformed.exceptionBytes;
  result.data = rmm::device_buffer(result.candidateBytes, stream);
  if (result.candidateBytes != 0) {
    UCX_ALP_CUDA_CHECK(cudaMemsetAsync(
        result.data.data(), 0, result.data.size(), stream.value()));
  }

  if (result.bitWidth != 0) {
    const uint32_t groups = (numValues + 31) / 32;
    const uint32_t packBlocks = (groups * 32 + kThreads - 1) / kThreads;
    packWarpBitPlanesKernel<<<packBlocks, kThreads, 0, stream.value()>>>(
        static_cast<const int64_t*>(transformed.encodedValues.data()),
        numValues,
        result.base,
        result.bitWidth,
        static_cast<uint32_t*>(result.data.data()));
    UCX_ALP_CUDA_CHECK(cudaGetLastError());
  }
  if (transformed.exceptionBytes != 0) {
    UCX_ALP_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(result.data.data()) + packedWireSize,
        transformed.exceptionData.data(),
        transformed.exceptionBytes,
        cudaMemcpyDeviceToDevice,
        stream.value()));
  }

  result.used = static_cast<double>(result.candidateBytes) <=
      (1.0 - minGain) * result.inputBytes;
  return result;
}

void reconstructFloat64AlpIntegersInto(
    const int64_t* encodedValues,
    const void* exceptionData,
    uint32_t numValues,
    uint32_t exceptionCount,
    uint32_t exponentIndex,
    uint32_t factorIndex,
    double* output,
    rmm::cuda_stream_view stream) {
  if (exponentIndex >= kNumExponents || factorIndex > exponentIndex ||
      exceptionCount > numValues) {
    throw std::invalid_argument("UCX FP64 ALP metadata is invalid");
  }
  if (numValues == 0) {
    return;
  }
  if (encodedValues == nullptr || output == nullptr ||
      (exceptionCount != 0 && exceptionData == nullptr)) {
    throw std::invalid_argument("UCX FP64 ALP reconstruction input is null");
  }

  const uint32_t blocks = (numValues + kThreads - 1) / kThreads;
  decodeIntegersKernel<<<blocks, kThreads, 0, stream.value()>>>(
      encodedValues,
      reinterpret_cast<uint64_t*>(output),
      numValues,
      exponentIndex,
      factorIndex);
  UCX_ALP_CUDA_CHECK(cudaGetLastError());

  if (exceptionCount != 0) {
    const auto* positions = static_cast<const uint32_t*>(exceptionData);
    const auto* values = reinterpret_cast<const uint64_t*>(
        static_cast<const uint8_t*>(exceptionData) +
        roundUp16(static_cast<std::size_t>(exceptionCount) * sizeof(uint32_t)));
    const uint32_t patchBlocks = (exceptionCount + kThreads - 1) / kThreads;
    patchExceptionsKernel<<<patchBlocks, kThreads, 0, stream.value()>>>(
        positions, values, exceptionCount, reinterpret_cast<uint64_t*>(output));
    UCX_ALP_CUDA_CHECK(cudaGetLastError());
  }
}

void reconstructFloat64AlpRawPlanesInto(
    const uint8_t* planes,
    uint32_t planeStride,
    uint32_t planeWidth,
    int64_t base,
    const void* exceptionData,
    uint32_t numValues,
    uint32_t exceptionCount,
    uint32_t exponentIndex,
    uint32_t factorIndex,
    double* output,
    rmm::cuda_stream_view stream) {
  if (planeWidth == 0 || planeWidth > 4 || planeStride < numValues ||
      exponentIndex >= kNumExponents || factorIndex > exponentIndex ||
      exceptionCount > numValues || planes == nullptr || output == nullptr ||
      (exceptionCount != 0 && exceptionData == nullptr)) {
    throw std::invalid_argument("UCX FP64 ALP raw-plane input is invalid");
  }
  const uint32_t blocks = (numValues + kThreads - 1) / kThreads;
  switch (planeWidth) {
    case 1:
      decodeRawPlanesKernel<1><<<blocks, kThreads, 0, stream.value()>>>(
          planes,
          reinterpret_cast<uint64_t*>(output),
          numValues,
          planeStride,
          base,
          exponentIndex,
          factorIndex);
      break;
    case 2:
      decodeRawPlanesKernel<2><<<blocks, kThreads, 0, stream.value()>>>(
          planes,
          reinterpret_cast<uint64_t*>(output),
          numValues,
          planeStride,
          base,
          exponentIndex,
          factorIndex);
      break;
    case 3:
      decodeRawPlanesKernel<3><<<blocks, kThreads, 0, stream.value()>>>(
          planes,
          reinterpret_cast<uint64_t*>(output),
          numValues,
          planeStride,
          base,
          exponentIndex,
          factorIndex);
      break;
    case 4:
      decodeRawPlanesKernel<4><<<blocks, kThreads, 0, stream.value()>>>(
          planes,
          reinterpret_cast<uint64_t*>(output),
          numValues,
          planeStride,
          base,
          exponentIndex,
          factorIndex);
      break;
  }
  UCX_ALP_CUDA_CHECK(cudaGetLastError());
  if (exceptionCount != 0) {
    const auto* positions = static_cast<const uint32_t*>(exceptionData);
    const auto* values = reinterpret_cast<const uint64_t*>(
        static_cast<const uint8_t*>(exceptionData) +
        roundUp16(static_cast<std::size_t>(exceptionCount) * sizeof(uint32_t)));
    const uint32_t patchBlocks = (exceptionCount + kThreads - 1) / kThreads;
    patchExceptionsKernel<<<patchBlocks, kThreads, 0, stream.value()>>>(
        positions, values, exceptionCount, reinterpret_cast<uint64_t*>(output));
    UCX_ALP_CUDA_CHECK(cudaGetLastError());
  }
}

void decompressFloat64AlpPayloadInto(
    const void* data,
    std::size_t candidateBytes,
    uint32_t numValues,
    uint32_t exceptionCount,
    uint32_t exponentIndex,
    uint32_t factorIndex,
    uint32_t bitWidth,
    int64_t base,
    double* output,
    rmm::cuda_stream_view stream) {
  if (exponentIndex >= kNumExponents || factorIndex > exponentIndex ||
      bitWidth > 64 || exceptionCount > numValues) {
    throw std::invalid_argument("UCX FP64 ALP metadata is invalid");
  }
  if (numValues == 0) {
    return;
  }
  if ((candidateBytes != 0 && data == nullptr) || output == nullptr) {
    throw std::invalid_argument("UCX FP64 ALP payload or output is null");
  }

  const std::size_t packedSize = packedBytes(numValues, bitWidth);
  const std::size_t packedWireSize = roundUp16(packedSize);
  const std::size_t positionsWireSize =
      roundUp16(static_cast<std::size_t>(exceptionCount) * sizeof(uint32_t));
  const std::size_t valuesWireSize =
      roundUp16(static_cast<std::size_t>(exceptionCount) * sizeof(uint64_t));
  if (packedWireSize + positionsWireSize + valuesWireSize != candidateBytes) {
    throw std::invalid_argument("UCX FP64 ALP payload size is invalid");
  }

  const uint32_t blocks = (numValues + kThreads - 1) / kThreads;
  decodeKernel<<<blocks, kThreads, 0, stream.value()>>>(
      static_cast<const uint32_t*>(data),
      reinterpret_cast<uint64_t*>(output),
      numValues,
      base,
      bitWidth,
      exponentIndex,
      factorIndex);
  UCX_ALP_CUDA_CHECK(cudaGetLastError());

  if (exceptionCount != 0) {
    const auto* positions = reinterpret_cast<const uint32_t*>(
        static_cast<const uint8_t*>(data) + packedWireSize);
    const auto* values = reinterpret_cast<const uint64_t*>(
        static_cast<const uint8_t*>(data) + packedWireSize + positionsWireSize);
    const uint32_t patchBlocks = (exceptionCount + kThreads - 1) / kThreads;
    patchExceptionsKernel<<<patchBlocks, kThreads, 0, stream.value()>>>(
        positions, values, exceptionCount, reinterpret_cast<uint64_t*>(output));
    UCX_ALP_CUDA_CHECK(cudaGetLastError());
  }
}

rmm::device_buffer decompressFloat64AlpPayload(
    const void* data,
    std::size_t candidateBytes,
    uint32_t numValues,
    uint32_t exceptionCount,
    uint32_t exponentIndex,
    uint32_t factorIndex,
    uint32_t bitWidth,
    int64_t base,
    rmm::cuda_stream_view stream) {
  rmm::device_buffer output(
      static_cast<std::size_t>(numValues) * sizeof(double), stream);
  decompressFloat64AlpPayloadInto(
      data,
      candidateBytes,
      numValues,
      exceptionCount,
      exponentIndex,
      factorIndex,
      bitWidth,
      base,
      static_cast<double*>(output.data()),
      stream);
  UCX_ALP_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  return output;
}

rmm::device_buffer decompressFloat64Alp(
    const Float64AlpCompressResult& compressed,
    rmm::cuda_stream_view stream) {
  if (compressed.inputBytes !=
      static_cast<std::size_t>(compressed.numValues) * sizeof(double)) {
    throw std::invalid_argument(
        "UCX FP64 ALP input size does not match its value count");
  }
  return decompressFloat64AlpPayload(
      compressed.data.data(),
      compressed.candidateBytes,
      compressed.numValues,
      compressed.exceptionCount,
      compressed.exponentIndex,
      compressed.factorIndex,
      compressed.bitWidth,
      compressed.base,
      stream);
}

} // namespace facebook::velox::ucx_exchange
