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
#pragma once

#include <cstddef>
#include <cstdint>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

namespace facebook::velox::ucx_exchange {

/// GPU-oriented Adaptive Lossless floating-Point candidate for FP64. Decimal
/// values are mapped to exact integers. Values that cannot be reconstructed
/// bit-for-bit are stored as exceptions and patched by one GPU thread per
/// exception.
struct Float64AlpTransformResult {
  rmm::device_buffer encodedValues;
  rmm::device_buffer exceptionData;
  uint32_t numValues{0};
  uint32_t exceptionCount{0};
  uint32_t exponentIndex{0};
  uint32_t factorIndex{0};
  uint32_t bitWidth{0};
  int64_t base{0};
  std::size_t inputBytes{0};
  std::size_t exceptionBytes{0};
};

Float64AlpTransformResult transformFloat64Alp(
    const double* input,
    uint32_t numValues,
    rmm::cuda_stream_view stream);

/// ALP transform emitted directly as byte planes for the shared plane
/// transport. Compressible planes use DietGPU rANS and entropy-flat planes
/// remain raw. This avoids an intermediate int64 column.
struct Float64AlpPlaneResult {
  rmm::device_buffer planes;
  rmm::device_buffer exceptionData;
  uint32_t numValues{0};
  uint32_t exceptionCount{0};
  uint32_t exponentIndex{0};
  uint32_t factorIndex{0};
  uint32_t bitWidth{0};
  uint32_t planeWidth{0};
  uint32_t planeStride{0};
  int64_t base{0};
  std::size_t inputBytes{0};
  std::size_t exceptionBytes{0};
};

Float64AlpPlaneResult encodeFloat64AlpPlanes(
    const double* input,
    uint32_t numValues,
    uint32_t planeStride,
    rmm::cuda_stream_view stream);

void finalizeFloat64AlpExceptions(
    const double* input,
    uint32_t exceptionCount,
    Float64AlpPlaneResult& result,
    rmm::cuda_stream_view stream);

struct Float64AlpCompressResult {
  bool used{false};
  rmm::device_buffer data;
  uint32_t numValues{0};
  uint32_t exceptionCount{0};
  uint32_t exponentIndex{0};
  uint32_t factorIndex{0};
  uint32_t bitWidth{0};
  int64_t base{0};
  std::size_t inputBytes{0};
  std::size_t candidateBytes{0};
};

Float64AlpCompressResult compressFloat64Alp(
    const double* input,
    uint32_t numValues,
    rmm::cuda_stream_view stream,
    double minGain = 0.02);

void reconstructFloat64AlpIntegersInto(
    const int64_t* encodedValues,
    const void* exceptionData,
    uint32_t numValues,
    uint32_t exceptionCount,
    uint32_t exponentIndex,
    uint32_t factorIndex,
    double* output,
    rmm::cuda_stream_view stream);

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
    rmm::cuda_stream_view stream);

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
    rmm::cuda_stream_view stream);

rmm::device_buffer decompressFloat64AlpPayload(
    const void* data,
    std::size_t candidateBytes,
    uint32_t numValues,
    uint32_t exceptionCount,
    uint32_t exponentIndex,
    uint32_t factorIndex,
    uint32_t bitWidth,
    int64_t base,
    rmm::cuda_stream_view stream);

rmm::device_buffer decompressFloat64Alp(
    const Float64AlpCompressResult& compressed,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::ucx_exchange
