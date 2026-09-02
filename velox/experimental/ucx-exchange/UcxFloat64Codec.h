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
#include <vector>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

namespace dietgpu {
class StackDeviceMemory;
}

namespace facebook::velox::ucx_exchange {

/// Byte-exact FP64 codec modeled on DietGPU's FP32 transform. Rotating the
/// IEEE-754 word left by one places the exponent in the high-order bytes. The
/// first one or two byte planes are entropy coded and the mantissa-bearing
/// remainder is copied without expansion.
struct Float64CompressResult {
  rmm::device_buffer data;
  std::vector<uint32_t> exponentSegmentSizes;
  uint32_t exponentPlanes{0};
  uint32_t numValues{0};
  std::size_t inputBytes{0};
  std::size_t candidateBytes{0};
};

/// `exponentPlanes` must be one or two. The returned payload contains the
/// padded rANS exponent segments followed by the raw residual byte planes.
Float64CompressResult compressFloat64(
    const double* input,
    uint32_t numValues,
    uint32_t exponentPlanes,
    rmm::cuda_stream_view stream);

/// Uses caller-owned DietGPU scratch so multiple regions in one packed batch
/// do not allocate and free a large arena for every region.
Float64CompressResult compressFloat64WithScratch(
    const double* input,
    uint32_t numValues,
    uint32_t exponentPlanes,
    rmm::cuda_stream_view stream,
    dietgpu::StackDeviceMemory& scratch);

/// Reconstructs every input bit, including NaN payloads and signed zero.
void decompressFloat64PayloadInto(
    const void* data,
    std::size_t candidateBytes,
    const std::vector<uint32_t>& exponentSegmentSizes,
    uint32_t exponentPlanes,
    uint32_t numValues,
    double* output,
    rmm::cuda_stream_view stream);

void decompressFloat64PayloadIntoWithScratch(
    const void* data,
    std::size_t candidateBytes,
    const std::vector<uint32_t>& exponentSegmentSizes,
    uint32_t exponentPlanes,
    uint32_t numValues,
    double* output,
    rmm::cuda_stream_view stream,
    dietgpu::StackDeviceMemory& scratch);

rmm::device_buffer decompressFloat64Payload(
    const void* data,
    std::size_t candidateBytes,
    const std::vector<uint32_t>& exponentSegmentSizes,
    uint32_t exponentPlanes,
    uint32_t numValues,
    rmm::cuda_stream_view stream);

rmm::device_buffer decompressFloat64(
    const Float64CompressResult& compressed,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::ucx_exchange
