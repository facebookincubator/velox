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

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <rmm/cuda_stream.hpp>

namespace facebook::velox::ucx_exchange {
namespace {

void checkCuda(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

void expectBitExactRoundTrip(
    const std::vector<uint64_t>& inputBits,
    uint32_t exponentPlanes) {
  rmm::cuda_stream stream;
  rmm::device_buffer input(
      inputBits.data(), inputBits.size() * sizeof(uint64_t), stream.view());
  auto compressed = compressFloat64(
      reinterpret_cast<const double*>(input.data()),
      static_cast<uint32_t>(inputBits.size()),
      exponentPlanes,
      stream.view());
  auto output = decompressFloat64(compressed, stream.view());

  std::vector<uint64_t> outputBits(inputBits.size());
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpy(
          outputBits.data(),
          output.data(),
          output.size(),
          cudaMemcpyDeviceToHost));
  EXPECT_EQ(outputBits, inputBits);
}

TEST(UcxFloat64CodecTest, preservesEveryBitWithOneOrTwoPlanes) {
  std::vector<uint64_t> bits{
      0x0000000000000000ULL, // +0
      0x8000000000000000ULL, // -0
      0x3ff0000000000000ULL, // +1
      0xbff0000000000000ULL, // -1
      0x0010000000000000ULL, // minimum normal
      0x0000000000000001ULL, // minimum subnormal
      0x7fefffffffffffffULL, // maximum finite
      0xffefffffffffffffULL, // minimum finite
      0x7ff0000000000000ULL, // +infinity
      0xfff0000000000000ULL, // -infinity
      0x7ff8000000000042ULL, // quiet NaN with payload
      0xfff0000000000042ULL, // negative signaling NaN payload
  };

  uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (uint32_t index = 0; index < 65'537; ++index) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    bits.push_back(state);
  }

  expectBitExactRoundTrip(bits, 1);
  expectBitExactRoundTrip(bits, 2);
}

TEST(UcxFloat64CodecTest, validatesPlaneCount) {
  rmm::cuda_stream stream;
  EXPECT_THROW(
      compressFloat64(nullptr, 0, 0, stream.view()), std::invalid_argument);
  EXPECT_THROW(
      compressFloat64(nullptr, 0, 3, stream.view()), std::invalid_argument);
}

} // namespace
} // namespace facebook::velox::ucx_exchange
