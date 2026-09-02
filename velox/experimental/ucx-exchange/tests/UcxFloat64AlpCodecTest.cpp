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

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <rmm/cuda_stream.hpp>

namespace facebook::velox::ucx_exchange {
namespace {

void checkAlpCuda(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

void expectAlpBitExactRoundTrip(const std::vector<uint64_t>& inputBits) {
  rmm::cuda_stream stream;
  rmm::device_buffer input(
      inputBits.data(), inputBits.size() * sizeof(uint64_t), stream.view());
  auto compressed = compressFloat64Alp(
      reinterpret_cast<const double*>(input.data()),
      static_cast<uint32_t>(inputBits.size()),
      stream.view());
  auto output = decompressFloat64Alp(compressed, stream.view());

  std::vector<uint64_t> outputBits(inputBits.size());
  checkAlpCuda(cudaMemcpy(
      outputBits.data(), output.data(), output.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(outputBits, inputBits);
}

TEST(UcxFloat64AlpCodecTest, preservesEveryBitIncludingExceptions) {
  std::vector<uint64_t> bits{
      0x0000000000000000ULL,
      0x8000000000000000ULL,
      0x3ff0000000000000ULL,
      0xbff0000000000000ULL,
      0x0010000000000000ULL,
      0x0000000000000001ULL,
      0x7fefffffffffffffULL,
      0xffefffffffffffffULL,
      0x7ff0000000000000ULL,
      0xfff0000000000000ULL,
      0x7ff8000000000042ULL,
      0xfff0000000000042ULL,
  };
  uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (uint32_t index = 0; index < 65'537; ++index) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    bits.push_back(state);
  }
  expectAlpBitExactRoundTrip(bits);
}

TEST(UcxFloat64AlpCodecTest, compressesDecimalLikeValues) {
  constexpr uint32_t kValues = 1u << 20;
  std::vector<double> values(kValues);
  for (uint32_t index = 0; index < kValues; ++index) {
    values[index] =
        static_cast<double>((index * 48'271ULL) % 10'000'000ULL) / 100.0;
  }

  rmm::cuda_stream stream;
  rmm::device_buffer input(
      values.data(), values.size() * sizeof(double), stream.view());
  auto compressed = compressFloat64Alp(
      static_cast<const double*>(input.data()), kValues, stream.view());
  EXPECT_TRUE(compressed.used);
  EXPECT_LE(compressed.exponentIndex, 18);
  EXPECT_LE(compressed.factorIndex, compressed.exponentIndex);
  EXPECT_LT(compressed.candidateBytes, compressed.inputBytes / 2);
  auto output = decompressFloat64Alp(compressed, stream.view());
  std::vector<double> decoded(kValues);
  checkAlpCuda(cudaMemcpy(
      decoded.data(), output.data(), output.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(0, std::memcmp(decoded.data(), values.data(), output.size()));
}

} // namespace
} // namespace facebook::velox::ucx_exchange
