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

// KEY MILESTONE TEST: First Velox simple function running on GPU.
// Instantiates PlusFunction<GpuExec> via GpuSimpleFunctionAdapter
// and verifies correct results from a CUDA kernel.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cudf/null_mask.hpp>

#include "velox/experimental/cudf/functions/GpuSimpleFunctionAdapter.cuh"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Comparisons.h"

using namespace facebook::velox::gpu;

namespace {

template <typename T>
T* deviceAlloc(size_t count) {
  T* ptr = nullptr;
  cudaMalloc(&ptr, count * sizeof(T));
  return ptr;
}

template <typename T>
void toDevice(T* dst, const T* src, size_t count) {
  cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void toHost(T* dst, const T* src, size_t count) {
  cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost);
}

size_t bitmaskWords(int n) {
  return (n + 31) / 32;
}

} // namespace

class GpuAdapterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaSetDevice(0);
  }
};

TEST_F(GpuAdapterTest, plusDoubleOnGpu) {
  constexpr int N = 8;
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  std::vector<double> b = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};
  std::vector<double> expected = {
      11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0};

  auto* dA = deviceAlloc<double>(N);
  auto* dB = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dNullMask = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  GpuColumnReader<double> readerA{cudf::device_span<const double>(dA, N)};
  GpuColumnReader<double> readerB{cudf::device_span<const double>(dB, N)};
  GpuColumnWriter<double> writer{dOut, dNullMask};

  using PlusFn = facebook::velox::functions::PlusFunction<GpuExec>;
  GpuSimpleFunctionAdapter<PlusFn, double, double, double>::apply(
      writer, readerA, readerB, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<double> result(N);
  toHost(result.data(), dOut, N);

  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(expected[i], result[i]) << "mismatch at row " << i;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  cudaFree(dNullMask);
}

TEST_F(GpuAdapterTest, plusInt64OnGpu) {
  constexpr int N = 4;
  std::vector<int64_t> a = {100, -50, 0, 9223372036854775806LL};
  std::vector<int64_t> b = {200, 50, 0, 1};
  std::vector<int64_t> expected = {300, 0, 0, 9223372036854775807LL};

  auto* dA = deviceAlloc<int64_t>(N);
  auto* dB = deviceAlloc<int64_t>(N);
  auto* dOut = deviceAlloc<int64_t>(N);
  auto* dNullMask = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  GpuColumnReader<int64_t> readerA{cudf::device_span<const int64_t>(dA, N)};
  GpuColumnReader<int64_t> readerB{cudf::device_span<const int64_t>(dB, N)};
  GpuColumnWriter<int64_t> writer{dOut, dNullMask};

  using PlusFn = facebook::velox::functions::PlusFunction<GpuExec>;
  GpuSimpleFunctionAdapter<PlusFn, int64_t, int64_t, int64_t>::apply(
      writer, readerA, readerB, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<int64_t> result(N);
  toHost(result.data(), dOut, N);

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(expected[i], result[i]) << "mismatch at row " << i;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  cudaFree(dNullMask);
}

TEST_F(GpuAdapterTest, nullPropagation) {
  constexpr int N = 4;
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> b = {10.0, 20.0, 30.0, 40.0};

  auto* dA = deviceAlloc<double>(N);
  auto* dB = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dOutNullMask = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));
  auto* dCombinedNull = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  // Row 1 is null (bit 1 cleared)
  std::vector<cudf::bitmask_type> nullMask = {0b00001101};
  toDevice(dCombinedNull, nullMask.data(), 1);

  // Initialize output null mask to all-valid
  std::vector<cudf::bitmask_type> allValid = {0xFFFFFFFF};
  toDevice(dOutNullMask, allValid.data(), 1);

  GpuColumnReader<double> readerA{cudf::device_span<const double>(dA, N)};
  GpuColumnReader<double> readerB{cudf::device_span<const double>(dB, N)};
  GpuColumnWriter<double> writer{dOut, dOutNullMask};

  using PlusFn = facebook::velox::functions::PlusFunction<GpuExec>;
  GpuSimpleFunctionAdapter<PlusFn, double, double, double>::apply(
      writer, readerA, readerB, dCombinedNull, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<double> result(N);
  toHost(result.data(), dOut, N);
  std::vector<cudf::bitmask_type> resultNullMask(1);
  toHost(resultNullMask.data(), dOutNullMask, 1);

  // Row 0: valid -> 11.0
  EXPECT_DOUBLE_EQ(11.0, result[0]);
  // Row 1: null input -> output should be null (bit 1 cleared)
  EXPECT_FALSE(cudf::bit_is_set(resultNullMask.data(), 1));
  // Row 2: valid -> 33.0
  EXPECT_DOUBLE_EQ(33.0, result[2]);
  // Row 3: valid -> 44.0
  EXPECT_DOUBLE_EQ(44.0, result[3]);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  cudaFree(dOutNullMask);
  cudaFree(dCombinedNull);
}

TEST_F(GpuAdapterTest, ltComparisonOnGpu) {
  constexpr int N = 4;
  std::vector<double> a = {1.0, 5.0, 3.0, 3.0};
  std::vector<double> b = {2.0, 3.0, 3.0, 1.0};
  std::vector<bool> expected = {true, false, false, false};

  auto* dA = deviceAlloc<double>(N);
  auto* dB = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<bool>(N);
  auto* dNullMask = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  GpuColumnReader<double> readerA{cudf::device_span<const double>(dA, N)};
  GpuColumnReader<double> readerB{cudf::device_span<const double>(dB, N)};
  GpuColumnWriter<bool> writer{dOut, dNullMask};

  using LtFn = facebook::velox::functions::LtFunction<GpuExec>;
  GpuSimpleFunctionAdapter<LtFn, bool, double, double>::apply(
      writer, readerA, readerB, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<bool> result(N);
  bool hostBuf[N];
  toHost(hostBuf, dOut, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(expected[i], hostBuf[i]) << "mismatch at row " << i;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  cudaFree(dNullMask);
}
