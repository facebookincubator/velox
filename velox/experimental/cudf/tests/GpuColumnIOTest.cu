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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cudf/null_mask.hpp>

#include "velox/experimental/cudf/functions/GpuColumnIO.cuh"

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

__global__ void copyDoublesKernel(
    GpuColumnReader<double> reader,
    GpuColumnWriter<double> writer,
    cudf::size_type n) {
  cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n)
    return;
  writer.write(row, reader.read(row));
  writer.setValid(row);
}

__global__ void copyInt64Kernel(
    GpuColumnReader<int64_t> reader,
    GpuColumnWriter<int64_t> writer,
    cudf::size_type n) {
  cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n)
    return;
  writer.write(row, reader.read(row));
  writer.setValid(row);
}

__global__ void readStringsKernel(
    GpuColumnReader<GpuStringView> reader,
    int32_t* outSizes,
    cudf::size_type n) {
  cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n)
    return;
  outSizes[row] = reader.read(row).size();
}

__global__ void readTimestampsKernel(
    GpuColumnReader<GpuTimestamp> reader,
    int64_t* outSeconds,
    uint64_t* outNanos,
    cudf::size_type n) {
  cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n)
    return;
  GpuTimestamp ts = reader.read(row);
  outSeconds[row] = ts.seconds;
  outNanos[row] = ts.nanos;
}

} // namespace

class GpuColumnIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaSetDevice(0);
  }
};

TEST_F(GpuColumnIOTest, doubleRoundTrip) {
  constexpr int N = 8;
  std::vector<double> hostInput = {
      1.0, 2.5, -3.14, 0.0, 100.0, -0.5, 42.0, 7.7};

  auto* dInput = deviceAlloc<double>(N);
  auto* dOutput = deviceAlloc<double>(N);
  auto* dNullMask = deviceAlloc<cudf::bitmask_type>(
      cudf::bitmask_allocation_size_bytes(N) / sizeof(cudf::bitmask_type));

  toDevice(dInput, hostInput.data(), N);

  GpuColumnReader<double> reader{
      cudf::device_span<const double>(dInput, N)};
  GpuColumnWriter<double> writer{dOutput, dNullMask};

  copyDoublesKernel<<<1, 256>>>(reader, writer, N);
  cudaDeviceSynchronize();

  std::vector<double> hostOutput(N);
  toHost(hostOutput.data(), dOutput, N);

  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(hostInput[i], hostOutput[i]) << "mismatch at row " << i;
  }

  cudaFree(dInput);
  cudaFree(dOutput);
  cudaFree(dNullMask);
}

TEST_F(GpuColumnIOTest, int64RoundTrip) {
  constexpr int N = 4;
  std::vector<int64_t> hostInput = {0, -1, 9223372036854775807LL, -100};

  auto* dInput = deviceAlloc<int64_t>(N);
  auto* dOutput = deviceAlloc<int64_t>(N);
  auto* dNullMask = deviceAlloc<cudf::bitmask_type>(
      cudf::bitmask_allocation_size_bytes(N) / sizeof(cudf::bitmask_type));

  toDevice(dInput, hostInput.data(), N);

  GpuColumnReader<int64_t> reader{
      cudf::device_span<const int64_t>(dInput, N)};
  GpuColumnWriter<int64_t> writer{dOutput, dNullMask};

  copyInt64Kernel<<<1, N>>>(reader, writer, N);
  cudaDeviceSynchronize();

  std::vector<int64_t> hostOutput(N);
  toHost(hostOutput.data(), dOutput, N);

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(hostInput[i], hostOutput[i]) << "mismatch at row " << i;
  }

  cudaFree(dInput);
  cudaFree(dOutput);
  cudaFree(dNullMask);
}

TEST_F(GpuColumnIOTest, stringRead) {
  constexpr int N = 4;
  std::string allChars = "hiworldgpu";
  std::vector<int32_t> offsets = {0, 2, 7, 7, 10};
  std::vector<int32_t> expectedSizes = {2, 5, 0, 3};

  auto* dChars = deviceAlloc<char>(allChars.size());
  auto* dOffsets = deviceAlloc<int32_t>(offsets.size());
  auto* dOutSizes = deviceAlloc<int32_t>(N);

  cudaMemcpy(dChars, allChars.data(), allChars.size(), cudaMemcpyHostToDevice);
  toDevice(dOffsets, offsets.data(), offsets.size());

  GpuColumnReader<GpuStringView> reader{dOffsets, dChars};

  readStringsKernel<<<1, N>>>(reader, dOutSizes, N);
  cudaDeviceSynchronize();

  std::vector<int32_t> hostSizes(N);
  toHost(hostSizes.data(), dOutSizes, N);

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(expectedSizes[i], hostSizes[i]) << "size mismatch at row " << i;
  }

  cudaFree(dChars);
  cudaFree(dOffsets);
  cudaFree(dOutSizes);
}

TEST_F(GpuColumnIOTest, timestampRead) {
  constexpr int N = 3;
  std::vector<int64_t> epochNanos = {
      1000000000LL * 100 + 500,
      0LL,
      1000000000LL * 1234567890 + 999999999};

  auto* dEpochNanos = deviceAlloc<int64_t>(N);
  auto* dOutSeconds = deviceAlloc<int64_t>(N);
  auto* dOutNanos = deviceAlloc<uint64_t>(N);

  toDevice(dEpochNanos, epochNanos.data(), N);

  GpuColumnReader<GpuTimestamp> reader{
      cudf::device_span<const int64_t>(dEpochNanos, N)};

  readTimestampsKernel<<<1, N>>>(reader, dOutSeconds, dOutNanos, N);
  cudaDeviceSynchronize();

  std::vector<int64_t> hostSeconds(N);
  std::vector<uint64_t> hostNanos(N);
  toHost(hostSeconds.data(), dOutSeconds, N);
  toHost(hostNanos.data(), dOutNanos, N);

  EXPECT_EQ(hostSeconds[0], 100);
  EXPECT_EQ(hostNanos[0], 500u);
  EXPECT_EQ(hostSeconds[1], 0);
  EXPECT_EQ(hostNanos[1], 0u);
  EXPECT_EQ(hostSeconds[2], 1234567890);
  EXPECT_EQ(hostNanos[2], 999999999u);

  cudaFree(dEpochNanos);
  cudaFree(dOutSeconds);
  cudaFree(dOutNanos);
}
