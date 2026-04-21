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

// Tests for GPU bitmask operations and special form kernels.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cudf/utilities/bit.hpp>

#include "velox/experimental/cudf/functions/GpuBitmaskOps.cuh"

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

} // namespace

class GpuBitmaskOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaSetDevice(0);
  }
};

TEST_F(GpuBitmaskOpsTest, bitmaskAndOrNot) {
  // 8 rows: lhs = 0b10110101, rhs = 0b11001100
  cudf::bitmask_type lhsVal = 0b10110101;
  cudf::bitmask_type rhsVal = 0b11001100;

  auto* dLhs = deviceAlloc<cudf::bitmask_type>(1);
  auto* dRhs = deviceAlloc<cudf::bitmask_type>(1);
  auto* dOut = deviceAlloc<cudf::bitmask_type>(1);

  toDevice(dLhs, &lhsVal, 1);
  toDevice(dRhs, &rhsVal, 1);

  // AND
  launchBitmaskAnd(dOut, dLhs, dRhs, 8);
  cudaDeviceSynchronize();
  cudf::bitmask_type result;
  toHost(&result, dOut, 1);
  EXPECT_EQ(0b10000100u, result & 0xFF);

  // OR
  launchBitmaskOr(dOut, dLhs, dRhs, 8);
  cudaDeviceSynchronize();
  toHost(&result, dOut, 1);
  EXPECT_EQ(0b11111101u, result & 0xFF);

  // NOT (lhs)
  launchBitmaskNot(dOut, dLhs, 8);
  cudaDeviceSynchronize();
  toHost(&result, dOut, 1);
  EXPECT_EQ(0b01001010u, result & 0xFF);

  cudaFree(dLhs);
  cudaFree(dRhs);
  cudaFree(dOut);
}

TEST_F(GpuBitmaskOpsTest, bitmaskAndNot) {
  cudf::bitmask_type lhsVal = 0b11111111;
  cudf::bitmask_type rhsVal = 0b00001010;

  auto* dLhs = deviceAlloc<cudf::bitmask_type>(1);
  auto* dRhs = deviceAlloc<cudf::bitmask_type>(1);
  auto* dOut = deviceAlloc<cudf::bitmask_type>(1);

  toDevice(dLhs, &lhsVal, 1);
  toDevice(dRhs, &rhsVal, 1);

  cudf::size_type words = 1;
  bitmaskAndNot<<<1, 32>>>(dOut, dLhs, dRhs, words);
  cudaDeviceSynchronize();

  cudf::bitmask_type result;
  toHost(&result, dOut, 1);
  EXPECT_EQ(0b11110101u, result & 0xFF);

  cudaFree(dLhs);
  cudaFree(dRhs);
  cudaFree(dOut);
}

TEST_F(GpuBitmaskOpsTest, ifKernelDouble) {
  constexpr int N = 4;
  // condition: true, false, true, null
  bool condHost[N] = {true, false, true, false};
  double thenHost[N] = {1.0, 2.0, 3.0, 4.0};
  double elseHost[N] = {10.0, 20.0, 30.0, 40.0};
  // Expected: if(T)=1, if(F)=20, if(T)=3, if(null)=40

  auto* dCond = deviceAlloc<bool>(N);
  auto* dThen = deviceAlloc<double>(N);
  auto* dElse = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dOutNull = deviceAlloc<cudf::bitmask_type>(1);
  auto* dCondNull = deviceAlloc<cudf::bitmask_type>(1);

  toDevice(dCond, condHost, N);
  toDevice(dThen, thenHost, N);
  toDevice(dElse, elseHost, N);

  // Row 3 has null condition (bit 3 cleared)
  cudf::bitmask_type condNullVal = 0b00000111;
  toDevice(dCondNull, &condNullVal, 1);

  // Init output null to all-valid
  cudf::bitmask_type allValid = 0xFFFFFFFF;
  toDevice(dOutNull, &allValid, 1);

  int blocks = (N + 255) / 256;
  gpuIfKernel<double><<<blocks, 256>>>(
      dOut, dOutNull, dCond, dCondNull,
      dThen, nullptr, dElse, nullptr, N);
  cudaDeviceSynchronize();

  double resultHost[N];
  toHost(resultHost, dOut, N);

  EXPECT_DOUBLE_EQ(1.0, resultHost[0]);   // cond=T -> then
  EXPECT_DOUBLE_EQ(20.0, resultHost[1]);  // cond=F -> else
  EXPECT_DOUBLE_EQ(3.0, resultHost[2]);   // cond=T -> then
  EXPECT_DOUBLE_EQ(40.0, resultHost[3]);  // cond=null -> else

  cudaFree(dCond);
  cudaFree(dThen);
  cudaFree(dElse);
  cudaFree(dOut);
  cudaFree(dOutNull);
  cudaFree(dCondNull);
}

TEST_F(GpuBitmaskOpsTest, coalesceKernelDouble) {
  constexpr int N = 4;
  // col0: [null, 2.0, null, 4.0]
  // col1: [10.0, null, 30.0, null]
  // col2: [100.0, 200.0, 300.0, 400.0]
  // Expected coalesce: [10.0, 2.0, 30.0, 4.0]

  double col0[N] = {0.0, 2.0, 0.0, 4.0};
  double col1[N] = {10.0, 0.0, 30.0, 0.0};
  double col2[N] = {100.0, 200.0, 300.0, 400.0};

  cudf::bitmask_type null0 = 0b00001010; // rows 1,3 valid
  cudf::bitmask_type null1 = 0b00000101; // rows 0,2 valid
  // col2 is all-valid (no null mask)

  auto* dCol0 = deviceAlloc<double>(N);
  auto* dCol1 = deviceAlloc<double>(N);
  auto* dCol2 = deviceAlloc<double>(N);
  auto* dNull0 = deviceAlloc<cudf::bitmask_type>(1);
  auto* dNull1 = deviceAlloc<cudf::bitmask_type>(1);
  auto* dOut = deviceAlloc<double>(N);
  auto* dOutNull = deviceAlloc<cudf::bitmask_type>(1);

  toDevice(dCol0, col0, N);
  toDevice(dCol1, col1, N);
  toDevice(dCol2, col2, N);
  toDevice(dNull0, &null0, 1);
  toDevice(dNull1, &null1, 1);

  cudf::bitmask_type initNull = 0xFFFFFFFF;
  toDevice(dOutNull, &initNull, 1);

  // Copy device pointers to host arrays, then to device
  auto* dColPtrs = deviceAlloc<const double*>(3);
  auto* dNullPtrs = deviceAlloc<const cudf::bitmask_type*>(3);

  const double* hColPtrs[3];
  const cudf::bitmask_type* hNullPtrs[3];
  cudaMemcpy(&hColPtrs[0], &dCol0, sizeof(double*), cudaMemcpyHostToHost);
  cudaMemcpy(&hColPtrs[1], &dCol1, sizeof(double*), cudaMemcpyHostToHost);
  cudaMemcpy(&hColPtrs[2], &dCol2, sizeof(double*), cudaMemcpyHostToHost);
  cudaMemcpy(&hNullPtrs[0], &dNull0, sizeof(cudf::bitmask_type*), cudaMemcpyHostToHost);
  cudaMemcpy(&hNullPtrs[1], &dNull1, sizeof(cudf::bitmask_type*), cudaMemcpyHostToHost);
  hNullPtrs[2] = nullptr; // col2 not nullable

  toDevice(dColPtrs, hColPtrs, 3);
  toDevice(dNullPtrs, hNullPtrs, 3);

  int blocks = (N + 255) / 256;
  gpuCoalesceKernel<double><<<blocks, 256>>>(
      dOut, dOutNull, dColPtrs, dNullPtrs, 3, N);
  cudaDeviceSynchronize();

  double resultHost[N];
  toHost(resultHost, dOut, N);

  EXPECT_DOUBLE_EQ(10.0, resultHost[0]);  // col0=null, col1=10
  EXPECT_DOUBLE_EQ(2.0, resultHost[1]);   // col0=2
  EXPECT_DOUBLE_EQ(30.0, resultHost[2]);  // col0=null, col1=30
  EXPECT_DOUBLE_EQ(4.0, resultHost[3]);   // col0=4

  cudaFree(dCol0);
  cudaFree(dCol1);
  cudaFree(dCol2);
  cudaFree(dNull0);
  cudaFree(dNull1);
  cudaFree(dOut);
  cudaFree(dOutNull);
  cudaFree(dColPtrs);
  cudaFree(dNullPtrs);
}
