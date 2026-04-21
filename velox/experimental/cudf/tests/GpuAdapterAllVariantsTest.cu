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

// PR6 tests: Verify that GpuSimpleFunctionAdapter handles all major
// function variants -- unary, binary, ternary, and bool-returning call().

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

class GpuAdapterAllVariantsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaSetDevice(0);
  }
};

// ---- Binary Arithmetic ----

TEST_F(GpuAdapterAllVariantsTest, minusDoubleOnGpu) {
  constexpr int N = 4;
  std::vector<double> a = {10.0, 20.0, -5.0, 100.0};
  std::vector<double> b = {3.0, 25.0, -5.0, 0.5};
  std::vector<double> expected = {7.0, -5.0, 0.0, 99.5};

  auto* dA = deviceAlloc<double>(N);
  auto* dB = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};
  GpuColumnReader<double> rB{cudf::device_span<const double>(dB, N)};
  GpuColumnWriter<double> w{dOut, dNM};

  using Fn = facebook::velox::functions::MinusFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, double, double, double>::apply(
      w, rA, rB, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<double> result(N);
  toHost(result.data(), dOut, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(expected[i], result[i]) << "at " << i;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  cudaFree(dNM);
}

TEST_F(GpuAdapterAllVariantsTest, multiplyInt64OnGpu) {
  constexpr int N = 4;
  std::vector<int64_t> a = {3, -7, 0, 1000000};
  std::vector<int64_t> b = {5, 11, 42, 1000000};
  std::vector<int64_t> expected = {15, -77, 0, 1000000000000LL};

  auto* dA = deviceAlloc<int64_t>(N);
  auto* dB = deviceAlloc<int64_t>(N);
  auto* dOut = deviceAlloc<int64_t>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  GpuColumnReader<int64_t> rA{cudf::device_span<const int64_t>(dA, N)};
  GpuColumnReader<int64_t> rB{cudf::device_span<const int64_t>(dB, N)};
  GpuColumnWriter<int64_t> w{dOut, dNM};

  using Fn = facebook::velox::functions::MultiplyFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, int64_t, int64_t, int64_t>::apply(
      w, rA, rB, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<int64_t> result(N);
  toHost(result.data(), dOut, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(expected[i], result[i]) << "at " << i;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  cudaFree(dNM);
}

TEST_F(GpuAdapterAllVariantsTest, divideDoubleOnGpu) {
  constexpr int N = 4;
  std::vector<double> a = {10.0, 9.0, -6.0, 1.0};
  std::vector<double> b = {2.0, 3.0, 2.0, 3.0};
  std::vector<double> expected = {5.0, 3.0, -3.0, 1.0 / 3.0};

  auto* dA = deviceAlloc<double>(N);
  auto* dB = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};
  GpuColumnReader<double> rB{cudf::device_span<const double>(dB, N)};
  GpuColumnWriter<double> w{dOut, dNM};

  using Fn = facebook::velox::functions::DivideFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, double, double, double>::apply(
      w, rA, rB, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<double> result(N);
  toHost(result.data(), dOut, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(expected[i], result[i]) << "at " << i;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  cudaFree(dNM);
}

TEST_F(GpuAdapterAllVariantsTest, modulusDoubleOnGpu) {
  constexpr int N = 4;
  std::vector<double> a = {10.0, 7.5, -10.0, 5.0};
  std::vector<double> b = {3.0, 2.0, 3.0, 5.0};
  std::vector<double> expected = {1.0, 1.5, -1.0, 0.0};

  auto* dA = deviceAlloc<double>(N);
  auto* dB = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};
  GpuColumnReader<double> rB{cudf::device_span<const double>(dB, N)};
  GpuColumnWriter<double> w{dOut, dNM};

  using Fn = facebook::velox::functions::ModulusFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, double, double, double>::apply(
      w, rA, rB, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<double> result(N);
  toHost(result.data(), dOut, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(expected[i], result[i]) << "at " << i;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  cudaFree(dNM);
}

// ---- Unary Arithmetic ----

TEST_F(GpuAdapterAllVariantsTest, negateInt64OnGpu) {
  constexpr int N = 4;
  std::vector<int64_t> a = {5, -3, 0, 9223372036854775807LL};
  std::vector<int64_t> expected = {-5, 3, 0, -9223372036854775807LL};

  auto* dA = deviceAlloc<int64_t>(N);
  auto* dOut = deviceAlloc<int64_t>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);

  GpuColumnReader<int64_t> rA{cudf::device_span<const int64_t>(dA, N)};
  GpuColumnWriter<int64_t> w{dOut, dNM};

  using Fn = facebook::velox::functions::NegateFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, int64_t, int64_t>::apply(
      w, rA, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<int64_t> result(N);
  toHost(result.data(), dOut, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(expected[i], result[i]) << "at " << i;
  }

  cudaFree(dA);
  cudaFree(dOut);
  cudaFree(dNM);
}

TEST_F(GpuAdapterAllVariantsTest, absDoubleOnGpu) {
  constexpr int N = 4;
  std::vector<double> a = {-5.5, 3.14, 0.0, -0.0};
  std::vector<double> expected = {5.5, 3.14, 0.0, 0.0};

  auto* dA = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};
  GpuColumnWriter<double> w{dOut, dNM};

  using Fn = facebook::velox::functions::AbsFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, double, double>::apply(
      w, rA, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<double> result(N);
  toHost(result.data(), dOut, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(expected[i], result[i]) << "at " << i;
  }

  cudaFree(dA);
  cudaFree(dOut);
  cudaFree(dNM);
}

TEST_F(GpuAdapterAllVariantsTest, ceilFloorOnGpu) {
  constexpr int N = 4;
  std::vector<double> a = {2.3, -2.3, 5.0, -0.1};
  std::vector<double> expectedCeil = {3.0, -2.0, 5.0, 0.0};
  std::vector<double> expectedFloor = {2.0, -3.0, 5.0, -1.0};

  auto* dA = deviceAlloc<double>(N);
  auto* dOutCeil = deviceAlloc<double>(N);
  auto* dOutFloor = deviceAlloc<double>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};

  {
    GpuColumnWriter<double> w{dOutCeil, dNM};
    using Fn = facebook::velox::functions::CeilFunction<GpuExec>;
    GpuSimpleFunctionAdapter<Fn, double, double>::apply(
        w, rA, nullptr, nullptr, N);
    cudaDeviceSynchronize();
  }
  {
    GpuColumnWriter<double> w{dOutFloor, dNM};
    using Fn = facebook::velox::functions::FloorFunction<GpuExec>;
    GpuSimpleFunctionAdapter<Fn, double, double>::apply(
        w, rA, nullptr, nullptr, N);
    cudaDeviceSynchronize();
  }

  std::vector<double> ceilResult(N), floorResult(N);
  toHost(ceilResult.data(), dOutCeil, N);
  toHost(floorResult.data(), dOutFloor, N);

  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(expectedCeil[i], ceilResult[i]) << "ceil at " << i;
    EXPECT_DOUBLE_EQ(expectedFloor[i], floorResult[i]) << "floor at " << i;
  }

  cudaFree(dA);
  cudaFree(dOutCeil);
  cudaFree(dOutFloor);
  cudaFree(dNM);
}

// ---- Math Functions (unary double→double) ----

TEST_F(GpuAdapterAllVariantsTest, expOnGpu) {
  constexpr int N = 3;
  std::vector<double> a = {0.0, 1.0, -1.0};
  std::vector<double> expected = {1.0, std::exp(1.0), std::exp(-1.0)};

  auto* dA = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));
  toDevice(dA, a.data(), N);

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};
  GpuColumnWriter<double> w{dOut, dNM};

  using Fn = facebook::velox::functions::ExpFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, double, double>::apply(
      w, rA, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<double> result(N);
  toHost(result.data(), dOut, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(expected[i], result[i], 1e-12) << "at " << i;
  }

  cudaFree(dA);
  cudaFree(dOut);
  cudaFree(dNM);
}

TEST_F(GpuAdapterAllVariantsTest, sqrtOnGpu) {
  constexpr int N = 3;
  std::vector<double> a = {4.0, 9.0, 2.0};
  std::vector<double> expected = {2.0, 3.0, std::sqrt(2.0)};

  auto* dA = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));
  toDevice(dA, a.data(), N);

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};
  GpuColumnWriter<double> w{dOut, dNM};

  using Fn = facebook::velox::functions::SqrtFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, double, double>::apply(
      w, rA, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<double> result(N);
  toHost(result.data(), dOut, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(expected[i], result[i], 1e-12) << "at " << i;
  }

  cudaFree(dA);
  cudaFree(dOut);
  cudaFree(dNM);
}

TEST_F(GpuAdapterAllVariantsTest, lnOnGpu) {
  constexpr int N = 3;
  std::vector<double> a = {1.0, std::exp(1.0), std::exp(5.0)};
  std::vector<double> expected = {0.0, 1.0, 5.0};

  auto* dA = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));
  toDevice(dA, a.data(), N);

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};
  GpuColumnWriter<double> w{dOut, dNM};

  using Fn = facebook::velox::functions::LnFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, double, double>::apply(
      w, rA, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<double> result(N);
  toHost(result.data(), dOut, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(expected[i], result[i], 1e-12) << "at " << i;
  }

  cudaFree(dA);
  cudaFree(dOut);
  cudaFree(dNM);
}

TEST_F(GpuAdapterAllVariantsTest, sinCosOnGpu) {
  constexpr int N = 3;
  std::vector<double> a = {0.0, M_PI / 2.0, M_PI};

  auto* dA = deviceAlloc<double>(N);
  auto* dOutSin = deviceAlloc<double>(N);
  auto* dOutCos = deviceAlloc<double>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));
  toDevice(dA, a.data(), N);

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};

  {
    GpuColumnWriter<double> w{dOutSin, dNM};
    using Fn = facebook::velox::functions::SinFunction<GpuExec>;
    GpuSimpleFunctionAdapter<Fn, double, double>::apply(
        w, rA, nullptr, nullptr, N);
    cudaDeviceSynchronize();
  }
  {
    GpuColumnWriter<double> w{dOutCos, dNM};
    using Fn = facebook::velox::functions::CosFunction<GpuExec>;
    GpuSimpleFunctionAdapter<Fn, double, double>::apply(
        w, rA, nullptr, nullptr, N);
    cudaDeviceSynchronize();
  }

  std::vector<double> sinResult(N), cosResult(N);
  toHost(sinResult.data(), dOutSin, N);
  toHost(cosResult.data(), dOutCos, N);

  EXPECT_NEAR(0.0, sinResult[0], 1e-12);
  EXPECT_NEAR(1.0, sinResult[1], 1e-12);
  EXPECT_NEAR(0.0, sinResult[2], 1e-12);

  EXPECT_NEAR(1.0, cosResult[0], 1e-12);
  EXPECT_NEAR(0.0, cosResult[1], 1e-12);
  EXPECT_NEAR(-1.0, cosResult[2], 1e-12);

  cudaFree(dA);
  cudaFree(dOutSin);
  cudaFree(dOutCos);
  cudaFree(dNM);
}

// ---- Comparison Functions ----

TEST_F(GpuAdapterAllVariantsTest, gtGteLteOnGpu) {
  constexpr int N = 4;
  std::vector<double> a = {1.0, 5.0, 3.0, 3.0};
  std::vector<double> b = {2.0, 3.0, 3.0, 1.0};

  auto* dA = deviceAlloc<double>(N);
  auto* dB = deviceAlloc<double>(N);
  auto* dOutGt = deviceAlloc<bool>(N);
  auto* dOutGte = deviceAlloc<bool>(N);
  auto* dOutLte = deviceAlloc<bool>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};
  GpuColumnReader<double> rB{cudf::device_span<const double>(dB, N)};

  {
    GpuColumnWriter<bool> w{dOutGt, dNM};
    using Fn = facebook::velox::functions::GtFunction<GpuExec>;
    GpuSimpleFunctionAdapter<Fn, bool, double, double>::apply(
        w, rA, rB, nullptr, nullptr, N);
    cudaDeviceSynchronize();
  }
  {
    GpuColumnWriter<bool> w{dOutGte, dNM};
    using Fn = facebook::velox::functions::GteFunction<GpuExec>;
    GpuSimpleFunctionAdapter<Fn, bool, double, double>::apply(
        w, rA, rB, nullptr, nullptr, N);
    cudaDeviceSynchronize();
  }
  {
    GpuColumnWriter<bool> w{dOutLte, dNM};
    using Fn = facebook::velox::functions::LteFunction<GpuExec>;
    GpuSimpleFunctionAdapter<Fn, bool, double, double>::apply(
        w, rA, rB, nullptr, nullptr, N);
    cudaDeviceSynchronize();
  }

  bool gtHost[N], gteHost[N], lteHost[N];
  toHost(gtHost, dOutGt, N);
  toHost(gteHost, dOutGte, N);
  toHost(lteHost, dOutLte, N);

  // a:   1   5   3   3
  // b:   2   3   3   1
  // gt:  F   T   F   T
  // gte: F   T   T   T
  // lte: T   F   T   F
  EXPECT_FALSE(gtHost[0]);
  EXPECT_TRUE(gtHost[1]);
  EXPECT_FALSE(gtHost[2]);
  EXPECT_TRUE(gtHost[3]);

  EXPECT_FALSE(gteHost[0]);
  EXPECT_TRUE(gteHost[1]);
  EXPECT_TRUE(gteHost[2]);
  EXPECT_TRUE(gteHost[3]);

  EXPECT_TRUE(lteHost[0]);
  EXPECT_FALSE(lteHost[1]);
  EXPECT_TRUE(lteHost[2]);
  EXPECT_FALSE(lteHost[3]);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOutGt);
  cudaFree(dOutGte);
  cudaFree(dOutLte);
  cudaFree(dNM);
}

// ---- Ternary Function: Between ----

TEST_F(GpuAdapterAllVariantsTest, betweenOnGpu) {
  constexpr int N = 4;
  std::vector<double> val = {5.0, 1.0, 10.0, 3.0};
  std::vector<double> lo = {2.0, 2.0, 2.0, 3.0};
  std::vector<double> hi = {8.0, 8.0, 8.0, 3.0};
  // between: 5 in [2,8]=T, 1 in [2,8]=F, 10 in [2,8]=F, 3 in [3,3]=T

  auto* dVal = deviceAlloc<double>(N);
  auto* dLo = deviceAlloc<double>(N);
  auto* dHi = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<bool>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dVal, val.data(), N);
  toDevice(dLo, lo.data(), N);
  toDevice(dHi, hi.data(), N);

  GpuColumnReader<double> rVal{cudf::device_span<const double>(dVal, N)};
  GpuColumnReader<double> rLo{cudf::device_span<const double>(dLo, N)};
  GpuColumnReader<double> rHi{cudf::device_span<const double>(dHi, N)};
  GpuColumnWriter<bool> w{dOut, dNM};

  using Fn = facebook::velox::functions::BetweenFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, bool, double, double, double>::apply(
      w, rVal, rLo, rHi, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  bool hostResult[N];
  toHost(hostResult, dOut, N);

  EXPECT_TRUE(hostResult[0]);
  EXPECT_FALSE(hostResult[1]);
  EXPECT_FALSE(hostResult[2]);
  EXPECT_TRUE(hostResult[3]);

  cudaFree(dVal);
  cudaFree(dLo);
  cudaFree(dHi);
  cudaFree(dOut);
  cudaFree(dNM);
}

// ---- Bool-returning call(): PModIntFunction ----

TEST_F(GpuAdapterAllVariantsTest, pmodIntNullableResultOnGpu) {
  constexpr int N = 4;
  std::vector<int64_t> a = {10, -10, 5, 7};
  std::vector<int64_t> b = {3, 3, 0, 2};
  // pmod(10,3)=1, pmod(-10,3)=2, pmod(5,0)=null, pmod(7,2)=1
  std::vector<int64_t> expected = {1, 2, 0 /*null*/, 1};
  std::vector<bool> expectedValid = {true, true, false, true};

  auto* dA = deviceAlloc<int64_t>(N);
  auto* dB = deviceAlloc<int64_t>(N);
  auto* dOut = deviceAlloc<int64_t>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  std::vector<cudf::bitmask_type> allValid = {0xFFFFFFFF};
  toDevice(dNM, allValid.data(), 1);

  GpuColumnReader<int64_t> rA{cudf::device_span<const int64_t>(dA, N)};
  GpuColumnReader<int64_t> rB{cudf::device_span<const int64_t>(dB, N)};
  GpuColumnWriter<int64_t> w{dOut, dNM};

  using Fn = facebook::velox::functions::PModIntFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, int64_t, int64_t, int64_t>::apply(
      w, rA, rB, nullptr, nullptr, N);
  cudaDeviceSynchronize();

  std::vector<int64_t> result(N);
  std::vector<cudf::bitmask_type> resultNM(1);
  toHost(result.data(), dOut, N);
  toHost(resultNM.data(), dNM, 1);

  for (int i = 0; i < N; ++i) {
    if (expectedValid[i]) {
      EXPECT_TRUE(cudf::bit_is_set(resultNM.data(), i)) << "at " << i;
      EXPECT_EQ(expected[i], result[i]) << "at " << i;
    } else {
      EXPECT_FALSE(cudf::bit_is_set(resultNM.data(), i))
          << "expected null at " << i;
    }
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  cudaFree(dNM);
}

// ---- Active row mask filtering ----

TEST_F(GpuAdapterAllVariantsTest, activeRowMask) {
  constexpr int N = 4;
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> b = {10.0, 20.0, 30.0, 40.0};

  auto* dA = deviceAlloc<double>(N);
  auto* dB = deviceAlloc<double>(N);
  auto* dOut = deviceAlloc<double>(N);
  auto* dNM = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));
  auto* dActive = deviceAlloc<cudf::bitmask_type>(bitmaskWords(N));

  toDevice(dA, a.data(), N);
  toDevice(dB, b.data(), N);

  // Only rows 0 and 2 are active (bits 0 and 2 set)
  std::vector<cudf::bitmask_type> activeMask = {0b00000101};
  toDevice(dActive, activeMask.data(), 1);

  // Zero output so we can detect untouched rows
  cudaMemset(dOut, 0, N * sizeof(double));

  GpuColumnReader<double> rA{cudf::device_span<const double>(dA, N)};
  GpuColumnReader<double> rB{cudf::device_span<const double>(dB, N)};
  GpuColumnWriter<double> w{dOut, dNM};

  using Fn = facebook::velox::functions::PlusFunction<GpuExec>;
  GpuSimpleFunctionAdapter<Fn, double, double, double>::apply(
      w, rA, rB, nullptr, dActive, N);
  cudaDeviceSynchronize();

  std::vector<double> result(N);
  toHost(result.data(), dOut, N);

  EXPECT_DOUBLE_EQ(11.0, result[0]);
  EXPECT_DOUBLE_EQ(0.0, result[1]); // inactive, untouched
  EXPECT_DOUBLE_EQ(33.0, result[2]);
  EXPECT_DOUBLE_EQ(0.0, result[3]); // inactive, untouched

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);
  cudaFree(dNM);
  cudaFree(dActive);
}
