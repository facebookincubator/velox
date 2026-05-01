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

// Smoke test for the gpu_portable lift of DecimalUtil::divideWithRoundUp.
// Verifies that the generated DecimalDivide.h (a) exposes a getter that
// returns a non-empty CUDA source string with the expected baked literals,
// and (b) compiles under NVRTC and runs a single-row division correctly
// via cudf::transform_extended. The lift is scale-oblivious on output
// (mirrors the CPU contract); the caller is responsible for reinterpreting
// the result column at its intended decimal scale.

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/gpu_portable/DecimalDivide.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/transform.hpp>
#include <cuda_runtime_api.h>

namespace facebook::velox::cudf_velox {
namespace {

class DecimalDivideGpuPortableTest
    : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "no CUDA devices available";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(nullptr)));
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    exec::test::OperatorTestBase::TearDown();
  }
};

TEST_F(DecimalDivideGpuPortableTest, sourceStringContainsBakedLiterals) {
  const auto src = gpu_portable::velox_decimal_divide_int128_int128_source(
      /*noRoundUp=*/false, /*pow10_aRescale=*/static_cast<__int128_t>(1));
  ASSERT_FALSE(src.empty());
  EXPECT_NE(src.find("__device__ void velox_decimal_divide_int128_int128"), std::string::npos);
  // Adapter-in unwraps the cuDF boundary decimals into raw __int128_t.
  EXPECT_NE(src.find("const __int128_t a = a_in.value();"), std::string::npos)
      << src;
  EXPECT_NE(src.find("const __int128_t b = b_in.value();"), std::string::npos)
      << src;
  // Baked literals spliced as integer text. `false` becomes `0`.
  EXPECT_NE(src.find("const bool noRoundUp = 0;"), std::string::npos)
      << src;
  EXPECT_NE(src.find("const __int128_t pow10_aRescale = 1;"), std::string::npos)
      << src;
  // The dual-output rewrite + out-wrap should produce a decimal128
  // construction at the out-pointer write.
  EXPECT_NE(
      src.find("*out = numeric::decimal128{quotient * resultSign, numeric::scale_type{0}}"),
      std::string::npos) << src;
  EXPECT_EQ(src.find("return remainder"), std::string::npos) << src;
  // CPU-only constructs must be gone.
  EXPECT_EQ(src.find("VELOX_USER_CHECK"), std::string::npos) << src;
  EXPECT_EQ(src.find("checkedMultiply"), std::string::npos) << src;
  EXPECT_EQ(src.find("DecimalUtil::kPowersOfTen"), std::string::npos) << src;
}

// Compute aRescale the same way Velox's decimal-divide expression code
// does: aRescale = s_out + s_b - s_a, where all scales are Velox-style
// (non-negative count of fractional digits).
namespace {
int32_t computeARescale(int32_t sOut, int32_t sA, int32_t sB) {
  return sOut + sB - sA;
}

__int128_t pow10Int128(int32_t exp) {
  __int128_t v = 1;
  for (int32_t i = 0; i < exp; ++i) {
    v *= 10;
  }
  return v;
}
} // namespace

TEST_F(DecimalDivideGpuPortableTest, divides10By2) {
  // a = 10, b = 2, pow10 = 1 (scale factor for aRescale=0). Expected: 5.
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int128_t>(
              {static_cast<int128_t>(10)}, DECIMAL(38, 0)),
          makeFlatVector<int128_t>(
              {static_cast<int128_t>(2)}, DECIMAL(38, 0)),
      });

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable =
      velox::cudf_velox::with_arrow::toCudfTable(input, pool_.get(), stream, mr);
  auto ownedCols = cudfTable->release();
  ASSERT_EQ(ownedCols.size(), 2u);

  const std::vector<cudf::transform_input> inputs = {
      ownedCols[0]->view(), ownedCols[1]->view()};

  const auto src = gpu_portable::velox_decimal_divide_int128_int128_source(
      /*noRoundUp=*/false, /*pow10_aRescale=*/static_cast<__int128_t>(1));

  auto outCol = cudf::transform_extended(
      std::span<const cudf::transform_input>(inputs.data(), inputs.size()),
      src,
      cudf::data_type{cudf::type_id::DECIMAL128, 0},
      cudf::udf_source_type::CUDA,
      std::nullopt,
      cudf::null_aware::NO,
      std::nullopt,
      cudf::output_nullability::PRESERVE,
      stream,
      mr);

  cudf::table_view resultTable({outCol->view()});
  auto resultVec = velox::cudf_velox::with_arrow::toVeloxColumn(
      resultTable, pool_.get(), "", stream, mr);
  auto flat = resultVec->childAt(0)->as<FlatVector<int128_t>>();
  ASSERT_NE(flat, nullptr);
  ASSERT_EQ(flat->size(), 1u);
  EXPECT_EQ(flat->valueAt(0), static_cast<int128_t>(5));
}

// Demonstrates that the lifted kernel is scale-oblivious: it always emits a
// scale-0 output column, and the caller stamps the desired output scale
// via cudf::bit_cast (metadata-only reinterpret). This mirrors the
// Velox-CPU contract where divideWithRoundUp writes a raw integer to r
// and the output vector carries the scale.
TEST_F(DecimalDivideGpuPortableTest, scaledDivideViaCallerReinterpret) {
  // Velox DECIMAL(38, 3) / DECIMAL(38, 3) -> DECIMAL(38, 3).
  // Raw representations: 10.000 -> 10000, 2.000 -> 2000. Expected:
  // 5.000 -> raw 5000 in the output column at scale 3.
  constexpr int32_t kScaleA = 3;
  constexpr int32_t kScaleB = 3;
  constexpr int32_t kScaleOut = 3;
  const int32_t aRescale =
      computeARescale(kScaleOut, kScaleA, kScaleB);
  const __int128_t pow10 = pow10Int128(aRescale);

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int128_t>(
              {static_cast<int128_t>(10000)}, DECIMAL(38, kScaleA)),
          makeFlatVector<int128_t>(
              {static_cast<int128_t>(2000)}, DECIMAL(38, kScaleB)),
      });

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable =
      velox::cudf_velox::with_arrow::toCudfTable(input, pool_.get(), stream, mr);
  auto ownedCols = cudfTable->release();
  ASSERT_EQ(ownedCols.size(), 2u);

  const std::vector<cudf::transform_input> inputs = {
      ownedCols[0]->view(), ownedCols[1]->view()};

  const auto src = gpu_portable::velox_decimal_divide_int128_int128_source(
      /*noRoundUp=*/false, pow10);

  // Kernel writes raw int directly into a scale-0 DECIMAL128 column.
  auto outCol = cudf::transform_extended(
      std::span<const cudf::transform_input>(inputs.data(), inputs.size()),
      src,
      cudf::data_type{cudf::type_id::DECIMAL128, 0},
      cudf::udf_source_type::CUDA,
      std::nullopt,
      cudf::null_aware::NO,
      std::nullopt,
      cudf::output_nullability::PRESERVE,
      stream,
      mr);

  // Metadata-only reinterpret to the output column's true scale. In cuDF's
  // scale convention, Velox scale s maps to cuDF scale -s.
  auto scaledView = cudf::bit_cast(
      outCol->view(),
      cudf::data_type{cudf::type_id::DECIMAL128, -kScaleOut});

  cudf::table_view resultTable({scaledView});
  auto resultVec = velox::cudf_velox::with_arrow::toVeloxColumn(
      resultTable, pool_.get(), "", stream, mr);
  auto flat = resultVec->childAt(0)->as<FlatVector<int128_t>>();
  ASSERT_NE(flat, nullptr);
  ASSERT_EQ(flat->size(), 1u);
  // 5.000 stored as raw 5000 in a scale-3 column.
  EXPECT_EQ(flat->valueAt(0), static_cast<int128_t>(5000));
}

} // namespace
} // namespace facebook::velox::cudf_velox
