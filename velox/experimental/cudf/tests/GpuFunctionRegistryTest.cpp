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

// Registration bookkeeping for GPU simple functions. These cases exercise the
// registry itself and need no GPU: the launchers are stand-ins that are never
// invoked. What is under test is that the registry reproduces the parts of
// Velox's registration contract that dialect separation depends on -- aliases,
// name sanitizing, overload coexistence, and overwrite-on-collision.

#include <gtest/gtest.h>

#include "velox/experimental/cudf/functions/GpuFunctionBridge.h"
#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

std::unique_ptr<cudf::column> launcherA(
    const std::vector<cudf::column_view>&,
    cudf::data_type,
    rmm::cuda_stream_view,
    rmm::device_async_resource_ref) {
  return nullptr;
}

std::unique_ptr<cudf::column> launcherB(
    const std::vector<cudf::column_view>&,
    cudf::data_type,
    rmm::cuda_stream_view,
    rmm::device_async_resource_ref) {
  return nullptr;
}

GpuFunctionSignature doubleBinary() {
  return GpuFunctionSignature{"double", {"double", "double"}};
}

GpuFunctionSignature bigintBinary() {
  return GpuFunctionSignature{"bigint", {"bigint", "bigint"}};
}

const std::vector<GpuFunctionEntry>* lookup(const std::string& name) {
  const auto& registry = gpuFunctionRegistry();
  auto it = registry.find(name);
  return it == registry.end() ? nullptr : &it->second;
}

class GpuFunctionRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    clearGpuFunctionRegistry();
  }
};

TEST_F(GpuFunctionRegistryTest, registersUnderEveryAlias) {
  ASSERT_TRUE(registerGpuKernel({"power", "pow"}, doubleBinary(), launcherA));

  ASSERT_NE(lookup("power"), nullptr);
  ASSERT_NE(lookup("pow"), nullptr);
  EXPECT_EQ(lookup("power")->front().launch, launcherA);
  EXPECT_EQ(lookup("pow")->front().launch, launcherA);
}

TEST_F(GpuFunctionRegistryTest, lowercasesNamesLikeVelox) {
  ASSERT_TRUE(registerGpuKernel({"MyFunc"}, doubleBinary(), launcherA));

  EXPECT_NE(lookup("myfunc"), nullptr);
  EXPECT_EQ(lookup("MyFunc"), nullptr);
}

TEST_F(GpuFunctionRegistryTest, differentSignaturesCoexistAsOverloads) {
  ASSERT_TRUE(registerGpuKernel({"plus"}, doubleBinary(), launcherA));
  ASSERT_TRUE(registerGpuKernel({"plus"}, bigintBinary(), launcherB));

  const auto* entries = lookup("plus");
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(entries->size(), 2);
}

// The mechanism dialect separation rests on: a second implementation under the
// same name and signature replaces the first, so whichever dialect registers
// last wins, exactly as on the CPU side.
TEST_F(GpuFunctionRegistryTest, sameSignatureOverwritesByDefault) {
  ASSERT_TRUE(registerGpuKernel({"divide"}, doubleBinary(), launcherA));
  ASSERT_TRUE(registerGpuKernel({"divide"}, doubleBinary(), launcherB));

  const auto* entries = lookup("divide");
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(entries->size(), 1);
  EXPECT_EQ(entries->front().launch, launcherB);
}

TEST_F(GpuFunctionRegistryTest, overwriteFalseKeepsTheIncumbent) {
  ASSERT_TRUE(registerGpuKernel({"divide"}, doubleBinary(), launcherA));
  EXPECT_FALSE(registerGpuKernel(
      {"divide"}, doubleBinary(), launcherB, /*overwrite=*/false));

  const auto* entries = lookup("divide");
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(entries->size(), 1);
  EXPECT_EQ(entries->front().launch, launcherA);
}

// Guards the naming contract between the two dialect files. Prefixing is how
// both can be loaded into one process without colliding.
TEST_F(GpuFunctionRegistryTest, prefixSeparatesDialects) {
  ASSERT_TRUE(registerGpuKernel({"presto.divide"}, doubleBinary(), launcherA));
  ASSERT_TRUE(registerGpuKernel({"spark.divide"}, doubleBinary(), launcherB));

  ASSERT_NE(lookup("presto.divide"), nullptr);
  ASSERT_NE(lookup("spark.divide"), nullptr);
  EXPECT_EQ(lookup("presto.divide")->front().launch, launcherA);
  EXPECT_EQ(lookup("spark.divide")->front().launch, launcherB);
}

// Exercises the real registrations rather than stand-ins, so it also proves the
// signature strings the device side derives from SimpleTypeTrait are the ones
// Velox would produce.
TEST_F(GpuFunctionRegistryTest, prestoRegistrationsCarryVeloxSignatures) {
  registerPrestoGpuFunctions("");

  const auto* plus = lookup("plus");
  ASSERT_NE(plus, nullptr);
  EXPECT_EQ(plus->size(), 2) << "expected double and bigint overloads";

  bool sawDouble = false;
  bool sawBigint = false;
  for (const auto& entry : *plus) {
    if (entry.signature.returnType == "double") {
      sawDouble = true;
      EXPECT_EQ(
          entry.signature.argumentTypes,
          (std::vector<std::string>{"double", "double"}));
    } else if (entry.signature.returnType == "bigint") {
      sawBigint = true;
    }
  }
  EXPECT_TRUE(sawDouble);
  EXPECT_TRUE(sawBigint);

  // Return type differs from argument type for predicates and comparisons.
  const auto* isNan = lookup("is_nan");
  ASSERT_NE(isNan, nullptr);
  EXPECT_EQ(isNan->front().signature.returnType, "boolean");
  EXPECT_EQ(
      isNan->front().signature.argumentTypes,
      (std::vector<std::string>{"double"}));
}

// The four functions whose bodies use VELOX_USER_CHECK must stay unregistered
// while the Exceptions shadow makes those checks no-ops, or GPU results would
// diverge from CPU silently instead of erroring.
TEST_F(GpuFunctionRegistryTest, checkBearingFunctionsAreNotRegistered) {
  registerPrestoGpuFunctions("");

  for (const auto* name :
       {"bit_count",
        "bitwise_arithmetic_shift_right",
        "bitwise_shift_left",
        "bitwise_logical_shift_right"}) {
    EXPECT_EQ(lookup(name), nullptr) << name << " should be held back";
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
