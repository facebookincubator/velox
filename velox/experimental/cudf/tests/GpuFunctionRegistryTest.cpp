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

// Tests of the GPU simple-function registry: aliases, name sanitizing,
// overload coexistence and overwrite-on-collision. They need no GPU; the
// launchers are stand-ins that are never invoked.

#include "velox/experimental/cudf/functions/GpuFunctionLookup.h"

#include "velox/expression/SignatureBinder.h"
#include "velox/type/TypeCoercer.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

std::unique_ptr<cudf::column> launcherA(
    const std::vector<GpuArgView>&,
    cudf::size_type,
    cudf::data_type,
    uint8_t*,
    cuda::stream_ref,
    rmm::device_async_resource_ref) {
  return nullptr;
}

std::unique_ptr<cudf::column> launcherB(
    const std::vector<GpuArgView>&,
    cudf::size_type,
    cudf::data_type,
    uint8_t*,
    cuda::stream_ref,
    rmm::device_async_resource_ref) {
  return nullptr;
}

GpuFunctionSignature doubleBinary() {
  return GpuFunctionSignature{
      "double", {"double", "double"}, /*variadicTail=*/false, {}};
}

GpuFunctionSignature bigintBinary() {
  return GpuFunctionSignature{
      "bigint", {"bigint", "bigint"}, /*variadicTail=*/false, {}};
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

// A second implementation under the same name and signature replaces the
// first, as on the CPU, so the dialect registered last wins.
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

// Prefixes let both dialects load into one process without colliding.
TEST_F(GpuFunctionRegistryTest, prefixSeparatesDialects) {
  ASSERT_TRUE(registerGpuKernel({"presto.divide"}, doubleBinary(), launcherA));
  ASSERT_TRUE(registerGpuKernel({"spark.divide"}, doubleBinary(), launcherB));

  ASSERT_NE(lookup("presto.divide"), nullptr);
  ASSERT_NE(lookup("spark.divide"), nullptr);
  EXPECT_EQ(lookup("presto.divide")->front().launch, launcherA);
  EXPECT_EQ(lookup("spark.divide")->front().launch, launcherB);
}

std::vector<std::string> signaturesOf(const std::string& name) {
  std::vector<std::string> out;
  if (const auto* entries = lookup(name)) {
    for (const auto& entry : *entries) {
      out.push_back(entry.signature->toString());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// The signatures derived from SimpleTypeTrait must equal the ones Velox
// registers for the same functions, through the helpers that mirror
// RegistrationHelpers.h, since SignatureBinder matches calls against them.
TEST_F(GpuFunctionRegistryTest, registrationsCarryVeloxSignatures) {
  registerPrestoGpuFunctions("");

  const std::vector<std::pair<std::string, std::vector<std::string>>> expected{
      // PlusFunction for floating point, CheckedPlusFunction for integers.
      {"plus",
       {"(bigint,bigint) -> bigint",
        "(double,double) -> double",
        "(integer,integer) -> integer",
        "(real,real) -> real",
        "(smallint,smallint) -> smallint",
        "(tinyint,tinyint) -> tinyint"}},
      {"abs",
       {"(bigint) -> bigint",
        "(double) -> double",
        "(integer) -> integer",
        "(real) -> real",
        "(smallint) -> smallint",
        "(tinyint) -> tinyint"}},
      {"ln", {"(double) -> double"}},
      {"is_nan", {"(double) -> boolean"}},
  };
  for (const auto& [name, signatures] : expected) {
    SCOPED_TRACE(name);
    EXPECT_EQ(signaturesOf(name), signatures);
  }

  EXPECT_EQ(signaturesOf("ceil"), signaturesOf("ceiling"));
  // Two floating-point types at two arities.
  EXPECT_EQ(signaturesOf("truncate").size(), 4);
  EXPECT_TRUE(lookup("and")->front().signature->variableArity());
}

// A decimal signature names its precision and scale as variables, which must
// be declared for the signature to build. Otherwise the function registers as
// taking a bigint, since SimpleTypeTrait<ShortDecimal<P, S>> inherits
// TypeTraits<BIGINT>, and never matches a decimal call.
TEST_F(GpuFunctionRegistryTest, decimalSignaturesCarryPrecisionAndScale) {
  GpuFunctionSignature signature{
      "decimal(i1,i5)",
      {"decimal(i1,i5)", "decimal(i1,i5)"},
      /*variadicTail=*/false,
      // Named once per occurrence, as the device side collects them; the
      // builder rejects a redeclaration, so the duplicates must be dropped.
      {"i1", "i5", "i1", "i5", "i1", "i5"}};
  ASSERT_TRUE(registerGpuKernel({"decimal_add"}, signature, launcherA));

  const auto* entries = lookup("decimal_add");
  ASSERT_NE(entries, nullptr);
  EXPECT_EQ(
      entries->front().signature->toString(),
      "(decimal(i1,i5),decimal(i1,i5)) -> decimal(i1,i5)");

  // The signature binds a concrete decimal call.
  const std::vector<TypePtr> arguments{DECIMAL(10, 2), DECIMAL(10, 2)};
  exec::SignatureBinder binder(
      *entries->front().signature, arguments, TypeCoercer::defaults());
  ASSERT_TRUE(binder.tryBind());
  EXPECT_TRUE(binder.tryResolveReturnType()->equivalent(*DECIMAL(10, 2)));

  // A bigint call does not bind to it.
  const std::vector<TypePtr> bigints{BIGINT(), BIGINT()};
  exec::SignatureBinder wrongType(
      *entries->front().signature, bigints, TypeCoercer::defaults());
  EXPECT_FALSE(wrongType.tryBind());
}

// The four functions whose bodies use VELOX_USER_CHECK are not registered
// yet; see the TODO at the registration site.
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
