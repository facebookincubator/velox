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

#include "velox/experimental/cudf/functions/GpuFunctionLookup.h"
#include "velox/type/TypeCoercer.h"
#include "velox/expression/SignatureBinder.h"

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

std::unique_ptr<cudf::column> launcherA(
    const std::vector<GpuArgView>&,
    cudf::size_type,
    cudf::data_type,
    rmm::cuda_stream_view,
    rmm::device_async_resource_ref) {
  return nullptr;
}

std::unique_ptr<cudf::column> launcherB(
    const std::vector<GpuArgView>&,
    cudf::size_type,
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

// Exercises the real registrations rather than stand-ins. The strings the
// device side derives from SimpleTypeTrait have to parse into the signature
// Velox would have built for the same function, because that signature is what
// exec::SignatureBinder later matches a call against -- a string Velox cannot
// parse would leave the function permanently unresolvable.
TEST_F(GpuFunctionRegistryTest, prestoRegistrationsCarryVeloxSignatures) {
  registerPrestoGpuFunctions("");

  const auto* plus = lookup("plus");
  ASSERT_NE(plus, nullptr);

  std::vector<std::string> plusSignatures;
  for (const auto& entry : *plus) {
    plusSignatures.push_back(entry.signature->toString());
  }
  std::sort(plusSignatures.begin(), plusSignatures.end());
  // Two floating-point overloads from the plain struct and four integral ones
  // from CheckedPlusFunction, under one name. Overloads of the same name
  // coexist rather than replacing each other, and which struct backs which type
  // is the whole checked/unchecked distinction.
  EXPECT_EQ(
      plusSignatures,
      (std::vector<std::string>{
          "(bigint,bigint) -> bigint",
          "(double,double) -> double",
          "(integer,integer) -> integer",
          "(real,real) -> real",
          "(smallint,smallint) -> smallint",
          "(tinyint,tinyint) -> tinyint"}));

  // Return type differs from argument type for predicates and comparisons.
  const auto* isNan = lookup("is_nan");
  ASSERT_NE(isNan, nullptr);
  EXPECT_EQ(isNan->front().signature->toString(), "(double) -> boolean");

  // A variadic registration has to survive the round trip as variable arity
  // rather than as a one-argument signature.
  const auto* conjunction = lookup("and");
  ASSERT_NE(conjunction, nullptr);
  EXPECT_TRUE(conjunction->front().signature->variableArity());
}

// Type coverage is registered through helpers that mirror Velox's
// RegistrationHelpers.h, so a function's breadth is decided by which helper it
// is passed to. That makes a whole type set easy to gain and equally easy to
// lose in one edit, which is what this pins.
TEST_F(GpuFunctionRegistryTest, numericBreadthMatchesVeloxTypeSets) {
  registerPrestoGpuFunctions("");

  auto signaturesOf = [](const std::vector<GpuFunctionEntry>* entries) {
    std::vector<std::string> out;
    for (const auto& entry : *entries) {
      out.push_back(entry.signature->toString());
    }
    std::sort(out.begin(), out.end());
    return out;
  };

  // registerUnaryNumeric: four integral widths plus double and real.
  const auto* abs = lookup("abs");
  ASSERT_NE(abs, nullptr);
  EXPECT_EQ(
      signaturesOf(abs),
      (std::vector<std::string>{
          "(bigint) -> bigint",
          "(double) -> double",
          "(integer) -> integer",
          "(real) -> real",
          "(smallint) -> smallint",
          "(tinyint) -> tinyint"}));

  // ceiling is an alias of ceil upstream, so both names carry the same set.
  EXPECT_EQ(signaturesOf(lookup("ceil")), signaturesOf(lookup("ceiling")));

  // Floating point only, matching registerUnaryFloatingPoint for negate.
  EXPECT_EQ(
      signaturesOf(lookup("truncate")).size(), 4u); // 2 types x 2 arities

  // Genuinely double-only upstream: breadth here would be a divergence, not an
  // improvement.
  EXPECT_EQ(
      signaturesOf(lookup("ln")), (std::vector<std::string>{"(double) -> double"}));
}

// A decimal signature is the case where the type string is not just the type
// name: it spells out precision and scale as variables, and those have to be
// declared before the signature will build. Getting this wrong is invisible at
// registration -- SimpleTypeTrait<ShortDecimal<P, S>> inherits
// TypeTraits<BIGINT>, so the function would register cleanly as taking a bigint
// and then simply never match a decimal call.
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

  // The signature is only useful if it binds a concrete decimal call, which is
  // what SignatureBinder is asked at resolution time.
  const std::vector<TypePtr> arguments{DECIMAL(10, 2), DECIMAL(10, 2)};
  exec::SignatureBinder binder(
      *entries->front().signature, arguments, TypeCoercer::defaults());
  ASSERT_TRUE(binder.tryBind());
  EXPECT_TRUE(binder.tryResolveReturnType()->equivalent(*DECIMAL(10, 2)));

  // A bigint call must not bind to it -- the failure the old derivation caused,
  // in reverse.
  const std::vector<TypePtr> bigints{BIGINT(), BIGINT()};
  exec::SignatureBinder wrongType(
      *entries->front().signature, bigints, TypeCoercer::defaults());
  EXPECT_FALSE(wrongType.tryBind());
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
