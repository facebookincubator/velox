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

#include "velox/expression/SignatureBinder.h"
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/functions/prestosql/DecimalFunctions.h"
#include "velox/type/TypeCoercer.h"

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

std::unique_ptr<cudf::column> launcherA(
    const std::vector<GpuArgView>&,
    const GpuFunctionInstance&,
    cudf::size_type,
    cudf::data_type,
    cuda::stream_ref,
    rmm::device_async_resource_ref) {
  return nullptr;
}

std::unique_ptr<cudf::column> launcherB(
    const std::vector<GpuArgView>&,
    const GpuFunctionInstance&,
    cudf::size_type,
    cudf::data_type,
    cuda::stream_ref,
    rmm::device_async_resource_ref) {
  return nullptr;
}

GpuFunctionSignature doubleBinary() {
  return GpuFunctionSignature{
      "double",
      {"double", "double"},
      /*variadicTail=*/false,
      /*integerVariables=*/{},
      /*variableConstraints=*/{}};
}

GpuFunctionSignature bigintBinary() {
  return GpuFunctionSignature{
      "bigint",
      {"bigint", "bigint"},
      /*variadicTail=*/false,
      /*integerVariables=*/{},
      /*variableConstraints=*/{}};
}

// Stand-in for a function with no initialize(): the instance is an empty
// struct, so one byte with no setup, which is what most functions register.
GpuFunctionInstanceSpec noInitialize() {
  return GpuFunctionInstanceSpec{nullptr, 1, 1};
}

/// The return type Velox's own CPU registry resolves for this call, or null if
/// no CPU overload accepts it.
TypePtr resolveVeloxReturnType(
    const std::string& name,
    const std::vector<TypePtr>& arguments) {
  auto resolved = exec::simpleFunctions().resolveFunction(name, arguments);
  return resolved.has_value() ? resolved->type() : nullptr;
}

/// The same question asked of the GPU registry: the first registered overload
/// whose signature binds, resolved through the same SignatureBinder the
/// evaluator uses.
TypePtr resolveGpuReturnType(
    const std::string& name,
    const std::vector<TypePtr>& arguments) {
  const auto& registry = gpuFunctionRegistry();
  auto it = registry.find(name);
  if (it == registry.end()) {
    return nullptr;
  }
  for (const auto& entry : it->second) {
    exec::SignatureBinder binder(
        *entry.signature, arguments, TypeCoercer::defaults());
    if (binder.tryBind()) {
      if (auto type = binder.tryResolveReturnType()) {
        return type;
      }
    }
  }
  return nullptr;
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
  ASSERT_TRUE(registerGpuKernel(
      {"power", "pow"}, doubleBinary(), launcherA, noInitialize()));

  ASSERT_NE(lookup("power"), nullptr);
  ASSERT_NE(lookup("pow"), nullptr);
  EXPECT_EQ(lookup("power")->front().launch, launcherA);
  EXPECT_EQ(lookup("pow")->front().launch, launcherA);
}

TEST_F(GpuFunctionRegistryTest, lowercasesNamesLikeVelox) {
  ASSERT_TRUE(
      registerGpuKernel({"MyFunc"}, doubleBinary(), launcherA, noInitialize()));

  EXPECT_NE(lookup("myfunc"), nullptr);
  EXPECT_EQ(lookup("MyFunc"), nullptr);
}

TEST_F(GpuFunctionRegistryTest, differentSignaturesCoexistAsOverloads) {
  ASSERT_TRUE(
      registerGpuKernel({"plus"}, doubleBinary(), launcherA, noInitialize()));
  ASSERT_TRUE(
      registerGpuKernel({"plus"}, bigintBinary(), launcherB, noInitialize()));

  const auto* entries = lookup("plus");
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(entries->size(), 2);
}

// The mechanism dialect separation rests on: a second implementation under the
// same name and signature replaces the first, so whichever dialect registers
// last wins, exactly as on the CPU side.
TEST_F(GpuFunctionRegistryTest, sameSignatureOverwritesByDefault) {
  ASSERT_TRUE(
      registerGpuKernel({"divide"}, doubleBinary(), launcherA, noInitialize()));
  ASSERT_TRUE(
      registerGpuKernel({"divide"}, doubleBinary(), launcherB, noInitialize()));

  const auto* entries = lookup("divide");
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(entries->size(), 1);
  EXPECT_EQ(entries->front().launch, launcherB);
}

TEST_F(GpuFunctionRegistryTest, overwriteFalseKeepsTheIncumbent) {
  ASSERT_TRUE(
      registerGpuKernel({"divide"}, doubleBinary(), launcherA, noInitialize()));
  EXPECT_FALSE(registerGpuKernel(
      {"divide"},
      doubleBinary(),
      launcherB,
      noInitialize(),
      /*overwrite=*/false));

  const auto* entries = lookup("divide");
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(entries->size(), 1);
  EXPECT_EQ(entries->front().launch, launcherA);
}

// Guards the naming contract between the two dialect files. Prefixing is how
// both can be loaded into one process without colliding.
TEST_F(GpuFunctionRegistryTest, prefixSeparatesDialects) {
  ASSERT_TRUE(registerGpuKernel(
      {"presto.divide"}, doubleBinary(), launcherA, noInitialize()));
  ASSERT_TRUE(registerGpuKernel(
      {"spark.divide"}, doubleBinary(), launcherB, noInitialize()));

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
  // Two floating-point overloads from the plain struct, four integral ones from
  // CheckedPlusFunction, and one decimal, under one name. Overloads of the same
  // name coexist rather than replacing each other, and which struct backs which
  // type is the whole checked/unchecked distinction.
  EXPECT_EQ(
      plusSignatures,
      (std::vector<std::string>{
          "(bigint,bigint) -> bigint",
          "(decimal(i1,i5),decimal(i2,i6)) -> decimal(i3,i7)",
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
  // ceil and ceiling agree on the numeric overloads but not on decimal:
  // Velox registers the numeric ceil under both names and the decimal one
  // under "ceil" alone, and this mirrors that rather than improving on it.
  EXPECT_EQ(
      signaturesOf(lookup("ceiling")).size(),
      signaturesOf(lookup("ceil")).size() - 1);

  // Floating point only, matching registerUnaryFloatingPoint for negate.
  // 2 floating-point types x 2 arities, plus 2 decimal entries -- one per
  // arity. Decimal contributes only two because ShortDecimal and LongDecimal
  // render the same signature string; see decimalOverloadsCollapseBySignature.
  EXPECT_EQ(signaturesOf(lookup("truncate")).size(), 6u);

  // Genuinely double-only upstream: breadth here would be a divergence, not an
  // improvement.
  EXPECT_EQ(
      signaturesOf(lookup("ln")),
      (std::vector<std::string>{"(double) -> double"}));
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
      /*integerVariables=*/{"i1", "i5", "i1", "i5", "i1", "i5"},
      // Free here: the result precision and scale come from the argument
      // ones only because this signature repeats i1 and i5 in the return.
      /*variableConstraints=*/{}};
  ASSERT_TRUE(
      registerGpuKernel({"decimal_add"}, signature, launcherA, noInitialize()));

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

// A decimal function's result precision and scale are computed from the
// argument ones by a constraint expression, and this is the test that keeps
// ours honest: resolve the same call through Velox's own CPU registration and
// through the GPU one, and require the same answer.
//
// Comparing against the CPU registry rather than against expected literals is
// deliberate. A transcription slip in a constraint -- naming i2 where i5 was
// meant, dropping the carry digit -- produces a signature that still binds and
// still resolves, just to a different type than the CPU would, and the kernel
// was compiled for the CPU's answer.
TEST_F(GpuFunctionRegistryTest, decimalResultTypesMatchTheCpuRegistry) {
  registerPrestoGpuFunctions("");

  // The CPU side, registered here so the comparison reads off one source.
  functions::registerDecimalPlus("");
  functions::registerDecimalMinus("");
  functions::registerDecimalMultiply("");
  functions::registerDecimalDivide("");
  functions::registerDecimalModulus("");
  functions::registerDecimalFloor("");
  functions::registerDecimalCeil("");
  functions::registerDecimalRound("");
  functions::registerDecimalTruncate("");

  // Short and long on both sides, equal and unequal scales, and a pair whose
  // sum would overflow 38 digits so the min(38, ...) clamp is exercised.
  const std::vector<std::vector<TypePtr>> calls{
      {DECIMAL(10, 2), DECIMAL(10, 2)},
      {DECIMAL(10, 2), DECIMAL(12, 4)},
      {DECIMAL(38, 10), DECIMAL(38, 10)},
      {DECIMAL(18, 0), DECIMAL(38, 20)},
      {DECIMAL(5, 5), DECIMAL(5, 0)},
  };

  for (const auto* name : {"plus", "minus", "multiply", "divide", "mod"}) {
    for (const auto& arguments : calls) {
      SCOPED_TRACE(
          fmt::format(
              "{}({}, {})",
              name,
              arguments[0]->toString(),
              arguments[1]->toString()));

      const auto cpuType = resolveVeloxReturnType(name, arguments);
      const auto gpuType = resolveGpuReturnType(name, arguments);

      // Either both resolve or neither does; a GPU registration that claims a
      // call the CPU declines is as wrong as one that declines a valid call.
      ASSERT_EQ(cpuType != nullptr, gpuType != nullptr)
          << "cpu=" << (cpuType ? cpuType->toString() : "none")
          << " gpu=" << (gpuType ? gpuType->toString() : "none");
      if (cpuType != nullptr) {
        EXPECT_TRUE(gpuType->equivalent(*cpuType))
            << "cpu=" << cpuType->toString() << " gpu=" << gpuType->toString();
      }
    }
  }

  // The unary forms, where the result scale collapses to zero.
  for (const auto* name : {"floor", "ceil", "round", "truncate"}) {
    for (const auto& type :
         {DECIMAL(10, 2), DECIMAL(38, 38), DECIMAL(5, 0), DECIMAL(20, 19)}) {
      SCOPED_TRACE(fmt::format("{}({})", name, type->toString()));
      const std::vector<TypePtr> arguments{type};

      const auto cpuType = resolveVeloxReturnType(name, arguments);
      const auto gpuType = resolveGpuReturnType(name, arguments);
      ASSERT_EQ(cpuType != nullptr, gpuType != nullptr)
          << "cpu=" << (cpuType ? cpuType->toString() : "none")
          << " gpu=" << (gpuType ? gpuType->toString() : "none");
      if (cpuType != nullptr) {
        EXPECT_TRUE(gpuType->equivalent(*cpuType))
            << "cpu=" << cpuType->toString() << " gpu=" << gpuType->toString();
      }
    }
  }
}

// initialize() is what makes a decimal function work at all: the rescale
// factors come from the argument types, not the values. This checks the state
// actually reaches the registry as a sized, initializable instance -- the
// mechanism the launcher depends on.
TEST_F(GpuFunctionRegistryTest, decimalFunctionsCarryAnInitializedInstance) {
  registerPrestoGpuFunctions("");

  const auto* entries = lookup("plus");
  ASSERT_NE(entries, nullptr);

  const auto decimalEntry = std::find_if(
      entries->begin(), entries->end(), [](const GpuFunctionEntry& entry) {
        return entry.signature->returnType().baseName() == "decimal";
      });
  ASSERT_NE(decimalEntry, entries->end());

  // Unlike every function registered before this, a decimal plus has state.
  ASSERT_NE(decimalEntry->instanceSpec.initialize, nullptr);
  EXPECT_GT(decimalEntry->instanceSpec.size, 0);

  // Running it must produce the rescale factors the argument scales imply:
  // DECIMAL(20,2) + DECIMAL(20,4) rescales the first operand by 10^2 and the
  // second not at all.
  std::vector<std::byte> instance(decimalEntry->instanceSpec.size);
  const std::vector<TypePtr> arguments{DECIMAL(20, 2), DECIMAL(20, 4)};
  const core::QueryConfig config{{}};
  decimalEntry->instanceSpec.initialize(instance.data(), arguments, config);
  EXPECT_EQ(static_cast<int>(instance[0]), 2);
  EXPECT_EQ(static_cast<int>(instance[1]), 0);

  // Different scales must give different state, which is what distinguishes a
  // shipped instance from a default-constructed one.
  std::vector<std::byte> other(decimalEntry->instanceSpec.size);
  const std::vector<TypePtr> reversed{DECIMAL(20, 6), DECIMAL(20, 1)};
  decimalEntry->instanceSpec.initialize(other.data(), reversed, config);
  EXPECT_EQ(static_cast<int>(other[0]), 0);
  EXPECT_EQ(static_cast<int>(other[1]), 5);

  // A function without initialize() registers no initializer at all, so the
  // launcher's default-constructed instance stays correct.
  const auto* doubles = lookup("power");
  ASSERT_NE(doubles, nullptr);
  EXPECT_EQ(doubles->front().instanceSpec.initialize, nullptr);
}

// A gap this registry has and the CPU one does not, recorded as a test so it
// cannot be forgotten.
//
// Velox registers five type combinations per binary decimal function -- (short,
// short) -> short, (long, long) -> long, and three mixed ones -- and each needs
// its own kernel, because behind the resolver a short decimal is an int64_t and
// a long one an __int128. But all five produce the *same* signature string,
// since ShortDecimal<P, S> and LongDecimal<P, S> both render "decimal(i1,i5)".
// This registry keys on the signature, so the five collapse to one entry and
// the last kernel registered wins.
//
// A (short, short) -> short call therefore resolves to whichever combination
// registered last, reading int64_t operands through an __int128 kernel. The
// result type is still correct -- decimalResultTypesMatchTheCpuRegistry passes
// -- which is exactly what makes this dangerous.
//
// TODO(gpu-sfi-decimal): key entries on the physical argument types as well as
// the signature, so resolution can pick the kernel compiled for the resolved
// precision rather than the one that registered last.
TEST_F(GpuFunctionRegistryTest, decimalOverloadsCollapseBySignature) {
  registerPrestoGpuFunctions("");

  const auto* entries = lookup("plus");
  ASSERT_NE(entries, nullptr);

  const auto decimals = std::count_if(
      entries->begin(), entries->end(), [](const GpuFunctionEntry& entry) {
        return entry.signature->returnType().baseName() == "decimal";
      });

  // One, where Velox's five combinations would want five distinguishable
  // kernels. Raise this the moment physical-type keying lands.
  EXPECT_EQ(decimals, 1)
      << "if this is no longer 1, decimal dispatch has been made "
         "physical-type aware and the TODO above can go";
}

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
