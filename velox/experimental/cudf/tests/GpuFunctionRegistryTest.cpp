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
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/functions/prestosql/DecimalFunctions.h"
#include "velox/type/TypeCoercer.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

std::unique_ptr<cudf::column> launcherA(
    const std::vector<GpuArgView>&,
    const GpuFunctionInstance&,
    cudf::size_type,
    cudf::data_type,
    uint8_t*,
    cuda::stream_ref,
    rmm::device_async_resource_ref) {
  return nullptr;
}

std::unique_ptr<cudf::column> launcherB(
    const std::vector<GpuArgView>&,
    const GpuFunctionInstance&,
    cudf::size_type,
    cudf::data_type,
    uint8_t*,
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
      /*variableConstraints=*/{},
      /*argumentKinds=*/{TypeKind::DOUBLE, TypeKind::DOUBLE},
      /*returnKind=*/TypeKind::DOUBLE};
}

GpuFunctionSignature bigintBinary() {
  return GpuFunctionSignature{
      "bigint",
      {"bigint", "bigint"},
      /*variadicTail=*/false,
      /*integerVariables=*/{},
      /*variableConstraints=*/{},
      /*argumentKinds=*/{TypeKind::BIGINT, TypeKind::BIGINT},
      /*returnKind=*/TypeKind::BIGINT};
}

// Stand-in for a function with no initialize(): a one-byte empty instance.
GpuFunctionInstanceSpec noInitialize() {
  return GpuFunctionInstanceSpec{nullptr, 1, 1};
}

// The return type Velox's CPU registry resolves for this call, or null if no
// CPU overload accepts it.
TypePtr resolveVeloxReturnType(
    const std::string& name,
    const std::vector<TypePtr>& arguments) {
  auto resolved = exec::simpleFunctions().resolveFunction(name, arguments);
  return resolved.has_value() ? resolved->type() : nullptr;
}

// The same for the GPU registry: the first registered overload that binds,
// through the SignatureBinder the evaluator uses.
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

// A second implementation under the same name and signature replaces the
// first, as on the CPU, so the dialect registered last wins.
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

// Prefixes let both dialects load into one process without colliding.
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

std::vector<std::string> signaturesOf(const std::string& name) {
  std::vector<std::string> out;
  if (const auto* entries = lookup(name)) {
    for (const auto& entry : *entries) {
      out.push_back(entry.signature->toString());
    }
  }
  std::sort(out.begin(), out.end());
  // A decimal signature is backed by five kernels that differ only in physical
  // type; list it once.
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// The signatures derived from SimpleTypeTrait must equal the ones Velox
// registers for the same functions, through the helpers that mirror
// RegistrationHelpers.h, since SignatureBinder matches calls against them.
TEST_F(GpuFunctionRegistryTest, registrationsCarryVeloxSignatures) {
  registerPrestoGpuFunctions("");

  const std::vector<std::pair<std::string, std::vector<std::string>>> expected{
      // PlusFunction for floating point, CheckedPlusFunction for integers, and
      // the decimal function.
      {"plus",
       {"(bigint,bigint) -> bigint",
        "(decimal(i1,i5),decimal(i2,i6)) -> decimal(i3,i7)",
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

  // Velox registers the numeric ceil under both names and the decimal one
  // under "ceil" alone.
  EXPECT_EQ(signaturesOf("ceiling").size(), signaturesOf("ceil").size() - 1);
  // Two floating-point types at two arities, plus one decimal signature per
  // arity.
  EXPECT_EQ(signaturesOf("truncate").size(), 6);
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
      /*integerVariables=*/{"i1", "i5", "i1", "i5", "i1", "i5"},
      // Free here: the result precision and scale come from the argument
      // ones only because this signature repeats i1 and i5 in the return.
      /*variableConstraints=*/{},
      // Long decimals: int128 storage, unlike the short-decimal registration
      // with the same signature string.
      /*argumentKinds=*/{TypeKind::HUGEINT, TypeKind::HUGEINT},
      /*returnKind=*/TypeKind::HUGEINT};
  ASSERT_TRUE(
      registerGpuKernel({"decimal_add"}, signature, launcherA, noInitialize()));

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

// Resolves the same decimal calls through Velox's CPU registry and the GPU one
// and requires the same result type. A slip in a constraint expression still
// binds, just to a type the kernel was not compiled for.
TEST_F(GpuFunctionRegistryTest, decimalResultTypesMatchTheCpuRegistry) {
  registerPrestoGpuFunctions("");

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

// A decimal function's rescale factors come from initialize(), so its
// registration must carry a sized, initializable instance.
TEST_F(GpuFunctionRegistryTest, decimalFunctionsCarryAnInitializedInstance) {
  registerPrestoGpuFunctions("");

  const auto* entries = lookup("plus");
  ASSERT_NE(entries, nullptr);

  const auto decimalEntry = std::find_if(
      entries->begin(), entries->end(), [](const GpuFunctionEntry& entry) {
        return entry.signature->returnType().baseName() == "decimal";
      });
  ASSERT_NE(decimalEntry, entries->end());

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

  // Different scales give different state.
  std::vector<std::byte> other(decimalEntry->instanceSpec.size);
  const std::vector<TypePtr> reversed{DECIMAL(20, 6), DECIMAL(20, 1)};
  decimalEntry->instanceSpec.initialize(other.data(), reversed, config);
  EXPECT_EQ(static_cast<int>(other[0]), 0);
  EXPECT_EQ(static_cast<int>(other[1]), 5);

  // A function without initialize() registers no initializer.
  const auto* doubles = lookup("power");
  ASSERT_NE(doubles, nullptr);
  EXPECT_EQ(doubles->front().instanceSpec.initialize, nullptr);
}

// Velox registers five storage-width combinations per decimal function, and
// all render the same signature string, so the registry keys each kernel on
// its physical argument and return types as well.
TEST_F(GpuFunctionRegistryTest, decimalOverloadsAreKeyedByPhysicalType) {
  registerPrestoGpuFunctions("");

  const auto* entries = lookup("plus");
  ASSERT_NE(entries, nullptr);

  std::vector<std::pair<std::vector<TypeKind>, TypeKind>> decimalKinds;
  for (const auto& entry : *entries) {
    if (entry.signature->returnType().baseName() == "decimal") {
      decimalKinds.emplace_back(entry.argumentKinds, entry.returnKind);
    }
  }

  // registerDecimalBinary's five combinations.
  EXPECT_EQ(decimalKinds.size(), 5);

  // All distinct: a duplicate would be two kernels the registry cannot tell
  // apart.
  std::sort(decimalKinds.begin(), decimalKinds.end());
  EXPECT_EQ(
      std::adjacent_find(decimalKinds.begin(), decimalKinds.end()),
      decimalKinds.end())
      << "two decimal kernels share a signature and a physical shape";

  // Each is one of the two decimal storage widths.
  for (const auto& [arguments, returnKind] : decimalKinds) {
    for (const auto kind : arguments) {
      EXPECT_TRUE(kind == TypeKind::BIGINT || kind == TypeKind::HUGEINT);
    }
    EXPECT_TRUE(
        returnKind == TypeKind::BIGINT || returnKind == TypeKind::HUGEINT);
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
