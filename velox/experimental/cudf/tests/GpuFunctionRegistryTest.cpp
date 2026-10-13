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

// Tests of the GPU simple-function registry: naming and collision policy, the
// signatures it builds from the strings the device side supplies, and what the
// Presto registration puts in it. They need no GPU; the launchers are stand-ins
// that are never invoked.

#include "velox/experimental/cudf/functions/GpuFunctionLookup.h"
#include "velox/experimental/cudf/functions/GpuSfiExpression.h"
#include "velox/experimental/cudf/tests/GpuTestFunctions.h"

#include "velox/core/Expressions.h"
#include "velox/core/QueryConfig.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/TypeCoercer.h"

#include <folly/String.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

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

// A decimal signature whose result repeats the argument precision and scale,
// stored in `kind`: int64 for short decimals, int128 for long ones.
GpuFunctionSignature decimalBinary(TypeKind kind) {
  return GpuFunctionSignature{
      "decimal(i1,i5)",
      {"decimal(i1,i5)", "decimal(i1,i5)"},
      /*variadicTail=*/false,
      // Named once per occurrence, as the device side collects them; the
      // builder rejects a redeclaration, so the duplicates must be dropped.
      /*integerVariables=*/{"i1", "i5", "i1", "i5", "i1", "i5"},
      /*variableConstraints=*/{},
      /*argumentKinds=*/{kind, kind},
      /*returnKind=*/kind};
}

// Stand-in for a function with no initialize(): a one-byte empty instance.
GpuFunctionInstanceSpec noInitialize() {
  return GpuFunctionInstanceSpec{nullptr, 1, 1};
}

// The launchers registered under `name`, in registry order; empty when the
// name is unknown.
std::vector<GpuLaunchFn> launchersOf(const std::string& name) {
  std::vector<GpuLaunchFn> launchers;
  const auto& registry = gpuFunctionRegistry();
  if (auto it = registry.find(name); it != registry.end()) {
    for (const auto& entry : it->second) {
      launchers.push_back(entry.launch);
    }
  }
  return launchers;
}

// Renders argument types for a trace message.
std::string describe(const std::vector<TypePtr>& types) {
  std::vector<std::string> names;
  for (const auto& type : types) {
    names.push_back(type->toString());
  }
  return folly::join(", ", names);
}

// The rendered signatures registered under `name`, sorted and listed once
// each, since a decimal signature is backed by five kernels that differ only
// in physical type; empty when the name is unknown.
std::vector<std::string> signaturesOf(const std::string& name) {
  std::vector<std::string> signatures;
  const auto& registry = gpuFunctionRegistry();
  if (auto it = registry.find(name); it != registry.end()) {
    for (const auto& entry : it->second) {
      signatures.push_back(entry.signature->toString());
    }
  }
  std::sort(signatures.begin(), signatures.end());
  signatures.erase(
      std::unique(signatures.begin(), signatures.end()), signatures.end());
  return signatures;
}

// The return type the GPU registry resolves for this call through the
// SignatureBinder the evaluator uses, or null when no overload binds.
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

// The decimal entries registered under `name`.
std::vector<const GpuFunctionEntry*> decimalEntriesOf(const std::string& name) {
  std::vector<const GpuFunctionEntry*> entries;
  for (const auto& entry : gpuFunctionRegistry().at(name)) {
    if (entry.signature->returnType().baseName() == "decimal") {
      entries.push_back(&entry);
    }
  }
  return entries;
}

class GpuFunctionRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    clearGpuFunctionRegistry();
    // Also registers TIMESTAMP WITH TIME ZONE, which the GPU registration
    // needs known to parse a signature naming it.
    functions::prestosql::registerAllScalarFunctions();
  }
};

// Registers under every alias, lowercased as Velox does; a second entry with
// the same name and signature replaces the first unless overwrite is false, so
// the dialect registered last wins; different signatures, different physical
// types behind one signature, and different prefixes coexist.
TEST_F(GpuFunctionRegistryTest, namesAndCollisions) {
  struct Registration {
    std::vector<std::string> aliases;
    GpuFunctionSignature signature;
    GpuLaunchFn launch;
    bool overwrite;
    bool accepted;
  };
  struct Case {
    std::string name;
    std::vector<Registration> registrations;
    std::vector<std::pair<std::string, std::vector<GpuLaunchFn>>> launchers;
  };
  const std::vector<Case> cases = {
      {"aliases",
       {{{"Power", "pow"}, doubleBinary(), launcherA, true, true}},
       {{"power", {launcherA}}, {"pow", {launcherA}}, {"Power", {}}}},
      {"overloads",
       {{{"plus"}, doubleBinary(), launcherA, true, true},
        {{"plus"}, bigintBinary(), launcherB, true, true}},
       {{"plus", {launcherA, launcherB}}}},
      {"physical overloads",
       {{{"plus"}, decimalBinary(TypeKind::BIGINT), launcherA, true, true},
        {{"plus"}, decimalBinary(TypeKind::HUGEINT), launcherB, true, true}},
       {{"plus", {launcherA, launcherB}}}},
      {"overwrite",
       {{{"divide"}, doubleBinary(), launcherA, true, true},
        {{"divide"}, doubleBinary(), launcherB, true, true}},
       {{"divide", {launcherB}}}},
      {"keep incumbent",
       {{{"divide"}, doubleBinary(), launcherA, true, true},
        {{"divide"}, doubleBinary(), launcherB, false, false}},
       {{"divide", {launcherA}}}},
      {"prefixes",
       {{{"presto.divide"}, doubleBinary(), launcherA, true, true},
        {{"spark.divide"}, doubleBinary(), launcherB, true, true}},
       {{"presto.divide", {launcherA}}, {"spark.divide", {launcherB}}}},
  };
  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    clearGpuFunctionRegistry();
    for (const auto& registration : testCase.registrations) {
      EXPECT_EQ(
          registerGpuKernel(
              registration.aliases,
              registration.signature,
              registration.launch,
              noInitialize(),
              registration.overwrite),
          registration.accepted);
    }
    for (const auto& [name, launchers] : testCase.launchers) {
      EXPECT_EQ(launchersOf(name), launchers) << name;
    }
  }
}

// Builds a Velox FunctionSignature from the device side's strings: a variadic
// tail matches any number of trailing arguments, and a decimal signature
// declares its precision and scale variables, named once per occurrence by the
// device side, so that it binds a decimal call rather than a bigint one.
TEST_F(GpuFunctionRegistryTest, signatures) {
  struct Case {
    GpuFunctionSignature signature;
    std::string rendered;
    std::vector<std::pair<std::vector<TypePtr>, TypePtr>> binds;
    std::vector<std::vector<TypePtr>> rejects;
  };
  const std::vector<Case> cases = {
      {doubleBinary(),
       "(double,double) -> double",
       {{{DOUBLE(), DOUBLE()}, DOUBLE()}},
       {{BIGINT(), BIGINT()}, {DOUBLE()}}},
      {GpuFunctionSignature{
           "boolean",
           {"boolean"},
           /*variadicTail=*/true,
           /*integerVariables=*/{},
           /*variableConstraints=*/{},
           /*argumentKinds=*/{TypeKind::BOOLEAN},
           /*returnKind=*/TypeKind::BOOLEAN},
       "(boolean...) -> boolean",
       {{{}, BOOLEAN()},
        {{BOOLEAN()}, BOOLEAN()},
        {{BOOLEAN(), BOOLEAN(), BOOLEAN()}, BOOLEAN()}},
       {{BIGINT(), BIGINT()}}},
      {decimalBinary(TypeKind::HUGEINT),
       "(decimal(i1,i5),decimal(i1,i5)) -> decimal(i1,i5)",
       {{{DECIMAL(10, 2), DECIMAL(10, 2)}, DECIMAL(10, 2)}},
       {{BIGINT(), BIGINT()}}},
  };
  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.rendered);
    clearGpuFunctionRegistry();
    ASSERT_TRUE(registerGpuKernel(
        {"f"}, testCase.signature, launcherA, noInitialize()));
    const auto& signature = *gpuFunctionRegistry().at("f").front().signature;
    EXPECT_EQ(signature.toString(), testCase.rendered);

    for (const auto& [arguments, returnType] : testCase.binds) {
      SCOPED_TRACE(describe(arguments));
      exec::SignatureBinder binder(
          signature, arguments, TypeCoercer::defaults());
      ASSERT_TRUE(binder.tryBind());
      EXPECT_TRUE(binder.tryResolveReturnType()->equivalent(*returnType));
    }
    for (const auto& arguments : testCase.rejects) {
      SCOPED_TRACE(describe(arguments));
      exec::SignatureBinder binder(
          signature, arguments, TypeCoercer::defaults());
      EXPECT_FALSE(binder.tryBind());
    }
  }
}

// The Presto registration mirrors RegistrationHelpers.h type set for type set,
// so every GPU signature is one Velox registers under the same name on the CPU,
// which SignatureBinder matches calls against; the explicit rows pin the type
// sets per helper, the aliases, both round arities, the decimal functions, and
// the checked bitwise functions.
TEST_F(GpuFunctionRegistryTest, prestoRegistrations) {
  registerPrestoGpuFunctions("");

  // Kleene and/or are special forms on the CPU and is_null is generic there,
  // so they have no signature string to compare against.
  const std::unordered_set<std::string> notComparable = {
      "and", "or", "is_null"};
  for (const auto& [name, entries] : gpuFunctionRegistry()) {
    if (notComparable.count(name) != 0) {
      continue;
    }
    SCOPED_TRACE(name);
    std::vector<std::string> cpuSignatures;
    for (const auto* signature : facebook::velox::getFunctionSignatures(name)) {
      cpuSignatures.push_back(signature->toString());
    }
    for (const auto& entry : entries) {
      EXPECT_THAT(
          cpuSignatures, testing::Contains(entry.signature->toString()));
    }
  }

  const std::vector<std::string> unaryNumeric = {
      "(bigint) -> bigint",
      "(double) -> double",
      "(integer) -> integer",
      "(real) -> real",
      "(smallint) -> smallint",
      "(tinyint) -> tinyint",
  };
  auto unaryNumericAndDecimal = unaryNumeric;
  unaryNumericAndDecimal.push_back("(decimal(i1,i5)) -> decimal(i2,i6)");
  const std::vector<std::pair<std::string, std::vector<std::string>>> expected =
      {
          // PlusFunction for floating point, CheckedPlusFunction for integers,
          // and the decimal function.
          {"plus",
           {"(bigint,bigint) -> bigint",
            "(decimal(i1,i5),decimal(i2,i6)) -> decimal(i3,i7)",
            "(double,double) -> double",
            "(integer,integer) -> integer",
            "(real,real) -> real",
            "(smallint,smallint) -> smallint",
            "(tinyint,tinyint) -> tinyint"}},
          {"abs", unaryNumeric},
          // Velox registers the numeric ceil under both names and the decimal
          // one under "ceil" alone.
          {"ceil", unaryNumericAndDecimal},
          {"ceiling", unaryNumeric},
          {"ln", {"(double) -> double"}},
          {"is_nan", {"(double) -> boolean"}},
          // Two floating-point types at two arities, and a decimal signature
          // per arity.
          {"truncate",
           {"(decimal(i1,i5),integer) -> decimal(i1,i5)",
            "(decimal(i1,i5)) -> decimal(i2,i6)",
            "(double) -> double",
            "(double,integer) -> double",
            "(real) -> real",
            "(real,integer) -> real"}},
          {"and", {"(boolean...) -> boolean"}},
          // Every integral pair widens to bigint, as the CPU's bitwise
          // registration does.
          {"bit_count",
           {"(bigint,bigint) -> bigint",
            "(integer,integer) -> bigint",
            "(smallint,smallint) -> bigint",
            "(tinyint,tinyint) -> bigint"}},
          {"bitwise_arithmetic_shift_right", {"(bigint,bigint) -> bigint"}},
          {"bitwise_shift_left", {"(bigint,bigint,bigint) -> bigint"}},
          {"bitwise_logical_shift_right", {"(bigint,bigint,bigint) -> bigint"}},
      };
  for (const auto& [name, signatures] : expected) {
    SCOPED_TRACE(name);
    EXPECT_THAT(
        signaturesOf(name), testing::UnorderedElementsAreArray(signatures));
  }
}

// TIMESTAMP WITH TIME ZONE is BIGINT underneath. The signature names the
// logical type and the kernel is compiled for the packed int64, so a call binds
// on a TIMESTAMP WITH TIME ZONE column alone, as the DATE registrations do on
// theirs.
TEST_F(GpuFunctionRegistryTest, timestampWithTimeZoneRendersItsLogicalType) {
  registerGpuTestFunctions();

  const auto& entries = gpuFunctionRegistry().at("test_millis_utc");
  ASSERT_EQ(entries.size(), 1);
  const auto& entry = entries.front();
  EXPECT_EQ(
      entry.signature->toString(), "(timestamp with time zone) -> bigint");
  EXPECT_THAT(entry.argumentKinds, testing::ElementsAre(TypeKind::BIGINT));
  EXPECT_EQ(entry.returnKind, TypeKind::BIGINT);

  auto call = [](const TypePtr& argumentType) {
    return std::make_shared<core::CallTypedExpr>(
        BIGINT(),
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(argumentType, "c0")},
        "test_millis_utc");
  };
  EXPECT_TRUE(GpuSfiExpression::canEvaluate(call(TIMESTAMP_WITH_TIME_ZONE())));
  EXPECT_FALSE(GpuSfiExpression::canEvaluate(call(BIGINT())));
}

// Resolves the same decimal calls through Velox's CPU registry and the GPU one
// and requires the same result type, or that neither resolves: a slip in a
// constraint expression still binds, just to a type the kernel was not
// compiled for, and a GPU registration that claims a call the CPU declines is
// as wrong as one that declines a valid call.
TEST_F(GpuFunctionRegistryTest, decimalResultTypesMatchTheCpu) {
  registerPrestoGpuFunctions("");

  struct Case {
    std::vector<std::string> names;
    std::vector<std::vector<TypePtr>> calls;
  };
  const std::vector<Case> cases = {
      // Short and long on both sides, equal and unequal scales, and a pair
      // whose sum would overflow 38 digits so the min(38, ...) clamp is
      // exercised.
      {{"plus", "minus", "multiply", "divide", "mod"},
       {
           {DECIMAL(10, 2), DECIMAL(10, 2)},
           {DECIMAL(10, 2), DECIMAL(12, 4)},
           {DECIMAL(38, 10), DECIMAL(38, 10)},
           {DECIMAL(18, 0), DECIMAL(38, 20)},
           {DECIMAL(5, 5), DECIMAL(5, 0)},
       }},
      // The unary forms, where the result scale collapses to zero.
      {{"floor", "ceil", "round", "truncate"},
       {
           {DECIMAL(10, 2)},
           {DECIMAL(38, 38)},
           {DECIMAL(5, 0)},
           {DECIMAL(20, 19)},
       }},
  };
  for (const auto& testCase : cases) {
    for (const auto& name : testCase.names) {
      for (const auto& arguments : testCase.calls) {
        SCOPED_TRACE(fmt::format("{}({})", name, describe(arguments)));
        const auto cpuType = facebook::velox::resolveFunction(name, arguments);
        const auto gpuType = resolveGpuReturnType(name, arguments);
        ASSERT_EQ(cpuType != nullptr, gpuType != nullptr)
            << "cpu=" << (cpuType ? cpuType->toString() : "none")
            << " gpu=" << (gpuType ? gpuType->toString() : "none");
        if (cpuType != nullptr) {
          EXPECT_TRUE(gpuType->equivalent(*cpuType))
              << "cpu=" << cpuType->toString()
              << " gpu=" << gpuType->toString();
        }
      }
    }
  }
}

// Velox registers five storage-width combinations per decimal function, all
// rendering the same signature string, so the registry keys each kernel on its
// physical argument and return types as well; a duplicate would be two kernels
// it cannot tell apart.
TEST_F(GpuFunctionRegistryTest, decimalKernelsAreKeyedByPhysicalType) {
  registerPrestoGpuFunctions("");

  std::vector<std::pair<std::vector<TypeKind>, TypeKind>> shapes;
  for (const auto* entry : decimalEntriesOf("plus")) {
    shapes.emplace_back(entry->argumentKinds, entry->returnKind);
    for (const auto kind : entry->argumentKinds) {
      EXPECT_THAT(kind, testing::AnyOf(TypeKind::BIGINT, TypeKind::HUGEINT));
    }
    EXPECT_THAT(
        entry->returnKind, testing::AnyOf(TypeKind::BIGINT, TypeKind::HUGEINT));
  }
  std::sort(shapes.begin(), shapes.end());
  EXPECT_EQ(shapes.size(), 5);
  EXPECT_EQ(std::adjacent_find(shapes.begin(), shapes.end()), shapes.end());
}

// A decimal function's rescale factors come from initialize(), so its
// registration carries a sized, initializable instance that derives them from
// the argument scales, while a function without initialize() registers none.
TEST_F(GpuFunctionRegistryTest, decimalInstancesAreInitialized) {
  registerPrestoGpuFunctions("");

  EXPECT_EQ(
      gpuFunctionRegistry().at("power").front().instanceSpec.initialize,
      nullptr);

  const auto& spec = decimalEntriesOf("plus").front()->instanceSpec;
  ASSERT_NE(spec.initialize, nullptr);
  ASSERT_GT(spec.size, 0);

  // DecimalPlusFunction rescales each operand to the wider scale, and stores
  // the two factors first.
  const std::vector<std::pair<int8_t, int8_t>> scales = {
      {2, 4}, {6, 1}, {3, 3}};
  for (const auto& [left, right] : scales) {
    SCOPED_TRACE(fmt::format("scales {} and {}", left, right));
    std::vector<std::byte> instance(spec.size);
    const std::vector<TypePtr> arguments{DECIMAL(20, left), DECIMAL(20, right)};
    const core::QueryConfig config{{}};
    spec.initialize(instance.data(), arguments, config);
    const auto wider = std::max(left, right);
    EXPECT_EQ(static_cast<int>(instance[0]), wider - left);
    EXPECT_EQ(static_cast<int>(instance[1]), wider - right);
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
