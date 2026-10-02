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

#include "velox/expression/SignatureBinder.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
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
    cudf::size_type,
    cudf::data_type,
    cuda::stream_ref,
    rmm::device_async_resource_ref) {
  return nullptr;
}

std::unique_ptr<cudf::column> launcherB(
    const std::vector<GpuArgView>&,
    cudf::size_type,
    cudf::data_type,
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

// The rendered signatures registered under `name`, sorted; empty when the name
// is unknown.
std::vector<std::string> signaturesOf(const std::string& name) {
  std::vector<std::string> signatures;
  const auto& registry = gpuFunctionRegistry();
  if (auto it = registry.find(name); it != registry.end()) {
    for (const auto& entry : it->second) {
      signatures.push_back(entry.signature->toString());
    }
  }
  std::sort(signatures.begin(), signatures.end());
  return signatures;
}

class GpuFunctionRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    clearGpuFunctionRegistry();
  }
};

// Registers under every alias, lowercased as Velox does; a second entry with
// the same name and signature replaces the first unless overwrite is false, so
// the dialect registered last wins; different signatures and different
// prefixes coexist.
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
      {GpuFunctionSignature{"boolean", {"boolean"}, /*variadicTail=*/true, {}},
       "(boolean...) -> boolean",
       {{{}, BOOLEAN()},
        {{BOOLEAN()}, BOOLEAN()},
        {{BOOLEAN(), BOOLEAN(), BOOLEAN()}, BOOLEAN()}},
       {{BIGINT(), BIGINT()}}},
      {GpuFunctionSignature{
           "decimal(i1,i5)",
           {"decimal(i1,i5)", "decimal(i1,i5)"},
           /*variadicTail=*/false,
           {"i1", "i5", "i1", "i5", "i1", "i5"}},
       "(decimal(i1,i5),decimal(i1,i5)) -> decimal(i1,i5)",
       {{{DECIMAL(10, 2), DECIMAL(10, 2)}, DECIMAL(10, 2)}},
       {{BIGINT(), BIGINT()}}},
  };
  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.rendered);
    clearGpuFunctionRegistry();
    ASSERT_TRUE(registerGpuKernel({"f"}, testCase.signature, launcherA));
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
// sets per helper, the aliases, both round arities, and the functions held back
// because their VELOX_USER_CHECKs would be no-ops. TODO(gpu-sfi-checks).
TEST_F(GpuFunctionRegistryTest, prestoRegistrations) {
  functions::prestosql::registerAllScalarFunctions();
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

  const std::vector<std::pair<std::string, std::vector<std::string>>> expected =
      {
          {"plus", {"(double,double) -> double", "(real,real) -> real"}},
          {"abs",
           {"(bigint) -> bigint",
            "(double) -> double",
            "(integer) -> integer",
            "(real) -> real",
            "(smallint) -> smallint",
            "(tinyint) -> tinyint"}},
          {"ln", {"(double) -> double"}},
          {"is_nan", {"(double) -> boolean"}},
          {"ceiling", signaturesOf("ceil")},
          // Two floating-point types at two arities.
          {"truncate",
           {"(double) -> double",
            "(double,integer) -> double",
            "(real) -> real",
            "(real,integer) -> real"}},
          {"and", {"(boolean...) -> boolean"}},
          {"bit_count", {}},
          {"bitwise_arithmetic_shift_right", {}},
          {"bitwise_shift_left", {}},
          {"bitwise_logical_shift_right", {}},
      };
  for (const auto& [name, signatures] : expected) {
    SCOPED_TRACE(name);
    EXPECT_EQ(signaturesOf(name), signatures);
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
