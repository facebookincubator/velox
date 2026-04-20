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

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/expression/Expr.h"
#include "velox/expression/FieldReference.h"
#include "velox/expression/FunctionSignature.h"

#include <gtest/gtest.h>

using namespace facebook::velox;

namespace facebook::velox::cudf_velox {

namespace {

// Stand-in for a real `CudfFunction` used to mock registrations in the tests
// below. The tag is intended to identy the registration that was used, for
// testing purposes.
class TagFunction : public CudfFunction {
 public:
  explicit TagFunction(std::string tag) : tag_(std::move(tag)) {}

  const std::string& tag() const {
    return tag_;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& /*inputColumns*/,
      rmm::cuda_stream_view /*stream*/,
      rmm::device_async_resource_ref /*mr*/) const override {
    VELOX_UNREACHABLE("TagFunction::eval should not be called in these tests");
  }

 private:
  std::string tag_;
};

CudfFunctionFactory tagFactory(std::string tag) {
  return [tag](const std::string&, const std::shared_ptr<exec::Expr>&) {
    return std::make_shared<TagFunction>(tag);
  };
}

std::string tagOf(const std::shared_ptr<CudfFunction>& fn) {
  auto* tagged = dynamic_cast<TagFunction*>(fn.get());
  return tagged ? tagged->tag() : std::string{"<null>"};
}

// Necessary to mock `exec::Expr` call nodes successfully because the fake
// function names used in these tests aren't registered with Velox's scalar
// registry, so the usual `ExprSet` compilation path would reject them.
std::shared_ptr<exec::Expr> makeCall(
    const std::string& name,
    const TypePtr& returnType,
    const std::vector<TypePtr>& argTypes) {
  std::vector<std::shared_ptr<exec::Expr>> inputs;
  inputs.reserve(argTypes.size());
  for (const auto& argType : argTypes) {
    inputs.push_back(
        std::make_shared<exec::Expr>(
            argType,
            std::vector<std::shared_ptr<exec::Expr>>{},
            "field",
            /*specialFormKind=*/std::nullopt,
            /*supportsFlatNoNullsFastPath=*/false,
            /*trackCpuUsage=*/false));
  }
  return std::make_shared<exec::Expr>(
      returnType,
      std::move(inputs),
      name,
      /*specialFormKind=*/std::nullopt,
      /*supportsFlatNoNullsFastPath=*/false,
      /*trackCpuUsage=*/false);
}

} // namespace

class FunctionRegistryTest : public ::testing::Test {};

// One function registration, test mock `TagFunction` testing infrastructure.
TEST_F(FunctionRegistryTest, singleSignatureDispatch) {
  const std::string name = "regtest_single";
  registerCudfFunction(
      name,
      tagFactory("double"),
      {exec::FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .argumentType("double")
           .build()});

  auto expr = makeCall(name, DOUBLE(), {DOUBLE(), DOUBLE()});
  auto fn = createCudfFunction(name, expr);
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(tagOf(fn), "double");
}

// Two function registrations, same name collision with different signatures.
// Test that both can coexist.
TEST_F(FunctionRegistryTest, multipleSignaturesDispatchByInputTypes) {
  const std::string name = "regtest_multi";

  registerCudfFunction(
      name,
      tagFactory("double"),
      {exec::FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .argumentType("double")
           .build()});

  registerCudfFunction(
      name,
      tagFactory("date_interval"),
      {exec::FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .argumentType("interval day to second")
           .build()});

  auto doubleExpr = makeCall(name, DOUBLE(), {DOUBLE(), DOUBLE()});
  ASSERT_NE(createCudfFunction(name, doubleExpr), nullptr);
  EXPECT_EQ(tagOf(createCudfFunction(name, doubleExpr)), "double");

  auto dateExpr = makeCall(name, DATE(), {DATE(), INTERVAL_DAY_TIME()});
  ASSERT_NE(createCudfFunction(name, dateExpr), nullptr);
  EXPECT_EQ(tagOf(createCudfFunction(name, dateExpr)), "date_interval");
}

// Function registration whose types don't match any registered spec returns
// nullptr.
TEST_F(FunctionRegistryTest, noMatchingSignatureReturnsNull) {
  const std::string name = "regtest_nomatch";

  registerCudfFunction(
      name,
      tagFactory("double"),
      {exec::FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .argumentType("double")
           .build()});

  auto varcharExpr = makeCall(name, VARCHAR(), {VARCHAR(), VARCHAR()});
  EXPECT_EQ(createCudfFunction(name, varcharExpr), nullptr);
}

// Empty signature list acts as a wildcard match (used by `cast`/`try_cast`).
TEST_F(FunctionRegistryTest, emptySignaturesMatchAnyCall) {
  const std::string name = "regtest_empty_sigs";

  registerCudfFunction(name, tagFactory("dynamic"), {});

  auto anyExpr = makeCall(name, DOUBLE(), {VARCHAR(), BIGINT()});
  auto fn = createCudfFunction(name, anyExpr);
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(tagOf(fn), "dynamic");
}

// Typed function registered before empty spec: typed wins on exact match, empty
// catches the rest.
TEST_F(FunctionRegistryTest, emptySignaturesActAsFallback) {
  const std::string name = "regtest_typed_and_fallback";

  registerCudfFunction(
      name,
      tagFactory("typed"),
      {exec::FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .argumentType("double")
           .build()});
  registerCudfFunction(name, tagFactory("fallback"), {});

  auto matchExpr = makeCall(name, DOUBLE(), {DOUBLE(), DOUBLE()});
  EXPECT_EQ(tagOf(createCudfFunction(name, matchExpr)), "typed");

  auto mismatchExpr = makeCall(name, VARCHAR(), {VARCHAR(), VARCHAR()});
  EXPECT_EQ(tagOf(createCudfFunction(name, mismatchExpr)), "fallback");
}

// Two function registrations under the same name, second with
// `overwrite=false`. Test that the second is rejected wholesale (even with a
// non-overlapping signature) and the first remains the only callable one.
TEST_F(FunctionRegistryTest, overwriteFalsePreservesFirstRegistration) {
  const std::string name = "regtest_no_overwrite";

  ASSERT_TRUE(registerCudfFunction(
      name,
      tagFactory("first"),
      {exec::FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .argumentType("double")
           .build()}));
  EXPECT_FALSE(registerCudfFunction(
      name,
      tagFactory("second"),
      {exec::FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .argumentType("interval day to second")
           .build()},
      /*overwrite=*/false));

  auto expr = makeCall(name, DOUBLE(), {DOUBLE(), DOUBLE()});
  EXPECT_EQ(tagOf(createCudfFunction(name, expr)), "first");

  auto dateExpr = makeCall(name, DATE(), {DATE(), INTERVAL_DAY_TIME()});
  EXPECT_EQ(createCudfFunction(name, dateExpr), nullptr);
}

// Two function registrations under the same name with different signatures.
// Test that `canEvaluate` walks the spec list the same way `createCudfFunction`
// does, returning true when any spec matches and false otherwise.
TEST_F(FunctionRegistryTest, canEvaluateMultipleSignatures) {
  const std::string name = "regtest_can_eval_multi";

  registerCudfFunction(
      name,
      tagFactory("double"),
      {exec::FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .argumentType("double")
           .build()});
  registerCudfFunction(
      name,
      tagFactory("date_interval"),
      {exec::FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .argumentType("interval day to second")
           .build()});

  EXPECT_TRUE(
      FunctionExpression::canEvaluate(
          makeCall(name, DOUBLE(), {DOUBLE(), DOUBLE()})));
  EXPECT_TRUE(
      FunctionExpression::canEvaluate(
          makeCall(name, DATE(), {DATE(), INTERVAL_DAY_TIME()})));
  EXPECT_FALSE(
      FunctionExpression::canEvaluate(
          makeCall(name, VARCHAR(), {VARCHAR(), VARCHAR()})));
}

// `FieldReference` expression, no registry setup. Test that `canEvaluate`
// returns true without consulting the function registry at all.
TEST_F(FunctionRegistryTest, canEvaluateFieldReference) {
  auto fieldExpr = std::make_shared<exec::FieldReference>(
      DOUBLE(), std::vector<std::shared_ptr<exec::Expr>>{}, "c0");
  EXPECT_TRUE(FunctionExpression::canEvaluate(fieldExpr));
}

// `cast` and `try_cast` calls with cudf-supported type pairs. Test that
// `canEvaluate` dispatches these through `cudf::is_supported_cast` rather
// than through the normal signature-matching path.
TEST_F(FunctionRegistryTest, canEvaluateCastExpression) {
  EXPECT_TRUE(
      FunctionExpression::canEvaluate(makeCall("cast", BIGINT(), {DOUBLE()})));
  EXPECT_TRUE(
      FunctionExpression::canEvaluate(
          makeCall("try_cast", BIGINT(), {DOUBLE()})));
}

// Two function registrations under the same name with identical signatures,
// both using the default `overwrite=true`. Test that the second appends to the
// registry rather than replacing the first, and the first still wins on
// matching calls (first-registered wins ties). This is the behavior change
// introduced when the registry moved from one entry per name to many.
TEST_F(FunctionRegistryTest, overwriteTrueAppendsSpec) {
  const std::string name = "regtest_overwrite_true";
  auto sig = exec::FunctionSignatureBuilder()
                 .returnType("double")
                 .argumentType("double")
                 .argumentType("double")
                 .build();

  ASSERT_TRUE(registerCudfFunction(name, tagFactory("first"), {sig}));
  ASSERT_TRUE(registerCudfFunction(name, tagFactory("second"), {sig}));

  auto expr = makeCall(name, DOUBLE(), {DOUBLE(), DOUBLE()});
  EXPECT_EQ(tagOf(createCudfFunction(name, expr)), "first");
}

// One factory registered under two names via `registerCudfFunctions`. Test
// that calls to either name dispatch to the same factory.
TEST_F(FunctionRegistryTest, registerAliases) {
  const std::string primary = "regtest_alias_primary";
  const std::string alias = "regtest_alias_secondary";

  registerCudfFunctions(
      {primary, alias},
      tagFactory("shared"),
      {exec::FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .argumentType("double")
           .build()});

  auto primaryExpr = makeCall(primary, DOUBLE(), {DOUBLE(), DOUBLE()});
  EXPECT_EQ(tagOf(createCudfFunction(primary, primaryExpr)), "shared");

  auto aliasExpr = makeCall(alias, DOUBLE(), {DOUBLE(), DOUBLE()});
  EXPECT_EQ(tagOf(createCudfFunction(alias, aliasExpr)), "shared");
}

// Empty-signature (wildcard) registration added before a typed one under the
// same name. Test that the wildcard matches first and shadows the typed one,
// even on calls that the typed signature would bind exactly. Registration
// order determines match priority.
TEST_F(FunctionRegistryTest, emptySignatureRegisteredFirstShadowsTyped) {
  const std::string name = "regtest_empty_first";

  registerCudfFunction(name, tagFactory("fallback"), {});
  registerCudfFunction(
      name,
      tagFactory("typed"),
      {exec::FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .argumentType("double")
           .build()});

  auto doubleExpr = makeCall(name, DOUBLE(), {DOUBLE(), DOUBLE()});
  EXPECT_EQ(tagOf(createCudfFunction(name, doubleExpr)), "fallback");
}

// Call against a name that was never registered. Test that both
// `createCudfFunction` returns `nullptr` and `canEvaluate` returns false.
TEST_F(FunctionRegistryTest, unregisteredNameReturnsNullAndFalse) {
  const std::string name = "regtest_never_registered";
  auto expr = makeCall(name, DOUBLE(), {DOUBLE(), DOUBLE()});
  EXPECT_EQ(createCudfFunction(name, expr), nullptr);
  EXPECT_FALSE(FunctionExpression::canEvaluate(expr));
}

// `cast` call whose source/destination types `cudf::is_supported_cast`
// rejects. Test that `canEvaluate` returns false instead of falling through
// to the registry. Destination must be non-fixed-width to be rejected, since
// cuDF's check only requires the destination to be fixed-width for non-fixed-
// point casts.
TEST_F(FunctionRegistryTest, canEvaluateCastUnsupportedTypes) {
  auto unsupportedCast = makeCall("cast", ROW({DOUBLE()}), {INTEGER()});
  EXPECT_FALSE(FunctionExpression::canEvaluate(unsupportedCast));
}

// Malformed `cast` call with no input arguments. Test that `canEvaluate`
// returns false via its null-source-type guard rather than dereferencing a
// missing input.
TEST_F(FunctionRegistryTest, canEvaluateCastWithoutInputsReturnsFalse) {
  auto zeroArgCast = makeCall("cast", INTEGER(), {});
  EXPECT_FALSE(FunctionExpression::canEvaluate(zeroArgCast));
}

// `registerCudfFunctions` called with `overwrite=false` over a mix of an
// already-registered name and a brand-new one. Test that each name is handled
// independently: the existing one keeps its original factory, the new one
// gets the new factory.
TEST_F(FunctionRegistryTest, registerAliasesOverwriteFalse) {
  const std::string alreadyThere = "regtest_alias_existing";
  const std::string fresh = "regtest_alias_fresh";
  auto sig = exec::FunctionSignatureBuilder()
                 .returnType("double")
                 .argumentType("double")
                 .argumentType("double")
                 .build();

  registerCudfFunction(alreadyThere, tagFactory("original"), {sig});

  registerCudfFunctions(
      {alreadyThere, fresh},
      tagFactory("new"),
      {sig},
      /*overwrite=*/false);

  auto originalExpr = makeCall(alreadyThere, DOUBLE(), {DOUBLE(), DOUBLE()});
  EXPECT_EQ(tagOf(createCudfFunction(alreadyThere, originalExpr)), "original");

  auto freshExpr = makeCall(fresh, DOUBLE(), {DOUBLE(), DOUBLE()});
  EXPECT_EQ(tagOf(createCudfFunction(fresh, freshExpr)), "new");
}

// Call made with fewer arguments than the registered signature expects.
// Test that both `createCudfFunction` returns `nullptr` and `canEvaluate`
// returns false on arity mismatch.
TEST_F(FunctionRegistryTest, arityMismatchReturnsNull) {
  const std::string name = "regtest_arity";
  registerCudfFunction(
      name,
      tagFactory("two_args"),
      {exec::FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .argumentType("double")
           .build()});

  auto zeroArgExpr = makeCall(name, DOUBLE(), {});
  EXPECT_EQ(createCudfFunction(name, zeroArgExpr), nullptr);
  EXPECT_FALSE(FunctionExpression::canEvaluate(zeroArgExpr));
}

} // namespace facebook::velox::cudf_velox
