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
#include "velox/exec/AggregateCompanionAdapter.h"

#include <gtest/gtest.h>

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/tests/DummyAggregateFunction.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/FunctionRegistry.h"

using namespace facebook::velox::exec;

namespace facebook::velox::exec::test {

namespace {

class AggregateCompanionRegistryTest : public testing::Test {
 public:
  AggregateCompanionRegistryTest() {
    exec::aggregateFunctions().withWLock([&](auto& functionsMap) {
      functionsMap.clear();
      EXPECT_EQ(0, functionsMap.size());
    });
  }

 protected:
  void checkEqual(const TypePtr& actual, const TypePtr& expected) {
    if (expected) {
      EXPECT_NE(actual, nullptr);
      EXPECT_TRUE(actual->equivalent(*expected));
    } else {
      EXPECT_EQ(actual, nullptr);
    }
  }

  void checkAggregateSignaturesCount(
      const std::string& name,
      uint32_t numOfSignatures) {
    auto signatures = getAggregateFunctionSignatures(name);
    if (numOfSignatures > 0) {
      EXPECT_TRUE(signatures.has_value());
      EXPECT_EQ(signatures->size(), numOfSignatures);
    } else {
      EXPECT_FALSE(signatures.has_value());
    }
  }

  void checkScalarSignaturesCount(
      const std::string& name,
      uint32_t numOfSignatures) {
    auto signatures = getVectorFunctionSignatures(name);
    if (numOfSignatures > 0) {
      EXPECT_TRUE(signatures.has_value());
      EXPECT_EQ(signatures->size(), numOfSignatures);
    } else {
      EXPECT_FALSE(signatures.has_value());
    }
  }

  void checkAggregateTypeResolution(
      const std::string& name,
      const std::vector<TypePtr>& argTypes,
      const TypePtr& intermediateType,
      const TypePtr& resultType) {
    const auto& [resolvedResult, resolveIntermediate] =
        resolveAggregateFunction(name, argTypes);
    checkEqual(resolvedResult, resultType);
    checkEqual(resolveIntermediate, intermediateType);
  }

  void checkScalarTypeResolution(
      const std::string& name,
      const std::vector<TypePtr>& argTypes,
      const TypePtr& resultType) {
    const auto& resolvedResult = resolveFunction(name, argTypes);
    checkEqual(resolvedResult, resultType);
  }
};

TEST_F(AggregateCompanionRegistryTest, basic) {
  std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures{
      AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("array(double)")
          .argumentType("double")
          .build(),
      AggregateFunctionSignatureBuilder()
          .returnType("bigint")
          .intermediateType("array(bigint)")
          .argumentType("bigint")
          .build()};
  registerDummyAggregateFunction("aggregateFunc1", signatures);

  checkAggregateSignaturesCount("aggregateFunc1_partial", 2);
  checkAggregateTypeResolution(
      "aggregateFunc1_partial", {DOUBLE()}, ARRAY(DOUBLE()), ARRAY(DOUBLE()));
  checkAggregateTypeResolution(
      "aggregateFunc1_partial", {BIGINT()}, ARRAY(BIGINT()), ARRAY(BIGINT()));

  checkAggregateSignaturesCount("aggregateFunc1_merge", 2);
  checkAggregateTypeResolution(
      "aggregateFunc1_merge",
      {ARRAY(DOUBLE())},
      ARRAY(DOUBLE()),
      ARRAY(DOUBLE()));
  checkAggregateTypeResolution(
      "aggregateFunc1_merge",
      {ARRAY(BIGINT())},
      ARRAY(BIGINT()),
      ARRAY(BIGINT()));

  checkAggregateSignaturesCount("aggregateFunc1_merge_extract", 2);
  checkAggregateTypeResolution(
      "aggregateFunc1_merge_extract",
      {ARRAY(DOUBLE())},
      ARRAY(DOUBLE()),
      DOUBLE());
  checkAggregateTypeResolution(
      "aggregateFunc1_merge_extract",
      {ARRAY(BIGINT())},
      ARRAY(BIGINT()),
      BIGINT());

  checkScalarSignaturesCount("aggregateFunc1_extract", 2);
  checkScalarTypeResolution(
      "aggregateFunc1_extract", {ARRAY(DOUBLE())}, DOUBLE());
  checkScalarTypeResolution(
      "aggregateFunc1_extract", {ARRAY(BIGINT())}, BIGINT());
}

TEST_F(AggregateCompanionRegistryTest, extractFunctionNameWithSuffix) {
  std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures{
      AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("array(double)")
          .argumentType("double")
          .build(),
      AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("array(double)")
          .argumentType("double")
          .argumentType("double")
          .build(),
      AggregateFunctionSignatureBuilder()
          .returnType("bigint")
          .intermediateType("array(double)")
          .argumentType("bigint")
          .build(),
      AggregateFunctionSignatureBuilder()
          .returnType("map(bigint,array(bigint))")
          .intermediateType("array(double)")
          .argumentType("map(bigint,array(bigint))")
          .build(),
      AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("row(double, double)")
          .argumentType("bigint")
          .argumentType("double")
          .build()};
  registerDummyAggregateFunction("aggregateFunc2", signatures);

  checkAggregateSignaturesCount("aggregateFunc2_partial", 5);
  checkAggregateTypeResolution(
      "aggregateFunc2_partial", {DOUBLE()}, ARRAY(DOUBLE()), ARRAY(DOUBLE()));
  checkAggregateTypeResolution(
      "aggregateFunc2_partial",
      {DOUBLE(), DOUBLE()},
      ARRAY(DOUBLE()),
      ARRAY(DOUBLE()));
  checkAggregateTypeResolution(
      "aggregateFunc2_partial", {BIGINT()}, ARRAY(DOUBLE()), ARRAY(DOUBLE()));
  checkAggregateTypeResolution(
      "aggregateFunc2_partial",
      {MAP(BIGINT(), ARRAY(BIGINT()))},
      ARRAY(DOUBLE()),
      ARRAY(DOUBLE()));
  checkAggregateTypeResolution(
      "aggregateFunc2_partial",
      {BIGINT(), DOUBLE()},
      ROW({DOUBLE(), DOUBLE()}),
      ROW({DOUBLE(), DOUBLE()}));

  checkAggregateSignaturesCount("aggregateFunc2_merge", 2);
  checkAggregateTypeResolution(
      "aggregateFunc2_merge",
      {ARRAY(DOUBLE())},
      ARRAY(DOUBLE()),
      ARRAY(DOUBLE()));
  checkAggregateTypeResolution(
      "aggregateFunc2_merge",
      {ROW({DOUBLE(), DOUBLE()})},
      ROW({DOUBLE(), DOUBLE()}),
      ROW({DOUBLE(), DOUBLE()}));

  checkAggregateSignaturesCount("aggregateFunc2_merge_extract_double", 2);
  checkAggregateTypeResolution(
      "aggregateFunc2_merge_extract_double",
      {ARRAY(DOUBLE())},
      ARRAY(DOUBLE()),
      DOUBLE());
  checkAggregateTypeResolution(
      "aggregateFunc2_merge_extract_double",
      {ROW({DOUBLE(), DOUBLE()})},
      ROW({DOUBLE(), DOUBLE()}),
      DOUBLE());

  checkAggregateSignaturesCount("aggregateFunc2_merge_extract_bigint", 1);
  checkAggregateTypeResolution(
      "aggregateFunc2_merge_extract_bigint",
      {ARRAY(DOUBLE())},
      ARRAY(DOUBLE()),
      BIGINT());

  checkAggregateSignaturesCount(
      "aggregateFunc2_merge_extract_map_bigint_array_bigint", 1);
  checkAggregateTypeResolution(
      "aggregateFunc2_merge_extract_map_bigint_array_bigint",
      {ARRAY(DOUBLE())},
      ARRAY(DOUBLE()),
      MAP(BIGINT(), ARRAY(BIGINT())));

  checkScalarSignaturesCount("aggregateFunc2_extract_double", 2);
  checkScalarTypeResolution(
      "aggregateFunc2_extract_double", {ARRAY(DOUBLE())}, DOUBLE());
  checkScalarTypeResolution(
      "aggregateFunc2_extract_double", {ROW({DOUBLE(), DOUBLE()})}, DOUBLE());

  checkScalarSignaturesCount("aggregateFunc2_extract_bigint", 1);
  checkScalarTypeResolution(
      "aggregateFunc2_extract_bigint", {ARRAY(DOUBLE())}, BIGINT());

  checkScalarSignaturesCount(
      "aggregateFunc2_extract_map_bigint_array_bigint", 1);
  checkScalarTypeResolution(
      "aggregateFunc2_extract_map_bigint_array_bigint",
      {ARRAY(DOUBLE())},
      MAP(BIGINT(), ARRAY(BIGINT())));
}

TEST_F(
    AggregateCompanionRegistryTest,
    extractFunctionNameWithSuffixOfNestedTypes) {
  std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures{
      AggregateFunctionSignatureBuilder()
          .returnType("row(row(bigint, double))")
          .intermediateType("row(bigint, double)")
          .argumentType("bigint")
          .argumentType("double")
          .build(),
      AggregateFunctionSignatureBuilder()
          .returnType("row(row(bigint), double)")
          .intermediateType("row(bigint, double)")
          .argumentType("bigint")
          .build()};
  registerDummyAggregateFunction("aggregateFunc3", signatures);

  checkAggregateSignaturesCount(
      "aggregateFunc3_merge_extract_row_row_bigint_double_endrow_endrow", 1);
  checkAggregateTypeResolution(
      "aggregateFunc3_merge_extract_row_row_bigint_double_endrow_endrow",
      {ROW({BIGINT(), DOUBLE()})},
      ROW({BIGINT(), DOUBLE()}),
      ROW({ROW({BIGINT(), DOUBLE()})}));

  checkAggregateSignaturesCount(
      "aggregateFunc3_merge_extract_row_row_bigint_endrow_double_endrow", 1);
  checkAggregateTypeResolution(
      "aggregateFunc3_merge_extract_row_row_bigint_endrow_double_endrow",
      {ROW({BIGINT(), DOUBLE()})},
      ROW({BIGINT(), DOUBLE()}),
      ROW({ROW({BIGINT()}), DOUBLE()}));

  checkScalarSignaturesCount(
      "aggregateFunc3_extract_row_row_bigint_double_endrow_endrow", 1);
  checkScalarTypeResolution(
      "aggregateFunc3_extract_row_row_bigint_double_endrow_endrow",
      {ROW({BIGINT(), DOUBLE()})},
      ROW({ROW({BIGINT(), DOUBLE()})}));

  checkScalarSignaturesCount(
      "aggregateFunc3_extract_row_row_bigint_endrow_double_endrow", 1);
  checkScalarTypeResolution(
      "aggregateFunc3_extract_row_row_bigint_endrow_double_endrow",
      {ROW({BIGINT(), DOUBLE()})},
      ROW({ROW({BIGINT()}), DOUBLE()}));
}

TEST_F(AggregateCompanionRegistryTest, templateSignature) {
  std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures{
      AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("array(double)")
          .argumentType("double")
          .build(),
      AggregateFunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
          .intermediateType("array(T)")
          .argumentType("T")
          .build()};
  registerDummyAggregateFunction("aggregateFunc4", signatures);

  checkAggregateSignaturesCount("aggregateFunc4_partial", 2);
  checkAggregateTypeResolution(
      "aggregateFunc4_partial", {BIGINT()}, ARRAY(BIGINT()), ARRAY(BIGINT()));
  checkAggregateTypeResolution(
      "aggregateFunc4_partial", {DOUBLE()}, ARRAY(DOUBLE()), ARRAY(DOUBLE()));

  checkAggregateSignaturesCount("aggregateFunc4_merge", 2);
  checkAggregateTypeResolution(
      "aggregateFunc4_merge",
      {ARRAY(BIGINT())},
      ARRAY(BIGINT()),
      ARRAY(BIGINT()));
  checkAggregateTypeResolution(
      "aggregateFunc4_merge",
      {ARRAY(DOUBLE())},
      ARRAY(DOUBLE()),
      ARRAY(DOUBLE()));

  checkAggregateSignaturesCount("aggregateFunc4_merge_extract", 2);
  checkAggregateTypeResolution(
      "aggregateFunc4_merge_extract",
      {ARRAY(BIGINT())},
      ARRAY(BIGINT()),
      BIGINT());
  checkAggregateTypeResolution(
      "aggregateFunc4_merge_extract",
      {ARRAY(DOUBLE())},
      ARRAY(DOUBLE()),
      DOUBLE());

  checkScalarSignaturesCount("aggregateFunc4_extract", 2);
  checkScalarTypeResolution(
      "aggregateFunc4_extract", {ARRAY(BIGINT())}, BIGINT());
  checkScalarTypeResolution(
      "aggregateFunc4_extract", {ARRAY(DOUBLE())}, DOUBLE());
  checkScalarTypeResolution(
      "aggregateFunc4_extract",
      {ARRAY(ROW({MAP(VARCHAR(), BIGINT())}))},
      ROW({MAP(VARCHAR(), BIGINT())}));
}

TEST_F(
    AggregateCompanionRegistryTest,
    sameIntermediateTypeWithDifferentVariables) {
  std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures{
      AggregateFunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("double")
          .intermediateType("T")
          .argumentType("double")
          .argumentType("T")
          .build(),
      AggregateFunctionSignatureBuilder()
          .typeVariable("K")
          .returnType("bigint")
          .intermediateType("K")
          .argumentType("bigint")
          .argumentType("K")
          .build()};
  registerDummyAggregateFunction("aggregateFunc5", signatures);

  checkAggregateSignaturesCount("aggregateFunc5_partial", 2);
  checkAggregateTypeResolution(
      "aggregateFunc5_partial", {BIGINT(), BIGINT()}, BIGINT(), BIGINT());
  checkAggregateTypeResolution(
      "aggregateFunc5_partial", {DOUBLE(), DOUBLE()}, DOUBLE(), DOUBLE());

  checkAggregateSignaturesCount("aggregateFunc5_merge", 1);
  checkAggregateTypeResolution(
      "aggregateFunc5_merge", {BIGINT()}, BIGINT(), BIGINT());
  checkAggregateTypeResolution(
      "aggregateFunc5_merge", {DOUBLE()}, DOUBLE(), DOUBLE());

  checkAggregateSignaturesCount("aggregateFunc5_merge_extract_double", 1);
  checkAggregateTypeResolution(
      "aggregateFunc5_merge_extract_double",
      {VARBINARY()},
      VARBINARY(),
      DOUBLE());

  checkAggregateSignaturesCount("aggregateFunc5_merge_extract_bigint", 1);
  checkAggregateTypeResolution(
      "aggregateFunc5_merge_extract_bigint",
      {VARBINARY()},
      VARBINARY(),
      BIGINT());

  checkScalarSignaturesCount("aggregateFunc5_extract_double", 1);
  checkScalarTypeResolution(
      "aggregateFunc5_extract_double", {VARBINARY()}, DOUBLE());

  checkScalarSignaturesCount("aggregateFunc5_extract_bigint", 1);
  checkScalarTypeResolution(
      "aggregateFunc5_extract_bigint", {VARBINARY()}, BIGINT());
}

TEST_F(
    AggregateCompanionRegistryTest,
    resultTypeNotResolvableFromIntermediateType) {
  // We only register companion functions for original signatures whose result
  // type can be resolved from its intermediate type.
  std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures{
      AggregateFunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("array(T)")
          .intermediateType("varbinary")
          .argumentType("T")
          .build()};
  registerDummyAggregateFunction("aggregateFunc6", signatures);

  checkAggregateSignaturesCount("aggregateFunc6_partial", 0);

  checkAggregateSignaturesCount("aggregateFunc6_merge", 0);

  checkAggregateSignaturesCount("aggregateFunc6_merge_extract", 0);

  checkScalarSignaturesCount("aggregateFunc6_extract", 0);
}

namespace {
class DummyVectorFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& /* rows */,
      std::vector<VectorPtr>& /* args */,
      const TypePtr& /* outputType */,
      exec::EvalCtx& /* context */,
      VectorPtr& /* result */) const override {}

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("boolean")
                .argumentType("integer")
                .build()};
  }
};
} // namespace

TEST_F(AggregateCompanionRegistryTest, duplicateRegistration) {
  std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures{
      AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("array(double)")
          .argumentType("double")
          .build()};

  // Manual registration of a function with the same name as an already
  // registered companion function.
  registerDummyAggregateFunction("aggregateFunc7", signatures);
  EXPECT_FALSE(
      registerDummyAggregateFunction("aggregateFunc7_partial", signatures)
          .mainFunction);
  EXPECT_TRUE(
      registerDummyAggregateFunction("aggregateFunc7_partial", signatures, true)
          .mainFunction);

  // Register companion functions with the same name as an already registered
  // function.
  registerDummyAggregateFunction("aggregateFunc8_partial", signatures);
  EXPECT_FALSE(registerDummyAggregateFunction("aggregateFunc8", signatures)
                   .partialFunction);
  EXPECT_TRUE(registerDummyAggregateFunction("aggregateFunc8", signatures, true)
                  .partialFunction);

  exec::registerVectorFunction(
      "aggregateFunc9_extract",
      DummyVectorFunction::signatures(),
      std::make_unique<DummyVectorFunction>());
  EXPECT_FALSE(registerDummyAggregateFunction("aggregateFunc9", signatures)
                   .extractFunction);
  EXPECT_TRUE(registerDummyAggregateFunction("aggregateFunc9", signatures, true)
                  .extractFunction);
}

} // namespace

} // namespace facebook::velox::exec::test
