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
#include "velox/exec/AggregateFunctionAdapter.h"

#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/aggregates/tests/AggregationTestBase.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

namespace facebook::velox::aggregate::test {

class AggregateFunctionAdapterTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    // allowInputShuffle();
  }
};

TEST_F(AggregateFunctionAdapterTest, registerCompanionFunctions) {
  auto originalSignatures = getAggregateFunctionSignatures("avg");

  auto signatures = getAggregateFunctionSignatures("avg_partial");
  EXPECT_TRUE(signatures.has_value());
  EXPECT_EQ(signatures->size(), originalSignatures->size());
  for (const auto& signature : signatures.value()) {
    if (signature->argumentTypes()[0].baseName() != "DECIMAL") {
      EXPECT_EQ(signature->returnType().toString(), "row(double,bigint)");
    } else {
      EXPECT_EQ(signature->returnType().toString(), "varbinary");
    }
  }

  signatures = getAggregateFunctionSignatures("avg_merge");
  EXPECT_TRUE(signatures.has_value());
  EXPECT_EQ(signatures->size(), 2);
  for (const auto& signature : signatures.value()) {
    if (signature->argumentTypes()[0].baseName() == "varbinary") {
      EXPECT_EQ(signature->returnType().toString(), "varbinary");
    } else {
      EXPECT_EQ(signature->returnType().toString(), "row(double,bigint)");
    }
  }
}

TEST_F(AggregateFunctionAdapterTest, queryCompanionFunctions) {
  auto companionSignatures = getCompanionFunctionSignatures("avg").value();
  EXPECT_EQ(companionSignatures.size(), 4);

  auto checkSignatures = [&companionSignatures](
                             CompanionType companionType,
                             const std::string& suffix) {
    const auto& signatures = companionSignatures[companionType];
    EXPECT_GT(signatures.size(), 0);
    for (const auto& entry : signatures) {
      EXPECT_NE(entry.functionName_.find("avg_" + suffix), std::string::npos);
      EXPECT_GT(entry.signatures_.size(), 0);
    }
  };

  checkSignatures(CompanionType::kPartial, "partial");
  checkSignatures(CompanionType::kMerge, "merge");
  checkSignatures(CompanionType::kExtract, "extract");
  checkSignatures(CompanionType::kRetract, "retract");
}

TEST_F(AggregateFunctionAdapterTest, avgPartial) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<bool>({true, false, true, false, true})});
  auto expected = makeRowVector(
      {makeFlatVector<bool>({false, true}),
       makeRowVector(
           {makeFlatVector<double>({6, 9}), makeFlatVector<int64_t>({2, 3})})});
  testAggregations({data}, {"c1"}, {"avg_partial(c0)"}, {expected});
}

TEST_F(AggregateFunctionAdapterTest, avgMerge) {
  auto data = makeRowVector(
      {makeFlatVector<bool>({true, false, true, false, true}),
       makeRowVector(
           {makeFlatVector<double>({1.1, 2.2, 3.3, 4.4, 5.5}),
            makeFlatVector<int64_t>({1, 2, 3, 4, 5})})});
  auto expected = makeRowVector(
      {makeFlatVector<bool>({false, true}),
       makeRowVector(
           {makeFlatVector<double>({6.6, 9.9}),
            makeFlatVector<int64_t>({6, 9})})});
  testAggregations({data}, {"c0"}, {"avg_merge(c1)"}, {expected});
}

class AggregateFunctionAdapterFunctionTest
    : public functions::test::FunctionBaseTest {
 public:
  AggregateFunctionAdapterFunctionTest() {
    aggregate::prestosql::registerAllAggregateFunctions();
  }
};

TEST_F(AggregateFunctionAdapterFunctionTest, registerCompanionFunctions) {
  auto returnType =
      resolveVectorFunction("avg_extract_double", {ROW({DOUBLE(), BIGINT()})});
  EXPECT_TRUE(returnType != nullptr);
  EXPECT_TRUE(returnType->kind() == TypeKind::DOUBLE);

  returnType =
      resolveVectorFunction("avg_extract_real", {ROW({DOUBLE(), BIGINT()})});
  EXPECT_TRUE(returnType != nullptr);
  EXPECT_TRUE(returnType->kind() == TypeKind::REAL);

  returnType = resolveVectorFunction(
      "avg_retract", {ROW({DOUBLE(), BIGINT()}), ROW({DOUBLE(), BIGINT()})});
  EXPECT_TRUE(returnType != nullptr);
  EXPECT_TRUE(returnType->kind() == TypeKind::ROW);

  returnType = resolveVectorFunction("avg_retract", {VARBINARY(), VARBINARY()});
  EXPECT_TRUE(returnType != nullptr);
  EXPECT_TRUE(returnType->kind() == TypeKind::VARBINARY);
}

TEST_F(AggregateFunctionAdapterFunctionTest, avgExtract) {
  auto data = makeRowVector(
      {makeFlatVector<double>({6, 9}), makeFlatVector<int64_t>({2, 3})});
  auto expected = makeFlatVector<double>({3.0, 3.0});
  auto result = evaluate("avg_extract_double(c0)", makeRowVector({data}));
  assertEqualVectors(expected, result);
}

TEST_F(AggregateFunctionAdapterFunctionTest, avgRetract) {
  auto data = makeRowVector(
      {makeRowVector(
           {makeFlatVector<double>({6, 9}), makeFlatVector<int64_t>({2, 3})}),
       makeRowVector(
           {makeFlatVector<double>({1, 2}), makeFlatVector<int64_t>({1, 1})})});
  auto expected = makeRowVector(
      {makeFlatVector<double>({5, 7}), makeFlatVector<int64_t>({1, 2})});
  auto result = evaluate("avg_retract(c0, c1)", data);
  assertEqualVectors(expected, result);
}

} // namespace facebook::velox::aggregate::test
