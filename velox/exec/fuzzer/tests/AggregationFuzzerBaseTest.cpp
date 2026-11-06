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

#include <gflags/gflags.h>
#include <gmock/gmock.h>

#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/exec/fuzzer/AggregationFuzzer.h"
#include "velox/exec/fuzzer/AggregationFuzzerBase.h"
#include "velox/exec/fuzzer/DuckQueryRunner.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/tests/SimpleAggregateFunctionsRegistration.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/parse/TypeResolver.h"

DECLARE_bool(enable_sorted_aggregations);
DECLARE_bool(enable_streaming_aggregations);

namespace facebook::velox::exec::test {
namespace {

// Exposes signature selection and argument generation independently of a run.
class TestAggregationFuzzer : public AggregationFuzzerBase {
 public:
  using AggregationFuzzerBase::addSignature;
  using AggregationFuzzerBase::AggregationFuzzerBase;
  using AggregationFuzzerBase::generateConstantInputs;
  using AggregationFuzzerBase::pickSignature;
  using AggregationFuzzerBase::signatureStats_;
};

// Supplies bounded values and a known valid literal to the full fuzzer.
class FixedInputGenerator : public InputGenerator {
 public:
  explicit FixedInputGenerator(std::vector<VectorPtr> values)
      : values_{std::move(values)} {}

  std::vector<VectorPtr> generate(
      const std::vector<TypePtr>& /*types*/,
      VectorFuzzer& /*fuzzer*/,
      FuzzerGenerator& /*rng*/,
      memory::MemoryPool* /*pool*/) override {
    return values_;
  }

  void reset() override {}

 private:
  const std::vector<VectorPtr> values_;
};

// Checks that successful fuzzer comparisons use literal aggregate arguments.
class LiteralArgumentResultVerifier : public ResultVerifier {
 public:
  bool supportsCompare() override {
    return true;
  }

  bool supportsVerify() override {
    return false;
  }

  void initialize(
      const std::vector<RowVectorPtr>& /*input*/,
      const std::vector<core::ExprPtr>& /*projections*/,
      const std::vector<std::string>& /*groupingKeys*/,
      const core::AggregationNode::Aggregate& aggregate,
      const std::string& /*aggregateName*/) override {
    ASSERT_EQ(aggregate.call->inputs().size(), 2);
    ASSERT_NE(
        std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
            aggregate.call->inputs()[1]),
        nullptr);
    ++numInitializations_;
  }

  bool compare(const RowVectorPtr& result, const RowVectorPtr& otherResult)
      override {
    ++numComparisons_;
    return assertEqualResults({result}, {otherResult});
  }

  bool verify(const RowVectorPtr& /*result*/) override {
    VELOX_UNSUPPORTED();
  }

  void reset() override {}

  uint32_t numInitializations() const {
    return numInitializations_;
  }

  uint32_t numComparisons() const {
    return numComparisons_;
  }

 private:
  // Retain totals across per-iteration resets to detect vacuous coverage.
  uint32_t numInitializations_{0};
  uint32_t numComparisons_{0};
};

} // namespace

class AggregationFuzzerBaseTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    aggregate::registerSimpleConstantForwardingAggregate("constant_forward");
    parse::registerTypeResolver();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
    fuzzer_ = std::make_unique<TestAggregationFuzzer>(
        123,
        std::unordered_map<std::string, std::shared_ptr<ResultVerifier>>{},
        std::unordered_map<std::string, std::shared_ptr<InputGenerator>>{},
        VectorFuzzer::Options::TimestampPrecision::kMilliSeconds,
        std::unordered_map<std::string, std::string>{},
        std::unordered_map<std::string, std::string>{},
        false,
        std::make_unique<DuckQueryRunner>(rootPool_.get()));
  }

  void TearDown() override {
    releaseFuzzer();
    OperatorTestBase::TearDown();
  }

  void releaseFuzzer() {
    fuzzer_.reset();
    connector::ConnectorRegistry::global().erase(kHiveConnectorId);
    dwrf::unregisterDwrfReaderFactory();
    dwrf::unregisterDwrfWriterFactory();
  }

  std::unique_ptr<TestAggregationFuzzer> fuzzer_;
};

TEST_F(AggregationFuzzerBaseTest, constantSignatureMetadata) {
  std::vector<FunctionSignaturePtr> signatures;
  for (bool generic : {false, true}) {
    SCOPED_TRACE(generic);
    auto builder = AggregateFunctionSignatureBuilder();
    if (generic) {
      builder.typeVariable("T");
    }
    const auto valueType = generic ? "T" : "bigint";
    signatures.push_back(builder.returnType(valueType)
                             .intermediateType(valueType)
                             .argumentType(valueType)
                             .constantArgumentType("bigint")
                             .constantArgumentType("boolean")
                             .build());
    ASSERT_TRUE(fuzzer_->addSignature(
        generic ? "generic" : "fixed", signatures.back()));
  }
  fuzzer_->signatureStats_.resize(2);
  std::unordered_set<std::string> selected;
  for (auto i = 0; i < 20; ++i) {
    auto signature = fuzzer_->pickSignature().first;
    EXPECT_THAT(
        signature.constantArgs, testing::ElementsAre(false, true, true));
    selected.insert(signature.name);
  }
  EXPECT_THAT(selected, testing::UnorderedElementsAre("fixed", "generic"));
}

TEST_F(AggregationFuzzerBaseTest, constantInputsAcrossBatches) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({7, 9, 11}),
      makeFlatVector<bool>({true, false, true}),
      makeFlatVector<bool>({true, false, false}),
  });
  std::vector<RowVectorPtr> batches{
      input,
      makeRowVector({
          makeFlatVector<int64_t>({4, 5, 6}),
          makeFlatVector<int64_t>({13, 15, 17}),
          makeFlatVector<bool>({false, true, false}),
          makeFlatVector<bool>({false, true, true}),
      }),
  };
  CallableSignature signature{
      .name = "test",
      .args = {BIGINT(), BIGINT(), BOOLEAN(), BOOLEAN()},
      .returnType = BIGINT(),
      .constantArgs = {false, true, false, true},
  };
  const auto constants = fuzzer_->generateConstantInputs(signature, batches);
  EXPECT_THAT(
      constants->type()->asRow().names(), testing::ElementsAre("c1", "c3"));
  auto expected = makeConstant<int64_t>(7, 3);
  for (const auto& batch : batches) {
    velox::test::assertEqualVectors(expected, batch->childAt(1));
    velox::test::assertEqualVectors(
        makeConstant<bool>(true, 3), batch->childAt(3));
  }
  velox::test::assertEqualVectors(
      makeFlatVector<int64_t>({1, 2, 3}), batches.front()->childAt(0));
  EXPECT_EQ(batches.back()->childAt(0)->toString(0), "4");
  EXPECT_FALSE(
      batches.back()->childAt(2)->as<SimpleVector<bool>>()->valueAt(0));
}

TEST_F(AggregationFuzzerBaseTest, typedConstants) {
  std::vector<VectorPtr> values{
      makeFlatVector<bool>({true}),
      makeFlatVector<int64_t>({7}),
      makeNullableFlatVector<int64_t>({std::nullopt}),
      makeFlatVector<double>({0.125}),
      makeFlatVector<StringView>({StringView("a'b")}),
      makeFlatVector<StringView>({StringView("a\0b", 3)}, VARBINARY()),
      makeArrayVector<int64_t>({{1, 2}}),
      makeMapVector<int64_t, int64_t>({{{1, 2}}}),
      makeRowVector({makeFlatVector<int64_t>({3})}),
      makeFlatVector<int64_t>({123}, DECIMAL(10, 2)),
      makeFlatVector<int128_t>({123}, DECIMAL(25, 2)),
  };
  for (const auto& value : values) {
    SCOPED_TRACE(value->type()->toString());
    CallableSignature signature{
        .name = "arbitrary",
        .args = {value->type()},
        .returnType = value->type(),
        .constantArgs = {true},
    };
    std::vector<RowVectorPtr> input{makeRowVector({value})};
    auto constants = fuzzer_->generateConstantInputs(signature, input);
    auto builder = PlanBuilder(pool_.get(), constants);
    auto plan =
        builder.newPlan()
            .values(input)
            .singleAggregation({}, {"arbitrary(__complex_constant(c0))"})
            .planNode();
    auto aggregation =
        std::dynamic_pointer_cast<const core::AggregationNode>(plan);
    auto constant = std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
        aggregation->aggregates()[0].call->inputs()[0]);
    ASSERT_NE(constant, nullptr);
    EXPECT_TRUE(constant->type()->equivalent(*value->type()));
    AssertQueryBuilder(plan).assertResults(makeRowVector({value}));
  }
}

TEST_F(AggregationFuzzerBaseTest, constantForwardingPlans) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({7, 9, 11}),
      makeFlatVector<bool>({true, false, true}),
  });
  CallableSignature signature{
      .name = "constant_forward",
      .args = {BIGINT(), BIGINT()},
      .returnType = BIGINT(),
      .constantArgs = {false, true},
  };
  std::vector<RowVectorPtr> batches{
      input,
      makeRowVector({
          makeFlatVector<int64_t>({4, 5, 6}),
          makeFlatVector<int64_t>({13, 15, 17}),
          makeFlatVector<bool>({false, true, true}),
      }),
  };
  auto constants = fuzzer_->generateConstantInputs(signature, batches);
  auto call = makeFunctionCall(
      signature.name,
      {"c0", "__complex_constant(c1)"},
      true,
      false,
      false,
      {"c0", "c1"});
  auto builder = PlanBuilder(pool_.get(), constants);
  const auto directory = common::testutil::TempDirectoryPath::create();
  const auto writerPool = rootPool_->addAggregateChild("writer");
  const auto splits = makeSplits(batches, directory->getPath(), writerPool);
  const auto expected = makeRowVector({makeFlatVector<int64_t>({43})});
  for (bool tableScan : {false, true}) {
    for (bool partial : {false, true}) {
      SCOPED_TRACE(fmt::format("tableScan={}, partial={}", tableScan, partial));
      auto plan = builder.newPlan();
      if (tableScan) {
        plan.tableScan(asRowType(input->type()));
      } else {
        plan.values(batches);
      }
      if (partial) {
        plan.partialAggregation({}, {call}, {"c2"});
      } else {
        plan.singleAggregation({}, {call}, {"c2"});
      }
      auto aggregation = std::dynamic_pointer_cast<const core::AggregationNode>(
          plan.planNode());
      ASSERT_NE(aggregation, nullptr);
      EXPECT_NE(
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
              aggregation->aggregates()[0].call->inputs()[1]),
          nullptr);
      if (partial) {
        plan.intermediateAggregation().finalAggregation();
      }
      auto query = AssertQueryBuilder(plan.planNode());
      if (tableScan) {
        query.splits(splits);
      }
      query.assertResults(expected);
    }
  }
}

TEST_F(AggregationFuzzerBaseTest, constantArgumentsInFuzzer) {
  gflags::FlagSaver flags;
  FLAGS_steps = 5;
  FLAGS_duration_sec = 0;
  FLAGS_batch_size = 3;
  FLAGS_num_batches = 2;
  FLAGS_null_ratio = 0;
  FLAGS_persist_and_run_once = false;
  FLAGS_enable_sorted_aggregations = true;
  FLAGS_enable_streaming_aggregations = true;
  releaseFuzzer();

  auto verifier = std::make_shared<LiteralArgumentResultVerifier>();
  auto generator = std::make_shared<FixedInputGenerator>(std::vector<VectorPtr>{
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({7, 9, 11}),
  });
  aggregateFuzzer(
      {{"constant_forward",
        getAggregateFunctionSignatures().at("constant_forward")}},
      123,
      {},
      {{"constant_forward", verifier}},
      {{"constant_forward", generator}},
      {},
      VectorFuzzer::Options::TimestampPrecision::kMilliSeconds,
      {},
      {},
      true,
      std::nullopt,
      std::make_unique<DuckQueryRunner>(rootPool_.get()));
  EXPECT_GT(verifier->numInitializations(), 0);
  EXPECT_GT(verifier->numComparisons(), 0);
}

TEST_F(AggregationFuzzerBaseTest, emptyInput) {
  CallableSignature signature{
      .name = "test",
      .args = {BIGINT()},
      .returnType = BIGINT(),
      .constantArgs = {true},
  };
  std::vector<RowVectorPtr> input{makeRowVector({makeFlatVector<int64_t>({})})};
  auto constants = fuzzer_->generateConstantInputs(signature, input);
  ASSERT_NE(constants, nullptr);
  EXPECT_EQ(constants->size(), 1);
  EXPECT_EQ(input.front()->size(), 0);

  signature.constantArgs = {false};
  EXPECT_EQ(fuzzer_->generateConstantInputs(signature, input), nullptr);
}

} // namespace facebook::velox::exec::test
