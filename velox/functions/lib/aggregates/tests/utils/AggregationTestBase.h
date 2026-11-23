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
#pragma once

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {
class AssertQueryBuilder;
}

namespace facebook::velox::functions::aggregate::test {

class AggregationTestBase : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;

  std::vector<RowVectorPtr>
  makeVectors(const RowTypePtr& rowType, vector_size_t size, int numVectors);

  /// Ensure the function is working in streaming use case.  Create a first
  /// aggregation function, add the rawInput1, then extract the accumulator,
  /// read the accumulator to second function, and continue to add rawInput2 to
  /// second function.
  VectorPtr testStreaming(
      const std::string& functionName,
      bool testGlobal,
      const std::vector<VectorPtr>& rawInput1,
      const std::vector<VectorPtr>& rawInput2,
      const std::unordered_map<std::string, std::string>& config = {});

  /// Which companion plans AggregationTester::withCompanion() runs.
  enum class CompanionPlans {
    /// No companion plans.
    kNone,
    /// The plans that end with the '_extract', '_merge_extract' and '_merge'
    /// companions.
    kAll,
    /// Only the plans that end with the '_merge' companion, which compare
    /// partial states. For functions whose '_extract' and '_merge_extract'
    /// companions cannot resolve their result type from the intermediate type
    /// alone, such as Spark's decimal avg.
    kMergeOnly,
  };

  /// Runs an aggregation through the plans that AggregationTestBase generates
  /// and checks that they all produce the same results. Set the input, the
  /// aggregation and the options, then call one of the assert methods:
  ///
  ///   AggregationTester(*this)
  ///       .inputs(vectors)
  ///       .groupBy({"c0"})
  ///       .aggregates({"sum(c1)"})
  ///       .withCompanion()
  ///       .assertResults("SELECT c0, sum(c1) FROM tmp GROUP BY 1");
  ///
  /// The plans combine partial, final, single and intermediate aggregations,
  /// with and without local exchanges and spilling. The checks also add raw
  /// input after intermediate results (streaming), extract results twice from
  /// the same accumulators (incremental), and read the input from files.
  class AggregationTester {
   public:
    using AssertResults = std::function<std::shared_ptr<exec::Task>(
        exec::test::AssertQueryBuilder&)>;

    explicit AggregationTester(AggregationTestBase& test) : test_{test} {}

    /// The input batches of the aggregation.
    AggregationTester& inputs(std::vector<RowVectorPtr> inputs) {
      inputs_ = std::move(inputs);
      return *this;
    }

    /// The input of the aggregation, as a single batch.
    AggregationTester& inputs(RowVectorPtr input) {
      inputs_ = {std::move(input)};
      return *this;
    }

    /// Adds plan nodes between the input and the aggregation, such as a
    /// projection that computes the grouping keys.
    AggregationTester& preAggregation(
        std::function<void(exec::test::PlanBuilder&)> preAggregation) {
      preAggregation_ = std::move(preAggregation);
      return *this;
    }

    AggregationTester& groupBy(std::vector<std::string> groupingKeys) {
      groupingKeys_ = std::move(groupingKeys);
      return *this;
    }

    AggregationTester& aggregates(std::vector<std::string> aggregates) {
      aggregates_ = std::move(aggregates);
      return *this;
    }

    /// Projections to apply to the result of the aggregation.
    AggregationTester& project(std::vector<std::string> projections) {
      projections_ = std::move(projections);
      return *this;
    }

    /// Sets a query configuration property for every plan.
    AggregationTester& config(
        const std::string& key,
        const std::string& value) {
      config_[key] = value;
      return *this;
    }

    /// Also runs the plans that compute the aggregates with their companion
    /// functions: '_partial', '_merge', '_merge_extract' and '_extract'. These
    /// plans group the partial step by an extra key 'k0', so that the merge
    /// step has several values to merge for each group. 'plans' selects which
    /// of them run; each call overrides the previous one.
    AggregationTester& withCompanion(
        CompanionPlans plans = CompanionPlans::kAll) {
      companionPlans_ = plans;
      return *this;
    }

    /// Skips the check that adds raw input to accumulators after intermediate
    /// results.
    AggregationTester& disableStreaming() {
      streaming_ = false;
      return *this;
    }

    /// Skips the check that extracting results twice from the same accumulators
    /// gives the same results.
    AggregationTester& disableIncremental() {
      incremental_ = false;
      return *this;
    }

    /// Checks the results against an equivalent DuckDB query over a table 'tmp'
    /// that the test created from the input.
    void assertResults(const std::string& duckDbSql);

    /// Checks the results against 'expected'.
    void assertResults(const std::vector<RowVectorPtr>& expected);

    /// Checks the results against 'expected', a single batch.
    void assertResults(const RowVectorPtr& expected);

    /// Checks the results with 'assertResults', which runs the query.
    void assertResults(const AssertResults& assertResults);

    /// Checks that every plan fails with an error that contains 'message'.
    void assertError(const std::string& message);

   private:
    // Runs the plans and checks their results with 'assertResults'.
    void run(const AssertResults& assertResults);

    // Adds the input and the pre-aggregation nodes to 'builder'.
    void addSource(exec::test::PlanBuilder& builder) const;

    // Returns the argument types of each aggregate, as the aggregation over the
    // input resolves them.
    std::vector<std::vector<TypePtr>> aggregateArgTypes() const;

    AggregationTestBase& test_;
    std::vector<RowVectorPtr> inputs_;
    std::function<void(exec::test::PlanBuilder&)> preAggregation_;
    std::vector<std::string> groupingKeys_;
    std::vector<std::string> aggregates_;
    std::vector<std::string> projections_;
    std::unordered_map<std::string, std::string> config_;
    CompanionPlans companionPlans_{CompanionPlans::kNone};
    bool streaming_{true};
    bool incremental_{true};
  };

 private:
  // Test streaming use case where raw inputs are added after intermediate
  // results. Return the result of aggregates if successful.
  //
  // When 'projections' is not empty, compares the results after applying
  // 'projections' to them. The companion plans use it to compare intermediate
  // results after the extract companion, since equal intermediate results may
  // differ in form, as KLL sketches do.
  RowVectorPtr validateStreamingInTestAggregations(
      const std::function<void(exec::test::PlanBuilder&)>& makeSource,
      const std::vector<std::string>& aggregates,
      const std::unordered_map<std::string, std::string>& config,
      const std::vector<std::string>& projections = {});

  VectorPtr testStreaming(
      const std::string& functionName,
      bool testGlobal,
      const std::vector<VectorPtr>& rawInput1,
      vector_size_t rawInput1Size,
      const std::vector<VectorPtr>& rawInput2,
      vector_size_t rawInput2Size,
      const std::unordered_map<std::string, std::string>& config = {});

  // Test to ensure that when extractValues() or extractAccumulators() is called
  // twice on the same accumulator, the extracted results are the same. This
  // ensures that extractValues() and extractAccumulators() are free of side
  // effects.
  void testIncrementalAggregation(
      const std::function<void(exec::test::PlanBuilder&)>& makeSource,
      const std::vector<std::string>& aggregates,
      const std::unordered_map<std::string, std::string>& config = {});

  // Runs the plans and checks of AggregationTester over the input that
  // 'makeSource' builds. 'streaming' and 'incremental' enable the checks of
  // the same names.
  void testAggregations(
      std::function<void(exec::test::PlanBuilder&)> makeSource,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::vector<std::string>& postAggregationProjections,
      std::function<std::shared_ptr<exec::Task>(
          exec::test::AssertQueryBuilder&)> assertResults,
      const std::unordered_map<std::string, std::string>& config,
      bool streaming,
      bool incremental);

  // Writes the input to files and runs the plans over a scan of them. Can
  // reveal bugs in string life cycle management.
  void testReadFromFiles(
      std::function<void(exec::test::PlanBuilder&)> makeSource,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::vector<std::string>& postAggregationProjections,
      std::function<std::shared_ptr<exec::Task>(
          exec::test::AssertQueryBuilder&)> assertResults,
      const std::unordered_map<std::string, std::string>& config);

  // Runs the plans and checks that each fails with an error that contains
  // 'expectedMessage'.
  void testFailingAggregations(
      const std::vector<RowVectorPtr>& data,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::string& expectedMessage,
      const std::unordered_map<std::string, std::string>& config,
      bool streaming);

  // Input and companion aggregates that the companion plans are built from.
  // Defined in the .cpp.
  struct CompanionPlanInputs;

  // Runs the companion plans that end with the extract companion, including
  // the streaming check, and checks their results with 'assertResults'.
  void testCompanionExtract(
      const CompanionPlanInputs& inputs,
      const std::vector<std::string>& postAggregationProjections,
      const std::function<std::shared_ptr<exec::Task>(
          exec::test::AssertQueryBuilder&)>& assertResults,
      const std::unordered_map<std::string, std::string>& config,
      bool streaming);

  // Runs the companion plans that end with the merge_extract companion and
  // checks their results with 'assertResults'.
  void testCompanionMergeExtract(
      const CompanionPlanInputs& inputs,
      const std::vector<std::string>& postAggregationProjections,
      const std::function<std::shared_ptr<exec::Task>(
          exec::test::AssertQueryBuilder&)>& assertResults,
      const std::unordered_map<std::string, std::string>& config);

  // Runs the companion plans that end with the merge companion: partial merges
  // with and without abandoning partial aggregation, which compare partial
  // states.
  void testCompanionMerge(
      const CompanionPlanInputs& inputs,
      const std::unordered_map<std::string, std::string>& config);

  void testAggregationsImpl(
      std::function<void(exec::test::PlanBuilder&)> makeSource,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::vector<std::string>& postAggregationProjections,
      std::function<std::shared_ptr<exec::Task>(
          exec::test::AssertQueryBuilder&)> assertResults,
      const std::unordered_map<std::string, std::string>& config,
      bool streaming);

  void testStreamingAggregationsImpl(
      std::function<void(exec::test::PlanBuilder&)> makeSource,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::vector<std::string>& postAggregationProjections,
      std::function<std::shared_ptr<exec::Task>(
          exec::test::AssertQueryBuilder&)> assertResults,
      const std::unordered_map<std::string, std::string>& config);
};

/// Runs 'name' over a single group of ('values', 'comparisons') in a row of a
/// RowContainer, and returns the variable-length size the aggregate tracked for
/// that row. Zero means the aggregate never moved the row size counter, which
/// for an accumulator holding out-of-line bytes is the bug this measures.
uint32_t aggregateAndReadRowSize(
    memory::MemoryPool* pool,
    const std::string& name,
    const TypePtr& valueType,
    const VectorPtr& values,
    const VectorPtr& comparisons);

} // namespace facebook::velox::functions::aggregate::test
