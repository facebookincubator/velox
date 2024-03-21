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

#include <string>

#include "velox/core/PlanNode.h"
#include "velox/exec/fuzzer/ResultVerifier.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::exec::test {

// MinMaxByResultVerifier works by computing a superset and a subset of the
// expected values for each group and checking that the actual result contains
// the subset and is contained in the superset. For min_by/max_by(x, y) whose
// result is a single value, superset is an array of x values associated with
// the min/max y, and the subset is a nullptr. For min_by/max_by(x, y, n) whose
// result is an array, superset is an array of x values associated with the
// min/max n y values, and subset is an array of x values associated with the
// min/max n-1 y values.
//
// min_by/max_by(x, y) and min_by/max_by(x, y, n) can return NULL in three
// situations:
// (1) all x in the group are masked out,
// (2) all y in the group are NULL,
// (3) for min_by/max_by(x, y), one of x associated with min/max y is NULL.
class MinMaxByResultVerifier : public ResultVerifier {
 public:
  explicit MinMaxByResultVerifier(bool minBy) : minBy_{minBy} {}

  bool supportsCompare() override {
    return false;
  }

  bool supportsVerify() override {
    return true;
  }

  void initialize(
      const std::vector<RowVectorPtr>& input,
      const std::vector<std::string>& groupingKeys,
      const core::AggregationNode::Aggregate& aggregate,
      const std::string& aggregateName) override {
    if (aggregate.call->inputs().size() > 2) {
      minMaxByN_ = true;
    }

    std::stringstream ss;
    toTypeSql(aggregate.call->type(), ss);
    aggregateTypeSql_ = ss.str();

    // We'll first compute array_agg grouped by the original grouping keys plus
    // y. We then apply set_union on these results associated with the min/max n
    // y. We find the min/max n y by using the row_number window function over
    // partitions by the original grouping keys ordered by y.
    auto yColumn = extractYColumnName(aggregate);
    auto groupingKeysAndY = groupingKeys;
    groupingKeysAndY.push_back(yColumn);
    auto frame = makeWindowFrame(groupingKeys, yColumn);

    // projectColumnsBeforeUnion is used to prepare the result of array_agg for
    // union. Specifically, if 'y' is NULL, we turn the array_agg result of this
    // 'y' into NULL too that will be ignored by set_union later. This is for
    // case (2) above where min_by/max_by returns NULL.
    auto projectColumnsBeforeUnion = groupingKeys;
    std::string expectColumn;
    if (minMaxByN_) {
      expectColumn = fmt::format(
          "if ({} is null, cast(NULL as {}), expected) as expected",
          yColumn,
          aggregateTypeSql_);
    } else {
      expectColumn = fmt::format(
          "if ({} is null, cast(NULL as {}[]), expected) as expected",
          yColumn,
          aggregateTypeSql_);
    }
    projectColumnsBeforeUnion.push_back(expectColumn);

    // Add a column of aggregateTypeSql_ of all nulls so that we can union the
    // oracle result with the actual result in the verify method.
    auto finalProjectColumns = groupingKeys;
    finalProjectColumns.push_back(fmt::format(
        "cast(NULL as {}) as {}", aggregateTypeSql_, aggregateName));
    finalProjectColumns.push_back("expected");

    // Suppose the original query is 'SELECT min_by(x, y, 2) filter (where a)
    // from t group by b', we construct a superSet query as follows:
    // SELECT
    //     b,
    //     NULL as a0,
    //     SET_UNION(expected) AS expected
    // FROM (
    //     SELECT
    //         b,
    //         y,
    //         IF (y IS NULL, NULL, expected) AS expected,
    //         row_number
    //     FROM (
    //         SELECT
    //             b,
    //             y,
    //             expected,
    //             ROW_NUMBER() OVER (
    //                 PARTITION BY
    //                     b
    //                 ORDER BY
    //                     y ASC nulls last
    //             ) AS row_number
    //         FROM (
    //             SELECT
    //                 b,
    //                 y,
    //                 $internal$array_agg(x) AS expected
    //             FROM t
    //             WHERE
    //                 a
    //             GROUP BY
    //                 b,
    //                 y
    //         )
    //     )
    //     WHERE
    //         row_number <= 2
    // )
    // GROUP BY
    //     b
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto superSetPlan =
        PlanBuilder(planNodeIdGenerator, input[0]->pool()).values(input);
    // Filter out masked rows first so that groups with all rows filtered out
    // won't take a row_number during the filtering later.
    if (aggregate.mask != nullptr) {
      superSetPlan = superSetPlan.filter(aggregate.mask->name());
    }

    int64_t n = 1;
    if (minMaxByN_) {
      n = extractNValue(aggregate, input);
    }

    superSetPlan =
        superSetPlan
            .singleAggregation(groupingKeysAndY, {makeArrayAggCall(aggregate)})
            .window(
                {fmt::format("row_number() over ({}) as row_number", frame)})
            .filter(fmt::format("row_number <= {}", n))
            .project(projectColumnsBeforeUnion)
            .singleAggregation(
                groupingKeys, {"set_union(expected) as expected"})
            .project(finalProjectColumns);

    superset_ = AssertQueryBuilder(superSetPlan.planNode())
                    .copyResults(input[0]->pool());
    groupingKeys_ = groupingKeys;
    name_ = aggregateName;

    if (n > 1) {
      // Same as the superSet query, except that 'row_number <= 2' becomes
      // 'row_number <= 1'.
      auto subsetPlan =
          PlanBuilder(planNodeIdGenerator, input[0]->pool()).values(input);
      if (aggregate.mask != nullptr) {
        subsetPlan = subsetPlan.filter(aggregate.mask->name());
      }
      subsetPlan = subsetPlan
                       .singleAggregation(
                           groupingKeysAndY, {makeArrayAggCall(aggregate)})
                       .window({fmt::format(
                           "row_number() over ({}) as row_number", frame)})
                       .filter(fmt::format("row_number <= {}", n - 1))
                       .project(projectColumnsBeforeUnion)
                       .singleAggregation(
                           groupingKeys, {"set_union(expected) as expected"})
                       .project(finalProjectColumns);
      subset_ = AssertQueryBuilder(subsetPlan.planNode())
                    .copyResults(input[0]->pool());
    }
  }

  bool compare(
      const RowVectorPtr& /*result*/,
      const RowVectorPtr& /*altResult*/) override {
    VELOX_UNSUPPORTED();
  }

  bool verify(const RowVectorPtr& result) override {
    // Union 'result' with 'superset_', group by on 'groupingKeys_' and produce
    // pairs of actual and expected values per group. We cannot use join because
    // grouping keys may have nulls.
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto supersetSource =
        PlanBuilder(planNodeIdGenerator).values({superset_}).planNode();

    // Append a NULL column for expected so that we can union 'result' with
    // 'superset_' and 'subset_'.
    std::string nullExpected;
    if (minMaxByN_) {
      nullExpected =
          fmt::format("cast(NULL as {}) as expected", aggregateTypeSql_);
    } else {
      nullExpected =
          fmt::format("cast(NULL as {}[]) as expected", aggregateTypeSql_);
    }
    auto actualSource = PlanBuilder(planNodeIdGenerator, result->pool())
                            .values({result})
                            .appendColumns({nullExpected})
                            .planNode();

    core::PlanNodePtr plan;
    // Return an expression that returns true if 'set' contains NULL.
    auto containsNull = [](const std::string& set) {
      return fmt::format(
          "cardinality({0}) > cardinality(remove_nulls({0}))", set);
    };
    // Return an expression that returns true if every element in 'subset' is
    // also contained in 'superset'.
    auto checkContains = [&containsNull](
                             const std::string& subset,
                             const std::string& superset) {
      return fmt::format(
          "reduce({}, 0, (s, x) -> if (s < 0, -1, if (x is null, if ({}, 1, -1), if (\"$internal$contains\"({}, x), 1, -1))), s -> s >= 0)",
          subset,
          containsNull(superset),
          superset);
    };
    if (minMaxByN_) {
      // Construct a query that checks every element in 'actualSource.a0' is
      // contained in 'supersetSource.expected' of the corresponding group:
      // SELECT
      //     switch(
      //         CARDINALITY(a) > 0,
      //         CARDINALITY(e) > 0 AND REDUCE(
      //             a[1],
      //             0,
      //             (s, x) -> IF (
      //                     s < 0,
      //                     -1,
      //                     IF (
      //                         x IS NULL,
      //                         IF (
      //                             CARDINALITY(e[1]) >
      //                             CARDINALITY(remove_nulls(e[1])), 1, -1
      //                         ),
      //                         IF ($internal$contains(e[1], x), 1, -1)
      //                     )
      //                 ),
      //             s -> s >= 0
      //         ),
      //         CARDINALITY(e) = 0 OR CARDINALITY(e[1]) = 0
      //     )
      // FROM (
      //     SELECT
      //         remove_nulls(a) AS a,
      //         remove_nulls(e) AS e
      //     FROM (
      //         SELECT
      //             ARRAY_AGG(a0) AS a,
      //             ARRAY_AGG(expected) AS e
      //         FROM (
      //             SELECT
      //                 *
      //             FROM superset
      //
      //             UNION ALL
      //
      //             SELECT
      //                 *
      //             FROM actual
      //         )
      //         GROUP BY
      //             b
      //     )
      // )
      plan =
          PlanBuilder(planNodeIdGenerator, result->pool())
              .localPartition({}, {supersetSource, actualSource})
              .singleAggregation(
                  groupingKeys_,
                  {fmt::format("array_agg({}) as a", name_),
                   "array_agg(expected) as e"})
              .project({"remove_nulls(a) as a", "remove_nulls(e) as e"})
              // Check that if actual result is not NULL, all elements in a[1]
              // are also in e[1]. If the actual result is NULL, either the
              // expected result is NULL too (when all rows in the group are
              // masked out), or e[1] is empty (when all y in the group are
              // NULL).
              .project({fmt::format(
                  "switch(cardinality(a) > 0, cardinality(e) > 0 and {}, cardinality(e) = 0 or cardinality(e[1]) = 0)",
                  checkContains("a[1]", "e[1]"))})
              .planNode();
    } else {
      // Construct a query that checks 'actualSource.a0' is contained in
      // 'supersetSource.expected'. This is similar to the query above, but has
      // the top-level projection replaced with the last project below.
      plan =
          PlanBuilder(planNodeIdGenerator, result->pool())
              .localPartition({}, {supersetSource, actualSource})
              .singleAggregation(
                  groupingKeys_,
                  {fmt::format("array_agg({}) as a", name_),
                   "array_agg(expected) as e"})
              .project({"remove_nulls(a) as a", "remove_nulls(e) as e"})
              // Check that if actual result is not NULL, a[1]
              // is contained in e[1]. If the actual result is NULL, either the
              // expected result is NULL too (when all rows in the group are
              // masked out), e[1] is empty (when all y in the group are
              // NULL), or e[1] is [NULL] (when there is a NULL x associated
              // with the min/max y).
              .project({fmt::format(
                  "switch(cardinality(a) > 0, cardinality(e) > 0 and \"$internal$contains\"(e[1], a[1]), cardinality(e) = 0 or cardinality(e[1]) = 0 or {})",
                  containsNull("e[1]"))})
              .planNode();
    }

    auto assertTrueResults = [&result](const core::PlanNodePtr& plan) {
      auto contains = AssertQueryBuilder(plan).copyResults(result->pool());
      auto numGroups = result->size();
      VELOX_CHECK_EQ(numGroups, contains->size());

      VectorPtr expected = BaseVector::createConstant(
          BOOLEAN(), true, numGroups, result->pool());
      velox::test::VectorMaker vectorMaker(result->pool());
      auto expectedRow = vectorMaker.rowVector({expected});

      return assertEqualResults({expectedRow}, {contains});
    };
    assertTrueResults(plan);

    if (minMaxByN_) {
      // Construct a query that checks every element in 'subsetSource.expected'
      // is contained in 'actualSource.a0'. This is similar to the query above,
      // but has the top-level projection replaced with the last project below.
      auto subsetSource =
          PlanBuilder(planNodeIdGenerator).values({subset_}).planNode();
      plan =
          PlanBuilder(planNodeIdGenerator, result->pool())
              .localPartition({}, {subsetSource, actualSource})
              .singleAggregation(
                  groupingKeys_,
                  {fmt::format("array_agg({}) as a", name_),
                   "array_agg(expected) as e"})
              .project({"remove_nulls(a) as a", "remove_nulls(e) as e"})
              // The reverse of the check for superset. Check that all elements
              // in subset are contained in the actual result.
              .project({fmt::format(
                  "switch(cardinality(e) > 0 and cardinality(e[1]) > 0, cardinality(a) > 0 and {}, cardinality(a) = 0)",
                  checkContains("e[1]", "a[1]"))})
              .planNode();
      assertTrueResults(plan);
    }
    return true;
  }

  void reset() override {
    superset_.reset();
    subset_.reset();
    groupingKeys_.clear();
    name_.clear();
    aggregateTypeSql_.clear();
    minMaxByN_ = false;
  }

 private:
  std::string makeWindowFrame(
      const std::vector<std::string>& partitionByKeys,
      const std::string& orderByKey) {
    std::stringstream frame;
    if (!partitionByKeys.empty()) {
      frame << "partition by " << folly::join(", ", partitionByKeys);
    }
    frame << " order by " << orderByKey;
    if (minBy_) {
      frame << " asc";
    } else {
      frame << " desc";
    }
    frame << " nulls last";
    return frame.str();
  }

  int64_t extractNValue(
      const core::AggregationNode::Aggregate& aggregate,
      const std::vector<RowVectorPtr>& input) {
    const auto& args = aggregate.call->inputs();
    VELOX_CHECK_EQ(args.size(), 3);

    if (auto constant = core::TypedExprs::asConstant(args[2])) {
      return constant->toConstantVector(input[0]->pool())
          ->as<ConstantVector<int64_t>>()
          ->valueAt(0);
    } else if (auto field = core::TypedExprs::asFieldAccess(args[2])) {
      return input[0]
          ->childAt(field->name())
          ->as<SimpleVector<int64_t>>()
          ->valueAt(0);
    } else {
      VELOX_UNREACHABLE();
    }
  }

  std::string extractYColumnName(
      const core::AggregationNode::Aggregate& aggregate) {
    const auto& args = aggregate.call->inputs();
    VELOX_CHECK_GE(args.size(), 2)

    auto inputField = core::TypedExprs::asFieldAccess(args[1]);
    VELOX_CHECK_NOT_NULL(inputField)

    return inputField->name();
  }

  std::string makeArrayAggCall(
      const core::AggregationNode::Aggregate& aggregate) {
    const auto& args = aggregate.call->inputs();
    VELOX_CHECK_GE(args.size(), 1)

    auto inputField = core::TypedExprs::asFieldAccess(args[0]);
    VELOX_CHECK_NOT_NULL(inputField)

    // Use $internal$array_agg to ensure we don't ignore input nulls since they
    // may affect the result of min_by/max_by.
    std::string arrayAggCall =
        fmt::format("\"$internal$array_agg\"({})", inputField->name());
    arrayAggCall += " as expected";

    return arrayAggCall;
  }

  RowVectorPtr superset_;
  RowVectorPtr subset_{nullptr};
  std::vector<std::string> groupingKeys_;
  std::string name_;
  std::string aggregateTypeSql_;
  bool minBy_;
  bool minMaxByN_{false};
};

} // namespace facebook::velox::exec::test
