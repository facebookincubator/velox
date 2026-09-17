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
#include "velox/core/FixedPointPlanNodes.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace ::facebook::velox;
using namespace ::facebook::velox::core;

namespace {
class FixedPointPlanNodesTest : public testing::Test,
                                public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  // Nodes are built by hand rather than through PlanBuilder because most of
  // these plans are deliberately malformed -- a fragment with two incoming
  // shuffles, schemas that disagree across one, stages of differing width.
  // PlanBuilder composes a well-formed plan from a source outwards and has no
  // way to express those.
  const RowTypePtr schema_{ROW("x", BIGINT())};

  StateDeclarationPtr declaration(const std::string& name = "n") const {
    return std::make_shared<VectorStateDeclaration>(
        name, schema_, /*initialPlan=*/nullptr, /*append=*/true);
  }

  PlanNodePtr stateSource(
      const std::string& id = "b",
      const std::string& entry = "n") const {
    return std::make_shared<StateSourceNode>(
        id, entry, schema_, /*delta=*/true);
  }

  PlanNodePtr exchange(const std::string& id, const RowTypePtr& type) const {
    return std::make_shared<ExchangeNode>(id, type, "Presto");
  }

  PlanNodePtr exchange(const std::string& id) const {
    return exchange(id, schema_);
  }

  PlanNodePtr shuffleOut(
      const std::string& id,
      const PlanNodePtr& source,
      int32_t numPartitions,
      const RowTypePtr& type) const {
    return std::make_shared<PartitionedOutputNode>(
        id,
        PartitionedOutputNode::Kind::kPartitioned,
        std::vector<TypedExprPtr>{
            std::make_shared<FieldAccessTypedExpr>(BIGINT(), "x")},
        numPartitions,
        /*replicateNullsAndAny=*/false,
        std::make_shared<GatherPartitionFunctionSpec>(),
        type,
        "Presto",
        std::string(TransportKind::kInMemory),
        source);
  }

  PlanNodePtr shuffleOut(
      const std::string& id,
      const PlanNodePtr& source,
      int32_t numPartitions) const {
    return shuffleOut(id, source, numPartitions, schema_);
  }

  static ConvergenceConfig noConvergence() {
    return ConvergenceConfig{
        .maxIterations = 5, .errorWhenMaxIterationReached = false};
  }

  // A fixed point over a single "n" state entry.
  FixedPointNodePtr fixedPoint(
      std::vector<PlanNodePtr> plans,
      ConvergenceConfig convergence) const {
    return std::make_shared<FixedPointNode>(
        "fp",
        std::vector<StateDeclarationPtr>{declaration()},
        std::move(plans),
        std::move(convergence),
        "n");
  }

  // A fixed point over its own "m" entry whose body shuffles 'numPartitions'
  // ways -- what an enclosing loop must notice.
  PlanNodePtr shufflingFixedPoint(const std::string& id, int32_t numPartitions)
      const {
    return std::make_shared<FixedPointNode>(
        id,
        std::vector<StateDeclarationPtr>{declaration("m")},
        std::vector<PlanNodePtr>{
            shuffleOut(id + "p", stateSource(id + "s", "m"), numPartitions),
            exchange(id + "e")},
        noConvergence(),
        "m");
  }

  // Puts 'nested' beside the primary input so the plan still starts with a
  // StateSourceNode; only the recursive walk finds it.
  PlanNodePtr beside(
      const std::string& id,
      const PlanNodePtr& primary,
      const PlanNodePtr& nested) const {
    return std::make_shared<LocalPartitionNode>(
        id,
        LocalPartitionNode::Type::kGather,
        /*scaleWriter=*/false,
        std::make_shared<GatherPartitionFunctionSpec>(),
        std::vector<PlanNodePtr>{primary, nested});
  }
};

// The FixedPointNode constructor and the state declarations validate their
// inputs up front, so a malformed plan fails at construction rather than at
// execution.
TEST_F(FixedPointPlanNodesTest, validation) {
  auto vecSchema = ROW("x", BIGINT());
  auto htSchema = ROW({"k", "v"}, BIGINT());

  // A single body that reads the output entry -- the minimal valid body, reused
  // as the valid baseline that each case mutates one field of.
  auto body =
      std::make_shared<StateSourceNode>("b", "n", vecSchema, /*delta=*/true);
  auto vectorN = [&] {
    return std::make_shared<VectorStateDeclaration>(
        "n", vecSchema, /*initialPlan=*/nullptr, /*append=*/true);
  };

  // maxIterations must be positive.
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{vectorN()},
          std::vector<PlanNodePtr>{body},
          ConvergenceConfig{
              .maxIterations = 0, .errorWhenMaxIterationReached = false},
          "n"),
      "maxIterations must be positive");

  // A hash table needs at least one key column, and every key must be in the
  // schema -- checked when the declaration is built.
  VELOX_ASSERT_USER_THROW(
      std::make_shared<HashTableStateDeclaration>(
          "h", htSchema, std::vector<std::string>{}),
      "at least one key column");
  VELOX_ASSERT_USER_THROW(
      std::make_shared<HashTableStateDeclaration>(
          "h", htSchema, std::vector<std::string>{"missing"}),
      "key column is not in the schema");
  VELOX_ASSERT_USER_THROW(
      std::make_shared<HashTableStateDeclaration>(
          "h", htSchema, std::vector<std::string>{"k", "k"}),
      "key columns must be unique");

  // StateHashJoin needs a non-null probe source and at least one probe key.
  VELOX_ASSERT_USER_THROW(
      std::make_shared<StateHashJoinNode>(
          "j", "h", std::vector<std::string>{"k"}, htSchema, nullptr),
      "non-null probe source");
  VELOX_ASSERT_USER_THROW(
      std::make_shared<StateHashJoinNode>(
          "j", "h", std::vector<std::string>{}, htSchema, body),
      "at least one probe key");

  // State declaration names must be unique across all kinds.
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{
              vectorN(),
              std::make_shared<VectorStateDeclaration>("n", vecSchema)},
          std::vector<PlanNodePtr>{body},
          ConvergenceConfig{
              .maxIterations = 5, .errorWhenMaxIterationReached = false},
          "n"),
      "duplicate state declaration name");

  // A StateSource must reference a declared vector entry.
  auto typoBody =
      std::make_shared<StateSourceNode>("b", "typo", vecSchema, /*delta=*/true);
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{vectorN()},
          std::vector<PlanNodePtr>{typoBody},
          ConvergenceConfig{
              .maxIterations = 5, .errorWhenMaxIterationReached = false},
          "n"),
      "StateSource references no declared vector state entry");

  // errorWhenMaxIterationReached requires a convergence plan (a null plan never
  // converges, so it would always fail).
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{vectorN()},
          std::vector<PlanNodePtr>{body},
          ConvergenceConfig{
              .maxIterations = 5, .errorWhenMaxIterationReached = true},
          "n"),
      "errorWhenMaxIterationReached requires a convergence criterion");

  // A convergence plan must emit exactly one BOOLEAN column.
  auto nonBoolConvergence =
      std::make_shared<StateSourceNode>("c", "n", vecSchema, /*delta=*/false);
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{vectorN()},
          std::vector<PlanNodePtr>{body},
          ConvergenceConfig{.plans = {nonBoolConvergence}, .maxIterations = 5},
          "n"),
      "convergence plan output column must be BOOLEAN");

  auto twoColSchema = ROW({"a", "b"}, BOOLEAN());
  auto twoColConvergence = std::make_shared<StateSourceNode>(
      "c", "flags", twoColSchema, /*delta=*/false);
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{
              vectorN(),
              std::make_shared<VectorStateDeclaration>("flags", twoColSchema)},
          std::vector<PlanNodePtr>{body},
          ConvergenceConfig{.plans = {twoColConvergence}, .maxIterations = 5},
          "n"),
      "exactly one output column");

  // A StateHashJoin output arity must equal probe columns plus the hash table's
  // payload columns.
  auto badArityJoin = std::make_shared<StateHashJoinNode>(
      "j",
      "h",
      std::vector<std::string>{"x"},
      vecSchema,
      std::make_shared<StateSourceNode>("b", "n", vecSchema, /*delta=*/true));
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{
              vectorN(),
              std::make_shared<HashTableStateDeclaration>(
                  "h", htSchema, std::vector<std::string>{"k"})},
          std::vector<PlanNodePtr>{badArityJoin},
          ConvergenceConfig{
              .maxIterations = 5, .errorWhenMaxIterationReached = false},
          "n"),
      "output arity must equal probe columns plus hash table payload columns");

  // A StateHashJoin's leading probe key column types must match the hash
  // table's build key types (keys-first on both sides).
  auto varcharProbe = ROW("k", VARCHAR());
  auto badKeyTypeJoin = std::make_shared<StateHashJoinNode>(
      "j",
      "h",
      std::vector<std::string>{"k"},
      ROW({"k", "v"}, {VARCHAR(), BIGINT()}),
      std::make_shared<StateSourceNode>(
          "s", "probe", varcharProbe, /*delta=*/true));
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{
              std::make_shared<VectorStateDeclaration>("probe", varcharProbe),
              std::make_shared<HashTableStateDeclaration>(
                  "h", htSchema, std::vector<std::string>{"k"})},
          std::vector<PlanNodePtr>{badKeyTypeJoin},
          ConvergenceConfig{
              .maxIterations = 5, .errorWhenMaxIterationReached = false},
          "probe"),
      "probe key column type at channel 0 must match the hash table build key");

  // A null body plan is rejected with a clean error rather than crashing.
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{vectorN()},
          std::vector<PlanNodePtr>{nullptr},
          ConvergenceConfig{
              .maxIterations = 5, .errorWhenMaxIterationReached = false},
          "n"),
      "plan 0 must not be null");

  // Hash table key columns must be the leading schema columns (keys-first).
  VELOX_ASSERT_USER_THROW(
      std::make_shared<HashTableStateDeclaration>(
          "h", ROW({"v", "k"}, BIGINT()), std::vector<std::string>{"k"}),
      "leading schema columns in order");

  // An initial plan must not read state -- it runs in Phase 1 before state
  // exists.
  auto stateReadingInitial =
      std::make_shared<StateSourceNode>("s", "n", vecSchema, /*delta=*/true);
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{
              std::make_shared<VectorStateDeclaration>(
                  "n", vecSchema, stateReadingInitial, /*append=*/true)},
          std::vector<PlanNodePtr>{body},
          ConvergenceConfig{
              .maxIterations = 5, .errorWhenMaxIterationReached = false},
          "n"),
      "initial plan must not read state");
}

// A convergence or body chain is one logical plan cut at its shuffle
// boundaries.  These are the ways a caller can hand over something that is not
// that, each of which would otherwise wire exchanges to peers that do not
// match.
TEST_F(FixedPointPlanNodesTest, chainValidation) {
  auto schema = schema_;
  auto otherSchema = ROW({"x", "y"}, BIGINT());

  // A two-plan chain shuffling across two workers is the valid baseline each
  // case below mutates, and it is what numWorkers()/requiresSplits() report.
  {
    auto node = fixedPoint(
        {shuffleOut("p", stateSource(), 2, schema), exchange("e", schema)},
        noConvergence());
    EXPECT_EQ(node->numWorkers(), 2);
    EXPECT_TRUE(node->requiresSplits());
  }

  // A non-shuffling body is one worker and needs no peer splits.
  {
    auto node = fixedPoint({stateSource()}, noConvergence());
    EXPECT_EQ(node->numWorkers(), 1);
    EXPECT_FALSE(node->requiresSplits());
  }

  // A fragment reading a second shuffle is a branching topology, not one link
  // of a linear chain.  primaryLeaf() alone would not see the second branch.
  {
    auto branching = std::make_shared<LocalPartitionNode>(
        "lp",
        LocalPartitionNode::Type::kGather,
        /*scaleWriter=*/false,
        std::make_shared<GatherPartitionFunctionSpec>(),
        std::vector<PlanNodePtr>{
            exchange("e0", schema), exchange("e1", schema)});
    VELOX_ASSERT_USER_THROW(
        fixedPoint(
            {shuffleOut("p", stateSource(), 2, schema), branching},
            noConvergence()),
        "must read exactly one shuffle");
  }

  // What one fragment shuffles out is what the next reads back, so the schemas
  // must match.
  VELOX_ASSERT_USER_THROW(
      fixedPoint(
          {shuffleOut("p", stateSource(), 2, schema),
           exchange("e", otherSchema)},
          noConvergence()),
      "must match what the next one reads back");

  // Every shuffling stage of a chain crosses the same number of workers.
  VELOX_ASSERT_USER_THROW(
      fixedPoint(
          {shuffleOut("p0", stateSource(), 2, schema),
           shuffleOut("p1", exchange("e0", schema), 3, schema),
           exchange("e1", schema)},
          noConvergence()),
      "must partition across the same number of workers");

  // A convergence chain wider than the body would wait on peers the
  // coordinator never assigned it.
  {
    auto convergenceSchema = ROW("c", BOOLEAN());
    ConvergenceConfig convergence{
        .plans =
            {shuffleOut(
                 "cp",
                 std::make_shared<StateSourceNode>(
                     "cs", "n", schema, /*delta=*/false),
                 3,
                 schema),
             exchange("ce", schema)},
        .maxIterations = 5};
    VELOX_ASSERT_USER_THROW(
        fixedPoint(
            {shuffleOut("p", stateSource(), 2, schema), exchange("e", schema)},
            std::move(convergence)),
        "must shuffle across the same number of workers");
  }
}

// stopWhenDeltaEmpty reads a row count local to one worker, so it is only a
// sound verdict when nothing in the loop shuffles -- and it cannot be combined
// with a convergence sequence, which would be a second, disagreeing verdict.
TEST_F(FixedPointPlanNodesTest, deltaEmptyValidation) {
  auto schema = schema_;
  auto shuffleOut = [&](const std::string& id, const PlanNodePtr& source) {
    return this->shuffleOut(id, source, /*numPartitions=*/2);
  };

  // A purely local loop is the case it is for.
  EXPECT_NO_THROW(
      fixedPoint({stateSource("b")}, ConvergenceConfig::whenDeltaEmpty(10)));

  // The row count is already the verdict; a convergence sequence would be a
  // second one.
  auto withPlans = ConvergenceConfig::whenDeltaEmpty(10);
  withPlans.plans = {stateSource("c")};
  VELOX_ASSERT_USER_THROW(
      fixedPoint({stateSource("b")}, std::move(withPlans)),
      "mutually exclusive");

  // A shuffling body means peers, and one worker's frontier can empty while
  // theirs has not.
  VELOX_ASSERT_USER_THROW(
      fixedPoint(
          {shuffleOut("p", stateSource("b")),
           std::make_shared<ExchangeNode>("e", schema, "Presto")},
          ConvergenceConfig::whenDeltaEmpty(10)),
      "non-shuffling fixed point");

  // numWorkers() sees through to the nested loop's width, so a shuffling
  // nested loop is rejected by the same check as a shuffling body.
  auto nested = std::make_shared<FixedPointNode>(
      "inner",
      std::vector<StateDeclarationPtr>{std::make_shared<VectorStateDeclaration>(
          "m", schema, /*initialPlan=*/nullptr, /*append=*/true)},
      std::vector<PlanNodePtr>{
          shuffleOut(
              "ip",
              std::make_shared<StateSourceNode>(
                  "is", "m", schema, /*delta=*/true)),
          std::make_shared<ExchangeNode>("ie", schema, "Presto")},
      ConvergenceConfig{
          .maxIterations = 5, .errorWhenMaxIterationReached = false},
      "m");
  auto beside = std::make_shared<LocalPartitionNode>(
      "g",
      LocalPartitionNode::Type::kGather,
      /*scaleWriter=*/false,
      std::make_shared<GatherPartitionFunctionSpec>(),
      std::vector<PlanNodePtr>{stateSource("b"), nested});
  VELOX_ASSERT_USER_THROW(
      fixedPoint({beside}, ConvergenceConfig::whenDeltaEmpty(10)),
      "non-shuffling fixed point");
}

// A nested fixed point runs on the peers the enclosing loop was assigned, so
// the enclosing loop has to know it shuffles: it must report the same width and
// ask the coordinator for splits, or the nested exchanges wait on peers nobody
// named.
TEST_F(FixedPointPlanNodesTest, nestedWorkerPropagation) {
  auto schema = schema_;

  // The enclosing loop does not shuffle itself, but adopts the nested width and
  // the nested split requirement.
  {
    auto outer = std::make_shared<FixedPointNode>(
        "fp",
        std::vector<StateDeclarationPtr>{declaration("n")},
        std::vector<PlanNodePtr>{
            beside("g", stateSource("b", "n"), shufflingFixedPoint("in", 2))},
        noConvergence(),
        "n");
    EXPECT_EQ(outer->numWorkers(), 2);
    EXPECT_TRUE(outer->requiresSplits());
  }

  // A nested loop wider than the enclosing one is rejected rather than
  // silently widening it.
  VELOX_ASSERT_USER_THROW(
      std::make_shared<FixedPointNode>(
          "fp",
          std::vector<StateDeclarationPtr>{declaration("n")},
          std::vector<PlanNodePtr>{
              shuffleOut("p", stateSource("b", "n"), 2),
              beside(
                  "g",
                  std::make_shared<ExchangeNode>("e", schema, "Presto"),
                  shufflingFixedPoint("in", 3))},
          noConvergence(),
          "n"),
      "must shuffle across the same number of workers");
}

// A nested loop in the convergence chain widens the enclosing loop exactly as
// one in the body does -- the convergence sequence runs on the same peers.
// Only the body path was covered before.
TEST_F(FixedPointPlanNodesTest, nestedInConvergence) {
  auto verdictSchema = ROW("converged", BOOLEAN());
  auto verdictSource = [&](const std::string& id, const std::string& entry) {
    return std::make_shared<StateSourceNode>(
        id, entry, verdictSchema, /*delta=*/false);
  };
  // A loop over its own BOOLEAN entry, shuffling 'numPartitions' ways, so it
  // can sit in a convergence chain whose last plan must emit one BOOLEAN.
  auto shufflingVerdictLoop = [&](const std::string& id,
                                  int32_t numPartitions) {
    auto out = std::make_shared<PartitionedOutputNode>(
        id + "p",
        PartitionedOutputNode::Kind::kPartitioned,
        std::vector<TypedExprPtr>{
            std::make_shared<FieldAccessTypedExpr>(BOOLEAN(), "converged")},
        numPartitions,
        /*replicateNullsAndAny=*/false,
        std::make_shared<GatherPartitionFunctionSpec>(),
        verdictSchema,
        "Presto",
        std::string(TransportKind::kInMemory),
        verdictSource(id + "s", "v"));
    return std::make_shared<FixedPointNode>(
        id,
        std::vector<StateDeclarationPtr>{
            std::make_shared<VectorStateDeclaration>(
                "v", verdictSchema, /*initialPlan=*/nullptr, /*append=*/true)},
        std::vector<PlanNodePtr>{
            out,
            std::make_shared<ExchangeNode>(id + "e", verdictSchema, "Presto")},
        noConvergence(),
        "v");
  };
  auto convergenceHosting = [&](const PlanNodePtr& nested) {
    return std::vector<PlanNodePtr>{
        beside("cg", verdictSource("cs", "w"), nested)};
  };
  auto withVerdictState = [&](std::vector<PlanNodePtr> plans,
                              ConvergenceConfig convergence) {
    return std::make_shared<FixedPointNode>(
        "fp",
        std::vector<StateDeclarationPtr>{
            declaration(),
            std::make_shared<VectorStateDeclaration>(
                "w", verdictSchema, /*initialPlan=*/nullptr, /*append=*/true)},
        std::move(plans),
        std::move(convergence),
        "n");
  };

  // The enclosing loop adopts the nested width from the convergence chain.
  {
    auto convergence = ConvergenceConfig::converging(
        convergenceHosting(shufflingVerdictLoop("cin", 2)), 10);
    auto outer = withVerdictState({stateSource()}, std::move(convergence));
    EXPECT_EQ(outer->numWorkers(), 2);
    EXPECT_TRUE(outer->requiresSplits());
  }

  // And a nested width that disagrees with the body is rejected, naming the
  // convergence plan that hosts it.
  {
    auto convergence = ConvergenceConfig::converging(
        convergenceHosting(shufflingVerdictLoop("cin", 3)), 10);
    VELOX_ASSERT_USER_THROW(
        withVerdictState(
            {shuffleOut("p", stateSource(), 2), exchange("e")},
            std::move(convergence)),
        "nested in convergence plan 0");
  }
}

} // namespace
