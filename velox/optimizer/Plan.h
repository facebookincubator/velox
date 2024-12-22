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

#include "velox/connectors/Connector.h"
#include "velox/core/PlanNode.h"
#include "velox/optimizer/Cost.h"
#include "velox/optimizer/RelationOp.h"
#include "velox/runner/MultiFragmentPlan.h"

/// Planning-time data structures. Represent the state of the planning process
/// plus utilities.
namespace facebook::velox::optimizer {

struct ITypedExprHasher {
  size_t operator()(const velox::core::ITypedExpr* expr) const {
    return expr->hash();
  }
};

struct ITypedExprComparer {
  bool operator()(
      const velox::core::ITypedExpr* lhs,
      const velox::core::ITypedExpr* rhs) const {
    return *lhs == *rhs;
  }
};

// Map for deduplicating ITypedExpr trees.
using ExprDedupMap = folly::F14FastMap<
    const velox::core::ITypedExpr*,
    ExprCP,
    ITypedExprHasher,
    ITypedExprComparer>;

struct Plan;
struct PlanState;

using PlanPtr = Plan*;

/// A set of build sides. a candidate plan tracks all builds so that they can be
/// reused
using BuildSet = std::vector<HashBuildPtr>;

/// Item produced by optimization and kept in memo. Corresponds to
/// pre-costed physical plan with costs and data properties.
struct Plan {
  Plan(RelationOpPtr op, const PlanState& state);

  /// True if 'state' has a lower cost than 'this'.
  bool isStateBetter(const PlanState& state) const;

  // Root of the plan tree.
  RelationOpPtr op;

  // Total cost of 'op'. Setup costs and memory sizes are added up. The unit
  // cost is the sum of the unit costs of the left-deep branch of 'op', where
  // each unit cost is multiplied by the product of the fanouts of its inputs.
  Cost cost;

  // The tables from original join graph that are included in this
  // plan. If this is a derived table in the original plan, the
  // covered object is the derived table, not its constituent
  // tables.
  PlanObjectSet tables;

  // The produced columns. Includes input columns.
  PlanObjectSet columns;

  // Columns that are fixed on input. Applies to index path for a derived
  // table, e.g. a left (t1 left t2) dt on dt.t1pk = a.fk. In a memo of dt
  // inputs is dt.pkt1.
  PlanObjectSet input;

  // hash join builds placed in the plan. Allows reusing a build.
  BuildSet builds;

  // the tables/derived tables that are contained in this plan and need not be
  // addressed by enclosing plans. This is all the tables in a build side join
  // but not necessarily all tables that were added to a group by derived table.
  PlanObjectSet fullyImported;

  std::string printCost() const;
  std::string toString(bool detail) const;
};

/// The set of plans produced for a set of tables and columns. The plans may
/// have different output orders and  distributions.
struct PlanSet {
  // Interesting equivalent plans.
  std::vector<std::unique_ptr<Plan>> plans;

  // plan with lowest cost + setupCost. Member of 'plans'.
  PlanPtr bestPlan{nullptr};

  // Cost of 'bestPlan' plus shuffle. If a cutoff is applicable, nothing more
  // expensive than this should be tried.
  float bestCostWithShuffle{0};

  // Returns the best plan that produces 'distribution'. If the best plan has
  // some other distribution, sets 'needsShuffle ' to true.
  PlanPtr best(const Distribution& distribution, bool& needShuffle);

  /// Compares 'plan' to already seen plans and retains it if it is
  /// interesting, e.g. better than the best so far or has an interesting
  /// order. Returns the plan if retained, nullptr if not.
  PlanPtr addPlan(RelationOpPtr plan, PlanState& state);
};

// Represents the next table/derived table to join. May consist of several
// tables for a bushy build side.
struct JoinCandidate {
  JoinCandidate() = default;

  JoinCandidate(JoinEdgeP _join, PlanObjectCP _right, float _fanout)
      : join(_join), tables({_right}), fanout(_fanout) {}

  // Returns the join side info for 'table'. If 'other' is set, returns the
  // other side.
  JoinSide sideOf(PlanObjectCP side, bool other = false) const;

  std::string toString() const;

  // The join between already placed tables and the table(s) in 'this'.
  JoinEdgeP join{nullptr};

  // Tables to join on the build side. The tables must not be already placed in
  // the plan. side, i.e. be alread
  std::vector<PlanObjectCP> tables;

  // Joins imported from the left side for reducing a build
  // size. These could be ignored without affecting the result but can
  // be included to restrict the size of build, e.g. lineitem join
  // part left (partsupp exists part) would have the second part in
  // 'existences' and partsupp in 'tables' because we know that
  // partsupp will not be probed with keys that are not in part, so
  // there is no point building with these. This may involve tables already
  // placed in the plan.
  std::vector<PlanObjectSet> existences;

  // Number of right side hits for one row on the left. The join
  // selectivity in 'tables' affects this but the selectivity in
  // 'existences' does not.
  float fanout;

  // the selectivity from 'existences'. 0.2 means that the join of 'tables' is
  // reduced 5x.
  float existsFanout{1};
};

/// Represents a join to add to a partial plan. One join candidate can make
/// many NextJoins, e.g, for different join methods. If one is clearly best,
/// not all need be tried. If many NextJoins are disconnected (no JoinEdge
/// between them), these may be statically orderable without going through
/// permutations.
struct NextJoin {
  NextJoin(
      const JoinCandidate* candidate,
      const RelationOpPtr& plan,
      const Cost& cost,
      const PlanObjectSet& placed,
      const PlanObjectSet& columns,
      const BuildSet& builds)
      : candidate(candidate),
        plan(plan),
        cost(cost),
        placed(placed),
        columns(columns),
        newBuilds(builds) {}

  const JoinCandidate* candidate;
  RelationOpPtr plan;
  Cost cost;
  PlanObjectSet placed;
  PlanObjectSet columns;
  BuildSet newBuilds;

  /// If true, only 'other' should be tried. Use to compare equivalent joins
  /// with different join method or partitioning.
  bool isWorse(const NextJoin& other) const;
};

class Optimization;

/// Tracks the set of tables / columns that have been placed or are still needed
/// when constructing a partial plan.
struct PlanState {
  PlanState(Optimization& optimization, DerivedTableP dt)
      : optimization(optimization), dt(dt) {}

  PlanState(Optimization& optimization, DerivedTableP dt, PlanPtr plan)
      : optimization(optimization), dt(dt), cost(plan->cost) {}

  Optimization& optimization;
  // The derived table from which the tables are drawn.
  DerivedTableP dt{nullptr};

  // The tables that have been placed so far.
  PlanObjectSet placed;

  // The columns that have a value from placed tables.
  PlanObjectSet columns;

  // The columns that need a value at the end of the plan. A dt can be
  // planned for just join/filter columns or all payload. Initially,
  // columns the selected columns of the dt depend on.
  PlanObjectSet targetColumns;

  // lookup keys for an index based derived table.
  PlanObjectSet input;

  // The total cost for the PlanObjects placed thus far.
  Cost cost;

  // All the hash join builds in any branch of the partial plan constructed so
  // far.
  BuildSet builds;

  // True if we should backtrack when 'costs' exceeds the best cost with shuffle
  // from already generated plans.
  bool hasCutoff{true};

  // Interesting completed plans for the dt being planned. For
  // example, best by cost and maybe plans with interesting orders.
  PlanSet plans;

  // Caches results of downstreamColumns(). This is a pure function of
  // 'placed' a'targetColumns' and 'dt'.
  mutable std::unordered_map<PlanObjectSet, PlanObjectSet>
      downstreamPrecomputed;

  /// Updates 'cost_' to reflect 'op' being placed on top of the partial plan.
  void addCost(RelationOp& op);

  /// Adds 'added' to all hash join builds.
  void addBuilds(const BuildSet& added);

  // Specifies that the plan to make only references 'target' columns and
  // whatever these depend on. These refer to 'columns' of 'dt'.
  void setTargetColumnsForDt(const PlanObjectSet& target);

  /// Returns the  set of columns referenced in unplaced joins/filters union
  /// targetColumns. Gets smaller as more tables are placed.
  PlanObjectSet downstreamColumns() const;

  // Adds a placed join to the set of partial queries to be developed. No op if
  // cost exceeds best so far and cutoff is enabled.
  void addNextJoin(
      const JoinCandidate* candidate,
      RelationOpPtr plan,
      BuildSet builds,
      std::vector<NextJoin>& toTry) const;

  std::string printCost() const;

  /// Makes a string of 'op' with 'details'. Costs are annotated with percentage
  /// of total in 'this->cost'.
  std::string printPlan(RelationOpPtr op, bool detail) const;

  /// True if the costs accumulated so far are so high that this should not be
  /// explored further.
  bool isOverBest() const {
    return hasCutoff && plans.bestPlan &&
        cost.unitCost + cost.setupCost > plans.bestCostWithShuffle;
  }
};

/// A scoped guard that restores fields of PlanState on destruction.
struct PlanStateSaver {
 public:
  explicit PlanStateSaver(PlanState& state)
      : state_(state),
        placed_(state.placed),
        columns_(state.columns),
        cost_(state.cost),
        numBuilds_(state.builds.size()) {}

  ~PlanStateSaver() {
    state_.placed = std::move(placed_);
    state_.columns = std::move(columns_);
    state_.cost = cost_;
    state_.builds.resize(numBuilds_);
  }

 private:
  PlanState& state_;
  PlanObjectSet placed_;
  PlanObjectSet columns_;
  const Cost cost_;
  const int32_t numBuilds_;
};

/// Key for collection of memoized partial plans. These are all made for hash
/// join builds with different cardinality reducing joins pushed down. The first
/// table is the table for which the key represents the build side. The 'tables'
/// set is the set of reducing joins applied to 'firstTable', including the
/// table itself. 'existences' is another set of reducing joins that are
/// semijoined to the join of 'tables' in order to restrict the build side.
struct MemoKey {
  bool operator==(const MemoKey& other) const;
  size_t hash() const;

  PlanObjectCP firstTable;
  PlanObjectSet columns;
  PlanObjectSet tables;
  std::vector<PlanObjectSet> existences;
};

} // namespace facebook::velox::optimizer

namespace std {
template <>
struct hash<::facebook::velox::optimizer::MemoKey> {
  size_t operator()(const ::facebook::velox::optimizer::MemoKey& key) const {
    return key.hash();
  }
};
} // namespace std

namespace facebook::velox::optimizer {

/// Instance of query optimization. Comverts a plan and schema into an
/// optimized plan. Depends on QueryGraphContext being set on the
/// calling thread. There is one instance per query to plan. The
/// instance must stay live as long as a returned plan is live.
class Optimization {
 public:
  static constexpr int32_t kRetained = 1;
  static constexpr int32_t kExceededBest = 2;

  using PlanCostMap = std::unordered_map<
      velox::core::PlanNodeId,
      std::vector<std::pair<std::string, Cost>>>;

  Optimization(
      const velox::core::PlanNode& plan,
      const Schema& schema,
      History& history,
      velox::core::ExpressionEvaluator& evaluator,
      int32_t traceFlags = 0);

  /// Returns the optimized RelationOp plan for 'plan' given at construction.
  PlanPtr bestPlan();

  /// Returns a set of per-stage Velox PlanNode trees.
  velox::runner::MultiFragmentPlanPtr toVeloxPlan(
      RelationOpPtr plan,
      const velox::runner::MultiFragmentPlan::Options& options);

  // Produces trace output if event matches 'traceFlags_'.
  void trace(int32_t event, int32_t id, const Cost& cost, RelationOp& plan);

  void setLeafHandle(
      int32_t id,
      connector::ConnectorTableHandlePtr handle,
      std::vector<core::TypedExprPtr> extraFilters) {
    leafHandles_[id] = std::make_pair(handle, extraFilters);
  }

  std::pair<connector::ConnectorTableHandlePtr, std::vector<core::TypedExprPtr>>
  leafHandle(int32_t id) {
    auto it = leafHandles_.find(id);
    return it != leafHandles_.end()
        ? it->second
        : std::make_pair<
              std::shared_ptr<velox::connector::ConnectorTableHandle>,
              std::vector<core::TypedExprPtr>>(nullptr, {});
  }

  // Translates from Expr to Velox.
  velox::core::TypedExprPtr toTypedExpr(ExprCP expr);
  auto& idGenerator() {
    return idGenerator_;
  }

  /// Sets 'filterSelectivity' of 'baseTable' from history. Returns True if set.
  bool setLeafSelectivity(BaseTable& baseTable) {
    return history_.setLeafSelectivity(baseTable);
  }

  auto& memo() {
    return memo_;
  }

  auto& existenceDts() {
    return existenceDts_;
  }

  // Lists the possible joins based on 'state.placed' and adds each on top of
  // 'plan'. This is a set of plans extending 'plan' by one join (single table
  // or bush). Calls itself on the interesting next plans. If all tables have
  // been used, adds postprocess and adds the plan to 'plans' in 'state'. If
  // 'state' enables cutoff and a partial plan is worse than the best so far,
  // discards the candidate.
  void makeJoins(RelationOpPtr plan, PlanState& state);

  velox::core::ExpressionEvaluator* evaluator() {
    return &evaluator_;
  }

  Name newCName(const std::string& prefix) {
    return toName(fmt::format("{}{}", prefix, ++nameCounter_));
  }

  PlanCostMap planCostMap() {
    return costEstimates_;
  }

  bool& makeVeloxExprWithNoAlias() {
    return makeVeloxExprWithNoAlias_;
  }

  // Makes an output type for use in PlanNode et al. If 'columnType' is set,
  // only considers base relation columns of the given type.
  velox::RowTypePtr makeOutputType(const ColumnVector& columns);

 private:
  static constexpr uint64_t kAllAllowedInDt = ~0UL;

  // True if 'op' is in 'mask.
  static bool contains(uint64_t mask, PlanType op) {
    return 0 != (mask & (1UL << static_cast<int32_t>(op)));
  }

  // Returns a mask that allows 'op' in the same derived table.
  uint64_t allow(PlanType op) {
    return 1UL << static_cast<int32_t>(op);
  }

  // Removes 'op' from the set of operators allowed in the current derived
  // table. makeQueryGraph() starts a new derived table if it finds an operator
  // that does not belong to the mask.
  static uint64_t makeDtIf(uint64_t mask, PlanType op) {
    return mask & ~(1UL << static_cast<int32_t>(op));
  }

  // Initializes a tree of DerivedTables with JoinEdges from 'plan' given at
  // construction. Sets 'root_' to the root DerivedTable.
  DerivedTableP makeQueryGraph();

  // Converts 'plan' to PlanObjects and records join edges into
  // 'currentSelect_'. If 'node' does not match  allowedInDt, wraps 'node' in a
  // new DerivedTable.
  PlanObjectP makeQueryGraph(
      const velox::core::PlanNode& node,
      uint64_t allowedInDt);

  // Converts a table scan into a BaseTable wen building a DerivedTable.
  PlanObjectP makeBaseTable(const core::TableScanNode* tableScan);

  // Interprets a Project node and adds its information into the DerivedTable
  // being assembled.
  void addProjection(const core::ProjectNode* project);

  // Interprets a Filter node and adds its information into the DerivedTable
  // being assembled.
  void addFilter(const core::FilterNode* Filter);

  // Interprets an AggregationNode and adds its information to the DerivedTable
  // being assembled.
  PlanObjectP addAggregation(
      const core::AggregationNode& aggNode,
      uint64_t allowedInDt);

  // Sets the columns to project out from the root DerivedTable  based on
  // 'plan'.
  void setDerivedTableOutput(
      DerivedTableP dt,
      const velox::core::PlanNode& planNode);

  // Returns a literal from applying 'call' or 'cast' to 'literals'. nullptr if
  // not successful.
  ExprCP tryFoldConstant(
      const velox::core::CallTypedExpr* call,
      const velox::core::CastTypedExpr* cast,
      const ExprVector& literals);

  // Makes a deduplicated Expr tree from 'expr'.
  ExprCP translateExpr(const velox::core::TypedExprPtr& expr);

  // Adds conjuncts combined by any number of enclosing ands from 'input' to
  // 'flat'.
  void translateConjuncts(
      const velox::core::TypedExprPtr& input,
      ExprVector& flat);

  // Converts 'name' to a deduplicated ExprCP. If 'name' is assigned to an
  // expression in a projection, returns the deduplicated ExprPtr of the
  // expression.
  ExprCP translateColumn(const std::string& name);

  //  Applies translateColumn to a 'source'.
  ExprVector translateColumns(
      const std::vector<velox::core::FieldAccessTypedExprPtr>& source);

  // Adds a JoinEdge corresponding to 'join' to the enclosing DerivedTable.
  void translateJoin(const velox::core::AbstractJoinNode& join);

  // Makes an extra column for existence flag.
  ColumnCP makeMark(const velox::core::AbstractJoinNode& join);

  // Adds a join edge for a join with no equalities.
  void translateNonEqualityJoin(const velox::core::NestedLoopJoinNode& join);

  // Adds order by information to the enclosing DerivedTable.
  OrderByP translateOrderBy(const velox::core::OrderByNode& order);

  // Adds aggregation information to the enclosing DerivedTable.
  AggregationP translateAggregation(
      const velox::core::AggregationNode& aggregation);

  // Adds 'node' and descendants to query graph wrapped inside a
  // DerivedTable. Done for joins to the right of non-inner joins,
  // group bys as non-top operators, whenever descendents of 'node'
  // are not freely reorderable with its parents' descendents.
  PlanObjectP wrapInDt(const velox::core::PlanNode& node);

  /// Retrieves or makes a plan from 'key'. 'key' specifies a set of
  /// top level joined tables or a hash join build side table or
  /// join. 'distribution' is the desired output distribution or a
  /// distribution with no partitioning if this does
  /// matter. 'boundColumns' is a set of columns that are lookup keys
  /// for an index based path through the joins in
  /// 'key'. 'existsFanout' is the selectivity for the 'existences' in
  /// 'key', i.e. extra reducing joins for a hash join build side,
  /// reflecting reducing joins on the probe side, 1 if none. 'state'
  /// is the state of the caller, empty for a top level call and the
  /// state with the planned objects so far if planning a derived
  /// table. 'needsShuffle' is set to true if a shuffle is needed to
  /// align the result of the made plan with 'distribution'.
  PlanPtr makePlan(
      const MemoKey& key,
      const Distribution& distribution,
      const PlanObjectSet& boundColumns,
      float existsFanout,
      PlanState& state,
      bool& needsShuffle);

  // Returns a sorted list of candidates to add to the plan in
  // 'state'. The joinable tables depend on the tables already present
  // in 'plan'. A candidate will be a single table for all the single
  // tables that can be joined. Additionally, when the single table
  // can be joined to more tables not in 'state' to form a reducing
  // join, this is produced as a candidate for a bushy hash join. When
  // a single table or join to be used as a hash build side is made,
  // we further check if reducing joins applying to the probe can be
  // used to furthr reduce the build. These last joins are added as
  // 'existences' in the candidate.
  std::vector<JoinCandidate> nextJoins(PlanState& state);

  // Adds group by, order by, top k to 'plan'. Updates 'plan' if
  // relation ops added.  Sets cost in 'state'.
  void addPostprocess(DerivedTableP dt, RelationOpPtr& plan, PlanState& state);

  // Places a derived table as first table in a plan. Imports possibly reducing
  // joins into the plan if can.
  void placeDerivedTable(const DerivedTable* from, PlanState& state);

  // Adds the items from 'dt.conjuncts' that are not placed in 'state'
  // and whose prerequisite columns are placed. If conjuncts can be
  // placed, adds them to 'state.placed' and calls makeJoins()
  // recursively to make the rest of the plan. Returns false if no
  // unplaced conjuncts were found and and plan construction should
  // proceed.
  bool placeConjuncts(RelationOpPtr plan, PlanState& state);

  // Helper function that calls makeJoins recursively for each of
  // 'nextJoins'. The point of making 'nextJoins' first and only then
  // calling makeJoins is to allow detecting a star pattern of a fact
  // table and independently joined dimensions. These can be ordered
  // based on partitioning and size and we do not need to evaluate
  // their different permutations.
  void tryNextJoins(PlanState& state, const std::vector<NextJoin>& nextJoins);

  // Adds a cross join to access a single row from a non-correlated subquery.
  RelationOpPtr placeSingleRowDt(
      RelationOpPtr plan,
      const DerivedTable* subq,
      ExprCP filter,
      PlanState& state);

  // Adds the join represented by'candidate' on top of 'plan'. Tries index and
  // hash based methods and adds the index and hash based plans to 'result'. If
  // one of these is clearly superior, only adds the better one.
  void addJoin(
      const JoinCandidate& candidate,
      const RelationOpPtr& plan,
      PlanState& state,
      std::vector<NextJoin>& result);

  // If 'candidate' can be added on top 'plan' as a merge/index lookup, adds the
  // plan to 'toTry'. Adds any necessary repartitioning.
  void joinByIndex(
      const RelationOpPtr& plan,
      const JoinCandidate& candidate,
      PlanState& state,
      std::vector<NextJoin>& toTry);

  // Adds 'candidate' on top of 'plan as a hash join. Adds possibly needed
  // repartitioning to both probe and build and makes a broadcast build if
  // indicated. If 'candidate' calls for a join on the build ide, plans a
  // derived table with the build side tables and optionl 'existences' from
  // 'candidate'.
  void joinByHash(
      const RelationOpPtr& plan,
      const JoinCandidate& candidate,
      PlanState& state,
      std::vector<NextJoin>& toTry);

  /// Tries a right hash join variant of left outer or left semijoin.
  void joinByHashRight(
      const RelationOpPtr& plan,
      const JoinCandidate& candidate,
      PlanState& state,
      std::vector<NextJoin>& toTry);

  void crossJoin(
      const RelationOpPtr& plan,
      const JoinCandidate& candidate,
      PlanState& state,
      std::vector<NextJoin>& toTry);

  // Returns a filter expr that ands 'exprs'. nullptr if 'exprs' is empty.
  velox::core::TypedExprPtr toAnd(const ExprVector& exprs);

  // Translates 'exprs' and returns them in 'result'. If an expr is
  // other than a column, adds a projection node to evaluate the
  // expression. The projection is added on top of 'source' and
  // returned. If no projection is added, 'source' is returned.
  velox::core::PlanNodePtr maybeProject(
      const ExprVector& exprs,
      velox::core::PlanNodePtr source,
      std::vector<velox::core::FieldAccessTypedExprPtr>& result);

  // Makes a Velox AggregationNode for a RelationOp.
  velox::core::PlanNodePtr makeAggregation(
      Aggregation& agg,
      velox::runner::ExecutableFragment& fragment,
      std::vector<velox::runner::ExecutableFragment>& stages);

  // Makes partial + final order by fragments for order by with and without
  // limit.
  velox::core::PlanNodePtr makeOrderBy(
      OrderBy& op,
      velox::runner::ExecutableFragment& fragment,
      std::vector<velox::runner::ExecutableFragment>& stages);

  // Makes a tree of PlanNode for a tree of
  // RelationOp. 'fragment' is the fragment that 'op'
  // belongs to. If op or children are repartitions then the
  // source of each makes a separate fragment. These
  // fragments are referenced from 'fragment' via
  // 'inputStages' and are returned in 'stages'.
  velox::core::PlanNodePtr makeFragment(
      RelationOpPtr op,
      velox::runner::ExecutableFragment& fragment,
      std::vector<velox::runner::ExecutableFragment>& stages);

  // Returns a new PlanNodeId and associates the Cost of 'op' with it.
  velox::core::PlanNodeId nextId(const RelationOp& op);

  // Records 'cost' for 'id'. 'role' can be e.g. 'build; or
  // 'probe'. for nodes that produce multiple operators.
  void recordPlanNodeEstimate(
      const velox::core::PlanNodeId id,
      Cost cost,
      const std::string& role);

  const Schema& schema_;

  // Top level plan to optimize.
  const velox::core::PlanNode& inputPlan_;

  // Source of historical cost/cardinality information.
  History& history_;
  velox::core::ExpressionEvaluator& evaluator_;
  // Top DerivedTable when making a QueryGraph from PlanNode.
  DerivedTableP root_;

  // Innermost DerivedTable when making a QueryGraph from PlanNode.
  DerivedTableP currentSelect_;

  // Maps names in project noes of 'inputPlan_' to deduplicated Exprs.
  std::unordered_map<std::string, ExprCP> renames_;

  // Maps unique core::TypedExprs from 'inputPlan_' to deduplicated Exps.
  ExprDedupMap exprDedup_;

  // Counter for generating unique correlation names for BaseTables and
  // DerivedTables.
  int32_t nameCounter_{0};

  // Serial number for columns created for projections that name Exprs, e.g. in
  // join or grouping keys.
  int32_t resultNameCounter_{0};

  // Serial number for stages in executable plan.
  int32_t stageCounter_{0};

  std::unordered_map<MemoKey, PlanSet> memo_;

  // Set of previously planned dts for importing probe side reducing joins to a
  // build side
  std::unordered_map<MemoKey, DerivedTableP> existenceDts_;

  // The top level PlanState. Contains the set of top level interesting plans.
  // Must stay alive as long as the Plans and RelationOps are reeferenced.
  PlanState topState_{*this, nullptr};

  // Controls tracing.
  int32_t traceFlags_{0};

  // Generates unique ids for build sides.
  int32_t buildCounter_{0};

  // When making a graph from 'inputPlan_' the output of an aggregation comes
  // from the topmost (final) and the input from the lefmost (whichever consumes
  // raw values). Records the output type of the final aggregation.
  velox::RowTypePtr aggFinalType_;

  // Map from plan object id to pair of handle with pushdown filters and list of
  // filters to eval on the result from the handle.
  std::unordered_map<
      int32_t,
      std::pair<
          connector::ConnectorTableHandlePtr,
          std::vector<core::TypedExprPtr>>>
      leafHandles_;

  class PlanNodeIdGenerator {
   public:
    explicit PlanNodeIdGenerator(int startId = 0) : nextId_{startId} {}

    core::PlanNodeId next() {
      return fmt::format("{}", nextId_++);
    }

    void reset(int startId = 0) {
      nextId_ = startId;
    }

   private:
    int nextId_;
  };

  velox::runner::MultiFragmentPlan::Options options_;
  PlanNodeIdGenerator idGenerator_;
  // Limit for a possible limit/top k order by for while making a Velox plan. -1
  // means no limit.
  int32_t toVeloxLimit_{-1};
  int32_t toVeloxOffset_{0};

  // map from Velox PlanNode ids to RelationOp. For cases that have multiple
  // operators, e.g. probe and build, both RelationOps are mentioned.
  PlanCostMap costEstimates_;

  // On when producing a remaining filter for table scan, where columns must
  // correspond 1:1 to the schema.
  bool makeVeloxExprWithNoAlias_{false};
};

/// Returns bits describing function 'name'.
FunctionSet functionBits(Name name);

const JoinEdgeVector& joinedBy(PlanObjectCP table);

void filterUpdated(BaseTableCP baseTable);

} // namespace facebook::velox::optimizer
