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

#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/py/file/PyFile.h"
#include "velox/py/type/PyType.h"

namespace facebook::velox::py {

/// Called when the Python module is loaded/initialized to register Velox
/// resources like serde functions, type resolver, local filesystem and Presto
/// functions (scalar and aggregate).
void registerAllResources();

class PyVector;

// Stores the data associated with a table scan leaf operator. Map from the plan
// node id to a list of splits.
using TScanFiles = std::unordered_map<
    core::PlanNodeId,
    std::vector<std::shared_ptr<connector::ConnectorSplit>>>;
using TScanFilesPtr = std::shared_ptr<TScanFiles>;

/// Thin wrapper class used to expose Velox plan nodes to Python. It is only
/// used to maintain the internal Velox structure and expose basic string
/// serialization functionality.
class PyPlanNode {
 public:
  /// Creates a new PyPlanNode wrapper given a Velox plan node. Throws if
  /// planNode is nullptr.
  PyPlanNode(core::PlanNodePtr planNode, const TScanFilesPtr& scanFiles);

  const core::PlanNodePtr& planNode() const {
    return planNode_;
  }

  std::string id() const {
    return std::string{planNode_->id()};
  }

  std::string serialize() const {
    return folly::toJson(planNode_->serialize());
  }

  std::string name() const {
    return std::string(planNode_->name());
  }

  std::string toString(bool detailed = false, bool recursive = false) const {
    return planNode_->toString(detailed, recursive);
  }

  const TScanFilesPtr& scanFiles() const {
    return scanFiles_;
  }

 private:
  const core::PlanNodePtr planNode_;
  const TScanFilesPtr scanFiles_;
};

/// Wrapper class for PlanBuilder. It allows us to avoid exposing all details of
/// the class to users making it easier to use.
class PyPlanBuilder {
 public:
  /// Constructs a new PyPlanBuilder. If provided, the planNodeIdGenerator is
  /// used; otherwise a new one is created.
  PyPlanBuilder(
      const std::shared_ptr<core::PlanNodeIdGenerator>& generator = nullptr,
      const TScanFilesPtr& scanFiles = nullptr);

  // Returns the plan node at the head of the internal plan builder. If there is
  // no plan node (plan builder is empty), then std::nullopt will signal pybind
  // to return None to the Python client.
  std::optional<PyPlanNode> planNode() const;

  // Returns a new builder sharing the plan node id generator, such that the new
  // builder can safely be used to build other parts/pipelines of the same plan.
  PyPlanBuilder newBuilder() {
    DCHECK(planNodeIdGenerator_ != nullptr);
    DCHECK(scanFiles_ != nullptr);
    return PyPlanBuilder{planNodeIdGenerator_, scanFiles_};
  }

  /// Add table scan node with basic functionality. This API is Hive-connector
  /// specific.
  ///
  /// @param outputSchema The schema to be read from the file. Needs to be a
  /// RowType.
  ///
  /// @param aliases Aliases to apply, mapping from the desired output name to
  /// the input name in the file. If there are aliases, OutputSchema should be
  /// defined based on the aliased names. The Python dict should contain strings
  /// for keys and values.
  ///
  /// @param subfields Used for projecting subfields from containers (like map
  /// keys or struct fields), when one does not want to read the entire
  /// container. It's a dictionary mapping from column name (string) to the list
  /// of subitems to project from it. For now, a list of integers representing
  /// the subfields in a flatmap/struct.
  ///
  /// @param rowIndexColumnName If defined, create an output column with that
  /// name producing $row_ids. This name needs to be part of `output`.
  ///
  /// @param connectorId ID of the connector to use for this scan. By default
  /// the Python module will specify the default name used to register the
  /// Hive connector in the first place.
  ///
  /// @param inputFile List of files to be collected and converted into splits
  /// by runners during execution.
  ///
  /// Example (from Python API):
  ///
  /// tableScan(
  ///   output=ROW(
  ///     names=["row_number", "$row_group_id", "aliased_name"],
  ///     types=[BIGINT(), VARCHAR(), MAP(INTEGER(), VARCHAR())],
  ///   ),
  ///   subfields={"aliased_name": [1234, 5431, 65223]},
  ///   row_index="row_number",
  ///   aliases={
  ///     "aliased_name": "name_in_file",
  ///    },
  ///    input_files=[PARQUET("my_file.parquet")],
  ///  )
  PyPlanBuilder& tableScan(
      const PyType& outputSchema,
      const pybind11::dict& aliases,
      const pybind11::dict& subfields,
      const std::string& rowIndexColumnName,
      const std::string& connectorId,
      const std::optional<std::vector<PyFile>>& inputFiles);

  /// Adds a table writer node to write to an output file(s).
  ///
  /// @param outputFile The output file to be written.
  /// @param connectorId The id of the connector to use during the write
  /// process.
  /// @param outputSchema An optional schema to be used when writing the file
  /// (columns and types). By default use the schema produced by the upstream
  /// operator.
  PyPlanBuilder& tableWrite(
      const PyFile& outputFile,
      const std::string& connectorId,
      const std::optional<PyType>& outputSchema);

  /// Add the provided vectors straight into the operator tree.
  PyPlanBuilder& values(const std::vector<PyVector>& values);

  /// Add a list of projections. Projections are specified as SQL expressions
  /// and currently use DuckDB's parser semantics.
  PyPlanBuilder& project(const std::vector<std::string>& projections);

  /// Add a filter (selection) to the plan. Filters are specified as SQL
  /// expressions and currently use DuckDB's parser semantics.
  PyPlanBuilder& filter(const std::string& filter);

  /// Add a single-stage aggregation given a set of group keys, and
  /// aggregations. Aggregations are specified as SQL expressions and currently
  /// use DuckDB's parser semantics.
  PyPlanBuilder& aggregate(
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregations);

  /// Sorts the input based on the values of sorting keys.
  ///
  /// @param keys List of columns to order by. The strings can be column
  /// names and optionally contain the sort orientation ("col" or "col DESC").
  /// @param is_partial
  PyPlanBuilder& orderBy(const std::vector<std::string>& keys, bool isPartial);

  /// Limit how many rows from the input to produce as output.
  ///
  /// @param count How many rows to produce, at most.
  /// @param offset Hoy many rows from the beggining of the input to skip.
  /// @param is_partial If this is restricting partial results and hence can be
  /// applied once per driver, or if it's applied to the query output.
  PyPlanBuilder& limit(int64_t count, int64_t offset, bool isPartial);

  /// Add a merge join node to the plan. Assumes that both left and right
  /// subtrees will produce input sorted in join key order.
  ///
  /// @param leftKeys Set of join keys from the left (current plan builder)
  /// plan subtree.
  /// @param leftKeys Set of join keys from the right plan subtree.
  /// @param rightPlanSubtree Subtree to join to.
  /// @param output List of column names to project in the output of the
  /// join.
  /// @param filter An optional filter specified as a SQL expression to be
  /// applied during the join.
  /// @param joinType The type of join (kInner, kLeft, kRight, or kFull)
  PyPlanBuilder& mergeJoin(
      const std::vector<std::string>& leftKeys,
      const std::vector<std::string>& rightKeys,
      const PyPlanNode& rightPlanSubtree,
      const std::vector<std::string>& output,
      const std::string& filter,
      core::JoinType joinType);

  /// Takes N sorted `source` subtrees and merges them into a sorted output.
  /// Assumes that all sources are sorted on `keys`.
  ///
  /// @param keys The sorting keys.
  /// @param sources The list of sources to merge.
  PyPlanBuilder& sortedMerge(
      const std::vector<std::string>& keys,
      const std::vector<std::optional<PyPlanNode>>& sources);

  /// Generates TPC-H data on the fly using dbgen. Note that generating data on
  /// the fly is not terribly efficient, so for performance evaluation one
  /// should generate data using this node, write it to output storage files,
  /// (Parquet, ORC, or similar), then benchmark a query plan that reads those
  /// files.
  ///
  /// @param tableName The TPC-H table name to generate data for.
  /// @param columns The columns from `table_name` to generate data for. If
  /// empty (the default), generate data for all columns.
  /// @param scaleFactor TPC-H scale factor to use - controls the amount of
  /// data generated.
  /// @param numParts How many splits to generate. This controls the
  /// parallelism and the number of output files to be generated.
  /// @param connector_id ID of the connector to use for this scan.
  PyPlanBuilder& tpchGen(
      const std::string& tableName,
      const std::vector<std::string>& columns,
      double scaleFactor,
      size_t numParts,
      const std::string& connectorId);

  // TODO: Add other nodes.

 private:
  exec::test::PlanBuilder planBuilder_;
  std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator_;

  // For API convenience, we allow clients to specify the files themselves when
  // they add scans, so that we can automatically create and attach splits to
  // the task. If specified in the .tableScan() call, store these files here.
  TScanFilesPtr scanFiles_;
};

} // namespace facebook::velox::py
