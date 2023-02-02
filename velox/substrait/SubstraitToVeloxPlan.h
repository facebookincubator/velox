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

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/core/PlanNode.h"
#include "velox/substrait/SubstraitToVeloxExpr.h"

namespace facebook::velox::substrait {

/// This class is used to convert the Substrait plan into Velox plan.
class SubstraitVeloxPlanConverter {
 public:
  explicit SubstraitVeloxPlanConverter(memory::MemoryPool* pool)
      : pool_(pool) {}
  struct SplitInfo {
    /// The Partition index.
    u_int32_t partitionIndex;

    /// The file paths to be scanned.
    std::vector<std::string> paths;

    /// The file starts in the scan.
    std::vector<u_int64_t> starts;

    /// The lengths to be scanned.
    std::vector<u_int64_t> lengths;

    /// The file format of the files to be scanned.
    dwio::common::FileFormat format;
  };

  /// Convert Substrait AggregateRel into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::AggregateRel& aggRel);

  /// Convert Substrait ProjectRel into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::ProjectRel& projectRel);

  /// Convert Substrait FilterRel into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::FilterRel& filterRel);

  /// Convert Substrait ReadRel into Velox PlanNode.
  /// Index: the index of the partition this item belongs to.
  /// Starts: the start positions in byte to read from the items.
  /// Lengths: the lengths in byte to read from the items.
  core::PlanNodePtr toVeloxPlan(
      const ::substrait::ReadRel& readRel,
      std::shared_ptr<SplitInfo>& splitInfo);

  /// Convert Substrait FetchRel into Velox LimitNode or TopNNode according the
  /// different input of fetchRel.
  core::PlanNodePtr toVeloxPlan(const ::substrait::FetchRel& fetchRel);

  /// Convert Substrait ReadRel into Velox Values Node.
  core::PlanNodePtr toVeloxPlan(
      const ::substrait::ReadRel& readRel,
      const RowTypePtr& type);

  /// Convert Substrait Rel into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::Rel& rel);

  /// Convert Substrait RelRoot into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::RelRoot& root);

  /// Convert Substrait SortRel into Velox OrderByNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::SortRel& sortRel);

  /// Convert Substrait Plan into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::Plan& substraitPlan);

  /// Check the Substrait type extension only has one unknown extension.
  bool checkTypeExtension(const ::substrait::Plan& substraitPlan);

  /// Construct the function map between the index and the Substrait function
  /// name.
  void constructFunctionMap(const ::substrait::Plan& substraitPlan);

  /// Return the function map used by this plan converter.
  const std::unordered_map<uint64_t, std::string>& getFunctionMap() const {
    return functionMap_;
  }

  /// Return the splitInfo map used by this plan converter.
  const std::unordered_map<core::PlanNodeId, std::shared_ptr<SplitInfo>>&
  splitInfos() const {
    return splitInfoMap_;
  }

  /// Looks up a function by ID and returns function name if found. Throws if
  /// function with specified ID doesn't exist. Returns a compound
  /// function specification consisting of the function name and the input
  /// types. The format is as follows: <function
  /// name>:<arg_type0>_<arg_type1>_..._<arg_typeN>
  const std::string& findFunction(uint64_t id) const;

  /// Integrate Substrait emit feature. Here a given 'substrait::RelCommon'
  /// is passed and check if emit is defined for this relation. Basically a
  /// ProjectNode is added on top of 'noEmitNode' to represent output order
  /// specified in 'relCommon::emit'. Return 'noEmitNode' as is
  /// if output order is 'kDriect'.
  core::PlanNodePtr processEmit(
      const ::substrait::RelCommon& relCommon,
      const core::PlanNodePtr& noEmitNode);

 private:
  /// Returns unique ID to use for plan node. Produces sequential numbers
  /// starting from zero.
  std::string nextPlanNodeId();

  /// Used to convert Substrait Filter into Velox SubfieldFilters which will
  /// be used in TableScan.
  connector::hive::SubfieldFilters toVeloxFilter(
      const std::vector<std::string>& inputNameList,
      const std::vector<TypePtr>& inputTypeList,
      const ::substrait::Expression& substraitFilter);

  /// Multiple conditions are connected to a binary tree structure with
  /// the relation key words, including AND, OR, and etc. Currently, only
  /// AND is supported. This function is used to extract all the Substrait
  /// conditions in the binary tree structure into a vector.
  void flattenConditions(
      const ::substrait::Expression& substraitFilter,
      std::vector<::substrait::Expression_ScalarFunction>& scalarFunctions);

  /// The Substrait parser used to convert Substrait representations into
  /// recognizable representations.
  std::shared_ptr<SubstraitParser> substraitParser_{
      std::make_shared<SubstraitParser>()};

  /// Helper Function to convert Substrait sortField to Velox sortingKeys and
  /// sortingOrders.
  std::pair<
      std::vector<core::FieldAccessTypedExprPtr>,
      std::vector<core::SortOrder>>
  processSortField(
      const ::google::protobuf::RepeatedPtrField<::substrait::SortField>&
          sortField,
      const RowTypePtr& inputType);

  /// The Expression converter used to convert Substrait representations into
  /// Velox expressions.
  std::shared_ptr<SubstraitVeloxExprConverter> exprConverter_;

  /// The unique identification for each PlanNode.
  int planNodeId_ = 0;

  /// The map storing the relations between the function id and the function
  /// name. Will be constructed based on the Substrait representation.
  std::unordered_map<uint64_t, std::string> functionMap_;

  /// Mapping from leaf plan node ID to splits.
  std::unordered_map<core::PlanNodeId, std::shared_ptr<SplitInfo>>
      splitInfoMap_;

  /// Memory pool.
  memory::MemoryPool* pool_;

  /// Helper function to convert the input of Substrait Rel to Velox Node.
  template <typename T>
  core::PlanNodePtr convertSingleInput(T rel) {
    VELOX_CHECK(rel.has_input(), "Child Rel is expected here.");
    return toVeloxPlan(rel.input());
  }
};

} // namespace facebook::velox::substrait
