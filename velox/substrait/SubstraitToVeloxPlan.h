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
#include "velox/substrait/TypeUtils.h"

namespace facebook::velox::substrait {
struct SplitInfo {
  /// Whether the split comes from arrow array stream node.
  bool isStream = false;

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

/// This class is used to convert the Substrait plan into Velox plan.
class SubstraitVeloxPlanConverter {
 public:
  SubstraitVeloxPlanConverter(
      memory::MemoryPool* pool,
      bool validationMode = false)
      : pool_(pool), validationMode_(validationMode) {}

  /// This class is used to convert the Substrait plan into Velox plan.
  /// Used to convert Substrait JoinRel into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::JoinRel& sJoin);

  /// Used to convert Substrait AggregateRel into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::AggregateRel& sAgg);

  /// Convert Substrait ProjectRel into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::ProjectRel& projectRel);

  /// Convert Substrait FilterRel into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::FilterRel& filterRel);

  /// Convert Substrait ReadRel into Velox PlanNode.
  /// Index: the index of the partition this item belongs to.
  /// Starts: the start positions in byte to read from the items.
  /// Lengths: the lengths in byte to read from the items.
  core::PlanNodePtr toVeloxPlan(const ::substrait::ReadRel& sRead);

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

  /// Convert Substrait ReadRel into Velox PlanNode.
  /// Index: the index of the partition this item belongs to.
  /// Starts: the start positions in byte to read from the items.
  /// Lengths: the lengths in byte to read from the items.
  core::PlanNodePtr toVeloxPlan(const ::substrait::ReadRel& sRead);

  /// Used to convert Substrait Rel into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::Rel& sRel);

  /// Used to convert Substrait RelRoot into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::RelRoot& sRoot);

  /// Used to convert Substrait Plan into Velox PlanNode.
  core::PlanNodePtr toVeloxPlan(const ::substrait::Plan& substraitPlan);

  /// Used to construct the function map between the index
  /// and the Substrait function name. Initialize the expression
  /// converter based on the constructed function map.
  void constructFunctionMap(const ::substrait::Plan& substraitPlan);

  /// Will return the function map used by this plan converter.
  const std::unordered_map<uint64_t, std::string>& getFunctionMap() {
    return functionMap_;
  }

  /// Will return the index of Partition to be scanned.
  u_int32_t getPartitionIndex() {
    return partitionIndex_;
  }

  /// Return the splitInfo map used by this plan converter.
  const std::unordered_map<core::PlanNodeId, std::shared_ptr<SplitInfo>>&
  splitInfos() const {
    return splitInfoMap_;
  }

  /// Used to insert certain plan node as input. The plan node
  /// id will start from the setted one.
  void insertInputNode(
      uint64_t inputIdx,
      const std::shared_ptr<const core::PlanNode>& inputNode,
      int planNodeId) {
    inputNodesMap_[inputIdx] = inputNode;
    planNodeId_ = planNodeId;
  }

  /// Used to check if ReadRel specifies an input of stream.
  /// If yes, the index of input stream will be returned.
  /// If not, -1 will be returned.
  int32_t streamIsInput(const ::substrait::ReadRel& sRel);

  /// Multiple conditions are connected to a binary tree structure with
  /// the relation key words, including AND, OR, and etc. Currently, only
  /// AND is supported. This function is used to extract all the Substrait
  /// conditions in the binary tree structure into a vector.
  void flattenConditions(
      const ::substrait::Expression& sFilter,
      std::vector<::substrait::Expression_ScalarFunction>& scalarFunctions,
      std::vector<::substrait::Expression_SingularOrList>& singularOrLists);

  /// Used to find the function specification in the constructed function map.
  std::string findFuncSpec(uint64_t id);

  /// Extract join keys from joinExpression.
  /// joinExpression is a boolean condition that describes whether each record
  /// from the left set “match” the record from the right set. The condition
  /// must only include the following operations: AND, ==, field references.
  /// Field references correspond to the direct output order of the data.
  void extractJoinKeys(
      const ::substrait::Expression& joinExpression,
      std::vector<const ::substrait::Expression::FieldReference*>& leftExprs,
      std::vector<const ::substrait::Expression::FieldReference*>& rightExprs);

 private:
  /// Memory pool.
  memory::MemoryPool* pool_;

  /// Range filter recorder for a field is used to make sure only the conditions
  /// that can coexist for this field being pushed down with a range filter.
  class RangeRecorder {
   public:
    /// Set the existence of values range and returns whether this condition can
    /// coexist with existing conditions for one field. Conditions in OR
    /// relation can coexist with each other.
    bool setInRange(bool forOrRelation = false) {
      if (forOrRelation) {
        return true;
      }
      if (inRange_ || multiRange_ || leftBound_ || rightBound_) {
        return false;
      }
      inRange_ = true;
      return true;
    }

    /// Set the existence of left bound and returns whether it can coexist with
    /// existing conditions for this field.
    bool setLeftBound(bool forOrRelation = false) {
      if (forOrRelation) {
        return true;
      }
      if (leftBound_ || inRange_ || multiRange_) {
        return false;
      }
      leftBound_ = true;
      return true;
    }

    /// Set the existence of right bound and returns whether it can coexist with
    /// existing conditions for this field.
    bool setRightBound(bool forOrRelation = false) {
      if (forOrRelation) {
        return true;
      }
      if (rightBound_ || inRange_ || multiRange_) {
        return false;
      }
      rightBound_ = true;
      return true;
    }

    /// Set the multi-range and returns whether it can coexist with
    /// existing conditions for this field.
    bool setMultiRange() {
      if (inRange_ || multiRange_ || leftBound_ || rightBound_) {
        return false;
      }
      multiRange_ = true;
      return true;
    }

    /// Set certain existence according to function name and returns whether it
    /// can coexist with existing conditions for this field.
    bool setCertainRangeForFunction(
        const std::string& functionName,
        bool reverse = false,
        bool forOrRelation = false);

   private:
    /// The existence of values range.
    bool inRange_ = false;

    /// The existence of left bound.
    bool leftBound_ = false;

    /// The existence of right bound.
    bool rightBound_ = false;

    /// The existence of multi-range.
    bool multiRange_ = false;
  };

  /// Filter info for a column used in filter push down.
  class FilterInfo {
   public:
    // Disable null allow.
    void forbidsNull() {
      nullAllowed_ = false;
      if (!isInitialized_) {
        isInitialized_ = true;
      }
    }

    // Return the initialization status.
    bool isInitialized() {
      return isInitialized_ ? true : false;
    }

    // Add a lower bound to the range. Multiple lower bounds are
    // regarded to be in 'or' relation.
    void setLower(const std::optional<variant>& left, bool isExclusive) {
      lowerBounds_.emplace_back(left);
      lowerExclusives_.emplace_back(isExclusive);
      if (!isInitialized_) {
        isInitialized_ = true;
      }
    }

    // Add a upper bound to the range. Multiple upper bounds are
    // regarded to be in 'or' relation.
    void setUpper(const std::optional<variant>& right, bool isExclusive) {
      upperBounds_.emplace_back(right);
      upperExclusives_.emplace_back(isExclusive);
      if (!isInitialized_) {
        isInitialized_ = true;
      }
    }

    // Set a list of values to be used in the push down of 'in' expression.
    void setValues(const std::vector<variant>& values) {
      for (const auto& value : values) {
        valuesVector_.emplace_back(value);
      }
      if (!isInitialized_) {
        isInitialized_ = true;
      }
    }

    // Set a value for the not(equal) condition.
    void setNotValue(const std::optional<variant>& notValue) {
      notValue_ = notValue;
      if (!isInitialized_) {
        isInitialized_ = true;
      }
    }

    // Whether this filter map is initialized.
    bool isInitialized_ = false;

    // The null allow.
    bool nullAllowed_ = false;

    // If true, left bound will be exclusive.
    std::vector<bool> lowerExclusives_;

    // If true, right bound will be exclusive.
    std::vector<bool> upperExclusives_;

    // A value should not be equal to.
    std::optional<variant> notValue_ = std::nullopt;

    // The lower bounds in 'or' relation.
    std::vector<std::optional<variant>> lowerBounds_;

    // The upper bounds in 'or' relation.
    std::vector<std::optional<variant>> upperBounds_;

    // The list of values used in 'in' expression.
    std::vector<variant> valuesVector_;
  };

  /// Helper Function to convert Substrait sortField to Velox sortingKeys and
  /// sortingOrders.
  std::pair<
      std::vector<core::FieldAccessTypedExprPtr>,
      std::vector<core::SortOrder>>
  processSortField(
      const ::google::protobuf::RepeatedPtrField<::substrait::SortField>&
          sortField,
      const RowTypePtr& inputType);

  /// A function returning current function id and adding the plan node id by
  /// one once called.
  std::string nextPlanNodeId();

  /// Returns whether the args of a scalar function being field or
  /// field with literal. If yes, extract and set the field index.
  bool fieldOrWithLiteral(
      const ::google::protobuf::RepeatedPtrField<::substrait::FunctionArgument>&
          arguments,
      uint32_t& fieldIndex);

  /// Separate the functions to be two parts:
  /// subfield functions to be handled by the subfieldFilters in
  /// HiveConnector, and remaining functions to be handled by the
  /// remainingFilter in HiveConnector.
  void separateFilters(
      const std::unordered_map<uint32_t, std::shared_ptr<RangeRecorder>>&
          rangeRecorders,
      const std::vector<::substrait::Expression_ScalarFunction>&
          scalarFunctions,
      std::vector<::substrait::Expression_ScalarFunction>& subfieldFunctions,
      std::vector<::substrait::Expression_ScalarFunction>& remainingFunctions,
      const std::vector<::substrait::Expression_SingularOrList>&
          singularOrLists,
      std::vector<::substrait::Expression_SingularOrList>& subfieldrOrLists,
      std::vector<::substrait::Expression_SingularOrList>& remainingrOrLists);

  /// Returns whether a function can be pushed down.
  bool canPushdownCommonFunction(
      const ::substrait::Expression_ScalarFunction& scalarFunction,
      const std::string& filterName,
      uint32_t& fieldIdx);

  /// Returns whether a NOT function can be pushed down.
  bool canPushdownNot(
      const ::substrait::Expression_ScalarFunction& scalarFunction,
      const std::unordered_map<uint32_t, std::shared_ptr<RangeRecorder>>&
          rangeRecorders);

  /// Returns whether a OR function can be pushed down.
  bool canPushdownOr(
      const ::substrait::Expression_ScalarFunction& scalarFunction,
      const std::unordered_map<uint32_t, std::shared_ptr<RangeRecorder>>&
          rangeRecorders);

  /// Returns whether a SingularOrList can be pushed down.
  bool canPushdownSingularOrList(
      const ::substrait::Expression_SingularOrList& singularOrList,
      bool disableIntLike = false);

  /// Returns a set of unique column indices for IN function to be pushed down.
  std::unordered_set<uint32_t> getInColIndices(
      const std::vector<::substrait::Expression_SingularOrList>&
          singularOrLists);

  /// Check whether the chidren functions of this scalar function have the same
  /// column index. Curretly used to check whether the two chilren functions of
  /// 'or' expression are effective on the same column.
  bool chidrenFunctionsOnSameField(
      const ::substrait::Expression_ScalarFunction& function);

  /// Extract the scalar function, and set the filter info for different types
  /// of columns. If reverse is true, the opposite filter info will be set.
  void setFilterMap(
      const ::substrait::Expression_ScalarFunction& scalarFunction,
      const std::vector<TypePtr>& inputTypeList,
      std::unordered_map<uint32_t, std::shared_ptr<FilterInfo>>& colInfoMap,
      bool reverse = false);

  /// Extract SingularOrList and returns the field index.
  uint32_t getColumnIndexFromSingularOrList(
      const ::substrait::Expression_SingularOrList& singularOrList);

  /// Extract SingularOrList and set it to the filter info map.
  void setSingularListValues(
      const ::substrait::Expression_SingularOrList& singularOrList,
      std::unordered_map<uint32_t, std::shared_ptr<FilterInfo>>& colInfoMap);

  /// Set the filter info for a column base on the information
  /// extracted from filter condition.
  template <typename T>
  void setColInfoMap(
      const std::string& filterName,
      uint32_t colIdx,
      std::optional<variant> literalVariant,
      bool reverse,
      std::unordered_map<uint32_t, std::shared_ptr<FilterInfo>>& colInfoMap);

  /// Create a multirange to specify the filter 'x != notValue' with:
  /// x > notValue or x < notValue.
  template <TypeKind KIND, typename FilterType>
  void createNotEqualFilter(
      variant notVariant,
      bool nullAllowed,
      std::vector<std::unique_ptr<FilterType>>& colFilters);

  /// Create a values range to handle in filter.
  /// variants: the list of values extracted from the in expression.
  /// inputName: the column input name.
  template <TypeKind KIND>
  void setInFilter(
      const std::vector<variant>& variants,
      bool nullAllowed,
      const std::string& inputName,
      connector::hive::SubfieldFilters& filters);

  /// Set the constructed filters into SubfieldFilters.
  /// The FilterType is used to distinguish BigintRange and
  /// Filter (the base class). This is needed because BigintMultiRange
  /// can only accept the unique ptr of BigintRange as parameter.
  template <TypeKind KIND, typename FilterType>
  void setSubfieldFilter(
      std::vector<std::unique_ptr<FilterType>> colFilters,
      const std::string& inputName,
      bool nullAllowed,
      connector::hive::SubfieldFilters& filters);

  /// Create the subfield filter based on the constructed filter info.
  /// inputName: the input name of a column.
  template <TypeKind KIND, typename FilterType>
  void constructSubfieldFilters(
      uint32_t colIdx,
      const std::string& inputName,
      const std::shared_ptr<FilterInfo>& filterInfo,
      connector::hive::SubfieldFilters& filters);

  /// Construct subfield filters according to the pre-set map of filter info.
  connector::hive::SubfieldFilters mapToFilters(
      const std::vector<std::string>& inputNameList,
      const std::vector<TypePtr>& inputTypeList,
      std::unordered_map<uint32_t, std::shared_ptr<FilterInfo>> colInfoMap);

  /// Convert subfield functions into subfieldFilters to
  /// be used in Hive Connector.
  connector::hive::SubfieldFilters toSubfieldFilters(
      const std::vector<std::string>& inputNameList,
      const std::vector<TypePtr>& inputTypeList,
      const std::vector<::substrait::Expression_ScalarFunction>&
          subfieldFunctions,
      const std::vector<::substrait::Expression_SingularOrList>&
          singularOrLists);

  /// Connect all remaining functions with 'and' relation
  /// for the use of remaingFilter in Hive Connector.
  core::TypedExprPtr connectWithAnd(
      std::vector<std::string> inputNameList,
      std::vector<TypePtr> inputTypeList,
      const std::vector<::substrait::Expression_ScalarFunction>&
          remainingFunctions,
      const std::vector<::substrait::Expression_SingularOrList>&
          singularOrLists);

  /// Connect the left and right expressions with 'and' relation.
  core::TypedExprPtr connectWithAnd(
      core::TypedExprPtr leftExpr,
      core::TypedExprPtr rightExpr);

  /// Set the phase of Aggregation.
  void setPhase(
      const ::substrait::AggregateRel& sAgg,
      core::AggregationNode::Step& aggStep);

  /// Used to convert AggregateRel into Velox plan node.
  /// The output of child node will be used as the input of Aggregation.
  std::shared_ptr<const core::PlanNode> toVeloxAgg(
      const ::substrait::AggregateRel& sAgg,
      const std::shared_ptr<const core::PlanNode>& childNode,
      const core::AggregationNode::Step& aggStep);

  /// Helper function to convert the input of Substrait Rel to Velox Node.
  template <typename T>
  core::PlanNodePtr convertSingleInput(T rel) {
    VELOX_CHECK(rel.has_input(), "Child Rel is expected here.");
    return toVeloxPlan(rel.input());
  }

  /// The unique identification for each PlanNode.
  int planNodeId_ = 0;

  /// The map storing the relations between the function id and the function
  /// name. Will be constructed based on the Substrait representation.
  std::unordered_map<uint64_t, std::string> functionMap_;

  /// The map storing the split stats for each PlanNode.
  std::unordered_map<core::PlanNodeId, std::shared_ptr<SplitInfo>>
      splitInfoMap_;

  /// The map storing the pre-built plan nodes which can be accessed through
  /// index. This map is only used when the computation of a Substrait plan
  /// depends on other input nodes.
  std::unordered_map<uint64_t, std::shared_ptr<const core::PlanNode>>
      inputNodesMap_;

  /// The Substrait parser used to convert Substrait representations into
  /// recognizable representations.
  std::shared_ptr<SubstraitParser> subParser_{
      std::make_shared<SubstraitParser>()};

  /// The Expression converter used to convert Substrait representations into
  /// Velox expressions.
  std::shared_ptr<SubstraitVeloxExprConverter> exprConverter_;

  /// Memory pool.
  memory::MemoryPool* pool_;

  /// A flag used to specify validation.
  bool validationMode_ = false;
};

} // namespace facebook::velox::substrait
