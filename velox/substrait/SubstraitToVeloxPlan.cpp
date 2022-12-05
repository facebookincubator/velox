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

#include "velox/substrait/SubstraitToVeloxPlan.h"

#include <google/protobuf/wrappers.pb.h>

#include "velox/substrait/TypeUtils.h"
#include "velox/substrait/VariantToVectorConverter.h"
#include "velox/type/Type.h"

namespace facebook::velox::substrait {
namespace {

core::SortOrder toSortOrder(const ::substrait::SortField& sortField) {
  switch (sortField.direction()) {
    case ::substrait::SortField_SortDirection_SORT_DIRECTION_ASC_NULLS_FIRST:
      return core::kAscNullsFirst;
    case ::substrait::SortField_SortDirection_SORT_DIRECTION_ASC_NULLS_LAST:
      return core::kAscNullsLast;
    case ::substrait::SortField_SortDirection_SORT_DIRECTION_DESC_NULLS_FIRST:
      return core::kDescNullsFirst;
    case ::substrait::SortField_SortDirection_SORT_DIRECTION_DESC_NULLS_LAST:
      return core::kDescNullsLast;
    default:
      VELOX_FAIL("Sort direction is not supported.");
  }
}

} // namespace

template <typename T>
// Get the lowest value for numeric type.
T getLowest() {
  return std::numeric_limits<T>::lowest();
};

// Get the lowest value for string.
template <>
std::string getLowest<std::string>() {
  return "";
};

// Get the max value for numeric type.
template <typename T>
T getMax() {
  return std::numeric_limits<T>::max();
};

// The max value will be used in BytesRange. Return empty string here instead.
template <>
std::string getMax<std::string>() {
  return "";
};

// Substrait function names.
const std::string sIsNotNull = "is_not_null";
const std::string sGte = "gte";
const std::string sGt = "gt";
const std::string sLte = "lte";
const std::string sLt = "lt";
const std::string sEqual = "equal";
const std::string sOr = "or";
const std::string sNot = "not";

// Substrait types.
const std::string sI32 = "i32";
const std::string sI64 = "i64";

/// @brief Return whether a config is set as true in AdvancedExtension
/// optimization.
/// @param extension Substrait advanced extension.
/// @param config the key string of a config.
/// @return Whether the config is set as true.
bool configSetInOptimization(
    const ::substrait::extensions::AdvancedExtension& extension,
    const std::string& config) {
  if (extension.has_optimization()) {
    google::protobuf::StringValue msg;
    extension.optimization().UnpackTo(&msg);
    std::size_t pos = msg.value().find(config);
    if ((pos != std::string::npos) &&
        (msg.value().substr(pos + config.size(), 1) == "1")) {
      return true;
    }
  }
  return false;
}

/// @brief Get the input type from both sides of join.
/// @param leftNode the plan node of left side.
/// @param rightNode the plan node of right side.
/// @return the input type.
RowTypePtr getJoinInputType(
    const core::PlanNodePtr& leftNode,
    const core::PlanNodePtr& rightNode) {
  auto outputSize =
      leftNode->outputType()->size() + rightNode->outputType()->size();
  std::vector<std::string> outputNames;
  std::vector<std::shared_ptr<const Type>> outputTypes;
  outputNames.reserve(outputSize);
  outputTypes.reserve(outputSize);
  for (const auto& node : {leftNode, rightNode}) {
    const auto& names = node->outputType()->names();
    outputNames.insert(outputNames.end(), names.begin(), names.end());
    const auto& types = node->outputType()->children();
    outputTypes.insert(outputTypes.end(), types.begin(), types.end());
  }
  return std::make_shared<const RowType>(
      std::move(outputNames), std::move(outputTypes));
}

/// @brief Get the direct output type of join.
/// @param leftNode the plan node of left side.
/// @param rightNode the plan node of right side.
/// @param joinType the join type.
/// @return the output type.
RowTypePtr getJoinOutputType(
    const core::PlanNodePtr& leftNode,
    const core::PlanNodePtr& rightNode,
    const core::JoinType& joinType) {
  // Decide output type.
  // Output of right semi join cannot include columns from the left side.
  bool outputMayIncludeLeftColumns =
      !(core::isRightSemiFilterJoin(joinType) ||
        core::isRightSemiProjectJoin(joinType));

  // Output of left semi and anti joins cannot include columns from the right
  // side.
  bool outputMayIncludeRightColumns =
      !(core::isLeftSemiFilterJoin(joinType) ||
        core::isLeftSemiProjectJoin(joinType) || core::isAntiJoin(joinType) ||
        core::isNullAwareAntiJoin(joinType));

  if (outputMayIncludeLeftColumns && outputMayIncludeRightColumns) {
    return getJoinInputType(leftNode, rightNode);
  }

  if (outputMayIncludeLeftColumns) {
    if (core::isLeftSemiProjectJoin(joinType)) {
      auto outputSize = leftNode->outputType()->size() + 1;
      std::vector<std::string> outputNames = leftNode->outputType()->names();
      std::vector<std::shared_ptr<const Type>> outputTypes =
          leftNode->outputType()->children();
      outputNames.emplace_back("exists");
      outputTypes.emplace_back(BOOLEAN());
      return std::make_shared<const RowType>(
          std::move(outputNames), std::move(outputTypes));
    } else {
      return leftNode->outputType();
    }
  }

  if (outputMayIncludeRightColumns) {
    if (core::isRightSemiProjectJoin(joinType)) {
      auto outputSize = rightNode->outputType()->size() + 1;
      std::vector<std::string> outputNames = rightNode->outputType()->names();
      std::vector<std::shared_ptr<const Type>> outputTypes =
          rightNode->outputType()->children();
      outputNames.emplace_back("exists");
      outputTypes.emplace_back(BOOLEAN());
      return std::make_shared<const RowType>(
          std::move(outputNames), std::move(outputTypes));
    } else {
      return rightNode->outputType();
    }
  }
  VELOX_FAIL("Output should include left or right columns.");
}
} // namespace

core::AggregationNode::Step SubstraitVeloxPlanConverter::toAggregationStep(
    const ::substrait::AggregateRel& sAgg) {
  if (sAgg.measures().size() == 0) {
    // When only groupings exist, set the phase to be Single.
    return core::AggregationNode::Step::kSingle;
  }

  // Use the first measure to set aggregation phase.
  const auto& firstMeasure = sAgg.measures()[0];
  const auto& aggFunction = firstMeasure.measure();
  switch (aggFunction.phase()) {
    case ::substrait::AGGREGATION_PHASE_INITIAL_TO_INTERMEDIATE:
      return core::AggregationNode::Step::kPartial;
    case ::substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_INTERMEDIATE:
      return core::AggregationNode::Step::kIntermediate;
    case ::substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_RESULT:
      return core::AggregationNode::Step::kFinal;
    case ::substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT:
      return core::AggregationNode::Step::kSingle;
    default:
      VELOX_FAIL("Aggregate phase is not supported.");
  }
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::JoinRel& sJoin) {
  if (!sJoin.has_left()) {
    VELOX_FAIL("Left Rel is expected in JoinRel.");
  }
  if (!sJoin.has_right()) {
    VELOX_FAIL("Right Rel is expected in JoinRel.");
  }

  auto leftNode = toVeloxPlan(sJoin.left());
  auto rightNode = toVeloxPlan(sJoin.right());

  // Map join type.
  core::JoinType joinType;
  switch (sJoin.type()) {
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_INNER:
      joinType = core::JoinType::kInner;
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_OUTER:
      joinType = core::JoinType::kFull;
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_LEFT:
      joinType = core::JoinType::kLeft;
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_RIGHT:
      joinType = core::JoinType::kRight;
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_LEFT_SEMI:
      // Determine the semi join type based on extracted information.
      if (sJoin.has_advanced_extension() &&
          configSetInOptimization(
              sJoin.advanced_extension(), "isExistenceJoin=")) {
        joinType = core::JoinType::kLeftSemiProject;
      } else {
        joinType = core::JoinType::kLeftSemiFilter;
      }
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_RIGHT_SEMI:
      // Determine the semi join type based on extracted information.
      if (sJoin.has_advanced_extension() &&
          configSetInOptimization(
              sJoin.advanced_extension(), "isExistenceJoin=")) {
        joinType = core::JoinType::kRightSemiProject;
      } else {
        joinType = core::JoinType::kRightSemiFilter;
      }
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_ANTI: {
      // Determine the anti join type based on extracted information.
      if (sJoin.has_advanced_extension() &&
          configSetInOptimization(
              sJoin.advanced_extension(), "isNullAwareAntiJoin=")) {
        joinType = core::JoinType::kNullAwareAnti;
      } else {
        joinType = core::JoinType::kAnti;
      }
      break;
    }
    default:
      VELOX_NYI("Unsupported Join type: {}", sJoin.type());
  }

  // extract join keys from join expression
  std::vector<const ::substrait::Expression::FieldReference*> leftExprs,
      rightExprs;
  extractJoinKeys(sJoin.expression(), leftExprs, rightExprs);
  VELOX_CHECK_EQ(leftExprs.size(), rightExprs.size());
  size_t numKeys = leftExprs.size();

  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> leftKeys,
      rightKeys;
  leftKeys.reserve(numKeys);
  rightKeys.reserve(numKeys);
  auto inputRowType = getJoinInputType(leftNode, rightNode);
  for (size_t i = 0; i < numKeys; ++i) {
    leftKeys.emplace_back(
        exprConverter_->toVeloxExpr(*leftExprs[i], inputRowType));
    rightKeys.emplace_back(
        exprConverter_->toVeloxExpr(*rightExprs[i], inputRowType));
  }

  std::shared_ptr<const core::ITypedExpr> filter;
  if (sJoin.has_post_join_filter()) {
    filter =
        exprConverter_->toVeloxExpr(sJoin.post_join_filter(), inputRowType);
  }

  // Create join node
  return std::make_shared<core::HashJoinNode>(
      nextPlanNodeId(),
      joinType,
      leftKeys,
      rightKeys,
      filter,
      leftNode,
      rightNode,
      getJoinOutputType(leftNode, rightNode, joinType));
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::AggregateRel& sAgg) {
  auto childNode = convertSingleInput<::substrait::AggregateRel>(aggRel);
  core::AggregationNode::Step aggStep = toAggregationStep(sAgg);
  return toVeloxAgg(sAgg, childNode, aggStep);
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxAgg(
    const ::substrait::AggregateRel& sAgg,
    const std::shared_ptr<const core::PlanNode>& childNode,
    const core::AggregationNode::Step& aggStep) {
  const auto& inputType = childNode->outputType();
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      veloxGroupingExprs;

  // Get the grouping expressions.
  for (const auto& grouping : sAgg.groupings()) {
    for (const auto& groupingExpr : grouping.grouping_expressions()) {
      // Velox's groupings are limited to be Field.
      veloxGroupingExprs.emplace_back(
          exprConverter_->toVeloxExpr(groupingExpr.selection(), inputType));
    }
  }

  // Parse measures and get the aggregate expressions.
  // Each measure represents one aggregate expression.
  std::vector<core::CallTypedExprPtr> aggExprs;
  aggExprs.reserve(sAgg.measures().size());
  std::vector<core::FieldAccessTypedExprPtr> aggregateMasks;
  aggregateMasks.reserve(sAgg.measures().size());
  for (const auto& smea : sAgg.measures()) {
    core::FieldAccessTypedExprPtr aggregateMask;
    ::substrait::Expression substraitAggMask = smea.filter();
    // Get Aggregation Masks.
    if (smea.has_filter()) {
      if (substraitAggMask.ByteSizeLong() == 0) {
        aggregateMask = {};
      } else {
        aggregateMask =
            std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                exprConverter_->toVeloxExpr(substraitAggMask, inputType));
      }
      aggregateMasks.push_back(aggregateMask);
    }
    const auto& aggFunction = smea.measure();
    std::string funcName = subParser_->findVeloxFunction(
        functionMap_, aggFunction.function_reference());
    std::vector<std::shared_ptr<const core::ITypedExpr>> aggParams;
    aggParams.reserve(aggFunction.arguments().size());
    for (const auto& arg : aggFunction.arguments()) {
      aggParams.emplace_back(
          exprConverter_->toVeloxExpr(arg.value(), inputType));
    }
    auto aggVeloxType =
        toVeloxType(subParser_->parseType(aggFunction.output_type())->type);
    auto aggExpr = std::make_shared<const core::CallTypedExpr>(
        aggVeloxType, std::move(aggParams), funcName);
    aggExprs.emplace_back(aggExpr);
  }

  bool ignoreNullKeys = false;
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      preGroupingExprs = {};

  // Get the output names of Aggregation.
  std::vector<std::string> aggOutNames;
  aggOutNames.reserve(sAgg.measures().size());
  for (int idx = veloxGroupingExprs.size();
       idx < veloxGroupingExprs.size() + sAgg.measures().size();
       idx++) {
    aggOutNames.emplace_back(subParser_->makeNodeName(planNodeId_, idx));
  }

  // Create Aggregate node.
  auto aggNode = std::make_shared<core::AggregationNode>(
      nextPlanNodeId(),
      aggStep,
      veloxGroupingExprs,
      preGroupingExprs,
      aggOutNames,
      aggExprs,
      aggregateMasks,
      ignoreNullKeys,
      childNode);
  return aggNode;
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ProjectRel& projectRel) {
  auto childNode = convertSingleInput<::substrait::ProjectRel>(projectRel);
  // Construct Velox Expressions.
  const auto& projectExprs = sProject.expressions();
  std::vector<std::string> projectNames;
  std::vector<core::TypedExprPtr> expressions;
  projectNames.reserve(projectExprs.size());
  expressions.reserve(projectExprs.size());

  const auto& inputType = childNode->outputType();
  int colIdx = 0;
  for (const auto& expr : projectExprs) {
    expressions.emplace_back(exprConverter_->toVeloxExpr(expr, inputType));
    projectNames.emplace_back(subParser_->makeNodeName(planNodeId_, colIdx));
    colIdx += 1;
  }

  return std::make_shared<core::ProjectNode>(
      nextPlanNodeId(),
      std::move(projectNames),
      std::move(expressions),
      childNode);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ExpandRel& expandRel) {
  core::PlanNodePtr childNode;
  if (expandRel.has_input()) {
    childNode = toVeloxPlan(expandRel.input());
  } else {
    VELOX_FAIL("Child Rel is expected in ExpandRel.");
  }

  const auto& inputType = childNode->outputType();

  std::vector<std::vector<core::FieldAccessTypedExprPtr>> groupingSetExprs;
  groupingSetExprs.reserve(expandRel.groupings_size());

  for (const auto& grouping : expandRel.groupings()) {
    std::vector<core::FieldAccessTypedExprPtr> groupingExprs;
    groupingExprs.reserve(grouping.groupsets_expressions_size());

    for (const auto& groupingExpr : grouping.groupsets_expressions()) {
      auto expression =
          exprConverter_->toVeloxExpr(groupingExpr.selection(), inputType);
      auto expr_field =
          dynamic_cast<const core::FieldAccessTypedExpr*>(expression.get());
      VELOX_CHECK(
          expr_field != nullptr,
          " the group set key in Expand Operator only support field")

      groupingExprs.emplace_back(
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
              expression));
    }
    groupingSetExprs.emplace_back(groupingExprs);
  }

  std::vector<core::GroupIdNode::GroupingKeyInfo> groupingKeyInfos;
  std::set<std::string> names;
  auto index = 0;
  for (const auto& groupingSet : groupingSetExprs) {
    for (const auto& groupingKey : groupingSet) {
      if (names.find(groupingKey->name()) == names.end()) {
        core::GroupIdNode::GroupingKeyInfo keyInfos;
        keyInfos.output = groupingKey->name();
        keyInfos.input = groupingKey;
        groupingKeyInfos.push_back(keyInfos);
      }
      names.insert(groupingKey->name());
    }
  }

  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> aggExprs;

  for (const auto& aggExpr : expandRel.aggregate_expressions()) {
    auto expression = exprConverter_->toVeloxExpr(aggExpr, inputType);
    auto expr_field =
        dynamic_cast<const core::FieldAccessTypedExpr*>(expression.get());
    VELOX_CHECK(
        expr_field != nullptr,
        " the agg key in Expand Operator only support field");
    auto filed =
        std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expression);
    aggExprs.emplace_back(filed);
  }

  return std::make_shared<core::GroupIdNode>(
      nextPlanNodeId(),
      groupingSetExprs,
      std::move(groupingKeyInfos),
      aggExprs,
      std::move(expandRel.group_name()),
      childNode);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::SortRel& sortRel) {
  auto childNode = convertSingleInput<::substrait::SortRel>(sortRel);

  auto [sortingKeys, sortingOrders] =
      processSortField(sortRel.sorts(), childNode->outputType());

  return std::make_shared<core::OrderByNode>(
      nextPlanNodeId(),
      sortingKeys,
      sortingOrders,
      false /*isPartial*/,
      childNode);
}

std::pair<
    std::vector<core::FieldAccessTypedExprPtr>,
    std::vector<core::SortOrder>>
SubstraitVeloxPlanConverter::processSortField(
    const ::google::protobuf::RepeatedPtrField<::substrait::SortField>&
        sortFields,
    const RowTypePtr& inputType) {
  std::vector<core::FieldAccessTypedExprPtr> sortingKeys;
  std::vector<core::SortOrder> sortingOrders;
  sortingKeys.reserve(sortFields.size());
  sortingOrders.reserve(sortFields.size());

  for (const auto& sort : sortFields) {
    sortingOrders.emplace_back(toSortOrder(sort));

    if (sort.has_expr()) {
      auto expression = exprConverter_->toVeloxExpr(sort.expr(), inputType);
      auto fieldExpr =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
              expression);
      VELOX_CHECK_NOT_NULL(
          fieldExpr, " the sorting key in Sort Operator only support field");
      sortingKeys.emplace_back(fieldExpr);
    }
  }
  return {sortingKeys, sortingOrders};
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::FilterRel& filterRel) {
  auto childNode = convertSingleInput<::substrait::FilterRel>(filterRel);
  const auto& inputType = childNode->outputType();
  const auto& sExpr = filterRel.condition();

  return std::make_shared<core::FilterNode>(
      nextPlanNodeId(),
      exprConverter_->toVeloxExpr(sExpr, inputType),
      childNode);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::FetchRel& fetchRel) {
  core::PlanNodePtr childNode;
  if (fetchRel.has_input()) {
    childNode = toVeloxPlan(fetchRel.input());
  } else {
    VELOX_FAIL("Child Rel is expected in FetchRel.");
  }

  return std::make_shared<core::LimitNode>(
      nextPlanNodeId(),
      (int32_t)fetchRel.offset(),
      (int32_t)fetchRel.count(),
      false /*isPartial*/,
      childNode);
}

bool isPushDownSupportedByFormat(
    const dwio::common::FileFormat& format,
    connector::hive::SubfieldFilters& subfieldFilters) {
  switch (format) {
    case dwio::common::FileFormat::PARQUET:
    case dwio::common::FileFormat::ORC:
    case dwio::common::FileFormat::DWRF:
    case dwio::common::FileFormat::RC:
    case dwio::common::FileFormat::RC_TEXT:
    case dwio::common::FileFormat::RC_BINARY:
    case dwio::common::FileFormat::TEXT:
    case dwio::common::FileFormat::JSON:
    case dwio::common::FileFormat::ALPHA:
    case dwio::common::FileFormat::UNKNOWN:
    default:
      break;
  }
  return true;
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::FetchRel& fetchRel) {
  core::PlanNodePtr childNode;
  // Check the input of fetchRel, if it's sortRel, convert them into
  // topNNode. otherwise, to limitNode.
  ::substrait::SortRel sortRel;
  bool topNFlag;
  if (fetchRel.has_input()) {
    topNFlag = fetchRel.input().has_sort();
    if (topNFlag) {
      sortRel = fetchRel.input().sort();
      childNode = toVeloxPlan(sortRel.input());
    } else {
      childNode = toVeloxPlan(fetchRel.input());
    }
  } else {
    VELOX_FAIL("Child Rel is expected in FetchRel.");
  }

  if (topNFlag) {
    auto [sortingKeys, sortingOrders] =
        processSortField(sortRel.sorts(), childNode->outputType());

    VELOX_CHECK_EQ(fetchRel.offset(), 0);

    return std::make_shared<core::TopNNode>(
        nextPlanNodeId(),
        sortingKeys,
        sortingOrders,
        (int32_t)fetchRel.count(),
        false /*isPartial*/,
        childNode);

  } else {
    return std::make_shared<core::LimitNode>(
        nextPlanNodeId(),
        (int32_t)fetchRel.offset(),
        (int32_t)fetchRel.count(),
        false /*isPartial*/,
        childNode);
  }
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ReadRel& sRead) {
  // Check if the ReadRel specifies an input of stream. If yes, the pre-built
  // input node will be used as the data source.
  auto splitInfo = std::make_shared<SplitInfo>();
  auto streamIdx = streamIsInput(sRead);
  if (streamIdx >= 0) {
    if (inputNodesMap_.find(streamIdx) == inputNodesMap_.end()) {
      VELOX_FAIL(
          "Could not find source index {} in input nodes map.", streamIdx);
    }
    auto streamNode = inputNodesMap_[streamIdx];
    splitInfo->isStream = true;
    splitInfoMap_[streamNode->id()] = splitInfo;
    return streamNode;
  }

  // Otherwise, will create TableScan node for ReadRel.
  // Get output names and types.
  std::vector<std::string> colNameList;
  std::vector<TypePtr> veloxTypeList;
  std::vector<bool> isPartitionColumns;
  if (sRead.has_base_schema()) {
    const auto& baseSchema = sRead.base_schema();
    colNameList.reserve(baseSchema.names().size());
    for (const auto& name : baseSchema.names()) {
      colNameList.emplace_back(name);
    }
    auto substraitTypeList = subParser_->parseNamedStruct(baseSchema);
    isPartitionColumns = subParser_->parsePartitionColumns(baseSchema);
    veloxTypeList.reserve(substraitTypeList.size());
    for (const auto& substraitType : substraitTypeList) {
      veloxTypeList.emplace_back(toVeloxType(substraitType->type));
    }
  }

  // Parse local files and construct split info.
  if (sRead.has_local_files()) {
    using SubstraitFileFormatCase =
        ::substrait::ReadRel_LocalFiles_FileOrFiles::FileFormatCase;
    const auto& fileList = sRead.local_files().items();
    splitInfo->paths.reserve(fileList.size());
    splitInfo->starts.reserve(fileList.size());
    splitInfo->lengths.reserve(fileList.size());
    for (const auto& file : fileList) {
      // Expect all Partitions share the same index.
      splitInfo->partitionIndex = file.partition_index();
      splitInfo->paths.emplace_back(file.uri_file());
      splitInfo->starts.emplace_back(file.start());
      splitInfo->lengths.emplace_back(file.length());
      switch (file.file_format_case()) {
        case SubstraitFileFormatCase::kOrc:
        case SubstraitFileFormatCase::kDwrf:
          splitInfo->format = dwio::common::FileFormat::DWRF;
          break;
        case SubstraitFileFormatCase::kParquet:
          splitInfo->format = dwio::common::FileFormat::PARQUET;
          break;
        default:
          splitInfo->format = dwio::common::FileFormat::UNKNOWN;
      }
    }
  }
  // Do not hard-code connector ID and allow for connectors other than Hive.
  static const std::string kHiveConnectorId = "test-hive";

  // Velox requires Filter Pushdown must being enabled.
  bool filterPushdownEnabled = true;
  std::shared_ptr<connector::hive::HiveTableHandle> tableHandle;
  if (!sRead.has_filter()) {
    tableHandle = std::make_shared<connector::hive::HiveTableHandle>(
        kHiveConnectorId,
        "hive_table",
        filterPushdownEnabled,
        connector::hive::SubfieldFilters{},
        nullptr);
  } else {
    // Flatten the conditions connected with 'and'.
    std::vector<::substrait::Expression_ScalarFunction> scalarFunctions;
    std::vector<::substrait::Expression_SingularOrList> singularOrLists;
    std::vector<::substrait::Expression_IfThen> ifThens;
    flattenConditions(
        sRead.filter(), scalarFunctions, singularOrLists, ifThens);

    std::unordered_map<uint32_t, std::shared_ptr<RangeRecorder>> rangeRecorders;
    for (uint32_t idx = 0; idx < veloxTypeList.size(); idx++) {
      rangeRecorders[idx] = std::make_shared<RangeRecorder>();
    }

    // Separate the filters to be two parts. The subfield part can be
    // pushed down.
    std::vector<::substrait::Expression_ScalarFunction> subfieldFunctions;
    std::vector<::substrait::Expression_SingularOrList> subfieldrOrLists;

    std::vector<::substrait::Expression_ScalarFunction> remainingFunctions;
    std::vector<::substrait::Expression_SingularOrList> remainingrOrLists;

    separateFilters(
        rangeRecorders,
        scalarFunctions,
        subfieldFunctions,
        remainingFunctions,
        singularOrLists,
        subfieldrOrLists,
        remainingrOrLists);

    // Create subfield filters based on the constructed filter info map.
    connector::hive::SubfieldFilters subfieldFilters = toSubfieldFilters(
        colNameList, veloxTypeList, subfieldFunctions, subfieldrOrLists);
    // Connect the remaining filters with 'and'.
    std::shared_ptr<const core::ITypedExpr> remainingFilter;

    if (!isPushDownSupportedByFormat(splitInfo->format, subfieldFilters)) {
      // A subfieldFilter is not supported by the format,
      // mark all filter as remaining filters.
      subfieldFilters.clear();
      remainingFilter = connectWithAnd(
          colNameList,
          veloxTypeList,
          scalarFunctions,
          singularOrLists,
          ifThens);
    } else {
      remainingFilter = connectWithAnd(
          colNameList,
          veloxTypeList,
          remainingFunctions,
          remainingrOrLists,
          ifThens);
    }

    tableHandle = std::make_shared<connector::hive::HiveTableHandle>(
        kHiveConnectorId,
        "hive_table",
        filterPushdownEnabled,
        std::move(subfieldFilters),
        remainingFilter);
  }

  // Get assignments and out names.
  std::vector<std::string> outNames;
  outNames.reserve(colNameList.size());
  std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
      assignments;
  for (int idx = 0; idx < colNameList.size(); idx++) {
    auto outName = subParser_->makeNodeName(planNodeId_, idx);
    auto columnType = isPartitionColumns[idx]
        ? connector::hive::HiveColumnHandle::ColumnType::kPartitionKey
        : connector::hive::HiveColumnHandle::ColumnType::kRegular;
    assignments[outName] = std::make_shared<connector::hive::HiveColumnHandle>(
        colNameList[idx], columnType, veloxTypeList[idx]);
    outNames.emplace_back(outName);
  }
  auto outputType = ROW(std::move(outNames), std::move(veloxTypeList));

  if (sRead.has_virtual_table()) {
    return toVeloxPlan(sRead, outputType);
  } else {
    auto tableScanNode = std::make_shared<core::TableScanNode>(
        nextPlanNodeId(), outputType, tableHandle, assignments);
    // Set split info map.
    splitInfoMap_[tableScanNode->id()] = splitInfo;
    return tableScanNode;
  }
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ReadRel& readRel,
    const RowTypePtr& type) {
  ::substrait::ReadRel_VirtualTable readVirtualTable = readRel.virtual_table();
  int64_t numVectors = readVirtualTable.values_size();
  int64_t numColumns = type->size();
  int64_t valueFieldNums =
      readVirtualTable.values(numVectors - 1).fields_size();
  std::vector<RowVectorPtr> vectors;
  vectors.reserve(numVectors);

  int64_t batchSize = valueFieldNums / numColumns;

  for (int64_t index = 0; index < numVectors; ++index) {
    std::vector<VectorPtr> children;
    ::substrait::Expression_Literal_Struct rowValue =
        readRel.virtual_table().values(index);
    auto fieldSize = rowValue.fields_size();
    VELOX_CHECK_EQ(fieldSize, batchSize * numColumns);

    for (int64_t col = 0; col < numColumns; ++col) {
      const TypePtr& outputChildType = type->childAt(col);
      std::vector<variant> batchChild;
      batchChild.reserve(batchSize);
      for (int64_t batchId = 0; batchId < batchSize; batchId++) {
        // each value in the batch
        auto fieldIdx = col * batchSize + batchId;
        ::substrait::Expression_Literal field = rowValue.fields(fieldIdx);

        auto expr = exprConverter_->toVeloxExpr(field);
        if (auto constantExpr =
                std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                    expr)) {
          if (!constantExpr->hasValueVector()) {
            batchChild.emplace_back(constantExpr->value());
          } else {
            VELOX_UNSUPPORTED(
                "Values node with complex type values is not supported yet");
          }
        } else {
          VELOX_FAIL("Expected constant expression");
        }
      }
      children.emplace_back(
          setVectorFromVariants(outputChildType, batchChild, pool_));
    }
    vectors.emplace_back(
        std::make_shared<RowVector>(pool_, type, nullptr, batchSize, children));
  }
  return std::make_shared<core::ValuesNode>(nextPlanNodeId(), vectors);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::Rel& sRel) {
  if (sRel.has_aggregate()) {
    return toVeloxPlan(sRel.aggregate());
  }
  if (sRel.has_project()) {
    return toVeloxPlan(sRel.project());
  }
  if (sRel.has_filter()) {
    return toVeloxPlan(sRel.filter());
  }
  if (sRel.has_join()) {
    return toVeloxPlan(sRel.join());
  }
  if (sRel.has_read()) {
    return toVeloxPlan(sRel.read());
  }
  if (sRel.has_sort()) {
    return toVeloxPlan(sRel.sort());
  }
  if (rel.has_fetch()) {
    return toVeloxPlan(rel.fetch());
  }
  if (rel.has_sort()) {
    return toVeloxPlan(rel.sort());
  }
  if (sRel.has_expand()) {
    return toVeloxPlan(sRel.expand());
  }
  if (sRel.has_fetch()) {
    return toVeloxPlan(sRel.fetch());
  }
  VELOX_NYI("Substrait conversion not supported for Rel.");
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::RelRoot& sRoot) {
  // TODO: Use the names as the output names for the whole computing.
  const auto& sNames = sRoot.names();
  if (sRoot.has_input()) {
    const auto& sRel = sRoot.input();
    return toVeloxPlan(sRel);
  }
  VELOX_FAIL("Input is expected in RelRoot.");
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::Plan& substraitPlan) {
  VELOX_CHECK(
      checkTypeExtension(substraitPlan),
      "The type extension only have unknown type.")
  // Construct the function map based on the Substrait representation,
  // and initialize the expression converter with it.
  constructFunctionMap(substraitPlan);

  // In fact, only one RelRoot or Rel is expected here.
  VELOX_CHECK_EQ(substraitPlan.relations_size(), 1);
  const auto& sRel = substraitPlan.relations(0);
  if (sRel.has_root()) {
    return toVeloxPlan(sRel.root());
  }
  if (sRel.has_rel()) {
    return toVeloxPlan(sRel.rel());
  }
  VELOX_FAIL("RelRoot or Rel is expected in Plan.");
}

void SubstraitVeloxPlanConverter::constructFunctionMap(
    const ::substrait::Plan& substraitPlan) {
  // Construct the function map based on the Substrait representation.
  for (const auto& sExtension : substraitPlan.extensions()) {
    if (!sExtension.has_extension_function()) {
      continue;
    }
    const auto& sFmap = sExtension.extension_function();
    auto id = sFmap.function_anchor();
    auto name = sFmap.name();
    functionMap_[id] = name;
  }
  exprConverter_ =
      std::make_shared<SubstraitVeloxExprConverter>(pool_, functionMap_);
}

std::string SubstraitVeloxPlanConverter::nextPlanNodeId() {
  auto id = fmt::format("{}", planNodeId_);
  planNodeId_++;
  return id;
}

void SubstraitVeloxPlanConverter::flattenConditions(
    const ::substrait::Expression& substraitFilter,
    std::vector<::substrait::Expression_ScalarFunction>& scalarFunctions,
    std::vector<::substrait::Expression_SingularOrList>& singularOrLists,
    std::vector<::substrait::Expression_IfThen>& ifThens) {
  auto typeCase = substraitFilter.rex_type_case();
  switch (typeCase) {
    case ::substrait::Expression::RexTypeCase::kScalarFunction: {
      auto sFunc = substraitFilter.scalar_function();
      auto filterNameSpec = subParser_->findSubstraitFuncSpec(
          functionMap_, sFunc.function_reference());
      // TODO: Only and relation is supported here.
      if (subParser_->getSubFunctionName(filterNameSpec) == "and") {
        for (const auto& sCondition : sFunc.arguments()) {
          flattenConditions(
              sCondition.value(), scalarFunctions, singularOrLists, ifThens);
        }
      } else {
        scalarFunctions.emplace_back(sFunc);
      }
      break;
    }
    case ::substrait::Expression::RexTypeCase::kSingularOrList: {
      singularOrLists.emplace_back(substraitFilter.singular_or_list());
      break;
    }
    case ::substrait::Expression::RexTypeCase::kIfThen: {
      ifThens.emplace_back(substraitFilter.if_then());
      break;
    }
    default:
      VELOX_NYI("GetFlatConditions not supported for type '{}'", typeCase);
  }
}

std::string SubstraitVeloxPlanConverter::findFuncSpec(uint64_t id) {
  return subParser_->findSubstraitFuncSpec(functionMap_, id);
}

int32_t SubstraitVeloxPlanConverter::streamIsInput(
    const ::substrait::ReadRel& sRead) {
  if (sRead.has_local_files()) {
    const auto& fileList = sRead.local_files().items();
    if (fileList.size() == 0) {
      VELOX_FAIL("At least one file path is expected.");
    }

    // The stream input will be specified with the format of
    // "iterator:${index}".
    std::string filePath = fileList[0].uri_file();
    std::string prefix = "iterator:";
    std::size_t pos = filePath.find(prefix);
    if (pos == std::string::npos) {
      return -1;
    }

    // Get the index.
    std::string idxStr = filePath.substr(pos + prefix.size(), filePath.size());
    try {
      return stoi(idxStr);
    } catch (const std::exception& err) {
      VELOX_FAIL(err.what());
    }
  }
  if (validationMode_) {
    return -1;
  }
  VELOX_FAIL("Local file is expected.");
}

void SubstraitVeloxPlanConverter::extractJoinKeys(
    const ::substrait::Expression& joinExpression,
    std::vector<const ::substrait::Expression::FieldReference*>& leftExprs,
    std::vector<const ::substrait::Expression::FieldReference*>& rightExprs) {
  std::vector<const ::substrait::Expression*> expressions;
  expressions.push_back(&joinExpression);
  while (!expressions.empty()) {
    auto visited = expressions.back();
    expressions.pop_back();
    if (visited->rex_type_case() ==
        ::substrait::Expression::RexTypeCase::kScalarFunction) {
      const auto& funcName =
          subParser_->getSubFunctionName(subParser_->findVeloxFunction(
              functionMap_, visited->scalar_function().function_reference()));
      const auto& args = visited->scalar_function().arguments();
      if (funcName == "and") {
        expressions.push_back(&args[0].value());
        expressions.push_back(&args[1].value());
      } else if (funcName == "eq" || funcName == "equalto") {
        VELOX_CHECK(std::all_of(
            args.cbegin(),
            args.cend(),
            [](const ::substrait::FunctionArgument& arg) {
              return arg.value().has_selection();
            }));
        leftExprs.push_back(&args[0].value().selection());
        rightExprs.push_back(&args[1].value().selection());
      } else {
        VELOX_NYI("Join condition {} not supported.", funcName);
      }
    } else {
      VELOX_FAIL(
          "Unable to parse from join expression: {}",
          joinExpression.DebugString());
    }
  }
}

connector::hive::SubfieldFilters SubstraitVeloxPlanConverter::toSubfieldFilters(
    const std::vector<std::string>& inputNameList,
    const std::vector<TypePtr>& inputTypeList,
    const std::vector<::substrait::Expression_ScalarFunction>& scalarFunctions,
    const std::vector<::substrait::Expression_SingularOrList>&
        singularOrLists) {
  std::unordered_map<uint32_t, std::shared_ptr<FilterInfo>> colInfoMap;
  // A map between the column index and the FilterInfo.
  for (uint32_t idx = 0; idx < inputTypeList.size(); idx++) {
    colInfoMap[idx] = std::make_shared<FilterInfo>();
  }

  // Construct the FilterInfo for the related column.
  for (const auto& scalarFunction : scalarFunctions) {
    auto filterNameSpec = subParser_->findSubstraitFuncSpec(
        functionMap_, scalarFunction.function_reference());
    auto filterName = subParser_->getSubFunctionName(filterNameSpec);
    if (filterName == sNot) {
      VELOX_CHECK(scalarFunction.arguments().size() == 1);
      auto expr = scalarFunction.arguments()[0].value();
      if (expr.has_scalar_function()) {
        // Set its chid to filter info with reverse enabled.
        setFilterMap(
            scalarFunction.arguments()[0].value().scalar_function(),
            inputTypeList,
            colInfoMap,
            true);
      } else {
        // TODO: support push down of Not In.
        VELOX_NYI("Scalar function expected.");
      }
      continue;
    }

    if (filterName == sOr) {
      VELOX_CHECK(scalarFunction.arguments().size() == 2);
      VELOX_CHECK(std::all_of(
          scalarFunction.arguments().cbegin(),
          scalarFunction.arguments().cend(),
          [](const ::substrait::FunctionArgument& arg) {
            return arg.value().has_scalar_function() ||
                arg.value().has_singular_or_list();
          }));
      // Set the chidren functions to filter info. They should be
      // effective to the same field.
      for (const auto& arg : scalarFunction.arguments()) {
        auto expr = arg.value();
        if (expr.has_scalar_function()) {
          setFilterMap(
              arg.value().scalar_function(), inputTypeList, colInfoMap);
        } else if (expr.has_singular_or_list()) {
          setSingularListValues(expr.singular_or_list(), colInfoMap);
        } else {
          VELOX_NYI("Scalar function or SingularOrList expected.");
        }
      }
      continue;
    }

    setFilterMap(scalarFunction, inputTypeList, colInfoMap);
  }

  for (const auto& list : singularOrLists) {
    setSingularListValues(list, colInfoMap);
  }
  return mapToFilters(inputNameList, inputTypeList, colInfoMap);
}

bool SubstraitVeloxPlanConverter::fieldOrWithLiteral(
    const ::google::protobuf::RepeatedPtrField<::substrait::FunctionArgument>&
        arguments,
    uint32_t& fieldIndex) {
  if (arguments.size() == 1) {
    if (arguments[0].value().has_selection()) {
      // Only field exists.
      fieldIndex = subParser_->parseReferenceSegment(
          arguments[0].value().selection().direct_reference());
      return true;
    } else {
      return false;
    }
  }

  if (arguments.size() != 2) {
    // Not the field and literal combination.
    return false;
  }
  bool fieldExists = false;
  bool literalExists = false;
  for (const auto& param : arguments) {
    auto typeCase = param.value().rex_type_case();
    switch (typeCase) {
      case ::substrait::Expression::RexTypeCase::kSelection:
        fieldIndex = subParser_->parseReferenceSegment(
            param.value().selection().direct_reference());
        fieldExists = true;
        break;
      case ::substrait::Expression::RexTypeCase::kLiteral:
        literalExists = true;
        break;
      default:
        break;
    }
  }
  // Whether the field and literal both exist.
  return fieldExists && literalExists;
}

bool SubstraitVeloxPlanConverter::chidrenFunctionsOnSameField(
    const ::substrait::Expression_ScalarFunction& function) {
  // Get the column indices of the chidren functions.
  std::vector<int32_t> colIndices;
  for (const auto& arg : function.arguments()) {
    if (arg.value().has_scalar_function()) {
      auto scalarFunction = arg.value().scalar_function();
      for (const auto& param : scalarFunction.arguments()) {
        if (param.value().has_selection()) {
          auto field = param.value().selection();
          VELOX_CHECK(field.has_direct_reference());
          int32_t colIdx =
              subParser_->parseReferenceSegment(field.direct_reference());
          colIndices.emplace_back(colIdx);
        }
      }
    } else if (arg.value().has_singular_or_list()) {
      auto singularOrList = arg.value().singular_or_list();
      int32_t colIdx = getColumnIndexFromSingularOrList(singularOrList);
      colIndices.emplace_back(colIdx);
    } else {
      return false;
    }
  }

  if (std::all_of(colIndices.begin(), colIndices.end(), [&](uint32_t idx) {
        return idx == colIndices[0];
      })) {
    // All indices are the same.
    return true;
  }
  return false;
}

bool SubstraitVeloxPlanConverter::canPushdownCommonFunction(
    const ::substrait::Expression_ScalarFunction& scalarFunction,
    const std::string& filterName,
    uint32_t& fieldIdx) {
  // Condtions can be pushed down.
  std::unordered_set<std::string> supportedCommonFunctions = {
      sIsNotNull, sGte, sGt, sLte, sLt, sEqual};

  bool canPushdown = false;
  if (supportedCommonFunctions.find(filterName) !=
          supportedCommonFunctions.end() &&
      fieldOrWithLiteral(scalarFunction.arguments(), fieldIdx)) {
    // The arg should be field or field with literal.
    canPushdown = true;
  }
  return canPushdown;
}

bool SubstraitVeloxPlanConverter::canPushdownNot(
    const ::substrait::Expression_ScalarFunction& scalarFunction,
    const std::unordered_map<uint32_t, std::shared_ptr<RangeRecorder>>&
        rangeRecorders) {
  VELOX_CHECK(
      scalarFunction.arguments().size() == 1,
      "Only one arg is expected for Not.");
  auto notArg = scalarFunction.arguments()[0];
  if (!notArg.value().has_scalar_function()) {
    // Not for a Boolean Literal or Or List is not supported curretly.
    // It can be pushed down with an AlwaysTrue or AlwaysFalse Range.
    return false;
  }

  auto argFunction = subParser_->findSubstraitFuncSpec(
      functionMap_, notArg.value().scalar_function().function_reference());
  auto functionName = subParser_->getSubFunctionName(argFunction);

  std::unordered_set<std::string> supportedNotFunctions = {
      sGte, sGt, sLte, sLt, sEqual};

  uint32_t fieldIdx;
  bool isFieldOrWithLiteral = fieldOrWithLiteral(
      notArg.value().scalar_function().arguments(), fieldIdx);

  if (supportedNotFunctions.find(functionName) != supportedNotFunctions.end() &&
      isFieldOrWithLiteral &&
      rangeRecorders.at(fieldIdx)->setCertainRangeForFunction(
          functionName, true /*reverse*/)) {
    return true;
  }
  return false;
}

bool SubstraitVeloxPlanConverter::canPushdownOr(
    const ::substrait::Expression_ScalarFunction& scalarFunction,
    const std::unordered_map<uint32_t, std::shared_ptr<RangeRecorder>>&
        rangeRecorders) {
  // OR Conditon whose chidren functions are on different columns is not
  // supported to be pushed down.
  if (!chidrenFunctionsOnSameField(scalarFunction)) {
    return false;
  }

  std::unordered_set<std::string> supportedOrFunctions = {
      sIsNotNull, sGte, sGt, sLte, sLt, sEqual};

  for (const auto& arg : scalarFunction.arguments()) {
    if (arg.value().has_scalar_function()) {
      auto nameSpec = subParser_->findSubstraitFuncSpec(
          functionMap_, arg.value().scalar_function().function_reference());
      auto functionName = subParser_->getSubFunctionName(nameSpec);

      uint32_t fieldIdx;
      bool isFieldOrWithLiteral = fieldOrWithLiteral(
          arg.value().scalar_function().arguments(), fieldIdx);
      if (supportedOrFunctions.find(functionName) ==
              supportedOrFunctions.end() ||
          !isFieldOrWithLiteral ||
          !rangeRecorders.at(fieldIdx)->setCertainRangeForFunction(
              functionName, false /*reverse*/, true /*forOrRelation*/)) {
        // The arg should be field or field with literal.
        return false;
      }
    } else if (arg.value().has_singular_or_list()) {
      auto singularOrList = arg.value().singular_or_list();
      if (!canPushdownSingularOrList(singularOrList, true)) {
        return false;
      }
      uint32_t fieldIdx = getColumnIndexFromSingularOrList(singularOrList);
      // Disable IN pushdown for int-like types.
      if (!rangeRecorders.at(fieldIdx)->setInRange(true /*forOrRelation*/)) {
        return false;
      }
    } else {
      // Or relation betweeen other expressions is not supported to be pushded
      // down currently.
      return false;
    }
  }
  return true;
}

void SubstraitVeloxPlanConverter::separateFilters(
    const std::unordered_map<uint32_t, std::shared_ptr<RangeRecorder>>&
        rangeRecorders,
    const std::vector<::substrait::Expression_ScalarFunction>& scalarFunctions,
    std::vector<::substrait::Expression_ScalarFunction>& subfieldFunctions,
    std::vector<::substrait::Expression_ScalarFunction>& remainingFunctions,
    const std::vector<::substrait::Expression_SingularOrList>& singularOrLists,
    std::vector<::substrait::Expression_SingularOrList>& subfieldOrLists,
    std::vector<::substrait::Expression_SingularOrList>& remainingOrLists) {
  for (const auto& singularOrList : singularOrLists) {
    if (!canPushdownSingularOrList(singularOrList)) {
      remainingOrLists.emplace_back(singularOrList);
      continue;
    }
    uint32_t colIdx = getColumnIndexFromSingularOrList(singularOrList);
    if (rangeRecorders.at(colIdx)->setInRange()) {
      subfieldOrLists.emplace_back(singularOrList);
    } else {
      remainingOrLists.emplace_back(singularOrList);
    }
  }

  for (const auto& scalarFunction : scalarFunctions) {
    auto filterNameSpec = subParser_->findSubstraitFuncSpec(
        functionMap_, scalarFunction.function_reference());
    auto filterName = subParser_->getSubFunctionName(filterNameSpec);
    if (filterName != sNot && filterName != sOr) {
      // Check if the condition is supported to be pushed down.
      uint32_t fieldIdx;
      if (canPushdownCommonFunction(scalarFunction, filterName, fieldIdx) &&
          rangeRecorders.at(fieldIdx)->setCertainRangeForFunction(filterName)) {
        subfieldFunctions.emplace_back(scalarFunction);
      } else {
        remainingFunctions.emplace_back(scalarFunction);
      }
      continue;
    }

    // Check whether NOT and OR functions can be pushed down.
    // If yes, the scalar function will be added into the subfield functions.
    bool supported = false;
    if (filterName == sNot) {
      supported = canPushdownNot(scalarFunction, rangeRecorders);
    } else if (filterName == sOr) {
      supported = canPushdownOr(scalarFunction, rangeRecorders);
    }

    if (supported) {
      subfieldFunctions.emplace_back(scalarFunction);
    } else {
      remainingFunctions.emplace_back(scalarFunction);
    }
  }
}

bool SubstraitVeloxPlanConverter::RangeRecorder::setCertainRangeForFunction(
    const std::string& functionName,
    bool reverse,
    bool forOrRelation) {
  if (functionName == sLt || functionName == sLte) {
    if (reverse) {
      return setLeftBound(forOrRelation);
    } else {
      return setRightBound(forOrRelation);
    }
  }
  if (functionName == sGt || functionName == sGte) {
    if (reverse) {
      return setRightBound(forOrRelation);
    } else {
      return setLeftBound(forOrRelation);
    }
  }
  if (functionName == sEqual) {
    if (reverse) {
      // Not equal means lt or gt.
      return setMultiRange();
    } else {
      return setLeftBound(forOrRelation) && setRightBound(forOrRelation);
    }
  }
  if (functionName == sOr) {
    if (reverse) {
      // Not supported.
      return false;
    } else {
      return setMultiRange();
    }
  }
  if (functionName == sIsNotNull) {
    if (reverse) {
      // Not supported.
      return false;
    } else {
      // Is not null can always coexist with the other range.
      return true;
    }
  }
  return false;
}

template <typename T>
void SubstraitVeloxPlanConverter::setColInfoMap(
    const std::string& filterName,
    uint32_t colIdx,
    std::optional<variant> literalVariant,
    bool reverse,
    std::unordered_map<uint32_t, std::shared_ptr<FilterInfo>>& colInfoMap) {
  if (filterName == sIsNotNull) {
    if (reverse) {
      VELOX_NYI("Reverse not supported for filter name '{}'", filterName);
    }
    colInfoMap[colIdx]->forbidsNull();
    return;
  }

  if (filterName == sGte) {
    if (reverse) {
      colInfoMap[colIdx]->setUpper(literalVariant, true);
    } else {
      colInfoMap[colIdx]->setLower(literalVariant, false);
    }
    return;
  }

  if (filterName == sGt) {
    if (reverse) {
      colInfoMap[colIdx]->setUpper(literalVariant, false);
    } else {
      colInfoMap[colIdx]->setLower(literalVariant, true);
    }
    return;
  }

  if (filterName == sLte) {
    if (reverse) {
      colInfoMap[colIdx]->setLower(literalVariant, true);
    } else {
      colInfoMap[colIdx]->setUpper(literalVariant, false);
    }
    return;
  }

  if (filterName == sLt) {
    if (reverse) {
      colInfoMap[colIdx]->setLower(literalVariant, false);
    } else {
      colInfoMap[colIdx]->setUpper(literalVariant, true);
    }
    return;
  }

  if (filterName == sEqual) {
    if (reverse) {
      colInfoMap[colIdx]->setNotValue(literalVariant);
    } else {
      colInfoMap[colIdx]->setLower(literalVariant, false);
      colInfoMap[colIdx]->setUpper(literalVariant, false);
    }
    return;
  }
  VELOX_NYI("SetColInfoMap not supported for filter name '{}'", filterName);
}

void SubstraitVeloxPlanConverter::setFilterMap(
    const ::substrait::Expression_ScalarFunction& scalarFunction,
    const std::vector<TypePtr>& inputTypeList,
    std::unordered_map<uint32_t, std::shared_ptr<FilterInfo>>& colInfoMap,
    bool reverse) {
  auto nameSpec = subParser_->findSubstraitFuncSpec(
      functionMap_, scalarFunction.function_reference());
  auto functionName = subParser_->getSubFunctionName(nameSpec);

  // Extract the column index and column bound from the scalar function.
  std::optional<uint32_t> colIdx;
  std::optional<::substrait::Expression_Literal> substraitLit;
  for (const auto& param : scalarFunction.arguments()) {
    auto typeCase = param.value().rex_type_case();
    switch (typeCase) {
      case ::substrait::Expression::RexTypeCase::kSelection:
        colIdx = subParser_->parseReferenceSegment(
            param.value().selection().direct_reference());
        break;
      case ::substrait::Expression::RexTypeCase::kLiteral:
        substraitLit = param.value().literal();
        break;
      default:
        VELOX_NYI(
            "Substrait conversion not supported for arg type '{}'", typeCase);
    }
  }
  if (!colIdx.has_value()) {
    VELOX_NYI("Column index is expected in subfield filters creation.");
  }

  // Set the extracted bound to the specific column.
  uint32_t colIdxVal = colIdx.value();
  auto inputType = inputTypeList[colIdxVal];
  std::optional<variant> val;
  switch (inputType->kind()) {
    case TypeKind::INTEGER:
      if (substraitLit) {
        val = variant(substraitLit.value().i32());
      }
      setColInfoMap<int>(functionName, colIdxVal, val, reverse, colInfoMap);
      break;
    case TypeKind::BIGINT:
      if (substraitLit) {
        val = variant(substraitLit.value().i64());
      }
      setColInfoMap<int64_t>(functionName, colIdxVal, val, reverse, colInfoMap);
      break;
    case TypeKind::DOUBLE:
      if (substraitLit) {
        val = variant(substraitLit.value().fp64());
      }
      setColInfoMap<double>(functionName, colIdxVal, val, reverse, colInfoMap);
      break;
    case TypeKind::VARCHAR:
      if (substraitLit) {
        val = variant(substraitLit.value().string());
      }
      setColInfoMap<std::string>(
          functionName, colIdxVal, val, reverse, colInfoMap);
      break;
    default:
      VELOX_NYI(
          "Subfield filters creation not supported for input type '{}'",
          inputType);
  }
}

template <TypeKind KIND, typename FilterType>
void SubstraitVeloxPlanConverter::createNotEqualFilter(
    variant notVariant,
    bool nullAllowed,
    std::vector<std::unique_ptr<FilterType>>& colFilters) {
  using NativeType = typename RangeTraits<KIND>::NativeType;
  using RangeType = typename RangeTraits<KIND>::RangeType;

  // Value > lower
  std::unique_ptr<FilterType> lowerFilter = std::make_unique<RangeType>(
      notVariant.value<NativeType>(), /*lower*/
      false, /*lowerUnbounded*/
      true, /*lowerExclusive*/
      getMax<NativeType>(), /*upper*/
      true, /*upperUnbounded*/
      false, /*upperExclusive*/
      nullAllowed); /*nullAllowed*/
  colFilters.emplace_back(std::move(lowerFilter));

  // Value < upper
  std::unique_ptr<FilterType> upperFilter = std::make_unique<RangeType>(
      getLowest<NativeType>(), /*lower*/
      true, /*lowerUnbounded*/
      false, /*lowerExclusive*/
      notVariant.value<NativeType>(), /*upper*/
      false, /*upperUnbounded*/
      true, /*upperExclusive*/
      nullAllowed); /*nullAllowed*/
  colFilters.emplace_back(std::move(upperFilter));
}

template <TypeKind KIND>
void SubstraitVeloxPlanConverter::setInFilter(
    const std::vector<variant>& variants,
    bool nullAllowed,
    const std::string& inputName,
    connector::hive::SubfieldFilters& filters) {}

template <>
void SubstraitVeloxPlanConverter::setInFilter<TypeKind::DOUBLE>(
    const std::vector<variant>& variants,
    bool nullAllowed,
    const std::string& inputName,
    connector::hive::SubfieldFilters& filters) {
  std::vector<double> values;
  values.reserve(variants.size());
  for (const auto& variant : variants) {
    double value = variant.value<double>();
    values.emplace_back(value);
  }
  filters[common::Subfield(inputName)] =
      common::createDoubleValues(values, nullAllowed);
}

template <>
void SubstraitVeloxPlanConverter::setInFilter<TypeKind::BIGINT>(
    const std::vector<variant>& variants,
    bool nullAllowed,
    const std::string& inputName,
    connector::hive::SubfieldFilters& filters) {
  std::vector<int64_t> values;
  values.reserve(variants.size());
  for (const auto& variant : variants) {
    int64_t value = variant.value<int64_t>();
    values.emplace_back(value);
  }
  filters[common::Subfield(inputName)] =
      common::createBigintValues(values, nullAllowed);
}

template <>
void SubstraitVeloxPlanConverter::setInFilter<TypeKind::INTEGER>(
    const std::vector<variant>& variants,
    bool nullAllowed,
    const std::string& inputName,
    connector::hive::SubfieldFilters& filters) {
  // Use bigint values for int type.
  std::vector<int64_t> values;
  values.reserve(variants.size());
  for (const auto& variant : variants) {
    // Use the matched type to get value from variant.
    int64_t value = variant.value<int32_t>();
    values.emplace_back(value);
  }
  filters[common::Subfield(inputName)] =
      common::createBigintValues(values, nullAllowed);
}

template <>
void SubstraitVeloxPlanConverter::setInFilter<TypeKind::VARCHAR>(
    const std::vector<variant>& variants,
    bool nullAllowed,
    const std::string& inputName,
    connector::hive::SubfieldFilters& filters) {
  std::vector<std::string> values;
  values.reserve(variants.size());
  for (const auto& variant : variants) {
    std::string value = variant.value<std::string>();
    values.emplace_back(value);
  }
  filters[common::Subfield(inputName)] =
      std::make_unique<common::BytesValues>(values, nullAllowed);
}

template <TypeKind KIND, typename FilterType>
void SubstraitVeloxPlanConverter::setSubfieldFilter(
    std::vector<std::unique_ptr<FilterType>> colFilters,
    const std::string& inputName,
    bool nullAllowed,
    connector::hive::SubfieldFilters& filters) {
  using MultiRangeType = typename RangeTraits<KIND>::MultiRangeType;

  if (colFilters.size() == 1) {
    filters[common::Subfield(inputName)] = std::move(colFilters[0]);
  } else if (colFilters.size() > 1) {
    // BigintMultiRange should have been sorted
    if (colFilters[0]->kind() == common::FilterKind::kBigintRange) {
      std::sort(
          colFilters.begin(),
          colFilters.end(),
          [](const auto& a, const auto& b) {
            return dynamic_cast<common::BigintRange*>(a.get())->lower() <
                dynamic_cast<common::BigintRange*>(b.get())->lower();
          });
    }
    filters[common::Subfield(inputName)] =
        std::make_unique<MultiRangeType>(std::move(colFilters), nullAllowed);
  }
}

template <TypeKind KIND, typename FilterType>
void SubstraitVeloxPlanConverter::constructSubfieldFilters(
    uint32_t colIdx,
    const std::string& inputName,
    const std::shared_ptr<FilterInfo>& filterInfo,
    connector::hive::SubfieldFilters& filters) {
  using NativeType = typename RangeTraits<KIND>::NativeType;
  using RangeType = typename RangeTraits<KIND>::RangeType;
  using MultiRangeType = typename RangeTraits<KIND>::MultiRangeType;

  if (!filterInfo->isInitialized()) {
    return;
  }

  uint32_t rangeSize = std::max(
      filterInfo->lowerBounds_.size(), filterInfo->upperBounds_.size());
  bool nullAllowed = filterInfo->nullAllowed_;

  // Handle 'in' filter.
  if (filterInfo->valuesVector_.size() > 0) {
    // To filter out null is a default behaviour of Spark IN expression.
    nullAllowed = false;
    setInFilter<KIND>(
        filterInfo->valuesVector_, nullAllowed, inputName, filters);
    // Currently, In cannot coexist with other filter conditions
    // due to multirange is in 'OR' relation but 'AND' is needed.
    VELOX_CHECK(
        rangeSize == 0,
        "LowerBounds or upperBounds conditons cannot be supported after IN filter.");
    VELOX_CHECK(
        !filterInfo->notValue_.has_value(),
        "Not equal cannot be supported after IN filter.");
    return;
  }

  // Construct the Filters.
  std::vector<std::unique_ptr<FilterType>> colFilters;

  // Handle not(equal) filter.
  if (filterInfo->notValue_) {
    variant notVariant = filterInfo->notValue_.value();
    createNotEqualFilter<KIND, FilterType>(
        notVariant, filterInfo->nullAllowed_, colFilters);
    // Currently, Not-equal cannot coexist with other filter conditions
    // due to multirange is in 'OR' relation but 'AND' is needed.
    VELOX_CHECK(
        rangeSize == 0,
        "LowerBounds or upperBounds conditons cannot be supported after not-equal filter.");
    filters[common::Subfield(inputName)] =
        std::make_unique<MultiRangeType>(std::move(colFilters), nullAllowed);
    return;
  }

  // Handle null filtering.
  if (rangeSize == 0 && !nullAllowed) {
    std::unique_ptr<common::IsNotNull> filter =
        std::make_unique<common::IsNotNull>();
    filters[common::Subfield(inputName)] = std::move(filter);
    return;
  }

  // Handle other filter ranges.
  NativeType lowerBound = getLowest<NativeType>();
  NativeType upperBound = getMax<NativeType>();
  bool lowerUnbounded = true;
  bool upperUnbounded = true;
  bool lowerExclusive = false;
  bool upperExclusive = false;

  for (uint32_t idx = 0; idx < rangeSize; idx++) {
    if (idx < filterInfo->lowerBounds_.size() &&
        filterInfo->lowerBounds_[idx]) {
      lowerUnbounded = false;
      variant lowerVariant = filterInfo->lowerBounds_[idx].value();
      lowerBound = lowerVariant.value<NativeType>();
      lowerExclusive = filterInfo->lowerExclusives_[idx];
    }
    if (idx < filterInfo->upperBounds_.size() &&
        filterInfo->upperBounds_[idx]) {
      upperUnbounded = false;
      variant upperVariant = filterInfo->upperBounds_[idx].value();
      upperBound = upperVariant.value<NativeType>();
      upperExclusive = filterInfo->upperExclusives_[idx];
    }
    std::unique_ptr<FilterType> filter = std::make_unique<RangeType>(
        lowerBound,
        lowerUnbounded,
        lowerExclusive,
        upperBound,
        upperUnbounded,
        upperExclusive,
        nullAllowed);
    colFilters.emplace_back(std::move(filter));
  }

  // Set the SubfieldFilter.
  setSubfieldFilter<KIND, FilterType>(
      std::move(colFilters), inputName, filterInfo->nullAllowed_, filters);
}

bool SubstraitVeloxPlanConverter::checkTypeExtension(
    const ::substrait::Plan& substraitPlan) {
  for (const auto& sExtension : substraitPlan.extensions()) {
    if (!sExtension.has_extension_type()) {
      continue;
    }

    // Only support UNKNOWN type in UserDefined type extension.
    if (sExtension.extension_type().name() != "UNKNOWN") {
      return false;
    }
  }
  return true;
}

connector::hive::SubfieldFilters SubstraitVeloxPlanConverter::mapToFilters(
    const std::vector<std::string>& inputNameList,
    const std::vector<TypePtr>& inputTypeList,
    std::unordered_map<uint32_t, std::shared_ptr<FilterInfo>> colInfoMap) {
  // Construct the subfield filters based on the filter info map.
  connector::hive::SubfieldFilters filters;
  for (uint32_t colIdx = 0; colIdx < inputNameList.size(); colIdx++) {
    auto inputType = inputTypeList[colIdx];
    switch (inputType->kind()) {
      case TypeKind::INTEGER:
        constructSubfieldFilters<TypeKind::INTEGER, common::BigintRange>(
            colIdx, inputNameList[colIdx], colInfoMap[colIdx], filters);
        break;
      case TypeKind::BIGINT:
        constructSubfieldFilters<TypeKind::BIGINT, common::BigintRange>(
            colIdx, inputNameList[colIdx], colInfoMap[colIdx], filters);
        break;
      case TypeKind::DOUBLE:
        constructSubfieldFilters<TypeKind::DOUBLE, common::Filter>(
            colIdx, inputNameList[colIdx], colInfoMap[colIdx], filters);
        break;
      case TypeKind::VARCHAR:
        constructSubfieldFilters<TypeKind::VARCHAR, common::Filter>(
            colIdx, inputNameList[colIdx], colInfoMap[colIdx], filters);
        break;
      default:
        VELOX_NYI(
            "Subfield filters creation not supported for input type '{}'",
            inputType);
    }
  }
  return filters;
}

core::TypedExprPtr SubstraitVeloxPlanConverter::connectWithAnd(
    std::vector<std::string> inputNameList,
    std::vector<TypePtr> inputTypeList,
    const std::vector<::substrait::Expression_ScalarFunction>& scalarFunctions,
    const std::vector<::substrait::Expression_SingularOrList>& singularOrLists,
    const std::vector<::substrait::Expression_IfThen>& ifThens) {
  if (scalarFunctions.size() == 0 && singularOrLists.size() == 0 &&
      ifThens.size() == 0) {
    return nullptr;
  }
  auto inputType = ROW(std::move(inputNameList), std::move(inputTypeList));

  // Filter for scalar functions.
  std::vector<std::shared_ptr<const core::ITypedExpr>> allFilters;
  for (auto scalar : scalarFunctions) {
    auto filter = exprConverter_->toVeloxExpr(scalar, inputType);
    if (filter != nullptr) {
      allFilters.emplace_back(filter);
    }
  }
  for (auto orList : singularOrLists) {
    auto filter = exprConverter_->toVeloxExpr(orList, inputType);
    if (filter != nullptr) {
      allFilters.emplace_back(filter);
    }
  }
  for (auto ifThen : ifThens) {
    auto filter = exprConverter_->toVeloxExpr(ifThen, inputType);
    if (filter != nullptr) {
      allFilters.emplace_back(filter);
    }
  }
  VELOX_CHECK_GT(allFilters.size(), 0, "One filter should be valid.")
  std::shared_ptr<const core::ITypedExpr> andFilter = allFilters[0];
  for (auto i = 1; i < allFilters.size(); i++) {
    andFilter = connectWithAnd(andFilter, allFilters[i]);
  }
  return andFilter;
}

core::TypedExprPtr SubstraitVeloxPlanConverter::connectWithAnd(
    core::TypedExprPtr leftExpr,
    core::TypedExprPtr rightExpr) {
  std::vector<core::TypedExprPtr> params;
  params.reserve(2);
  params.emplace_back(leftExpr);
  params.emplace_back(rightExpr);
  return std::make_shared<const core::CallTypedExpr>(
      BOOLEAN(), std::move(params), "and");
}

bool SubstraitVeloxPlanConverter::canPushdownSingularOrList(
    const ::substrait::Expression_SingularOrList& singularOrList,
    bool disableIntLike) {
  VELOX_CHECK(
      singularOrList.options_size() > 0, "At least one option is expected.");
  // Check whether the value is field.
  bool hasField = singularOrList.value().has_selection();
  auto options = singularOrList.options();
  for (const auto& option : options) {
    VELOX_CHECK(option.has_literal(), "Literal is expected as option.");
    auto type = option.literal().literal_type_case();
    // Only BigintValues and BytesValues are supported.
    if (type != ::substrait::Expression_Literal::LiteralTypeCase::kI32 &&
        type != ::substrait::Expression_Literal::LiteralTypeCase::kI64 &&
        type != ::substrait::Expression_Literal::LiteralTypeCase::kString) {
      return false;
    }
    // BigintMultiRange can only accept BigintRange, so disableIntLike is set to
    // true for OR pushdown of int-like types.
    if (disableIntLike &&
        (type == ::substrait::Expression_Literal::LiteralTypeCase::kI32 ||
         type == ::substrait::Expression_Literal::LiteralTypeCase::kI64)) {
      return false;
    }
  }
  return hasField;
}

uint32_t SubstraitVeloxPlanConverter::getColumnIndexFromSingularOrList(
    const ::substrait::Expression_SingularOrList& singularOrList) {
  // Get the column index.
  ::substrait::Expression_FieldReference selection;
  if (singularOrList.value().has_scalar_function()) {
    selection = singularOrList.value()
                    .scalar_function()
                    .arguments()[0]
                    .value()
                    .selection();
  } else if (singularOrList.value().has_selection()) {
    selection = singularOrList.value().selection();
  } else {
    VELOX_FAIL("Unsupported type in IN pushdown.");
  }
  return subParser_->parseReferenceSegment(selection.direct_reference());
}

void SubstraitVeloxPlanConverter::setSingularListValues(
    const ::substrait::Expression_SingularOrList& singularOrList,
    std::unordered_map<uint32_t, std::shared_ptr<FilterInfo>>& colInfoMap) {
  VELOX_CHECK(
      singularOrList.options_size() > 0, "At least one option is expected.");
  // Get the column index.
  uint32_t colIdx = getColumnIndexFromSingularOrList(singularOrList);

  // Get the value list.
  auto options = singularOrList.options();
  std::vector<variant> variants;
  variants.reserve(options.size());
  for (const auto& option : options) {
    VELOX_CHECK(option.has_literal(), "Literal is expected as option.");
    variants.emplace_back(
        exprConverter_->toVeloxExpr(option.literal())->value());
  }
  // Set the value list to filter info.
  colInfoMap[colIdx]->setValues(variants);
}

} // namespace facebook::velox::substrait
