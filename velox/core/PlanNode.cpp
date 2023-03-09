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
#include "velox/core/PlanNode.h"

namespace facebook::velox::core {

const SortOrder kAscNullsFirst(true, true);
const SortOrder kAscNullsLast(true, false);
const SortOrder kDescNullsFirst(false, true);
const SortOrder kDescNullsLast(false, false);

namespace {
const std::vector<PlanNodePtr> kEmptySources;

RowTypePtr getAggregationOutputType(
    const std::vector<FieldAccessTypedExprPtr>& groupingKeys,
    const std::vector<std::string>& aggregateNames,
    const std::vector<CallTypedExprPtr>& aggregates) {
  VELOX_CHECK_EQ(
      aggregateNames.size(),
      aggregates.size(),
      "Number of aggregate names must be equal to number of aggregates");

  std::vector<std::string> names;
  std::vector<TypePtr> types;

  for (auto& key : groupingKeys) {
    auto field =
        std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(key);
    VELOX_CHECK(field, "Grouping key must be a field reference");
    names.push_back(field->name());
    types.push_back(field->type());
  }

  for (int32_t i = 0; i < aggregateNames.size(); i++) {
    names.push_back(aggregateNames[i]);
    types.push_back(aggregates[i]->type());
  }

  return std::make_shared<RowType>(std::move(names), std::move(types));
}
} // namespace

AggregationNode::AggregationNode(
    const PlanNodeId& id,
    Step step,
    const std::vector<FieldAccessTypedExprPtr>& groupingKeys,
    const std::vector<FieldAccessTypedExprPtr>& preGroupedKeys,
    const std::vector<std::string>& aggregateNames,
    const std::vector<CallTypedExprPtr>& aggregates,
    const std::vector<FieldAccessTypedExprPtr>& aggregateMasks,
    bool ignoreNullKeys,
    PlanNodePtr source)
    : PlanNode(id),
      step_(step),
      groupingKeys_(groupingKeys),
      preGroupedKeys_(preGroupedKeys),
      aggregateNames_(aggregateNames),
      aggregates_(aggregates),
      aggregateMasks_(aggregateMasks),
      ignoreNullKeys_(ignoreNullKeys),
      sources_{source},
      outputType_(getAggregationOutputType(
          groupingKeys_,
          aggregateNames_,
          aggregates_)) {
  // Empty grouping keys are used in global aggregation:
  //    SELECT sum(c) FROM t
  // Empty aggregates are used in distinct:
  //    SELECT distinct(b, c) FROM t GROUP BY a
  VELOX_CHECK(
      !groupingKeys_.empty() || !aggregates_.empty(),
      "Aggregation must specify either grouping keys or aggregates");

  std::unordered_set<std::string> groupingKeyNames;
  groupingKeyNames.reserve(groupingKeys.size());
  for (const auto& key : groupingKeys) {
    groupingKeyNames.insert(key->name());
  }

  for (const auto& key : preGroupedKeys) {
    VELOX_CHECK_EQ(
        1,
        groupingKeyNames.count(key->name()),
        "Pre-grouped key must be one of the grouping keys: {}.",
        key->name());
  }
}

namespace {
void addFields(
    std::stringstream& stream,
    const std::vector<FieldAccessTypedExprPtr>& keys) {
  for (auto i = 0; i < keys.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << keys[i]->name();
  }
}

void addKeys(std::stringstream& stream, const std::vector<TypedExprPtr>& keys) {
  for (auto i = 0; i < keys.size(); ++i) {
    const auto& expr = keys[i];
    if (i > 0) {
      stream << ", ";
    }
    if (auto field =
            std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
      stream << field->name();
    } else if (
        auto constant =
            std::dynamic_pointer_cast<const core::ConstantTypedExpr>(expr)) {
      stream << constant->toString();
    } else {
      stream << expr->toString();
    }
  }
}
} // namespace

void AggregationNode::addDetails(std::stringstream& stream) const {
  stream << stepName(step_) << " ";

  if (!groupingKeys_.empty()) {
    stream << "[";
    addFields(stream, groupingKeys_);
    stream << "] ";
  }

  for (auto i = 0; i < aggregateNames_.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << aggregateNames_[i] << " := " << aggregates_[i]->toString();
    if (aggregateMasks_.size() > i && aggregateMasks_[i]) {
      stream << " mask: " << aggregateMasks_[i]->name();
    }
  }
}

namespace {
RowTypePtr getGroupIdOutputType(
    const std::vector<GroupIdNode::GroupingKeyInfo>& groupingKeyInfos,
    const std::vector<FieldAccessTypedExprPtr>& aggregationInputs,
    const std::string& groupIdName) {
  // Grouping keys come first, followed by aggregation inputs and groupId
  // column.

  auto numOutputs = groupingKeyInfos.size() + aggregationInputs.size() + 1;

  std::vector<std::string> names;
  std::vector<TypePtr> types;

  names.reserve(numOutputs);
  types.reserve(numOutputs);

  for (const auto& groupingKeyInfo : groupingKeyInfos) {
    names.push_back(groupingKeyInfo.output);
    types.push_back(groupingKeyInfo.input->type());
  }

  for (const auto& input : aggregationInputs) {
    names.push_back(input->name());
    types.push_back(input->type());
  }

  names.push_back(groupIdName);
  types.push_back(BIGINT());

  return ROW(std::move(names), std::move(types));
}
} // namespace

GroupIdNode::GroupIdNode(
    PlanNodeId id,
    std::vector<std::vector<FieldAccessTypedExprPtr>> groupingSets,
    std::vector<GroupIdNode::GroupingKeyInfo> groupingKeyInfos,
    std::vector<FieldAccessTypedExprPtr> aggregationInputs,
    std::string groupIdName,
    PlanNodePtr source)
    : PlanNode(std::move(id)),
      sources_{source},
      outputType_(getGroupIdOutputType(
          groupingKeyInfos,
          aggregationInputs,
          groupIdName)),
      groupingSets_(std::move(groupingSets)),
      groupingKeyInfos_(std::move(groupingKeyInfos)),
      aggregationInputs_(std::move(aggregationInputs)),
      groupIdName_(std::move(groupIdName)) {
  VELOX_CHECK_GE(
      groupingSets_.size(),
      2,
      "GroupIdNode requires two or more grouping sets.");
}

void GroupIdNode::addDetails(std::stringstream& stream) const {
  for (auto i = 0; i < groupingSets_.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << "[";
    addFields(stream, groupingSets_[i]);
    stream << "]";
  }
}

const std::vector<PlanNodePtr>& ValuesNode::sources() const {
  return kEmptySources;
}

void ValuesNode::addDetails(std::stringstream& stream) const {
  vector_size_t totalCount = 0;
  for (const auto& vector : values_) {
    totalCount += vector->size();
  }
  stream << totalCount << " rows in " << values_.size() << " vectors";
}

void ProjectNode::addDetails(std::stringstream& stream) const {
  stream << "expressions: ";
  for (auto i = 0; i < projections_.size(); i++) {
    auto& projection = projections_[i];
    if (i > 0) {
      stream << ", ";
    }
    stream << "(" << names_[i] << ":" << projection->type()->toString() << ", "
           << projection->toString() << ")";
  }
}

const std::vector<PlanNodePtr>& TableScanNode::sources() const {
  return kEmptySources;
}

void TableScanNode::addDetails(std::stringstream& stream) const {
  stream << tableHandle_->toString();
}

const std::vector<PlanNodePtr>& ArrowStreamNode::sources() const {
  return kEmptySources;
}

void ArrowStreamNode::addDetails(std::stringstream& stream) const {
  // Nothing to add.
}

const std::vector<PlanNodePtr>& ExchangeNode::sources() const {
  return kEmptySources;
}

void ExchangeNode::addDetails(std::stringstream& /* stream */) const {
  // Nothing to add.
}

UnnestNode::UnnestNode(
    const PlanNodeId& id,
    std::vector<FieldAccessTypedExprPtr> replicateVariables,
    std::vector<FieldAccessTypedExprPtr> unnestVariables,
    const std::vector<std::string>& unnestNames,
    const std::optional<std::string>& ordinalityName,
    const PlanNodePtr& source)
    : PlanNode(id),
      replicateVariables_{std::move(replicateVariables)},
      unnestVariables_{std::move(unnestVariables)},
      withOrdinality_{ordinalityName.has_value()},
      sources_{source} {
  // Calculate output type. First come "replicate" columns, followed by
  // "unnest" columns, followed by an optional ordinality column.
  std::vector<std::string> names;
  std::vector<TypePtr> types;

  for (const auto& variable : replicateVariables_) {
    names.emplace_back(variable->name());
    types.emplace_back(variable->type());
  }

  int unnestIndex = 0;
  for (const auto& variable : unnestVariables_) {
    if (variable->type()->isArray()) {
      names.emplace_back(unnestNames[unnestIndex++]);
      types.emplace_back(variable->type()->asArray().elementType());
    } else if (variable->type()->isMap()) {
      const auto& mapType = variable->type()->asMap();

      names.emplace_back(unnestNames[unnestIndex++]);
      types.emplace_back(mapType.keyType());

      names.emplace_back(unnestNames[unnestIndex++]);
      types.emplace_back(mapType.valueType());
    } else {
      VELOX_FAIL(
          "Unexpected type of unnest variable. Expected ARRAY or MAP, but got {}.",
          variable->type()->toString());
    }
  }

  if (ordinalityName.has_value()) {
    names.emplace_back(ordinalityName.value());
    types.emplace_back(BIGINT());
  }
  outputType_ = ROW(std::move(names), std::move(types));
}

void UnnestNode::addDetails(std::stringstream& stream) const {
  addFields(stream, unnestVariables_);
}

AbstractJoinNode::AbstractJoinNode(
    const PlanNodeId& id,
    JoinType joinType,
    const std::vector<FieldAccessTypedExprPtr>& leftKeys,
    const std::vector<FieldAccessTypedExprPtr>& rightKeys,
    TypedExprPtr filter,
    PlanNodePtr left,
    PlanNodePtr right,
    const RowTypePtr outputType)
    : PlanNode(id),
      joinType_(joinType),
      leftKeys_(leftKeys),
      rightKeys_(rightKeys),
      filter_(std::move(filter)),
      sources_({std::move(left), std::move(right)}),
      outputType_(outputType) {
  VELOX_CHECK(!leftKeys_.empty(), "JoinNode requires at least one join key");
  VELOX_CHECK_EQ(
      leftKeys_.size(),
      rightKeys_.size(),
      "JoinNode requires same number of join keys on left and right sides");
  auto leftType = sources_[0]->outputType();
  for (auto key : leftKeys_) {
    VELOX_CHECK(
        leftType->containsChild(key->name()),
        "Left side join key not found in left side output: {}",
        key->name());
  }
  auto rightType = sources_[1]->outputType();
  for (auto key : rightKeys_) {
    VELOX_CHECK(
        rightType->containsChild(key->name()),
        "Right side join key not found in right side output: {}",
        key->name());
  }
  for (auto i = 0; i < leftKeys_.size(); ++i) {
    VELOX_CHECK_EQ(
        leftKeys_[i]->type()->kind(),
        rightKeys_[i]->type()->kind(),
        "Join key types on the left and right sides must match");
  }

  auto numOutputColumms = outputType_->size();
  if (core::isLeftSemiProjectJoin(joinType) ||
      core::isRightSemiProjectJoin(joinType)) {
    // Last output column must be a boolean 'match'.
    --numOutputColumms;
    VELOX_CHECK_EQ(outputType_->childAt(numOutputColumms), BOOLEAN());

    // Verify that 'match' column name doesn't match any column from left or
    // right source.
    const auto& name = outputType->nameOf(numOutputColumms);
    VELOX_CHECK(!leftType->containsChild(name));
    VELOX_CHECK(!rightType->containsChild(name));
  }

  // Output of right semi join cannot include columns from the left side.
  bool outputMayIncludeLeftColumns =
      !(core::isRightSemiFilterJoin(joinType) ||
        core::isRightSemiProjectJoin(joinType));

  // Output of left semi and anti joins cannot include columns from the right
  // side.
  bool outputMayIncludeRightColumns =
      !(core::isLeftSemiFilterJoin(joinType) ||
        core::isLeftSemiProjectJoin(joinType) || core::isAntiJoin(joinType));

  for (auto i = 0; i < numOutputColumms; ++i) {
    auto name = outputType_->nameOf(i);
    if (outputMayIncludeLeftColumns && leftType->containsChild(name)) {
      VELOX_CHECK(
          !rightType->containsChild(name),
          "Duplicate column name found on join's left and right sides: {}",
          name);
    } else if (outputMayIncludeRightColumns && rightType->containsChild(name)) {
      VELOX_CHECK(
          !leftType->containsChild(name),
          "Duplicate column name found on join's left and right sides: {}",
          name);
    } else {
      VELOX_FAIL(
          "Join's output column not found in either left or right sides: {}",
          name);
    }
  }
}

void AbstractJoinNode::addDetails(std::stringstream& stream) const {
  stream << joinTypeName(joinType_) << " ";

  for (auto i = 0; i < leftKeys_.size(); ++i) {
    if (i > 0) {
      stream << " AND ";
    }
    stream << leftKeys_[i]->name() << "=" << rightKeys_[i]->name();
  }

  if (filter_) {
    stream << ", filter: " << filter_->toString();
  }
}

void HashJoinNode::addDetails(std::stringstream& stream) const {
  AbstractJoinNode::addDetails(stream);
  if (nullAware_) {
    stream << ", null aware";
  }
}

CrossJoinNode::CrossJoinNode(
    const PlanNodeId& id,
    PlanNodePtr left,
    PlanNodePtr right,
    RowTypePtr outputType)
    : PlanNode(id),
      sources_({std::move(left), std::move(right)}),
      outputType_(std::move(outputType)) {}

void CrossJoinNode::addDetails(std::stringstream& /* stream */) const {
  // Nothing to add.
}

AssignUniqueIdNode::AssignUniqueIdNode(
    const PlanNodeId& id,
    const std::string& idName,
    const int32_t taskUniqueId,
    PlanNodePtr source)
    : PlanNode(id), taskUniqueId_(taskUniqueId), sources_{std::move(source)} {
  std::vector<std::string> names(sources_[0]->outputType()->names());
  std::vector<TypePtr> types(sources_[0]->outputType()->children());

  names.emplace_back(idName);
  types.emplace_back(BIGINT());
  outputType_ = ROW(std::move(names), std::move(types));
  uniqueIdCounter_ = std::make_shared<std::atomic_int64_t>();
}

void AssignUniqueIdNode::addDetails(std::stringstream& /* stream */) const {
  // Nothing to add.
}

namespace {
RowTypePtr getWindowOutputType(
    const RowTypePtr& inputType,
    const std::vector<std::string>& windowColumnNames,
    const std::vector<WindowNode::Function>& windowFunctions) {
  VELOX_CHECK_EQ(
      windowColumnNames.size(),
      windowFunctions.size(),
      "Number of window column names must be equal to number of window functions");

  std::vector<std::string> names = inputType->names();
  std::vector<TypePtr> types = inputType->children();

  for (int32_t i = 0; i < windowColumnNames.size(); i++) {
    names.push_back(windowColumnNames[i]);
    types.push_back(windowFunctions[i].functionCall->type());
  }
  return ROW(std::move(names), std::move(types));
}

const char* frameBoundString(const WindowNode::BoundType boundType) {
  switch (boundType) {
    case WindowNode::BoundType::kCurrentRow:
      return "CURRENT ROW";
    case WindowNode::BoundType::kPreceding:
      return "PRECEDING";
    case WindowNode::BoundType::kFollowing:
      return "FOLLOWING";
    case WindowNode::BoundType::kUnboundedPreceding:
      return "UNBOUNDED PRECEDING";
    case WindowNode::BoundType::kUnboundedFollowing:
      return "UNBOUNDED FOLLOWING";
  }
  VELOX_UNREACHABLE();
}

const char* windowTypeString(const WindowNode::WindowType windowType) {
  switch (windowType) {
    case WindowNode::WindowType::kRows:
      return "ROWS";
    case WindowNode::WindowType::kRange:
      return "RANGE";
  }
  VELOX_UNREACHABLE();
}

void addWindowFunction(
    std::stringstream& stream,
    const WindowNode::Function& windowFunction) {
  stream << windowFunction.functionCall->toString() << " ";
  auto frame = windowFunction.frame;
  if (frame.startType == WindowNode::BoundType::kUnboundedFollowing) {
    VELOX_USER_FAIL("Window frame start cannot be UNBOUNDED FOLLOWING");
  }
  if (frame.endType == WindowNode::BoundType::kUnboundedPreceding) {
    VELOX_USER_FAIL("Window frame end cannot be UNBOUNDED PRECEDING");
  }

  stream << windowTypeString(frame.type) << " between ";
  if (frame.startValue) {
    addKeys(stream, {frame.startValue});
    stream << " ";
  }
  stream << frameBoundString(frame.startType) << " and ";
  if (frame.endValue) {
    addKeys(stream, {frame.endValue});
    stream << " ";
  }
  stream << frameBoundString(frame.endType);
}

} // namespace

WindowNode::WindowNode(
    PlanNodeId id,
    std::vector<FieldAccessTypedExprPtr> partitionKeys,
    std::vector<FieldAccessTypedExprPtr> sortingKeys,
    std::vector<SortOrder> sortingOrders,
    std::vector<std::string> windowColumnNames,
    std::vector<Function> windowFunctions,
    PlanNodePtr source)
    : PlanNode(std::move(id)),
      partitionKeys_(std::move(partitionKeys)),
      sortingKeys_(std::move(sortingKeys)),
      sortingOrders_(std::move(sortingOrders)),
      windowFunctions_(std::move(windowFunctions)),
      sources_{std::move(source)},
      outputType_(getWindowOutputType(
          sources_[0]->outputType(),
          windowColumnNames,
          windowFunctions_)) {
  VELOX_CHECK_GT(
      windowFunctions_.size(),
      0,
      "Window node must have at least one window function");
  VELOX_CHECK_EQ(
      sortingKeys_.size(),
      sortingOrders_.size(),
      "Number of sorting keys must be equal to the number of sorting orders");
}

namespace {
void addSortingKeys(
    std::stringstream& stream,
    const std::vector<FieldAccessTypedExprPtr>& sortingKeys,
    const std::vector<SortOrder>& sortingOrders) {
  for (auto i = 0; i < sortingKeys.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << sortingKeys[i]->name() << " " << sortingOrders[i].toString();
  }
}
} // namespace

void LocalMergeNode::addDetails(std::stringstream& stream) const {
  addSortingKeys(stream, sortingKeys_, sortingOrders_);
}

void TableWriteNode::addDetails(std::stringstream& /* stream */) const {
  // TODO Add connector details.
}

void MergeExchangeNode::addDetails(std::stringstream& stream) const {
  addSortingKeys(stream, sortingKeys_, sortingOrders_);
}

void LocalPartitionNode::addDetails(std::stringstream& stream) const {
  // Nothing to add.
  switch (type_) {
    case Type::kGather:
      stream << "GATHER";
      break;
    case Type::kRepartition:
      stream << "REPARTITION";
      break;
  }
}

void EnforceSingleRowNode::addDetails(std::stringstream& /* stream */) const {
  // Nothing to add.
}

void PartitionedOutputNode::addDetails(std::stringstream& stream) const {
  if (broadcast_) {
    stream << "BROADCAST";
  } else if (numPartitions_ == 1) {
    stream << "SINGLE";
  } else {
    stream << "HASH(";
    addKeys(stream, keys_);
    stream << ") " << numPartitions_;
  }

  if (replicateNullsAndAny_) {
    stream << " replicate nulls and any";
  }
}

void TopNNode::addDetails(std::stringstream& stream) const {
  if (isPartial_) {
    stream << "PARTIAL ";
  }
  stream << count_ << " ";

  addSortingKeys(stream, sortingKeys_, sortingOrders_);
}

void LimitNode::addDetails(std::stringstream& stream) const {
  if (isPartial_) {
    stream << "PARTIAL ";
  }
  stream << count_;
  if (offset_) {
    stream << " offset " << offset_;
  }
}

void OrderByNode::addDetails(std::stringstream& stream) const {
  if (isPartial_) {
    stream << "PARTIAL ";
  }
  addSortingKeys(stream, sortingKeys_, sortingOrders_);
}

void WindowNode::addDetails(std::stringstream& stream) const {
  stream << "partition by [";
  if (!partitionKeys_.empty()) {
    addFields(stream, partitionKeys_);
  }
  stream << "] ";

  stream << "order by [";
  addSortingKeys(stream, sortingKeys_, sortingOrders_);
  stream << "] ";

  auto numInputCols = sources_[0]->outputType()->size();
  auto numOutputCols = outputType_->size();
  for (auto i = numInputCols; i < numOutputCols; i++) {
    if (i >= numInputCols + 1) {
      stream << ", ";
    }
    stream << outputType_->names()[i] << " := ";
    addWindowFunction(stream, windowFunctions_[i - numInputCols]);
  }
}

void PlanNode::toString(
    std::stringstream& stream,
    bool detailed,
    bool recursive,
    size_t indentationSize,
    std::function<void(
        const PlanNodeId& planNodeId,
        const std::string& indentation,
        std::stringstream& stream)> addContext) const {
  const std::string indentation(indentationSize, ' ');

  stream << indentation << "-- " << name();

  if (detailed) {
    stream << "[";
    addDetails(stream);
    stream << "]";
    stream << " -> ";
    outputType()->printChildren(stream, ", ");
  }
  stream << std::endl;

  if (addContext) {
    auto contextIndentation = indentation + "   ";
    stream << contextIndentation;
    addContext(id_, contextIndentation, stream);
    stream << std::endl;
  }

  if (recursive) {
    for (auto& source : sources()) {
      source->toString(stream, detailed, true, indentationSize + 2, addContext);
    }
  }
}

namespace {
void collectLeafPlanNodeIds(
    const core::PlanNode& planNode,
    std::unordered_set<core::PlanNodeId>& leafIds) {
  if (planNode.sources().empty()) {
    leafIds.insert(planNode.id());
    return;
  }

  for (const auto& child : planNode.sources()) {
    collectLeafPlanNodeIds(*child, leafIds);
  }
}
} // namespace

std::unordered_set<core::PlanNodeId> PlanNode::leafPlanNodeIds() const {
  std::unordered_set<core::PlanNodeId> leafIds;
  collectLeafPlanNodeIds(*this, leafIds);
  return leafIds;
}

} // namespace facebook::velox::core
