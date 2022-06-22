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
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/substrait/TypeUtils.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
namespace facebook::velox::substrait {

namespace {
template <TypeKind KIND>
VectorPtr setVectorFromVariantsByKind(
    const std::vector<velox::variant>& value,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<KIND>::NativeType;

  auto flatVector = std::dynamic_pointer_cast<FlatVector<T>>(
      BaseVector::create(CppToType<T>::create(), value.size(), pool));

  for (vector_size_t i = 0; i < value.size(); i++) {
    if (value[i].isNull()) {
      flatVector->setNull(i, true);
    } else {
      flatVector->set(i, value[i].value<T>());
    }
  }
  return flatVector;
}

template <>
VectorPtr setVectorFromVariantsByKind<TypeKind::VARBINARY>(
    const std::vector<velox::variant>& value,
    memory::MemoryPool* pool) {
  throw std::invalid_argument("Return of VARBINARY data is not supported");
}

template <>
VectorPtr setVectorFromVariantsByKind<TypeKind::VARCHAR>(
    const std::vector<velox::variant>& value,
    memory::MemoryPool* pool) {
  auto flatVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
      BaseVector::create(VARCHAR(), value.size(), pool));

  for (vector_size_t i = 0; i < value.size(); i++) {
    if (value[i].isNull()) {
      flatVector->setNull(i, true);
    } else {
      flatVector->set(i, StringView(value[i].value<Varchar>()));
    }
  }
  return flatVector;
}

VectorPtr setVectorFromVariants(
    const TypePtr& type,
    const std::vector<velox::variant>& value,
    velox::memory::MemoryPool* pool) {
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      setVectorFromVariantsByKind, type->kind(), value, pool);
}
} // namespace

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::AggregateRel& aggRel,
    memory::MemoryPool* pool) {
  core::PlanNodePtr childNode;
  if (aggRel.has_input()) {
    childNode = toVeloxPlan(aggRel.input(), pool);
  } else {
    VELOX_FAIL("Child Rel is expected in AggregateRel.");
  }
  core::AggregationNode::Step aggStep;
  // Get aggregation phase and check if there are input columns need to be
  // combined into row.
  if (needsRowConstruct(aggRel, aggStep)) {
    return toVeloxAggWithRowConstruct(aggRel, childNode, aggStep);
  }
  return toVeloxAgg(aggRel, childNode, aggStep);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxAggWithRowConstruct(
    const ::substrait::AggregateRel& sAgg,
    const core::PlanNodePtr& childNode,
    const core::AggregationNode::Step& aggStep) {
  // Will add a Project node before Aggregate node to combine columns into row.
  std::vector<core::TypedExprPtr> constructExprs;
  const auto& groupings = sAgg.groupings();
  const auto& constructInputType = childNode->outputType();

  // Handle groupings.
  uint32_t groupingOutIdx = 0;
  for (const auto& grouping : groupings) {
    const auto& groupingExprs = grouping.grouping_expressions();
    for (const auto& groupingExpr : groupingExprs) {
      // Velox's groupings are limited to be Field.
      auto fieldExpr = exprConverter_->toVeloxExpr(
          groupingExpr.selection(), constructInputType);
      constructExprs.push_back(fieldExpr);
      groupingOutIdx += 1;
    }
  }

  // Handle aggregations.
  std::vector<std::string> aggFuncNames;
  aggFuncNames.reserve(sAgg.measures().size());
  std::vector<TypePtr> aggOutTypes;
  aggOutTypes.reserve(sAgg.measures().size());

  for (const auto& smea : sAgg.measures()) {
    const auto& aggFunction = smea.measure();
    std::string funcName = substraitParser_->findVeloxFunction(
        functionMap_, aggFunction.function_reference());
    aggFuncNames.emplace_back(funcName);
    aggOutTypes.emplace_back(toVeloxType(
        substraitParser_->parseType(aggFunction.output_type())->type));
    if (isRowIntermediateType(funcName)) {
      // Will use row constructor to combine the sum and count columns into row.
      if (aggFunction.args().size() != 2) {
        VELOX_FAIL("Final average should have two args.");
      }
      std::vector<core::TypedExprPtr> aggParams;
      aggParams.reserve(aggFunction.args().size());
      for (const auto& arg : aggFunction.args()) {
        aggParams.emplace_back(
            exprConverter_->toVeloxExpr(arg, constructInputType));
      }
      auto constructExpr = std::make_shared<const core::CallTypedExpr>(
          ROW({"sum", "count"}, {DOUBLE(), BIGINT()}),
          std::move(aggParams),
          "row_constructor");
      constructExprs.emplace_back(constructExpr);
    } else {
      if (aggFunction.args().size() != 1) {
        VELOX_FAIL("Expect only one arg.");
      }
      for (const auto& arg : aggFunction.args()) {
        constructExprs.emplace_back(
            exprConverter_->toVeloxExpr(arg, constructInputType));
      }
    }
  }

  // Get the output names of row construct.
  std::vector<std::string> constructOutNames;
  constructOutNames.reserve(constructExprs.size());
  for (uint32_t colIdx = 0; colIdx < constructExprs.size(); colIdx++) {
    constructOutNames.emplace_back(
        substraitParser_->makeNodeName(planNodeId_, colIdx));
  }

  uint32_t totalOutColNum = constructExprs.size();
  // Create the row construct node.
  auto constructNode = std::make_shared<core::ProjectNode>(
      nextPlanNodeId(),
      std::move(constructOutNames),
      std::move(constructExprs),
      childNode);

  // Create the Aggregation node.
  bool ignoreNullKeys = false;
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> aggregateMasks(
      totalOutColNum - groupingOutIdx);
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      preGroupingExprs = {};

  // Get the output names of Aggregate node.
  std::vector<std::string> aggOutNames;
  aggOutNames.reserve(totalOutColNum - groupingOutIdx);
  for (uint32_t idx = groupingOutIdx; idx < totalOutColNum; idx++) {
    aggOutNames.emplace_back(substraitParser_->makeNodeName(planNodeId_, idx));
  }

  // Get the Aggregate expressions.
  std::vector<std::shared_ptr<const core::CallTypedExpr>> aggExprs;
  aggExprs.reserve(totalOutColNum - groupingOutIdx);
  const auto& constructOutType = constructNode->outputType();
  for (uint32_t colIdx = groupingOutIdx; colIdx < totalOutColNum; colIdx++) {
    std::vector<std::shared_ptr<const core::ITypedExpr>> aggArgs;
    aggArgs.reserve(1);
    // Use the colIdx to access the columns after grouping columns.
    aggArgs.emplace_back(std::make_shared<const core::FieldAccessTypedExpr>(
        constructOutType->childAt(colIdx), constructOutType->names()[colIdx]));
    // Use the another index to access the types and names of aggregation
    // columns.
    aggExprs.emplace_back(std::make_shared<const core::CallTypedExpr>(
        aggOutTypes[colIdx - groupingOutIdx],
        std::move(aggArgs),
        aggFuncNames[colIdx - groupingOutIdx]));
  }

  // Get the grouping expressions.
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> groupingExprs;
  groupingExprs.reserve(groupingOutIdx);
  for (uint32_t colIdx = 0; colIdx < groupingOutIdx; colIdx++) {
    // Velox's groupings are limited to be Field.
    groupingExprs.emplace_back(
        std::make_shared<const core::FieldAccessTypedExpr>(
            constructOutType->childAt(colIdx),
            constructOutType->names()[colIdx]));
  }

  // Create the Aggregation node.
  auto aggNode = std::make_shared<core::AggregationNode>(
      nextPlanNodeId(),
      aggStep,
      groupingExprs,
      preGroupingExprs,
      aggOutNames,
      aggExprs,
      aggregateMasks,
      ignoreNullKeys,
      constructNode);
  return aggNode;
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxAgg(
    const ::substrait::AggregateRel& sAgg,
    const std::shared_ptr<const core::PlanNode>& childNode,
    const core::AggregationNode::Step& aggStep) {
  const auto& inputType = childNode->outputType();
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      veloxGroupingExprs;

  // Get the grouping expressions.
  uint32_t groupingOutIdx = 0;
  for (const auto& grouping : sAgg.groupings()) {
    for (const auto& groupingExpr : grouping.grouping_expressions()) {
      // Velox's groupings are limited to be Field.
      veloxGroupingExprs.emplace_back(
          exprConverter_->toVeloxExpr(groupingExpr.selection(), inputType));
      groupingOutIdx += 1;
    }
  }

  // Parse measures and get the aggregate expressions.
  uint32_t aggOutIdx = groupingOutIdx;
  std::vector<std::shared_ptr<const core::CallTypedExpr>> aggExprs;
  aggExprs.reserve(sAgg.measures().size());
  for (const auto& smea : sAgg.measures()) {
    const auto& aggFunction = smea.measure();
    std::string funcName = substraitParser_->findVeloxFunction(
        functionMap_, aggFunction.function_reference());
    std::vector<core::TypedExprPtr> aggParams;
    aggParams.reserve(aggFunction.args().size());
    for (const auto& arg : aggFunction.args()) {
      aggParams.emplace_back(exprConverter_->toVeloxExpr(arg, inputType));
    }
    auto aggVeloxType = toVeloxType(
        substraitParser_->parseType(aggFunction.output_type())->type);
    if (isRowIntermediateType(funcName)) {
      // Will used sum and count to calculate the partial avg.
      auto sumExpr = std::make_shared<const core::CallTypedExpr>(
          aggVeloxType, aggParams, "sum");
      auto countExpr = std::make_shared<const core::CallTypedExpr>(
          BIGINT(), aggParams, "count");
      aggExprs.emplace_back(sumExpr);
      aggExprs.emplace_back(countExpr);
      aggOutIdx += 2;
    } else {
      auto aggExpr = std::make_shared<const core::CallTypedExpr>(
          aggVeloxType, std::move(aggParams), funcName);
      aggExprs.emplace_back(aggExpr);
      aggOutIdx += 1;
    }
  }

  bool ignoreNullKeys = false;
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> aggregateMasks(
      aggOutIdx - groupingOutIdx);
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      preGroupingExprs = {};

  // Get the output names of Aggregation.
  std::vector<std::string> aggOutNames;
  aggOutNames.reserve(aggOutIdx - groupingOutIdx);
  for (int idx = groupingOutIdx; idx < aggOutIdx; idx++) {
    aggOutNames.emplace_back(substraitParser_->makeNodeName(planNodeId_, idx));
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
    const ::substrait::ProjectRel& projectRel,
    memory::MemoryPool* pool) {
  core::PlanNodePtr childNode;
  if (projectRel.has_input()) {
    childNode = toVeloxPlan(projectRel.input(), pool);
  } else {
    VELOX_FAIL("Child Rel is expected in ProjectRel.");
  }

  // Construct Velox Expressions.
  auto projectExprs = projectRel.expressions();
  std::vector<std::string> projectNames;
  std::vector<core::TypedExprPtr> expressions;
  projectNames.reserve(projectExprs.size());
  expressions.reserve(projectExprs.size());

  const auto& inputType = childNode->outputType();
  int colIdx = 0;
  for (const auto& expr : projectExprs) {
    expressions.emplace_back(exprConverter_->toVeloxExpr(expr, inputType));
    projectNames.emplace_back(
        substraitParser_->makeNodeName(planNodeId_, colIdx));
    colIdx += 1;
  }

  return std::make_shared<core::ProjectNode>(
      nextPlanNodeId(),
      std::move(projectNames),
      std::move(expressions),
      childNode);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::FilterRel& filterRel,
    memory::MemoryPool* pool) {
  core::PlanNodePtr childNode;
  if (filterRel.has_input()) {
    childNode = toVeloxPlan(filterRel.input(), pool);
  } else {
    VELOX_FAIL("Child Rel is expected in FilterRel.");
  }

  const auto& inputType = childNode->outputType();
  const auto& sExpr = filterRel.condition();

  return std::make_shared<core::FilterNode>(
      nextPlanNodeId(),
      exprConverter_->toVeloxExpr(sExpr, inputType),
      childNode);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ReadRel& readRel,
    memory::MemoryPool* pool,
    std::shared_ptr<SplitInfo>& splitInfo) {
  // Get output names and types.
  std::vector<std::string> colNameList;
  std::vector<TypePtr> veloxTypeList;
  if (readRel.has_base_schema()) {
    const auto& baseSchema = readRel.base_schema();
    colNameList.reserve(baseSchema.names().size());
    for (const auto& name : baseSchema.names()) {
      colNameList.emplace_back(name);
    }
    auto substraitTypeList = substraitParser_->parseNamedStruct(baseSchema);
    veloxTypeList.reserve(substraitTypeList.size());
    for (const auto& substraitType : substraitTypeList) {
      veloxTypeList.emplace_back(toVeloxType(substraitType->type));
    }
  }

  // Parse local files
  if (readRel.has_local_files()) {
    const auto& fileList = readRel.local_files().items();
    splitInfo->paths.reserve(fileList.size());
    splitInfo->starts.reserve(fileList.size());
    splitInfo->lengths.reserve(fileList.size());
    for (const auto& file : fileList) {
      // Expect all files to share the same index.
      splitInfo->partitionIndex = file.partition_index();
      splitInfo->paths.emplace_back(file.uri_file());
      splitInfo->starts.emplace_back(file.start());
      splitInfo->lengths.emplace_back(file.length());
      switch (file.format()) {
        case 0:
          splitInfo->format = dwio::common::FileFormat::ORC;
          break;
        case 1:
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
  if (!readRel.has_filter()) {
    tableHandle = std::make_shared<connector::hive::HiveTableHandle>(
        kHiveConnectorId,
        "hive_table",
        filterPushdownEnabled,
        connector::hive::SubfieldFilters{},
        nullptr);
  } else {
    connector::hive::SubfieldFilters filters =
        toVeloxFilter(colNameList, veloxTypeList, readRel.filter());
    tableHandle = std::make_shared<connector::hive::HiveTableHandle>(
        kHiveConnectorId,
        "hive_table",
        filterPushdownEnabled,
        std::move(filters),
        nullptr);
  }

  // Get assignments and out names.
  std::vector<std::string> outNames;
  outNames.reserve(colNameList.size());
  std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
      assignments;
  for (int idx = 0; idx < colNameList.size(); idx++) {
    auto outName = substraitParser_->makeNodeName(planNodeId_, idx);
    assignments[outName] = std::make_shared<connector::hive::HiveColumnHandle>(
        colNameList[idx],
        connector::hive::HiveColumnHandle::ColumnType::kRegular,
        veloxTypeList[idx]);
    outNames.emplace_back(outName);
  }
  auto outputType = ROW(std::move(outNames), std::move(veloxTypeList));

  if (readRel.has_virtual_table()) {
    return toVeloxPlan(readRel, pool, outputType);

  } else {
    auto tableScanNode = std::make_shared<core::TableScanNode>(
        nextPlanNodeId(), outputType, tableHandle, assignments);
    return tableScanNode;
  }
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ReadRel& readRel,
    memory::MemoryPool* pool,
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
          setVectorFromVariants(outputChildType, batchChild, pool));
    }

    vectors.emplace_back(
        std::make_shared<RowVector>(pool, type, nullptr, batchSize, children));
  }

  return std::make_shared<core::ValuesNode>(nextPlanNodeId(), vectors);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::Rel& rel,
    memory::MemoryPool* pool) {
  if (rel.has_aggregate()) {
    return toVeloxPlan(rel.aggregate(), pool);
  }
  if (rel.has_project()) {
    return toVeloxPlan(rel.project(), pool);
  }
  if (rel.has_filter()) {
    return toVeloxPlan(rel.filter(), pool);
  }
  if (rel.has_read()) {
    auto splitInfo = std::make_shared<SplitInfo>();

    auto planNode = toVeloxPlan(rel.read(), pool, splitInfo);
    splitInfoMap_[planNode->id()] = splitInfo;
    return planNode;
  }
  VELOX_NYI("Substrait conversion not supported for Rel.");
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::RelRoot& root,
    memory::MemoryPool* pool) {
  // TODO: Use the names as the output names for the whole computing.
  const auto& names = root.names();
  if (root.has_input()) {
    const auto& rel = root.input();
    return toVeloxPlan(rel, pool);
  }
  VELOX_FAIL("Input is expected in RelRoot.");
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::Plan& substraitPlan,
    memory::MemoryPool* pool) {
  // Construct the function map based on the Substrait representation.
  constructFunctionMap(substraitPlan);

  // Construct the expression converter.
  exprConverter_ = std::make_shared<SubstraitVeloxExprConverter>(functionMap_);

  // In fact, only one RelRoot or Rel is expected here.
  for (const auto& rel : substraitPlan.relations()) {
    if (rel.has_root()) {
      return toVeloxPlan(rel.root(), pool);
    }
    if (rel.has_rel()) {
      return toVeloxPlan(rel.rel(), pool);
    }
  }
  VELOX_FAIL("RelRoot or Rel is expected in Plan.");
}

std::string SubstraitVeloxPlanConverter::nextPlanNodeId() {
  auto id = fmt::format("{}", planNodeId_);
  planNodeId_++;
  return id;
}

// This class contains the needed infos for Filter Pushdown.
// TODO: Support different types here.
class FilterInfo {
 public:
  // Used to set the left bound.
  void setLeft(double left, bool isExclusive) {
    left_ = left;
    leftExclusive_ = isExclusive;
    if (!isInitialized_) {
      isInitialized_ = true;
    }
  }

  // Used to set the right bound.
  void setRight(double right, bool isExclusive) {
    right_ = right;
    rightExclusive_ = isExclusive;
    if (!isInitialized_) {
      isInitialized_ = true;
    }
  }

  // Will fordis Null value if called once.
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

  // The left bound.
  std::optional<double> left_ = std::nullopt;
  // The right bound.
  std::optional<double> right_ = std::nullopt;
  // The Null allowing.
  bool nullAllowed_ = true;
  // If true, left bound will be exclusive.
  bool leftExclusive_ = false;
  // If true, right bound will be exclusive.
  bool rightExclusive_ = false;

 private:
  bool isInitialized_ = false;
};

connector::hive::SubfieldFilters SubstraitVeloxPlanConverter::toVeloxFilter(
    const std::vector<std::string>& inputNameList,
    const std::vector<TypePtr>& inputTypeList,
    const ::substrait::Expression& substraitFilter) {
  connector::hive::SubfieldFilters filters;
  // A map between the column index and the FilterInfo for that column.
  std::unordered_map<int, std::shared_ptr<FilterInfo>> colInfoMap;
  for (int idx = 0; idx < inputNameList.size(); idx++) {
    colInfoMap[idx] = std::make_shared<FilterInfo>();
  }

  std::vector<::substrait::Expression_ScalarFunction> scalarFunctions;
  flattenConditions(substraitFilter, scalarFunctions);
  // Construct the FilterInfo for the related column.
  for (const auto& scalarFunction : scalarFunctions) {
    auto filterNameSpec = substraitParser_->findFunctionSpec(
        functionMap_, scalarFunction.function_reference());
    auto filterName = substraitParser_->getFunctionName(filterNameSpec);
    int32_t colIdx;
    // TODO: Add different types' support here.
    double val;
    for (auto& param : scalarFunction.args()) {
      auto typeCase = param.rex_type_case();
      switch (typeCase) {
        case ::substrait::Expression::RexTypeCase::kSelection: {
          auto sel = param.selection();
          // TODO: Only direct reference is considered here.
          auto dRef = sel.direct_reference();
          colIdx = substraitParser_->parseReferenceSegment(dRef);
          break;
        }
        case ::substrait::Expression::RexTypeCase::kLiteral: {
          auto sLit = param.literal();
          // TODO: Only double is considered here.
          val = sLit.fp64();
          break;
        }
        default:
          VELOX_NYI(
              "Substrait conversion not supported for arg type '{}'", typeCase);
      }
    }
    if (filterName == "is_not_null") {
      colInfoMap[colIdx]->forbidsNull();
    } else if (filterName == "gte") {
      colInfoMap[colIdx]->setLeft(val, false);
    } else if (filterName == "gt") {
      colInfoMap[colIdx]->setLeft(val, true);
    } else if (filterName == "lte") {
      colInfoMap[colIdx]->setRight(val, false);
    } else if (filterName == "lt") {
      colInfoMap[colIdx]->setRight(val, true);
    } else {
      VELOX_NYI(
          "Substrait conversion not supported for filter name '{}'",
          filterName);
    }
  }

  // Construct the Filters.
  for (int idx = 0; idx < inputNameList.size(); idx++) {
    auto filterInfo = colInfoMap[idx];
    double leftBound;
    double rightBound;
    bool leftUnbounded = true;
    bool rightUnbounded = true;
    bool leftExclusive = false;
    bool rightExclusive = false;
    if (filterInfo->isInitialized()) {
      if (filterInfo->left_) {
        leftUnbounded = false;
        leftBound = filterInfo->left_.value();
        leftExclusive = filterInfo->leftExclusive_;
      }
      if (filterInfo->right_) {
        rightUnbounded = false;
        rightBound = filterInfo->right_.value();
        rightExclusive = filterInfo->rightExclusive_;
      }
      bool nullAllowed = filterInfo->nullAllowed_;
      filters[common::Subfield(inputNameList[idx])] =
          std::make_unique<common::DoubleRange>(
              leftBound,
              leftUnbounded,
              leftExclusive,
              rightBound,
              rightUnbounded,
              rightExclusive,
              nullAllowed);
    }
  }
  return filters;
}

void SubstraitVeloxPlanConverter::flattenConditions(
    const ::substrait::Expression& substraitFilter,
    std::vector<::substrait::Expression_ScalarFunction>& scalarFunctions) {
  auto typeCase = substraitFilter.rex_type_case();
  switch (typeCase) {
    case ::substrait::Expression::RexTypeCase::kScalarFunction: {
      auto sFunc = substraitFilter.scalar_function();
      auto filterNameSpec = substraitParser_->findFunctionSpec(
          functionMap_, sFunc.function_reference());
      // TODO: Only and relation is supported here.
      if (substraitParser_->getFunctionName(filterNameSpec) == "and") {
        for (const auto& sCondition : sFunc.args()) {
          flattenConditions(sCondition, scalarFunctions);
        }
      } else {
        scalarFunctions.emplace_back(sFunc);
      }
      break;
    }
    default:
      VELOX_NYI("GetFlatConditions not supported for type '{}'", typeCase);
  }
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
}

const std::string& SubstraitVeloxPlanConverter::findFunction(
    uint64_t id) const {
  return substraitParser_->findFunctionSpec(functionMap_, id);
}

bool SubstraitVeloxPlanConverter::isRowIntermediateType(std::string name) {
  auto signatures = exec::getAggregateFunctionSignatures(name);
  for (const auto& signature : signatures.value()) {
    if (signature->intermediateType().baseType() == "row") {
      return true;
    }
  }
  return false;
}

bool SubstraitVeloxPlanConverter::needsRowConstruct(
    const ::substrait::AggregateRel& sAgg,
    core::AggregationNode::Step& aggStep) {
  if (sAgg.measures().size() == 0) {
    // When only groupings exist, set the phase to be Single.
    aggStep = core::AggregationNode::Step::kSingle;
    return false;
  }
  for (const auto& smea : sAgg.measures()) {
    auto aggFunction = smea.measure();
    std::string funcName = substraitParser_->findVeloxFunction(
        functionMap_, aggFunction.function_reference());
    // Set the aggregation phase.
    switch (aggFunction.phase()) {
      case ::substrait::AGGREGATION_PHASE_INITIAL_TO_INTERMEDIATE:
        aggStep = core::AggregationNode::Step::kPartial;
        break;
      case ::substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_INTERMEDIATE:
        aggStep = core::AggregationNode::Step::kIntermediate;
        break;
      case ::substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_RESULT:
        aggStep = core::AggregationNode::Step::kFinal;
        // Only Final Average needs row construct currently.
        if (isRowIntermediateType(funcName)) {
          return true;
        }
        break;
      default:
        throw std::runtime_error("Aggregate phase is not supported.");
    }
  }
  return false;
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
} // namespace facebook::velox::substrait
