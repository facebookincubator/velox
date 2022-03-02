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

#include "SubstraitToVeloxPlan.h"
#include "TypeUtils.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::connector;
using namespace facebook::velox::dwio::common;

namespace facebook::velox::substrait {

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::AggregateRel& sAgg) {
  std::shared_ptr<const core::PlanNode> childNode;
  if (sAgg.has_input()) {
    childNode = toVeloxPlan(sAgg.input());
  } else {
    VELOX_FAIL("Child Rel is expected in AggregateRel.");
  }
  auto inputTypes = childNode->outputType();
  // Construct Velox grouping expressions.
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      veloxGroupingExprs;
  const auto& groupings = sAgg.groupings();
  int inputPlanNodeId = planNodeId_ - 1;
  int outIdx = 0;
  for (const auto& grouping : groupings) {
    auto groupingExprs = grouping.grouping_expressions();
    for (const auto& groupingExpr : groupingExprs) {
      auto fieldExpr =
          exprConverter_->toVeloxExpr(groupingExpr, inputPlanNodeId);
      // Velox's groupings are limited to be Field, and pre-projection for
      // grouping cols is not supported.
      auto typedFieldExpr =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
              fieldExpr);
      veloxGroupingExprs.push_back(typedFieldExpr);
      outIdx += 1;
    }
  }
  // Parse measures.
  core::AggregationNode::Step aggStep;
  bool phaseInited = false;
  std::vector<std::shared_ptr<const core::CallTypedExpr>> aggExprs;
  std::vector<std::shared_ptr<const core::ITypedExpr>> projectExprs;
  std::vector<std::string> projectOutNames;
  for (const auto& sMea : sAgg.measures()) {
    auto aggFunction = sMea.measure();
    if (!phaseInited) {
      auto phase = aggFunction.phase();
      switch (phase) {
        case ::substrait::AGGREGATION_PHASE_INITIAL_TO_INTERMEDIATE:
          aggStep = core::AggregationNode::Step::kPartial;
          break;
        case ::substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_INTERMEDIATE:
          aggStep = core::AggregationNode::Step::kIntermediate;
          break;
        case ::substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_RESULT:
          aggStep = core::AggregationNode::Step::kFinal;
          break;
        default:
          VELOX_NYI("Substrait conversion not supported for phase '{}'", phase);
      }
      phaseInited = true;
    }
    auto funcId = aggFunction.function_reference();
    auto funcName = subParser_->findVeloxFunction(functionMap_, funcId);
    // Construct Velox aggregate expressions.
    std::vector<std::shared_ptr<const core::ITypedExpr>> aggParams;
    auto args = aggFunction.args();
    for (auto arg : args) {
      auto typeCase = arg.rex_type_case();
      switch (typeCase) {
        case ::substrait::Expression::RexTypeCase::kSelection: {
          auto sel = arg.selection();
          auto fieldExpr = exprConverter_->toVeloxExpr(sel, inputPlanNodeId);
          aggParams.push_back(fieldExpr);
          break;
        }
        case ::substrait::Expression::RexTypeCase::kScalarFunction: {
          // Pre-projection is needed before Aggregate.
          // The input of Aggregatation will be the output of the
          // pre-projection.
          auto sFunc = arg.scalar_function();
          auto veloxExpr = exprConverter_->toVeloxExpr(sFunc, inputPlanNodeId);
          projectExprs.push_back(veloxExpr);
          auto colOutName = subParser_->makeNodeName(planNodeId_, outIdx);
          projectOutNames.push_back(colOutName);
          auto subType = subParser_->parseType(sFunc.output_type());
          auto veloxType = toVeloxType(subType->type);
          auto aggInputParam =
              std::make_shared<const core::FieldAccessTypedExpr>(
                  veloxType, colOutName);
          aggParams.push_back(aggInputParam);
          break;
        }
        default:
          VELOX_NYI(
              "Substrait conversion not supported for arg type '{}'", typeCase);
      }
    }
    auto aggOutType = aggFunction.output_type();
    auto aggVeloxType = toVeloxType(subParser_->parseType(aggOutType)->type);
    auto aggExpr = std::make_shared<const core::CallTypedExpr>(
        aggVeloxType, std::move(aggParams), funcName);
    aggExprs.push_back(aggExpr);
    outIdx += 1;
  }
  bool ignoreNullKeys = false;
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> aggregateMasks(
      outIdx);
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      preGroupingExprs;
  if (projectOutNames.size() > 0) {
    // A Project Node is needed before Aggregation.
    auto projectNode = std::make_shared<core::ProjectNode>(
        nextPlanNodeId(),
        std::move(projectOutNames),
        std::move(projectExprs),
        childNode);
    std::vector<std::string> aggOutNames;
    for (int idx = 0; idx < outIdx; idx++) {
      aggOutNames.push_back(subParser_->makeNodeName(planNodeId_, idx));
    }
    auto aggNode = std::make_shared<core::AggregationNode>(
        nextPlanNodeId(),
        aggStep,
        veloxGroupingExprs,
        preGroupingExprs,
        aggOutNames,
        aggExprs,
        aggregateMasks,
        ignoreNullKeys,
        projectNode);
    return aggNode;
  } else {
    // Conduct Aggregation directly.
    std::vector<std::string> aggOutNames;
    for (int idx = 0; idx < outIdx; idx++) {
      aggOutNames.push_back(subParser_->makeNodeName(planNodeId_, idx));
    }
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
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ProjectRel& sProject) {
  std::shared_ptr<const core::PlanNode> childNode;
  if (sProject.has_input()) {
    childNode = toVeloxPlan(sProject.input());
  } else {
    VELOX_FAIL("Child Rel is expected in ProjectRel.");
  }
  // Construct Velox Expressions.
  std::vector<std::string> projectNames;
  std::vector<std::shared_ptr<const core::ITypedExpr>> expressions;
  auto prePlanNodeId = planNodeId_ - 1;
  int colIdx = 0;
  for (const auto& expr : sProject.expressions()) {
    auto veloxExpr = exprConverter_->toVeloxExpr(expr, prePlanNodeId);
    expressions.push_back(veloxExpr);
    auto colOutName = subParser_->makeNodeName(planNodeId_, colIdx);
    projectNames.push_back(colOutName);
    colIdx += 1;
  }
  auto projectNode = std::make_shared<core::ProjectNode>(
      nextPlanNodeId(),
      std::move(projectNames),
      std::move(expressions),
      childNode);
  return projectNode;
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::FilterRel& sFilter) {
  // TODO: Currently Filter is skipped because Filter is Pushdowned to
  // TableScan.
  std::shared_ptr<const core::PlanNode> childNode;
  if (sFilter.has_input()) {
    childNode = toVeloxPlan(sFilter.input());
  } else {
    VELOX_FAIL("Child Rel is expected in FilterRel.");
  }
  return childNode;
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ReadRel& sRead,
    u_int32_t* index,
    std::vector<std::string>* paths,
    std::vector<u_int64_t>* starts,
    std::vector<u_int64_t>* lengths) {
  std::vector<std::string> colNameList;
  std::vector<std::shared_ptr<SubstraitParser::SubstraitType>>
      substraitTypeList;
  // Get output names and types.
  if (sRead.has_base_schema()) {
    const auto& baseSchema = sRead.base_schema();
    for (const auto& name : baseSchema.names()) {
      colNameList.push_back(name);
    }
    auto typeList = subParser_->parseNamedStruct(baseSchema);
    for (auto type : typeList) {
      substraitTypeList.push_back(type);
    }
  }
  std::vector<TypePtr> veloxTypeList;
  for (auto subType : substraitTypeList) {
    veloxTypeList.push_back(toVeloxType(subType->type));
  }
  // Parse local files
  if (sRead.has_local_files()) {
    const auto& localFiles = sRead.local_files();
    const auto& fileList = localFiles.items();
    for (const auto& file : fileList) {
      // Expect all Partitions share the same index.
      (*index) = file.partition_index();
      (*paths).push_back(file.uri_file());
      (*starts).push_back(file.start());
      (*lengths).push_back(file.length());
    }
  }
  // Velox requires Filter Pushdown must being enabled.
  bool filterPushdownEnabled = true;
  std::shared_ptr<connector::hive::HiveTableHandle> tableHandle;
  if (!sRead.has_filter()) {
    tableHandle = std::make_shared<hive::HiveTableHandle>(
        filterPushdownEnabled, hive::SubfieldFilters{}, nullptr);
  } else {
    const auto& sFilter = sRead.filter();
    connector::hive::SubfieldFilters filters =
        exprConverter_->toVeloxFilter(colNameList, veloxTypeList, sFilter);
    tableHandle = std::make_shared<connector::hive::HiveTableHandle>(
        filterPushdownEnabled, std::move(filters), nullptr);
  }
  std::vector<std::string> outNames;
  std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
      assignments;
  for (int idx = 0; idx < colNameList.size(); idx++) {
    auto outName = subParser_->makeNodeName(planNodeId_, idx);
    assignments[outName] = std::make_shared<connector::hive::HiveColumnHandle>(
        colNameList[idx],
        connector::hive::HiveColumnHandle::ColumnType::kRegular,
        veloxTypeList[idx]);
    outNames.push_back(outName);
  }
  auto outputType = ROW(std::move(outNames), std::move(veloxTypeList));
  auto tableScanNode = std::make_shared<core::TableScanNode>(
      nextPlanNodeId(), outputType, tableHandle, assignments);
  return tableScanNode;
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::Rel& sRel) {
  if (sRel.has_aggregate()) {
    return toVeloxPlan(sRel.aggregate());
  } else if (sRel.has_project()) {
    return toVeloxPlan(sRel.project());
  } else if (sRel.has_filter()) {
    return toVeloxPlan(sRel.filter());
  } else if (sRel.has_read()) {
    return toVeloxPlan(
        sRel.read(), &partitionIndex_, &paths_, &starts_, &lengths_);
  } else {
    VELOX_NYI("Substrait conversion not supported for Rel.");
  }
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::RelRoot& sRoot) {
  const auto& sNames = sRoot.names();
  int nameIdx = 0;
  // TODO: Use the names as the output names for the whole computing.
  for (const auto& sName : sNames) {
    nameIdx += 1;
  }
  if (sRoot.has_input()) {
    const auto& sRel = sRoot.input();
    return toVeloxPlan(sRel);
  } else {
    VELOX_FAIL("Input is expected in RelRoot.");
  }
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::Plan& sPlan) {
  // Construct the function map based on the Substrait representation.
  for (const auto& sExtension : sPlan.extensions()) {
    if (!sExtension.has_extension_function()) {
      continue;
    }
    const auto& sFmap = sExtension.extension_function();
    auto id = sFmap.function_anchor();
    auto name = sFmap.name();
    functionMap_[id] = name;
  }
  exprConverter_ =
      std::make_shared<SubstraitVeloxExprConverter>(subParser_, functionMap_);
  std::shared_ptr<const core::PlanNode> planNode;
  // In fact, only one RelRoot is expected here.
  for (const auto& sRel : sPlan.relations()) {
    if (sRel.has_root()) {
      planNode = toVeloxPlan(sRel.root());
    }
    if (sRel.has_rel()) {
      planNode = toVeloxPlan(sRel.rel());
    }
  }
  return planNode;
}

std::string SubstraitVeloxPlanConverter::nextPlanNodeId() {
  auto id = fmt::format("{}", planNodeId_);
  planNodeId_++;
  return id;
}

} // namespace facebook::velox::substrait
