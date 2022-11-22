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

#include "velox/substrait/SubstraitToVeloxPlanValidator.h"
#include "TypeUtils.h"

namespace facebook::velox::substrait {

bool SubstraitToVeloxPlanValidator::validateInputTypes(
    const ::substrait::extensions::AdvancedExtension& extension,
    std::vector<TypePtr>& types) {
  // The input type is wrapped in enhancement.
  if (!extension.has_enhancement()) {
    return false;
  }
  const auto& enhancement = extension.enhancement();
  ::substrait::Type inputType;
  if (!enhancement.UnpackTo(&inputType)) {
    return false;
  }
  if (!inputType.has_struct_()) {
    return false;
  }

  // Get the input types.
  const auto& sTypes = inputType.struct_().types();
  for (const auto& sType : sTypes) {
    try {
      types.emplace_back(toVeloxType(subParser_->parseType(sType)->type));
    } catch (const VeloxException& err) {
      std::cout << "Type is not supported in ProjectRel due to:"
                << err.message() << std::endl;
      return false;
    }
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::FetchRel& fetchRel) {
  if (fetchRel.offset() < 0 || fetchRel.count() < 0) {
    std::cout << "Offset and count should be valid." << std::endl;
    return false;
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::ExpandRel& sExpand) {
  if (sExpand.has_input() && !validate(sExpand.input())) {
    return false;
  }
  // Get and validate the input types from extension.
  if (!sExpand.has_advanced_extension()) {
    std::cout << "Input types are expected in ExpandRel." << std::endl;
    return false;
  }
  const auto& extension = sExpand.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    std::cout << "Validation failed for input types in ExpandRel." << std::endl;
    return false;
  }

  int32_t inputPlanNodeId = 0;
  std::vector<std::string> names;
  names.reserve(types.size());
  for (auto colIdx = 0; colIdx < types.size(); colIdx++) {
    names.emplace_back(subParser_->makeNodeName(inputPlanNodeId, colIdx));
  }
  auto rowType = std::make_shared<RowType>(std::move(names), std::move(types));

  // Validate the expand agg expressions.
  const auto& aggExprs = sExpand.aggregate_expressions();
  std::vector<std::shared_ptr<const core::ITypedExpr>> expressions;
  expressions.reserve(aggExprs.size());

  try {
    for (const auto& expr : aggExprs) {
      expressions.emplace_back(exprConverter_->toVeloxExpr(expr, rowType));
    }
    // Try to compile the expressions. If there is any unregistred funciton or
    // mismatched type, exception will be thrown.
    exec::ExprSet exprSet(std::move(expressions), execCtx_);
  } catch (const VeloxException& err) {
    std::cout << "Validation failed for agg expression in ExpandRel due to:"
              << err.message() << std::endl;
    return false;
  }

  // Validate groupings.
  for (const auto& grouping : sExpand.groupings()) {
    for (const auto& groupingExpr : grouping.groupsets_expressions()) {
      const auto& typeCase = groupingExpr.rex_type_case();
      switch (typeCase) {
        case ::substrait::Expression::RexTypeCase::kSelection:
          break;
        default:
          std::cout << "Only field is supported in groupings." << std::endl;
          return false;
      }
    }
  }

  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::SortRel& sSort) {
  if (sSort.has_input() && !validate(sSort.input())) {
    return false;
  }
  // Get and validate the input types from extension.
  if (!sSort.has_advanced_extension()) {
    std::cout << "Input types are expected in SortRel." << std::endl;
    return false;
  }
  const auto& extension = sSort.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    std::cout << "Validation failed for input types in SortRel." << std::endl;
    return false;
  }

  int32_t inputPlanNodeId = 0;
  std::vector<std::string> names;
  names.reserve(types.size());
  for (auto colIdx = 0; colIdx < types.size(); colIdx++) {
    names.emplace_back(subParser_->makeNodeName(inputPlanNodeId, colIdx));
  }
  auto rowType = std::make_shared<RowType>(std::move(names), std::move(types));

  const auto& sorts = sSort.sorts();
  for (const auto& sort : sorts) {
    switch (sort.direction()) {
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_ASC_NULLS_FIRST:
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_ASC_NULLS_LAST:
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_DESC_NULLS_FIRST:
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_DESC_NULLS_LAST:
        break;
      default:
        return false;
    }

    if (sort.has_expr()) {
      try {
        auto expression = exprConverter_->toVeloxExpr(sort.expr(), rowType);
        auto expr_field =
            dynamic_cast<const core::FieldAccessTypedExpr*>(expression.get());
        VELOX_CHECK(
            expr_field != nullptr,
            " the sorting key in Sort Operator only support field")

        exec::ExprSet exprSet({std::move(expression)}, execCtx_);
      } catch (const VeloxException& err) {
        std::cout << "Validation failed for expression in SortRel due to:"
                  << err.message() << std::endl;
        return false;
      }
    }
  }

  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::ProjectRel& sProject) {
  if (sProject.has_input() && !validate(sProject.input())) {
    return false;
  }

  // Get and validate the input types from extension.
  if (!sProject.has_advanced_extension()) {
    std::cout << "Input types are expected in ProjectRel." << std::endl;
    return false;
  }
  const auto& extension = sProject.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    std::cout << "Validation failed for input types in ProjectRel."
              << std::endl;
    return false;
  }

  // In velox, the supported hash type in projectNode is 
  // BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT, VARCHAR, VARBINARY
  // REAL, DOUBLE. Hash.cpp (L148 - L156)
  for (auto i = 0; i < types.size(); i++) {
    switch (types[i]->kind()) {
      case TypeKind::DATE:
        return false;
      default:;
    }
  }

  int32_t inputPlanNodeId = 0;
  // Create the fake input names to be used in row type.
  std::vector<std::string> names;
  names.reserve(types.size());
  for (uint32_t colIdx = 0; colIdx < types.size(); colIdx++) {
    names.emplace_back(subParser_->makeNodeName(inputPlanNodeId, colIdx));
  }
  auto rowType = std::make_shared<RowType>(std::move(names), std::move(types));

  // Validate the project expressions.
  const auto& projectExprs = sProject.expressions();
  std::vector<std::shared_ptr<const core::ITypedExpr>> expressions;
  expressions.reserve(projectExprs.size());
  try {
    for (const auto& expr : projectExprs) {
      expressions.emplace_back(exprConverter_->toVeloxExpr(expr, rowType));
    }
    // Try to compile the expressions. If there is any unregistred funciton or
    // mismatched type, exception will be thrown.
    exec::ExprSet exprSet(std::move(expressions), execCtx_);
  } catch (const VeloxException& err) {
    std::cout << "Validation failed for expression in ProjectRel due to:"
              << err.message() << std::endl;
    return false;
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::FilterRel& sFilter) {
  if (sFilter.has_input() && !validate(sFilter.input())) {
    return false;
  }

  // Get and validate the input types from extension.
  if (!sFilter.has_advanced_extension()) {
    std::cout << "Input types are expected in FilterRel." << std::endl;
    return false;
  }
  const auto& extension = sFilter.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    std::cout << "Validation failed for input types in FilterRel." << std::endl;
    return false;
  }

  int32_t inputPlanNodeId = 0;
  // Create the fake input names to be used in row type.
  std::vector<std::string> names;
  names.reserve(types.size());
  for (uint32_t colIdx = 0; colIdx < types.size(); colIdx++) {
    names.emplace_back(subParser_->makeNodeName(inputPlanNodeId, colIdx));
  }
  auto rowType = std::make_shared<RowType>(std::move(names), std::move(types));

  std::vector<std::shared_ptr<const core::ITypedExpr>> expressions;
  expressions.reserve(1);
  try {
    expressions.emplace_back(
        exprConverter_->toVeloxExpr(sFilter.condition(), rowType));
    // Try to compile the expressions. If there is any unregistred funciton
    // or mismatched type, exception will be thrown.
    exec::ExprSet exprSet(std::move(expressions), execCtx_);
  } catch (const VeloxException& err) {
    std::cout << "Validation failed for expression in ProjectRel due to:"
              << err.message() << std::endl;
    return false;
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::JoinRel& sJoin) {
  if (sJoin.has_left() && !validate(sJoin.left())) {
    return false;
  }
  if (sJoin.has_right() && !validate(sJoin.right())) {
    return false;
  }

  switch (sJoin.type()) {
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_INNER:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_OUTER:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_LEFT:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_RIGHT:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_LEFT_SEMI:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_RIGHT_SEMI:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_ANTI:
      break;
    default:
      return false;
  }

  // Validate input types.
  if (!sJoin.has_advanced_extension()) {
    std::cout << "Input types are expected in JoinRel." << std::endl;
    return false;
  }

  const auto& extension = sJoin.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    std::cout << "Validation failed for input types in JoinRel" << std::endl;
    return false;
  }

  int32_t inputPlanNodeId = 0;
  std::vector<std::string> names;
  names.reserve(types.size());
  for (auto colIdx = 0; colIdx < types.size(); colIdx++) {
    names.emplace_back(subParser_->makeNodeName(inputPlanNodeId, colIdx));
  }
  auto rowType = std::make_shared<RowType>(std::move(names), std::move(types));

  if (sJoin.has_expression()) {
    std::vector<const ::substrait::Expression::FieldReference*> leftExprs,
        rightExprs;
    try {
      planConverter_->extractJoinKeys(
          sJoin.expression(), leftExprs, rightExprs);
    } catch (const VeloxException& err) {
      std::cout << "Validation failed for expression in JoinRel due to:"
                << err.message() << std::endl;
      return false;
    }
  }

  if (sJoin.has_post_join_filter()) {
    try {
      auto expression =
          exprConverter_->toVeloxExpr(sJoin.post_join_filter(), rowType);
      exec::ExprSet exprSet({std::move(expression)}, execCtx_);
    } catch (const VeloxException& err) {
      std::cout << "Validation failed for expression in ProjectRel due to:"
                << err.message() << std::endl;
      return false;
    }
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::AggregateRel& sAgg) {
  if (sAgg.has_input() && !validate(sAgg.input())) {
    return false;
  }

  // Validate input types.
  if (sAgg.has_advanced_extension()) {
    const auto& extension = sAgg.advanced_extension();
    std::vector<TypePtr> types;
    if (!validateInputTypes(extension, types)) {
      std::cout << "Validation failed for input types in AggregateRel"
                << std::endl;
      return false;
    }
  }

  // Validate groupings.
  for (const auto& grouping : sAgg.groupings()) {
    for (const auto& groupingExpr : grouping.grouping_expressions()) {
      const auto& typeCase = groupingExpr.rex_type_case();
      switch (typeCase) {
        case ::substrait::Expression::RexTypeCase::kSelection:
          break;
        default:
          std::cout << "Only field is supported in groupings." << std::endl;
          return false;
      }
    }
  }

  // Validate aggregate functions.
  std::vector<std::string> funcSpecs;
  funcSpecs.reserve(sAgg.measures().size());
  for (const auto& smea : sAgg.measures()) {
    try {
      const auto& aggFunction = smea.measure();
      funcSpecs.emplace_back(
          planConverter_->findFuncSpec(aggFunction.function_reference()));
      toVeloxType(subParser_->parseType(aggFunction.output_type())->type);
      for (const auto& arg : aggFunction.arguments()) {
        auto typeCase = arg.value().rex_type_case();
        switch (typeCase) {
          case ::substrait::Expression::RexTypeCase::kSelection:
          case ::substrait::Expression::RexTypeCase::kLiteral:
            break;
          default:
            std::cout << "Only field is supported in aggregate functions."
                      << std::endl;
            return false;
        }
      }
    } catch (const VeloxException& err) {
      std::cout << "Validation failed for aggregate function due to: "
                << err.message() << std::endl;
      return false;
    }
  }

  std::unordered_set<std::string> supportedFuncs = {
      "sum", "count", "avg", "min", "max", "stddev_samp"};
  for (const auto& funcSpec : funcSpecs) {
    auto funcName = subParser_->getSubFunctionName(funcSpec);
    if (supportedFuncs.find(funcName) == supportedFuncs.end()) {
      std::cout << "Validation failed due to " << funcName
                << " was not supported in AggregateRel." << std::endl;
      return false;
    }
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::ReadRel& sRead) {
  try {
    planConverter_->toVeloxPlan(sRead);
  } catch (const VeloxException& err) {
    std::cout << "ReadRel validation failed due to:" << err.message()
              << std::endl;
    return false;
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(const ::substrait::Rel& sRel) {
  if (sRel.has_aggregate()) {
    return validate(sRel.aggregate());
  }
  if (sRel.has_project()) {
    return validate(sRel.project());
  }
  if (sRel.has_filter()) {
    return validate(sRel.filter());
  }
  if (sRel.has_join()) {
    return validate(sRel.join());
  }
  if (sRel.has_read()) {
    return validate(sRel.read());
  }
  if (sRel.has_sort()) {
    return validate(sRel.sort());
  }
  if (sRel.has_expand()) {
    return validate(sRel.expand());
  }
  if (sRel.has_fetch()) {
    return validate(sRel.fetch());
  }
  return false;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::RelRoot& sRoot) {
  if (sRoot.has_input()) {
    const auto& sRel = sRoot.input();
    return validate(sRel);
  }
  return false;
}

bool SubstraitToVeloxPlanValidator::validate(const ::substrait::Plan& sPlan) {
  // Create plan converter and expression converter to help the validation.
  planConverter_->constructFunctionMap(sPlan);
  exprConverter_ = std::make_shared<SubstraitVeloxExprConverter>(
      pool_, planConverter_->getFunctionMap());

  for (const auto& sRel : sPlan.relations()) {
    if (sRel.has_root()) {
      return validate(sRel.root());
    }
    if (sRel.has_rel()) {
      return validate(sRel.rel());
    }
  }
  return false;
}

} // namespace facebook::velox::substrait
