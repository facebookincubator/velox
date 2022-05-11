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

#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

namespace facebook::velox::substrait {

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::Type& substraitType) {
  switch (substraitType.kind_case()) {
    case ::substrait::Type::KindCase::kBool:
    case ::substrait::Type::KindCase::kI32:
    case ::substrait::Type::KindCase::kI64:
    case ::substrait::Type::KindCase::kFp64:
    case ::substrait::Type::KindCase::kString:
      return true;
    default:
      return false;
  }
}

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
  const auto& substraitTypes = inputType.struct_().types();
  for (const auto& substraitType : substraitTypes) {
    try {
      types.emplace_back(
          toVeloxType(subParser_->parseType(substraitType)->type));
    } catch (const VeloxException& err) {
      VLOG(0) << "Type is not supported in ProjectRel due to:" << err.message();
      return false;
    }
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::ProjectRel& projectRel) {
  if (projectRel.has_input() && !validate(projectRel.input())) {
    return false;
  }

  // Get and validate the input types from extension.
  if (!projectRel.has_advanced_extension()) {
    VLOG(0) << "Input types are expected in ProjectRel.";
    return false;
  }
  const auto& extension = projectRel.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    VLOG(0) << "Validation failed for input types in ProjectRel";
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

  // Validate the project expressions.
  const auto& projectExprs = projectRel.expressions();
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
    VLOG(0) << "Validation failed for expression in ProjectRel due to:"
            << err.message();
    return false;
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::FilterRel& filterRel) {
  // Not verified the Filter and fallback it currently.
  return false;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::AggregateRel& aggRel) {
  if (aggRel.has_input() && !validate(aggRel.input())) {
    return false;
  }

  // Validate input types.
  if (aggRel.has_advanced_extension()) {
    const auto& extension = aggRel.advanced_extension();
    std::vector<TypePtr> types;
    if (!validateInputTypes(extension, types)) {
      VLOG(0) << "Validation failed for input types in AggregateRel";
      return false;
    }
  }

  // Validate groupings.
  for (const auto& grouping : aggRel.groupings()) {
    for (const auto& groupingExpr : grouping.grouping_expressions()) {
      const auto& typeCase = groupingExpr.rex_type_case();
      switch (typeCase) {
        case ::substrait::Expression::RexTypeCase::kSelection:
          break;
        default:
          VLOG(0) << "Only field is supported in groupings.";
          return false;
      }
    }
  }

  // Validate aggregate functions.
  std::vector<std::string> funcSpecs;
  funcSpecs.reserve(aggRel.measures().size());
  for (const auto& smea : aggRel.measures()) {
    try {
      const auto& aggFunction = smea.measure();
      funcSpecs.emplace_back(
          planConverter_->findFuncSpec(aggFunction.function_reference()));
      toVeloxType(subParser_->parseType(aggFunction.output_type())->type);
      for (const auto& arg : aggFunction.args()) {
        auto typeCase = arg.rex_type_case();
        switch (typeCase) {
          case ::substrait::Expression::RexTypeCase::kSelection:
            break;
          default:
            VLOG(0) << "Only field is supported in aggregate functions.";
            return false;
        }
      }
    } catch (const VeloxException& err) {
      VLOG(0) << "Validation failed for aggregate function due to: "
              << err.message();
      return false;
    }
  }

  std::unordered_set<std::string> supportedFuncs = {"sum", "count", "avg"};
  for (const auto& funcSpec : funcSpecs) {
    auto funcName = subParser_->getSubFunctionName(funcSpec);
    if (supportedFuncs.find(funcName) == supportedFuncs.end()) {
      VLOG(0) << "Validation failed due to " << funcName
              << " was not supported in AggregateRel.";
      return false;
    }
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::ReadRel& readRel) {
  if (!readRel.has_base_schema()) {
    VLOG(0) << "Validation failed due to schema was not found in ReadRel.";
    return false;
  }
  const auto& substraitTypes = readRel.base_schema().struct_().types();
  for (const auto& substraitType : substraitTypes) {
    if (!validate(substraitType)) {
      VLOG(0) << "Validation failed due to type was not supported in ReadRel.";
      return false;
    }
  }
  std::vector<::substrait::Expression_ScalarFunction> scalarFunctions;
  if (readRel.has_filter()) {
    try {
      planConverter_->flattenConditions(readRel.filter(), scalarFunctions);
    } catch (const VeloxException& err) {
      VLOG(0)
          << "Validation failed due to flattening conditions failed in ReadRel due to:"
          << err.message();
      return false;
    }
  }
  // Get and validate the filter functions.
  std::vector<std::string> funcSpecs;
  funcSpecs.reserve(scalarFunctions.size());
  for (const auto& scalarFunction : scalarFunctions) {
    try {
      funcSpecs.emplace_back(
          planConverter_->findFuncSpec(scalarFunction.function_reference()));
    } catch (const VeloxException& err) {
      VLOG(0) << "Validation failed in ReadRel due to:" << err.message();
      return false;
    }

    if (scalarFunction.args().size() == 1) {
      // Field is expected.
      for (const auto& param : scalarFunction.args()) {
        auto typeCase = param.rex_type_case();
        switch (typeCase) {
          case ::substrait::Expression::RexTypeCase::kSelection:
            break;
          default:
            VLOG(0) << "Field is Expected.";
            return false;
        }
      }
    } else if (scalarFunction.args().size() == 2) {
      // Expect there being two args. One is field and the other is literal.
      bool fieldExists = false;
      bool litExists = false;
      for (const auto& param : scalarFunction.args()) {
        auto typeCase = param.rex_type_case();
        switch (typeCase) {
          case ::substrait::Expression::RexTypeCase::kSelection: {
            fieldExists = true;
            break;
          }
          case ::substrait::Expression::RexTypeCase::kLiteral: {
            litExists = true;
            break;
          }
          default:
            VLOG(0) << "Type case: " << typeCase
                    << " is not supported in ReadRel.";
            return false;
        }
      }
      if (!fieldExists || !litExists) {
        VLOG(0) << "Only the case of Field and Literal is supported.";
        return false;
      }
    } else {
      VLOG(0) << "More than two args is not supported in ReadRel.";
      return false;
    }
  }
  std::unordered_set<std::string> supportedFilters = {
      "is_not_null", "gte", "gt", "lte", "lt"};
  std::unordered_set<std::string> supportedTypes = {"opt", "req", "fp64"};
  for (const auto& funcSpec : funcSpecs) {
    // Validate the functions.
    auto funcName = subParser_->getSubFunctionName(funcSpec);
    if (supportedFilters.find(funcName) == supportedFilters.end()) {
      VLOG(0) << "Validation failed due to " << funcName
              << " was not supported in ReadRel.";
      return false;
    }

    // Validate the types.
    std::vector<std::string> funcTypes;
    subParser_->getSubFunctionTypes(funcSpec, funcTypes);
    for (const auto& funcType : funcTypes) {
      if (supportedTypes.find(funcType) == supportedTypes.end()) {
        VLOG(0) << "Validation failed due to " << funcType
                << " was not supported in ReadRel.";
        return false;
      }
    }
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::Rel& substraitRel) {
  if (substraitRel.has_aggregate()) {
    return validate(substraitRel.aggregate());
  }
  if (substraitRel.has_project()) {
    return validate(substraitRel.project());
  }
  if (substraitRel.has_filter()) {
    return validate(substraitRel.filter());
  }
  if (substraitRel.has_read()) {
    return validate(substraitRel.read());
  }
  return false;
}

bool SubstraitToVeloxPlanValidator::validate(const ::substrait::RelRoot& root) {
  if (root.has_input()) {
    const auto& substraitRel = root.input();
    return validate(substraitRel);
  }
  return false;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::Plan& substraitPlan) {
  functions::prestosql::registerAllScalarFunctions();
  // Create plan converter and expression converter to help the validation.
  planConverter_->constructFuncMap(substraitPlan);
  exprConverter_ = std::make_shared<SubstraitVeloxExprConverter>(
      planConverter_->getFunctionMap());

  for (const auto& substraitRel : substraitPlan.relations()) {
    if (substraitRel.has_root()) {
      return validate(substraitRel.root());
    }
    if (substraitRel.has_rel()) {
      return validate(substraitRel.rel());
    }
  }
  return false;
}

} // namespace facebook::velox::substrait
