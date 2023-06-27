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
#include <google/protobuf/wrappers.pb.h>
#include <string>
#include "TypeUtils.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/Tokenizer.h"

namespace facebook::velox::substrait {
namespace {
bool validateColNames(const ::substrait::NamedStruct& schema) {
  for (auto& name : schema.names()) {
    common::Tokenizer token(name);
    for (auto i = 0; i < name.size(); i++) {
      auto c = name[i];
      if (!token.isUnquotedPathCharacter(c)) {
        std::cout
            << "native validation failed due to: Illegal column charactor " << c
            << "in column " << name << std::endl;
        return false;
      }
    }
  }
  return true;
}
} // namespace
bool SubstraitToVeloxPlanValidator::validateInputTypes(
    const ::substrait::extensions::AdvancedExtension& extension,
    std::vector<TypePtr>& types) {
  // The input type is wrapped in enhancement.
  if (!extension.has_enhancement()) {
    logValidateMsg(
        "native validation failed due to: {input type is not wrapped in enhancement}.");
    return false;
  }
  const auto& enhancement = extension.enhancement();
  ::substrait::Type inputType;
  if (!enhancement.UnpackTo(&inputType)) {
    logValidateMsg(
        "native validation failed due to: {enhancement can't be unpacked to inputType}.");
    return false;
  }
  if (!inputType.has_struct_()) {
    logValidateMsg(
        "native validation failed due to: {input type has no struct}.");
    return false;
  }

  // Get the input types.
  const auto& sTypes = inputType.struct_().types();
  for (const auto& sType : sTypes) {
    try {
      types.emplace_back(toVeloxType(subParser_->parseType(sType)->type));
    } catch (const VeloxException& err) {
      logValidateMsg(
          "native validation failed due to: Type is not supported, " +
          err.message());
      return false;
    }
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validateRound(
    const ::substrait::Expression::ScalarFunction& scalarFunction,
    const RowTypePtr& inputType) {
  const auto& arguments = scalarFunction.arguments();
  if (arguments.size() < 2) {
    return false;
  }
  if (!arguments[1].value().has_literal()) {
    logValidateMsg(
        "native validation failued due to: Round scale is expected.");
    return false;
  }
  // Velox has different result with Spark on negative scale.
  auto typeCase = arguments[1].value().literal().literal_type_case();
  switch (typeCase) {
    case ::substrait::Expression_Literal::LiteralTypeCase::kI32:
      return (arguments[1].value().literal().i32() >= 0);
    case ::substrait::Expression_Literal::LiteralTypeCase::kI64:
      return (arguments[1].value().literal().i64() >= 0);
    default:
      logValidateMsg(
          "native validation failed due to: Round scale validation is not supported for type case '{}'" +
          typeCase);
      return false;
  }
}

bool SubstraitToVeloxPlanValidator::validateExtractExpr(
    const std::vector<std::shared_ptr<const core::ITypedExpr>>& params) {
  if (params.size() != 2) {
    logValidateMsg(
        "native validation failed due to: Value expected in variant in ExtractExpr.");
    return false;
  }
  auto functionArg =
      std::dynamic_pointer_cast<const core::ConstantTypedExpr>(params[0]);
  if (functionArg) {
    // Get the function argument.
    auto variant = functionArg->value();
    if (!variant.hasValue()) {
      logValidateMsg(
          "native validation failed due to: Value expected in variant in ExtractExpr.");
      return false;
    }
    // The first parameter specifies extracting from which field.
    std::string from = variant.value<std::string>();
    // Hour causes incorrect result.
    if (from == "HOUR") {
      logValidateMsg("native validation failed due to: {extract from hour}.");
      return false;
    }
    return true;
  }
  logValidateMsg(
      "native validation failed due to: Constant is expected to be the first parameter in extract.");
  return false;
}

bool SubstraitToVeloxPlanValidator::validateScalarFunction(
    const ::substrait::Expression::ScalarFunction& scalarFunction,
    const RowTypePtr& inputType) {
  std::vector<core::TypedExprPtr> params;
  params.reserve(scalarFunction.arguments().size());
  for (const auto& argument : scalarFunction.arguments()) {
    if (argument.has_value() &&
        !validateExpression(argument.value(), inputType)) {
      return false;
    }
    params.emplace_back(
        exprConverter_->toVeloxExpr(argument.value(), inputType));
  }

  const auto& function = subParser_->findFunctionSpec(
      planConverter_->getFunctionMap(), scalarFunction.function_reference());
  const auto& name = subParser_->getSubFunctionName(function);
  std::vector<std::string> types;
  subParser_->getSubFunctionTypes(function, types);
  if (name == "round") {
    return validateRound(scalarFunction, inputType);
  }
  if (name == "extract") {
    return validateExtractExpr(params);
  }
  if (name == "regexp_extract_all" &&
      !scalarFunction.arguments()[1].value().has_literal()) {
    logValidateMsg(
        "native validation failed due to: pattern is not literal for regex_extract_all.");
    return false;
  }
  if (name == "char_length") {
    VELOX_CHECK(types.size() == 1);
    if (types[0] == "vbin") {
      logValidateMsg(
          "native validation failed due to: Binary type is not supported in " +
          name);
      return false;
    }
  }
  if (name == "murmur3hash") {
    for (const auto& type : types) {
      if (type.find("struct") != std::string::npos ||
          type.find("map") != std::string::npos ||
          type.find("list") != std::string::npos) {
        logValidateMsg(
            "native validation failed due to: " + type +
            "is not supported in murmur3hash.");
        return false;
      }
    }
  }
  std::unordered_set<std::string> functions = {
      "regexp_replace",    "split",         "split_part",
      "factorial",         "concat_ws",     "rand",
      "json_array_length", "from_unixtime", "to_unix_timestamp",
      "unix_timestamp",    "repeat",        "translate",
      "add_months",        "date_format",   "trunc",
      "sequence",          "posexplode",    "arrays_overlap",
      "array_min",         "array_max"};
  if (functions.find(name) != functions.end()) {
    logValidateMsg(
        "native validation failed due to: Function is not supported: " + name);
    return false;
  }

  return true;
}

bool SubstraitToVeloxPlanValidator::validateLiteral(
    const ::substrait::Expression_Literal& literal,
    const RowTypePtr& inputType) {
  if (literal.has_list() && literal.list().values_size() == 0) {
    logValidateMsg(
        "native validation failed due to: literal is a list but has no value.");
    return false;
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validateCast(
    const ::substrait::Expression::Cast& castExpr,
    const RowTypePtr& inputType) {
  if (!validateExpression(castExpr.input(), inputType)) {
    return false;
  }

  const auto& toType =
      toVeloxType(subParser_->parseType(castExpr.type())->type);
  if (toType->kind() == TypeKind::TIMESTAMP) {
    logValidateMsg(
        "native validation failed due to: Casting to TIMESTAMP is not supported.");
    return false;
  }

  core::TypedExprPtr input =
      exprConverter_->toVeloxExpr(castExpr.input(), inputType);

  // Casting from some types is not supported. See CastExpr::applyCast.
  switch (input->type()->kind()) {
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
    case TypeKind::VARBINARY:
      logValidateMsg(
          "native validation failed due to: Invalid input type in casting: ARRAY/MAP/ROW/VARBINARY");
      return false;
    case TypeKind::DATE: {
      if (toType->kind() == TypeKind::TIMESTAMP) {
        logValidateMsg(
            "native validation failed due to: Casting from DATE to TIMESTAMP is not supported.");
        return false;
      }
    }
    case TypeKind::TIMESTAMP: {
      logValidateMsg(
          "native validation failed due to: Casting from TIMESTAMP is not supported or has incorrect result.");
      return false;
    }
    default: {
    }
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validateExpression(
    const ::substrait::Expression& expression,
    const RowTypePtr& inputType) {
  std::shared_ptr<const core::ITypedExpr> veloxExpr;
  auto typeCase = expression.rex_type_case();
  switch (typeCase) {
    case ::substrait::Expression::RexTypeCase::kScalarFunction:
      return validateScalarFunction(expression.scalar_function(), inputType);
    case ::substrait::Expression::RexTypeCase::kLiteral:
      return validateLiteral(expression.literal(), inputType);
    case ::substrait::Expression::RexTypeCase::kCast:
      return validateCast(expression.cast(), inputType);
    default:
      return true;
  }
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::FetchRel& fetchRel) {
  const auto& extension = fetchRel.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    logValidateMsg(
        "native validation failed due to: unsupported input types in FetchRel.");
    return false;
  }

  if (fetchRel.offset() < 0 || fetchRel.count() < 0) {
    logValidateMsg(
        "native validation failed due to: Offset and count should be valid in FetchRel.");
    return false;
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::ExpandRel& expandRel) {
  if (expandRel.has_input() && !validate(expandRel.input())) {
    logValidateMsg(
        "native validation failed due to: input validation fails in ExpandRel.");
    return false;
  }
  RowTypePtr rowType = nullptr;
  // Get and validate the input types from extension.
  if (expandRel.has_advanced_extension()) {
    const auto& extension = expandRel.advanced_extension();
    std::vector<TypePtr> types;
    if (!validateInputTypes(extension, types)) {
      logValidateMsg(
          "native validation failed due to: unsupported input types in ExpandRel.");
      return false;
    }
    int32_t inputPlanNodeId = 0;
    std::vector<std::string> names;
    names.reserve(types.size());
    for (auto colIdx = 0; colIdx < types.size(); colIdx++) {
      names.emplace_back(subParser_->makeNodeName(inputPlanNodeId, colIdx));
    }
    rowType = std::make_shared<RowType>(std::move(names), std::move(types));
  }

  int32_t projectSize = 0;
  // Validate fields.
  for (const auto& fields : expandRel.fields()) {
    std::vector<core::TypedExprPtr> expressions;
    if (fields.has_switching_field()) {
      auto projectExprs = fields.switching_field().duplicates();
      expressions.reserve(projectExprs.size());
      if (projectSize == 0) {
        projectSize = projectExprs.size();
      } else if (projectSize != projectExprs.size()) {
        logValidateMsg(
            "native validation failed due to: SwitchingField expressions size should be constant in ExpandRel.");
        return false;
      }

      try {
        for (const auto& projectExpr : projectExprs) {
          const auto& typeCase = projectExpr.rex_type_case();
          switch (typeCase) {
            case ::substrait::Expression::RexTypeCase::kSelection:
            case ::substrait::Expression::RexTypeCase::kLiteral:
              break;
            default:
              logValidateMsg(
                  "native validation failed due to: Only field or literal is supported in project of ExpandRel.");
              return false;
          }
          if (rowType) {
            expressions.emplace_back(
                exprConverter_->toVeloxExpr(projectExpr, rowType));
          }
        }

        if (rowType) {
          // Try to compile the expressions. If there is any unregistered
          // function or mismatched type, exception will be thrown.
          exec::ExprSet exprSet(std::move(expressions), execCtx_);
        }

      } catch (const VeloxException& err) {
        logValidateMsg(
            "native validation failed due to: in ExpandRel, " + err.message());
        return false;
      }
    } else {
      logValidateMsg(
          "native validation failed due to: Only SwitchingField is supported in ExpandRel.");
      return false;
    }
  }

  return true;
}

bool validateBoundType(::substrait::Expression_WindowFunction_Bound boundType) {
  switch (boundType.kind_case()) {
    case ::substrait::Expression_WindowFunction_Bound::kUnboundedFollowing:
    case ::substrait::Expression_WindowFunction_Bound::kUnboundedPreceding:
    case ::substrait::Expression_WindowFunction_Bound::kCurrentRow:
    case ::substrait::Expression_WindowFunction_Bound::kFollowing:
    case ::substrait::Expression_WindowFunction_Bound::kPreceding:
      break;
    default:
      VLOG(1)
          << "native validation failed due to: The Bound Type is not supported. ";
      return false;
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::WindowRel& windowRel) {
  if (windowRel.has_input() && !validate(windowRel.input())) {
    logValidateMsg(
        "native validation failed due to: windowRel input fails to validate. ");
    return false;
  }

  // Get and validate the input types from extension.
  if (!windowRel.has_advanced_extension()) {
    logValidateMsg(
        "native validation failed due to: Input types are expected in WindowRel.");
    return false;
  }
  const auto& extension = windowRel.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    logValidateMsg(
        "native validation failed due to: Validation failed for input types in WindowRel.");
    return false;
  }

  int32_t inputPlanNodeId = 0;
  std::vector<std::string> names;
  names.reserve(types.size());
  for (auto colIdx = 0; colIdx < types.size(); colIdx++) {
    names.emplace_back(subParser_->makeNodeName(inputPlanNodeId, colIdx));
  }
  auto rowType = std::make_shared<RowType>(std::move(names), std::move(types));

  // Validate WindowFunction
  std::vector<std::string> funcSpecs;
  funcSpecs.reserve(windowRel.measures().size());
  for (const auto& smea : windowRel.measures()) {
    try {
      const auto& windowFunction = smea.measure();
      funcSpecs.emplace_back(
          planConverter_->findFuncSpec(windowFunction.function_reference()));
      toVeloxType(subParser_->parseType(windowFunction.output_type())->type);
      for (const auto& arg : windowFunction.arguments()) {
        auto typeCase = arg.value().rex_type_case();
        switch (typeCase) {
          case ::substrait::Expression::RexTypeCase::kSelection:
          case ::substrait::Expression::RexTypeCase::kLiteral:
            break;
          default:
            logValidateMsg(
                "native validation failed due to: Only field is supported in window functions.");
            return false;
        }
      }
      // Validate BoundType and Frame Type
      switch (windowFunction.window_type()) {
        case ::substrait::WindowType::ROWS:
        case ::substrait::WindowType::RANGE:
          break;
        default:
          logValidateMsg(
              "native validation failed due to: the window type only support ROWS and RANGE, and the input type is " +
              windowFunction.window_type());
          return false;
      }

      bool boundTypeSupported =
          validateBoundType(windowFunction.upper_bound()) &&
          validateBoundType(windowFunction.lower_bound());
      if (!boundTypeSupported) {
        logValidateMsg(
            "native validation failed due to: The Bound Type is not supported. " +
            windowFunction.lower_bound().kind_case());
        return false;
      }
    } catch (const VeloxException& err) {
      logValidateMsg(
          "native validation failed due to: in window function, " +
          err.message());
      return false;
    }
  }

  // Validate supported aggregate functions.
  std::unordered_set<std::string> unsupportedFuncs = {"collect_list"};
  for (const auto& funcSpec : funcSpecs) {
    auto funcName = subParser_->getSubFunctionName(funcSpec);
    if (unsupportedFuncs.find(funcName) != unsupportedFuncs.end()) {
      logValidateMsg(
          "native validation failed due to: " + funcName +
          " was not supported in WindowRel.");
      return false;
    }
  }

  // Validate groupby expression
  const auto& groupByExprs = windowRel.partition_expressions();
  std::vector<std::shared_ptr<const core::ITypedExpr>> expressions;
  expressions.reserve(groupByExprs.size());
  try {
    for (const auto& expr : groupByExprs) {
      auto expression = exprConverter_->toVeloxExpr(expr, rowType);
      auto expr_field =
          dynamic_cast<const core::FieldAccessTypedExpr*>(expression.get());
      if (expr_field == nullptr) {
        logValidateMsg(
            "native validation failed due to: Only field is supported for partition key in Window Operator!");
        return false;
      } else {
        expressions.emplace_back(expression);
      }
    }
    // Try to compile the expressions. If there is any unregistred funciton or
    // mismatched type, exception will be thrown.
    exec::ExprSet exprSet(std::move(expressions), execCtx_);
  } catch (const VeloxException& err) {
    logValidateMsg(
        "native validation failed due to: in WindowRel, " + err.message());
    return false;
  }

  // Validate Sort expression
  const auto& sorts = windowRel.sorts();
  for (const auto& sort : sorts) {
    switch (sort.direction()) {
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_ASC_NULLS_FIRST:
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_ASC_NULLS_LAST:
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_DESC_NULLS_FIRST:
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_DESC_NULLS_LAST:
        break;
      default:
        logValidateMsg(
            "native validation failed due to: in windowRel, unsupported Sort direction " +
            sort.direction());
        return false;
    }

    if (sort.has_expr()) {
      try {
        auto expression = exprConverter_->toVeloxExpr(sort.expr(), rowType);
        auto expr_field =
            dynamic_cast<const core::FieldAccessTypedExpr*>(expression.get());
        if (!expr_field) {
          logValidateMsg(
              "native validation failed due to: in windowRel, the sorting key in Sort Operator only support field");
          return false;
        }
        exec::ExprSet exprSet({std::move(expression)}, execCtx_);
      } catch (const VeloxException& err) {
        logValidateMsg(
            "native validation failed due to: in WindowRel, " + err.message());
        return false;
      }
    }
  }

  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::SortRel& sortRel) {
  if (sortRel.has_input() && !validate(sortRel.input())) {
    return false;
  }
  // Get and validate the input types from extension.
  if (!sortRel.has_advanced_extension()) {
    logValidateMsg(
        "native validation failed due to: Input types are expected in SortRel.");
    return false;
  }
  const auto& extension = sortRel.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    logValidateMsg(
        "native validation failed due to: Validation failed for input types in SortRel.");
    return false;
  }
  for (const auto& type : types) {
    if (type->kind() == TypeKind::TIMESTAMP) {
      logValidateMsg(
          "native validation failed due to: Timestamp type is not supported in SortRel.");
      return false;
    }
  }

  int32_t inputPlanNodeId = 0;
  std::vector<std::string> names;
  names.reserve(types.size());
  for (auto colIdx = 0; colIdx < types.size(); colIdx++) {
    names.emplace_back(subParser_->makeNodeName(inputPlanNodeId, colIdx));
  }
  auto rowType = std::make_shared<RowType>(std::move(names), std::move(types));

  const auto& sorts = sortRel.sorts();
  for (const auto& sort : sorts) {
    switch (sort.direction()) {
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_ASC_NULLS_FIRST:
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_ASC_NULLS_LAST:
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_DESC_NULLS_FIRST:
      case ::substrait::SortField_SortDirection_SORT_DIRECTION_DESC_NULLS_LAST:
        break;
      default:
        logValidateMsg(
            "native validation failed due to: in sortRel, unsupported Sort direction " +
            sort.direction());
        return false;
    }

    if (sort.has_expr()) {
      try {
        auto expression = exprConverter_->toVeloxExpr(sort.expr(), rowType);
        auto expr_field =
            dynamic_cast<const core::FieldAccessTypedExpr*>(expression.get());
        if (!expr_field) {
          logValidateMsg(
              "native validation failed due to: in SortRel, the sorting key in Sort Operator only support field");
          return false;
        }
        exec::ExprSet exprSet({std::move(expression)}, execCtx_);
      } catch (const VeloxException& err) {
        logValidateMsg(
            "native validation failed due to: in SortRel, expression validation fails as" +
            err.message());
        return false;
      }
    }
  }

  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::ProjectRel& projectRel) {
  if (projectRel.has_input() && !validate(projectRel.input())) {
    logValidateMsg(
        "native validation failed due to: Validation failed for ProjectRel input.");
    return false;
  }

  // Get and validate the input types from extension.
  if (!projectRel.has_advanced_extension()) {
    logValidateMsg(
        "native validation failed due to: Input types are expected in ProjectRel.");
    return false;
  }
  const auto& extension = projectRel.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    logValidateMsg(
        "native validation failed due to: Validation failed for input types in ProjectRel.");
    return false;
  }

  for (auto i = 0; i < types.size(); i++) {
    switch (types[i]->kind()) {
      case TypeKind::ARRAY:
        logValidateMsg(
            "native validation failed due to: unsupported input type ARRAY in ProjectRel.");
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
  const auto& projectExprs = projectRel.expressions();
  std::vector<std::shared_ptr<const core::ITypedExpr>> expressions;
  expressions.reserve(projectExprs.size());
  try {
    for (const auto& expr : projectExprs) {
      if (!validateExpression(expr, rowType)) {
        return false;
      }
      expressions.emplace_back(exprConverter_->toVeloxExpr(expr, rowType));
    }
    // Try to compile the expressions. If there is any unregistered function or
    // mismatched type, exception will be thrown.
    exec::ExprSet exprSet(std::move(expressions), execCtx_);
  } catch (const VeloxException& err) {
    logValidateMsg(
        "native validation failed due to: in ProjectRel, " + err.message());
    return false;
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::FilterRel& filterRel) {
  if (filterRel.has_input() && !validate(filterRel.input())) {
    logValidateMsg(
        "native validation failed due to: input of FilterRel validation fails");
    return false;
  }

  // Get and validate the input types from extension.
  if (!filterRel.has_advanced_extension()) {
    logValidateMsg(
        "native validation failed due to: Input types are expected in FilterRel.");
    return false;
  }
  const auto& extension = filterRel.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    logValidateMsg(
        "native validation failed due to: Validation failed for input types in FilterRel.");
    return false;
  }
  for (const auto& type : types) {
    if (type->kind() == TypeKind::TIMESTAMP) {
      logValidateMsg(
          "native validation failed due to: Timestamp is not fully supported in Filter");
      return false;
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

  std::vector<std::shared_ptr<const core::ITypedExpr>> expressions;
  expressions.reserve(1);
  try {
    if (!validateExpression(filterRel.condition(), rowType)) {
      return false;
    }
    expressions.emplace_back(
        exprConverter_->toVeloxExpr(filterRel.condition(), rowType));
    // Try to compile the expressions. If there is any unregistered function
    // or mismatched type, exception will be thrown.
    exec::ExprSet exprSet(std::move(expressions), execCtx_);
  } catch (const VeloxException& err) {
    logValidateMsg(
        "native validation failed due to: validation fails for FilterRel expression, " +
        err.message());
    return false;
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::JoinRel& joinRel) {
  if (joinRel.has_left() && !validate(joinRel.left())) {
    logValidateMsg(
        "native validation failed due to: validation fails for join left input. ");
    return false;
  }
  if (joinRel.has_right() && !validate(joinRel.right())) {
    logValidateMsg(
        "native validation failed due to: validation fails for join right input. ");
    return false;
  }

  if (joinRel.has_advanced_extension() &&
      subParser_->configSetInOptimization(
          joinRel.advanced_extension(), "isSMJ=")) {
    switch (joinRel.type()) {
      case ::substrait::JoinRel_JoinType_JOIN_TYPE_INNER:
      case ::substrait::JoinRel_JoinType_JOIN_TYPE_LEFT:
        break;
      default:
        logValidateMsg(
            "native validation failed due to: Sort merge join only support inner and left join");
        return false;
    }
  }
  switch (joinRel.type()) {
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_INNER:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_OUTER:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_LEFT:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_RIGHT:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_LEFT_SEMI:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_RIGHT_SEMI:
    case ::substrait::JoinRel_JoinType_JOIN_TYPE_ANTI:
      break;
    default:
      logValidateMsg(
          "native validation failed due to: Sort merge join only support inner and left join");
      return false;
  }

  // Validate input types.
  if (!joinRel.has_advanced_extension()) {
    logValidateMsg(
        "native validation failed due to: Input types are expected in JoinRel.");
    return false;
  }

  const auto& extension = joinRel.advanced_extension();
  std::vector<TypePtr> types;
  if (!validateInputTypes(extension, types)) {
    logValidateMsg(
        "native validation failed due to: Validation failed for input types in JoinRel");
    return false;
  }

  int32_t inputPlanNodeId = 0;
  std::vector<std::string> names;
  names.reserve(types.size());
  for (auto colIdx = 0; colIdx < types.size(); colIdx++) {
    names.emplace_back(subParser_->makeNodeName(inputPlanNodeId, colIdx));
  }
  auto rowType = std::make_shared<RowType>(std::move(names), std::move(types));

  if (joinRel.has_expression()) {
    std::vector<const ::substrait::Expression::FieldReference*> leftExprs,
        rightExprs;
    try {
      planConverter_->extractJoinKeys(
          joinRel.expression(), leftExprs, rightExprs);
    } catch (const VeloxException& err) {
      logValidateMsg(
          "native validation failed due to: JoinRel expression validation fails, " +
          err.message());
      return false;
    }
  }

  if (joinRel.has_post_join_filter()) {
    try {
      auto expression =
          exprConverter_->toVeloxExpr(joinRel.post_join_filter(), rowType);
      exec::ExprSet exprSet({std::move(expression)}, execCtx_);
    } catch (const VeloxException& err) {
      logValidateMsg(
          "native validation failed due to: in post join filter, " +
          err.message());
      return false;
    }
  }
  return true;
}

TypePtr SubstraitToVeloxPlanValidator::getDecimalType(
    const std::string& decimalType) {
  // Decimal info is in the format of dec<precision,scale>.
  auto precisionStart = decimalType.find_first_of('<');
  auto tokenIndex = decimalType.find_first_of(',');
  auto scaleStart = decimalType.find_first_of('>');
  auto precision = stoi(decimalType.substr(
      precisionStart + 1, (tokenIndex - precisionStart - 1)));
  auto scale =
      stoi(decimalType.substr(tokenIndex + 1, (scaleStart - tokenIndex - 1)));

  if (precision <= 18) {
    return SHORT_DECIMAL(precision, scale);
  } else {
    return LONG_DECIMAL(precision, scale);
  }
}

TypePtr SubstraitToVeloxPlanValidator::getRowType(
    const std::string& structType) {
  // Struct info is in the format of struct<T1,T2, ...,Tn>.
  // TODO: nested struct is not supported.
  auto structStart = structType.find_first_of('<');
  auto structEnd = structType.find_last_of('>');
  if (structEnd - structStart > 1) {
    logValidateMsg(
        "native validation failed due to: More information is needed to create RowType");
  }
  VELOX_CHECK(
      structEnd - structStart > 1,
      "native validation failed due to: More information is needed to create RowType");
  std::string childrenTypes =
      structType.substr(structStart + 1, structEnd - structStart - 1);

  // Split the types with delimiter.
  std::string delimiter = ",";
  std::size_t pos;
  std::vector<TypePtr> types;
  std::vector<std::string> names;
  while ((pos = childrenTypes.find(delimiter)) != std::string::npos) {
    const auto& typeStr = childrenTypes.substr(0, pos);
    std::string decDelimiter = ">";
    if (typeStr.find("dec") != std::string::npos) {
      std::size_t endPos = childrenTypes.find(decDelimiter);
      VELOX_CHECK(endPos >= pos + 1, "Decimal scale is expected.");
      const auto& decimalStr =
          typeStr + childrenTypes.substr(pos, endPos - pos) + decDelimiter;
      types.emplace_back(getDecimalType(decimalStr));
      names.emplace_back("");
      childrenTypes.erase(
          0, endPos + delimiter.length() + decDelimiter.length());
      continue;
    }

    types.emplace_back(toVeloxType(subParser_->parseType(typeStr)));
    names.emplace_back("");
    childrenTypes.erase(0, pos + delimiter.length());
  }
  types.emplace_back(toVeloxType(subParser_->parseType(childrenTypes)));
  names.emplace_back("");
  return std::make_shared<RowType>(std::move(names), std::move(types));
}

bool SubstraitToVeloxPlanValidator::validateAggRelFunctionType(
    const ::substrait::AggregateRel& aggRel) {
  if (aggRel.measures_size() == 0) {
    return true;
  }

  for (const auto& smea : aggRel.measures()) {
    const auto& aggFunction = smea.measure();
    auto funcSpec =
        planConverter_->findFuncSpec(aggFunction.function_reference());
    std::vector<TypePtr> types;
    bool isDecimal = false;
    try {
      std::vector<std::string> funcTypes;
      subParser_->getSubFunctionTypes(funcSpec, funcTypes);
      types.reserve(funcTypes.size());
      for (auto& type : funcTypes) {
        if (!isDecimal && type.find("dec") != std::string::npos) {
          isDecimal = true;
        }
        if (type.find("struct") != std::string::npos) {
          types.emplace_back(getRowType(type));
        } else if (type.find("dec") != std::string::npos) {
          types.emplace_back(getDecimalType(type));
        } else {
          types.emplace_back(toVeloxType(subParser_->parseType(type)));
        }
      }
    } catch (const VeloxException& err) {
      logValidateMsg(
          "native validation failed due to: Validation failed for input type in AggregateRel function due to:" +
          err.message());
      return false;
    }
    auto funcName = subParser_->mapToVeloxFunction(
        subParser_->getSubFunctionName(funcSpec), isDecimal);
    if (auto signatures = exec::getAggregateFunctionSignatures(funcName)) {
      for (const auto& signature : signatures.value()) {
        exec::SignatureBinder binder(*signature, types);
        if (binder.tryBind()) {
          auto resolveType = binder.tryResolveType(
              exec::isPartialOutput(planConverter_->toAggregationStep(aggRel))
                  ? signature->intermediateType()
                  : signature->returnType());
          if (resolveType == nullptr) {
            logValidateMsg(
                "native validation failed due to: Validation failed for function " +
                funcName + "resolve type in AggregateRel.");
            return false;
          }
          return true;
        }
      }
      logValidateMsg(
          "native validation failed due to: Validation failed for function " +
          funcName + " bind in AggregateRel.");
      return false;
    }
  }
  logValidateMsg(
      "native validation failed due to: Validation failed for function resolve in AggregateRel.");
  return false;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::AggregateRel& aggRel) {
  if (aggRel.has_input() && !validate(aggRel.input())) {
    logValidateMsg(
        "native validation failed due to: input validation fails in AggregateRel.");
    return false;
  }

  // Validate input types.
  if (aggRel.has_advanced_extension()) {
    std::vector<TypePtr> types;
    const auto& extension = aggRel.advanced_extension();
    if (!validateInputTypes(extension, types)) {
      logValidateMsg(
          "native validation failed due to: Validation failed for input types in AggregateRel.");
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
          logValidateMsg(
              "native validation failed due to: Only field is supported in groupings.");
          return false;
      }
    }
  }

  // Validate aggregate functions.
  std::vector<std::string> funcSpecs;
  funcSpecs.reserve(aggRel.measures().size());
  for (const auto& smea : aggRel.measures()) {
    try {
      // Validate the filter expression
      if (smea.has_filter()) {
        ::substrait::Expression aggRelMask = smea.filter();
        if (aggRelMask.ByteSizeLong() > 0) {
          auto typeCase = aggRelMask.rex_type_case();
          switch (typeCase) {
            case ::substrait::Expression::RexTypeCase::kSelection:
              break;
            default:
              logValidateMsg(
                  "native validation failed due to: Only field is supported in aggregate filter expression.");
              return false;
          }
        }
      }

      const auto& aggFunction = smea.measure();
      const auto& functionSpec =
          planConverter_->findFuncSpec(aggFunction.function_reference());
      funcSpecs.emplace_back(functionSpec);
      toVeloxType(subParser_->parseType(aggFunction.output_type())->type);
      // Validate the size of arguments.
      if (subParser_->getSubFunctionName(functionSpec) == "count" &&
          aggFunction.arguments().size() > 1) {
        logValidateMsg(
            "native validation failed due to: count should have only one argument");
        // Count accepts only one argument.
        return false;
      }
      for (const auto& arg : aggFunction.arguments()) {
        auto typeCase = arg.value().rex_type_case();
        switch (typeCase) {
          case ::substrait::Expression::RexTypeCase::kSelection:
          case ::substrait::Expression::RexTypeCase::kLiteral:
            break;
          default:
            logValidateMsg(
                "native validation failed due to: Only field is supported in aggregate functions.");
            return false;
        }
      }
    } catch (const VeloxException& err) {
      logValidateMsg(
          "native validation failed due to: aggregate function validation fails, " +
          err.message());
      return false;
    }
  }

  std::unordered_set<std::string> supportedFuncs = {
      "sum",
      "sum_merge",
      "count",
      "count_merge",
      "avg",
      "avg_merge",
      "min",
      "min_merge",
      "max",
      "max_merge",
      "stddev_samp",
      "stddev_samp_merge",
      "stddev_pop",
      "stddev_pop_merge",
      "bloom_filter_agg",
      "var_samp",
      "var_samp_merge",
      "var_pop",
      "var_pop_merge",
      "bit_and",
      "bit_and_merge",
      "bit_or",
      "bit_or_merge",
      "bit_xor",
      "bit_xor_merge",
      "first",
      "first_merge",
      "first_ignore_null",
      "first_ignore_null_merge",
      "last",
      "last_merge",
      "last_ignore_null",
      "last_ignore_null_merge",
      "corr",
      "corr_merge",
      "covar_pop",
      "covar_pop_merge",
      "covar_samp",
      "covar_samp_merge",
      "approx_distinct"};
  for (const auto& funcSpec : funcSpecs) {
    auto funcName = subParser_->getSubFunctionName(funcSpec);
    if (supportedFuncs.find(funcName) == supportedFuncs.end()) {
      logValidateMsg(
          "native validation failed due to:  " + funcName +
          " was not supported in AggregateRel.");
      return false;
    }
  }

  if (!validateAggRelFunctionType(aggRel)) {
    return false;
  }

  // Validate both groupby and aggregates input are empty, which is corner case.
  if (aggRel.measures_size() == 0) {
    bool hasExpr = false;
    for (const auto& grouping : aggRel.groupings()) {
      for (const auto& groupingExpr : grouping.grouping_expressions()) {
        hasExpr = true;
        break;
      }
      if (hasExpr) {
        break;
      }
    }
    if (!hasExpr) {
      logValidateMsg(
          "native validation failed due to: aggregation must specify either grouping keys or aggregates.");
      return false;
    }
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::ReadRel& readRel) {
  try {
    planConverter_->toVeloxPlan(readRel);
  } catch (const VeloxException& err) {
    logValidateMsg(
        "native validation failed due to: in ReadRel, " + err.message());
    return false;
  }

  // Validate filter in ReadRel.
  if (readRel.has_filter()) {
    std::vector<std::shared_ptr<const core::ITypedExpr>> expressions;
    expressions.reserve(1);

    std::vector<TypePtr> veloxTypeList;
    if (readRel.has_base_schema()) {
      const auto& baseSchema = readRel.base_schema();
      auto substraitTypeList = subParser_->parseNamedStruct(baseSchema);
      veloxTypeList.reserve(substraitTypeList.size());
      for (const auto& substraitType : substraitTypeList) {
        veloxTypeList.emplace_back(toVeloxType(substraitType->type));
      }
    }
    std::vector<std::string> names;
    int32_t inputPlanNodeId = 0;
    names.reserve(veloxTypeList.size());
    for (auto colIdx = 0; colIdx < veloxTypeList.size(); colIdx++) {
      names.emplace_back(subParser_->makeNodeName(inputPlanNodeId, colIdx));
    }
    auto rowType =
        std::make_shared<RowType>(std::move(names), std::move(veloxTypeList));

    try {
      expressions.emplace_back(
          exprConverter_->toVeloxExpr(readRel.filter(), rowType));
      // Try to compile the expressions. If there is any unregistered function
      // or mismatched type, exception will be thrown.
      exec::ExprSet exprSet(std::move(expressions), execCtx_);
    } catch (const VeloxException& err) {
      logValidateMsg(
          "native validation failed due to: filter expression validation fails in ReadRel, " +
          err.message());
      return false;
    }
  }
  if (readRel.has_base_schema()) {
    const auto& baseSchema = readRel.base_schema();
    if (!validateColNames(baseSchema)) {
      logValidateMsg(
          "native validation failed due to: Validation failed for column name contains illegal charactor.");
      return false;
    }
  }
  return true;
}

bool SubstraitToVeloxPlanValidator::validate(const ::substrait::Rel& rel) {
  if (rel.has_aggregate()) {
    return validate(rel.aggregate());
  }
  if (rel.has_project()) {
    return validate(rel.project());
  }
  if (rel.has_filter()) {
    return validate(rel.filter());
  }
  if (rel.has_join()) {
    return validate(rel.join());
  }
  if (rel.has_read()) {
    return validate(rel.read());
  }
  if (rel.has_sort()) {
    return validate(rel.sort());
  }
  if (rel.has_expand()) {
    return validate(rel.expand());
  }
  if (rel.has_fetch()) {
    return validate(rel.fetch());
  }
  if (rel.has_window()) {
    return validate(rel.window());
  }
  return false;
}

bool SubstraitToVeloxPlanValidator::validate(
    const ::substrait::RelRoot& relRoot) {
  if (relRoot.has_input()) {
    const auto& rel = relRoot.input();
    return validate(rel);
  }
  return false;
}

bool SubstraitToVeloxPlanValidator::validate(const ::substrait::Plan& plan) {
  // Create plan converter and expression converter to help the validation.
  planConverter_->constructFunctionMap(plan);
  exprConverter_ = std::make_shared<SubstraitVeloxExprConverter>(
      pool_, planConverter_->getFunctionMap());

  for (const auto& rel : plan.relations()) {
    if (rel.has_root()) {
      return validate(rel.root());
    }
    if (rel.has_rel()) {
      return validate(rel.rel());
    }
  }
  return false;
}

} // namespace facebook::velox::substrait
