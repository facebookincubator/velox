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

#include "substrait_to_velox_expr.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::connector;
using namespace facebook::velox::dwio::common;

namespace facebook::velox::substrait {

SubstraitVeloxExprConverter::SubstraitVeloxExprConverter(
    const std::shared_ptr<SubstraitParser>& sub_parser,
    const std::unordered_map<uint64_t, std::string>& functions_map) {
  sub_parser_ = sub_parser;
  functions_map_ = functions_map;
}

std::shared_ptr<const core::FieldAccessTypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const substrait::Expression::FieldReference& sfield,
    const int32_t& input_plan_node_id) {
  switch (sfield.reference_type_case()) {
    case substrait::Expression::FieldReference::ReferenceTypeCase::kDirectReference: {
      auto dref = sfield.direct_reference();
      int32_t col_idx = parseReferenceSegment(dref);
      auto field_name = sub_parser_->makeNodeName(input_plan_node_id, col_idx);
      // FIXME: get the input type
      return std::make_shared<const core::FieldAccessTypedExpr>(DOUBLE(), field_name);
      break;
    }
    case substrait::Expression::FieldReference::ReferenceTypeCase::kMaskedReference: {
      throw new std::runtime_error("not supported");
      break;
    }
    default:
      throw new std::runtime_error("not supported");
      break;
  }
}

std::shared_ptr<const core::ITypedExpr> SubstraitVeloxExprConverter::toVeloxExpr(
    const substrait::Expression::ScalarFunction& sfunc,
    const int32_t& input_plan_node_id) {
  std::vector<std::shared_ptr<const core::ITypedExpr>> params;
  for (auto& sarg : sfunc.args()) {
    auto expr = toVeloxExpr(sarg, input_plan_node_id);
    params.push_back(expr);
  }
  auto function_id = sfunc.function_reference();
  auto function_name = sub_parser_->findFunction(functions_map_, function_id);
  auto velox_function = sub_parser_->substrait_velox_function_map[function_name];
  auto sub_type = sub_parser_->parseType(sfunc.output_type());
  auto velox_type = getVeloxType(sub_type->type);
  return std::make_shared<const core::CallTypedExpr>(velox_type, std::move(params),
                                                     velox_function);
}

std::shared_ptr<const core::ConstantTypedExpr> SubstraitVeloxExprConverter::toVeloxExpr(
    const substrait::Expression::Literal& slit) {
  switch (slit.literal_type_case()) {
    case substrait::Expression_Literal::LiteralTypeCase::kFp64: {
      double val = slit.fp64();
      return std::make_shared<core::ConstantTypedExpr>(val);
      break;
    }
    case substrait::Expression_Literal::LiteralTypeCase::kBoolean: {
      bool val = slit.boolean();
      throw new std::runtime_error("Type is not supported.");
      break;
    }
    default:
      throw new std::runtime_error("Type is not supported.");
      break;
  }
}

std::shared_ptr<const core::ITypedExpr> SubstraitVeloxExprConverter::toVeloxExpr(
    const substrait::Expression& sexpr, const int32_t& input_plan_node_id) {
  std::shared_ptr<const core::ITypedExpr> velox_expr;
  switch (sexpr.rex_type_case()) {
    case substrait::Expression::RexTypeCase::kLiteral: {
      auto slit = sexpr.literal();
      velox_expr = toVeloxExpr(slit);
      break;
    }
    case substrait::Expression::RexTypeCase::kScalarFunction: {
      auto sfunc = sexpr.scalar_function();
      velox_expr = toVeloxExpr(sfunc, input_plan_node_id);
      break;
    }
    case substrait::Expression::RexTypeCase::kSelection: {
      auto sel = sexpr.selection();
      velox_expr = toVeloxExpr(sel, input_plan_node_id);
      break;
    }
    default:
      throw new std::runtime_error("Expression not supported");
      break;
  }
  return velox_expr;
}

TypePtr SubstraitVeloxExprConverter::getVeloxType(std::string type_name) {
  if (type_name == "BOOL") {
    return BOOLEAN();
  } else if (type_name == "FP64") {
    return DOUBLE();
  } else if (type_name == "STRING") {
    return VARCHAR();
  } else {
    throw std::runtime_error("Type name is not supported");
  }
}

int32_t SubstraitVeloxExprConverter::parseReferenceSegment(
    const substrait::Expression::ReferenceSegment& sref) {
  switch (sref.reference_type_case()) {
    case substrait::Expression::ReferenceSegment::ReferenceTypeCase::kStructField: {
      auto sfield = sref.struct_field();
      auto field_id = sfield.field();
      return field_id;
      break;
    }
    default:
      throw new std::runtime_error("not supported");
      break;
  }
}

// This class is used by Filter PushDown.
class SubstraitVeloxExprConverter::FilterInfo {
 public:
  FilterInfo() {}
  void setLeft(double left, bool isExclusive) {
    left_ = left;
    left_exclusive_ = isExclusive;
    if (!is_initialized_) {
      is_initialized_ = true;
    }
  }
  void setRight(double right, bool isExclusive) {
    right_ = right;
    right_exclusive_ = isExclusive;
    if (!is_initialized_) {
      is_initialized_ = true;
    }
  }
  void forbidsNull() {
    null_allowed_ = false;
    if (!is_initialized_) {
      is_initialized_ = true;
    }
  }
  bool isInitialized() { return is_initialized_ ? true : false; }

  std::optional<double> left_ = std::nullopt;
  std::optional<double> right_ = std::nullopt;
  bool null_allowed_ = true;
  bool left_exclusive_ = false;
  bool right_exclusive_ = false;

 private:
  bool is_initialized_ = false;
};

void SubstraitVeloxExprConverter::getFlatConditions(
    const substrait::Expression& sfilter,
    std::vector<substrait::Expression_ScalarFunction>* scalar_functions) {
  switch (sfilter.rex_type_case()) {
    case substrait::Expression::RexTypeCase::kScalarFunction: {
      auto sfunc = sfilter.scalar_function();
      auto filter_name =
          sub_parser_->findFunction(functions_map_, sfunc.function_reference());
      if (filter_name == "AND") {
        for (auto& scondition : sfunc.args()) {
          getFlatConditions(scondition, scalar_functions);
        }
      } else {
        (*scalar_functions).push_back(sfunc);
      }
    }
  }
}

hive::SubfieldFilters SubstraitVeloxExprConverter::toVeloxFilter(
    const std::vector<std::string>& input_name_list,
    const std::vector<TypePtr>& input_type_list, const substrait::Expression& sfilter) {
  hive::SubfieldFilters filters;
  std::unordered_map<int, std::shared_ptr<FilterInfo>> col_info_map;
  for (int idx = 0; idx < input_name_list.size(); idx++) {
    auto filter_info = std::make_shared<FilterInfo>();
    col_info_map[idx] = filter_info;
  }
  std::vector<substrait::Expression_ScalarFunction> scalar_functions;
  getFlatConditions(sfilter, &scalar_functions);
  for (auto& scalar_function : scalar_functions) {
    auto filter_name =
        sub_parser_->findFunction(functions_map_, scalar_function.function_reference());
    int32_t col_idx;
    // FIXME: different type support
    double val;
    for (auto& param : scalar_function.args()) {
      switch (param.rex_type_case()) {
        case substrait::Expression::RexTypeCase::kSelection: {
          auto sel = param.selection();
          // FIXME: only direct reference is considered here.
          auto dref = sel.direct_reference();
          col_idx = parseReferenceSegment(dref);
          break;
        }
        case substrait::Expression::RexTypeCase::kLiteral: {
          auto slit = param.literal();
          // FIXME: only double is considered here.
          val = slit.fp64();
          break;
        }
        default:
          throw new std::runtime_error("Condition arg is not supported.");
          break;
      }
    }
    if (filter_name == "IS_NOT_NULL") {
      col_info_map[col_idx]->forbidsNull();
    } else if (filter_name == "GREATER_THAN_OR_EQUAL") {
      col_info_map[col_idx]->setLeft(val, false);
    } else if (filter_name == "GREATER_THAN") {
      col_info_map[col_idx]->setLeft(val, true);
    } else if (filter_name == "LESS_THAN_OR_EQUAL") {
      col_info_map[col_idx]->setRight(val, false);
    } else if (filter_name == "LESS_THAN") {
      col_info_map[col_idx]->setRight(val, true);
    } else {
      throw new std::runtime_error("Function name is not supported.");
    }
  }
  for (int idx = 0; idx < input_name_list.size(); idx++) {
    auto filter_info = col_info_map[idx];
    double left_bound = 0;
    double right_bound = 0;
    bool left_unbounded = true;
    bool right_unbounded = true;
    bool left_exclusive = false;
    bool right_exclusive = false;
    if (filter_info->isInitialized()) {
      if (filter_info->left_) {
        left_unbounded = false;
        left_bound = filter_info->left_.value();
        left_exclusive = filter_info->left_exclusive_;
      }
      if (filter_info->right_) {
        right_unbounded = false;
        right_bound = filter_info->right_.value();
        right_exclusive = filter_info->right_exclusive_;
      }
      bool null_allowed = filter_info->null_allowed_;
      filters[common::Subfield(input_name_list[idx])] =
          std::make_unique<common::DoubleRange>(
              left_bound, left_unbounded, left_exclusive, right_bound, right_unbounded,
              right_exclusive, null_allowed);
    }
  }
  return filters;
}

}  // namespace facebook::velox::substrait
