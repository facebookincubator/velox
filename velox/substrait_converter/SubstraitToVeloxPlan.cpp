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
#include "velox/buffer/Buffer.h"
#include "velox/vector/arrow/Bridge.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::connector;
using namespace facebook::velox::dwio::common;

namespace facebook::velox::substraitconverter {

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const substrait::AggregateRel& sagg) {
  std::shared_ptr<const core::PlanNode> child_node;
  if (sagg.has_input()) {
    child_node = toVeloxPlan(sagg.input());
  } else {
    throw std::runtime_error("Child expected");
  }
  auto input_types = child_node->outputType();
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      velox_grouping_exprs;
  auto& groupings = sagg.groupings();
  int input_plan_node_id = plan_node_id_ - 1;
  int out_idx = 0;
  for (auto& grouping : groupings) {
    auto grouping_exprs = grouping.grouping_expressions();
    for (auto& grouping_expr : grouping_exprs) {
      auto field_expr =
          expr_converter_->toVeloxExpr(grouping_expr, input_plan_node_id);
      // Velox's groupings are limited to be Field, and pre-projection for
      // grouping cols is not supported.
      auto typed_field_expr =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
              field_expr);
      velox_grouping_exprs.push_back(typed_field_expr);
      out_idx += 1;
    }
  }
  // Parse measures
  core::AggregationNode::Step agg_step;
  bool phase_inited = false;
  std::vector<std::shared_ptr<const core::CallTypedExpr>> agg_exprs;
  std::vector<std::shared_ptr<const core::ITypedExpr>> project_exprs;
  std::vector<std::string> project_out_names;
  for (auto& smea : sagg.measures()) {
    auto agg_function = smea.measure();
    if (!phase_inited) {
      switch (agg_function.phase()) {
        case substrait::AGGREGATION_PHASE_INITIAL_TO_INTERMEDIATE:
          agg_step = core::AggregationNode::Step::kPartial;
          break;
        case substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_INTERMEDIATE:
          agg_step = core::AggregationNode::Step::kIntermediate;
          break;
        case substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_RESULT:
          agg_step = core::AggregationNode::Step::kFinal;
          break;
        default:
          throw new std::runtime_error("Aggregate phase is not supported.");
          break;
      }
      phase_inited = true;
    }
    auto func_id = agg_function.function_reference();
    auto sub_func_name = sub_parser_->findFunction(functions_map_, func_id);
    auto func_name = sub_parser_->substrait_velox_function_map[sub_func_name];
    std::vector<std::shared_ptr<const core::ITypedExpr>> agg_params;
    auto args = agg_function.args();
    for (auto arg : args) {
      switch (arg.rex_type_case()) {
        case substrait::Expression::RexTypeCase::kSelection: {
          auto sel = arg.selection();
          auto field_expr =
              expr_converter_->toVeloxExpr(sel, input_plan_node_id);
          agg_params.push_back(field_expr);
          break;
        }
        case substrait::Expression::RexTypeCase::kScalarFunction: {
          // Pre-projection is needed before Aggregate.
          auto sfunc = arg.scalar_function();
          auto velox_expr =
              expr_converter_->toVeloxExpr(sfunc, input_plan_node_id);
          project_exprs.push_back(velox_expr);
          auto col_out_name = sub_parser_->makeNodeName(plan_node_id_, out_idx);
          project_out_names.push_back(col_out_name);
          auto sub_type = sub_parser_->parseType(sfunc.output_type());
          auto velox_type = expr_converter_->getVeloxType(sub_type->type);
          auto agg_input_param =
              std::make_shared<const core::FieldAccessTypedExpr>(
                  velox_type, col_out_name);
          agg_params.push_back(agg_input_param);
          break;
        }
        default:
          throw new std::runtime_error("Expression not supported");
          break;
      }
    }
    auto agg_out_type = agg_function.output_type();
    auto agg_velox_type = expr_converter_->getVeloxType(
        sub_parser_->parseType(agg_out_type)->type);
    auto agg_expr = std::make_shared<const core::CallTypedExpr>(
        agg_velox_type, std::move(agg_params), func_name);
    agg_exprs.push_back(agg_expr);
    out_idx += 1;
  }
  bool ignoreNullKeys = false;
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> aggregateMasks(
      out_idx);
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      pre_grouping_exprs;
  if (project_out_names.size() > 0) {
    auto project_node = std::make_shared<core::ProjectNode>(
        nextPlanNodeId(),
        std::move(project_out_names),
        std::move(project_exprs),
        child_node);
    std::vector<std::string> agg_out_names;
    for (int idx = 0; idx < out_idx; idx++) {
      agg_out_names.push_back(sub_parser_->makeNodeName(plan_node_id_, idx));
    }
    auto agg_node = std::make_shared<core::AggregationNode>(
        nextPlanNodeId(),
        agg_step,
        velox_grouping_exprs,
        pre_grouping_exprs,
        agg_out_names,
        agg_exprs,
        aggregateMasks,
        ignoreNullKeys,
        project_node);
    return agg_node;
  } else {
    std::vector<std::string> agg_out_names;
    for (int idx = 0; idx < out_idx; idx++) {
      agg_out_names.push_back(sub_parser_->makeNodeName(plan_node_id_, idx));
    }
    auto agg_node = std::make_shared<core::AggregationNode>(
        nextPlanNodeId(),
        agg_step,
        velox_grouping_exprs,
        pre_grouping_exprs,
        agg_out_names,
        agg_exprs,
        aggregateMasks,
        ignoreNullKeys,
        child_node);
    return agg_node;
  }
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const substrait::ProjectRel& sproject) {
  std::shared_ptr<const core::PlanNode> child_node;
  if (sproject.has_input()) {
    child_node = toVeloxPlan(sproject.input());
  } else {
    throw std::runtime_error("Child expected");
  }
  // Expressions
  std::vector<std::string> project_names;
  std::vector<std::shared_ptr<const core::ITypedExpr>> expressions;
  auto pre_plan_node_id = plan_node_id_ - 1;
  int col_idx = 0;
  for (auto& expr : sproject.expressions()) {
    auto velox_expr = expr_converter_->toVeloxExpr(expr, pre_plan_node_id);
    expressions.push_back(velox_expr);
    auto col_out_name = sub_parser_->makeNodeName(plan_node_id_, col_idx);
    project_names.push_back(col_out_name);
    col_idx += 1;
  }
  auto project_node = std::make_shared<core::ProjectNode>(
      nextPlanNodeId(),
      std::move(project_names),
      std::move(expressions),
      child_node);
  return project_node;
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const substrait::FilterRel& sfilter) {
  // FIXME: currently Filter is skipped.
  std::shared_ptr<const core::PlanNode> child_node;
  if (sfilter.has_input()) {
    child_node = toVeloxPlan(sfilter.input());
  } else {
    throw std::runtime_error("Child expected");
  }
  /*
  if (sfilter.has_condition()) {
    ParseExpression(sfilter.condition());
  }
  for (auto& stype : sfilter.input_types()) {
    ParseType(stype);
  }
  */
  return child_node;
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const substrait::ReadRel& sread,
    u_int32_t* index,
    std::vector<std::string>* paths,
    std::vector<u_int64_t>* starts,
    std::vector<u_int64_t>* lengths) {
  std::vector<std::string> col_name_list;
  std::vector<std::shared_ptr<SubstraitParser::SubstraitType>>
      substrait_type_list;
  if (sread.has_base_schema()) {
    auto& base_schema = sread.base_schema();
    for (auto& name : base_schema.names()) {
      col_name_list.push_back(name);
    }
    auto type_list = sub_parser_->parseNamedStruct(base_schema);
    for (auto type : type_list) {
      substrait_type_list.push_back(type);
    }
  }
  // Parse local files
  if (sread.has_local_files()) {
    auto& local_files = sread.local_files();
    auto& files_list = local_files.items();
    for (auto& file : files_list) {
      // Expect all partions share the same index.
      (*index) = file.partition_index();
      (*paths).push_back(file.uri_file());
      (*starts).push_back(file.start());
      (*lengths).push_back(file.length());
    }
  }
  std::vector<TypePtr> velox_type_list;
  for (auto sub_type : substrait_type_list) {
    velox_type_list.push_back(expr_converter_->getVeloxType(sub_type->type));
  }
  // Note: Velox require Filter pushdown must being enabled.
  bool filter_pushdown_enabled = true;
  std::shared_ptr<hive::HiveTableHandle> table_handle;
  if (!sread.has_filter()) {
    std::cout << "no filter" << std::endl;
    table_handle = std::make_shared<hive::HiveTableHandle>(
        filter_pushdown_enabled, hive::SubfieldFilters{}, nullptr);
  } else {
    auto& sfilter = sread.filter();
    hive::SubfieldFilters filters =
        expr_converter_->toVeloxFilter(col_name_list, velox_type_list, sfilter);
    table_handle = std::make_shared<hive::HiveTableHandle>(
        filter_pushdown_enabled, std::move(filters), nullptr);
  }
  std::vector<std::string> out_names;
  std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
      assignments;
  for (int idx = 0; idx < col_name_list.size(); idx++) {
    auto out_name = sub_parser_->makeNodeName(plan_node_id_, idx);
    assignments[out_name] = std::make_shared<hive::HiveColumnHandle>(
        col_name_list[idx],
        hive::HiveColumnHandle::ColumnType::kRegular,
        velox_type_list[idx]);
    out_names.push_back(out_name);
  }
  auto output_type = ROW(std::move(out_names), std::move(velox_type_list));
  auto table_scan_node = std::make_shared<core::TableScanNode>(
      nextPlanNodeId(), output_type, table_handle, assignments);
  return table_scan_node;
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const substrait::Rel& srel) {
  if (srel.has_aggregate()) {
    return toVeloxPlan(srel.aggregate());
  } else if (srel.has_project()) {
    return toVeloxPlan(srel.project());
  } else if (srel.has_filter()) {
    return toVeloxPlan(srel.filter());
  } else if (srel.has_read()) {
    return toVeloxPlan(
        srel.read(), &partition_index_, &paths_, &starts_, &lengths_);
  } else {
    throw new std::runtime_error("Rel is not supported.");
  }
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const substrait::RelRoot& sroot) {
  auto& snames = sroot.names();
  int name_idx = 0;
  for (auto& sname : snames) {
    // TODO: Use names as output columns' names
    name_idx += 1;
  }
  if (sroot.has_input()) {
    auto& srel = sroot.input();
    return toVeloxPlan(srel);
  } else {
    throw new std::runtime_error("Input is expected in RelRoot.");
  }
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxPlan(
    const substrait::Plan& splan) {
  for (auto& sextension : splan.extensions()) {
    if (!sextension.has_extension_function()) {
      continue;
    }
    auto& sfmap = sextension.extension_function();
    auto id = sfmap.function_anchor();
    auto name = sfmap.name();
    functions_map_[id] = name;
  }
  expr_converter_ = std::make_shared<SubstraitVeloxExprConverter>(
      sub_parser_, functions_map_);
  std::shared_ptr<const core::PlanNode> plan_node;
  // In fact, only one RelRoot is expected here.
  for (auto& srel : splan.relations()) {
    if (srel.has_root()) {
      plan_node = toVeloxPlan(srel.root());
    }
    if (srel.has_rel()) {
      plan_node = toVeloxPlan(srel.rel());
    }
  }
  return plan_node;
}

std::string SubstraitVeloxPlanConverter::nextPlanNodeId() {
  auto id = fmt::format("{}", plan_node_id_);
  plan_node_id_++;
  return id;
}

} // namespace facebook::velox::substraitconverter
