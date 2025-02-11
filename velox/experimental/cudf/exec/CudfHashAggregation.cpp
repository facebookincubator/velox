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

#include "CudfHashAggregation.h"

#include "cudf/column/column_factories.hpp"
#include "cudf/stream_compaction.hpp"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/Task.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/expression/Expr.h"

#include <cudf/concatenate.hpp>
#include <cudf/reduction.hpp>
#include <optional>

namespace {

using namespace facebook::velox;

auto toAggregationsMap(const core::AggregationNode& aggregationNode) {
  auto step = aggregationNode.step();
  std::map<uint32_t, std::vector<std::pair<cudf::aggregation::Kind, uint32_t>>>
      requests;
  const auto& inputRowSchema = aggregationNode.sources()[0]->outputType();

  uint32_t outputIndex = aggregationNode.groupingKeys().size();

  for (auto& aggregate : aggregationNode.aggregates()) {
    std::vector<column_index_t> agg_inputs;
    for (const auto& arg : aggregate.call->inputs()) {
      if (auto field =
              dynamic_cast<const core::FieldAccessTypedExpr*>(arg.get())) {
        agg_inputs.push_back(inputRowSchema->getChildIdx(field->name()));
      } else {
        VELOX_NYI("Constants and lambdas not yet supported");
      }
    }
    // DM: This above seems to suggest that there can be multiple inputs to an
    // aggregate. I don't really know which kinds of aggregations support this
    // so I'm going to ignore it for now.
    VELOX_CHECK(agg_inputs.size() == 1);

    if (aggregate.distinct) {
      VELOX_NYI("De-dup before aggregation is not yet supported");
    }

    auto& agg_name = aggregate.call->name();
    if (agg_name == "sum") {
      requests[agg_inputs[0]].push_back(
          std::make_pair(cudf::aggregation::SUM, outputIndex));
    } else if (agg_name == "min") {
      requests[agg_inputs[0]].push_back(
          std::make_pair(cudf::aggregation::MIN, outputIndex));
    } else if (agg_name == "max") {
      requests[agg_inputs[0]].push_back(
          std::make_pair(cudf::aggregation::MAX, outputIndex));
    } else if (agg_name == "count") {
      if (facebook::velox::exec::isPartialOutput(step)) {
        // TODO (dm): Count valid and count all are separate aggregations. Fix
        // this
        requests[agg_inputs[0]].push_back(
            std::make_pair(cudf::aggregation::COUNT_ALL, outputIndex));
      } else {
        requests[agg_inputs[0]].push_back(
            std::make_pair(cudf::aggregation::SUM, outputIndex));
      }
    }
    outputIndex++;
  }

  return requests;
}

std::unique_ptr<cudf::groupby_aggregation> toGroupbyAggregationRequest(
    cudf::aggregation::Kind kind) {
  switch (kind) {
    case cudf::aggregation::SUM:
      return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::COUNT_ALL:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MIN:
      return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MAX:
      return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    default:
      VELOX_NYI("Aggregation not yet supported");
  }
}

std::unique_ptr<cudf::reduce_aggregation> toGlobalAggregationRequest(
    cudf::aggregation::Kind kind) {
  switch (kind) {
    case cudf::aggregation::SUM:
      return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::MIN:
      return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::MAX:
      return cudf::make_max_aggregation<cudf::reduce_aggregation>();
    default:
      VELOX_NYI("Aggregation not yet supported");
  }
}

} // namespace

namespace facebook::velox::cudf_velox {

CudfHashAggregation::CudfHashAggregation(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::AggregationNode>& aggregationNode)
    : Operator(
          driverCtx,
          aggregationNode->outputType(),
          operatorId,
          aggregationNode->id(),
          aggregationNode->step() == core::AggregationNode::Step::kPartial
              ? "CudfPartialAggregation"
              : "CudfAggregation",
          aggregationNode->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(operatorId)
              : std::nullopt),
      aggregationNode_(aggregationNode),
      isPartialOutput_(exec::isPartialOutput(aggregationNode->step())),
      isGlobal_(aggregationNode->groupingKeys().empty()),
      isDistinct_(!isGlobal_ && aggregationNode->aggregates().empty()) {}

void CudfHashAggregation::initialize() {
  Operator::initialize();

  VELOX_CHECK(pool()->trackUsage());

  const auto& inputType = aggregationNode_->sources()[0]->outputType();
  ignoreNullKeys_ = aggregationNode_->ignoreNullKeys();
  setupGroupingKeyChannelProjections(
      groupingKeyInputChannels_, groupingKeyOutputChannels_);

  const auto numGroupingKeys = groupingKeyOutputChannels_.size();

  // DM: Velox CPU does optimizations related to pre-grouped keys. We can also
  // do that in cudf. I'm skipping it for now

  requests_map_ = toAggregationsMap(*aggregationNode_);
  numAggregates_ = aggregationNode_->aggregates().size();

  // Check that aggregate result type match the output type.
  // TODO (dm): This is output schema validation. In velox CPU, it's done using
  // output types reported by aggregation functions. We can't do that in cudf
  // groupby.

  // DM: Set identity projections used by HashProbe to pushdown dynamic filters
  // to table scan.

  // TODO (dm): Add support for grouping sets and group ids

  aggregationNode_.reset();
}

void CudfHashAggregation::setupGroupingKeyChannelProjections(
    std::vector<column_index_t>& groupingKeyInputChannels,
    std::vector<column_index_t>& groupingKeyOutputChannels) const {
  VELOX_CHECK(groupingKeyInputChannels.empty());
  VELOX_CHECK(groupingKeyOutputChannels.empty());

  const auto& inputType = aggregationNode_->sources()[0]->outputType();
  const auto& groupingKeys = aggregationNode_->groupingKeys();
  // The map from the grouping key output channel to the input channel.
  //
  // NOTE: grouping key output order is specified as 'groupingKeys' in
  // 'aggregationNode_'.
  std::vector<exec::IdentityProjection> groupingKeyProjections;
  groupingKeyProjections.reserve(groupingKeys.size());
  for (auto i = 0; i < groupingKeys.size(); ++i) {
    groupingKeyProjections.emplace_back(
        exec::exprToChannel(groupingKeys[i].get(), inputType), i);
  }

  groupingKeyInputChannels.reserve(groupingKeys.size());
  for (auto i = 0; i < groupingKeys.size(); ++i) {
    groupingKeyInputChannels.push_back(groupingKeyProjections[i].inputChannel);
  }

  groupingKeyOutputChannels.resize(groupingKeys.size());

  std::iota(
      groupingKeyOutputChannels.begin(), groupingKeyOutputChannels.end(), 0);
}

void CudfHashAggregation::addInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudf_input = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudf_input);
    inputs_.push_back(std::move(cudf_input));
  }
}

RowVectorPtr CudfHashAggregation::doGroupByAggregation(
    std::unique_ptr<cudf::table> tbl) {
  auto groupby_key_tbl = tbl->select(
      groupingKeyInputChannels_.begin(), groupingKeyInputChannels_.end());

  size_t num_grouping_keys = groupby_key_tbl.num_columns();

  // TODO (dm): Support args like include_null_keys, keys_are_sorted,
  // column_order, null_precedence. We're fine for now because very few nullable
  // columns in tpch
  cudf::groupby::groupby group_by_owner(
      groupby_key_tbl,
      ignoreNullKeys_ ? cudf::null_policy::EXCLUDE
                      : cudf::null_policy::INCLUDE);

  // convert aggregation map into aggregation requests
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::vector<uint32_t>> output_indices;
  for (auto& [val_col_idx, agg_kinds] : requests_map_) {
    auto& request = requests.emplace_back();
    request.values = tbl->get_column(val_col_idx).view();
    auto& output_idx = output_indices.emplace_back();
    for (auto const& [aggKind, outIdx] : agg_kinds) {
      request.aggregations.push_back(toGroupbyAggregationRequest(aggKind));
      output_idx.push_back(outIdx);
    }
  }

  auto [group_keys, results] = group_by_owner.aggregate(requests);
  // flatten the results
  std::vector<std::unique_ptr<cudf::column>> result_columns;

  // first fill the grouping keys
  auto group_keys_columns = group_keys->release();
  result_columns.insert(
      result_columns.begin(),
      std::make_move_iterator(group_keys_columns.begin()),
      std::make_move_iterator(group_keys_columns.end()));

  // then fill the aggregation results
  result_columns.resize(num_grouping_keys + numAggregates_);
  for (auto i = 0; i < results.size(); i++) {
    auto& per_column_results = results[i].results;
    for (auto j = 0; j < per_column_results.size(); j++) {
      result_columns[output_indices[i][j]] = std::move(per_column_results[j]);
    }
  }

  // make a cudf table out of columns
  auto result_table = std::make_unique<cudf::table>(std::move(result_columns));

  // velox expects nullptr instead of a table with 0 rows
  if (result_table->num_rows() == 0) {
    return nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, result_table->num_rows(), std::move(result_table));
}

RowVectorPtr CudfHashAggregation::doGlobalAggregation(
    std::unique_ptr<cudf::table> tbl) {
  std::vector<std::unique_ptr<cudf::scalar>> result_scalars;
  result_scalars.resize(numAggregates_);

  for (auto const& [inColIdx, aggs] : requests_map_) {
    for (auto const& [aggKind, outIdx] : aggs) {
      auto inCol = tbl->get_column(inColIdx);
      auto result = cudf::reduce(
          inCol,
          *toGlobalAggregationRequest(aggKind),
          cudf::data_type(
              cudf_velox::velox_to_cudf_type_id(outputType_->childAt(outIdx))));
      result_scalars[outIdx] = std::move(result);
    }
  }

  // Convert scalars to columns
  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.reserve(result_scalars.size());
  for (auto& scalar : result_scalars) {
    result_columns.push_back(cudf::make_column_from_scalar(*scalar, 1));
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(),
      outputType_,
      1,
      std::make_unique<cudf::table>(std::move(result_columns)));

  VELOX_NYI("CudfHashAggregation::doGlobalAggregation()");
}

RowVectorPtr CudfHashAggregation::getDistinctKeys(
    std::unique_ptr<cudf::table> tbl) {
  std::vector<cudf::size_type> key_indices(
      groupingKeyInputChannels_.begin(), groupingKeyInputChannels_.end());
  auto result = cudf::distinct(tbl->view(), key_indices);

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, result->num_rows(), std::move(result));
}

RowVectorPtr CudfHashAggregation::getOutput() {
  if (finished_) {
    input_ = nullptr;
    return nullptr;
  }

  if (!noMoreInput_ && !newDistincts_) {
    input_ = nullptr;
    return nullptr;
  }

  if (inputs_.empty()) {
    return nullptr;
  }

  finished_ = true;

  auto cudf_tables = std::vector<std::unique_ptr<cudf::table>>(inputs_.size());
  auto cudf_table_views = std::vector<cudf::table_view>(inputs_.size());
  for (int i = 0; i < inputs_.size(); i++) {
    VELOX_CHECK_NOT_NULL(inputs_[i]);
    cudf_tables[i] = inputs_[i]->release();
    cudf_table_views[i] = cudf_tables[i]->view();
  }
  auto tbl = cudf::concatenate(cudf_table_views);

  cudf_table_views.clear();
  cudf_tables.clear();
  inputs_.clear();

  VELOX_CHECK_NOT_NULL(tbl);

  if (!isGlobal_) {
    return doGroupByAggregation(std::move(tbl));
  } else if (isDistinct_) {
    return getDistinctKeys(std::move(tbl));
  } else {
    return doGlobalAggregation(std::move(tbl));
  }
}

void CudfHashAggregation::noMoreInput() {
  Operator::noMoreInput();
}

bool CudfHashAggregation::isFinished() {
  return finished_;
}

void CudfHashAggregation::close() {
  Operator::close();
}

} // namespace facebook::velox::cudf_velox
