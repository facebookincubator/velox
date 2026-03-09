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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/unary.hpp>

namespace {

using namespace facebook::velox;
using cudf_velox::get_output_mr;
using cudf_velox::get_temp_mr;
using cudf_velox::GroupbyAggregator;

#define DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Name, name, KIND)                    \
  struct Groupby##Name##Aggregator : GroupbyAggregator {                       \
    Groupby##Name##Aggregator(                                                \
        core::AggregationNode::Step step,                                     \
        uint32_t inputIndex,                                                  \
        VectorPtr constant,                                                   \
        const TypePtr& resultType)                                            \
        : GroupbyAggregator(step, inputIndex, constant, resultType) {}        \
                                                                              \
    void addGroupbyRequest(                                                   \
        cudf::table_view const& tbl,                                          \
        std::vector<cudf::groupby::aggregation_request>& requests) override { \
      VELOX_CHECK(                                                            \
          constant == nullptr,                                                \
          #Name "Aggregator does not yet support constant input");            \
      auto& request = requests.emplace_back();                                \
      output_idx = requests.size() - 1;                                       \
      request.values = tbl.column(inputIndex);                                \
      request.aggregations.push_back(                                         \
          cudf::make_##name##_aggregation<cudf::groupby_aggregation>());      \
    }                                                                         \
                                                                              \
    std::unique_ptr<cudf::column> makeOutputColumn(                           \
        std::vector<cudf::groupby::aggregation_result>& results,              \
        rmm::cuda_stream_view stream) override {                              \
      auto col = std::move(results[output_idx].results[0]);                   \
      const auto cudfType =                                                   \
          cudf::data_type(cudf_velox::veloxToCudfTypeId(resultType));         \
      if (col->type() != cudfType) {                                          \
        col = cudf::cast(*col, cudfType, stream, get_output_mr());            \
      }                                                                       \
      return col;                                                             \
    }                                                                         \
                                                                              \
   private:                                                                   \
    uint32_t output_idx;                                                      \
  };

DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Sum, sum, SUM)
DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Min, min, MIN)
DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Max, max, MAX)

struct GroupbyCountAggregator : GroupbyAggregator {
  GroupbyCountAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, constant, resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) override {
    auto& request = requests.emplace_back();
    outputIdx_ = requests.size() - 1;
    request.values = tbl.column(constant == nullptr ? inputIndex : 0);
    std::unique_ptr<cudf::groupby_aggregation> aggRequest =
        exec::isRawInput(step)
        ? cudf::make_count_aggregation<cudf::groupby_aggregation>(
              constant == nullptr ? cudf::null_policy::EXCLUDE
                                  : cudf::null_policy::INCLUDE)
        : cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    request.aggregations.push_back(std::move(aggRequest));
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    // cudf produces int32 for count(0) but velox expects int64
    auto col = std::move(results[outputIdx_].results[0]);
    const auto cudfOutputType =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(resultType));
    if (col->type() != cudfOutputType) {
      col = cudf::cast(*col, cudfOutputType, stream, get_output_mr());
    }
    return col;
  }

 private:
  uint32_t outputIdx_;
};

struct GroupbyMeanAggregator : GroupbyAggregator {
  GroupbyMeanAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, constant, resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        auto& request = requests.emplace_back();
        meanIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case core::AggregationNode::Step::kPartial: {
        auto& request = requests.emplace_back();
        sumIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        request.aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
        break;
      }
      case core::AggregationNode::Step::kIntermediate:
      case core::AggregationNode::Step::kFinal: {
        // In intermediate and final aggregation, the previously computed sum
        // and count are in the child columns of the input column.
        auto& request = requests.emplace_back();
        sumIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex).child(0);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());

        auto& request2 = requests.emplace_back();
        countIdx_ = requests.size() - 1;
        request2.values = tbl.column(inputIndex).child(1);
        // The counts are already computed in partial aggregation, so we just
        // need to sum them up again.
        request2.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        break;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    const auto& outputType = asRowType(resultType);
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return std::move(results[meanIdx_].results[0]);
      case core::AggregationNode::Step::kPartial: {
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[sumIdx_].results[1]);

        auto const size = sum->size();
        auto const cudfSumType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(0)));
        auto const cudfCountType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(1)));
        if (sum->type() != cudf::data_type(cudfSumType)) {
          sum = cudf::cast(
              *sum, cudf::data_type(cudfSumType), stream, get_output_mr());
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count = cudf::cast(
              *count, cudf::data_type(cudfCountType), stream, get_output_mr());
        }

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count));

        // TODO: Handle nulls. This can happen if all values are null in a
        // group.
        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kIntermediate: {
        // The difference between intermediate and partial is in where the
        // sum and count are coming from. In partial, since the input column is
        // the same, the sum and count are in the same agg result. In
        // intermediate, the input columns are different (it's the child
        // columns of the input column) and so the sum and count are in
        // different agg results.
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[countIdx_].results[0]);

        auto size = sum->size();
        auto const cudfSumType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(0)));
        auto const cudfCountType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(1)));
        if (sum->type() != cudf::data_type(cudfSumType)) {
          sum = cudf::cast(
              *sum, cudf::data_type(cudfSumType), stream, get_output_mr());
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count = cudf::cast(
              *count, cudf::data_type(cudfCountType), stream, get_output_mr());
        }

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count));

        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kFinal: {
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[countIdx_].results[0]);
        auto avg = cudf::binary_operation(
            *sum,
            *count,
            cudf::binary_operator::DIV,
            cudf::data_type(cudf_velox::veloxToCudfTypeId(resultType)),
            stream,
            get_output_mr());
        return avg;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

 private:
  // These indices are used to track where the desired result columns
  // (mean/<sum, count>) are in the output of cudf::groupby::aggregate().
  uint32_t meanIdx_;
  uint32_t sumIdx_;
  uint32_t countIdx_;
};

std::unique_ptr<GroupbyAggregator> createGroupbyAggregator(
    core::AggregationNode::Step step,
    std::string const& kind,
    uint32_t inputIndex,
    VectorPtr constant,
    const TypePtr& resultType) {
  auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (kind.rfind(prefix + "sum", 0) == 0) {
    return std::make_unique<GroupbySumAggregator>(
        step, inputIndex, constant, resultType);
  } else if (kind.rfind(prefix + "count", 0) == 0) {
    return std::make_unique<GroupbyCountAggregator>(
        step, inputIndex, constant, resultType);
  } else if (kind.rfind(prefix + "min", 0) == 0) {
    return std::make_unique<GroupbyMinAggregator>(
        step, inputIndex, constant, resultType);
  } else if (kind.rfind(prefix + "max", 0) == 0) {
    return std::make_unique<GroupbyMaxAggregator>(
        step, inputIndex, constant, resultType);
  } else if (kind.rfind(prefix + "avg", 0) == 0) {
    return std::make_unique<GroupbyMeanAggregator>(
        step, inputIndex, constant, resultType);
  } else {
    VELOX_NYI("Aggregation not yet supported, kind: {}", kind);
  }
}

} // namespace

namespace facebook::velox::cudf_velox {

std::vector<std::unique_ptr<GroupbyAggregator>> toGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx) {
  auto resolved = resolveAggregateInputs(aggregationNode, operatorCtx);

  std::vector<std::unique_ptr<GroupbyAggregator>> aggregators;
  for (auto& info : resolved) {
    aggregators.push_back(createGroupbyAggregator(
        info.companionStep, info.kind, info.inputIndex, info.constant,
        info.resultType));
  }
  return aggregators;
}

std::vector<std::unique_ptr<GroupbyAggregator>> toIntermediateGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx) {
  auto const step = core::AggregationNode::Step::kIntermediate;
  auto const& inputRowSchema = aggregationNode.outputType();

  std::vector<std::unique_ptr<GroupbyAggregator>> aggregators;
  for (size_t i = 0; i < aggregationNode.aggregates().size(); i++) {
    // Intermediate aggregation has a 1:1 mapping between input and output.
    // We don't need to figure out input from the aggregate function.
    auto const& aggregate = aggregationNode.aggregates()[i];
    auto const inputIndex = aggregationNode.groupingKeys().size() + i;
    auto const kind = aggregate.call->name();
    auto const constant = nullptr;
    const auto originalName = getOriginalName(kind);
    auto const companionStep = getCompanionStep(kind, step);
    if (exec::isPartialOutput(companionStep)) {
      const auto resultType =
          exec::resolveIntermediateType(originalName, aggregate.rawInputTypes);
      aggregators.push_back(createGroupbyAggregator(
          step, kind, inputIndex, constant, resultType));
    } else {
      // Final step aggregator will not use the intermediate aggregator.
      aggregators.push_back(nullptr);
    }
  }
  return aggregators;
}

StepAwareAggregationRegistry& getGroupbyAggregationRegistry() {
  static StepAwareAggregationRegistry registry;
  return registry;
}

bool registerGroupbyAggregationFunctions(const std::string& prefix) {
  auto& registry = getGroupbyAggregationRegistry();
  registerCommonAggregationFunctions(registry, prefix);
  return true;
}

bool canGroupbyAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx) {
  return canAggregationBeEvaluatedByRegistry(
      getGroupbyAggregationRegistry(), call, step, rawInputTypes, queryCtx);
}

bool canGroupbyBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx) {
  const core::PlanNode* sourceNode = aggregationNode.sources().empty()
      ? nullptr
      : aggregationNode.sources()[0].get();

  // Get the aggregation step from the node
  auto step = aggregationNode.step();

  // Check supported aggregation functions using step-aware aggregation registry
  for (const auto& aggregate : aggregationNode.aggregates()) {
    // Use step-aware validation that handles partial/final/intermediate steps
    if (!canGroupbyAggregationBeEvaluatedByCudf(
            *aggregate.call, step, aggregate.rawInputTypes, queryCtx)) {
      return false;
    }

    // `distinct` aggregations are not supported, in testing fails with "De-dup
    // before aggregation is not yet supported"
    if (aggregate.distinct) {
      return false;
    }

    // `mask` is NOT supported (in testing do not appear to be be applied and
    // return incorrect results )
    if (aggregate.mask) {
      return false;
    }

    // Check input expressions can be evaluated by CUDF, expand the input first
    for (const auto& input : aggregate.call->inputs()) {
      auto expandedInput = expandFieldReference(input, sourceNode);
      std::vector<core::TypedExprPtr> exprs = {expandedInput};
      if (!canBeEvaluatedByCudf(exprs, queryCtx)) {
        return false;
      }
    }
  }

  // Check grouping key expressions
  if (!canGroupingKeysBeEvaluatedByCudf(
          aggregationNode.groupingKeys(), sourceNode, queryCtx)) {
    return false;
  }

  return true;
}

CudfGroupby::CudfGroupby(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<core::AggregationNode const> const& aggregationNode)
    : Operator(
          driverCtx,
          aggregationNode->outputType(),
          operatorId,
          aggregationNode->id(),
          aggregationNode->step() == core::AggregationNode::Step::kPartial
              ? "CudfPartialGroupby"
              : "CudfGroupby",
          std::nullopt),
      NvtxHelper(
          nvtx3::rgb{34, 139, 34}, // Forest Green
          operatorId,
          fmt::format("[{}]", aggregationNode->id())),
      aggregationNode_(aggregationNode),
      isPartialOutput_(
          exec::isPartialOutput(aggregationNode->step()) &&
          !hasFinalAggs(aggregationNode->aggregates())),
      maxPartialAggregationMemoryUsage_(
          driverCtx->queryConfig().maxPartialAggregationMemoryUsage()) {}

void CudfGroupby::initialize() {
  Operator::initialize();

  inputType_ = aggregationNode_->sources()[0]->outputType();
  ignoreNullKeys_ = aggregationNode_->ignoreNullKeys();
  setupGroupingKeyChannelProjections(
      *aggregationNode_, groupingKeyInputChannels_, groupingKeyOutputChannels_);

  // Velox CPU does optimizations related to pre-grouped keys. This can be
  // done in cudf by passing sort information to cudf::groupby() constructor.
  // We're postponing this for now.

  numAggregates_ = aggregationNode_->aggregates().size();
  aggregators_ = toGroupbyAggregators(*aggregationNode_, *operatorCtx_);
  intermediateAggregators_ =
      toIntermediateGroupbyAggregators(*aggregationNode_, *operatorCtx_);

  // Check that aggregate result type match the output type.
  // TODO: This is output schema validation. In velox CPU, it's done using
  // output types reported by aggregation functions. We can't do that in cudf
  // groupby.

  // TODO: Set identity projections used by HashProbe to pushdown dynamic
  // filters to table scan.

  // TODO: Add support for grouping sets and group ids.

  aggregationNode_.reset();
}

void CudfGroupby::computeIntermediateGroupbyPartial(CudfVectorPtr tbl) {
  // For every input, we'll do a groupby and compact results with the existing
  // intermediate groupby results.

  auto inputTableStream = tbl->stream();
  // Use getTableView() to avoid expensive materialization for packed_table.
  // tbl stays alive during this function call, keeping the view valid.
  auto groupbyOnInput = doGroupByAggregation(
      tbl->getTableView(),
      groupingKeyInputChannels_,
      aggregators_,
      inputTableStream);

  // If we already have partial output, concatenate the new results with it.
  if (partialOutput_) {
    // Create a vector of tables to concatenate
    std::vector<cudf::table_view> tablesToConcat;
    tablesToConcat.push_back(partialOutput_->getTableView());
    tablesToConcat.push_back(groupbyOnInput->getTableView());

    auto partialOutputStream = partialOutput_->stream();
    // We need to join the input table stream on the partial output stream to
    // make sure the intermediate results are available when we do the concat.
    cudf::detail::join_streams(
        std::vector<rmm::cuda_stream_view>{inputTableStream},
        partialOutputStream);

    // Concatenate the tables
    auto concatenatedTable =
        cudf::concatenate(tablesToConcat, partialOutputStream, get_output_mr());

    // Now we have to groupby again but this time with intermediate aggregators.
    // Keep concatenatedTable alive while we use its view.
    auto compactedOutput = doGroupByAggregation(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        partialOutputStream);
    partialOutput_ = compactedOutput;
  } else {
    // First time processing, just store the result of the input batch's groupby
    // This means we're storing the stream from the first batch.
    partialOutput_ = groupbyOnInput;
  }
}

void CudfGroupby::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (input->size() == 0) {
    return;
  }
  numInputRows_ += input->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  if (isPartialOutput_) {
    computeIntermediateGroupbyPartial(cudfInput);
    return;
  }

  inputs_.push_back(std::move(cudfInput));
}

CudfVectorPtr CudfGroupby::doGroupByAggregation(
    cudf::table_view tableView,
    std::vector<column_index_t> const& groupByKeys,
    std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
    rmm::cuda_stream_view stream) {
  auto groupbyKeyView =
      tableView.select(groupByKeys.begin(), groupByKeys.end());

  // TODO: All other args to groupby are related to sort groupby. We don't
  // support optimizations related to it yet.
  cudf::groupby::groupby groupByOwner(
      groupbyKeyView,
      ignoreNullKeys_ ? cudf::null_policy::EXCLUDE
                      : cudf::null_policy::INCLUDE);

  std::vector<cudf::groupby::aggregation_request> requests;
  for (auto& aggregator : aggregators) {
    aggregator->addGroupbyRequest(tableView, requests);
  }

  auto [groupKeys, results] =
      groupByOwner.aggregate(requests, stream, get_output_mr());
  // flatten the results
  std::vector<std::unique_ptr<cudf::column>> resultColumns;

  // first fill the grouping keys
  auto groupKeysColumns = groupKeys->release();
  resultColumns.insert(
      resultColumns.begin(),
      std::make_move_iterator(groupKeysColumns.begin()),
      std::make_move_iterator(groupKeysColumns.end()));

  // then fill the aggregation results
  for (auto& aggregator : aggregators) {
    resultColumns.push_back(aggregator->makeOutputColumn(results, stream));
  }

  // make a cudf table out of columns
  auto resultTable = std::make_unique<cudf::table>(std::move(resultColumns));

  auto numRows = resultTable->num_rows();

  // velox expects nullptr instead of a table with 0 rows
  if (numRows == 0) {
    return nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, numRows, std::move(resultTable), stream);
}

CudfVectorPtr CudfGroupby::releaseAndResetPartialOutput() {
  auto numOutputRows = partialOutput_->size();
  const double aggregationPct =
      numOutputRows == 0 ? 0 : (numOutputRows * 1.0) / numInputRows_ * 100;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kFlushRowCount),
        RuntimeCounter(numOutputRows));
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kFlushTimes), RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kPartialAggregationPct),
        RuntimeCounter(aggregationPct));
  }

  numInputRows_ = 0;
  // We're moving partialOutput_ to the caller because we want it to be null
  // after this call.
  return std::move(partialOutput_);
}

RowVectorPtr CudfGroupby::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (isPartialOutput_) {
    if (partialOutput_ &&
        partialOutput_->estimateFlatSize() >
            maxPartialAggregationMemoryUsage_) {
      return releaseAndResetPartialOutput();
    }
    if (not noMoreInput_) {
      // Don't produce output if the partial output hasn't reached memory limit
      // and there's more batches to come.
      return nullptr;
    }
    if (!partialOutput_ && finished_) {
      return nullptr;
    }
    return releaseAndResetPartialOutput();
  }

  if (finished_) {
    return nullptr;
  }

  if (!noMoreInput_) {
    // Final aggregation has to wait for all batches to arrive so we cannot
    // return any results here.
    return nullptr;
  }

  if (inputs_.empty() && !noMoreInput_) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  auto tbl = getConcatenatedTable(inputs_, inputType_, stream, get_output_mr());

  // Release input data after synchronizing.
  stream.synchronize();
  inputs_.clear();

  if (noMoreInput_) {
    finished_ = true;
  }

  VELOX_CHECK_NOT_NULL(tbl);

  return doGroupByAggregation(
      tbl->view(), groupingKeyInputChannels_, aggregators_, stream);
}

void CudfGroupby::noMoreInput() {
  Operator::noMoreInput();
  if (isPartialOutput_ && inputs_.empty()) {
    finished_ = true;
  }
}

bool CudfGroupby::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
