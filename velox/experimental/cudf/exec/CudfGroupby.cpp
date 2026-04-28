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
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
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
using cudf_velox::CountInputKind;
using cudf_velox::get_output_mr;
using cudf_velox::get_temp_mr;
using cudf_velox::GroupbyAggregator;
using cudf_velox::ResolvedAggregateInfo;

#define DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Name, name, KIND)                    \
  struct Groupby##Name##Aggregator : GroupbyAggregator {                      \
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
      CountInputKind inputKind,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, nullptr, resultType),
        inputKind_(inputKind) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) override {
    auto& request = requests.emplace_back();
    outputIndex_ = requests.size() - 1;
    // kCountAll and kNullConstant both submit a count-all-rows request;
    // kNullConstant overrides the result with zeros in makeOutputColumn.
    const bool countAll = (inputKind_ != CountInputKind::kColumn);
    // For raw input, count(*) can use any column (column 0) since we just
    // need a row count. For non-raw input (intermediate/final in streaming),
    // the input is partial results where column 0 is the grouping key;
    // we must use inputIndex to access the partial count column.
    request.values =
        tbl.column((countAll && exec::isRawInput(step)) ? 0 : inputIndex);
    std::unique_ptr<cudf::groupby_aggregation> aggRequest =
        exec::isRawInput(step)
        ? cudf::make_count_aggregation<cudf::groupby_aggregation>(
              countAll ? cudf::null_policy::INCLUDE
                       : cudf::null_policy::EXCLUDE)
        : cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    request.aggregations.push_back(std::move(aggRequest));
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    auto col = std::move(results[outputIndex_].results[0]);
    if (inputKind_ == CountInputKind::kNullConstant) {
      auto zero = cudf::numeric_scalar<int64_t>(0, true, stream, get_temp_mr());
      col = cudf::make_column_from_scalar(
          zero, col->size(), stream, get_output_mr());
    }
    // cudf produces int32 for count but velox expects int64.
    const auto cudfOutputType =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(resultType));
    if (col->type() != cudfOutputType) {
      col = cudf::cast(*col, cudfOutputType, stream, get_output_mr());
    }
    return col;
  }

 private:
  CountInputKind inputKind_;
  uint32_t outputIndex_;
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
    const ResolvedAggregateInfo& p) {
  auto const& kind = p.kind;
  auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (kind.rfind(prefix + "sum", 0) == 0) {
    return std::make_unique<GroupbySumAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "count", 0) == 0) {
    VELOX_CHECK(p.countInputKind.has_value());
    return std::make_unique<GroupbyCountAggregator>(
        p.companionStep, p.inputIndex, *p.countInputKind, p.resultType);
  } else if (kind.rfind(prefix + "min", 0) == 0) {
    return std::make_unique<GroupbyMinAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "max", 0) == 0) {
    return std::make_unique<GroupbyMaxAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "avg", 0) == 0) {
    return std::make_unique<GroupbyMeanAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else {
    VELOX_NYI("Aggregation not yet supported, kind: {}", kind);
  }
}

} // namespace

namespace facebook::velox::cudf_velox {

std::vector<std::unique_ptr<GroupbyAggregator>> toGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants) {
  auto params =
      resolveAggregateInfos(aggregationNode, step, outputType, constants);

  std::vector<std::unique_ptr<GroupbyAggregator>> aggregators;
  aggregators.reserve(params.size());
  for (const auto& p : params) {
    aggregators.push_back(createGroupbyAggregator(p));
  }
  return aggregators;
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

    if (isCountFunctionName(aggregate.call->name())) {
      continue;
    }

    // Check input expressions can be evaluated by cuDF, expand the input first.
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
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          aggregationNode->outputType(),
          aggregationNode->id(),
          std::string{"CudfGroupby"} +
              std::string{
                  core::AggregationNode::toName(aggregationNode->step())},
          nvtx3::rgb{34, 139, 34}, // Forest Green
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          aggregationNode),
      aggregationNode_(aggregationNode),
      isPartialOutput_(
          exec::isPartialOutput(aggregationNode->step()) &&
          !hasFinalAggs(aggregationNode->aggregates())),
      isSingleStep_(
          aggregationNode->step() == core::AggregationNode::Step::kSingle),
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
  const auto inputRowSchema = asRowType(inputType_);
  auto aggregationInput = buildAggregationInputChannels(
      *aggregationNode_,
      *operatorCtx_,
      inputRowSchema,
      groupingKeyInputChannels_);
  aggregationInputChannels_ = std::move(aggregationInput.channels);
  aggregators_ = toGroupbyAggregators(
      *aggregationNode_,
      aggregationNode_->step(),
      outputType_,
      aggregationInput.constants);
  streamingEnabled_ = !hasCompanionAggregates(aggregationNode_->aggregates());

  // Make aggregators for intermediate step when streaming is enabled.
  if (streamingEnabled_) {
    const bool isFinalOrSingle =
        aggregationNode_->step() == core::AggregationNode::Step::kFinal ||
        aggregationNode_->step() == core::AggregationNode::Step::kSingle;
    bufferedResultType_ = isFinalOrSingle
        ? getBufferedResultType(*aggregationNode_)
        : outputType_;

    std::vector<VectorPtr> nullConstants(numAggregates_);
    intermediateAggregators_ = toGroupbyAggregators(
        *aggregationNode_,
        core::AggregationNode::Step::kIntermediate,
        bufferedResultType_,
        nullConstants);

    if (isSingleStep_) {
      partialAggregators_ = toGroupbyAggregators(
          *aggregationNode_,
          core::AggregationNode::Step::kPartial,
          bufferedResultType_,
          aggregationInput.constants);
      finalAggregators_ = toGroupbyAggregators(
          *aggregationNode_,
          core::AggregationNode::Step::kFinal,
          outputType_,
          nullConstants);
    }
  }

  // Check that aggregate result type match the output type.
  // TODO: This is output schema validation. In velox CPU, it's done using
  // output types reported by aggregation functions. We can't do that in cudf
  // groupby.

  // TODO: Set identity projections used by HashProbe to pushdown dynamic
  // filters to table scan.

  // TODO: Add support for grouping sets and group ids.

  aggregationNode_.reset();
}

void CudfGroupby::computePartialGroupbyStreaming(CudfVectorPtr tbl) {
  // For every input, we'll do a groupby and compact results with the existing
  // intermediate groupby results.

  auto inputTableStream = tbl->stream();
  // Use getTableView() to avoid expensive materialization for packed_table.
  // tbl stays alive during this function call, keeping the view valid.
  auto permutedInputView = tbl->getTableView().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  auto groupbyOnInput = doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      aggregators_,
      bufferedResultType_,
      inputTableStream);

  // If we already have partial output, concatenate the new results with it.
  if (bufferedResult_) {
    auto partialOutputStream = bufferedResult_->stream();
    std::vector<CudfVectorPtr> tablesToConcat;
    tablesToConcat.push_back(bufferedResult_);
    tablesToConcat.push_back(groupbyOnInput);
    auto concatenatedTable = getConcatenatedTable(
        std::move(tablesToConcat),
        bufferedResultType_,
        partialOutputStream,
        get_output_mr());

    // Now we have to groupby again but this time with intermediate aggregators.
    // Keep concatenatedTable alive while we use its view.
    auto compactedOutput = doGroupByAggregation(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        partialOutputStream);
    bufferedResult_ = compactedOutput;
  } else {
    // First time processing, just store the result of the input batch's groupby
    // This means we're storing the stream from the first batch.
    bufferedResult_ = groupbyOnInput;
  }
}

void CudfGroupby::computeFinalGroupbyStreaming(CudfVectorPtr tbl) {
  auto inputTableStream = tbl->stream();
  auto permutedInputView = tbl->getTableView().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());

  if (!bufferedResult_) {
    auto groupbyOnInput = doGroupByAggregation(
        permutedInputView,
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        inputTableStream);
    if (!groupbyOnInput) {
      return;
    }
    bufferedResult_ = groupbyOnInput;
    return;
  }

  std::vector<cudf::table_view> tablesToConcat;
  tablesToConcat.push_back(bufferedResult_->getTableView());
  tablesToConcat.push_back(permutedInputView);

  auto finalStream = bufferedResult_->stream();
  cudf::detail::join_streams(
      std::vector<rmm::cuda_stream_view>{inputTableStream}, finalStream);

  auto concatenatedTable =
      cudf::concatenate(tablesToConcat, finalStream, get_temp_mr());
  cudf::detail::join_streams(
      std::vector<rmm::cuda_stream_view>{finalStream}, inputTableStream);
  auto compactedOutput = doGroupByAggregation(
      concatenatedTable->view(),
      groupingKeyOutputChannels_,
      intermediateAggregators_,
      bufferedResultType_,
      finalStream);
  bufferedResult_ = compactedOutput;
}

void CudfGroupby::computeSingleGroupbyStreaming(CudfVectorPtr tbl) {
  auto inputTableStream = tbl->stream();
  auto permutedInputView = tbl->getTableView().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  auto groupbyOnInput = doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      partialAggregators_,
      bufferedResultType_,
      inputTableStream);

  if (bufferedResult_) {
    auto partialOutputStream = bufferedResult_->stream();
    std::vector<CudfVectorPtr> tablesToConcat;
    tablesToConcat.push_back(bufferedResult_);
    tablesToConcat.push_back(groupbyOnInput);
    auto concatenatedTable = getConcatenatedTable(
        std::move(tablesToConcat),
        bufferedResultType_,
        partialOutputStream,
        get_temp_mr());

    auto compactedOutput = doGroupByAggregation(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        partialOutputStream);
    bufferedResult_ = compactedOutput;
  } else {
    bufferedResult_ = groupbyOnInput;
  }
}

void CudfGroupby::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  numInputRows_ += input->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  if (streamingEnabled_) {
    if (isPartialOutput_) {
      computePartialGroupbyStreaming(cudfInput);
      return;
    } else if (isSingleStep_) {
      computeSingleGroupbyStreaming(cudfInput);
      return;
    } else {
      computeFinalGroupbyStreaming(cudfInput);
      return;
    }
  }

  // Handle non-streaming cases.
  inputs_.push_back(std::move(cudfInput));
}

CudfVectorPtr CudfGroupby::doGroupByAggregation(
    cudf::table_view tableView,
    std::vector<column_index_t> const& groupByKeys,
    std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
    TypePtr const& outputType,
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
      pool(), outputType, numRows, std::move(resultTable), stream);
}

CudfVectorPtr CudfGroupby::releaseAndResetBufferedResult() {
  auto numOutputRows = bufferedResult_->size();
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
  // We're moving bufferedResult_ to the caller because we want it to be null
  // after this call.
  return std::move(bufferedResult_);
}

RowVectorPtr CudfGroupby::doGetOutput() {
  // Handle partial streaming groupby.
  if (isPartialOutput_ && streamingEnabled_) {
    if (bufferedResult_ &&
        bufferedResult_->estimateFlatSize() >
            maxPartialAggregationMemoryUsage_) {
      return releaseAndResetBufferedResult();
    }
    if (not noMoreInput_) {
      // Don't produce output if the partial output hasn't reached memory limit
      // and there's more batches to come.
      return nullptr;
    }
    if (!bufferedResult_ && finished_) {
      return nullptr;
    }
    return releaseAndResetBufferedResult();
  }

  if (finished_) {
    return nullptr;
  }

  if (!isPartialOutput_ && !noMoreInput_) {
    // Final aggregation has to wait for all batches to arrive so we cannot
    // return any results here.
    return nullptr;
  }

  // Streaming finalization: single step uses finalAggregators_ to convert
  // intermediate results to final output; final step uses aggregators_.
  // At this point isPartialOutput_ is false (handled above) and noMoreInput_
  // is true (guarded by the check above).
  if (streamingEnabled_) {
    finished_ = true;
    if (!bufferedResult_) {
      return nullptr;
    }
    auto& aggs = isSingleStep_ ? finalAggregators_ : aggregators_;
    auto stream = bufferedResult_->stream();
    auto result = doGroupByAggregation(
        bufferedResult_->getTableView(),
        groupingKeyOutputChannels_,
        aggs,
        outputType_,
        stream);
    stream.synchronize();
    bufferedResult_.reset();
    return result;
  }

  if (inputs_.empty() && !noMoreInput_) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  auto tbl = getConcatenatedTable(
      std::exchange(inputs_, {}), inputType_, stream, get_output_mr());

  // Release input data after synchronizing.
  stream.synchronize();
  inputs_.clear();

  if (noMoreInput_) {
    finished_ = true;
  }

  VELOX_CHECK_NOT_NULL(tbl);

  auto permutedInputView = tbl->view().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  return doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      aggregators_,
      outputType_,
      stream);
}

void CudfGroupby::doNoMoreInput() {
  Operator::noMoreInput();
  if (isPartialOutput_ && inputs_.empty()) {
    finished_ = true;
  }
}

bool CudfGroupby::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
