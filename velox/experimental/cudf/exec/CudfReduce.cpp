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
#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfReduce.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

namespace {

using namespace facebook::velox;
using facebook::velox::cudf_velox::CountInputKind;
using facebook::velox::cudf_velox::get_output_mr;
using facebook::velox::cudf_velox::get_temp_mr;
using facebook::velox::cudf_velox::ReduceAggregator;
using facebook::velox::cudf_velox::ResolvedAggregateInfo;

#define DEFINE_SIMPLE_REDUCE_AGGREGATOR(Name, name)                    \
  struct Reduce##Name##Aggregator : ReduceAggregator {                 \
    Reduce##Name##Aggregator(                                          \
        core::AggregationNode::Step step,                              \
        uint32_t inputIndex,                                           \
        VectorPtr constant,                                            \
        const TypePtr& resultType)                                     \
        : ReduceAggregator(step, inputIndex, constant, resultType) {}  \
                                                                       \
    std::unique_ptr<cudf::column> doReduce(                            \
        cudf::table_view const& input,                                 \
        TypePtr const& outputType,                                     \
        rmm::cuda_stream_view stream,                                  \
        vector_size_t /*inputRowCount*/) override {                    \
      auto const aggRequest =                                          \
          cudf::make_##name##_aggregation<cudf::reduce_aggregation>(); \
      auto const cudfOutputType =                                      \
          cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType));  \
      auto const resultScalar = cudf::reduce(                          \
          input.column(inputIndex),                                    \
          *aggRequest,                                                 \
          cudfOutputType,                                              \
          stream,                                                      \
          get_temp_mr());                                              \
      return cudf::make_column_from_scalar(                            \
          *resultScalar, 1, stream, get_output_mr());                  \
    }                                                                  \
  };

DEFINE_SIMPLE_REDUCE_AGGREGATOR(Sum, sum)
DEFINE_SIMPLE_REDUCE_AGGREGATOR(Min, min)
DEFINE_SIMPLE_REDUCE_AGGREGATOR(Max, max)

struct ReduceCountAggregator : ReduceAggregator {
  ReduceCountAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      CountInputKind inputKind,
      const TypePtr& resultType)
      : ReduceAggregator(step, inputIndex, nullptr, resultType),
        inputKind_(inputKind) {}

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream,
      vector_size_t inputRowCount) override {
    if (exec::isRawInput(step)) {
      int64_t count;
      switch (inputKind_) {
        case CountInputKind::kNullConstant:
          count = 0;
          break;
        case CountInputKind::kCountAll:
          count = input.num_columns() > 0 ? input.num_rows() : inputRowCount;
          break;
        case CountInputKind::kColumn: {
          VELOX_CHECK_GT(
              input.num_columns(),
              0,
              "count(column) requires at least one input column");
          auto inputCol = input.column(inputIndex);
          count = inputCol.size() - inputCol.null_count();
          break;
        }
        default:
          VELOX_UNREACHABLE();
      }

      auto resultScalar =
          cudf::numeric_scalar<int64_t>(count, true, stream, get_temp_mr());

      return cudf::make_column_from_scalar(
          resultScalar, 1, stream, get_output_mr());
    } else {
      // For non-raw input (intermediate/final), use sum aggregation
      auto const aggRequest =
          cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto const cudfOutputType = cudf::data_type(cudf::type_id::INT64);
      auto const resultScalar = cudf::reduce(
          input.column(inputIndex),
          *aggRequest,
          cudfOutputType,
          stream,
          get_temp_mr());
      resultScalar->set_valid_async(true, stream);
      return cudf::make_column_from_scalar(
          *resultScalar, 1, stream, get_output_mr());
    }
  }

 private:
  CountInputKind inputKind_;
};

struct ReduceMeanAggregator : ReduceAggregator {
  ReduceMeanAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : ReduceAggregator(step, inputIndex, constant, resultType) {}

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream,
      vector_size_t /*inputRowCount*/) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        auto const aggRequest =
            cudf::make_mean_aggregation<cudf::reduce_aggregation>();
        auto const cudfOutputType =
            cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType));
        auto const resultScalar = cudf::reduce(
            input.column(inputIndex),
            *aggRequest,
            cudfOutputType,
            stream,
            get_temp_mr());
        return cudf::make_column_from_scalar(
            *resultScalar, 1, stream, get_output_mr());
      }
      case core::AggregationNode::Step::kPartial: {
        VELOX_CHECK(outputType->isRow());
        auto const& rowType = outputType->asRow();
        auto const sumType = rowType.childAt(0);
        auto const countType = rowType.childAt(1);
        auto const cudfSumType =
            cudf::data_type(cudf_velox::veloxToCudfTypeId(sumType));
        auto const cudfCountType =
            cudf::data_type(cudf_velox::veloxToCudfTypeId(countType));

        // sum
        auto const aggRequest =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const sumResultScalar = cudf::reduce(
            input.column(inputIndex),
            *aggRequest,
            cudfSumType,
            stream,
            get_temp_mr());
        auto sumCol = cudf::make_column_from_scalar(
            *sumResultScalar, 1, stream, get_output_mr());

        // libcudf doesn't have a count agg for reduce. What we want is to
        // count the number of valid rows.
        auto countCol = cudf::make_column_from_scalar(
            cudf::numeric_scalar<int64_t>(
                input.column(inputIndex).size() -
                    input.column(inputIndex).null_count(),
                true,
                stream,
                get_temp_mr()),
            1,
            stream,
            get_output_mr());

        // Assemble into struct as expected by velox.
        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sumCol));
        children.push_back(std::move(countCol));
        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            1,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kFinal: {
        // Input column has two children: sum and count
        auto const sumCol = input.column(inputIndex).child(0);
        auto const countCol = input.column(inputIndex).child(1);

        // sum the sums
        auto const sumAggRequest =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const sumResultScalar = cudf::reduce(
            sumCol, *sumAggRequest, sumCol.type(), stream, get_temp_mr());
        auto sumResultCol = cudf::make_column_from_scalar(
            *sumResultScalar, 1, stream, get_output_mr());

        // sum the counts
        auto const countAggRequest =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const countResultScalar = cudf::reduce(
            countCol, *countAggRequest, countCol.type(), stream, get_temp_mr());

        // divide the sums by the counts
        auto const cudfOutputType =
            cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType));
        return cudf::binary_operation(
            *sumResultCol,
            *countResultScalar,
            cudf::binary_operator::DIV,
            cudfOutputType,
            stream,
            get_output_mr());
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }
};

struct ApproxDistinctAggregator : ReduceAggregator {
  static constexpr cudf::null_policy kNullPolicy = cudf::null_policy::EXCLUDE;
  static constexpr cudf::nan_policy kNanPolicy = cudf::nan_policy::NAN_IS_VALID;

  ApproxDistinctAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType,
      std::int32_t precision = 11) // Default 11 matches Velox's 2.3% standard
                                   // error (2^11 = 2048 buckets)
      : ReduceAggregator{step, inputIndex, constant, resultType},
        precision_{precision} {
    VELOX_CHECK(
        constant == nullptr,
        "ApproxDistinctAggregator does not support constant input");
  }

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream,
      vector_size_t /*inputRowCount*/) override {
    if (exec::isRawInput(step)) {
      return doPartialReduce(input, stream);
    } else if (step == core::AggregationNode::Step::kIntermediate) {
      return doIntermediateReduce(input, stream);
    } else {
      return doFinalReduce(input, stream);
    }
  }

 private:
  std::unique_ptr<cudf::column> makeSketchColumn(
      cuda::std::span<cuda::std::byte const> sketch_bytes,
      rmm::cuda_stream_view stream) {
    auto sketch_size = static_cast<cudf::size_type>(sketch_bytes.size());

    cudf::size_type offsets[2] = {0, sketch_size};
    rmm::device_buffer offsets_device{2 * sizeof(cudf::size_type), stream};
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        offsets_device.data(),
        offsets,
        2 * sizeof(cudf::size_type),
        cudaMemcpyHostToDevice,
        stream.value()));

    rmm::device_buffer chars_buffer{sketch_bytes.size(), stream};
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        chars_buffer.data(),
        sketch_bytes.data(),
        sketch_bytes.size(),
        cudaMemcpyDeviceToDevice,
        stream.value()));

    // Sync stream before stack-allocated offsets goes out of scope
    stream.synchronize();

    auto offsets_column = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        2,
        std::move(offsets_device),
        rmm::device_buffer{},
        0);

    return cudf::make_strings_column(
        1,
        std::move(offsets_column),
        std::move(chars_buffer),
        0,
        rmm::device_buffer{});
  }

  template <typename Func>
  auto mergeSketchesAndApply(
      cudf::column_view const& sketch_column,
      Func&& func,
      rmm::cuda_stream_view stream) {
    auto strings_col = cudf::strings_column_view(sketch_column);
    auto offsets_col = strings_col.offsets();
    auto chars_ptr = strings_col.chars_begin(stream);

    auto num_offsets = sketch_column.size() + 1;
    std::vector<cudf::size_type> host_offsets(num_offsets);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        host_offsets.data(),
        offsets_col.begin<cudf::size_type>(),
        num_offsets * sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize(); // Need host_offsets before proceeding

    cudf::size_type first_offset = host_offsets[0];
    cudf::size_type first_size = host_offsets[1] - first_offset;

    // Copy to mutable aligned buffer - cudf::approx_distinct_count requires
    // non-const span and proper alignment for int32 registers
    rmm::device_buffer aligned_sketch{
        static_cast<std::size_t>(first_size), stream};
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        aligned_sketch.data(),
        chars_ptr + first_offset,
        static_cast<std::size_t>(first_size),
        cudaMemcpyDeviceToDevice,
        stream.value()));

    cudf::approx_distinct_count merged_sketch(
        cuda::std::span<cuda::std::byte>(
            static_cast<cuda::std::byte*>(aligned_sketch.data()), first_size),
        precision_,
        kNullPolicy,
        kNanPolicy);

    for (cudf::size_type i = 1; i < sketch_column.size(); ++i) {
      cudf::size_type start_offset = host_offsets[i];
      cudf::size_type end_offset = host_offsets[i + 1];
      cudf::size_type size = end_offset - start_offset;

      if (size > 0) {
        rmm::device_buffer temp_sketch{static_cast<std::size_t>(size), stream};
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            temp_sketch.data(),
            chars_ptr + start_offset,
            size,
            cudaMemcpyDeviceToDevice,
            stream.value()));

        merged_sketch.merge(
            cuda::std::span<cuda::std::byte>(
                static_cast<cuda::std::byte*>(temp_sketch.data()), size),
            stream);
      }
    }

    return func(merged_sketch);
  }

  std::unique_ptr<cudf::column> doPartialReduce(
      cudf::table_view const& input,
      rmm::cuda_stream_view stream) {
    auto inputTable = cudf::table_view({input.column(inputIndex)});

    cudf::approx_distinct_count sketch{
        inputTable, precision_, kNullPolicy, kNanPolicy, stream};

    return makeSketchColumn(sketch.sketch(), stream);
  }

  std::unique_ptr<cudf::column> doIntermediateReduce(
      cudf::table_view const& input,
      rmm::cuda_stream_view stream) {
    auto sketch_column = input.column(inputIndex);

    if (sketch_column.size() == 0) {
      return makeSketchColumn({}, stream);
    }

    return mergeSketchesAndApply(
        sketch_column,
        [this, stream](cudf::approx_distinct_count& sketch) {
          return makeSketchColumn(sketch.sketch(), stream);
        },
        stream);
  }

  std::unique_ptr<cudf::column> doFinalReduce(
      cudf::table_view const& input,
      rmm::cuda_stream_view stream) {
    auto sketch_column = input.column(inputIndex);

    if (sketch_column.size() == 0) {
      return cudf::make_column_from_scalar(
          cudf::numeric_scalar<int64_t>(0, true, stream, get_temp_mr()),
          1,
          stream,
          get_output_mr());
    }

    return mergeSketchesAndApply(
        sketch_column,
        [stream](cudf::approx_distinct_count& sketch) {
          std::size_t estimate = sketch.estimate(stream);
          return cudf::make_column_from_scalar(
              cudf::numeric_scalar<int64_t>(
                  static_cast<int64_t>(estimate), true, stream, get_temp_mr()),
              1,
              stream,
              get_output_mr());
        },
        stream);
  }

  std::int32_t precision_;
};

std::unique_ptr<ReduceAggregator> createReduceAggregator(
    const ResolvedAggregateInfo& p) {
  auto const& kind = p.kind;
  auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (kind.rfind(prefix + "sum", 0) == 0) {
    return std::make_unique<ReduceSumAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "count", 0) == 0) {
    VELOX_CHECK(p.countInputKind.has_value());
    return std::make_unique<ReduceCountAggregator>(
        p.companionStep, p.inputIndex, *p.countInputKind, p.resultType);
  } else if (kind.rfind(prefix + "min", 0) == 0) {
    return std::make_unique<ReduceMinAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "max", 0) == 0) {
    return std::make_unique<ReduceMaxAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "avg", 0) == 0) {
    return std::make_unique<ReduceMeanAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "approx_distinct", 0) == 0) {
    return std::make_unique<ApproxDistinctAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else {
    VELOX_NYI("Reduce aggregation not yet supported, kind: {}", kind);
  }
}

} // namespace

namespace facebook::velox::cudf_velox {

std::vector<std::unique_ptr<ReduceAggregator>> toReduceAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants) {
  auto params =
      resolveAggregateInfos(aggregationNode, step, outputType, constants);

  std::vector<std::unique_ptr<ReduceAggregator>> aggregators;
  aggregators.reserve(params.size());
  for (const auto& p : params) {
    aggregators.push_back(createReduceAggregator(p));
  }
  return aggregators;
}

bool canReduceAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx) {
  return canAggregationBeEvaluatedByRegistry(
      getReduceAggregationRegistry(), call, step, rawInputTypes, queryCtx);
}

bool canReduceBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx) {
  const core::PlanNode* sourceNode = aggregationNode.sources().empty()
      ? nullptr
      : aggregationNode.sources()[0].get();

  // Get the aggregation step from the node
  auto step = aggregationNode.step();

  // Check supported aggregation functions using reduce registry
  for (const auto& aggregate : aggregationNode.aggregates()) {
    // Use step-aware validation that handles partial/final/intermediate steps
    if (!canReduceAggregationBeEvaluatedByCudf(
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

  return true;
}

CudfReduce::CudfReduce(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<core::AggregationNode const> const& aggregationNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          aggregationNode->outputType(),
          aggregationNode->id(),
          std::string{"CudfReduce"} +
              std::string{
                  core::AggregationNode::toName(aggregationNode->step())},
          nvtx3::rgb{34, 139, 34}, // Forest Green
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          aggregationNode),
      aggregationNode_(aggregationNode),
      isPartialOutput_(
          exec::isPartialOutput(aggregationNode->step()) &&
          !hasFinalAggs(aggregationNode->aggregates())) {}

void CudfReduce::initialize() {
  Operator::initialize();

  inputType_ = aggregationNode_->sources()[0]->outputType();

  numAggregates_ = aggregationNode_->aggregates().size();
  const auto inputRowSchema = asRowType(inputType_);
  std::vector<column_index_t> emptyKeys;
  auto aggregationInput = buildAggregationInputChannels(
      *aggregationNode_, *operatorCtx_, inputRowSchema, emptyKeys);
  aggregationInputChannels_ = std::move(aggregationInput.channels);
  aggregators_ = toReduceAggregators(
      *aggregationNode_,
      aggregationNode_->step(),
      outputType_,
      aggregationInput.constants);

  aggregationNode_.reset();
}

void CudfReduce::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  numInputRows_ += input->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  inputs_.push_back(std::move(cudfInput));
}

CudfVectorPtr CudfReduce::doGlobalAggregation(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> resultColumns;
  resultColumns.reserve(aggregators_.size());
  for (auto i = 0; i < aggregators_.size(); i++) {
    resultColumns.push_back(
        aggregators_[i]->doReduce(
            tableView, outputType_->childAt(i), stream, numInputRows_));
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(),
      outputType_,
      1,
      std::make_unique<cudf::table>(std::move(resultColumns)),
      stream);
}

RowVectorPtr CudfReduce::doGetOutput() {
  if (finished_) {
    return nullptr;
  }

  if (!isPartialOutput_ && !noMoreInput_) {
    // Final aggregation has to wait for all batches to arrive so we cannot
    // return any results here.
    return nullptr;
  }

  if (inputs_.empty() && !noMoreInput_) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  auto tbl = getConcatenatedTable(
      std::move(inputs_), inputType_, stream, get_output_mr());

  // Release input data after synchronizing.
  stream.synchronize();
  inputs_.clear();

  if (noMoreInput_) {
    finished_ = true;
  }

  VELOX_CHECK_NOT_NULL(tbl);

  auto tableView = tbl->view().num_columns() == 0
      ? tbl->view()
      : tbl->view().select(
            aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  auto output = doGlobalAggregation(tableView, stream);
  if (isPartialOutput_ && !noMoreInput_) {
    numInputRows_ = 0;
  }
  return output;
}

void CudfReduce::doNoMoreInput() {
  Operator::noMoreInput();
  if (isPartialOutput_ && inputs_.empty()) {
    finished_ = true;
  }
}

bool CudfReduce::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
