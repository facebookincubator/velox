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

#include <fmt/format.h>

#include <string_view>

namespace {

using namespace facebook::velox;
using cudf_velox::CountInputKind;
using cudf_velox::get_output_mr;
using cudf_velox::get_temp_mr;
using cudf_velox::GroupbyAggregator;
using cudf_velox::ResolvedAggregateInfo;
using cudf_velox::StreamingGroupbyAggregator;

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
    const bool countAll = (inputKind_ != CountInputKind::kColumn);
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

namespace {

constexpr std::string_view kStreamingGroupbyApiUsedStat{
    "streamingGroupbyApiUsed"};
constexpr std::string_view kStreamingGroupbyApiRebuildsStat{
    "streamingGroupbyApiRebuilds"};

struct GroupbyLeafState final : public BufferedState {
  explicit GroupbyLeafState(InputChunk chunk) : chunk(std::move(chunk)) {}

  InputChunk chunk;
};

bool isStreamingGroupbyCapacityError(const std::exception& e) {
  return std::string_view{e.what()}.find("max_groups") != std::string_view::npos;
}

uint64_t estimateStreamingRowWidth(const RowTypePtr& rowType) {
  uint64_t width = 0;
  for (const auto& child : rowType->children()) {
    width += child->isFixedWidth() ? child->cppSizeInBytes() : 16;
  }
  return std::max<uint64_t>(width, 1);
}

std::unique_ptr<cudf::column> castStreamingResult(
    std::unique_ptr<cudf::column> col,
    const TypePtr& type,
    rmm::cuda_stream_view stream) {
  const auto cudfType = cudf::data_type(cudf_velox::veloxToCudfTypeId(type));
  if (col->type() != cudfType) {
    col = cudf::cast(*col, cudfType, stream, get_output_mr());
  }
  return col;
}

std::unique_ptr<cudf::column> copyAndCastBufferedColumn(
    cudf::column_view const& col,
    const TypePtr& type,
    rmm::cuda_stream_view stream) {
  return castStreamingResult(
      std::make_unique<cudf::column>(col, stream, get_output_mr()),
      type,
      stream);
}

#define DEFINE_SIMPLE_STREAMING_GROUPBY_AGGREGATOR(Name, name)                  \
  struct StreamingGroupby##Name##Aggregator final : StreamingGroupbyAggregator { \
    StreamingGroupby##Name##Aggregator(                                         \
        core::AggregationNode::Step step,                                       \
        uint32_t aggregateIndex,                                                \
        uint32_t inputIndex,                                                    \
        VectorPtr constant,                                                     \
        const TypePtr& inputType,                                               \
        const TypePtr& bufferedType,                                            \
        const TypePtr& finalType)                                               \
        : StreamingGroupbyAggregator(                                           \
              step,                                                             \
              aggregateIndex,                                                   \
              inputIndex,                                                       \
              constant,                                                         \
              inputType,                                                        \
              bufferedType,                                                     \
              finalType) {}                                                     \
                                                                               \
    void addPreparedColumns(                                                    \
        std::vector<StreamingPreparedColumn>& columns) override {               \
      VELOX_CHECK(constant == nullptr, #Name " does not support constant input"); \
      preparedInputIndex_ = addPreparedColumn(                                  \
          columns,                                                              \
          std::nullopt,                                                         \
          inputType,                                                            \
          fmt::format("a{}_{}", aggregateIndex, #name));                        \
    }                                                                          \
                                                                               \
    void addStreamingRequest(                                                   \
        std::vector<cudf::groupby::streaming_aggregation_request>& requests)    \
        override {                                                              \
      requests.push_back(cudf::groupby::streaming_aggregation_request{          \
          static_cast<cudf::size_type>(preparedInputIndex_),                    \
          cudf::make_##name##_aggregation<cudf::groupby_aggregation>()});       \
      resultIndex_ = requests.size() - 1;                                       \
    }                                                                          \
                                                                               \
    std::unique_ptr<cudf::column> makeBufferedOutputColumn(                     \
        std::vector<cudf::groupby::aggregation_result>& results,                \
        rmm::cuda_stream_view stream) override {                                \
      return castStreamingResult(                                               \
          std::move(results[resultIndex_].results[0]), bufferedType, stream);   \
    }                                                                          \
                                                                               \
    std::unique_ptr<cudf::column> makeFinalOutputColumn(                        \
        cudf::column_view const& bufferedColumn,                                \
        rmm::cuda_stream_view stream) override {                                \
      return copyAndCastBufferedColumn(bufferedColumn, finalType, stream);      \
    }                                                                          \
                                                                               \
   private:                                                                    \
    column_index_t preparedInputIndex_;                                         \
    uint32_t resultIndex_;                                                      \
  };

DEFINE_SIMPLE_STREAMING_GROUPBY_AGGREGATOR(Sum, sum)
DEFINE_SIMPLE_STREAMING_GROUPBY_AGGREGATOR(Min, min)
DEFINE_SIMPLE_STREAMING_GROUPBY_AGGREGATOR(Max, max)

struct StreamingGroupbyCountAggregator final : StreamingGroupbyAggregator {
  StreamingGroupbyCountAggregator(
      core::AggregationNode::Step step,
      uint32_t aggregateIndex,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& inputType,
      CountInputKind inputKind,
      const TypePtr& bufferedType,
      const TypePtr& finalType)
      : StreamingGroupbyAggregator(
            step,
            aggregateIndex,
            inputIndex,
            constant,
            inputType,
            bufferedType,
            finalType),
        inputKind_(inputKind) {}

  void addPreparedColumns(
      std::vector<StreamingPreparedColumn>& columns) override {
    preparedInputIndex_ = addPreparedColumn(
        columns,
        std::nullopt,
        inputType,
        fmt::format("a{}_count", aggregateIndex));
  }

  void addStreamingRequest(
      std::vector<cudf::groupby::streaming_aggregation_request>& requests)
      override {
    const bool countAll = inputKind_ != CountInputKind::kColumn;
    requests.push_back(cudf::groupby::streaming_aggregation_request{
        static_cast<cudf::size_type>(preparedInputIndex_),
        exec::isRawInput(step)
            ? cudf::make_count_aggregation<cudf::groupby_aggregation>(
                  countAll ? cudf::null_policy::INCLUDE
                           : cudf::null_policy::EXCLUDE)
            : cudf::make_sum_aggregation<cudf::groupby_aggregation>()});
    resultIndex_ = requests.size() - 1;
  }

  std::unique_ptr<cudf::column> makeBufferedOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    auto col = std::move(results[resultIndex_].results[0]);
    if (exec::isRawInput(step) && inputKind_ == CountInputKind::kNullConstant) {
      auto zero = cudf::numeric_scalar<int64_t>(0, true, stream, get_temp_mr());
      col = cudf::make_column_from_scalar(
          zero, col->size(), stream, get_output_mr());
    }
    return castStreamingResult(std::move(col), bufferedType, stream);
  }

  std::unique_ptr<cudf::column> makeFinalOutputColumn(
      cudf::column_view const& bufferedColumn,
      rmm::cuda_stream_view stream) override {
    return copyAndCastBufferedColumn(bufferedColumn, finalType, stream);
  }

 private:
  CountInputKind inputKind_;
  column_index_t preparedInputIndex_;
  uint32_t resultIndex_;
};

struct StreamingGroupbyMeanAggregator final : StreamingGroupbyAggregator {
  StreamingGroupbyMeanAggregator(
      core::AggregationNode::Step step,
      uint32_t aggregateIndex,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& inputType,
      const TypePtr& bufferedType,
      const TypePtr& finalType)
      : StreamingGroupbyAggregator(
            step,
            aggregateIndex,
            inputIndex,
            constant,
            inputType,
            bufferedType,
            finalType) {}

  void addPreparedColumns(
      std::vector<StreamingPreparedColumn>& columns) override {
    VELOX_CHECK(constant == nullptr, "Avg does not support constant input");
    if (exec::isRawInput(step)) {
      sumInputIndex_ = addPreparedColumn(
          columns,
          std::nullopt,
          inputType,
          fmt::format("a{}_avg", aggregateIndex));
      countInputIndex_ = sumInputIndex_;
      return;
    }

    auto inputRowType = asRowType(inputType);
    sumInputIndex_ = addPreparedColumn(
        columns,
        0,
        inputRowType->childAt(0),
        fmt::format("a{}_avg_sum", aggregateIndex));
    countInputIndex_ = addPreparedColumn(
        columns,
        1,
        inputRowType->childAt(1),
        fmt::format("a{}_avg_count", aggregateIndex));
  }

  void addStreamingRequest(
      std::vector<cudf::groupby::streaming_aggregation_request>& requests)
      override {
    requests.push_back(cudf::groupby::streaming_aggregation_request{
        static_cast<cudf::size_type>(sumInputIndex_),
        cudf::make_sum_aggregation<cudf::groupby_aggregation>()});
    sumResultIndex_ = requests.size() - 1;
    requests.push_back(cudf::groupby::streaming_aggregation_request{
        static_cast<cudf::size_type>(countInputIndex_),
        exec::isRawInput(step)
            ? cudf::make_count_aggregation<cudf::groupby_aggregation>(
                  cudf::null_policy::EXCLUDE)
            : cudf::make_sum_aggregation<cudf::groupby_aggregation>()});
    countResultIndex_ = requests.size() - 1;
  }

  std::unique_ptr<cudf::column> makeBufferedOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    auto outputType = asRowType(bufferedType);
    auto sum = castStreamingResult(
        std::move(results[sumResultIndex_].results[0]),
        outputType->childAt(0),
        stream);
    auto count = castStreamingResult(
        std::move(results[countResultIndex_].results[0]),
        outputType->childAt(1),
        stream);
    auto size = sum->size();
    std::vector<std::unique_ptr<cudf::column>> children;
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

  std::unique_ptr<cudf::column> makeFinalOutputColumn(
      cudf::column_view const& bufferedColumn,
      rmm::cuda_stream_view stream) override {
    return cudf::binary_operation(
        bufferedColumn.child(0),
        bufferedColumn.child(1),
        cudf::binary_operator::DIV,
        cudf::data_type(cudf_velox::veloxToCudfTypeId(finalType)),
        stream,
        get_output_mr());
  }

 private:
  column_index_t sumInputIndex_;
  column_index_t countInputIndex_;
  uint32_t sumResultIndex_;
  uint32_t countResultIndex_;
};

std::unique_ptr<StreamingGroupbyAggregator> createStreamingGroupbyAggregator(
    uint32_t aggregateIndex,
    const ResolvedAggregateInfo& p,
    uint32_t rawInputIndex,
    const TypePtr& inputType,
    const TypePtr& bufferedType,
    const TypePtr& finalType) {
  auto const& kind = p.kind;
  auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (kind.rfind(prefix + "sum", 0) == 0) {
    return std::make_unique<StreamingGroupbySumAggregator>(
        p.companionStep,
        aggregateIndex,
        rawInputIndex,
        p.constant,
        inputType,
        bufferedType,
        finalType);
  } else if (kind.rfind(prefix + "count", 0) == 0) {
    VELOX_CHECK(p.countInputKind.has_value());
    return std::make_unique<StreamingGroupbyCountAggregator>(
        p.companionStep,
        aggregateIndex,
        rawInputIndex,
        p.constant,
        inputType,
        *p.countInputKind,
        bufferedType,
        finalType);
  } else if (kind.rfind(prefix + "min", 0) == 0) {
    return std::make_unique<StreamingGroupbyMinAggregator>(
        p.companionStep,
        aggregateIndex,
        rawInputIndex,
        p.constant,
        inputType,
        bufferedType,
        finalType);
  } else if (kind.rfind(prefix + "max", 0) == 0) {
    return std::make_unique<StreamingGroupbyMaxAggregator>(
        p.companionStep,
        aggregateIndex,
        rawInputIndex,
        p.constant,
        inputType,
        bufferedType,
        finalType);
  } else if (kind.rfind(prefix + "avg", 0) == 0) {
    return std::make_unique<StreamingGroupbyMeanAggregator>(
        p.companionStep,
        aggregateIndex,
        rawInputIndex,
        p.constant,
        inputType,
        bufferedType,
        finalType);
  } else {
    VELOX_NYI("Streaming aggregation not yet supported, kind: {}", kind);
  }
}

#undef DEFINE_SIMPLE_STREAMING_GROUPBY_AGGREGATOR

} // namespace

class StreamingGroupbyLeafState final : public BufferedState {
 public:
  explicit StreamingGroupbyLeafState(CudfGroupby& owner)
      : owner_(owner),
        rowWidthBytes_(estimateStreamingRowWidth(owner.bufferedResultType_)) {}

  void addChunk(InputChunk input) {
    if (input.empty()) {
      return;
    }

    lastStream_ = input.stream;
    auto const inputRows = input.size();
    auto const inputFlatSize = input.owner ? input.owner->estimateFlatSize() : 0;

    if (!groupby_) {
      currentCapacity_ = std::min<size_t>(
          owner_.maxBufferedRows_, std::max<size_t>(inputRows * 4, 1));
      groupby_ = owner_.createStreamingGroupby(currentCapacity_);
      groupby_->aggregate(input.view, input.stream);
    } else if (!tryAggregate(*groupby_, input.view, input.stream)) {
      growAndAggregate(input);
    }

    totalRows_ += inputRows;
    estimatedFlatSize_ =
        std::max<uint64_t>(
            estimatedFlatSize_ + inputFlatSize,
            currentCapacity_ * rowWidthBytes_);
  }

  size_t totalRows() const {
    return totalRows_;
  }

  uint64_t estimatedFlatSize() const {
    return estimatedFlatSize_;
  }

  CudfVectorPtr finalizeBuffered() const {
    if (!groupby_) {
      return nullptr;
    }
    return owner_.materializeStreamingBufferedOutput(*groupby_, lastStream_);
  }

 private:
  bool tryAggregate(
      cudf::groupby::streaming_groupby& groupby,
      cudf::table_view inputView,
      rmm::cuda_stream_view stream) const {
    try {
      groupby.aggregate(inputView, stream);
      return true;
    } catch (const std::exception& e) {
      if (isStreamingGroupbyCapacityError(e)) {
        return false;
      }
      throw;
    }
  }

  void growAndAggregate(const InputChunk& input) {
    while (currentCapacity_ < owner_.maxBufferedRows_) {
      auto newCapacity = std::min<size_t>(
          owner_.maxBufferedRows_,
          std::max<size_t>(currentCapacity_ * 2, input.size()));
      if (newCapacity == currentCapacity_) {
        break;
      }

      auto grown = owner_.createStreamingGroupby(newCapacity);
      try {
        grown->aggregate(input.view, input.stream);
        if (groupby_) {
          grown->merge(*groupby_, input.stream);
        }
        groupby_ = std::move(grown);
        currentCapacity_ = newCapacity;
        {
          auto lockedStats = owner_.stats_.wlock();
          lockedStats->addRuntimeStat(
              std::string{kStreamingGroupbyApiRebuildsStat}, RuntimeCounter(1));
        }
        return;
      } catch (const std::exception& e) {
        if (!isStreamingGroupbyCapacityError(e)) {
          throw;
        }
      }
    }

    VELOX_FAIL(
        "streaming_groupby reached the capacity ceiling of {} rows",
        owner_.maxBufferedRows_);
  }

  CudfGroupby& owner_;
  const uint64_t rowWidthBytes_;
  std::unique_ptr<cudf::groupby::streaming_groupby> groupby_;
  size_t currentCapacity_{0};
  size_t totalRows_{0};
  uint64_t estimatedFlatSize_{0};
  rmm::cuda_stream_view lastStream_{rmm::cuda_stream_default};
};

class BufferedGroupbyStateOps final : public BufferedStateOps {
 public:
  explicit BufferedGroupbyStateOps(CudfGroupby& owner) : owner_(owner) {
    keyIndices_.reserve(owner_.groupingKeyOutputChannels_.size());
    for (auto keyIndex : owner_.groupingKeyOutputChannels_) {
      keyIndices_.push_back(static_cast<cudf::size_type>(keyIndex));
    }
  }

  InputChunk prepareInput(CudfVectorPtr rawInput) override {
    auto stream = rawInput->stream();
    auto permutedInputView = rawInput->getTableView().select(
        owner_.aggregationInputChannels_.begin(),
        owner_.aggregationInputChannels_.end());

    if (owner_.isPartialOutput_) {
      auto compacted = owner_.doGroupByAggregation(
          permutedInputView,
          owner_.groupingKeyOutputChannels_,
          owner_.aggregators_,
          owner_.bufferedResultType_,
          stream);
      return compacted
          ? makeOwnedChunk(std::move(compacted), owner_.bufferedResultType_)
          : InputChunk{};
    }

    if (!owner_.isSingleStep_) {
      return makeBorrowedChunk(
          std::move(rawInput), owner_.bufferedResultType_, permutedInputView);
    }

    auto compacted = owner_.doGroupByAggregation(
        permutedInputView,
        owner_.groupingKeyOutputChannels_,
        owner_.partialAggregators_,
        owner_.bufferedResultType_,
        stream);
    return compacted
        ? makeOwnedChunk(std::move(compacted), owner_.bufferedResultType_)
        : InputChunk{};
  }

  size_t estimatedMergedRowUpperBound(
      const BufferedState& leaf,
      const InputChunk& input) const override {
    return asLeafState(leaf).chunk.size() + input.size();
  }

  std::unique_ptr<BufferedState> createLeaf(InputChunk input) override {
    return std::make_unique<GroupbyLeafState>(std::move(input));
  }

  void addInputToLeaf(BufferedState& leaf, InputChunk input) override {
    auto& groupbyLeaf = asLeafState(leaf);
    groupbyLeaf.chunk =
        mergeChunks(std::move(groupbyLeaf.chunk), std::move(input));
  }

  size_t leafRowCount(const BufferedState& leaf) const override {
    return asLeafState(leaf).chunk.size();
  }

  uint64_t leafFlatSize(const BufferedState& leaf) const override {
    const auto& chunk = asLeafState(leaf).chunk;
    return chunk.owner ? chunk.owner->estimateFlatSize() : 0;
  }

  std::vector<InputChunk> partitionInput(
      InputChunk input,
      const PartitionSpec& spec) override {
    if (input.empty()) {
      return std::vector<InputChunk>(spec.numPartitions);
    }

    auto partitions = hashPartitionTable(
        input.view,
        input.pool,
        input.type,
        input.stream,
        spec.keyIndices,
        spec.numPartitions,
        spec.hashId,
        spec.seed,
        input.stream);

    std::vector<InputChunk> chunks(spec.numPartitions);
    for (int32_t i = 0; i < spec.numPartitions; ++i) {
      if (partitions[i]) {
        chunks[i] = makeOwnedChunk(std::move(partitions[i]), input.type);
      }
    }
    return chunks;
  }

  std::vector<std::unique_ptr<BufferedState>> repartitionLeaf(
      std::unique_ptr<BufferedState> leaf,
      const PartitionSpec& spec) override {
    auto groupbyLeaf = std::unique_ptr<GroupbyLeafState>(
        static_cast<GroupbyLeafState*>(leaf.release()));
    auto partitions = partitionInput(std::move(groupbyLeaf->chunk), spec);

    std::vector<std::unique_ptr<BufferedState>> leaves(spec.numPartitions);
    for (int32_t i = 0; i < spec.numPartitions; ++i) {
      if (!partitions[i].empty()) {
        leaves[i] =
            std::make_unique<GroupbyLeafState>(std::move(partitions[i]));
      }
    }
    return leaves;
  }

  CudfVectorPtr finalizeLeaf(std::unique_ptr<BufferedState> leaf) override {
    auto groupbyLeaf = std::unique_ptr<GroupbyLeafState>(
        static_cast<GroupbyLeafState*>(leaf.release()));
    if (owner_.isPartialOutput_) {
      return std::move(groupbyLeaf->chunk.owner);
    }
    auto& finalAggregators =
        owner_.isSingleStep_ ? owner_.finalAggregators_ : owner_.aggregators_;
    return owner_.doGroupByAggregation(
        groupbyLeaf->chunk.view,
        owner_.groupingKeyOutputChannels_,
        finalAggregators,
        owner_.outputType_,
        groupbyLeaf->chunk.stream);
  }

  const std::vector<cudf::size_type>& keyIndices() const override {
    return keyIndices_;
  }

 private:
  CudfGroupby& owner_;
  std::vector<cudf::size_type> keyIndices_;

  GroupbyLeafState& asLeafState(BufferedState& leaf) const {
    return static_cast<GroupbyLeafState&>(leaf);
  }

  const GroupbyLeafState& asLeafState(const BufferedState& leaf) const {
    return static_cast<const GroupbyLeafState&>(leaf);
  }

  InputChunk makeOwnedChunk(CudfVectorPtr owner, const TypePtr& type) const {
    return InputChunk{
        owner->pool(),
        type,
        owner->getTableView(),
        owner->stream(),
        std::move(owner)};
  }

  InputChunk makeBorrowedChunk(
      CudfVectorPtr owner,
      const TypePtr& type,
      cudf::table_view view) const {
    return InputChunk{
        owner->pool(), type, view, owner->stream(), std::move(owner)};
  }

  InputChunk mergeChunks(InputChunk left, InputChunk right) const {
    if (left.empty()) {
      return right;
    }
    if (right.empty()) {
      return left;
    }

    auto stream = left.stream;
    std::vector<cudf::table_view> views{left.view, right.view};
    std::vector<rmm::cuda_stream_view> inputStreams{left.stream, right.stream};
    auto concatenatedTable =
        concatenateViews(views, inputStreams, stream, get_temp_mr());
    auto merged = owner_.doGroupByAggregation(
        concatenatedTable->view(),
        owner_.groupingKeyOutputChannels_,
        owner_.intermediateAggregators_,
        owner_.bufferedResultType_,
        stream);
    return merged
        ? makeOwnedChunk(std::move(merged), owner_.bufferedResultType_)
        : InputChunk{};
  }
};

class StreamingGroupbyBufferedStateOps final : public BufferedStateOps {
 public:
  explicit StreamingGroupbyBufferedStateOps(CudfGroupby& owner) : owner_(owner) {
    keyIndices_.reserve(owner_.groupingKeyOutputChannels_.size());
    for (auto keyIndex : owner_.groupingKeyOutputChannels_) {
      keyIndices_.push_back(static_cast<cudf::size_type>(keyIndex));
    }
  }

  InputChunk prepareInput(CudfVectorPtr rawInput) override {
    auto preparedView =
        owner_.makeStreamingPreparedInputView(rawInput->getTableView());
    return makeBorrowedChunk(
        std::move(rawInput), owner_.streamingPreparedType_, preparedView);
  }

  size_t estimatedMergedRowUpperBound(
      const BufferedState& leaf,
      const InputChunk& input) const override {
    return asLeafState(leaf).totalRows() + input.size();
  }

  std::unique_ptr<BufferedState> createLeaf(InputChunk input) override {
    auto leaf = std::make_unique<StreamingGroupbyLeafState>(owner_);
    leaf->addChunk(std::move(input));
    return leaf;
  }

  void addInputToLeaf(BufferedState& leaf, InputChunk input) override {
    asLeafState(leaf).addChunk(std::move(input));
  }

  size_t leafRowCount(const BufferedState& leaf) const override {
    return asLeafState(leaf).totalRows();
  }

  uint64_t leafFlatSize(const BufferedState& leaf) const override {
    return asLeafState(leaf).estimatedFlatSize();
  }

  std::vector<InputChunk> partitionInput(
      InputChunk input,
      const PartitionSpec& spec) override {
    if (input.empty()) {
      return std::vector<InputChunk>(spec.numPartitions);
    }

    auto partitions = hashPartitionTable(
        input.view,
        input.pool,
        input.type,
        input.stream,
        spec.keyIndices,
        spec.numPartitions,
        spec.hashId,
        spec.seed,
        input.stream);

    std::vector<InputChunk> chunks(spec.numPartitions);
    for (int32_t i = 0; i < spec.numPartitions; ++i) {
      if (partitions[i]) {
        chunks[i] =
            makeOwnedChunk(std::move(partitions[i]), owner_.streamingPreparedType_);
      }
    }
    return chunks;
  }

  std::vector<std::unique_ptr<BufferedState>> repartitionLeaf(
      std::unique_ptr<BufferedState> leaf,
      const PartitionSpec& spec) override {
    auto streamingLeaf = std::unique_ptr<StreamingGroupbyLeafState>(
        static_cast<StreamingGroupbyLeafState*>(leaf.release()));
    auto buffered = streamingLeaf->finalizeBuffered();
    if (!buffered) {
      return std::vector<std::unique_ptr<BufferedState>>(spec.numPartitions);
    }

    InputChunk bufferedChunk{
        buffered->pool(),
        owner_.bufferedResultType_,
        buffered->getTableView(),
        buffered->stream(),
        std::move(buffered)};

    auto partitions = hashPartitionTable(
        bufferedChunk.view,
        bufferedChunk.pool,
        bufferedChunk.type,
        bufferedChunk.stream,
        spec.keyIndices,
        spec.numPartitions,
        spec.hashId,
        spec.seed,
        bufferedChunk.stream);

    std::vector<std::unique_ptr<BufferedState>> leaves(spec.numPartitions);
    for (int32_t i = 0; i < spec.numPartitions; ++i) {
      if (!partitions[i]) {
        continue;
      }
      auto prepared = prepareOwnedChunk(std::move(partitions[i]));
      if (!prepared.empty()) {
        leaves[i] = createLeaf(std::move(prepared));
      }
    }
    return leaves;
  }

  CudfVectorPtr finalizeLeaf(std::unique_ptr<BufferedState> leaf) override {
    auto streamingLeaf = std::unique_ptr<StreamingGroupbyLeafState>(
        static_cast<StreamingGroupbyLeafState*>(leaf.release()));
    auto buffered = streamingLeaf->finalizeBuffered();
    if (owner_.isPartialOutput_) {
      return buffered;
    }
    return owner_.finalizeStreamingBufferedOutput(std::move(buffered));
  }

  const std::vector<cudf::size_type>& keyIndices() const override {
    return keyIndices_;
  }

 private:
  CudfGroupby& owner_;
  std::vector<cudf::size_type> keyIndices_;

  StreamingGroupbyLeafState& asLeafState(BufferedState& leaf) const {
    return static_cast<StreamingGroupbyLeafState&>(leaf);
  }

  const StreamingGroupbyLeafState& asLeafState(
      const BufferedState& leaf) const {
    return static_cast<const StreamingGroupbyLeafState&>(leaf);
  }

  InputChunk makeBorrowedChunk(
      CudfVectorPtr owner,
      const TypePtr& type,
      cudf::table_view view) const {
    return InputChunk{
        owner->pool(), type, view, owner->stream(), std::move(owner)};
  }

  InputChunk makeOwnedChunk(CudfVectorPtr owner, const TypePtr& type) const {
    return InputChunk{
        owner->pool(),
        type,
        owner->getTableView(),
        owner->stream(),
        std::move(owner)};
  }

  InputChunk prepareOwnedChunk(CudfVectorPtr owner) const {
    auto preparedView =
        owner_.makeStreamingPreparedInputView(owner->getTableView());
    return makeBorrowedChunk(
        std::move(owner), owner_.streamingPreparedType_, preparedView);
  }
};

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

std::vector<std::unique_ptr<StreamingGroupbyAggregator>>
toStreamingGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    RowTypePtr const& inputType,
    std::vector<column_index_t> const& aggregationInputChannels,
    TypePtr const& bufferedOutputType,
    TypePtr const& finalOutputType,
    std::vector<VectorPtr> const& constants) {
  auto params =
      resolveAggregateInfos(aggregationNode, step, finalOutputType, constants);
  auto const numKeys = aggregationNode.groupingKeys().size();
  auto const bufferedRowType = asRowType(bufferedOutputType);
  auto const finalRowType = asRowType(finalOutputType);

  std::vector<std::unique_ptr<StreamingGroupbyAggregator>> aggregators;
  aggregators.reserve(params.size());
  for (size_t i = 0; i < params.size(); ++i) {
    auto const& param = params[i];
    auto const rawInputIndex = aggregationInputChannels[param.inputIndex];
    aggregators.push_back(createStreamingGroupbyAggregator(
        i,
        param,
        rawInputIndex,
        inputType->childAt(rawInputIndex),
        bufferedRowType->childAt(numKeys + i),
        finalRowType->childAt(numKeys + i)));
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

bool CudfGroupby::canUseStreamingGroupbyApi(
    const RowTypePtr& inputRowSchema,
    const std::vector<VectorPtr>& constants) const {
  auto const& config = CudfConfig::getInstance();
  if (!config.streamingGroupbyApiEnabled || !streamingEnabled_ || isSingleStep_ ||
      aggregationNode_->groupingKeys().empty() ||
      aggregationNode_->aggregates().empty()) {
    return false;
  }

  auto const step = aggregationNode_->step();
  if (step != core::AggregationNode::Step::kPartial &&
      step != core::AggregationNode::Step::kFinal) {
    return false;
  }

  auto const numKeys = groupingKeyOutputChannels_.size();
  auto const params =
      resolveAggregateInfos(*aggregationNode_, step, outputType_, constants);
  auto const prefix = config.functionNamePrefix;

  for (size_t i = 0; i < params.size(); ++i) {
    auto const& param = params[i];
    auto const& kind = param.kind;
    auto const inputChannel = aggregationInputChannels_[numKeys + i];
    auto const inputType = inputRowSchema->childAt(inputChannel);

    if (kind.rfind(prefix + "sum", 0) == 0 ||
        kind.rfind(prefix + "min", 0) == 0 ||
        kind.rfind(prefix + "max", 0) == 0) {
      if (param.constant != nullptr || !inputType->isFixedWidth()) {
        return false;
      }
      continue;
    }

    if (kind.rfind(prefix + "avg", 0) == 0) {
      if (step == core::AggregationNode::Step::kPartial) {
        if (param.constant != nullptr || !inputType->isFixedWidth()) {
          return false;
        }
      } else {
        if (inputType->kind() != TypeKind::ROW) {
          return false;
        }
        auto const rowType = asRowType(inputType);
        if (!rowType->childAt(0)->isFixedWidth() ||
            !rowType->childAt(1)->isFixedWidth()) {
          return false;
        }
      }
      continue;
    }

    if (kind.rfind(prefix + "count", 0) == 0) {
      if (!param.countInputKind.has_value() ||
          *param.countInputKind == CountInputKind::kNullConstant) {
        return false;
      }
      if (*param.countInputKind == CountInputKind::kColumn &&
          !inputType->isFixedWidth()) {
        return false;
      }
      continue;
    }

    return false;
  }

  return true;
}

cudf::table_view CudfGroupby::makeStreamingPreparedInputView(
    cudf::table_view rawInputView) const {
  std::vector<cudf::column_view> columns;
  columns.reserve(streamingPreparedColumns_.size());
  for (const auto& column : streamingPreparedColumns_) {
    auto view = rawInputView.column(column.inputIndex);
    if (column.childIndex.has_value()) {
      view = view.child(*column.childIndex);
    }
    columns.push_back(view);
  }
  return cudf::table_view(columns);
}

std::unique_ptr<cudf::groupby::streaming_groupby>
CudfGroupby::createStreamingGroupby(size_t maxGroups) const {
  VELOX_CHECK_LE(
      maxGroups,
      static_cast<size_t>(std::numeric_limits<cudf::size_type>::max()));

  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  for (auto const& aggregator : streamingGroupbyAggregators_) {
    aggregator->addStreamingRequest(requests);
  }

  std::vector<cudf::size_type> keyIndices;
  keyIndices.reserve(groupingKeyOutputChannels_.size());
  for (auto keyIndex : groupingKeyOutputChannels_) {
    keyIndices.push_back(static_cast<cudf::size_type>(keyIndex));
  }

  return std::make_unique<cudf::groupby::streaming_groupby>(
      keyIndices,
      requests,
      static_cast<cudf::size_type>(maxGroups),
      ignoreNullKeys_ ? cudf::null_policy::EXCLUDE
                      : cudf::null_policy::INCLUDE);
}

CudfVectorPtr CudfGroupby::materializeStreamingBufferedOutput(
    const cudf::groupby::streaming_groupby& groupby,
    rmm::cuda_stream_view stream) const {
  auto [groupKeys, results] = groupby.finalize(stream, get_output_mr());
  std::vector<std::unique_ptr<cudf::column>> resultColumns;

  auto groupKeysColumns = groupKeys->release();
  resultColumns.reserve(
      groupKeysColumns.size() + streamingGroupbyAggregators_.size());
  resultColumns.insert(
      resultColumns.end(),
      std::make_move_iterator(groupKeysColumns.begin()),
      std::make_move_iterator(groupKeysColumns.end()));

  for (auto const& aggregator : streamingGroupbyAggregators_) {
    resultColumns.push_back(
        aggregator->makeBufferedOutputColumn(results, stream));
  }

  auto resultTable = std::make_unique<cudf::table>(std::move(resultColumns));
  auto numRows = resultTable->num_rows();
  if (numRows == 0) {
    return nullptr;
  }

  auto outputType = isPartialOutput_ ? outputType_ : bufferedResultType_;
  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType, numRows, std::move(resultTable), stream);
}

CudfVectorPtr CudfGroupby::finalizeStreamingBufferedOutput(
    CudfVectorPtr bufferedOutput) const {
  if (!bufferedOutput) {
    return nullptr;
  }

  auto stream = bufferedOutput->stream();
  auto bufferedView = bufferedOutput->getTableView();
  std::vector<std::unique_ptr<cudf::column>> outputColumns;
  outputColumns.reserve(outputType_->size());

  for (size_t i = 0; i < groupingKeyOutputChannels_.size(); ++i) {
    outputColumns.push_back(std::make_unique<cudf::column>(
        bufferedView.column(i), stream, get_output_mr()));
  }

  for (size_t i = 0; i < streamingGroupbyAggregators_.size(); ++i) {
    auto bufferedIndex = groupingKeyOutputChannels_.size() + i;
    outputColumns.push_back(
        streamingGroupbyAggregators_[i]->makeFinalOutputColumn(
            bufferedView.column(bufferedIndex), stream));
  }

  auto resultTable = std::make_unique<cudf::table>(std::move(outputColumns));
  auto numRows = resultTable->num_rows();
  if (numRows == 0) {
    return nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, numRows, std::move(resultTable), stream);
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

    auto const& cudfConfig = CudfConfig::getInstance();
    maxBufferedRows_ = cudfConfig.batchSizeMaxThreshold
        ? static_cast<size_t>(cudfConfig.batchSizeMaxThreshold.value())
        : static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
    VELOX_CHECK_GT(maxBufferedRows_, 0);
    streamingGroupbyApiEnabled_ =
        canUseStreamingGroupbyApi(inputRowSchema, aggregationInput.constants);

    if (streamingGroupbyApiEnabled_) {
      auto const outputRowType = asRowType(outputType_);
      streamingPreparedColumns_.clear();
      streamingGroupbyAggregators_ = toStreamingGroupbyAggregators(
          *aggregationNode_,
          aggregationNode_->step(),
          inputRowSchema,
          aggregationInputChannels_,
          bufferedResultType_,
          outputType_,
          aggregationInput.constants);

      for (size_t i = 0; i < groupingKeyOutputChannels_.size(); ++i) {
        auto inputIndex = aggregationInputChannels_[i];
        streamingPreparedColumns_.push_back(
            StreamingPreparedColumn{
                inputIndex,
                std::nullopt,
                inputRowSchema->childAt(inputIndex),
                outputRowType->nameOf(i)});
      }

      for (auto& aggregator : streamingGroupbyAggregators_) {
        aggregator->addPreparedColumns(streamingPreparedColumns_);
      }

      std::vector<std::string> names;
      std::vector<TypePtr> types;
      names.reserve(streamingPreparedColumns_.size());
      types.reserve(streamingPreparedColumns_.size());
      for (auto const& column : streamingPreparedColumns_) {
        names.push_back(column.name);
        types.push_back(column.type);
      }
      streamingPreparedType_ = ROW(std::move(names), std::move(types));
      {
        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            std::string{kStreamingGroupbyApiUsedStat}, RuntimeCounter(1));
      }
    }

    if (isFinalOrSingle) {
      if (streamingGroupbyApiEnabled_) {
        partitionedBufferedState_ = std::make_unique<PartitionedBufferedState>(
            std::make_unique<StreamingGroupbyBufferedStateOps>(*this),
            maxBufferedRows_);
      } else {
        partitionedBufferedState_ = std::make_unique<PartitionedBufferedState>(
            std::make_unique<BufferedGroupbyStateOps>(*this),
            maxBufferedRows_);
      }
    } else if (isPartialOutput_) {
      if (streamingGroupbyApiEnabled_) {
        flushableBufferedState_ = std::make_unique<FlushableBufferedState>(
            std::make_unique<StreamingGroupbyBufferedStateOps>(*this),
            maxBufferedRows_,
            maxPartialAggregationMemoryUsage_);
      } else {
        flushableBufferedState_ = std::make_unique<FlushableBufferedState>(
            std::make_unique<BufferedGroupbyStateOps>(*this),
            maxBufferedRows_,
            maxPartialAggregationMemoryUsage_);
      }
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
  flushableBufferedState_->addInput(std::move(tbl));
}

void CudfGroupby::computeFinalGroupbyStreaming(CudfVectorPtr tbl) {
  partitionedBufferedState_->addInput(std::move(tbl));
}

void CudfGroupby::computeSingleGroupbyStreaming(CudfVectorPtr tbl) {
  partitionedBufferedState_->addInput(std::move(tbl));
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

CudfVectorPtr CudfGroupby::releasePartialOutput(CudfVectorPtr output) {
  auto numOutputRows = output->size();
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
  return output;
}

RowVectorPtr CudfGroupby::doGetOutput() {
  // Handle partial streaming groupby.
  if (isPartialOutput_ && streamingEnabled_) {
    if (!flushableBufferedState_) {
      return nullptr;
    }

    if (auto output = flushableBufferedState_->getOutput(noMoreInput_)) {
      return releasePartialOutput(std::move(output));
    }

    if (noMoreInput_) {
      finished_ = true;
    }
    return nullptr;
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
    auto result = partitionedBufferedState_
        ? partitionedBufferedState_->drainNextOutput()
        : nullptr;
    if (!result) {
      finished_ = true;
    }
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
  if (isPartialOutput_ && !streamingEnabled_ && inputs_.empty()) {
    finished_ = true;
  }
}

bool CudfGroupby::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
