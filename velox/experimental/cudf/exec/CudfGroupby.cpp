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
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/unary.hpp>

#include <fmt/format.h>

#include <atomic>
#include <cstdlib>
#include <limits>
#include <sstream>
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

struct GroupbyStddevSampAggregator : GroupbyAggregator {
  GroupbyStddevSampAggregator(
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
    request.values = tbl.column(inputIndex);

    switch (step) {
      case core::AggregationNode::Step::kSingle:
        // Use cuDF's built-in std aggregation with ddof=1 (sample stddev)
        request.aggregations.push_back(
            cudf::make_std_aggregation<cudf::groupby_aggregation>(1));
        break;
      case core::AggregationNode::Step::kPartial:
        // Compute count, mean, m2 from raw values
        request.aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
        request.aggregations.push_back(
            cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        request.aggregations.push_back(
            cudf::make_m2_aggregation<cudf::groupby_aggregation>());
        break;
      case core::AggregationNode::Step::kIntermediate:
      case core::AggregationNode::Step::kFinal:
        // Input is struct(count, mean, m2) - use MERGE_M2 to merge
        request.aggregations.push_back(
            cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>());
        break;
      default:
        VELOX_NYI("Unsupported aggregation step for stddev_samp");
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return std::move(results[outputIdx_].results[0]);
      case core::AggregationNode::Step::kPartial: {
        auto count = std::move(results[outputIdx_].results[0]);
        auto mean = std::move(results[outputIdx_].results[1]);
        auto m2 = std::move(results[outputIdx_].results[2]);
        return makeM2StructColumn(
            std::move(count), std::move(mean), std::move(m2), stream);
      }
      case core::AggregationNode::Step::kIntermediate: {
        auto merged = std::move(results[outputIdx_].results[0]);

        // Check if types already match expected output - avoid copies if so
        const auto& outputType = asRowType(resultType);
        auto const cudfCountType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(0)));
        auto const cudfMeanType = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(1)));
        auto const cudfM2Type = cudf::data_type(
            cudf_velox::veloxToCudfTypeId(outputType->childAt(2)));

        auto mergedView = merged->view();
        bool typesMatch = mergedView.child(0).type() == cudfCountType &&
            mergedView.child(1).type() == cudfMeanType &&
            mergedView.child(2).type() == cudfM2Type;

        if (typesMatch) {
          // Types match - return merged directly to avoid device copies
          return merged;
        }

        // Types don't match - need to copy and cast (use output_mr since
        // these become part of the output)
        auto count = std::make_unique<cudf::column>(
            mergedView.child(0), stream, get_output_mr());
        auto mean = std::make_unique<cudf::column>(
            mergedView.child(1), stream, get_output_mr());
        auto m2 = std::make_unique<cudf::column>(
            mergedView.child(2), stream, get_output_mr());
        return makeM2StructColumn(
            std::move(count), std::move(mean), std::move(m2), stream);
      }
      case core::AggregationNode::Step::kFinal: {
        // MERGE_M2 returns struct(count, mean, m2)
        // Compute sqrt(m2 / (count - 1)) with NULL where count < 2
        auto merged = std::move(results[outputIdx_].results[0]);
        auto mergedView = merged->view();
        auto countView = mergedView.child(0);
        auto m2View = mergedView.child(2);

        // count - 1 (binary_operation handles type promotion)
        cudf::numeric_scalar<double> one(1.0, true, stream, get_temp_mr());
        auto countMinus1 = cudf::binary_operation(
            countView,
            one,
            cudf::binary_operator::SUB,
            cudf::data_type{cudf::type_id::FLOAT64},
            stream,
            get_temp_mr());

        // m2 / (count - 1)
        auto variance = cudf::binary_operation(
            m2View,
            *countMinus1,
            cudf::binary_operator::DIV,
            cudf::data_type{cudf::type_id::FLOAT64},
            stream,
            get_temp_mr());

        // sqrt(variance)
        auto stddev = cudf::unary_operation(
            *variance, cudf::unary_operator::SQRT, stream, get_temp_mr());

        // count >= 2
        cudf::numeric_scalar<int64_t> two(2, true, stream, get_temp_mr());
        auto validMask = cudf::binary_operation(
            countView,
            two,
            cudf::binary_operator::GREATER_EQUAL,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());

        // Apply mask: where count < 2, result is NULL
        cudf::numeric_scalar<double> nullDouble(
            0.0, false, stream, get_temp_mr());
        return cudf::copy_if_else(
            *stddev, nullDouble, *validMask, stream, get_output_mr());
      }
      default:
        VELOX_NYI("Unsupported aggregation step for stddev_samp");
    }
  }

 private:
  // Build a struct column with (count, mean, m2), casting to expected types.
  std::unique_ptr<cudf::column> makeM2StructColumn(
      std::unique_ptr<cudf::column> count,
      std::unique_ptr<cudf::column> mean,
      std::unique_ptr<cudf::column> m2,
      rmm::cuda_stream_view stream) {
    const auto& outputType = asRowType(resultType);
    auto const cudfCountType =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType->childAt(0)));
    auto const cudfMeanType =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType->childAt(1)));
    auto const cudfM2Type =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType->childAt(2)));

    if (count->type() != cudfCountType) {
      count = cudf::cast(*count, cudfCountType, stream, get_output_mr());
    }
    if (mean->type() != cudfMeanType) {
      mean = cudf::cast(*mean, cudfMeanType, stream, get_output_mr());
    }
    if (m2->type() != cudfM2Type) {
      m2 = cudf::cast(*m2, cudfM2Type, stream, get_output_mr());
    }

    auto const size = count->size();
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::move(count));
    children.push_back(std::move(mean));
    children.push_back(std::move(m2));

    return std::make_unique<cudf::column>(
        cudf::data_type(cudf::type_id::STRUCT),
        size,
        rmm::device_buffer{},
        rmm::device_buffer{},
        0,
        std::move(children));
  }

  uint32_t outputIdx_;
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
  } else if (kind.rfind(prefix + "stddev_samp", 0) == 0) {
    return std::make_unique<GroupbyStddevSampAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "stddev", 0) == 0) {
    // stddev is an alias for stddev_samp
    return std::make_unique<GroupbyStddevSampAggregator>(
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
constexpr std::string_view kStreamingGroupbyApiRepartitionsStat{
    "streamingGroupbyApiRepartitions"};
constexpr std::string_view kStreamingGroupbyApiCapacityLimitStat{
    "streamingGroupbyApiCapacityLimit"};

constexpr std::string_view kStreamingGroupbyDiagnosticEnv{
    "VELOX_CUDF_STREAMING_GROUPBY_DIAGNOSTICS"};
constexpr std::string_view kStreamingGroupbyDiagnosticSyncEnv{
    "VELOX_CUDF_STREAMING_GROUPBY_DIAGNOSTICS_SYNC"};

std::atomic<uint64_t> nextStreamingLeafDiagnosticId{1};

bool environmentFlagEnabled(std::string_view name) {
  const auto* value = std::getenv(std::string{name}.c_str());
  return value != nullptr && std::string_view{value} != "0" &&
      std::string_view{value} != "false";
}

bool streamingGroupbyDiagnosticsEnabled() {
  static const bool enabled =
      environmentFlagEnabled(kStreamingGroupbyDiagnosticEnv);
  return enabled;
}

bool streamingGroupbyDiagnosticSyncEnabled() {
  static const bool enabled =
      environmentFlagEnabled(kStreamingGroupbyDiagnosticSyncEnv);
  return enabled;
}

std::string streamingGroupbyOperatorContext(const CudfGroupby& owner) {
  const auto* driver = owner.operatorCtx()->driverCtx();
  return fmt::format(
      "task={} planNode={} operator={} operatorType={} pipeline={} driver={} "
      "partition={} splitGroup={}",
      owner.taskId(),
      owner.planNodeId(),
      owner.operatorId(),
      owner.operatorType(),
      driver->pipelineId,
      driver->driverId,
      driver->partitionId,
      driver->splitGroupId);
}

void appendColumnDescription(
    std::ostringstream& out,
    const cudf::column_view& column,
    int depth) {
  out << "{type=" << static_cast<int>(column.type().id())
      << ",size=" << column.size() << ",offset=" << column.offset()
      << ",nullCount=" << column.null_count()
      << ",nullable=" << column.nullable() << ",data=" << column.head()
      << ",mask=" << column.null_mask()
      << ",children=" << column.num_children();
  if (depth > 0 && column.num_children() > 0) {
    out << ",childViews=[";
    for (cudf::size_type i = 0; i < column.num_children(); ++i) {
      if (i != 0) {
        out << ',';
      }
      appendColumnDescription(out, column.child(i), depth - 1);
    }
    out << ']';
  }
  out << '}';
}

std::string tableDescription(const cudf::table_view& table) {
  std::ostringstream out;
  out << "rows=" << table.num_rows() << " columns=" << table.num_columns()
      << " columnViews=[";
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << i << ':';
    appendColumnDescription(out, table.column(i), 2);
  }
  out << ']';
  return out.str();
}

template <typename T>
std::string indicesDescription(const std::vector<T>& indices) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << indices[i];
  }
  out << ']';
  return out.str();
}

void emitStreamingGroupbyDiagnostic(
    const CudfGroupby& owner,
    std::string_view event,
    std::string_view details) {
  LOG(ERROR) << "[SG_DIAG] " << streamingGroupbyOperatorContext(owner)
               << " event=" << event << (details.empty() ? "" : " ") << details;
}

// Keep all diagnostic argument construction, including table/schema strings,
// out of the normal execution path when diagnostics are disabled.
#define logStreamingGroupbyDiagnostic(owner, event, details)       \
  do {                                                             \
    if (streamingGroupbyDiagnosticsEnabled()) {                    \
      emitStreamingGroupbyDiagnostic((owner), (event), (details)); \
    }                                                              \
  } while (false)

void streamingGroupbyCudaCheckpoint(
    const CudfGroupby& owner,
    std::string_view stage,
    rmm::cuda_stream_view stream) {
  if (!streamingGroupbyDiagnosticsEnabled()) {
    return;
  }

  const auto pendingStatus = cudaPeekAtLastError();
  auto syncStatus = cudaSuccess;
  if (streamingGroupbyDiagnosticSyncEnabled()) {
    syncStatus = cudaStreamSynchronize(stream.value());
  }
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  const auto memoryStatus = cudaMemGetInfo(&freeBytes, &totalBytes);

  LOG(ERROR) << "[SG_DIAG] " << streamingGroupbyOperatorContext(owner)
               << " event=cuda_checkpoint stage=" << stage
               << " stream=" << stream.value()
               << " syncEnabled=" << streamingGroupbyDiagnosticSyncEnabled()
               << " pendingStatus=" << static_cast<int>(pendingStatus)
               << " pendingError=" << cudaGetErrorString(pendingStatus)
               << " syncStatus=" << static_cast<int>(syncStatus)
               << " syncError=" << cudaGetErrorString(syncStatus)
               << " memoryStatus=" << static_cast<int>(memoryStatus)
               << " memoryError=" << cudaGetErrorString(memoryStatus)
               << " freeBytes=" << freeBytes << " totalBytes=" << totalBytes;

  if (streamingGroupbyDiagnosticSyncEnabled() &&
      (pendingStatus != cudaSuccess || syncStatus != cudaSuccess)) {
    const auto failureStatus =
        pendingStatus != cudaSuccess ? pendingStatus : syncStatus;
    VELOX_FAIL(
        "streaming_groupby CUDA diagnostic checkpoint '{}' failed on stream "
        "{}: {} ({})",
        stage,
        reinterpret_cast<uintptr_t>(stream.value()),
        cudaGetErrorString(failureStatus),
        static_cast<int>(failureStatus));
  }
}

struct GroupbyLeafState final : public BufferedState {
  explicit GroupbyLeafState(InputChunk chunk) : chunk(std::move(chunk)) {}

  InputChunk chunk;
};

bool isStreamingGroupbyCapacityError(const std::exception& e) {
  return std::string_view{e.what()}.find("max_distinct_keys") !=
      std::string_view::npos;
}

size_t cudfSizeTypeMaxRows() {
  return static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
}

size_t streamingGroupbyApiSafeCapacity() {
  // streaming_groupby's cuco set currently uses a 0.5 load factor, hence a
  // requested capacity of N allocates approximately 2N physical slots. cuDF
  // stores offsets into those slots in signed 32-bit cudf::size_type values.
  // Keeping both the requested capacity and every merge source at or below
  // half of size_type max prevents physical slot-offset overflow as well as
  // overflow of the transient `max_distinct_keys + row_index` encoding.
  return cudfSizeTypeMaxRows() / 2;
}

size_t saturatingAdd(size_t a, size_t b) {
  if (a > std::numeric_limits<size_t>::max() - b) {
    return std::numeric_limits<size_t>::max();
  }
  return a + b;
}

size_t saturatingMultiply(size_t a, size_t b) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    return std::numeric_limits<size_t>::max();
  }
  return a * b;
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

#define DEFINE_SIMPLE_STREAMING_GROUPBY_AGGREGATOR(Name, name)                \
  struct StreamingGroupby##Name##Aggregator final                             \
      : StreamingGroupbyAggregator {                                          \
    StreamingGroupby##Name##Aggregator(                                       \
        core::AggregationNode::Step step,                                     \
        uint32_t aggregateIndex,                                              \
        uint32_t inputIndex,                                                  \
        VectorPtr constant,                                                   \
        const TypePtr& inputType,                                             \
        const TypePtr& bufferedType,                                          \
        const TypePtr& finalType)                                             \
        : StreamingGroupbyAggregator(                                         \
              step,                                                           \
              aggregateIndex,                                                 \
              inputIndex,                                                     \
              constant,                                                       \
              inputType,                                                      \
              bufferedType,                                                   \
              finalType) {}                                                   \
                                                                              \
    void addPreparedColumns(                                                  \
        std::vector<StreamingPreparedColumn>& columns) override {             \
      VELOX_CHECK(                                                            \
          constant == nullptr, #Name " does not support constant input");     \
      preparedInputIndex_ = addPreparedColumn(                                \
          columns,                                                            \
          std::nullopt,                                                       \
          inputType,                                                          \
          fmt::format("a{}_{}", aggregateIndex, #name));                      \
    }                                                                         \
                                                                              \
    void addStreamingRequest(                                                 \
        std::vector<cudf::groupby::streaming_aggregation_request>& requests)  \
        override {                                                            \
      requests.push_back(                                                     \
          cudf::groupby::streaming_aggregation_request{                       \
              static_cast<cudf::size_type>(preparedInputIndex_),              \
              cudf::make_##name##_aggregation<cudf::groupby_aggregation>()}); \
      resultIndex_ = requests.size() - 1;                                     \
    }                                                                         \
                                                                              \
    std::unique_ptr<cudf::column> makeBufferedOutputColumn(                   \
        std::vector<cudf::groupby::aggregation_result>& results,              \
        rmm::cuda_stream_view stream) override {                              \
      return castStreamingResult(                                             \
          std::move(results[resultIndex_].results[0]), bufferedType, stream); \
    }                                                                         \
                                                                              \
    std::unique_ptr<cudf::column> makeFinalOutputColumn(                      \
        cudf::column_view const& bufferedColumn,                              \
        rmm::cuda_stream_view stream) override {                              \
      return copyAndCastBufferedColumn(bufferedColumn, finalType, stream);    \
    }                                                                         \
                                                                              \
   private:                                                                   \
    column_index_t preparedInputIndex_;                                       \
    uint32_t resultIndex_;                                                    \
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
    requests.push_back(
        cudf::groupby::streaming_aggregation_request{
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
    requests.push_back(
        cudf::groupby::streaming_aggregation_request{
            static_cast<cudf::size_type>(sumInputIndex_),
            cudf::make_sum_aggregation<cudf::groupby_aggregation>()});
    sumResultIndex_ = requests.size() - 1;
    requests.push_back(
        cudf::groupby::streaming_aggregation_request{
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
        rowWidthBytes_(estimateStreamingRowWidth(owner.bufferedResultType_)),
        diagnosticId_(nextStreamingLeafDiagnosticId.fetch_add(
            1,
            std::memory_order_relaxed)) {
    logStreamingGroupbyDiagnostic(
        owner_,
        "leaf_create",
        fmt::format("leaf={} rowWidthBytes={}", diagnosticId_, rowWidthBytes_));
  }

  ~StreamingGroupbyLeafState() override {
    if (streamingGroupbyDiagnosticsEnabled()) {
      size_t distinct = 0;
      try {
        distinct = distinctKeys();
      } catch (...) {
      }
      logStreamingGroupbyDiagnostic(
          owner_,
          "leaf_destroy",
          fmt::format(
              "leaf={} chunks={} totalRows={} distinct={} capacity={} "
              "estimatedFlatSize={} stateStream={}",
              diagnosticId_,
              chunkCount_,
              totalRows_,
              distinct,
              currentCapacity_,
              estimatedFlatSize_,
              stateStream_.has_value()
                  ? reinterpret_cast<uintptr_t>(stateStream_->value())
                  : 0));
    }
  }

  void addChunk(InputChunk input) {
    if (input.empty()) {
      return;
    }

    auto const stream = prepareInput(input.stream);
    auto const inputRows = input.size();
    auto const inputFlatSize =
        input.owner ? input.owner->estimateFlatSize() : 0;
    auto const chunk = ++chunkCount_;
    auto const distinctBefore = distinctKeys();
    auto const rebuildRequired = groupby_ &&
        (saturatingAdd(distinctBefore, inputRows) > currentCapacity_ ||
         currentCapacity_ > maxCapacityForBatch(inputRows));

    logStreamingGroupbyDiagnostic(
        owner_,
        "add_chunk_begin",
        fmt::format(
            "leaf={} chunk={} inputRows={} totalRowsBefore={} "
            "distinctBefore={} capacity={} first={} rebuildRequired={} "
            "inputStream={} stateStream={} inputFlatSize={} table={}",
            diagnosticId_,
            chunk,
            inputRows,
            totalRows_,
            distinctBefore,
            currentCapacity_,
            groupby_ == nullptr,
            rebuildRequired,
            reinterpret_cast<uintptr_t>(input.stream.value()),
            reinterpret_cast<uintptr_t>(stream.value()),
            inputFlatSize,
            tableDescription(input.view)));
    streamingGroupbyCudaCheckpoint(owner_, "before_add_chunk", stream);

    try {
      if (!groupby_) {
        currentCapacity_ = initialCapacity(inputRows);
        logStreamingGroupbyDiagnostic(
            owner_,
            "initial_state_begin",
            fmt::format(
                "leaf={} chunk={} inputRows={} capacity={} maxAllowed={}",
                diagnosticId_,
                chunk,
                inputRows,
                currentCapacity_,
                maxAllowedCapacity(inputRows)));
        groupby_ = owner_.createStreamingGroupby(currentCapacity_);
        streamingGroupbyCudaCheckpoint(
            owner_, "after_initial_state_create", stream);
        groupby_->aggregate(input.view, stream);
        streamingGroupbyCudaCheckpoint(
            owner_, "after_initial_state_aggregate", stream);
      } else if (rebuildRequired) {
        rebuildAndAggregate(input, stream);
      } else {
        groupby_->aggregate(input.view, stream);
        streamingGroupbyCudaCheckpoint(
            owner_, "after_subsequent_aggregate", stream);
      }
    } catch (const std::exception& e) {
      logStreamingGroupbyDiagnostic(
          owner_,
          "add_chunk_exception",
          fmt::format(
              "leaf={} chunk={} inputRows={} totalRowsBefore={} "
              "distinctBefore={} capacity={} what={}",
              diagnosticId_,
              chunk,
              inputRows,
              totalRows_,
              distinctBefore,
              currentCapacity_,
              e.what()));
      // aggregate() and merge() can throw after launching asynchronous work.
      // Preserve the original exception while still trying to keep input
      // deallocation behind that work.
      try {
        orderInputDeallocationAfterState(input.stream);
      } catch (const std::exception& cleanupError) {
        LOG(ERROR) << "Failed to order streaming_groupby input deallocation: "
                   << cleanupError.what();
      } catch (...) {
        LOG(ERROR) << "Failed to order streaming_groupby input deallocation";
      }
      throw;
    } catch (...) {
      logStreamingGroupbyDiagnostic(
          owner_,
          "add_chunk_unknown_exception",
          fmt::format(
              "leaf={} chunk={} inputRows={} totalRowsBefore={} "
              "distinctBefore={} capacity={}",
              diagnosticId_,
              chunk,
              inputRows,
              totalRows_,
              distinctBefore,
              currentCapacity_));
      // aggregate() and merge() can throw after launching asynchronous work.
      // Preserve the original exception while still trying to keep input
      // deallocation behind that work.
      try {
        orderInputDeallocationAfterState(input.stream);
      } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to order streaming_groupby input deallocation: "
                   << e.what();
      } catch (...) {
        LOG(ERROR) << "Failed to order streaming_groupby input deallocation";
      }
      throw;
    }
    orderInputDeallocationAfterState(input.stream);

    totalRows_ += inputRows;
    estimatedFlatSize_ = std::max<uint64_t>(
        estimatedFlatSize_ + inputFlatSize, currentCapacity_ * rowWidthBytes_);
    logStreamingGroupbyDiagnostic(
        owner_,
        "add_chunk_end",
        fmt::format(
            "leaf={} chunk={} inputRows={} totalRowsAfter={} distinctAfter={} "
            "capacity={} estimatedFlatSize={}",
            diagnosticId_,
            chunk,
            inputRows,
            totalRows_,
            distinctKeys(),
            currentCapacity_,
            estimatedFlatSize_));
  }

  size_t totalRows() const {
    return totalRows_;
  }

  uint64_t estimatedFlatSize() const {
    return estimatedFlatSize_;
  }

  uint64_t diagnosticId() const {
    return diagnosticId_;
  }

  size_t capacity() const {
    return currentCapacity_;
  }

  size_t diagnosticDistinctKeys() const {
    return distinctKeys();
  }

  CudfVectorPtr finalizeBuffered() const {
    if (!groupby_) {
      return nullptr;
    }
    VELOX_CHECK(stateStream_.has_value());
    logStreamingGroupbyDiagnostic(
        owner_,
        "finalize_buffered_begin",
        fmt::format(
            "leaf={} chunks={} totalRows={} distinct={} capacity={} "
            "estimatedFlatSize={} stateStream={}",
            diagnosticId_,
            chunkCount_,
            totalRows_,
            distinctKeys(),
            currentCapacity_,
            estimatedFlatSize_,
            reinterpret_cast<uintptr_t>(stateStream_->value())));
    streamingGroupbyCudaCheckpoint(
        owner_, "before_finalize_buffered", *stateStream_);
    auto output =
        owner_.materializeStreamingBufferedOutput(*groupby_, *stateStream_);
    streamingGroupbyCudaCheckpoint(
        owner_, "after_finalize_buffered", *stateStream_);
    logStreamingGroupbyDiagnostic(
        owner_,
        "finalize_buffered_end",
        fmt::format(
            "leaf={} outputRows={} output={}",
            diagnosticId_,
            output ? output->size() : 0,
            output ? tableDescription(output->getTableView()) : "null"));
    return output;
  }

 private:
  CudaEvent& stateEvent() {
    if (!stateEvent_) {
      stateEvent_ = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    }
    return *stateEvent_;
  }

  rmm::cuda_stream_view prepareInput(rmm::cuda_stream_view inputStream) {
    if (!stateStream_.has_value()) {
      stateStream_ = inputStream;
    } else if (stateStream_->value() != inputStream.value()) {
      // Keep every operation and allocation owned by streaming_groupby on one
      // stream for the lifetime of the leaf. This avoids depending on all of
      // cuDF's internal persistent allocations correctly retaining and using
      // their individual creation streams.
      stateEvent().recordFrom(inputStream).waitOn(*stateStream_);
    }
    return *stateStream_;
  }

  void orderInputDeallocationAfterState(rmm::cuda_stream_view inputStream) {
    VELOX_CHECK(stateStream_.has_value());
    if (stateStream_->value() != inputStream.value()) {
      // InputChunk destruction frees its buffers on inputStream. aggregate()
      // reads those buffers asynchronously on stateStream_, so make the input
      // stream wait before addChunk() releases the owner.
      stateEvent().recordFrom(*stateStream_).waitOn(inputStream);
    }
  }

  size_t distinctKeys() const {
    if (!groupby_) {
      return 0;
    }
    auto keys = groupby_->distinct_keys();
    VELOX_CHECK_GE(keys, 0);
    return static_cast<size_t>(keys);
  }

  size_t maxCapacityForBatch(size_t inputRows) const {
    auto const cudfMaxRows = cudfSizeTypeMaxRows();
    VELOX_CHECK_LE(
        inputRows,
        cudfMaxRows,
        "streaming_groupby input batch of {} rows exceeds cudf::size_type "
        "limit of {} rows",
        inputRows,
        cudfMaxRows);
    VELOX_CHECK_LE(
        inputRows,
        cudfMaxRows / 2,
        "streaming_groupby cannot safely aggregate a batch of {} rows: cuDF "
        "requires max_distinct_keys to be at least the batch size, and "
        "max_distinct_keys + batch_size must not exceed cudf::size_type max "
        "({}). Reduce the upstream GPU batch size.",
        inputRows,
        cudfMaxRows);
    return cudfMaxRows - inputRows;
  }

  size_t maxAllowedCapacity(size_t inputRows) const {
    return std::min(owner_.maxBufferedRows_, maxCapacityForBatch(inputRows));
  }

  size_t initialCapacity(size_t inputRows) const {
    auto const maxAllowed = maxAllowedCapacity(inputRows);
    auto const requested = std::max<size_t>(
        std::max<size_t>(inputRows, 1), saturatingMultiply(inputRows, 4));
    auto const capacity = std::min(requested, maxAllowed);
    VELOX_CHECK_GE(
        capacity,
        inputRows,
        "streaming_groupby input batch has {} rows, exceeding the configured "
        "capacity ceiling of {} rows",
        inputRows,
        maxAllowed);
    return capacity;
  }

  size_t rebuildCapacity(size_t inputRows) const {
    auto const currentDistinctKeys = distinctKeys();
    auto const maxAllowed = maxAllowedCapacity(inputRows);

    VELOX_CHECK_GE(
        maxAllowed,
        inputRows,
        "streaming_groupby input batch has {} rows, exceeding the configured "
        "capacity ceiling of {} rows",
        inputRows,
        maxAllowed);
    VELOX_CHECK_GE(
        maxAllowed,
        currentDistinctKeys,
        "streaming_groupby cannot aggregate a batch of {} rows with {} "
        "existing distinct keys without exceeding cudf::size_type limits. "
        "Reduce the upstream GPU batch size.",
        inputRows,
        currentDistinctKeys);

    auto const requiredWorstCase =
        saturatingAdd(currentDistinctKeys, inputRows);
    auto const growth = saturatingAdd(
        currentCapacity_, std::max(currentCapacity_ / 2, inputRows));
    return std::min(std::max(requiredWorstCase, growth), maxAllowed);
  }

  size_t nextRebuildCapacity(size_t currentAttempt, size_t inputRows) const {
    auto const maxAllowed = maxAllowedCapacity(inputRows);
    if (currentAttempt >= maxAllowed) {
      return currentAttempt;
    }
    return std::min(
        maxAllowed,
        saturatingAdd(currentAttempt, std::max<size_t>(currentAttempt / 2, 1)));
  }

  void rebuildAndAggregate(
      const InputChunk& input,
      rmm::cuda_stream_view stream) {
    auto capacity = rebuildCapacity(input.size());
    auto const oldDistinct = distinctKeys();
    auto const oldCapacity = currentCapacity_;
    auto attempt = uint32_t{0};
    for (;;) {
      ++attempt;
      logStreamingGroupbyDiagnostic(
          owner_,
          "rebuild_attempt_begin",
          fmt::format(
              "leaf={} attempt={} inputRows={} oldDistinct={} oldCapacity={} "
              "requestedCapacity={} maxAllowed={} requiredWorstCase={}",
              diagnosticId_,
              attempt,
              input.size(),
              oldDistinct,
              oldCapacity,
              capacity,
              maxAllowedCapacity(input.size()),
              saturatingAdd(oldDistinct, input.size())));
      auto rebuilt = owner_.createStreamingGroupby(capacity);
      streamingGroupbyCudaCheckpoint(owner_, "after_rebuild_create", stream);
      try {
        rebuilt->aggregate(input.view, stream);
        streamingGroupbyCudaCheckpoint(
            owner_, "after_rebuild_new_input_aggregate", stream);
        const auto newInputDistinct = rebuilt->distinct_keys();
        logStreamingGroupbyDiagnostic(
            owner_,
            "rebuild_before_merge",
            fmt::format(
                "leaf={} attempt={} newInputDistinct={} oldDistinct={} "
                "sumDistinctUpperBound={} requestedCapacity={}",
                diagnosticId_,
                attempt,
                newInputDistinct,
                oldDistinct,
                saturatingAdd(
                    static_cast<size_t>(newInputDistinct), oldDistinct),
                capacity));
        if (groupby_) {
          rebuilt->merge(*groupby_, stream);
          streamingGroupbyCudaCheckpoint(
              owner_, "after_rebuild_old_state_merge", stream);
        }

        // All old and new state was allocated and consumed on stream, so
        // move-assignment queues old-state deallocation behind merge().
        groupby_ = std::move(rebuilt);
        currentCapacity_ = capacity;
        logStreamingGroupbyDiagnostic(
            owner_,
            "rebuild_attempt_success",
            fmt::format(
                "leaf={} attempt={} capacity={} distinctAfter={}",
                diagnosticId_,
                attempt,
                currentCapacity_,
                distinctKeys()));
        {
          auto lockedStats = owner_.stats_.wlock();
          lockedStats->addRuntimeStat(
              std::string{kStreamingGroupbyApiRebuildsStat}, RuntimeCounter(1));
        }
        return;
      } catch (const std::exception& e) {
        logStreamingGroupbyDiagnostic(
            owner_,
            "rebuild_attempt_exception",
            fmt::format(
                "leaf={} attempt={} requestedCapacity={} inputRows={} "
                "oldDistinct={} what={}",
                diagnosticId_,
                attempt,
                capacity,
                input.size(),
                oldDistinct,
                e.what()));
        if (!isStreamingGroupbyCapacityError(e)) {
          throw;
        }
        auto const nextCapacity = nextRebuildCapacity(capacity, input.size());
        if (nextCapacity == capacity) {
          break;
        }
        capacity = nextCapacity;
      }
    }

    VELOX_FAIL(
        "streaming_groupby reached the capacity ceiling of {} rows",
        owner_.maxBufferedRows_);
  }

  CudfGroupby& owner_;
  const uint64_t rowWidthBytes_;
  std::unique_ptr<CudaEvent> stateEvent_;
  std::unique_ptr<cudf::groupby::streaming_groupby> groupby_;
  size_t currentCapacity_{0};
  size_t totalRows_{0};
  uint64_t estimatedFlatSize_{0};
  std::optional<rmm::cuda_stream_view> stateStream_;
  const uint64_t diagnosticId_;
  uint64_t chunkCount_{0};
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
      const InputChunk& input,
      const PartitionSpec& spec) override {
    if (input.empty()) {
      return std::vector<InputChunk>(spec.numPartitions);
    }

    std::vector<rmm::cuda_stream_view> inputStreams{input.stream};
    cudf::detail::join_streams(inputStreams, input.stream);

    auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
        input.view,
        spec.keyIndices,
        spec.numPartitions,
        spec.hashId,
        spec.seed,
        input.stream,
        get_output_mr());

    VELOX_CHECK_EQ(partitionOffsets.size(), spec.numPartitions + 1);
    VELOX_CHECK_EQ(partitionOffsets.front(), 0);

    partitionOffsets.erase(partitionOffsets.begin());
    partitionOffsets.pop_back();

    auto partitionedTableOwner =
        std::shared_ptr<cudf::table>(std::move(partitionedTable));
    auto partitionViews = cudf::split(
        partitionedTableOwner->view(), partitionOffsets, input.stream);
    std::vector<InputChunk> chunks(spec.numPartitions);
    for (int32_t i = 0; i < spec.numPartitions; ++i) {
      auto partition = partitionViews[i];
      if (partition.num_rows() > 0) {
        chunks[i] = makeBorrowedChunk(
            input.pool,
            input.type,
            partition,
            input.stream,
            partitionedTableOwner);
      }
    }

    CudaEvent event(cudaEventDisableTiming);
    streamsWaitForStream(event, inputStreams, input.stream);
    return chunks;
  }

  std::vector<std::unique_ptr<BufferedState>> repartitionLeaf(
      const BufferedState& leaf,
      const PartitionSpec& spec) override {
    auto partitions = partitionInput(asLeafState(leaf).chunk, spec);

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

  InputChunk makeBorrowedChunk(
      memory::MemoryPool* pool,
      const TypePtr& type,
      cudf::table_view view,
      rmm::cuda_stream_view stream,
      std::shared_ptr<cudf::table> tableOwner) const {
    return InputChunk{pool, type, view, stream, nullptr, std::move(tableOwner)};
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
  explicit StreamingGroupbyBufferedStateOps(CudfGroupby& owner)
      : owner_(owner) {
    keyIndices_.reserve(owner_.groupingKeyOutputChannels_.size());
    for (auto keyIndex : owner_.groupingKeyOutputChannels_) {
      keyIndices_.push_back(static_cast<cudf::size_type>(keyIndex));
    }
  }

  InputChunk prepareInput(CudfVectorPtr rawInput) override {
    VELOX_CHECK_NOT_NULL(rawInput);
    auto rawView = rawInput->getTableView();
    logStreamingGroupbyDiagnostic(
        owner_,
        "prepare_input_begin",
        fmt::format(
            "rawRows={} rawStream={} rawType={} rawTable={}",
            rawInput->size(),
            reinterpret_cast<uintptr_t>(rawInput->stream().value()),
            rawInput->type()->toString(),
            tableDescription(rawView)));
    auto preparedView = owner_.makeStreamingPreparedInputView(rawView);
    VELOX_CHECK_EQ(
        preparedView.num_rows(),
        rawView.num_rows(),
        "streaming_groupby prepared input row count changed");
    VELOX_CHECK_EQ(
        preparedView.num_columns(),
        owner_.streamingPreparedColumns_.size(),
        "streaming_groupby prepared input column count mismatch");
    logStreamingGroupbyDiagnostic(
        owner_,
        "prepare_input_end",
        fmt::format(
            "preparedType={} preparedTable={}",
            owner_.streamingPreparedType_->toString(),
            tableDescription(preparedView)));
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
      const InputChunk& input,
      const PartitionSpec& spec) override {
    if (input.empty()) {
      return std::vector<InputChunk>(spec.numPartitions);
    }

    logStreamingGroupbyDiagnostic(
        owner_,
        "partition_input_begin",
        fmt::format(
            "inputRows={} inputColumns={} stream={} partitions={} seed={} "
            "keyIndices={} table={}",
            input.size(),
            input.view.num_columns(),
            reinterpret_cast<uintptr_t>(input.stream.value()),
            spec.numPartitions,
            spec.seed,
            indicesDescription(spec.keyIndices),
            tableDescription(input.view)));
    streamingGroupbyCudaCheckpoint(
        owner_, "before_partition_input", input.stream);
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
    streamingGroupbyCudaCheckpoint(
        owner_, "after_partition_input", input.stream);

    std::vector<InputChunk> chunks(spec.numPartitions);
    std::ostringstream partitionRows;
    partitionRows << '[';
    for (int32_t i = 0; i < spec.numPartitions; ++i) {
      if (i != 0) {
        partitionRows << ',';
      }
      partitionRows << (partitions[i] ? partitions[i]->size() : 0);
      if (partitions[i]) {
        chunks[i] = makeOwnedChunk(
            std::move(partitions[i]), owner_.streamingPreparedType_);
      }
    }
    partitionRows << ']';
    logStreamingGroupbyDiagnostic(
        owner_,
        "partition_input_end",
        fmt::format(
            "inputRows={} partitions={} seed={} partitionRows={}",
            input.size(),
            spec.numPartitions,
            spec.seed,
            partitionRows.str()));
    return chunks;
  }

  std::vector<std::unique_ptr<BufferedState>> repartitionLeaf(
      const BufferedState& leaf,
      const PartitionSpec& spec) override {
    auto const& streamingLeaf = asLeafState(leaf);
    {
      auto lockedStats = owner_.stats_.wlock();
      lockedStats->addRuntimeStat(
          std::string{kStreamingGroupbyApiRepartitionsStat}, RuntimeCounter(1));
    }
    logStreamingGroupbyDiagnostic(
        owner_,
        "repartition_leaf_begin",
        fmt::format(
            "leaf={} totalRows={} distinct={} capacity={} partitions={} "
            "seed={} keyIndices={}",
            streamingLeaf.diagnosticId(),
            streamingLeaf.totalRows(),
            streamingLeaf.diagnosticDistinctKeys(),
            streamingLeaf.capacity(),
            spec.numPartitions,
            spec.seed,
            indicesDescription(spec.keyIndices)));
    auto buffered = streamingLeaf.finalizeBuffered();
    if (!buffered) {
      logStreamingGroupbyDiagnostic(
          owner_,
          "repartition_leaf_empty",
          fmt::format("leaf={}", streamingLeaf.diagnosticId()));
      return std::vector<std::unique_ptr<BufferedState>>(spec.numPartitions);
    }

    InputChunk bufferedChunk{
        buffered->pool(),
        owner_.bufferedResultType_,
        buffered->getTableView(),
        buffered->stream(),
        std::move(buffered)};

    logStreamingGroupbyDiagnostic(
        owner_,
        "repartition_buffered_begin",
        fmt::format(
            "leaf={} bufferedRows={} stream={} table={}",
            streamingLeaf.diagnosticId(),
            bufferedChunk.size(),
            reinterpret_cast<uintptr_t>(bufferedChunk.stream.value()),
            tableDescription(bufferedChunk.view)));
    streamingGroupbyCudaCheckpoint(
        owner_, "before_repartition_buffered", bufferedChunk.stream);
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
    streamingGroupbyCudaCheckpoint(
        owner_, "after_repartition_buffered", bufferedChunk.stream);

    std::vector<std::unique_ptr<BufferedState>> leaves(spec.numPartitions);
    std::ostringstream partitionRows;
    partitionRows << '[';
    for (int32_t i = 0; i < spec.numPartitions; ++i) {
      if (i != 0) {
        partitionRows << ',';
      }
      partitionRows << (partitions[i] ? partitions[i]->size() : 0);
      if (!partitions[i]) {
        continue;
      }
      auto prepared = prepareOwnedChunk(std::move(partitions[i]));
      if (!prepared.empty()) {
        leaves[i] = createLeaf(std::move(prepared));
      }
    }
    partitionRows << ']';
    logStreamingGroupbyDiagnostic(
        owner_,
        "repartition_leaf_end",
        fmt::format(
            "leaf={} bufferedRows={} partitions={} seed={} partitionRows={}",
            streamingLeaf.diagnosticId(),
            bufferedChunk.size(),
            spec.numPartitions,
            spec.seed,
            partitionRows.str()));
    return leaves;
  }

  CudfVectorPtr finalizeLeaf(std::unique_ptr<BufferedState> leaf) override {
    auto streamingLeaf = std::unique_ptr<StreamingGroupbyLeafState>(
        static_cast<StreamingGroupbyLeafState*>(leaf.release()));
    auto const leafId = streamingLeaf->diagnosticId();
    logStreamingGroupbyDiagnostic(
        owner_,
        "finalize_leaf_begin",
        fmt::format(
            "leaf={} totalRows={} distinct={} capacity={}",
            leafId,
            streamingLeaf->totalRows(),
            streamingLeaf->diagnosticDistinctKeys(),
            streamingLeaf->capacity()));
    auto buffered = streamingLeaf->finalizeBuffered();
    if (owner_.isPartialOutput_) {
      logStreamingGroupbyDiagnostic(
          owner_,
          "finalize_leaf_end",
          fmt::format(
              "leaf={} partial=true outputRows={}",
              leafId,
              buffered ? buffered->size() : 0));
      return buffered;
    }
    auto output = owner_.finalizeStreamingBufferedOutput(std::move(buffered));
    logStreamingGroupbyDiagnostic(
        owner_,
        "finalize_leaf_end",
        fmt::format(
            "leaf={} partial=false outputRows={} output={}",
            leafId,
            output ? output->size() : 0,
            output ? tableDescription(output->getTableView()) : "null"));
    return output;
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
  if (!config.streamingGroupbyApiEnabled || !streamingEnabled_ ||
      isSingleStep_ || aggregationNode_->groupingKeys().empty() ||
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
  for (size_t preparedIndex = 0;
       preparedIndex < streamingPreparedColumns_.size();
       ++preparedIndex) {
    const auto& column = streamingPreparedColumns_[preparedIndex];
    VELOX_CHECK_LT(
        column.inputIndex,
        rawInputView.num_columns(),
        "streaming_groupby prepared column {} refers to input column {}, but "
        "the input only has {} columns",
        preparedIndex,
        column.inputIndex,
        rawInputView.num_columns());
    auto view = rawInputView.column(column.inputIndex);
    if (column.childIndex.has_value()) {
      VELOX_CHECK_LT(
          *column.childIndex,
          view.num_children(),
          "streaming_groupby prepared column {} refers to child {} of input "
          "column {}, but it only has {} children",
          preparedIndex,
          *column.childIndex,
          column.inputIndex,
          view.num_children());
      view = view.child(*column.childIndex);
    }
    VELOX_CHECK_EQ(
        view.size(),
        rawInputView.num_rows(),
        "streaming_groupby prepared column {} has {} rows, expected {}. "
        "inputColumn={}, childIndex={}",
        preparedIndex,
        view.size(),
        rawInputView.num_rows(),
        column.inputIndex,
        column.childIndex.has_value() ? static_cast<int64_t>(*column.childIndex)
                                      : int64_t{-1});
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
  logStreamingGroupbyDiagnostic(
      *this,
      "cudf_finalize_begin",
      fmt::format(
          "distinct={} stream={}",
          groupby.distinct_keys(),
          reinterpret_cast<uintptr_t>(stream.value())));
  auto [groupKeys, results] = groupby.finalize(stream, get_output_mr());
  streamingGroupbyCudaCheckpoint(*this, "after_cudf_finalize", stream);
  logStreamingGroupbyDiagnostic(
      *this,
      "cudf_finalize_result",
      fmt::format(
          "groupKeyRows={} groupKeyColumns={} resultRequests={}",
          groupKeys->num_rows(),
          groupKeys->num_columns(),
          results.size()));
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
  streamingGroupbyCudaCheckpoint(
      *this, "after_buffered_result_columns", stream);

  auto resultTable = std::make_unique<cudf::table>(std::move(resultColumns));
  auto numRows = resultTable->num_rows();
  if (numRows == 0) {
    return nullptr;
  }

  auto outputType = isPartialOutput_ ? outputType_ : bufferedResultType_;
  auto output = std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType, numRows, std::move(resultTable), stream);
  logStreamingGroupbyDiagnostic(
      *this,
      "cudf_finalize_end",
      fmt::format(
          "outputRows={} outputType={} output={}",
          output->size(),
          outputType->toString(),
          tableDescription(output->getTableView())));
  return output;
}

CudfVectorPtr CudfGroupby::finalizeStreamingBufferedOutput(
    CudfVectorPtr bufferedOutput) const {
  if (!bufferedOutput) {
    return nullptr;
  }

  auto stream = bufferedOutput->stream();
  auto bufferedView = bufferedOutput->getTableView();
  logStreamingGroupbyDiagnostic(
      *this,
      "final_output_conversion_begin",
      fmt::format(
          "bufferedType={} outputType={} stream={} buffered={}",
          bufferedOutput->type()->toString(),
          outputType_->toString(),
          reinterpret_cast<uintptr_t>(stream.value()),
          tableDescription(bufferedView)));
  streamingGroupbyCudaCheckpoint(
      *this, "before_final_output_conversion", stream);
  std::vector<std::unique_ptr<cudf::column>> outputColumns;
  outputColumns.reserve(outputType_->size());

  for (size_t i = 0; i < groupingKeyOutputChannels_.size(); ++i) {
    outputColumns.push_back(
        std::make_unique<cudf::column>(
            bufferedView.column(i), stream, get_output_mr()));
  }

  for (size_t i = 0; i < streamingGroupbyAggregators_.size(); ++i) {
    auto bufferedIndex = groupingKeyOutputChannels_.size() + i;
    outputColumns.push_back(
        streamingGroupbyAggregators_[i]->makeFinalOutputColumn(
            bufferedView.column(bufferedIndex), stream));
  }
  streamingGroupbyCudaCheckpoint(*this, "after_final_output_columns", stream);

  auto resultTable = std::make_unique<cudf::table>(std::move(outputColumns));
  auto numRows = resultTable->num_rows();
  if (numRows == 0) {
    return nullptr;
  }

  auto output = std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, numRows, std::move(resultTable), stream);
  logStreamingGroupbyDiagnostic(
      *this,
      "final_output_conversion_end",
      fmt::format(
          "outputRows={} output={}",
          output->size(),
          tableDescription(output->getTableView())));
  return output;
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
      // This limit is also passed to PartitionedBufferedState below. Once a
      // leaf reaches the safe streaming_groupby capacity, PBS repartitions it
      // before another batch is added instead of attempting an unsafe rebuild.
      maxBufferedRows_ =
          std::min(maxBufferedRows_, streamingGroupbyApiSafeCapacity());
    }

    if (streamingGroupbyApiEnabled_ && !isPartialOutput_) {
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
        lockedStats->addRuntimeStat(
            std::string{kStreamingGroupbyApiCapacityLimitStat},
            RuntimeCounter(maxBufferedRows_));
      }
    }

    if (isFinalOrSingle) {
      if (streamingGroupbyApiEnabled_) {
        partitionedBufferedState_ = std::make_unique<PartitionedBufferedState>(
            std::make_unique<StreamingGroupbyBufferedStateOps>(*this),
            maxBufferedRows_);
      } else {
        partitionedBufferedState_ = std::make_unique<PartitionedBufferedState>(
            std::make_unique<BufferedGroupbyStateOps>(*this), maxBufferedRows_);
      }
    } else if (isPartialOutput_) {
      if (!streamingGroupbyApiEnabled_) {
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

  if (streamingGroupbyDiagnosticsEnabled()) {
    std::ostringstream preparedColumns;
    preparedColumns << '[';
    for (size_t i = 0; i < streamingPreparedColumns_.size(); ++i) {
      if (i != 0) {
        preparedColumns << ',';
      }
      const auto& column = streamingPreparedColumns_[i];
      preparedColumns << i << ":name=" << column.name
                      << "/input=" << column.inputIndex << "/child=";
      if (column.childIndex.has_value()) {
        preparedColumns << *column.childIndex;
      } else {
        preparedColumns << "none";
      }
      preparedColumns << "/type=" << column.type->toString();
    }
    preparedColumns << ']';

    logStreamingGroupbyDiagnostic(
        *this,
        "initialize",
        fmt::format(
            "streamingEnabled={} streamingApiEnabled={} partialOutput={} "
            "singleStep={} ignoreNullKeys={} maxBufferedRows={} "
            "cudfSizeTypeMax={} pbsState={} diagnosticSync={} inputType={} "
            "outputType={} bufferedType={} preparedType={} groupingKeyInputs={} "
            "groupingKeyOutputs={} aggregationInputs={} preparedColumns={}",
            streamingEnabled_,
            streamingGroupbyApiEnabled_,
            isPartialOutput_,
            isSingleStep_,
            ignoreNullKeys_,
            maxBufferedRows_,
            cudfSizeTypeMaxRows(),
            partitionedBufferedState_
                ? partitionedBufferedState_->diagnosticId()
                : 0,
            streamingGroupbyDiagnosticSyncEnabled(),
            inputType_->toString(),
            outputType_->toString(),
            bufferedResultType_ ? bufferedResultType_->toString() : "null",
            streamingPreparedType_ ? streamingPreparedType_->toString()
                                   : "null",
            indicesDescription(groupingKeyInputChannels_),
            indicesDescription(groupingKeyOutputChannels_),
            indicesDescription(aggregationInputChannels_),
            preparedColumns.str()));
  }

  aggregationNode_.reset();
}

void CudfGroupby::computePartialGroupbyStreaming(CudfVectorPtr tbl) {
  if (!streamingGroupbyApiEnabled_) {
    logStreamingGroupbyDiagnostic(
        *this,
        "partial_buffer_input",
        fmt::format("inputRows={} apiEnabled=false", tbl->size()));
    flushableBufferedState_->addInput(std::move(tbl));
    return;
  }

  auto const inputRows = tbl->size();
  auto stream = tbl->stream();
  auto permutedInputView = tbl->getTableView().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  logStreamingGroupbyDiagnostic(
      *this,
      "partial_groupby_begin",
      fmt::format(
          "inputRows={} stream={} permuted={}",
          inputRows,
          reinterpret_cast<uintptr_t>(stream.value()),
          tableDescription(permutedInputView)));
  auto output = doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      aggregators_,
      outputType_,
      stream);
  if (output) {
    logStreamingGroupbyDiagnostic(
        *this,
        "partial_groupby_buffered_output",
        fmt::format(
            "inputRows={} outputRows={} table={}",
            inputRows,
            output->size(),
            tableDescription(output->getTableView())));
    pendingPartialOutputs_.emplace_back(std::move(output), inputRows);
  } else {
    logStreamingGroupbyDiagnostic(
        *this,
        "partial_groupby_empty_output",
        fmt::format("inputRows={}", inputRows));
  }
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

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  const auto batch = ++diagnosticInputBatchCount_;
  logStreamingGroupbyDiagnostic(
      *this,
      "operator_add_input_begin",
      fmt::format(
          "batch={} rows={} stream={} streamingEnabled={} apiEnabled={} "
          "partialOutput={} singleStep={} type={} table={}",
          batch,
          cudfInput->size(),
          reinterpret_cast<uintptr_t>(cudfInput->stream().value()),
          streamingEnabled_,
          streamingGroupbyApiEnabled_,
          isPartialOutput_,
          isSingleStep_,
          cudfInput->type()->toString(),
          tableDescription(cudfInput->getTableView())));
  streamingGroupbyCudaCheckpoint(
      *this, "operator_add_input_entry", cudfInput->stream());

  if (streamingEnabled_) {
    if (isPartialOutput_) {
      if (!streamingGroupbyApiEnabled_) {
        numInputRows_ += input->size();
      }
      computePartialGroupbyStreaming(cudfInput);
      streamingGroupbyCudaCheckpoint(
          *this, "operator_add_input_after_partial", cudfInput->stream());
      logStreamingGroupbyDiagnostic(
          *this,
          "operator_add_input_end",
          fmt::format("batch={} path=partial", batch));
      return;
    } else if (isSingleStep_) {
      auto stream = cudfInput->stream();
      computeSingleGroupbyStreaming(cudfInput);
      streamingGroupbyCudaCheckpoint(
          *this, "operator_add_input_after_single", stream);
      logStreamingGroupbyDiagnostic(
          *this,
          "operator_add_input_end",
          fmt::format("batch={} path=single", batch));
      return;
    } else {
      auto stream = cudfInput->stream();
      computeFinalGroupbyStreaming(cudfInput);
      streamingGroupbyCudaCheckpoint(
          *this, "operator_add_input_after_final", stream);
      logStreamingGroupbyDiagnostic(
          *this,
          "operator_add_input_end",
          fmt::format("batch={} path=final", batch));
      return;
    }
  }

  // Handle non-streaming cases.
  if (isPartialOutput_) {
    numInputRows_ += input->size();
  }
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

  logStreamingGroupbyDiagnostic(
      *this,
      "legacy_groupby_begin",
      fmt::format(
          "rows={} columns={} groupByKeys={} aggregators={} stream={} table={}",
          tableView.num_rows(),
          tableView.num_columns(),
          indicesDescription(groupByKeys),
          aggregators.size(),
          reinterpret_cast<uintptr_t>(stream.value()),
          tableDescription(tableView)));
  streamingGroupbyCudaCheckpoint(*this, "before_legacy_groupby", stream);

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
  streamingGroupbyCudaCheckpoint(*this, "after_legacy_groupby", stream);
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
    logStreamingGroupbyDiagnostic(*this, "legacy_groupby_end", "outputRows=0");
    return nullptr;
  }

  auto output = std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType, numRows, std::move(resultTable), stream);
  logStreamingGroupbyDiagnostic(
      *this,
      "legacy_groupby_end",
      fmt::format(
          "outputRows={} outputType={} output={}",
          output->size(),
          outputType->toString(),
          tableDescription(output->getTableView())));
  return output;
}

CudfVectorPtr CudfGroupby::releasePartialOutput(
    CudfVectorPtr output,
    int64_t inputRows) {
  auto numOutputRows = output->size();
  const auto batch = ++diagnosticPartialOutputBatchCount_;
  const double aggregationPct =
      inputRows == 0 ? 0 : (numOutputRows * 1.0) / inputRows * 100;
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
  logStreamingGroupbyDiagnostic(
      *this,
      "partial_output_release",
      fmt::format(
          "outputBatch={} inputRows={} outputRows={} aggregationPct={} "
          "stream={} table={}",
          batch,
          inputRows,
          numOutputRows,
          aggregationPct,
          reinterpret_cast<uintptr_t>(output->stream().value()),
          tableDescription(output->getTableView())));
  return output;
}

RowVectorPtr CudfGroupby::doGetOutput() {
  // Handle partial streaming groupby.
  if (isPartialOutput_ && streamingEnabled_) {
    if (streamingGroupbyApiEnabled_) {
      if (!pendingPartialOutputs_.empty()) {
        auto [output, inputRows] = std::move(pendingPartialOutputs_.front());
        pendingPartialOutputs_.pop_front();
        return releasePartialOutput(std::move(output), inputRows);
      }

      if (noMoreInput_) {
        finished_ = true;
      }
      return nullptr;
    }

    if (!flushableBufferedState_) {
      return nullptr;
    }

    if (auto output = flushableBufferedState_->getOutput(noMoreInput_)) {
      auto released = releasePartialOutput(std::move(output), numInputRows_);
      numInputRows_ = 0;
      return released;
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
  logStreamingGroupbyDiagnostic(
      *this,
      "no_more_input",
      fmt::format(
          "inputBatches={} partialOutputBatches={} streamingEnabled={} "
          "apiEnabled={}",
          diagnosticInputBatchCount_,
          diagnosticPartialOutputBatchCount_,
          streamingEnabled_,
          streamingGroupbyApiEnabled_));
  Operator::noMoreInput();
  if (isPartialOutput_ && !streamingEnabled_ && inputs_.empty()) {
    finished_ = true;
  }
}

bool CudfGroupby::isFinished() {
  return finished_;
}

#undef logStreamingGroupbyDiagnostic

} // namespace facebook::velox::cudf_velox
