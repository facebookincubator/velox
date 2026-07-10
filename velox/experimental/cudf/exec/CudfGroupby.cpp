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
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
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
#include <cudf/partitioning.hpp>
#include <cudf/reduction.hpp>
#include <cudf/unary.hpp>

#include <limits>

namespace {

using namespace facebook::velox;
using cudf_velox::castDecimal64InputToDecimal128;
using cudf_velox::CountInputKind;
using cudf_velox::finalizeDecimalAverage;
using cudf_velox::get_output_mr;
using cudf_velox::get_temp_mr;
using cudf_velox::GroupbyAggregator;
using cudf_velox::ResolvedAggregateInfo;
using cudf_velox::serializeDecimalPartialOrIntermediateState;
using cudf_velox::validateIntermediateColumnType;

#define DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Name, name, KIND)               \
  struct Groupby##Name##Aggregator : GroupbyAggregator {                 \
    Groupby##Name##Aggregator(                                           \
        core::AggregationNode::Step step,                                \
        uint32_t inputIndex,                                             \
        VectorPtr constant,                                              \
        const TypePtr& resultType)                                       \
        : GroupbyAggregator(step, inputIndex, constant, resultType) {}   \
                                                                         \
    void addGroupbyRequest(                                              \
        cudf::table_view const& tbl,                                     \
        std::vector<cudf::groupby::aggregation_request>& requests,       \
        rmm::cuda_stream_view stream) override {                         \
      VELOX_CHECK(                                                       \
          constant == nullptr,                                           \
          #Name "Aggregator does not yet support constant input");       \
      auto& request = requests.emplace_back();                           \
      output_idx = requests.size() - 1;                                  \
      request.values = tbl.column(inputIndex);                           \
      request.aggregations.push_back(                                    \
          cudf::make_##name##_aggregation<cudf::groupby_aggregation>()); \
    }                                                                    \
                                                                         \
    std::unique_ptr<cudf::column> makeOutputColumn(                      \
        std::vector<cudf::groupby::aggregation_result>& results,         \
        rmm::cuda_stream_view stream,                                    \
        rmm::device_async_resource_ref mr) override {                    \
      auto col = std::move(results[output_idx].results[0]);              \
      const auto cudfType = cudf_velox::veloxToCudfDataType(resultType); \
      if (col->type() != cudfType) {                                     \
        col = cudf::cast(*col, cudfType, stream, mr);                    \
      }                                                                  \
      return col;                                                        \
    }                                                                    \
                                                                         \
   private:                                                              \
    uint32_t output_idx;                                                 \
  };

DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Sum, sum, SUM)
DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Min, min, MIN)
DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Max, max, MAX)

// Decimal SUM and AVG aggregators are separate implementations, as they need to
// handle the VARBINARY encoded intermediate state for streaming aggregation.
// Due to the packing and unpacking of that intermediate state, and the special
// handling required for the decimal divide, we cannot just use the existing
// cudf::make_mean_aggregation class. Also, unlike other aggregators, these
// classes hold state (the decoded intermediate sum and count columns and
// associated indices) in order to guarantee a lifetime constraint between
// aggregation steps.

void addDecimalSumCountRequestsAfterDecode(
    cudf::column_view encodedColumn,
    int32_t scale,
    std::vector<cudf::groupby::aggregation_request>& requests,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    uint32_t& countIdx,
    std::unique_ptr<cudf::column>& decodedSum,
    std::unique_ptr<cudf::column>& decodedCount) {
  auto sumAndCount =
      cudf_velox::deserializeDecimalSumState(encodedColumn, scale, stream);
  decodedSum.swap(sumAndCount.sum);
  decodedCount.swap(sumAndCount.count);

  sumIdx = requests.size();
  auto& sumRequest = requests.emplace_back();
  sumRequest.values = decodedSum->view();
  sumRequest.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  countIdx = requests.size();
  auto& countRequest = requests.emplace_back();
  countRequest.values = decodedCount->view();
  countRequest.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

// Decodes serialized state and adds sum + count groupby requests, used by the
// intermediate step (both SUM and AVG) and the final AVG step. resultType is
// DECIMAL for final AVG and DECIMAL or VARBINARY for intermediate; VARBINARY
// carries no scale, so decode at scale 0.
void addDecimalDecodedSumCountRequests(
    cudf::table_view const& tbl,
    uint32_t inputIndex,
    const TypePtr& resultType,
    std::vector<cudf::groupby::aggregation_request>& requests,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    uint32_t& countIdx,
    std::unique_ptr<cudf::column>& decodedSum,
    std::unique_ptr<cudf::column>& decodedCount) {
  validateIntermediateColumnType(tbl.column(inputIndex));
  auto scale = resultType->isDecimal()
      ? getDecimalPrecisionScale(*resultType).second
      : 0;
  addDecimalSumCountRequestsAfterDecode(
      tbl.column(inputIndex),
      scale,
      requests,
      stream,
      sumIdx,
      countIdx,
      decodedSum,
      decodedCount);
}

void addDecimalFinalSumOnlyRequest(
    cudf::table_view const& tbl,
    uint32_t inputIndex,
    const TypePtr& resultType,
    std::vector<cudf::groupby::aggregation_request>& requests,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    std::unique_ptr<cudf::column>& decodedSum) {
  validateIntermediateColumnType(tbl.column(inputIndex));
  auto scale = getDecimalPrecisionScale(*resultType).second;
  auto& request = requests.emplace_back();
  sumIdx = requests.size() - 1;
  auto sumAndCount = cudf_velox::deserializeDecimalSumState(
      tbl.column(inputIndex), scale, stream);
  decodedSum.swap(sumAndCount.sum);
  request.values = decodedSum->view();
  request.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

void addDecimalRawPartialSingleSumRequest(
    cudf::table_view const& tbl,
    uint32_t inputIndex,
    std::vector<cudf::groupby::aggregation_request>& requests,
    bool includeCountAggregation,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    std::unique_ptr<cudf::column>& castedInput) {
  auto inputView = castDecimal64InputToDecimal128(
      tbl.column(inputIndex), castedInput, stream);
  auto& request = requests.emplace_back();
  sumIdx = requests.size() - 1;
  request.values = inputView;
  request.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  if (includeCountAggregation) {
    request.aggregations.push_back(
        cudf::make_count_aggregation<cudf::groupby_aggregation>(
            cudf::null_policy::EXCLUDE));
  }
}

struct GroupbyDecimalSumAggregator : GroupbyAggregator {
  GroupbyDecimalSumAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, constant, resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
    if (step == core::AggregationNode::Step::kIntermediate) {
      addDecimalDecodedSumCountRequests(
          tbl,
          inputIndex,
          resultType,
          requests,
          stream,
          sumIdx_,
          countIdx_,
          decodedSum_,
          decodedCount_);
    } else if (step == core::AggregationNode::Step::kFinal) {
      addDecimalFinalSumOnlyRequest(
          tbl, inputIndex, resultType, requests, stream, sumIdx_, decodedSum_);
    } else {
      addDecimalRawPartialSingleSumRequest(
          tbl,
          inputIndex,
          requests,
          step == core::AggregationNode::Step::kPartial,
          stream,
          sumIdx_,
          castedInput_);
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto col = std::move(results[sumIdx_].results[0]);
    if (step == core::AggregationNode::Step::kPartial) {
      auto count = std::move(results[sumIdx_].results[1]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    if (step == core::AggregationNode::Step::kIntermediate) {
      auto count = std::move(results[countIdx_].results[0]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    auto const cudfResType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfResType) {
      col = cudf::cast(*col, cudfResType, stream, mr);
    }
    return col;
  }

 private:
  uint32_t sumIdx_{0};
  uint32_t countIdx_{0};
  std::unique_ptr<cudf::column> decodedSum_;
  std::unique_ptr<cudf::column> decodedCount_;
  // Holds the DECIMAL64->DECIMAL128 cast of raw input (kPartial/kSingle), kept
  // alive while the groupby request references its view.
  std::unique_ptr<cudf::column> castedInput_;
};

struct GroupbyDecimalAvgAggregator : GroupbyAggregator {
  GroupbyDecimalAvgAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, constant, resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
    if (step == core::AggregationNode::Step::kIntermediate ||
        step == core::AggregationNode::Step::kFinal) {
      addDecimalDecodedSumCountRequests(
          tbl,
          inputIndex,
          resultType,
          requests,
          stream,
          sumIdx_,
          countIdx_,
          decodedSum_,
          decodedCount_);
    } else {
      addDecimalRawPartialSingleSumRequest(
          tbl,
          inputIndex,
          requests,
          step == core::AggregationNode::Step::kPartial ||
              step == core::AggregationNode::Step::kSingle,
          stream,
          sumIdx_,
          castedInput_);
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto col = std::move(results[sumIdx_].results[0]);
    if (step == core::AggregationNode::Step::kSingle) {
      auto count = std::move(results[sumIdx_].results[1]);
      return finalizeDecimalAverage(
          std::move(col), std::move(count), resultType, stream, mr);
    }
    if (step == core::AggregationNode::Step::kPartial) {
      auto count = std::move(results[sumIdx_].results[1]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    if (step == core::AggregationNode::Step::kIntermediate) {
      auto count = std::move(results[countIdx_].results[0]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    if (step == core::AggregationNode::Step::kFinal) {
      auto count = std::move(results[countIdx_].results[0]);
      return finalizeDecimalAverage(
          std::move(col), std::move(count), resultType, stream, mr);
    }
    // All four aggregation steps are handled above.
    VELOX_UNREACHABLE();
  }

 private:
  uint32_t sumIdx_{0};
  uint32_t countIdx_{0};
  std::unique_ptr<cudf::column> decodedSum_;
  std::unique_ptr<cudf::column> decodedCount_;
  // Holds the DECIMAL64->DECIMAL128 cast of raw input (kPartial/kSingle), kept
  // alive while the groupby request references its view.
  std::unique_ptr<cudf::column> castedInput_;
};

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
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
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
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto col = std::move(results[outputIndex_].results[0]);
    if (inputKind_ == CountInputKind::kNullConstant) {
      auto zero = cudf::numeric_scalar<int64_t>(0, true, stream, get_temp_mr());
      col = cudf::make_column_from_scalar(zero, col->size(), stream, mr);
    }
    // cudf produces int32 for count but velox expects int64.
    const auto cudfOutputType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfOutputType) {
      col = cudf::cast(*col, cudfOutputType, stream, mr);
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
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
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
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    const auto& outputType = asRowType(resultType);
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return std::move(results[meanIdx_].results[0]);
      case core::AggregationNode::Step::kPartial: {
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[sumIdx_].results[1]);

        auto const size = sum->size();
        auto const cudfSumType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        if (sum->type() != cudf::data_type(cudfSumType)) {
          sum = cudf::cast(*sum, cudf::data_type(cudfSumType), stream, mr);
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count =
              cudf::cast(*count, cudf::data_type(cudfCountType), stream, mr);
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
        auto const cudfSumType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        if (sum->type() != cudf::data_type(cudfSumType)) {
          sum = cudf::cast(*sum, cudf::data_type(cudfSumType), stream, mr);
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count =
              cudf::cast(*count, cudf::data_type(cudfCountType), stream, mr);
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
            cudf_velox::veloxToCudfDataType(resultType),
            stream,
            mr);
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
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
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
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return std::move(results[outputIdx_].results[0]);
      case core::AggregationNode::Step::kPartial: {
        auto count = std::move(results[outputIdx_].results[0]);
        auto mean = std::move(results[outputIdx_].results[1]);
        auto m2 = std::move(results[outputIdx_].results[2]);
        return makeM2StructColumn(
            std::move(count), std::move(mean), std::move(m2), stream, mr);
      }
      case core::AggregationNode::Step::kIntermediate: {
        auto merged = std::move(results[outputIdx_].results[0]);

        // Check if types already match expected output - avoid copies if so
        const auto& outputType = asRowType(resultType);
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfMeanType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        auto const cudfM2Type =
            cudf_velox::veloxToCudfDataType(outputType->childAt(2));

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
        auto count =
            std::make_unique<cudf::column>(mergedView.child(0), stream, mr);
        auto mean =
            std::make_unique<cudf::column>(mergedView.child(1), stream, mr);
        auto m2 =
            std::make_unique<cudf::column>(mergedView.child(2), stream, mr);
        return makeM2StructColumn(
            std::move(count), std::move(mean), std::move(m2), stream, mr);
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
        return cudf::copy_if_else(*stddev, nullDouble, *validMask, stream, mr);
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
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    const auto& outputType = asRowType(resultType);
    auto const cudfCountType =
        cudf_velox::veloxToCudfDataType(outputType->childAt(0));
    auto const cudfMeanType =
        cudf_velox::veloxToCudfDataType(outputType->childAt(1));
    auto const cudfM2Type =
        cudf_velox::veloxToCudfDataType(outputType->childAt(2));
    if (count->type() != cudfCountType) {
      count = cudf::cast(*count, cudfCountType, stream, mr);
    }
    if (mean->type() != cudfMeanType) {
      mean = cudf::cast(*mean, cudfMeanType, stream, mr);
    }
    if (m2->type() != cudfM2Type) {
      m2 = cudf::cast(*m2, cudfM2Type, stream, mr);
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
    if (p.isDecimalAggregate) {
      return std::make_unique<GroupbyDecimalSumAggregator>(
          p.companionStep, p.inputIndex, p.constant, p.resultType);
    }
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
    if (p.isDecimalAggregate) {
      return std::make_unique<GroupbyDecimalAvgAggregator>(
          p.companionStep, p.inputIndex, p.constant, p.resultType);
    }
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

struct GroupbyLeafState final : public BufferedState {
  explicit GroupbyLeafState(InputChunk chunk) : chunk(std::move(chunk)) {}

  InputChunk chunk;
};

} // namespace

class GroupbyBufferedStateOps final : public BufferedStateOps {
 public:
  explicit GroupbyBufferedStateOps(CudfGroupby& owner) : owner_(owner) {
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
      // Handles both kPartial and kIntermediate aggregation steps.
      auto compacted = owner_.doGroupByAggregation(
          permutedInputView,
          owner_.groupingKeyOutputChannels_,
          owner_.aggregators_,
          owner_.bufferedResultType_,
          stream,
          get_output_mr());
      return compacted
          ? makeOwnedChunk(std::move(compacted), owner_.bufferedResultType_)
          : InputChunk{};
    }

    if (!owner_.isSingleStep_) {
      // kFinal aggregation step
      // Here we can avoid one initial compaction because the input and buffered
      // result type are the same but possibly permuted.
      //
      // Note: When the input batch has low cardinality, this can result in an
      // InputChunk that doesn't split easily when needed in
      // PartitionedBufferedState::splitLeafAndAddInput. But that's okay because
      // it's rare that we get low cardinality data that is not sufficiently
      // compacted in the partial aggregation step that always precedes a final
      // aggregation operator. Avoiding an additional groupby here is more
      // important for performance reasons

      // TODO (dm): Investigate introspecting the batch data with lightweight
      // functions like HLL to see if we can made a dynamic runtime decision on
      // pre-compacting the input.
      return makeBorrowedChunk(
          std::move(rawInput), owner_.bufferedResultType_, permutedInputView);
    }

    // kSingle step.
    auto compacted = owner_.doGroupByAggregation(
        permutedInputView,
        owner_.groupingKeyOutputChannels_,
        owner_.partialAggregators_,
        owner_.bufferedResultType_,
        stream,
        get_output_mr());
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
        groupbyLeaf->chunk.stream,
        get_output_mr());
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
        stream,
        get_output_mr());
    return merged
        ? makeOwnedChunk(std::move(merged), owner_.bufferedResultType_)
        : InputChunk{};
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

    auto const& cudfConfig = CudfConfig::getInstance();
    const auto maxBufferedRows = cudfConfig.batchSizeMaxThreshold.value_or(
        std::numeric_limits<int32_t>::max());
    VELOX_CHECK_GT(maxBufferedRows, 0);
    if (isFinalOrSingle) {
      partitionedBufferedState_ = std::make_unique<PartitionedBufferedState>(
          std::make_unique<GroupbyBufferedStateOps>(*this), maxBufferedRows);
    } else if (isPartialOutput_) {
      flushableBufferedState_ = std::make_unique<FlushableBufferedState>(
          std::make_unique<GroupbyBufferedStateOps>(*this),
          maxBufferedRows,
          maxPartialAggregationMemoryUsage_);
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
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
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
    aggregator->addGroupbyRequest(tableView, requests, stream);
  }

  auto [groupKeys, results] = groupByOwner.aggregate(requests, stream, mr);
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
    resultColumns.push_back(aggregator->makeOutputColumn(results, stream, mr));
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

void CudfGroupby::recordPartialFlushStats(const CudfVector& output) {
  const auto numOutputRows = output.size();
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
}

RowVectorPtr CudfGroupby::doGetOutput() {
  // Handle partial streaming groupby.
  if (isPartialOutput_ && streamingEnabled_) {
    if (!flushableBufferedState_) {
      return nullptr;
    }

    if (auto output = flushableBufferedState_->getOutput(noMoreInput_)) {
      recordPartialFlushStats(*output);
      return output;
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
      std::exchange(inputs_, {}), inputType_, stream, get_temp_mr());

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
      stream,
      get_output_mr());
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
