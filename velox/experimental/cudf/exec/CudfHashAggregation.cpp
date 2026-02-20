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
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/Type.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <vector>

namespace {

using namespace facebook::velox;

#define DEFINE_SIMPLE_AGGREGATOR(Name, name, KIND)                            \
  struct Name##Aggregator : cudf_velox::CudfHashAggregation::Aggregator {     \
    Name##Aggregator(                                                         \
        core::AggregationNode::Step step,                                     \
        uint32_t inputIndex,                                                  \
        VectorPtr constant,                                                   \
        bool is_global,                                                       \
        const TypePtr& resultType)                                            \
        : Aggregator(                                                         \
              step,                                                           \
              cudf::aggregation::KIND,                                        \
              inputIndex,                                                     \
              constant,                                                       \
              is_global,                                                      \
              resultType) {}                                                  \
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
        col = cudf::cast(*col, cudfType, stream);                             \
      }                                                                       \
      return col;                                                             \
    }                                                                         \
                                                                              \
    std::unique_ptr<cudf::column> doReduce(                                   \
        cudf::table_view const& input,                                        \
        TypePtr const& outputType,                                            \
        rmm::cuda_stream_view stream) override {                              \
      auto const aggRequest =                                                 \
          cudf::make_##name##_aggregation<cudf::reduce_aggregation>();        \
      auto const cudfOutputType =                                             \
          cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType));         \
      auto const resultScalar = cudf::reduce(                                 \
          input.column(inputIndex), *aggRequest, cudfOutputType, stream);     \
      return cudf::make_column_from_scalar(*resultScalar, 1, stream);         \
    }                                                                         \
                                                                              \
   private:                                                                   \
    uint32_t output_idx;                                                      \
  };

DEFINE_SIMPLE_AGGREGATOR(Sum, sum, SUM)
DEFINE_SIMPLE_AGGREGATOR(Min, min, MIN)
DEFINE_SIMPLE_AGGREGATOR(Max, max, MAX)

struct CountAggregator : cudf_velox::CudfHashAggregation::Aggregator {
  CountAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      bool isGlobal,
      const TypePtr& resultType)
      : Aggregator(
            step,
            cudf::aggregation::COUNT_VALID,
            inputIndex,
            constant,
            isGlobal,
            resultType) {}

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

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream) override {
    if (exec::isRawInput(step)) {
      // For raw input, implement count using size + null count
      auto inputCol = input.column(constant == nullptr ? inputIndex : 0);

      // count_valid: size - null_count, count_all: just the size
      int64_t count = constant == nullptr
          ? inputCol.size() - inputCol.null_count()
          : inputCol.size();

      auto resultScalar = cudf::numeric_scalar<int64_t>(count);

      return cudf::make_column_from_scalar(resultScalar, 1, stream);
    } else {
      // For non-raw input (intermediate/final), use sum aggregation
      auto const aggRequest =
          cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto const cudfOutputType = cudf::data_type(cudf::type_id::INT64);
      auto const resultScalar = cudf::reduce(
          input.column(inputIndex), *aggRequest, cudfOutputType, stream);
      resultScalar->set_valid_async(true, stream);
      return cudf::make_column_from_scalar(*resultScalar, 1, stream);
    }
    return nullptr;
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    // cudf produces int32 for count(0) but velox expects int64
    auto col = std::move(results[outputIdx_].results[0]);
    const auto cudfOutputType =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(resultType));
    if (col->type() != cudfOutputType) {
      col = cudf::cast(*col, cudfOutputType, stream);
    }
    return col;
  }

 private:
  uint32_t outputIdx_;
};

struct MeanAggregator : cudf_velox::CudfHashAggregation::Aggregator {
  MeanAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      bool isGlobal,
      const TypePtr& resultType)
      : Aggregator(
            step,
            cudf::aggregation::MEAN,
            inputIndex,
            constant,
            isGlobal,
            resultType) {}

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
        // We don't know how to handle kIntermediate step for mean
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
          sum = cudf::cast(*sum, cudf::data_type(cudfSumType), stream);
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count = cudf::cast(*count, cudf::data_type(cudfCountType), stream);
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
          sum = cudf::cast(*sum, cudf::data_type(cudfSumType), stream);
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count = cudf::cast(*count, cudf::data_type(cudfCountType), stream);
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
            stream);
        return avg;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        auto const aggRequest =
            cudf::make_mean_aggregation<cudf::reduce_aggregation>();
        auto const cudfOutputType =
            cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType));
        auto const resultScalar = cudf::reduce(
            input.column(inputIndex), *aggRequest, cudfOutputType, stream);
        return cudf::make_column_from_scalar(*resultScalar, 1, stream);
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
            input.column(inputIndex), *aggRequest, cudfSumType, stream);
        auto sumCol =
            cudf::make_column_from_scalar(*sumResultScalar, 1, stream);

        // libcudf doesn't have a count agg for reduce. What we want is to
        // count the number of valid rows.
        auto countCol = cudf::make_column_from_scalar(
            cudf::numeric_scalar<int64_t>(
                input.column(inputIndex).size() -
                input.column(inputIndex).null_count()),
            1,
            stream);

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
        auto const sumResultScalar =
            cudf::reduce(sumCol, *sumAggRequest, sumCol.type(), stream);
        auto sumResultCol =
            cudf::make_column_from_scalar(*sumResultScalar, 1, stream);

        // sum the counts
        auto const countAggRequest =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const countResultScalar =
            cudf::reduce(countCol, *countAggRequest, countCol.type(), stream);

        // divide the sums by the counts
        auto const cudfOutputType =
            cudf::data_type(cudf_velox::veloxToCudfTypeId(outputType));
        return cudf::binary_operation(
            *sumResultCol,
            *countResultScalar,
            cudf::binary_operator::DIV,
            cudfOutputType,
            stream);
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

struct ApproxDistinctAggregator : cudf_velox::CudfHashAggregation::Aggregator {
  static constexpr cudf::null_policy kNullPolicy = cudf::null_policy::EXCLUDE;
  static constexpr cudf::nan_policy kNanPolicy = cudf::nan_policy::NAN_IS_VALID;

  ApproxDistinctAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      bool isGlobal,
      const TypePtr& resultType,
      std::int32_t precision = 11) // Default 11 matches Velox's 2.3% standard
                                   // error (2^11 = 2048 buckets)
      : Aggregator{step, cudf::aggregation::INVALID, inputIndex, constant, isGlobal, resultType},
        precision_{precision} {
    VELOX_CHECK(
        constant == nullptr,
        "ApproxDistinctAggregator does not support constant input");
    VELOX_CHECK(
        isGlobal,
        "ApproxDistinctAggregator currently only supports global aggregation");
  }

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) override {
    VELOX_UNSUPPORTED(
        "approx_distinct is not supported as a group aggregation");
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    VELOX_UNSUPPORTED(
        "approx_distinct is not supported as a group aggregation");
  }

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream) override {
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
        kNanPolicy,
        stream);

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
          cudf::numeric_scalar<int64_t>(0, true, stream), 1, stream);
    }

    return mergeSketchesAndApply(
        sketch_column,
        [stream](cudf::approx_distinct_count& sketch) {
          std::size_t estimate = sketch.estimate(stream);
          return cudf::make_column_from_scalar(
              cudf::numeric_scalar<int64_t>(
                  static_cast<int64_t>(estimate), true, stream),
              1,
              stream);
        },
        stream);
  }

  std::int32_t precision_;
};

std::unique_ptr<cudf_velox::CudfHashAggregation::Aggregator> createAggregator(
    core::AggregationNode::Step step,
    std::string const& kind,
    uint32_t inputIndex,
    VectorPtr constant,
    bool isGlobal,
    const TypePtr& resultType) {
  auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (kind.rfind(prefix + "sum", 0) == 0) {
    return std::make_unique<SumAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else if (kind.rfind(prefix + "count", 0) == 0) {
    return std::make_unique<CountAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else if (kind.rfind(prefix + "min", 0) == 0) {
    return std::make_unique<MinAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else if (kind.rfind(prefix + "max", 0) == 0) {
    return std::make_unique<MaxAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else if (kind.rfind(prefix + "avg", 0) == 0) {
    return std::make_unique<MeanAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else if (kind.rfind(prefix + "approx_distinct", 0) == 0) {
    return std::make_unique<ApproxDistinctAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else {
    VELOX_NYI("Aggregation not yet supported, kind: {}", kind);
  }
}

/// \brief Convert companion function to step for the aggregation function
///
/// Companion functions are functions that are registered in velox along with
/// their main aggregation functions. These are designed to always function
/// with a fixed `step`. This is to allow spark style planNodes where `step` is
/// the property of the aggregation function rather than the planNode.
/// Companion functions allow us to override the planNode's step and use
/// aggregations of different steps in the same planNode
/// If an agg function name contains companionStep keyword, may cause error, now
/// it does not exist.
core::AggregationNode::Step getCompanionStep(
    std::string const& kind,
    core::AggregationNode::Step step) {
  if (kind.ends_with("_merge")) {
    return core::AggregationNode::Step::kIntermediate;
  }

  if (kind.ends_with("_partial")) {
    return core::AggregationNode::Step::kPartial;
  }

  // The format is count_merge_extract_BIGINT or count_merge_extract.
  if (kind.find("_merge_extract") != std::string::npos) {
    return core::AggregationNode::Step::kFinal;
  }

  return step;
}

std::string getOriginalName(const std::string& kind) {
  if (kind.ends_with("_merge")) {
    return kind.substr(0, kind.size() - std::string("_merge").size());
  }

  if (kind.ends_with("_partial")) {
    return kind.substr(0, kind.size() - std::string("_partial").size());
  }
  // The format is count_merge_extract_BIGINT or count_merge_extract.
  if (auto pos = kind.find("_merge_extract"); pos != std::string::npos) {
    return kind.substr(0, pos);
  }

  return kind;
}

bool hasFinalAggs(
    std::vector<core::AggregationNode::Aggregate> const& aggregates) {
  return std::any_of(aggregates.begin(), aggregates.end(), [](auto const& agg) {
    return agg.call->name().ends_with("_merge_extract");
  });
}

bool isCompanionAggregateName(std::string const& kind) {
  return kind.ends_with("_merge") || kind.ends_with("_partial") ||
      kind.find("_merge_extract") != std::string::npos;
}

bool hasCompanionAggregates(
    std::vector<core::AggregationNode::Aggregate> const& aggregates) {
  return std::any_of(aggregates.begin(), aggregates.end(), [](auto const& agg) {
    return isCompanionAggregateName(agg.call->name());
  });
}

struct AggregationInputChannels {
  std::vector<column_index_t> channels;
  std::vector<VectorPtr> constants;
};

AggregationInputChannels buildAggregationInputChannels(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx,
    RowTypePtr const& inputRowSchema,
    std::vector<column_index_t> const& groupingKeyInputChannels) {
  AggregationInputChannels result;
  result.constants.resize(aggregationNode.aggregates().size());
  result.channels.reserve(
      groupingKeyInputChannels.size() + aggregationNode.aggregates().size());
  result.channels.insert(
      result.channels.end(),
      groupingKeyInputChannels.begin(),
      groupingKeyInputChannels.end());

  const auto fallbackChannel =
      groupingKeyInputChannels.empty() ? 0 : groupingKeyInputChannels.front();

  for (auto i = 0; i < aggregationNode.aggregates().size(); ++i) {
    auto const& aggregate = aggregationNode.aggregates()[i];
    std::vector<column_index_t> aggInputs;
    for (auto const& arg : aggregate.call->inputs()) {
      if (auto const field =
              dynamic_cast<core::FieldAccessTypedExpr const*>(arg.get())) {
        aggInputs.push_back(inputRowSchema->getChildIdx(field->name()));
      } else if (
          auto constant =
              dynamic_cast<const core::ConstantTypedExpr*>(arg.get())) {
        result.constants[i] = constant->toConstantVector(operatorCtx.pool());
        aggInputs.push_back(fallbackChannel);
      } else {
        VELOX_NYI("Constants and lambdas not yet supported");
      }
    }

    VELOX_CHECK(aggInputs.size() <= 1);
    if (aggInputs.empty()) {
      aggInputs.push_back(fallbackChannel);
    }

    if (aggregate.distinct) {
      VELOX_NYI("De-dup before aggregation is not yet supported");
    }

    result.channels.push_back(aggInputs[0]);
  }

  return result;
}

auto toAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants) {
  bool const isGlobal = aggregationNode.groupingKeys().empty();
  const auto numKeys = aggregationNode.groupingKeys().size();

  std::vector<std::unique_ptr<cudf_velox::CudfHashAggregation::Aggregator>>
      aggregators;
  aggregators.reserve(aggregationNode.aggregates().size());
  for (size_t i = 0; i < aggregationNode.aggregates().size(); ++i) {
    // Positional mapping: inputs are keys first, then aggregate columns in
    // aggregate order.
    auto const& aggregate = aggregationNode.aggregates()[i];
    auto const inputIndex = numKeys + i;
    auto const kind = aggregate.call->name();
    auto const constant = constants[i];
    auto const companionStep = getCompanionStep(kind, step);
    const auto originalName = getOriginalName(kind);
    const auto resultType = exec::isPartialOutput(companionStep)
        ? exec::resolveIntermediateType(originalName, aggregate.rawInputTypes)
        : outputType->childAt(numKeys + i);

    aggregators.push_back(createAggregator(
        companionStep, kind, inputIndex, constant, isGlobal, resultType));
  }
  return aggregators;
}

RowTypePtr getFinalStepBufferedType(
    core::AggregationNode const& aggregationNode) {
  const auto outputRowType = asRowType(aggregationNode.outputType());
  const auto numKeys = aggregationNode.groupingKeys().size();

  std::vector<std::string> names = outputRowType->names();
  std::vector<TypePtr> types = outputRowType->children();

  VELOX_CHECK_EQ(names.size(), types.size());
  VELOX_CHECK_GE(types.size(), numKeys + aggregationNode.aggregates().size());

  for (auto i = 0; i < aggregationNode.aggregates().size(); ++i) {
    auto const& aggregate = aggregationNode.aggregates()[i];
    const auto originalName = getOriginalName(aggregate.call->name());
    types[numKeys + i] =
        exec::resolveIntermediateType(originalName, aggregate.rawInputTypes);
  }

  return ROW(std::move(names), std::move(types));
}

std::string makeCudfAggregationOperatorName(core::AggregationNode::Step step) {
  return std::string{"CudfAggregation"} +
      std::string{core::AggregationNode::toName(step)};
}

} // namespace

namespace facebook::velox::cudf_velox {

CudfHashAggregation::CudfHashAggregation(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<core::AggregationNode const> const& aggregationNode)
    : Operator(
          driverCtx,
          aggregationNode->outputType(),
          operatorId,
          aggregationNode->id(),
          makeCudfAggregationOperatorName(aggregationNode->step()),
          std::nullopt),
      NvtxHelper(
          nvtx3::rgb{34, 139, 34}, // Forest Green
          operatorId,
          fmt::format("[{}]", aggregationNode->id())),
      aggregationNode_(aggregationNode),
      isPartialOutput_(
          exec::isPartialOutput(aggregationNode->step()) &&
          !hasFinalAggs(aggregationNode->aggregates())),
      isGlobal_(aggregationNode->groupingKeys().empty()),
      isDistinct_(!isGlobal_ && aggregationNode->aggregates().empty()),
      isSingleStep_(
          aggregationNode->step() == core::AggregationNode::Step::kSingle),
      maxPartialAggregationMemoryUsage_(
          driverCtx->queryConfig().maxPartialAggregationMemoryUsage()) {}

void CudfHashAggregation::initialize() {
  Operator::initialize();

  inputType_ = aggregationNode_->sources()[0]->outputType();
  ignoreNullKeys_ = aggregationNode_->ignoreNullKeys();
  setupGroupingKeyChannelProjections(
      groupingKeyInputChannels_, groupingKeyOutputChannels_);

  auto const numGroupingKeys = groupingKeyOutputChannels_.size();

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
  aggregators_ = toAggregators(
      *aggregationNode_,
      aggregationNode_->step(),
      outputType_,
      aggregationInput.constants);
  streamingEnabled_ = !hasCompanionAggregates(aggregationNode_->aggregates()) &&
      !isGlobal_;

  // Make aggregators for intermediate step when streaming is enabled.
  // Distinct does not need any aggregators.
  if (streamingEnabled_ && !isDistinct_) {
    const bool isFinalOrSingle =
        aggregationNode_->step() == core::AggregationNode::Step::kFinal ||
        aggregationNode_->step() == core::AggregationNode::Step::kSingle;
    bufferedResultType_ = isFinalOrSingle
        ? getFinalStepBufferedType(*aggregationNode_)
        : outputType_;

    std::vector<VectorPtr> nullConstants(numAggregates_);
    intermediateAggregators_ = toAggregators(
        *aggregationNode_,
        core::AggregationNode::Step::kIntermediate,
        bufferedResultType_,
        nullConstants);

    if (isSingleStep_) {
      partialAggregators_ = toAggregators(
          *aggregationNode_,
          core::AggregationNode::Step::kPartial,
          bufferedResultType_,
          aggregationInput.constants);
      finalAggregators_ = toAggregators(
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

void CudfHashAggregation::setupGroupingKeyChannelProjections(
    std::vector<column_index_t>& groupingKeyInputChannels,
    std::vector<column_index_t>& groupingKeyOutputChannels) const {
  VELOX_CHECK(groupingKeyInputChannels.empty());
  VELOX_CHECK(groupingKeyOutputChannels.empty());

  auto const& inputType = aggregationNode_->sources()[0]->outputType();
  auto const& groupingKeys = aggregationNode_->groupingKeys();
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

void CudfHashAggregation::computePartialGroupbyStreaming(CudfVectorPtr tbl) {
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
    // Create a vector of tables to concatenate
    std::vector<cudf::table_view> tablesToConcat;
    tablesToConcat.push_back(bufferedResult_->getTableView());
    tablesToConcat.push_back(groupbyOnInput->getTableView());

    auto partialOutputStream = bufferedResult_->stream();
    // We need to join the input table stream on the partial output stream to
    // make sure the intermediate results are available when we do the concat.
    cudf::detail::join_streams(
        std::vector<rmm::cuda_stream_view>{inputTableStream},
        partialOutputStream);

    // Concatenate the tables
    auto concatenatedTable =
        cudf::concatenate(tablesToConcat, partialOutputStream);

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

void CudfHashAggregation::computePartialDistinctStreaming(CudfVectorPtr tbl) {
  // For every input, we'll concat with existing distinct results and then do a
  // distinct on the concatenated results.

  auto inputTableStream = tbl->stream();

  if (bufferedResult_) {
    // Concatenate the input table with the existing distinct results.
    std::vector<cudf::table_view> tablesToConcat;
    tablesToConcat.push_back(bufferedResult_->getTableView());
    tablesToConcat.push_back(tbl->getTableView().select(
        groupingKeyInputChannels_.begin(), groupingKeyInputChannels_.end()));

    auto partialOutputStream = bufferedResult_->stream();
    // We need to join the input table stream on the partial output stream to
    // make sure the input table is available when we do the concat.
    cudf::detail::join_streams(
        std::vector<rmm::cuda_stream_view>{inputTableStream},
        partialOutputStream);

    auto concatenatedTable =
        cudf::concatenate(tablesToConcat, partialOutputStream);

    // Do a distinct on the concatenated results.
    // Keep concatenatedTable alive while we use its view.
    auto distinctOutput = getDistinctKeys(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        inputTableStream);
    bufferedResult_ = distinctOutput;
  } else {
    // First time processing, just store the result of the input batch's
    // distinct. Use getTableView() to avoid expensive materialization for
    // packed_table. tbl stays alive during this function call.
    bufferedResult_ = getDistinctKeys(
        tbl->getTableView(), groupingKeyInputChannels_, inputTableStream);
  }
}

void CudfHashAggregation::computeFinalGroupbyStreaming(CudfVectorPtr tbl) {
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

  auto concatenatedTable = cudf::concatenate(tablesToConcat, finalStream);
  auto compactedOutput = doGroupByAggregation(
      concatenatedTable->view(),
      groupingKeyOutputChannels_,
      intermediateAggregators_,
      bufferedResultType_,
      finalStream);
  bufferedResult_ = compactedOutput;
}

void CudfHashAggregation::computeSingleGroupbyStreaming(CudfVectorPtr tbl) {
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
    std::vector<cudf::table_view> tablesToConcat;
    tablesToConcat.push_back(bufferedResult_->getTableView());
    tablesToConcat.push_back(groupbyOnInput->getTableView());

    auto partialOutputStream = bufferedResult_->stream();
    cudf::detail::join_streams(
        std::vector<rmm::cuda_stream_view>{inputTableStream},
        partialOutputStream);

    auto concatenatedTable =
        cudf::concatenate(tablesToConcat, partialOutputStream);

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

void CudfHashAggregation::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (input->size() == 0) {
    return;
  }
  numInputRows_ += input->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  if (isPartialOutput_ && !isGlobal_ && streamingEnabled_) {
    if (isDistinct_) {
      // Handle partial distinct aggregation.
      computePartialDistinctStreaming(cudfInput);
    } else {
      // Handle partial groupby aggregation.
      computePartialGroupbyStreaming(cudfInput);
    }
    return;
  }

  if (isSingleStep_ && streamingEnabled_ && !isGlobal_ && !isDistinct_) {
    computeSingleGroupbyStreaming(cudfInput);
    return;
  }

  if (!isPartialOutput_ && streamingEnabled_ && !isGlobal_ && !isDistinct_) {
    computeFinalGroupbyStreaming(cudfInput);
    return;
  }

  // Handle non-streaming or global cases.
  inputs_.push_back(std::move(cudfInput));
}

CudfVectorPtr CudfHashAggregation::doGroupByAggregation(
    cudf::table_view tableView,
    std::vector<column_index_t> const& groupByKeys,
    std::vector<std::unique_ptr<Aggregator>>& aggregators,
    TypePtr const& outputType,
    rmm::cuda_stream_view stream) {
  auto groupbyKeyView =
      tableView.select(groupByKeys.begin(), groupByKeys.end());

  size_t const numGroupingKeys = groupbyKeyView.num_columns();

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

  auto [groupKeys, results] = groupByOwner.aggregate(requests, stream);
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

CudfVectorPtr CudfHashAggregation::doGlobalAggregation(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> resultColumns;
  resultColumns.reserve(aggregators_.size());
  for (auto i = 0; i < aggregators_.size(); i++) {
    resultColumns.push_back(
        aggregators_[i]->doReduce(tableView, outputType_->childAt(i), stream));
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(),
      outputType_,
      1,
      std::make_unique<cudf::table>(std::move(resultColumns)),
      stream);
}

CudfVectorPtr CudfHashAggregation::getDistinctKeys(
    cudf::table_view tableView,
    std::vector<column_index_t> const& groupByKeys,
    rmm::cuda_stream_view stream) {
  auto result = cudf::distinct(
      tableView.select(groupByKeys.begin(), groupByKeys.end()),
      {groupingKeyOutputChannels_.begin(), groupingKeyOutputChannels_.end()},
      cudf::duplicate_keep_option::KEEP_FIRST,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream);

  auto numRows = result->num_rows();

  // velox expects nullptr instead of a table with 0 rows
  if (numRows == 0) {
    return nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, numRows, std::move(result), stream);
}

CudfVectorPtr CudfHashAggregation::releaseAndResetPartialOutput() {
  VELOX_DCHECK(!isGlobal_);
  auto numOutputRows = bufferedResult_->size();
  const double aggregationPct =
      numOutputRows == 0 ? 0 : (numOutputRows * 1.0) / numInputRows_ * 100;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat("flushRowCount", RuntimeCounter(numOutputRows));
    lockedStats->addRuntimeStat("flushTimes", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "partialAggregationPct", RuntimeCounter(aggregationPct));
  }

  numInputRows_ = 0;
  // We're moving bufferedResult_ to the caller because we want it to be null
  // after this call.
  return std::move(bufferedResult_);
}

RowVectorPtr CudfHashAggregation::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  // Handle partial groupby and distinct.
  if (isPartialOutput_ && !isGlobal_ && streamingEnabled_) {
    if (bufferedResult_ &&
        bufferedResult_->estimateFlatSize() >
            maxPartialAggregationMemoryUsage_) {
      // This is basically a flush of the partial output.
      return releaseAndResetPartialOutput();
    }
    if (not noMoreInput_) {
      // Don't produce output if the partial output hasn't reached memory limit
      // and there's more batches to come.
      return nullptr;
    }
    if (!bufferedResult_ && finished_) {
      return nullptr;
    }
    return releaseAndResetPartialOutput();
  }

  if (finished_) {
    return nullptr;
  }

  if (!isPartialOutput_ && !noMoreInput_) {
    // Final aggregation has to wait for all batches to arrive so we cannot
    // return any results here.
    return nullptr;
  }

  // Single streaming: finalize with final-step aggregators.
  if (isSingleStep_ && !isGlobal_ && !isDistinct_ && streamingEnabled_) {
    finished_ = true;
    if (!bufferedResult_) {
      return nullptr;
    }
    auto stream = bufferedResult_->stream();
    auto result = doGroupByAggregation(
        bufferedResult_->getTableView(),
        groupingKeyOutputChannels_,
        finalAggregators_,
        outputType_,
        stream);
    stream.synchronize();
    bufferedResult_.reset();
    return result;
  }

  // Final streaming: finalize with the step's own aggregators.
  if (!isPartialOutput_ && !isSingleStep_ && !isGlobal_ && !isDistinct_ &&
      streamingEnabled_) {
    if (!noMoreInput_) {
      return nullptr;
    }
    finished_ = true;
    if (!bufferedResult_) {
      return nullptr;
    }
    auto stream = bufferedResult_->stream();
    auto result = doGroupByAggregation(
        bufferedResult_->getTableView(),
        groupingKeyOutputChannels_,
        aggregators_,
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

  auto tbl = getConcatenatedTable(inputs_, inputType_, stream);

  // Release input data after synchronizing.
  stream.synchronize();
  inputs_.clear();

  if (noMoreInput_) {
    finished_ = true;
  }

  VELOX_CHECK_NOT_NULL(tbl);

  // Use tbl->view() instead of moving the table.
  // tbl stays alive until the end of this function, keeping the view valid.
  if (isDistinct_) {
    return getDistinctKeys(tbl->view(), groupingKeyInputChannels_, stream);
  }

  auto permutedInputView = tbl->view().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  if (isGlobal_) {
    return doGlobalAggregation(permutedInputView, stream);
  } else {
    return doGroupByAggregation(
        permutedInputView,
        groupingKeyOutputChannels_,
        aggregators_,
        outputType_,
        stream);
  }
}

void CudfHashAggregation::noMoreInput() {
  Operator::noMoreInput();
  if (isPartialOutput_ && inputs_.empty()) {
    finished_ = true;
  }
}

bool CudfHashAggregation::isFinished() {
  return finished_;
}

// Step-aware aggregation registry implementation
StepAwareAggregationRegistry& getStepAwareAggregationRegistry() {
  static StepAwareAggregationRegistry registry;
  return registry;
}

bool registerAggregationFunctionForStep(
    const std::string& name,
    core::AggregationNode::Step step,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite) {
  auto& registry = getStepAwareAggregationRegistry();

  if (!overwrite && registry.find(name) != registry.end() &&
      registry[name].find(step) != registry[name].end()) {
    return false;
  }

  registry[name][step] = signatures;
  return true;
}

// Register step-aware builtin aggregation functions
bool registerStepAwareBuiltinAggregationFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  // Register sum function (split by aggregation step)
  auto sumSingleSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};

  registerAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kSingle,
      sumSingleSignatures);

  auto sumPartialSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};
  registerAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kPartial,
      sumPartialSignatures);

  auto sumFinalIntermediateSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};
  registerAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kFinal,
      sumFinalIntermediateSignatures);
  registerAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kIntermediate,
      sumFinalIntermediateSignatures);

  // Register count function (split by aggregation step)
  auto countSinglePartialSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("double")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("varchar")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("boolean")
          .build(),
      FunctionSignatureBuilder().returnType("bigint").build()};

  registerAggregationFunctionForStep(
      prefix + "count",
      core::AggregationNode::Step::kSingle,
      countSinglePartialSignatures);
  registerAggregationFunctionForStep(
      prefix + "count",
      core::AggregationNode::Step::kPartial,
      countSinglePartialSignatures);

  auto countFinalIntermediateSignatures =
      std::vector<exec::FunctionSignaturePtr>{FunctionSignatureBuilder()
                                                  .returnType("bigint")
                                                  .argumentType("bigint")
                                                  .build()};
  registerAggregationFunctionForStep(
      prefix + "count",
      core::AggregationNode::Step::kFinal,
      countFinalIntermediateSignatures);
  registerAggregationFunctionForStep(
      prefix + "count",
      core::AggregationNode::Step::kIntermediate,
      countFinalIntermediateSignatures);

  // Register min function (same signatures for all steps)
  auto minMaxSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("tinyint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("smallint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("integer")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};

  registerAggregationFunctionForStep(
      prefix + "min", core::AggregationNode::Step::kSingle, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "min", core::AggregationNode::Step::kPartial, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "min", core::AggregationNode::Step::kFinal, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "min",
      core::AggregationNode::Step::kIntermediate,
      minMaxSignatures);

  // Register max function (same signatures for all steps)
  registerAggregationFunctionForStep(
      prefix + "max", core::AggregationNode::Step::kSingle, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "max", core::AggregationNode::Step::kPartial, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "max", core::AggregationNode::Step::kFinal, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "max",
      core::AggregationNode::Step::kIntermediate,
      minMaxSignatures);

  // Register avg function (different signatures for different steps)

  // Single step: avg(input_type) -> double
  auto avgSingleSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};
  registerAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kSingle,
      avgSingleSignatures);

  // Partial step: avg(input_type) -> row(sum input_type, count bigint)
  auto avgPartialSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("double")
          .build()};
  registerAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kPartial,
      avgPartialSignatures);

  // Final step: avg(row(double, bigint)) -> double
  auto avgFinalSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("row(double,bigint)")
          .build()};
  registerAggregationFunctionForStep(
      prefix + "avg", core::AggregationNode::Step::kFinal, avgFinalSignatures);

  // Intermediate step: avg(row(sum input_type, count bigint)) -> row(sum
  // input_type, count bigint)
  auto avgIntermediateSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("row(double,bigint)")
          .build()};
  registerAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kIntermediate,
      avgIntermediateSignatures);

  // Register approx_distinct function
  auto approxDistinctSingleSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("double")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("varchar")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("varbinary")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("date")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("timestamp")
          .build()};
  registerAggregationFunctionForStep(
      prefix + "approx_distinct",
      core::AggregationNode::Step::kSingle,
      approxDistinctSingleSignatures);

  auto approxDistinctPartialSignatures =
      std::vector<exec::FunctionSignaturePtr>{
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("tinyint")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("smallint")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("integer")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("bigint")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("real")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("double")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("varchar")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("varbinary")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("date")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("timestamp")
              .build()};
  registerAggregationFunctionForStep(
      prefix + "approx_distinct",
      core::AggregationNode::Step::kPartial,
      approxDistinctPartialSignatures);

  auto approxDistinctIntermediateSignatures =
      std::vector<exec::FunctionSignaturePtr>{FunctionSignatureBuilder()
                                                  .returnType("varbinary")
                                                  .argumentType("varbinary")
                                                  .build()};
  registerAggregationFunctionForStep(
      prefix + "approx_distinct",
      core::AggregationNode::Step::kIntermediate,
      approxDistinctIntermediateSignatures);

  auto approxDistinctFinalSignatures =
      std::vector<exec::FunctionSignaturePtr>{FunctionSignatureBuilder()
                                                  .returnType("bigint")
                                                  .argumentType("varbinary")
                                                  .build()};
  registerAggregationFunctionForStep(
      prefix + "approx_distinct",
      core::AggregationNode::Step::kFinal,
      approxDistinctFinalSignatures);

  return true;
}

bool matchTypedCallAgainstSignatures(
    const core::CallTypedExpr& call,
    const std::vector<exec::FunctionSignaturePtr>& sigs) {
  const auto n = call.inputs().size();
  std::vector<TypePtr> argTypes;
  argTypes.reserve(n);
  for (const auto& input : call.inputs()) {
    argTypes.push_back(input->type());
  }
  for (const auto& sig : sigs) {
    std::vector<Coercion> coercions(n);
    exec::SignatureBinder binder(*sig, argTypes);
    if (!binder.tryBindWithCoercions(coercions)) {
      continue;
    }

    // For simplicity we skip checking for constant agruments, this may be added
    // in the future

    return true;
  }
  return false;
}

// Step-aware aggregation validation function
bool canAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx) {
  // Check against step-aware aggregation registry
  auto& stepAwareRegistry = getStepAwareAggregationRegistry();
  auto funcIt = stepAwareRegistry.find(call.name());
  if (funcIt == stepAwareRegistry.end()) {
    return false;
  }

  auto stepIt = funcIt->second.find(step);
  if (stepIt == funcIt->second.end()) {
    return false;
  }

  // Validate against step-specific signatures from registry
  return matchTypedCallAgainstSignatures(call, stepIt->second);
}

bool canBeEvaluatedByCudf(
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
    if (!canAggregationBeEvaluatedByCudf(
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

core::TypedExprPtr expandFieldReference(
    const core::TypedExprPtr& expr,
    const core::PlanNode* sourceNode) {
  // If this is a field reference and we have a source projection, expand it
  if (expr->kind() == core::ExprKind::kFieldAccess && sourceNode) {
    auto projectNode = dynamic_cast<const core::ProjectNode*>(sourceNode);
    if (projectNode) {
      auto fieldExpr =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr);
      if (fieldExpr) {
        // Find the corresponding projection expression
        const auto& projections = projectNode->projections();
        const auto& names = projectNode->names();
        for (size_t i = 0; i < names.size(); ++i) {
          if (names[i] == fieldExpr->name()) {
            return projections[i];
          }
        }
      }
    }
  }
  return expr;
}

bool canGroupingKeysBeEvaluatedByCudf(
    const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
    const core::PlanNode* sourceNode,
    core::QueryCtx* queryCtx) {
  // Check grouping key expressions (with expansion)
  for (const auto& groupingKey : groupingKeys) {
    auto expandedKey = expandFieldReference(groupingKey, sourceNode);
    std::vector<core::TypedExprPtr> exprs = {expandedKey};
    if (!canBeEvaluatedByCudf(exprs, queryCtx)) {
      return false;
    }
  }

  return true;
}

} // namespace facebook::velox::cudf_velox
