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
#include "velox/experimental/cudf/expression/NullMask.h"
#include "velox/experimental/cudf/expression/sparksql/SubStringFunction.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/vector/BaseVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::sparksql {
namespace {

class SubStringFunction : public CudfFunction {
 public:
  explicit SubStringFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "substring expects at least 2 inputs");
    VELOX_CHECK_LE(
        expr->inputs().size(), 3, "substring expects at most 3 inputs");

    if (auto inputExpr =
            std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0])) {
      inputIsConstant_ = true;
      inputIsNull_ = inputExpr->value()->isNullAt(0);
      if (!inputIsNull_) {
        input_ = inputExpr->value()->toString(0);
        inputLength_ = utf8Length(input_);
      }
    }

    if (auto startExpr =
            std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1])) {
      startIsConstant_ = true;
      startIsNull_ = startExpr->value()->isNullAt(0);
      if (!startIsNull_) {
        rawStartValue_ =
            startExpr->value()->as<SimpleVector<int32_t>>()->valueAt(0);
        if (rawStartValue_ > 0) {
          start_ = normalizePositiveStart(rawStartValue_);
        } else if (rawStartValue_ == 0) {
          start_ = 0;
        }
      }
    }

    hasLength_ = expr->inputs().size() == 3;
    if (hasLength_) {
      if (auto lengthExpr =
              std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2])) {
        lengthIsConstant_ = true;
        lengthIsNull_ = lengthExpr->value()->isNullAt(0);
        if (!lengthIsNull_) {
          length_ = static_cast<cudf::size_type>(
              lengthExpr->value()->as<SimpleVector<int32_t>>()->valueAt(0));
        }
      }
    }

    if (inputIsConstant_ && startIsConstant_ &&
        (!hasLength_ || lengthIsConstant_)) {
      VELOX_NYI("substring with only literal inputs is not supported");
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(
        !inputColumns.empty(),
        "substring requires at least one non-literal input column");

    size_t nextInput = 0;
    auto rowCount = asView(inputColumns[0]).size();

    auto makeAllNullResult = [&]() {
      cudf::string_scalar nullString("", false, stream, mr);
      return cudf::make_column_from_scalar(nullString, rowCount, stream, mr);
    };

    if ((inputIsConstant_ && inputIsNull_) ||
        (startIsConstant_ && startIsNull_) ||
        (hasLength_ && lengthIsConstant_ && lengthIsNull_)) {
      return makeAllNullResult();
    }

    std::unique_ptr<cudf::column> inputColumnHolder;
    cudf::column_view inputColumn;
    if (inputIsConstant_) {
      cudf::string_scalar inputScalar(input_, !inputIsNull_, stream, mr);
      inputColumnHolder =
          cudf::make_column_from_scalar(inputScalar, rowCount, stream, mr);
      inputColumn = inputColumnHolder->view();
    } else {
      inputColumn = asView(inputColumns[nextInput++]);
    }

    if (startIsConstant_ && rawStartValue_ >= 0 &&
        (!hasLength_ || lengthIsConstant_)) {
      // Non-negative constant starts can use cuDF's scalar slice API directly:
      // Spark's 1-based positive start has already been normalized to cuDF's
      // 0-based start, and Spark start zero maps to cuDF start zero.
      auto clampedLength = std::max<cudf::size_type>(0, length_);
      cudf::numeric_scalar<cudf::size_type> startScalar(
          start_, true, stream, mr);
      cudf::numeric_scalar<cudf::size_type> endScalar(
          hasLength_ ? saturatingAddNonNegative(start_, clampedLength) : 0,
          hasLength_,
          stream,
          mr);
      cudf::numeric_scalar<cudf::size_type> stepScalar(1, true, stream, mr);
      return cudf::strings::slice_strings(
          inputColumn, startScalar, endScalar, stepScalar, stream, mr);
    }

    cudf::column_view originalStartColumn;
    bool mergeStartNulls = false;
    cudf::column_view originalLengthColumn;
    bool mergeLengthNulls = false;
    std::unique_ptr<cudf::column> result;
    {
      std::unique_ptr<cudf::column> inputLengthColumn64;
      if (inputIsConstant_) {
        inputLengthColumn64 =
            makeInputLengthColumn64(inputLength_, rowCount, stream, mr);
      } else {
        auto inputCharacterCounts =
            cudf::strings::count_characters(inputColumn, stream, mr);
        inputLengthColumn64 =
            makeWideIndexColumn(inputCharacterCounts->view(), stream, mr);
      }
      std::unique_ptr<cudf::column> startColumn;
      std::unique_ptr<cudf::column> preLengthStartColumn;
      if (startIsConstant_) {
        // `startColumn` is the lower bound passed to cuDF and must be clamped
        // to zero. `preLengthStartColumn` preserves Spark's negative-start
        // value before clamping so that stop = start + length matches Spark.
        startColumn = makeAdjustedConstantStartColumn(
            inputLengthColumn64->view(), rawStartValue_, rowCount, stream, mr);
        preLengthStartColumn = makePreLengthConstantStartColumn(
            inputLengthColumn64->view(), rawStartValue_, rowCount, stream, mr);
      } else {
        originalStartColumn = asView(inputColumns[nextInput++]);
        mergeStartNulls = originalStartColumn.has_nulls();
        auto startColumns = makeStartColumns(
            originalStartColumn, inputLengthColumn64->view(), stream, mr);
        startColumn = std::move(startColumns.start);
        preLengthStartColumn = std::move(startColumns.preLengthStart);
      }

      std::unique_ptr<cudf::column> stopColumn;
      if (!hasLength_) {
        // Spark's two-argument form behaves like an extremely large length,
        // not like cuDF's open-ended slice, for extreme negative starts.
        stopColumn = makeStopColumn(
            preLengthStartColumn->view(),
            std::numeric_limits<int32_t>::max(),
            inputLengthColumn64->view(),
            stream,
            mr);
      } else if (lengthIsConstant_) {
        stopColumn = makeStopColumn(
            preLengthStartColumn->view(),
            std::max<cudf::size_type>(0, length_),
            inputLengthColumn64->view(),
            stream,
            mr);
      } else {
        originalLengthColumn = asView(inputColumns[nextInput++]);
        mergeLengthNulls = originalLengthColumn.has_nulls();
        auto lengthColumn64 =
            makeWideIndexColumn(originalLengthColumn, stream, mr);
        cudf::numeric_scalar<int64_t> zero(0, true, stream, mr);
        cudf::numeric_scalar<int64_t> noUpperBound(0, false, stream, mr);
        auto nonNegativeLengthColumn =
            cudf::clamp(lengthColumn64->view(), zero, noUpperBound, stream, mr);
        stopColumn = makeStopColumn(
            preLengthStartColumn->view(),
            nonNegativeLengthColumn->view(),
            inputLengthColumn64->view(),
            stream,
            mr);
      }

      result = cudf::strings::slice_strings(
          inputColumn, startColumn->view(), stopColumn->view(), stream, mr);
    }

    inputColumnHolder.reset();
    std::vector<cudf::column_view> nullSourceColumns;
    if (mergeStartNulls) {
      nullSourceColumns.push_back(originalStartColumn);
    }
    if (mergeLengthNulls) {
      nullSourceColumns.push_back(originalLengthColumn);
    }
    if (!nullSourceColumns.empty()) {
      // Null start and length values are replaced with zero while computing
      // numeric slice bounds. Merge the original null masks back after slicing
      // to restore Spark null propagation.
      mergeNullSourceNullsIntoResult(*result, nullSourceColumns, stream, mr);
    }
    return result;
  }

 private:
  static cudf::size_type normalizePositiveStart(int32_t startValue) {
    VELOX_DCHECK_GT(startValue, 0);
    return static_cast<cudf::size_type>(startValue - 1);
  }

  static cudf::size_type utf8Length(const std::string& input) {
    size_t length = 0;
    for (const auto byte : input) {
      if ((static_cast<unsigned char>(byte) & 0xC0) != 0x80) {
        ++length;
      }
    }
    VELOX_CHECK_LE(
        length,
        static_cast<size_t>(std::numeric_limits<cudf::size_type>::max()));
    return static_cast<cudf::size_type>(length);
  }

  static cudf::size_type saturatingAddNonNegative(
      cudf::size_type lhs,
      cudf::size_type rhs) {
    VELOX_DCHECK_GE(lhs, 0);
    VELOX_DCHECK_GE(rhs, 0);
    constexpr auto maxSize = std::numeric_limits<cudf::size_type>::max();
    return lhs > maxSize - rhs ? maxSize : lhs + rhs;
  }

  struct StartColumns {
    std::unique_ptr<cudf::column> start;
    std::unique_ptr<cudf::column> preLengthStart;
  };

  static std::unique_ptr<cudf::column> makeWideIndexColumn(
      cudf::column_view indexColumn,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto casted = cudf::cast(
        indexColumn, cudf::data_type{cudf::type_to_id<int64_t>()}, stream, mr);
    if (!indexColumn.has_nulls()) {
      return casted;
    }

    cudf::numeric_scalar<int64_t> zero(0, true, stream, mr);
    auto filled = cudf::replace_nulls(casted->view(), zero, stream, mr);
    casted.reset();
    return filled;
  }

  static std::unique_ptr<cudf::column> makeIndexColumn(
      cudf::column_view indexColumn,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto casted = cudf::cast(
        indexColumn,
        cudf::data_type{cudf::type_to_id<cudf::size_type>()},
        stream,
        mr);
    if (!indexColumn.has_nulls()) {
      return casted;
    }

    cudf::numeric_scalar<cudf::size_type> zero(0, true, stream, mr);
    auto filled = cudf::replace_nulls(casted->view(), zero, stream, mr);
    casted.reset();
    return filled;
  }

  static std::unique_ptr<cudf::column> makeInputLengthColumn64(
      cudf::size_type inputLength,
      cudf::size_type rowCount,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    cudf::numeric_scalar<int64_t> inputLengthScalar(
        inputLength, true, stream, mr);
    return cudf::make_column_from_scalar(
        inputLengthScalar, rowCount, stream, mr);
  }

  static std::unique_ptr<cudf::column> clampStopColumn(
      cudf::column_view stopColumn,
      cudf::column_view inputLengthColumn64,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    cudf::numeric_scalar<int64_t> zero(0, true, stream, mr);
    cudf::numeric_scalar<int64_t> noUpperBound(0, false, stream, mr);
    auto nonNegativeStop =
        cudf::clamp(stopColumn, zero, noUpperBound, stream, mr);
    auto stopPastEnd = cudf::binary_operation(
        nonNegativeStop->view(),
        inputLengthColumn64,
        cudf::binary_operator::GREATER,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto clampedStop64 = cudf::copy_if_else(
        inputLengthColumn64,
        nonNegativeStop->view(),
        stopPastEnd->view(),
        stream,
        mr);
    nonNegativeStop.reset();
    stopPastEnd.reset();
    return cudf::cast(
        clampedStop64->view(),
        cudf::data_type{cudf::type_to_id<cudf::size_type>()},
        stream,
        mr);
  }

  template <typename WideLength>
  static std::unique_ptr<cudf::column> makeStopColumnFromWideInputs(
      cudf::column_view preLengthStartColumn64,
      WideLength const& length64,
      cudf::column_view inputLengthColumn64,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto unclampedStop = cudf::binary_operation(
        preLengthStartColumn64,
        length64,
        cudf::binary_operator::ADD,
        cudf::data_type{cudf::type_to_id<int64_t>()},
        stream,
        mr);
    return clampStopColumn(
        unclampedStop->view(), inputLengthColumn64, stream, mr);
  }

  static std::unique_ptr<cudf::column> makeStopColumn(
      cudf::column_view preLengthStartColumn64,
      cudf::size_type length,
      cudf::column_view inputLengthColumn64,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    cudf::numeric_scalar<int64_t> length64(length, true, stream, mr);
    return makeStopColumnFromWideInputs(
        preLengthStartColumn64, length64, inputLengthColumn64, stream, mr);
  }

  static std::unique_ptr<cudf::column> makeStopColumn(
      cudf::column_view preLengthStartColumn64,
      cudf::column_view lengthColumn64,
      cudf::column_view inputLengthColumn64,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    return makeStopColumnFromWideInputs(
        preLengthStartColumn64,
        lengthColumn64,
        inputLengthColumn64,
        stream,
        mr);
  }

  static StartColumns makeStartColumns(
      cudf::column_view startColumn,
      cudf::column_view inputLengthColumn64,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto startValues = makeWideIndexColumn(startColumn, stream, mr);
    cudf::numeric_scalar<int64_t> zero(0, true, stream, mr);
    cudf::numeric_scalar<int64_t> one(1, true, stream, mr);
    auto positiveStarts = cudf::binary_operation(
        startValues->view(),
        one,
        cudf::binary_operator::GREATER_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto zeroStarts = cudf::binary_operation(
        startValues->view(),
        zero,
        cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto shiftedStart = cudf::binary_operation(
        startValues->view(),
        one,
        cudf::binary_operator::SUB,
        cudf::data_type{cudf::type_to_id<int64_t>()},
        stream,
        mr);
    auto adjustedNegativeStarts = cudf::binary_operation(
        inputLengthColumn64,
        startValues->view(),
        cudf::binary_operator::ADD,
        cudf::data_type{cudf::type_to_id<int64_t>()},
        stream,
        mr);
    auto zeroColumn = cudf::make_column_from_scalar(
        zero, inputLengthColumn64.size(), stream, mr);
    auto nonPositivePreLengthStart = cudf::copy_if_else(
        zeroColumn->view(),
        adjustedNegativeStarts->view(),
        zeroStarts->view(),
        stream,
        mr);
    auto preLengthStart = cudf::copy_if_else(
        shiftedStart->view(),
        nonPositivePreLengthStart->view(),
        positiveStarts->view(),
        stream,
        mr);

    cudf::numeric_scalar<int64_t> noUpperBound(0, false, stream, mr);
    auto clampedNegativeStarts = cudf::clamp(
        adjustedNegativeStarts->view(), zero, noUpperBound, stream, mr);
    auto nonPositiveAdjustedStart = cudf::copy_if_else(
        zeroColumn->view(),
        clampedNegativeStarts->view(),
        zeroStarts->view(),
        stream,
        mr);
    auto adjustedStart = cudf::copy_if_else(
        shiftedStart->view(),
        nonPositiveAdjustedStart->view(),
        positiveStarts->view(),
        stream,
        mr);

    return {
        cudf::cast(
            adjustedStart->view(),
            cudf::data_type{cudf::type_to_id<cudf::size_type>()},
            stream,
            mr),
        std::move(preLengthStart)};
  }

  static std::unique_ptr<cudf::column> makePreLengthConstantStartColumn(
      cudf::column_view inputLengthColumn64,
      int32_t rawStartValue,
      cudf::size_type rowCount,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    cudf::numeric_scalar<int64_t> zero(0, true, stream, mr);
    if (rawStartValue > 0) {
      cudf::numeric_scalar<int64_t> adjustedStart(
          normalizePositiveStart(rawStartValue), true, stream, mr);
      return cudf::make_column_from_scalar(adjustedStart, rowCount, stream, mr);
    }
    if (rawStartValue == 0) {
      return cudf::make_column_from_scalar(zero, rowCount, stream, mr);
    }

    cudf::numeric_scalar<int64_t> rawStart(rawStartValue, true, stream, mr);
    return cudf::binary_operation(
        inputLengthColumn64,
        rawStart,
        cudf::binary_operator::ADD,
        cudf::data_type{cudf::type_to_id<int64_t>()},
        stream,
        mr);
  }

  static std::unique_ptr<cudf::column> makeAdjustedConstantStartColumn(
      cudf::column_view inputLengthColumn64,
      int32_t rawStartValue,
      cudf::size_type rowCount,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    if (rawStartValue > 0) {
      cudf::numeric_scalar<cudf::size_type> adjustedStart(
          normalizePositiveStart(rawStartValue), true, stream, mr);
      return cudf::make_column_from_scalar(adjustedStart, rowCount, stream, mr);
    }

    cudf::numeric_scalar<cudf::size_type> zero(0, true, stream, mr);
    if (rawStartValue == 0) {
      return cudf::make_column_from_scalar(zero, rowCount, stream, mr);
    }

    cudf::numeric_scalar<int64_t> rawStart(rawStartValue, true, stream, mr);
    auto adjustedNegativeStart = cudf::binary_operation(
        inputLengthColumn64,
        rawStart,
        cudf::binary_operator::ADD,
        cudf::data_type{cudf::type_to_id<int64_t>()},
        stream,
        mr);
    cudf::numeric_scalar<int64_t> zero64(0, true, stream, mr);
    cudf::numeric_scalar<int64_t> noUpperBound(0, false, stream, mr);
    auto clampedAdjustedStart = cudf::clamp(
        adjustedNegativeStart->view(), zero64, noUpperBound, stream, mr);
    adjustedNegativeStart.reset();
    return cudf::cast(
        clampedAdjustedStart->view(),
        cudf::data_type{cudf::type_to_id<cudf::size_type>()},
        stream,
        mr);
  }

  bool inputIsConstant_{false};
  bool inputIsNull_{false};
  bool startIsConstant_{false};
  bool startIsNull_{false};
  bool hasLength_{false};
  bool lengthIsConstant_{false};
  bool lengthIsNull_{false};
  std::string input_;
  cudf::size_type inputLength_{0};
  int32_t rawStartValue_{0};
  cudf::size_type start_{0};
  cudf::size_type length_{0};
};

} // namespace

std::shared_ptr<CudfFunction> makeSubStringFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  return std::make_shared<SubStringFunction>(expr);
}

} // namespace facebook::velox::cudf_velox::sparksql
