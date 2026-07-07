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
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/base/Exceptions.h"
#include "velox/core/Expressions.h"
#include "velox/exec/Operator.h"
#include "velox/type/Type.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/rolling.hpp>
#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <nvtx3/nvtx3.hpp>

#include <fmt/format.h>

#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

namespace facebook::velox::cudf_velox {

namespace {

// Returns true when the window function's value argument is a cast expression.
// The CPU Window operator only accepts field-access or constant arguments
// (exec::exprToChannel throws on anything else), so casts on window arguments
// are normally materialized in a projection below the WindowNode. We reject
// them here rather than silently reading the un-cast column, which would
// evaluate f(x) instead of f(cast(x)).
bool windowArgIsCast(const core::WindowNode::Function& func) {
  const auto& inputs = func.functionCall->inputs();
  return !inputs.empty() &&
      std::dynamic_pointer_cast<const core::CastTypedExpr>(inputs[0]) !=
      nullptr;
}

std::optional<column_index_t> resolveInputChannel(
    const core::WindowNode::Function& func,
    const RowTypePtr& inputType) {
  const auto& inputs = func.functionCall->inputs();
  if (inputs.empty()) {
    return std::nullopt;
  }
  // Match the CPU Window operator: the argument must be a field access or
  // constant. canRunOnGPU rejects cast arguments (windowArgIsCast) before we
  // reach here, so we never strip a cast and reinterpret its semantics.
  return exec::exprToChannel(inputs[0].get(), inputType);
}

bool isFullPartitionFrame(
    const core::WindowNode::Function& func,
    bool hasSortKeys) {
  const bool isUnboundedPreceding =
      func.frame.startType == core::WindowNode::BoundType::kUnboundedPreceding;
  const bool isUnboundedFollowing =
      func.frame.endType == core::WindowNode::BoundType::kUnboundedFollowing;
  const bool isCurrentRowFollowing =
      func.frame.endType == core::WindowNode::BoundType::kCurrentRow;
  const bool isRange = func.frame.type == core::WindowNode::WindowType::kRange;
  return isUnboundedPreceding &&
      (isUnboundedFollowing ||
       (isRange && isCurrentRowFollowing && !hasSortKeys));
}

std::optional<std::pair<cudf::range_window_type, cudf::range_window_type>>
toBatchRangeWindowTypes(
    const core::WindowNode::Function& func,
    bool isFullPartition) {
  if (func.frame.type != core::WindowNode::WindowType::kRange) {
    return std::nullopt;
  }
  if (isFullPartition) {
    return std::make_pair(
        cudf::range_window_type{cudf::unbounded{}},
        cudf::range_window_type{cudf::unbounded{}});
  }
  cudf::range_window_type following;
  if (func.frame.endType == core::WindowNode::BoundType::kCurrentRow) {
    following = cudf::current_row{};
  } else if (
      func.frame.endType == core::WindowNode::BoundType::kUnboundedFollowing) {
    following = cudf::unbounded{};
  } else {
    return std::nullopt;
  }
  return std::make_pair(cudf::range_window_type{cudf::unbounded{}}, following);
}

struct PendingRangeRolling {
  size_t funcIndex;
  cudf::column_view inputCol;
  std::unique_ptr<cudf::rolling_aggregation> agg;
};

struct RangeRollingBatch {
  cudf::range_window_type preceding;
  cudf::range_window_type following;
  std::vector<PendingRangeRolling> requests;
};

bool rangeWindowTypesEqual(
    const cudf::range_window_type& left,
    const cudf::range_window_type& right) {
  // Batch gating only uses unbounded/current_row; index distinguishes kinds.
  return left.index() == right.index();
}

RangeRollingBatch* findRangeRollingBatch(
    std::vector<RangeRollingBatch>& batches,
    const std::pair<cudf::range_window_type, cudf::range_window_type>&
        rangeTypes) {
  for (auto& batch : batches) {
    if (rangeWindowTypesEqual(batch.preceding, rangeTypes.first) &&
        rangeWindowTypesEqual(batch.following, rangeTypes.second)) {
      return &batch;
    }
  }
  return nullptr;
}

void addRangeRollingRequest(
    std::vector<RangeRollingBatch>& batches,
    const std::pair<cudf::range_window_type, cudf::range_window_type>&
        rangeTypes,
    PendingRangeRolling pending) {
  if (auto* batch = findRangeRollingBatch(batches, rangeTypes)) {
    batch->requests.push_back(std::move(pending));
    return;
  }
  RangeRollingBatch batch{rangeTypes.first, rangeTypes.second, {}};
  batch.requests.push_back(std::move(pending));
  batches.push_back(std::move(batch));
}

cudf::rank_method toRankMethod(const std::string& name) {
  if (name == "row_number") {
    return cudf::rank_method::FIRST;
  }
  if (name == "rank") {
    return cudf::rank_method::MIN;
  }
  if (name == "dense_rank") {
    return cudf::rank_method::DENSE;
  }
  VELOX_FAIL("Unsupported rank window function: {}", name);
}

std::unique_ptr<cudf::column> makeConstantOnesColumn(
    cudf::size_type numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto oneScalar = cudf::numeric_scalar<int64_t>(1, true, stream, mr);
  return cudf::make_column_from_scalar(oneScalar, numRows, stream, mr);
}

cudf::column_view makeCountStarInputColumn(
    const cudf::table_view& sortedView,
    cudf::size_type logicalRowCount,
    ColumnOrView& owner,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (sortedView.num_columns() > 0) {
    return sortedView.column(0);
  }
  // Reuse a single materialized ones-column across all zero-column count(*)
  // calls in this batch. Overwriting `owner` on every call would invalidate
  // column_views already captured by earlier calls (e.g. deferred requests
  // held in PendingRangeRolling::inputCol).
  if (!std::holds_alternative<std::unique_ptr<cudf::column>>(owner)) {
    auto oneScalar = cudf::numeric_scalar<int64_t>(1, true, stream, mr);
    owner =
        cudf::make_column_from_scalar(oneScalar, logicalRowCount, stream, mr);
  }
  return asView(owner);
}

std::unique_ptr<cudf::column> computeGlobalAggregate(
    cudf::column_view inputCol,
    const std::string& baseName,
    bool isCountStar,
    cudf::data_type resultType,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::unique_ptr<cudf::scalar> resultScalar;
  if (baseName == "sum") {
    auto agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    resultScalar = cudf::reduce(inputCol, *agg, resultType, stream, mr);
  } else if (baseName == "min") {
    auto agg = cudf::make_min_aggregation<cudf::reduce_aggregation>();
    resultScalar = cudf::reduce(inputCol, *agg, inputCol.type(), stream, mr);
  } else if (baseName == "max") {
    auto agg = cudf::make_max_aggregation<cudf::reduce_aggregation>();
    resultScalar = cudf::reduce(inputCol, *agg, inputCol.type(), stream, mr);
  } else if (baseName == "count") {
    auto agg = cudf::make_count_aggregation<cudf::reduce_aggregation>(
        isCountStar ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE);
    resultScalar = cudf::reduce(
        inputCol, *agg, cudf::data_type(cudf::type_id::INT64), stream, mr);
  } else if (baseName == "avg") {
    auto agg = cudf::make_mean_aggregation<cudf::reduce_aggregation>();
    resultScalar = cudf::reduce(
        inputCol, *agg, cudf::data_type(cudf::type_id::FLOAT64), stream, mr);
  } else {
    VELOX_FAIL("Unsupported global aggregate window function: {}", baseName);
  }
  return cudf::make_column_from_scalar(*resultScalar, numRows, stream, mr);
}

bool containsCustomComparison(const TypePtr& type) {
  if (type->providesCustomComparison()) {
    return true;
  }
  for (uint32_t i = 0; i < type->size(); ++i) {
    if (containsCustomComparison(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

cudf::size_type constantRowsBoundValue(const core::TypedExprPtr& value) {
  VELOX_USER_CHECK_NOT_NULL(value, "ROWS frame offset must be specified");

  auto constExpr =
      std::dynamic_pointer_cast<const core::ConstantTypedExpr>(value);
  VELOX_USER_CHECK_NOT_NULL(
      constExpr, "ROWS frame offset must be a constant expression");
  VELOX_USER_CHECK(!constExpr->isNull(), "ROWS frame offset must not be null");
  VELOX_USER_CHECK(
      constExpr->type()->isInteger() || constExpr->type()->isBigint(),
      "ROWS frame offset must be INTEGER or BIGINT type, got {}",
      constExpr->type()->toString());

  int64_t offset;
  if (constExpr->hasValueVector()) {
    auto vec = constExpr->valueVector();
    if (vec->type()->kind() == TypeKind::INTEGER) {
      offset = vec->as<SimpleVector<int32_t>>()->valueAt(0);
    } else {
      offset = vec->as<SimpleVector<int64_t>>()->valueAt(0);
    }
  } else if (constExpr->type()->kind() == TypeKind::INTEGER) {
    offset = constExpr->value().value<int32_t>();
  } else {
    offset = constExpr->value().value<int64_t>();
  }

  VELOX_USER_CHECK_GE(
      offset, 0, "ROWS frame offset must not be negative: {}", offset);
  VELOX_USER_CHECK_LE(
      offset,
      std::numeric_limits<cudf::size_type>::max(),
      "ROWS frame offset {} exceeds cudf size_type limit",
      offset);

  return static_cast<cudf::size_type>(offset);
}

cudf::window_bounds toRowsPrecedingBound(
    core::WindowNode::BoundType type,
    const core::TypedExprPtr& value) {
  switch (type) {
    case core::WindowNode::BoundType::kUnboundedPreceding:
      return cudf::window_bounds::unbounded();
    case core::WindowNode::BoundType::kCurrentRow:
      return cudf::window_bounds::get(1);
    case core::WindowNode::BoundType::kPreceding: {
      const auto offset = constantRowsBoundValue(value);
      VELOX_USER_CHECK_LT(
          offset,
          std::numeric_limits<cudf::size_type>::max(),
          "ROWS PRECEDING offset {} exceeds cudf preceding bound limit",
          offset);
      return cudf::window_bounds::get(offset + 1);
    }
    case core::WindowNode::BoundType::kFollowing: {
      const auto offset = constantRowsBoundValue(value);
      return cudf::window_bounds::get(1 - offset);
    }
    default:
      VELOX_UNREACHABLE(
          "Invalid ROWS start bound type: {}", static_cast<int>(type));
  }
}

cudf::window_bounds toRowsFollowingBound(
    core::WindowNode::BoundType type,
    const core::TypedExprPtr& value) {
  switch (type) {
    case core::WindowNode::BoundType::kUnboundedFollowing:
      return cudf::window_bounds::unbounded();
    case core::WindowNode::BoundType::kCurrentRow:
      return cudf::window_bounds::get(0);
    case core::WindowNode::BoundType::kFollowing:
      return cudf::window_bounds::get(constantRowsBoundValue(value));
    case core::WindowNode::BoundType::kPreceding:
      return cudf::window_bounds::get(-constantRowsBoundValue(value));
    default:
      VELOX_UNREACHABLE(
          "Invalid ROWS end bound type: {}", static_cast<int>(type));
  }
}

// Convert Velox RANGE frame bounds to cudf range_window_bounds.
int64_t constantIntegerOrBigintValue(const core::ConstantTypedExpr& constExpr) {
  VELOX_USER_CHECK(!constExpr.isNull(), "Constant offset must not be null");
  VELOX_USER_CHECK(
      constExpr.type()->isInteger() || constExpr.type()->isBigint(),
      "Constant must be INTEGER or BIGINT type, got {}",
      constExpr.type()->toString());
  if (constExpr.hasValueVector()) {
    auto vec = constExpr.valueVector();
    if (vec->type()->kind() == TypeKind::INTEGER) {
      return vec->as<SimpleVector<int32_t>>()->valueAt(0);
    }
    return vec->as<SimpleVector<int64_t>>()->valueAt(0);
  }
  if (constExpr.type()->kind() == TypeKind::INTEGER) {
    return constExpr.value().value<int32_t>();
  }
  return constExpr.value().value<int64_t>();
}

std::unique_ptr<cudf::rolling_aggregation> makeRollingAggregation(
    const std::string& baseName,
    bool isCountStar) {
  if (baseName == "sum") {
    return cudf::make_sum_aggregation<cudf::rolling_aggregation>();
  }
  if (baseName == "min") {
    return cudf::make_min_aggregation<cudf::rolling_aggregation>();
  }
  if (baseName == "max") {
    return cudf::make_max_aggregation<cudf::rolling_aggregation>();
  }
  if (baseName == "count") {
    auto nullPolicy =
        isCountStar ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    return cudf::make_count_aggregation<cudf::rolling_aggregation>(nullPolicy);
  }
  if (baseName == "avg") {
    return cudf::make_mean_aggregation<cudf::rolling_aggregation>();
  }
  VELOX_FAIL("Unsupported rolling window function: {}", baseName);
}

} // namespace

bool CudfWindow::isSupportedWindowFunction(
    const std::string& baseName,
    size_t numArgs) {
  static const std::unordered_set<std::string> kSupportedFunctions = {
      "lag",
      "lead",
      "row_number",
      "rank",
      "dense_rank",
      "first_value",
      "last_value",
      "sum",
      "min",
      "max",
      "count",
      "avg"};
  if (!kSupportedFunctions.contains(baseName)) {
    return false;
  }
  // lag/lead only support up to 2 arguments (value, offset)
  if ((baseName == "lag" || baseName == "lead") && numArgs > 2) {
    return false;
  }
  return true;
}

bool CudfWindow::canRunOnGPU(const core::WindowNode& windowNode) {
  return canRunOnGPU(windowNode, nullptr);
}

bool CudfWindow::canRunOnGPU(
    const core::WindowNode& windowNode,
    std::string* reason) {
  if (windowNode.sortingKeys().size() > 1) {
    if (reason) {
      *reason = "Multi-column ORDER BY requires a cuDF upgrade (follow-on PR)";
    }
    return false;
  }

  for (const auto& key : windowNode.partitionKeys()) {
    if (containsCustomComparison(key->type())) {
      if (reason) {
        *reason = fmt::format(
            "PARTITION BY key {} uses custom comparison type {}",
            key->name(),
            key->type()->toString());
      }
      return false;
    }
  }

  for (const auto& key : windowNode.sortingKeys()) {
    if (containsCustomComparison(key->type())) {
      if (reason) {
        *reason = fmt::format(
            "ORDER BY key {} uses custom comparison type {}",
            key->name(),
            key->type()->toString());
      }
      return false;
    }
  }

  const auto& prefix = CudfConfig::getInstance().functionNamePrefix;
  const auto inputType = asRowType(windowNode.inputType());
  for (const auto& func : windowNode.windowFunctions()) {
    const auto baseName =
        stripFunctionPrefix(func.functionCall->name(), prefix);
    if (!isSupportedWindowFunction(
            baseName, func.functionCall->inputs().size())) {
      if (reason) {
        *reason = fmt::format(
            "Unsupported window function: {}", func.functionCall->name());
      }
      return false;
    }

    if (windowArgIsCast(func)) {
      if (reason) {
        *reason = fmt::format(
            "Cast expression in {} argument is not supported", baseName);
      }
      return false;
    }

    // Reject NULL constant frame offsets so we fall back to the CPU operator
    // instead of reinterpreting a null as a garbage integer offset at runtime.
    const auto isNullConstantBound = [](core::WindowNode::BoundType type,
                                        const core::TypedExprPtr& value) {
      if (type != core::WindowNode::BoundType::kPreceding &&
          type != core::WindowNode::BoundType::kFollowing) {
        return false;
      }
      auto constExpr =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(value);
      return constExpr != nullptr && constExpr->isNull();
    };
    if (isNullConstantBound(func.frame.startType, func.frame.startValue) ||
        isNullConstantBound(func.frame.endType, func.frame.endValue)) {
      if (reason) {
        *reason = "NULL frame offset is not supported";
      }
      return false;
    }

    if (baseName == "lag" || baseName == "lead") {
      if (func.ignoreNulls) {
        if (reason) {
          *reason = fmt::format("{} IGNORE NULLS is not supported", baseName);
        }
        return false;
      }

      const auto inputChannel = resolveInputChannel(func, inputType);
      if (!inputChannel.has_value() || *inputChannel == kConstantChannel) {
        if (reason) {
          *reason = fmt::format(
              "Constant value argument for {} is not supported", baseName);
        }
        return false;
      }

      if (func.functionCall->inputs().size() >= 2) {
        const auto offsetExpr =
            std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                func.functionCall->inputs()[1]);
        if (!offsetExpr) {
          if (reason) {
            *reason = fmt::format(
                "Non-constant offset for {} is not supported", baseName);
          }
          return false;
        }

        if (offsetExpr->isNull()) {
          if (reason) {
            *reason =
                fmt::format("Null offset for {} is not supported", baseName);
          }
          return false;
        }

        const int64_t offset = constantIntegerOrBigintValue(*offsetExpr);
        constexpr int64_t kMaxCudfLeadLagOffset =
            std::numeric_limits<cudf::size_type>::max() - 1LL;
        if (offset < 0 || offset > kMaxCudfLeadLagOffset) {
          if (reason) {
            *reason = fmt::format(
                "{} offset {} is outside cuDF's supported range [0, {}]",
                baseName,
                offset,
                kMaxCudfLeadLagOffset);
          }
          return false;
        }
      }
    }

    if (!func.functionCall->inputs().empty()) {
      const auto& argumentType = func.functionCall->inputs()[0]->type();

      const bool unsupportedRealSum =
          baseName == "sum" && argumentType->isReal();
      const bool unsupportedDecimalAverage =
          baseName == "avg" && argumentType->isDecimal();
      const bool unsupportedNestedMinMax =
          (baseName == "min" || baseName == "max") &&
          (argumentType->isArray() || argumentType->isMap() ||
           argumentType->isRow());

      if (unsupportedRealSum || unsupportedDecimalAverage ||
          unsupportedNestedMinMax) {
        if (reason) {
          *reason = fmt::format(
              "{} does not support input type {} on cuDF",
              baseName,
              argumentType->toString());
        }
        return false;
      }
    }

    const bool isFullPartition =
        isFullPartitionFrame(func, !windowNode.sortingKeys().empty());
    const bool usesRollingFullPartitionPath =
        !windowNode.partitionKeys().empty() ||
        !windowNode.sortingKeys().empty();

    if (baseName == "avg" && isFullPartition && usesRollingFullPartitionPath) {
      if (reason) {
        *reason = "Full-partition AVG requires optimized cuDF MEAN support";
      }
      return false;
    }

    const bool usesFrame = baseName == "first_value" ||
        baseName == "last_value" || baseName == "sum" || baseName == "min" ||
        baseName == "max" || baseName == "count" || baseName == "avg";

    if (usesFrame) {
      if (auto channel = resolveInputChannel(func, inputType)) {
        if (*channel == kConstantChannel) {
          if (reason) {
            *reason = "Constant window aggregate input not supported";
          }
          return false;
        }
      }

      if (func.frame.type == core::WindowNode::WindowType::kRange) {
        const bool startOk = func.frame.startType ==
            core::WindowNode::BoundType::kUnboundedPreceding;
        const bool endOk = func.frame.endType ==
                core::WindowNode::BoundType::kUnboundedFollowing ||
            func.frame.endType == core::WindowNode::BoundType::kCurrentRow;
        if (!startOk || !endOk) {
          if (reason) {
            *reason =
                "RANGE frame with non-unbounded/current bounds not supported";
          }
          return false;
        }
      }

      const auto isConstantBound = [](core::WindowNode::BoundType type,
                                      const core::TypedExprPtr& value) {
        if (type == core::WindowNode::BoundType::kPreceding ||
            type == core::WindowNode::BoundType::kFollowing) {
          // A PRECEDING/FOLLOWING bound requires a constant offset; a null
          // value here would throw at execution time (see
          // constantRowsBoundValue) instead of falling back to CPU cleanly.
          return value != nullptr &&
              std::dynamic_pointer_cast<const core::ConstantTypedExpr>(value) !=
              nullptr;
        }
        return true;
      };
      if (!isConstantBound(func.frame.startType, func.frame.startValue) ||
          !isConstantBound(func.frame.endType, func.frame.endValue)) {
        if (reason) {
          *reason = "Non-constant frame bound not supported";
        }
        return false;
      }
    }

    const auto isNonNegativeConstantBound =
        [](core::WindowNode::BoundType type,
           const core::TypedExprPtr& value) -> std::optional<int64_t> {
      if (type != core::WindowNode::BoundType::kPreceding &&
          type != core::WindowNode::BoundType::kFollowing) {
        return std::nullopt;
      }
      if (!value) {
        return std::nullopt;
      }
      auto constExpr =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(value);
      if (!constExpr) {
        return std::nullopt;
      }
      if (constExpr->hasValueVector()) {
        auto vec = constExpr->valueVector();
        if (vec->type()->kind() == TypeKind::INTEGER) {
          return vec->as<SimpleVector<int32_t>>()->valueAt(0);
        }
        return vec->as<SimpleVector<int64_t>>()->valueAt(0);
      }
      if (constExpr->type()->kind() == TypeKind::INTEGER) {
        return constExpr->value().value<int32_t>();
      }
      return constExpr->value().value<int64_t>();
    };
    const auto validateRowsBound = [&](core::WindowNode::BoundType type,
                                       const core::TypedExprPtr& value,
                                       bool isStartBound) {
      const auto constant = isNonNegativeConstantBound(type, value);
      if (!constant.has_value()) {
        return true;
      }

      if (*constant < 0) {
        if (reason) {
          *reason = fmt::format(
              "Window frame {} offset must not be negative", *constant);
        }
        return false;
      }

      const int64_t maxOffset =
          static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()) -
          (isStartBound && type == core::WindowNode::BoundType::kPreceding ? 1
                                                                           : 0);

      if (*constant > maxOffset) {
        if (reason) {
          *reason = fmt::format(
              "Window frame offset {} exceeds cuDF limit {}",
              *constant,
              maxOffset);
        }
        return false;
      }

      return true;
    };

    if (func.frame.type == core::WindowNode::WindowType::kRows &&
        (!validateRowsBound(
             func.frame.startType, func.frame.startValue, true) ||
         !validateRowsBound(func.frame.endType, func.frame.endValue, false))) {
      return false;
    }
  }
  return true;
}

CudfWindow::CudfWindow(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::WindowNode>& windowNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          windowNode->outputType(),
          windowNode->id(),
          "CudfWindow",
          nvtx3::rgb{255, 165, 0},
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput),
      windowNode_(windowNode),
      inputRowType_(asRowType(windowNode->inputType())) {
  const auto& inputType = windowNode->inputType();

  for (const auto& key : windowNode->partitionKeys()) {
    partitionKeyIndices_.push_back(inputType->getChildIdx(key->name()));
  }

  for (size_t i = 0; i < windowNode->sortingKeys().size(); ++i) {
    sortKeyIndices_.push_back(
        inputType->getChildIdx(windowNode->sortingKeys()[i]->name()));
    const auto& order = windowNode->sortingOrders()[i];
    sortOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    // Velox isNullsFirst() is absolute; cuDF null_order is relative to sort
    // direction. BEFORE means nulls precede values in that direction.
    bool nullsBefore = (order.isNullsFirst() && order.isAscending()) ||
        (!order.isNullsFirst() && !order.isAscending());
    nullOrders_.push_back(
        nullsBefore ? cudf::null_order::BEFORE : cudf::null_order::AFTER);
  }
}

void CudfWindow::doAddInput(RowVectorPtr input) {
  // Queue inputs, process all at once.
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput, "CudfWindow expects CudfVector input");
    inputBatches_.push_back(std::move(cudfInput));
  }
}

void CudfWindow::computeRankColumnsBatch(
    const cudf::table_view& sortedInput,
    const std::vector<std::pair<size_t, std::string>>& pendingRanks,
    cudf::groupby::groupby* rankGrouper,
    std::vector<std::unique_ptr<cudf::column>>& windowResultCols,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  if (pendingRanks.empty()) {
    return;
  }

  const auto numRows = logicalRowCount_;
  // Single-column ORDER BY only (multi-column rejected in canRunOnGPU).
  auto colOrder =
      sortKeyIndices_.empty() ? cudf::order::ASCENDING : sortOrders_[0];
  auto nullOrd =
      sortKeyIndices_.empty() ? cudf::null_order::BEFORE : nullOrders_[0];

  if (partitionKeyIndices_.empty()) {
    for (const auto& [funcIndex, baseName] : pendingRanks) {
      if (sortKeyIndices_.empty() && baseName != "row_number") {
        windowResultCols[funcIndex] =
            makeConstantOnesColumn(numRows, stream, mr);
        continue;
      }
      if (baseName == "row_number") {
        auto oneScalar = cudf::numeric_scalar<int64_t>(1, true, stream, mr);
        windowResultCols[funcIndex] =
            cudf::sequence(numRows, oneScalar, oneScalar, stream, mr);
        continue;
      }
      auto valuesCol = sortedInput.column(sortKeyIndices_[0]);
      auto method = toRankMethod(baseName);
      auto agg = cudf::make_rank_aggregation<cudf::scan_aggregation>(
          method, colOrder, cudf::null_policy::INCLUDE, nullOrd);
      windowResultCols[funcIndex] = cudf::scan(
          valuesCol,
          *agg,
          cudf::scan_type::INCLUSIVE,
          cudf::null_policy::INCLUDE,
          stream,
          mr);
    }
    return;
  }

  std::vector<cudf::groupby::scan_request> scanRequests;
  scanRequests.reserve(pendingRanks.size());
  std::vector<size_t> batchedFuncIndices;
  batchedFuncIndices.reserve(pendingRanks.size());

  for (size_t i = 0; i < pendingRanks.size(); ++i) {
    const auto& [funcIndex, baseName] = pendingRanks[i];
    if (sortKeyIndices_.empty() && baseName != "row_number") {
      windowResultCols[funcIndex] = makeConstantOnesColumn(numRows, stream, mr);
      continue;
    }

    cudf::groupby::scan_request request;
    if (!sortKeyIndices_.empty()) {
      request.values = sortedInput.column(sortKeyIndices_[0]);
    } else {
      // row_number without ORDER BY: values column is unused for tie detection.
      request.values = sortedInput.column(partitionKeyIndices_[0]);
    }
    request.aggregations.push_back(
        cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
            toRankMethod(baseName),
            colOrder,
            cudf::null_policy::INCLUDE,
            nullOrd));
    scanRequests.push_back(std::move(request));
    batchedFuncIndices.push_back(funcIndex);
  }

  if (scanRequests.empty()) {
    return;
  }

  VELOX_CHECK_NOT_NULL(rankGrouper);
  auto scanResult = rankGrouper->scan(scanRequests, stream, mr);
  auto& aggResults = scanResult.second;
  VELOX_CHECK_EQ(aggResults.size(), scanRequests.size());
  for (size_t i = 0; i < scanRequests.size(); ++i) {
    VELOX_CHECK_EQ(aggResults[i].results.size(), 1);
    windowResultCols[batchedFuncIndices[i]] =
        std::move(aggResults[i].results[0]);
  }
}

std::unique_ptr<cudf::column> CudfWindow::computeLeadLagColumn(
    const cudf::table_view& partKeys,
    cudf::column_view inputCol,
    const core::WindowNode::Function& func,
    const std::string& baseName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_LE(
      func.functionCall->inputs().size(),
      2,
      "cudf {} does not support default value (3rd argument)",
      baseName);

  // Extract offset from the second argument, defaulting to 1.
  auto getOffset = [&]() -> cudf::size_type {
    const auto& args = func.functionCall->inputs();
    if (args.size() < 2) {
      return 1;
    }
    auto constExpr =
        std::dynamic_pointer_cast<const core::ConstantTypedExpr>(args[1]);
    VELOX_USER_CHECK_NOT_NULL(
        constExpr,
        "cudf {} requires constant offset, non-constant offset not supported",
        baseName);
    const int64_t offset = constantIntegerOrBigintValue(*constExpr);
    VELOX_USER_CHECK_GE(
        offset, 0, "cudf {} offset must not be negative: {}", baseName, offset);
    // We pass offset + 1 as the cudf window size below, so cap one below the
    // size_type max to avoid overflowing cudf::size_type.
    VELOX_USER_CHECK_LE(
        offset,
        std::numeric_limits<cudf::size_type>::max() - 1,
        "cudf {} offset {} exceeds cudf size_type limit",
        baseName,
        offset);
    return static_cast<cudf::size_type>(offset);
  };
  auto offset = getOffset();

  if (baseName == "lag") {
    auto agg = cudf::make_lag_aggregation<cudf::rolling_aggregation>(offset);
    return cudf::grouped_rolling_window(
        partKeys, inputCol, offset + 1, 0, offset + 1, *agg, stream, mr);
  }
  auto agg = cudf::make_lead_aggregation<cudf::rolling_aggregation>(offset);
  return cudf::grouped_rolling_window(
      partKeys, inputCol, 0, offset + 1, offset + 1, *agg, stream, mr);
}

std::unique_ptr<cudf::column> CudfWindow::invokeGroupedRollingWindow(
    const cudf::table_view& partKeys,
    cudf::column_view inputCol,
    const core::WindowNode::Function& func,
    std::unique_ptr<cudf::rolling_aggregation> agg,
    bool isFullPartition,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  // RANGE frames are handled by the batched grouped_range_rolling_window path
  // in doGetOutput (see toBatchRangeWindowTypes). canRunOnGPU only accepts
  // RANGE frames that path can express, so RANGE never reaches here.
  VELOX_CHECK(
      func.frame.type != core::WindowNode::WindowType::kRange,
      "RANGE window frames must be handled by the batched range rolling path");

  if (isFullPartition) {
    return cudf::grouped_rolling_window(
        partKeys,
        inputCol,
        cudf::window_bounds::unbounded(),
        cudf::window_bounds::unbounded(),
        1,
        *agg,
        stream,
        mr);
  }

  auto preceding =
      toRowsPrecedingBound(func.frame.startType, func.frame.startValue);
  auto following =
      toRowsFollowingBound(func.frame.endType, func.frame.endValue);
  return cudf::grouped_rolling_window(
      partKeys, inputCol, preceding, following, 1, *agg, stream, mr);
}

std::unique_ptr<cudf::column> CudfWindow::computeNthValueColumn(
    const cudf::table_view& partKeys,
    cudf::column_view inputCol,
    const core::WindowNode::Function& func,
    const std::string& baseName,
    bool isFullPartition,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto nullPolicy = func.ignoreNulls ? cudf::null_policy::EXCLUDE
                                     : cudf::null_policy::INCLUDE;

  if (baseName == "first_value") {
    auto agg = cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
        0, nullPolicy);
    return invokeGroupedRollingWindow(
        partKeys, inputCol, func, std::move(agg), isFullPartition, stream, mr);
  }
  // last_value: use -1 to get the last element in the frame.
  auto agg = cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
      -1, nullPolicy);
  return invokeGroupedRollingWindow(
      partKeys, inputCol, func, std::move(agg), isFullPartition, stream, mr);
}

std::unique_ptr<cudf::column> CudfWindow::computeAggregateColumn(
    const cudf::table_view& partKeys,
    cudf::column_view inputCol,
    const core::WindowNode::Function& func,
    const std::string& baseName,
    bool isCountStar,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  // The unpartitioned, sort-key-less full-partition case is already
  // dispatched to computeGlobalAggregate() directly from doGetOutput(), so
  // this is always reached with a non-trivial partition/rolling window.
  auto agg = makeRollingAggregation(baseName, isCountStar);
  const bool isFullPartition =
      isFullPartitionFrame(func, !sortKeyIndices_.empty());
  return invokeGroupedRollingWindow(
      partKeys, inputCol, func, std::move(agg), isFullPartition, stream, mr);
}

void CudfWindow::doNoMoreInput() {
  Operator::noMoreInput();
  if (inputBatches_.empty()) {
    finished_ = true;
    stream_ = cudfGlobalStreamPool().get_stream();
    streamAcquired_ = true;
    return;
  }

  // Verify total row count doesn't exceed cudf's int32 limit.
  int64_t totalRows = 0;
  for (const auto& batch : inputBatches_) {
    totalRows += batch->size();
  }
  VELOX_CHECK_LE(
      totalRows,
      std::numeric_limits<cudf::size_type>::max(),
      "Total row count {} exceeds cudf size_type limit",
      totalRows);
  logicalRowCount_ = static_cast<cudf::size_type>(totalRows);

  stream_ = cudfGlobalStreamPool().get_stream();
  streamAcquired_ = true;
  auto mr = get_output_mr();

  // Concatenate all input batches into one table with proper stream sync.
  auto allData = getConcatenatedTable(
      std::exchange(inputBatches_, {}), inputRowType_, stream_, mr);

  // Sort by partition keys + sort keys if the plan is not already sorted.
  if (!windowNode_->inputsSorted()) {
    std::vector<cudf::size_type> allSortKeys;
    std::vector<cudf::order> allOrders;
    std::vector<cudf::null_order> allNullOrders;

    for (auto idx : partitionKeyIndices_) {
      allSortKeys.push_back(idx);
      allOrders.push_back(cudf::order::ASCENDING);
      allNullOrders.push_back(cudf::null_order::BEFORE);
    }
    for (size_t i = 0; i < sortKeyIndices_.size(); ++i) {
      allSortKeys.push_back(sortKeyIndices_[i]);
      allOrders.push_back(sortOrders_[i]);
      allNullOrders.push_back(nullOrders_[i]);
    }

    // Skip sorting if there are no sort keys (global window without ORDER BY).
    if (allSortKeys.empty()) {
      sortedData_ = std::move(allData);
    } else {
      auto allView = allData->view();
      auto keyTable = allView.select(allSortKeys);
      sortedData_ = cudf::stable_sort_by_key(
          allView, keyTable, allOrders, allNullOrders, stream_, mr);
    }
  } else {
    sortedData_ = std::move(allData);
  }
}

bool CudfWindow::isFinished() {
  return finished_;
}

RowVectorPtr CudfWindow::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (!sortedData_) {
    finished_ = true;
    return nullptr;
  }

  auto mr = get_output_mr();
  auto sortedView = sortedData_->view();
  ColumnOrView countStarColOwner;

  // Build partition key table for grouped_rolling_window.
  auto partKeys = sortedView.select(partitionKeyIndices_);

  std::unique_ptr<cudf::groupby::groupby> rankGrouper;
  if (!partitionKeyIndices_.empty()) {
    bool needsRankGrouper = false;
    const auto& prefix = CudfConfig::getInstance().functionNamePrefix;
    for (const auto& func : windowNode_->windowFunctions()) {
      const auto baseName =
          stripFunctionPrefix(func.functionCall->name(), prefix);
      if (baseName == "row_number" || baseName == "rank" ||
          baseName == "dense_rank") {
        needsRankGrouper = true;
        break;
      }
    }
    if (needsRankGrouper) {
      rankGrouper = std::make_unique<cudf::groupby::groupby>(
          sortedView.select(partitionKeyIndices_),
          cudf::null_policy::INCLUDE,
          cudf::sorted::YES,
          std::vector<cudf::order>(
              partitionKeyIndices_.size(), cudf::order::ASCENDING),
          std::vector<cudf::null_order>(
              partitionKeyIndices_.size(), cudf::null_order::BEFORE));
    }
  }

  std::vector<std::unique_ptr<cudf::column>> windowResultCols(
      windowNode_->windowFunctions().size());
  std::vector<std::unique_ptr<cudf::column>> decimalSumInputOwners(
      windowNode_->windowFunctions().size());
  std::vector<RangeRollingBatch> rangeRollingBatches;
  std::vector<std::pair<size_t, std::string>> pendingRanks;
  const auto& prefix = CudfConfig::getInstance().functionNamePrefix;

  for (size_t funcIndex = 0; funcIndex < windowNode_->windowFunctions().size();
       ++funcIndex) {
    const auto& func = windowNode_->windowFunctions()[funcIndex];
    const auto baseName =
        stripFunctionPrefix(func.functionCall->name(), prefix);

    if (baseName == "row_number" || baseName == "rank" ||
        baseName == "dense_rank") {
      pendingRanks.emplace_back(funcIndex, baseName);
    } else if (baseName == "lag" || baseName == "lead") {
      auto inputColIdx = resolveInputChannel(func, inputRowType_);
      VELOX_CHECK(
          inputColIdx.has_value(),
          "Window function {} requires an input column",
          baseName);
      auto inputCol = sortedView.column(*inputColIdx);
      windowResultCols[funcIndex] =
          computeLeadLagColumn(partKeys, inputCol, func, baseName, stream_, mr);
    } else if (baseName == "first_value" || baseName == "last_value") {
      auto inputColIdx = resolveInputChannel(func, inputRowType_);
      VELOX_CHECK(
          inputColIdx.has_value(),
          "Window function {} requires an input column",
          baseName);
      auto inputCol = sortedView.column(*inputColIdx);
      const bool isFullPartition =
          isFullPartitionFrame(func, !sortKeyIndices_.empty());
      if (auto rangeTypes = toBatchRangeWindowTypes(func, isFullPartition)) {
        auto nullPolicy = func.ignoreNulls ? cudf::null_policy::EXCLUDE
                                           : cudf::null_policy::INCLUDE;
        std::unique_ptr<cudf::rolling_aggregation> agg;
        if (baseName == "first_value") {
          agg = cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
              0, nullPolicy);
        } else {
          agg = cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
              -1, nullPolicy);
        }
        addRangeRollingRequest(
            rangeRollingBatches,
            rangeTypes.value(),
            PendingRangeRolling{funcIndex, inputCol, std::move(agg)});
      } else {
        windowResultCols[funcIndex] = computeNthValueColumn(
            partKeys, inputCol, func, baseName, isFullPartition, stream_, mr);
      }
    } else if (
        baseName == "sum" || baseName == "min" || baseName == "max" ||
        baseName == "count" || baseName == "avg") {
      auto inputColIdx = resolveInputChannel(func, inputRowType_);
      const bool isCountStar = baseName == "count" && !inputColIdx.has_value();
      if (!isCountStar) {
        VELOX_CHECK(
            inputColIdx.has_value(),
            "Window function {} requires an input column",
            baseName);
      }
      cudf::column_view inputCol = isCountStar
          ? makeCountStarInputColumn(
                sortedView, logicalRowCount_, countStarColOwner, stream_, mr)
          : sortedView.column(*inputColIdx);
      if (baseName == "sum" &&
          func.functionCall->inputs()[0]->type()->isDecimal()) {
        inputCol = castDecimal64InputToDecimal128(
            inputCol, decimalSumInputOwners[funcIndex], stream_);
      }
      const bool isFullPartition =
          isFullPartitionFrame(func, !sortKeyIndices_.empty());
      if (isFullPartition && sortKeyIndices_.empty() &&
          partKeys.num_columns() == 0) {
        windowResultCols[funcIndex] = computeGlobalAggregate(
            inputCol,
            baseName,
            isCountStar,
            veloxToCudfDataType(func.functionCall->type()),
            logicalRowCount_,
            stream_,
            mr);
      } else if (
          auto rangeTypes = toBatchRangeWindowTypes(func, isFullPartition)) {
        addRangeRollingRequest(
            rangeRollingBatches,
            rangeTypes.value(),
            PendingRangeRolling{
                funcIndex,
                inputCol,
                makeRollingAggregation(baseName, isCountStar)});
      } else {
        windowResultCols[funcIndex] = computeAggregateColumn(
            partKeys, inputCol, func, baseName, isCountStar, stream_, mr);
      }
    } else {
      VELOX_FAIL("Unsupported window function for cudf: {}", baseName);
    }
  }

  computeRankColumnsBatch(
      sortedView,
      pendingRanks,
      rankGrouper.get(),
      windowResultCols,
      stream_,
      mr);

  if (!rangeRollingBatches.empty()) {
    ColumnOrView orderbyColHolder{cudf::column_view{}};
    cudf::column_view orderbyCol;
    cudf::order order = cudf::order::ASCENDING;
    cudf::null_order nullOrder = cudf::null_order::BEFORE;
    if (sortKeyIndices_.empty()) {
      auto oneScalar = cudf::numeric_scalar<int64_t>(1, true, stream_, mr);
      orderbyColHolder =
          cudf::sequence(logicalRowCount_, oneScalar, oneScalar, stream_, mr);
      orderbyCol = asView(orderbyColHolder);
    } else {
      orderbyCol = sortedView.column(sortKeyIndices_[0]);
      order = sortOrders_[0];
      nullOrder = nullOrders_[0];
    }
    for (auto& batch : rangeRollingBatches) {
      auto& pendingRequests = batch.requests;
      std::vector<cudf::rolling_request> rollingRequests;
      rollingRequests.reserve(pendingRequests.size());
      for (auto& pending : pendingRequests) {
        rollingRequests.push_back(
            cudf::rolling_request{pending.inputCol, 1, std::move(pending.agg)});
      }
      auto batchResult = cudf::grouped_range_rolling_window(
          partKeys,
          orderbyCol,
          order,
          nullOrder,
          batch.preceding,
          batch.following,
          cudf::host_span<cudf::rolling_request const>(
              rollingRequests.data(), rollingRequests.size()),
          stream_,
          mr);
      auto resultCols = batchResult->release();
      VELOX_CHECK_EQ(resultCols.size(), pendingRequests.size());
      for (size_t i = 0; i < pendingRequests.size(); ++i) {
        windowResultCols[pendingRequests[i].funcIndex] =
            std::move(resultCols[i]);
      }
    }
  }

  // Build the output table: input columns + window result columns.
  // Cast window result columns to expected output types if needed.
  auto sortedCols = sortedData_->release();
  sortedData_.reset();
  const auto numInputCols = inputRowType_->size();
  for (size_t i = 0; i < windowResultCols.size(); ++i) {
    auto& resultColumn = windowResultCols[i];
    VELOX_CHECK_NOT_NULL(resultColumn);
    auto expectedType =
        veloxToCudfDataType(outputType_->childAt(numInputCols + i));
    if (resultColumn->type() != expectedType) {
      resultColumn =
          cudf::cast(resultColumn->view(), expectedType, stream_, mr);
    }

    const auto baseName = stripFunctionPrefix(
        windowNode_->windowFunctions()[i].functionCall->name(), prefix);
    if (baseName == "count" && resultColumn->null_count() > 0) {
      cudf::numeric_scalar<int64_t> zero(0, true, stream_, mr);
      resultColumn =
          cudf::replace_nulls(resultColumn->view(), zero, stream_, mr);
    }

    sortedCols.push_back(std::move(resultColumn));
  }
  auto resultTable = std::make_unique<cudf::table>(std::move(sortedCols));
  auto resultSize = resultTable->num_rows();
  if (resultSize == 0 && logicalRowCount_ > 0) {
    resultSize = logicalRowCount_;
  }

  finished_ = true;
  return std::make_shared<CudfVector>(
      pool(), outputType_, resultSize, std::move(resultTable), stream_);
}

void CudfWindow::doClose() {
  Operator::close();
  // Release GPU allocations only after pending work on stream_ completes.
  if (streamAcquired_) {
    stream_.synchronize();
  }
  inputBatches_.clear();
  sortedData_.reset();
}

} // namespace facebook::velox::cudf_velox
