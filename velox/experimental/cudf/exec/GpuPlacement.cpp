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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergConnector.h"
#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfLocalPartition.h"
#include "velox/experimental/cudf/exec/CudfNestedLoopJoin.h"
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/GpuPlacement.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::cudf_velox {
namespace {

bool reject(std::string* reason, std::string message) {
  if (reason != nullptr) {
    *reason = std::move(message);
  }
  return false;
}

bool containsCustomComparison(const TypePtr& type) {
  if (type->providesCustomComparison()) {
    return true;
  }
  for (size_t i = 0; i < type->size(); ++i) {
    if (containsCustomComparison(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

bool containsCustomComparison(const core::TypedExprPtr& expression) {
  if (containsCustomComparison(expression->type())) {
    return true;
  }
  return std::any_of(
      expression->inputs().begin(),
      expression->inputs().end(),
      [](const auto& input) { return containsCustomComparison(input); });
}

// Returns whether legacy_cast leaves a cast from 'source' to 'target' as is: a
// cast between BOOLEAN, integer, REAL, DOUBLE, DECIMAL and DATE types, or from
// VARCHAR to DATE or DECIMAL.
// NOTE: The list is short on purpose. legacy_cast changes how a VARCHAR parses
// as a TIMESTAMP, how a TIMESTAMP, REAL or DOUBLE prints as a VARCHAR, and
// whether a VARCHAR with non-ASCII characters casts to an integer.
bool isLegacyCastIndependent(const TypePtr& source, const TypePtr& target) {
  const auto isNumberOrDate = [](const TypePtr& type) {
    return type->isDate() || type->isDecimal() ||
        (std::string_view(type->name()) == type->kindName() &&
         (type->isBoolean() || type->isTinyint() || type->isSmallint() ||
          type->isInteger() || type->isBigint() || type->isReal() ||
          type->isDouble()));
  };
  if (source->isVarchar()) {
    return target->isDate() || target->isDecimal();
  }
  return isNumberOrDate(source) && isNumberOrDate(target);
}

// Returns whether 'expr' holds a CAST or TRY_CAST that legacy_cast may change.
bool containsLegacyCastDependentCast(const core::TypedExprPtr& expr) {
  if (expr->isCastKind() &&
      !isLegacyCastIndependent(expr->inputs().at(0)->type(), expr->type())) {
    return true;
  }
  return std::any_of(
      expr->inputs().begin(), expr->inputs().end(), [](const auto& input) {
        return containsLegacyCastDependentCast(input);
      });
}

} // namespace

bool canRunFilterProjectOnGpu(
    const core::FilterNode* filter,
    const core::ProjectNode* project,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason) {
  // cuDF sizes an expression result from an input column. Over a zero-column
  // input a filter has nothing to evaluate against, and only constant
  // projections, which CudfFilterProject fills to the row count of each batch,
  // can run. A FilterNode keeps its input type, so the source of the Project
  // gives the input type of a fused FilterProject too.
  const auto& inputType =
      (project != nullptr ? project->sources()[0] : filter->sources()[0])
          ->outputType();
  if (inputType->size() == 0) {
    if (filter != nullptr) {
      return reject(reason, "FilterProject filter over a zero-column input");
    }
    if (!std::all_of(
            project->projections().begin(),
            project->projections().end(),
            [&](const auto& projection) {
              return isConstantProjectionOverZeroColumns(inputType, projection);
            })) {
      return reject(
          reason,
          "FilterProject non-constant projection over a zero-column input");
    }
  }
  if (filter != nullptr && !canExprRunOnGpu(filter->filter(), queryCtx, pool)) {
    return reject(reason, "FilterProject filter cannot be evaluated by cuDF");
  }
  if (project != nullptr) {
    for (const auto& projection : project->projections()) {
      if (!canExprRunOnGpu(projection, queryCtx, pool)) {
        return reject(
            reason, "FilterProject projections cannot be evaluated by cuDF");
      }
    }
  }
  return true;
}

bool canRunAggregationOnGpu(
    const core::AggregationNode& node,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason) {
  if (!canBeEvaluatedByCudf(node, queryCtx, pool)) {
    return reject(reason, "Aggregation cannot be evaluated by cuDF");
  }
  return true;
}

bool isHashJoinSupportedByCudf(
    core::JoinType joinType,
    bool nullAware,
    bool hasFilter,
    std::string* reason) {
  if (!CudfHashJoinProbe::isSupportedJoinType(joinType)) {
    return reject(reason, "HashJoin unsupported join type");
  }
  // Disabling null-aware anti join with filter until we implement it right
  if (joinType == core::JoinType::kAnti && nullAware && hasFilter) {
    return reject(
        reason, "HashJoin null-aware anti join with filter not implemented");
  }
  return true;
}

bool canRunHashJoinOnGpu(
    const core::HashJoinNode& node,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason) {
  if (!isHashJoinSupportedByCudf(
          node.joinType(),
          node.isNullAware(),
          node.filter() != nullptr,
          reason)) {
    return false;
  }
  if (node.filter() && !canJoinFilterRunOnGpu(node.filter(), queryCtx, pool)) {
    return reject(reason, "HashJoin join filter cannot be evaluated by cuDF");
  }
  return true;
}

bool canRunNestedLoopJoinOnGpu(
    const core::NestedLoopJoinNode& node,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason) {
  if (!CudfNestedLoopJoinProbe::isSupportedJoinType(node.joinType())) {
    return reject(
        reason,
        fmt::format(
            "NestedLoopJoin unsupported join type: {}",
            static_cast<int>(node.joinType())));
  }
  if (node.joinCondition()) {
    if (containsCustomComparison(node.joinCondition())) {
      return reject(
          reason, "NestedLoopJoin condition has custom comparison semantics");
    }
    if (!canJoinFilterRunOnGpu(node.joinCondition(), queryCtx, pool)) {
      return reject(
          reason, "NestedLoopJoin filter cannot be evaluated by cuDF");
    }
  }
  return true;
}

bool canRunTopNRowNumberOnGpu(
    const core::TopNRowNumberNode& node,
    std::string* reason) {
  if (!CudfTopNRowNumber::canRunOnGPU(node)) {
    return reject(
        reason,
        "TopNRowNumber ranking function or key types are not supported by cuDF");
  }
  return true;
}

bool canRunMarkDistinctOnGpu(
    const core::MarkDistinctNode& node,
    std::string* reason) {
  if (!node.masks().empty()) {
    return reject(reason, "MarkDistinct masks are not supported by cuDF");
  }
  return true;
}

bool canRunLocalPartitionOnGpu(
    const std::shared_ptr<const core::LocalPartitionNode>& node,
    std::string* reason) {
  if (node->scaleWriter()) {
    return reject(reason, "LocalPartition scale writer is unsupported by cuDF");
  }
  if (!CudfLocalPartition::shouldReplace(node)) {
    return reject(
        reason, "LocalPartition partition spec is unsupported by cuDF");
  }
  return true;
}

bool isGpuTableScan(
    const core::TableScanNode& node,
    const velox::connector::ConnectorQueryCtx& connectorQueryCtx,
    const core::QueryConfig& queryConfig,
    std::string* reason) {
  auto scanConnector = velox::connector::ConnectorRegistry::tryGet(
      node.tableHandle()->connectorId());
  if (dynamic_cast<const connector::hive::CudfHiveConnector*>(
          scanConnector.get()) == nullptr &&
      dynamic_cast<const connector::hive::iceberg::CudfIcebergConnector*>(
          scanConnector.get()) == nullptr) {
    return reject(
        reason,
        "TableScan connector is not CudfHiveConnector or CudfIcebergConnector");
  }
  if (!isTypeSupportedByCudf(node.outputType())) {
    return reject(
        reason,
        fmt::format(
            "TableScan output type is unsupported by cuDF: {}",
            node.outputType()->toString()));
  }
  if (const auto unsupported =
          connector::hive::CudfHiveDataSource::unsupportedReason(
              node.tableHandle(), connectorQueryCtx)) {
    return reject(
        reason,
        fmt::format(
            "TableScan is not supported by the cuDF data source: {}",
            *unsupported));
  }

  // TODO: Remove this check once the data source folds with the session
  // QueryConfig (facebookincubator/velox#19476).
  const auto hiveTableHandle =
      std::dynamic_pointer_cast<const velox::connector::hive::HiveTableHandle>(
          node.tableHandle());
  VELOX_CHECK_NOT_NULL(hiveTableHandle);
  const auto& remainingFilter = hiveTableHandle->remainingFilter();
  if (queryConfig.isLegacyCast() && remainingFilter &&
      containsLegacyCastDependentCast(remainingFilter)) {
    VELOX_UNSUPPORTED(
        "Remaining filter with a cast that legacy_cast may change is not supported by the cuDF data source: {}",
        remainingFilter->toString());
  }
  return true;
}

bool isGpuTableScan(
    const core::TableScanNode& node,
    const core::QueryCtx& queryCtx,
    std::string* reason) {
  // The check folds the remaining filter, which takes an expression evaluator
  // and a pool. 'queryCtx' is const, so the evaluator runs on a private
  // QueryCtx with the same QueryConfig.
  const auto pool = memory::memoryManager()->addLeafPool();
  const auto evaluatorQueryCtx = core::QueryCtx::create(
      nullptr, core::QueryConfig{queryCtx.queryConfig().rawConfigsCopy()});
  const auto connectorQueryCtx =
      velox::connector::ConnectorQueryCtx::Builder()
          .operatorPool(pool.get())
          .sessionProperties(queryCtx.connectorSessionProperties(
              node.tableHandle()->connectorId()))
          .expressionEvaluator(
              std::make_unique<exec::SimpleExpressionEvaluator>(
                  evaluatorQueryCtx.get(), pool.get()))
          .queryId(queryCtx.queryId())
          .planNodeId(node.id())
          .sessionTimezone(queryCtx.queryConfig().sessionTimezone())
          .adjustTimestampToTimezone(
              queryCtx.queryConfig().adjustTimestampToTimezone())
          .build();
  return isGpuTableScan(
      node, *connectorQueryCtx, queryCtx.queryConfig(), reason);
}

} // namespace facebook::velox::cudf_velox
