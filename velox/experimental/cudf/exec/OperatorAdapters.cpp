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
#include "velox/experimental/cudf/exec/CudfAssignUniqueId.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfLimit.h"
#include "velox/experimental/cudf/exec/CudfLocalPartition.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/exec/AssignUniqueId.h"
#include "velox/exec/CallbackSink.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/HashBuild.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/Limit.h"
#include "velox/exec/LocalPartition.h"
#include "velox/exec/OrderBy.h"
#include "velox/exec/StreamingAggregation.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/Task.h"
#include "velox/exec/TopN.h"
#include "velox/exec/Values.h"

namespace facebook::velox::cudf_velox {

/// OperatorAdapterRegistry Implementation
OperatorAdapterRegistry& OperatorAdapterRegistry::getInstance() {
  static OperatorAdapterRegistry instance;
  return instance;
}

void OperatorAdapterRegistry::registerAdapter(
    std::unique_ptr<OperatorAdapter> adapter) {
  adapters_.push_back(std::move(adapter));
}

const OperatorAdapter* OperatorAdapterRegistry::findAdapter(
    const exec::Operator* op) const {
  for (const auto& adapter : adapters_) {
    if (adapter->canHandle(op)) {
      return adapter.get();
    }
  }
  // Note: It is possible to have priority based adapter search.
  // But this is not implemented because it is not needed for now.
  return nullptr;
}

const std::vector<std::unique_ptr<OperatorAdapter>>&
OperatorAdapterRegistry::getAdapters() const {
  return adapters_;
}

void OperatorAdapterRegistry::clear() {
  adapters_.clear();
}

/// TableScanAdapter - Keeps original operator (produces GPU output)
class TableScanAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::TableScan*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    auto tableScanNode =
        std::dynamic_pointer_cast<const core::TableScanNode>(planNode);
    if (!tableScanNode) {
      return false;
    }
    auto const& connector = velox::connector::getConnector(
        tableScanNode->tableHandle()->connectorId());
    auto cudfHiveConnector = std::dynamic_pointer_cast<
        facebook::velox::cudf_velox::connector::hive::CudfHiveConnector>(
        connector);
    return cudfHiveConnector != nullptr;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  int keepOperator() const override {
    return 1;
  }

  std::string name() const override {
    return "TableScan";
  }
};

/// FilterProjectAdapter - Replaces with CudfFilterProject
class FilterProjectAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::FilterProject*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    auto filterProjectOp = dynamic_cast<const exec::FilterProject*>(op);
    if (!filterProjectOp) {
      return false;
    }

    auto projectPlanNode =
        std::dynamic_pointer_cast<const core::ProjectNode>(planNode);
    auto filterNode = filterProjectOp->filterNode();

    if (projectPlanNode) {
      if (projectPlanNode->sources()[0]->outputType()->size() == 0 ||
          projectPlanNode->outputType()->size() == 0) {
        return false;
      }
    }

    // Check filter separately
    if (filterNode) {
      if (!canBeEvaluatedByCudf(
              {filterNode->filter()}, ctx->task->queryCtx().get())) {
        return false;
      }
    }

    // Check projects separately
    if (projectPlanNode) {
      if (!canBeEvaluatedByCudf(
              projectPlanNode->projections(), ctx->task->queryCtx().get())) {
        return false;
      }
    }
    return true;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto filterProjectOp = dynamic_cast<const exec::FilterProject*>(op);
    auto projectPlanNode =
        std::dynamic_pointer_cast<const core::ProjectNode>(planNode);
    auto filterPlanNode = filterProjectOp->filterNode();

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfFilterProject>(
            operatorId, ctx, filterPlanNode, projectPlanNode));
    return result;
  }

  std::string name() const override {
    return "FilterProject";
  }
};

/// AggregationAdapter - Replaces with CudfHashAggregation
class AggregationAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashAggregation*>(op) != nullptr ||
        dynamic_cast<const exec::StreamingAggregation*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    if (!canHandle(op)) {
      return false;
    }

    auto aggregationPlanNode =
        std::dynamic_pointer_cast<const core::AggregationNode>(planNode);
    if (!aggregationPlanNode) {
      return false;
    }

    if (aggregationPlanNode->sources()[0]->outputType()->size() == 0) {
      // We cannot handle RowVectors with a length but no data.
      // This is the case with count(*) global (without groupby)
      return false;
    }

    return canBeEvaluatedByCudf(
        *aggregationPlanNode, ctx->task->queryCtx().get());
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto aggregationPlanNode =
        std::dynamic_pointer_cast<const core::AggregationNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfHashAggregation>(
            operatorId, ctx, aggregationPlanNode));
    return result;
  }

  std::string name() const override {
    return "Aggregation";
  }
};

class CudfHashJoinBaseAdapter : public OperatorAdapter {
 public:
  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    if (!canHandle(op)) {
      return false;
    }

    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);
    if (!joinPlanNode) {
      return false;
    }

    if (!CudfHashJoinProbe::isSupportedJoinType(joinPlanNode->joinType())) {
      return false;
    }

    // Disabling null-aware anti join with filter until we implement it right
    if (joinPlanNode->joinType() == core::JoinType::kAnti &&
        joinPlanNode->isNullAware() && joinPlanNode->filter()) {
      return false;
    }

    if (joinPlanNode->filter()) {
      if (!canBeEvaluatedByCudf(
              {joinPlanNode->filter()}, ctx->task->queryCtx().get())) {
        return false;
      }
    }
    return true;
  }
};

/// HashJoinBuildAdapter - Replaces with CudfHashJoinBuild
class HashJoinBuildAdapter : public CudfHashJoinBaseAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashBuild*>(op) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfHashJoinBuild>(operatorId, ctx, joinPlanNode));
    return result;
  }

  std::string name() const override {
    return "HashJoinBuild";
  }
};

/// HashJoinProbeAdapter - Replaces with CudfHashJoinProbe
class HashJoinProbeAdapter : public CudfHashJoinBaseAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashProbe*>(op) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfHashJoinProbe>(operatorId, ctx, joinPlanNode));
    return result;
  }

  std::string name() const override {
    return "HashJoinProbe";
  }
};

/// OrderByAdapter - Replaces with CudfOrderBy
class OrderByAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::OrderBy*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::OrderByNode>(planNode) !=
        nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto orderByPlanNode =
        std::dynamic_pointer_cast<const core::OrderByNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfOrderBy>(operatorId, ctx, orderByPlanNode));
    return result;
  }

  std::string name() const override {
    return "OrderBy";
  }
};

/// TopNAdapter - Replaces with CudfTopN
class TopNAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::TopN*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::TopNNode>(planNode) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto topNPlanNode =
        std::dynamic_pointer_cast<const core::TopNNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(std::make_unique<CudfTopN>(operatorId, ctx, topNPlanNode));
    return result;
  }

  std::string name() const override {
    return "TopN";
  }
};

/// LimitAdapter - Replaces with CudfLimit
class LimitAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::Limit*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::LimitNode>(planNode) !=
        nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto limitPlanNode =
        std::dynamic_pointer_cast<const core::LimitNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfLimit>(operatorId, ctx, limitPlanNode));
    return result;
  }

  std::string name() const override {
    return "Limit";
  }
};

/// LocalPartitionAdapter - Conditionally replaces with CudfLocalPartition
class LocalPartitionAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::LocalPartition*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    auto localPartitionPlanNode =
        std::dynamic_pointer_cast<const core::LocalPartitionNode>(planNode);
    return canHandle(op) && localPartitionPlanNode &&
        CudfLocalPartition::shouldReplace(localPartitionPlanNode);
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto localPartitionPlanNode =
        std::dynamic_pointer_cast<const core::LocalPartitionNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfLocalPartition>(
            operatorId, ctx, localPartitionPlanNode));
    // }
    return result;
  }

  int keepOperator() const override {
    return 0;
  }

  std::string name() const override {
    return "LocalPartition";
  }
};

/// LocalExchangeAdapter - Keeps original operator
class LocalExchangeAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::LocalExchange*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return true;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  int keepOperator() const override {
    return 1;
  }

  std::string name() const override {
    return "LocalExchange";
  }
};

/// AssignUniqueIdAdapter - Replaces with CudfAssignUniqueId
class AssignUniqueIdAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::AssignUniqueId*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(
               planNode) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto assignUniqueIdPlanNode =
        std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfAssignUniqueId>(
            operatorId,
            ctx,
            assignUniqueIdPlanNode,
            assignUniqueIdPlanNode->taskUniqueId(),
            assignUniqueIdPlanNode->uniqueIdCounter()));
    return result;
  }

  std::string name() const override {
    return "AssignUniqueId";
  }
};

/// ValuesAdapter - Keeps original operator
class ValuesAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::Values*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return false;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  int keepOperator() const override {
    return 1;
  }

  std::string name() const override {
    return "Values";
  }
};

/// CallbackSinkAdapter - Keeps original operator
class CallbackSinkAdapter : public OperatorAdapter {
 public:
  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::CallbackSink*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return false;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  int keepOperator() const override {
    return 1;
  }

  std::string name() const override {
    return "CallbackSink";
  }
};

/// Registration Function
void registerAllOperatorAdapters() {
  auto& registry = OperatorAdapterRegistry::getInstance();

  // Clear any existing adapters
  registry.clear();

  // Register all adapters
  registry.registerAdapter(std::make_unique<TableScanAdapter>());
  registry.registerAdapter(std::make_unique<FilterProjectAdapter>());
  registry.registerAdapter(std::make_unique<AggregationAdapter>());
  registry.registerAdapter(std::make_unique<HashJoinBuildAdapter>());
  registry.registerAdapter(std::make_unique<HashJoinProbeAdapter>());
  registry.registerAdapter(std::make_unique<OrderByAdapter>());
  registry.registerAdapter(std::make_unique<TopNAdapter>());
  registry.registerAdapter(std::make_unique<LimitAdapter>());
  registry.registerAdapter(std::make_unique<LocalPartitionAdapter>());
  registry.registerAdapter(std::make_unique<LocalExchangeAdapter>());
  registry.registerAdapter(std::make_unique<AssignUniqueIdAdapter>());
  registry.registerAdapter(std::make_unique<ValuesAdapter>());
  registry.registerAdapter(std::make_unique<CallbackSinkAdapter>());
}

} // namespace facebook::velox::cudf_velox
