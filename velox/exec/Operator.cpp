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
#include "velox/exec/Operator.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/process/ProcessBase.h"
#include "velox/exec/Driver.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::exec {
namespace {
// Basic implementation of the connector::ExpressionEvaluator interface.
class SimpleExpressionEvaluator : public connector::ExpressionEvaluator {
 public:
  explicit SimpleExpressionEvaluator(
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool)
      : queryCtx_(queryCtx), pool_(pool) {}

  std::unique_ptr<exec::ExprSet> compile(
      const core::TypedExprPtr& expression) const override {
    auto expressions = {expression};
    return std::make_unique<exec::ExprSet>(
        std::move(expressions), ensureExecCtx());
  }

  void evaluate(
      exec::ExprSet* exprSet,
      const SelectivityVector& rows,
      RowVectorPtr& input,
      VectorPtr* result) const override {
    exec::EvalCtx context(ensureExecCtx(), exprSet, input.get());

    std::vector<VectorPtr> results = {*result};
    exprSet->eval(0, 1, true, rows, context, results);

    *result = results[0];
  }

 private:
  core::ExecCtx* ensureExecCtx() const {
    if (!execCtx_) {
      execCtx_ = std::make_unique<core::ExecCtx>(pool_, queryCtx_);
    }
    return execCtx_.get();
  }

  core::QueryCtx* const queryCtx_;
  memory::MemoryPool* const pool_;
  mutable std::unique_ptr<core::ExecCtx> execCtx_;
};
} // namespace

OperatorCtx::OperatorCtx(
    DriverCtx* driverCtx,
    const core::PlanNodeId& planNodeId,
    int32_t operatorId,
    const std::string& operatorType)
    : driverCtx_(driverCtx),
      planNodeId_(planNodeId),
      operatorId_(operatorId),
      operatorType_(operatorType),
      pool_(driverCtx_->addOperatorPool(planNodeId, operatorType)) {}

core::ExecCtx* OperatorCtx::execCtx() const {
  if (!execCtx_) {
    execCtx_ = std::make_unique<core::ExecCtx>(
        pool_, driverCtx_->task->queryCtx().get());
  }
  return execCtx_.get();
}

std::shared_ptr<connector::ConnectorQueryCtx>
OperatorCtx::createConnectorQueryCtx(
    const std::string& connectorId,
    const std::string& planNodeId,
    bool forScan) const {
  return std::make_shared<connector::ConnectorQueryCtx>(
      pool_,
      forScan ? nullptr
              : driverCtx_->task->addConnectorWriterPoolLocked(
                    planNodeId_,
                    driverCtx_->pipelineId,
                    driverCtx_->driverId,
                    operatorType_,
                    connectorId),
      driverCtx_->task->queryCtx()->getConnectorConfig(connectorId),
      std::make_unique<SimpleExpressionEvaluator>(
          execCtx()->queryCtx(), execCtx()->pool()),
      driverCtx_->task->queryCtx()->allocator(),
      taskId(),
      planNodeId,
      driverCtx_->driverId);
}

std::optional<Spiller::Config> OperatorCtx::makeSpillConfig(
    Spiller::Type type) const {
  const auto& queryConfig = driverCtx_->task->queryCtx()->queryConfig();
  if (!queryConfig.spillEnabled()) {
    return std::nullopt;
  }
  if (driverCtx_->task->spillDirectory().empty()) {
    return std::nullopt;
  }
  return Spiller::Config(
      makeOperatorSpillPath(
          driverCtx_->task->spillDirectory(),
          driverCtx()->pipelineId,
          driverCtx()->driverId,
          operatorId_),
      queryConfig.maxSpillFileSize(),
      queryConfig.minSpillRunSize(),
      driverCtx_->task->queryCtx()->spillExecutor(),
      queryConfig.spillableReservationGrowthPct(),
      HashBitRange(
          queryConfig.spillStartPartitionBit(),
          queryConfig.spillStartPartitionBit() +
              queryConfig.spillPartitionBits()),
      queryConfig.maxSpillLevel(),
      queryConfig.testingSpillPct());
}

Operator::Operator(
    DriverCtx* driverCtx,
    RowTypePtr outputType,
    int32_t operatorId,
    std::string planNodeId,
    std::string operatorType)
    : operatorCtx_(std::make_unique<OperatorCtx>(
          driverCtx,
          planNodeId,
          operatorId,
          operatorType)),
      stats_(OperatorStats{
          operatorId,
          driverCtx->pipelineId,
          std::move(planNodeId),
          std::move(operatorType)}),
      outputType_(std::move(outputType)) {
  auto memoryUsageTracker = pool()->getMemoryUsageTracker();
  if (memoryUsageTracker) {
    memoryUsageTracker->setMakeMemoryCapExceededMessage(
        [&](memory::MemoryUsageTracker& tracker) {
          VELOX_DCHECK(pool()->getMemoryUsageTracker().get() == &tracker);
          std::stringstream out;
          out << "\nFailed Operator: " << this->operatorType() << "."
              << this->operatorId() << ": "
              << succinctBytes(tracker.currentBytes());
          return out.str();
        });
  }
}

std::vector<std::unique_ptr<Operator::PlanNodeTranslator>>&
Operator::translators() {
  static std::vector<std::unique_ptr<PlanNodeTranslator>> translators;
  return translators;
}

// static
std::unique_ptr<Operator> Operator::fromPlanNode(
    DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& planNode,
    std::shared_ptr<ExchangeClient> exchangeClient) {
  VELOX_CHECK_EQ(exchangeClient != nullptr, planNode->requiresExchangeClient());
  for (auto& translator : translators()) {
    std::unique_ptr<Operator> op;
    if (planNode->requiresExchangeClient()) {
      op = translator->toOperator(ctx, id, planNode, exchangeClient);
    } else {
      op = translator->toOperator(ctx, id, planNode);
    }

    if (op) {
      return op;
    }
  }
  return nullptr;
}

// static
std::unique_ptr<JoinBridge> Operator::joinBridgeFromPlanNode(
    const core::PlanNodePtr& planNode) {
  for (auto& translator : translators()) {
    auto joinBridge = translator->toJoinBridge(planNode);
    if (joinBridge) {
      return joinBridge;
    }
  }
  return nullptr;
}

// static
OperatorSupplier Operator::operatorSupplierFromPlanNode(
    const core::PlanNodePtr& planNode) {
  for (auto& translator : translators()) {
    auto supplier = translator->toOperatorSupplier(planNode);
    if (supplier) {
      return supplier;
    }
  }
  return nullptr;
}

// static
void Operator::registerOperator(
    std::unique_ptr<PlanNodeTranslator> translator) {
  translators().emplace_back(std::move(translator));
}

std::optional<uint32_t> Operator::maxDrivers(
    const core::PlanNodePtr& planNode) {
  for (auto& translator : translators()) {
    auto current = translator->maxDrivers(planNode);
    if (current) {
      return current;
    }
  }
  return std::nullopt;
}

const std::string& OperatorCtx::taskId() const {
  return driverCtx_->task->taskId();
}

static bool isSequence(
    const vector_size_t* numbers,
    vector_size_t start,
    vector_size_t end) {
  for (vector_size_t i = start; i < end; ++i) {
    if (numbers[i] != i) {
      return false;
    }
  }
  return true;
}

RowVectorPtr Operator::fillOutput(vector_size_t size, BufferPtr mapping) {
  bool wrapResults = true;
  if (size == input_->size() &&
      (!mapping || isSequence(mapping->as<vector_size_t>(), 0, size))) {
    if (isIdentityProjection_) {
      return std::move(input_);
    }
    wrapResults = false;
  }

  std::vector<VectorPtr> columns(outputType_->size());
  if (!identityProjections_.empty()) {
    auto input = input_->children();
    for (auto& projection : identityProjections_) {
      columns[projection.outputChannel] = wrapResults
          ? wrapChild(size, mapping, input[projection.inputChannel])
          : input[projection.inputChannel];
    }
  }
  for (auto& projection : resultProjections_) {
    columns[projection.outputChannel] = wrapResults
        ? wrapChild(size, mapping, results_[projection.inputChannel])
        : results_[projection.inputChannel];
  }

  return std::make_shared<RowVector>(
      operatorCtx_->pool(),
      outputType_,
      BufferPtr(nullptr),
      size,
      std::move(columns));
}

OperatorStats Operator::stats(bool clear) {
  if (!clear) {
    return *stats_.rlock();
  }

  auto lockedStats = stats_.wlock();
  OperatorStats ret{*lockedStats};
  lockedStats->clear();
  return ret;
}

void Operator::recordBlockingTime(uint64_t start, BlockingReason reason) {
  uint64_t now =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  const auto wallNanos = (now - start) * 1000;
  const auto blockReason = blockingReasonToString(reason).substr(1);

  auto lockedStats = stats_.wlock();
  lockedStats->blockedWallNanos += wallNanos;
  lockedStats->addRuntimeStat(
      fmt::format("blocked{}WallNanos", blockReason),
      RuntimeCounter(wallNanos, RuntimeCounter::Unit::kNanos));
  lockedStats->addRuntimeStat(
      fmt::format("blocked{}Times", blockReason), RuntimeCounter(1));
}

std::string Operator::toString() const {
  std::stringstream out;
  if (auto task = operatorCtx_->task()) {
    auto driverCtx = operatorCtx_->driverCtx();
    out << operatorType() << "(" << operatorId() << ")<" << task->taskId()
        << ":" << driverCtx->pipelineId << "." << driverCtx->driverId << " "
        << this;
  } else {
    out << "<Terminated, no task>";
  }
  return out.str();
}

std::vector<column_index_t> toChannels(
    const RowTypePtr& rowType,
    const std::vector<core::TypedExprPtr>& exprs) {
  std::vector<column_index_t> channels;
  channels.reserve(exprs.size());
  for (const auto& expr : exprs) {
    auto channel = exprToChannel(expr.get(), rowType);
    channels.push_back(channel);
  }
  return channels;
}

column_index_t exprToChannel(
    const core::ITypedExpr* expr,
    const TypePtr& type) {
  if (auto field = dynamic_cast<const core::FieldAccessTypedExpr*>(expr)) {
    return type->as<TypeKind::ROW>().getChildIdx(field->name());
  }
  if (dynamic_cast<const core::ConstantTypedExpr*>(expr)) {
    return kConstantChannel;
  }
  VELOX_FAIL(
      "Expression must be field access or constant, got: {}", expr->toString());
  return 0; // not reached.
}

std::vector<column_index_t> calculateOutputChannels(
    const RowTypePtr& sourceOutputType,
    const RowTypePtr& targetInputType,
    const RowTypePtr& targetOutputType) {
  // Note that targetInputType may have more columns than sourceOutputType as
  // some columns can be duplicated.
  bool identicalProjection =
      sourceOutputType->size() == targetInputType->size();
  const auto& outputNames = targetInputType->names();

  std::vector<column_index_t> outputChannels;
  outputChannels.resize(outputNames.size());
  for (auto i = 0; i < outputNames.size(); i++) {
    outputChannels[i] = sourceOutputType->getChildIdx(outputNames[i]);
    if (outputChannels[i] != i) {
      identicalProjection = false;
    }
    if (outputNames[i] != targetOutputType->nameOf(i)) {
      identicalProjection = false;
    }
  }
  if (identicalProjection) {
    outputChannels.clear();
  }
  return outputChannels;
}

void OperatorStats::addRuntimeStat(
    const std::string& name,
    const RuntimeCounter& value) {
  addOperatorRuntimeStats(name, value, runtimeStats);
}

void OperatorStats::add(const OperatorStats& other) {
  numSplits += other.numSplits;
  rawInputBytes += other.rawInputBytes;
  rawInputPositions += other.rawInputPositions;

  addInputTiming.add(other.addInputTiming);
  inputBytes += other.inputBytes;
  inputPositions += other.inputPositions;
  inputVectors += other.inputVectors;

  getOutputTiming.add(other.getOutputTiming);
  outputBytes += other.outputBytes;
  outputPositions += other.outputPositions;
  outputVectors += other.outputVectors;

  physicalWrittenBytes += other.physicalWrittenBytes;

  blockedWallNanos += other.blockedWallNanos;

  finishTiming.add(other.finishTiming);

  memoryStats.add(other.memoryStats);

  for (const auto& [name, stats] : other.runtimeStats) {
    if (UNLIKELY(runtimeStats.count(name) == 0)) {
      runtimeStats.insert(std::make_pair(name, stats));
    } else {
      runtimeStats.at(name).merge(stats);
    }
  }

  numDrivers += other.numDrivers;
  spilledBytes += other.spilledBytes;
  spilledRows += other.spilledRows;
  spilledPartitions += other.spilledPartitions;
  spilledFiles += other.spilledFiles;
}

void OperatorStats::clear() {
  numSplits = 0;
  rawInputBytes = 0;
  rawInputPositions = 0;

  addInputTiming.clear();
  inputBytes = 0;
  inputPositions = 0;

  getOutputTiming.clear();
  outputBytes = 0;
  outputPositions = 0;

  physicalWrittenBytes = 0;

  blockedWallNanos = 0;

  finishTiming.clear();

  memoryStats.clear();

  runtimeStats.clear();
}

} // namespace facebook::velox::exec
