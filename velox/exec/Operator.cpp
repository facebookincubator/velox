/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/common/base/Counters.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/Driver.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::exec {

OperatorCtx::OperatorCtx(
    DriverCtx* driverCtx,
    const core::PlanNodeId& planNodeId,
    int32_t operatorId,
    std::string_view operatorType)
    : driverCtx_(driverCtx),
      planNodeId_(planNodeId),
      operatorId_(operatorId),
      operatorType_(operatorType),
      pool_(driverCtx_->addOperatorPool(planNodeId, operatorType_)) {}

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
    memory::MemoryPool* connectorPool,
    const common::SpillConfig* spillConfig) const {
  const auto& task = driverCtx_->task;
  auto connectorQueryCtx = std::make_shared<connector::ConnectorQueryCtx>(
      pool_,
      connectorPool,
      task->queryCtx()->connectorSessionProperties(connectorId),
      spillConfig,
      driverCtx_->prefixSortConfig(),
      std::make_unique<SimpleExpressionEvaluator>(
          execCtx()->queryCtx(), execCtx()->pool()),
      task->queryCtx()->cache(),
      task->queryCtx()->queryId(),
      taskId(),
      planNodeId,
      driverCtx_->driverId,
      driverCtx_->queryConfig().sessionTimezone(),
      driverCtx_->queryConfig().adjustTimestampToTimezone(),
      task->getCancellationToken(),
      task->queryCtx()->fsTokenProvider());
  connectorQueryCtx->setSelectiveNimbleReaderEnabled(
      driverCtx_->queryConfig().selectiveNimbleReaderEnabled());
  connectorQueryCtx->setRowSizeTrackingMode(
      driverCtx_->queryConfig().rowSizeTrackingMode());
  return connectorQueryCtx;
}

Operator::Operator(
    DriverCtx* driverCtx,
    RowTypePtr outputType,
    int32_t operatorId,
    std::string planNodeId,
    std::string_view operatorType,
    std::optional<common::SpillConfig> spillConfig)
    : operatorCtx_(
          std::make_unique<OperatorCtx>(
              driverCtx,
              planNodeId,
              operatorId,
              operatorType)),
      outputType_(std::move(outputType)),
      spillConfig_(std::move(spillConfig)),
      dryRun_(
          operatorCtx_->driverCtx()->traceCtx() &&
          operatorCtx_->driverCtx()->traceCtx()->dryRun()),
      stats_(
          OperatorStats{
              operatorId,
              driverCtx->pipelineId,
              std::move(planNodeId),
              std::string(operatorType)}) {}

void Operator::maybeSetReclaimer() {
  VELOX_CHECK_NULL(pool()->reclaimer());

  if (pool()->parent()->reclaimer() == nullptr) {
    return;
  }
  pool()->setReclaimer(
      Operator::MemoryReclaimer::create(operatorCtx_->driverCtx(), this));
}

void Operator::maybeSetTracer() {
  const auto* traceCtx = operatorCtx_->driverCtx()->traceCtx();

  if (traceCtx && traceCtx->shouldTrace(*this)) {
    if (dynamic_cast<SourceOperator*>(this) != nullptr) {
      splitTracer_ = traceCtx->createSplitTracer(*this);
    } else {
      inputTracer_ = traceCtx->createInputTracer(*this);
    }
  }
}

bool Operator::traceInput(const RowVectorPtr& input, ContinueFuture* future) {
  if (FOLLY_UNLIKELY(inputTracer_ != nullptr)) {
    return inputTracer_->write(input, future);
  }
  return false;
}

void Operator::finishTrace() {
  VELOX_CHECK(inputTracer_ == nullptr || splitTracer_ == nullptr);
  if (inputTracer_ != nullptr) {
    inputTracer_->finish();
  }

  if (splitTracer_ != nullptr) {
    splitTracer_->finish();
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

void Operator::initialize() {
  VELOX_CHECK(!initialized_);
  VELOX_CHECK_EQ(
      pool()->usedBytes(),
      0,
      "Unexpected memory usage from pool {} before operator init",
      pool()->name());
  initialized_ = true;
  maybeSetReclaimer();
  maybeSetTracer();
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

// static
void Operator::unregisterAllOperators() {
  translators().clear();
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

RowVectorPtr Operator::fillOutput(
    vector_size_t size,
    const BufferPtr& mapping,
    const std::vector<VectorPtr>& results) {
  bool wrapResults = true;
  if (size == input_->size() &&
      (!mapping || isSequence(mapping->as<vector_size_t>(), 0, size))) {
    if (isIdentityProjection_) {
      return std::move(input_);
    }
    wrapResults = false;
  }

  std::vector<VectorPtr> projectedChildren(outputType_->size());
  projectChildren(
      projectedChildren,
      input_,
      identityProjections_,
      size,
      wrapResults ? mapping : nullptr);
  projectChildren(
      projectedChildren,
      results,
      resultProjections_,
      size,
      wrapResults ? mapping : nullptr);

  return std::make_shared<RowVector>(
      operatorCtx_->pool(),
      outputType_,
      nullptr,
      size,
      std::move(projectedChildren));
}

RowVectorPtr Operator::fillOutput(
    vector_size_t size,
    const BufferPtr& mapping) {
  return fillOutput(size, mapping, results_);
}

OperatorStats Operator::stats(bool clear) {
  OperatorStats stats;
  if (!clear) {
    stats = *stats_.rlock();
  } else {
    auto lockedStats = stats_.wlock();
    stats = *lockedStats;
    lockedStats->clear();
  }

  stats.memoryStats = MemoryStats::memStatsFromPool(pool());
  return stats;
}

void Operator::close() {
  input_ = nullptr;
  results_.clear();
  recordSpillStats();
  finishTrace();

  // Release the unused memory reservation on close.
  operatorCtx_->pool()->release();
}

vector_size_t Operator::outputBatchRows(
    std::optional<uint64_t> averageRowSize) const {
  const auto& queryConfig = operatorCtx_->task()->queryCtx()->queryConfig();
  if (!averageRowSize.has_value()) {
    return queryConfig.preferredOutputBatchRows();
  }

  if (averageRowSize.value() == 0) {
    return queryConfig.maxOutputBatchRows();
  }

  const uint64_t batchSize =
      queryConfig.preferredOutputBatchBytes() / averageRowSize.value();
  if (batchSize > queryConfig.maxOutputBatchRows()) {
    return queryConfig.maxOutputBatchRows();
  }
  return std::max<vector_size_t>(batchSize, 1);
}

void Operator::loadLazyReclaimable(RowVectorPtr& vector) {
  ReclaimableSectionGuard guard(this);
  vector->loadedVector();
}

void Operator::recordBlockingTime(uint64_t start, BlockingReason reason) {
  uint64_t now =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  const auto wallNanos = (now - start) * 1000;
  const auto blockReason = BlockingReasonName::toName(reason).substr(1);

  auto lockedStats = stats_.wlock();
  lockedStats->blockedWallNanos += wallNanos;
  lockedStats->addRuntimeStat(
      fmt::format("blocked{}WallNanos", blockReason),
      RuntimeCounter(wallNanos, RuntimeCounter::Unit::kNanos));
  lockedStats->addRuntimeStat(
      fmt::format("blocked{}Times", blockReason), RuntimeCounter(1));
}

void Operator::recordSpillStats() {
  auto lockedStats = stats_.wlock();
  lockedStats->spilledInputBytes +=
      spillStats_->spilledInputBytes.load(std::memory_order_relaxed);
  lockedStats->spilledBytes +=
      spillStats_->spilledBytes.load(std::memory_order_relaxed);
  lockedStats->spilledRows +=
      spillStats_->spilledRows.load(std::memory_order_relaxed);
  lockedStats->spilledPartitions +=
      spillStats_->spilledPartitions.load(std::memory_order_relaxed);
  lockedStats->spilledFiles +=
      spillStats_->spilledFiles.load(std::memory_order_relaxed);

  const auto fillTime =
      spillStats_->spillFillTimeNanos.load(std::memory_order_relaxed);
  if (fillTime != 0) {
    lockedStats->addRuntimeStat(
        kSpillFillTime,
        RuntimeCounter{
            static_cast<int64_t>(fillTime), RuntimeCounter::Unit::kNanos});
  }
  const auto sortTime =
      spillStats_->spillSortTimeNanos.load(std::memory_order_relaxed);
  if (sortTime != 0) {
    lockedStats->addRuntimeStat(
        kSpillSortTime,
        RuntimeCounter{
            static_cast<int64_t>(sortTime), RuntimeCounter::Unit::kNanos});
  }
  const auto extractTime =
      spillStats_->spillExtractVectorTimeNanos.load(std::memory_order_relaxed);
  if (extractTime != 0) {
    lockedStats->addRuntimeStat(
        kSpillExtractVectorTime,
        RuntimeCounter{
            static_cast<int64_t>(extractTime), RuntimeCounter::Unit::kNanos});
  }
  const auto serializationTime =
      spillStats_->spillSerializationTimeNanos.load(std::memory_order_relaxed);
  if (serializationTime != 0) {
    lockedStats->addRuntimeStat(
        kSpillSerializationTime,
        RuntimeCounter{
            static_cast<int64_t>(serializationTime),
            RuntimeCounter::Unit::kNanos});
  }
  const auto flushTime =
      spillStats_->spillFlushTimeNanos.load(std::memory_order_relaxed);
  if (flushTime != 0) {
    lockedStats->addRuntimeStat(
        kSpillFlushTime,
        RuntimeCounter{
            static_cast<int64_t>(flushTime), RuntimeCounter::Unit::kNanos});
  }
  const auto writes = spillStats_->spillWrites.load(std::memory_order_relaxed);
  if (writes != 0) {
    lockedStats->addRuntimeStat(
        kSpillWrites, RuntimeCounter{static_cast<int64_t>(writes)});
  }
  const auto writeTime =
      spillStats_->spillWriteTimeNanos.load(std::memory_order_relaxed);
  if (writeTime != 0) {
    lockedStats->addRuntimeStat(
        kSpillWriteTime,
        RuntimeCounter{
            static_cast<int64_t>(writeTime), RuntimeCounter::Unit::kNanos});
  }
  const auto runs = spillStats_->spillRuns.load(std::memory_order_relaxed);
  if (runs != 0) {
    lockedStats->addRuntimeStat(
        kSpillRuns, RuntimeCounter{static_cast<int64_t>(runs)});
    updateGlobalSpillRunStats(runs);
  }

  const auto maxLevelExceeded =
      spillStats_->spillMaxLevelExceededCount.load(std::memory_order_relaxed);
  if (maxLevelExceeded != 0) {
    lockedStats->addRuntimeStat(
        kExceededMaxSpillLevel,
        RuntimeCounter{static_cast<int64_t>(maxLevelExceeded)});
    updateGlobalMaxSpillLevelExceededCount(maxLevelExceeded);
  }

  const auto readBytes =
      spillStats_->spillReadBytes.load(std::memory_order_relaxed);
  if (readBytes != 0) {
    lockedStats->addRuntimeStat(
        kSpillReadBytes,
        RuntimeCounter{
            static_cast<int64_t>(readBytes), RuntimeCounter::Unit::kBytes});
  }

  const auto reads = spillStats_->spillReads.load(std::memory_order_relaxed);
  if (reads != 0) {
    lockedStats->addRuntimeStat(
        kSpillReads, RuntimeCounter{static_cast<int64_t>(reads)});
  }

  const auto readTime =
      spillStats_->spillReadTimeNanos.load(std::memory_order_relaxed);
  if (readTime != 0) {
    lockedStats->addRuntimeStat(
        kSpillReadTime,
        RuntimeCounter{
            static_cast<int64_t>(readTime), RuntimeCounter::Unit::kNanos});
  }

  const auto deserializationTime =
      spillStats_->spillDeserializationTimeNanos.load(
          std::memory_order_relaxed);
  if (deserializationTime != 0) {
    lockedStats->addRuntimeStat(
        kSpillDeserializationTime,
        RuntimeCounter{
            static_cast<int64_t>(deserializationTime),
            RuntimeCounter::Unit::kNanos});
  }

  // Collect filesystem I/O stats for spilling.
  const auto ioStatsMap = spillStats_->ioStats.stats();
  for (const auto& [statName, statValue] : ioStatsMap) {
    lockedStats->addRuntimeStat(
        statName, RuntimeCounter(statValue.sum, statValue.unit));
  }

  spillStats_->reset();
}

std::string Operator::toString() const {
  std::stringstream out;
  out << operatorType() << "[" << planNodeId() << "] " << operatorId();
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
  VELOX_UNREACHABLE(
      "Expression must be field access or constant, got: {}", expr->toString());
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

  isBlockedTiming.add(other.isBlockedTiming);

  backgroundTiming.add(other.backgroundTiming);

  memoryStats.add(other.memoryStats);

  for (const auto& [name, stats] : other.runtimeStats) {
    if (UNLIKELY(runtimeStats.count(name) == 0)) {
      runtimeStats.insert(std::make_pair(name, stats));
    } else {
      runtimeStats.at(name).merge(stats);
    }
  }

  for (const auto& [name, exprStats] : other.expressionStats) {
    if (UNLIKELY(expressionStats.count(name) == 0)) {
      expressionStats.insert(std::make_pair(name, exprStats));
    } else {
      expressionStats.at(name).add(exprStats);
    }
  }

  numDrivers += other.numDrivers;
  spilledInputBytes += other.spilledInputBytes;
  spilledBytes += other.spilledBytes;
  spilledRows += other.spilledRows;
  spilledPartitions += other.spilledPartitions;
  spilledFiles += other.spilledFiles;

  numNullKeys += other.numNullKeys;

  dynamicFilterStats.add(other.dynamicFilterStats);
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

  backgroundTiming.clear();

  memoryStats.clear();

  runtimeStats.clear();
  expressionStats.clear();

  numDrivers = 0;
  spilledInputBytes = 0;
  spilledBytes = 0;
  spilledRows = 0;
  spilledPartitions = 0;
  spilledFiles = 0;

  dynamicFilterStats.clear();
}

bool Operator::isDraining() const {
  return operatorCtx_->driver() != nullptr &&
      operatorCtx_->driver()->isDraining(operatorId());
}

bool Operator::hasDrained() const {
  return operatorCtx_->driver()->hasDrained(operatorId());
}

void Operator::finishDrain() {
  VELOX_CHECK(isDraining());
  operatorCtx_->driver()->finishDrain(operatorId());
  VELOX_CHECK(!isDraining());
}

bool Operator::shouldDropOutput() const {
  return operatorCtx_->driver()->shouldDropOutput(operatorId());
}

std::unique_ptr<memory::MemoryReclaimer> Operator::MemoryReclaimer::create(
    DriverCtx* driverCtx,
    Operator* op) {
  return std::unique_ptr<memory::MemoryReclaimer>(
      new Operator::MemoryReclaimer(driverCtx->driver->shared_from_this(), op));
}

void Operator::MemoryReclaimer::enterArbitration() {
  DriverThreadContext* driverThreadCtx = driverThreadContext();
  if (FOLLY_UNLIKELY(driverThreadCtx == nullptr)) {
    // Skips the driver suspension handling if this memory arbitration request
    // is not issued from a driver thread. For example, async streaming shuffle
    // and table scan prefetch execution path might initiate memory arbitration
    // request from non-driver thread.
    return;
  }

  Driver* const runningDriver = driverThreadCtx->driverCtx()->driver;
  if (!FLAGS_velox_memory_pool_capacity_transfer_across_tasks) {
    if (auto opDriver = ensureDriver()) {
      // NOTE: the current running driver might not be the driver of the
      // operator that requests memory arbitration. The reason is that an
      // operator might extend the buffer allocated from the other operator
      // either from the same or different drivers. But they must be from the
      // same task as the following check. User could set
      // FLAGS_transferred_arbitration_allowed=true to bypass this check.
      VELOX_CHECK_EQ(
          runningDriver->task()->taskId(),
          opDriver->task()->taskId(),
          "The current running driver and the request driver must be from the same task");
    }
  }
  if (runningDriver->task()->enterSuspended(runningDriver->state()) !=
      StopReason::kNone) {
    // There is no need for arbitration if the associated task has already
    // terminated.
    VELOX_FAIL("Terminate detected when entering suspension");
  }
}

void Operator::MemoryReclaimer::leaveArbitration() noexcept {
  DriverThreadContext* driverThreadCtx = driverThreadContext();
  if (FOLLY_UNLIKELY(driverThreadCtx == nullptr)) {
    // Skips the driver suspension handling if this memory arbitration request
    // is not issued from a driver thread.
    return;
  }
  Driver* const runningDriver = driverThreadCtx->driverCtx()->driver;
  if (!FLAGS_velox_memory_pool_capacity_transfer_across_tasks) {
    if (auto opDriver = ensureDriver()) {
      VELOX_CHECK_EQ(
          runningDriver->task()->taskId(),
          opDriver->task()->taskId(),
          "The current running driver and the request driver must be from the same task");
    }
  }
  runningDriver->task()->leaveSuspended(runningDriver->state());
}

bool Operator::MemoryReclaimer::reclaimableBytes(
    const memory::MemoryPool& pool,
    uint64_t& reclaimableBytes) const {
  reclaimableBytes = 0;
  std::shared_ptr<Driver> driver = ensureDriver();
  if (FOLLY_UNLIKELY(driver == nullptr)) {
    return false;
  }
  VELOX_CHECK_EQ(pool.name(), op_->pool()->name());
  return op_->reclaimableBytes(reclaimableBytes);
}

uint64_t Operator::MemoryReclaimer::reclaim(
    memory::MemoryPool* pool,
    uint64_t targetBytes,
    uint64_t /*unused*/,
    memory::MemoryReclaimer::Stats& stats) {
  std::shared_ptr<Driver> driver = ensureDriver();
  if (FOLLY_UNLIKELY(driver == nullptr)) {
    return 0;
  }
  if (!op_->canReclaim()) {
    return 0;
  }
  VELOX_CHECK_EQ(pool->name(), op_->pool()->name());
  VELOX_CHECK(
      !driver->state().isOnThread() || driver->state().suspended() ||
          driver->state().isTerminated,
      "driverOnThread {}, driverSuspended {} driverTerminated {} {}",
      driver->state().isOnThread(),
      driver->state().suspended(),
      driver->state().isTerminated,
      pool->name());
  VELOX_CHECK(driver->task()->pauseRequested());

  TestValue::adjust(
      "facebook::velox::exec::Operator::MemoryReclaimer::reclaim", pool);

  // NOTE: we can't reclaim memory from an operator which is under
  // non-reclaimable section.
  if (op_->nonReclaimableSection_) {
    // TODO: reduce the log frequency if it is too verbose.
    ++stats.numNonReclaimableAttempts;
    RECORD_METRIC_VALUE(kMetricMemoryNonReclaimableCount);
    LOG(WARNING) << "Can't reclaim from memory pool " << pool->name()
                 << " which is under non-reclaimable section, memory usage: "
                 << succinctBytes(pool->usedBytes())
                 << ", reservation: " << succinctBytes(pool->reservedBytes());
    return 0;
  }

  RuntimeStatWriterScopeGuard opStatsGuard(op_);

  return memory::MemoryReclaimer::run(
      [&]() {
        int64_t reclaimedBytes{0};
        {
          memory::ScopedReclaimedBytesRecorder recoder(pool, &reclaimedBytes);
          op_->reclaim(targetBytes, stats);
        }
        VELOX_CHECK_GE(
            reclaimedBytes,
            0,
            "Unexpected memory growth after reclaim from operator memory pool {}",
            pool->name());
        return reclaimedBytes;
      },
      stats);
}

void Operator::MemoryReclaimer::abort(
    memory::MemoryPool* pool,
    const std::exception_ptr& /* error */) {
  std::shared_ptr<Driver> driver = ensureDriver();
  if (driver == nullptr) {
    return;
  }
  VELOX_CHECK_EQ(pool->name(), op_->pool()->name());
  VELOX_CHECK(
      !driver->state().isOnThread() || driver->state().suspended() ||
      driver->state().isTerminated);
  VELOX_CHECK(driver->task()->isCancelled());
  if (driver->state().isOnThread()) {
    // We can't abort an operator if it is running on a driver thread for memory
    // arbitration. Otherwise, it might cause random crash when the driver
    // thread throws after detects the aborted query.
    return;
  }
  // Calls operator close to free up major memory usage.
  op_->close();
}
} // namespace facebook::velox::exec
