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

#pragma once

#include "velox/core/PlanNode.h"
#include "velox/core/TableWriteTraits.h"
#include "velox/exec/ColumnStatsCollector.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

class TableWriter : public Operator {
 public:
  TableWriter(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::TableWriteNodePtr& tableWriteNode);

  BlockingReason isBlocked(ContinueFuture* future) override;

  void initialize() override;

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  virtual bool needsInput() const override {
    return true;
  }

  void close() override;

  RowVectorPtr getOutput() override;

  bool isFinished() override {
    return finished_;
  }

  /// NOTE: we don't reclaim memory from table write operator directly but from
  /// its paired connector pool which reclaims memory from the file writers
  /// created inside the connector.
  bool canReclaim() const override {
    return false;
  }

  OperatorStats stats(bool clear) override {
    auto stats = Operator::stats(clear);
    // NOTE: file writers allocates memory through 'connectorPool_', not from
    // the table writer operator pool. So we report the memory usage from
    // 'connectorPool_'.
    stats.memoryStats = MemoryStats::memStatsFromPool(connectorPool_);

    if (FOLLY_LIKELY(dataSink_ != nullptr)) {
      const auto connectorStats = dataSink_->runtimeStats();
      for (const auto& [name, counter] : connectorStats) {
        stats.runtimeStats[name] = RuntimeMetric(counter.value, counter.unit);
      }
    }

    return stats;
  }

  /// The name of runtime stats specific to table writer.
  /// The running wall time of a writer operator from creation to close.
  static inline const std::string kRunningWallNanos{"runningWallNanos"};
  /// The number of files written by this writer operator.
  static inline const std::string kNumWrittenFiles{"numWrittenFiles"};
  /// The file write IO walltime.
  static inline const std::string kWriteIOTime{"writeIOWallNanos"};
  /// The walltime spend on file write data recoding.
  static inline const std::string kWriteRecodeTime{"writeRecodeWallNanos"};
  /// The walltime spent on file write data compression.
  static inline const std::string kWriteCompressionTime{
      "writeCompressionWallNanos"};

 private:
  // The memory reclaimer customized for connector which interface with the
  // memory arbitrator to reclaim memory from the file writers created within
  // the connector.
  class ConnectorReclaimer : public exec::ParallelMemoryReclaimer {
   public:
    static std::unique_ptr<memory::MemoryReclaimer> create(
        const std::optional<common::SpillConfig>& spillConfig,
        DriverCtx* driverCtx,
        Operator* op);

    void enterArbitration() override {}

    void leaveArbitration() noexcept override {}

    bool reclaimableBytes(
        const memory::MemoryPool& pool,
        uint64_t& reclaimableBytes) const override;

    uint64_t reclaim(
        memory::MemoryPool* pool,
        uint64_t targetBytes,
        uint64_t maxWaitMs,
        memory::MemoryReclaimer::Stats& stats) override;

    void abort(memory::MemoryPool* pool, const std::exception_ptr& /* error */)
        override {}

    std::shared_ptr<Driver> ensureDriver() const {
      return driver_.lock();
    }

   private:
    ConnectorReclaimer(
        const std::optional<common::SpillConfig>& spillConfig,
        const std::shared_ptr<Driver>& driver,
        Operator* op)
        : ParallelMemoryReclaimer(
              spillConfig.has_value() ? spillConfig.value().executor : nullptr,
              0),
          canReclaim_(spillConfig.has_value()),
          driver_(driver),
          op_(op) {}

    const bool canReclaim_{false};
    const std::weak_ptr<Driver> driver_;
    Operator* const op_;
  };

  void createDataSink();

  bool finishDataSink();

  std::vector<std::string> closeDataSink();

  void abortDataSink();

  void updateStats(const connector::DataSink::Stats& stats);

  // Sets type mappings in `inputMapping_`, `mappedInputType_`, and
  // `mappedOutputType_`.
  void setTypeMappings(const core::TableWriteNodePtr& tableWriteNode);

  std::string createTableCommitContext(bool lastOutput);

  void setConnectorMemoryReclaimer();

  const DriverCtx* const driverCtx_;
  memory::MemoryPool* const connectorPool_;
  const connector::ConnectorInsertTableHandlePtr insertTableHandle_;
  const connector::CommitStrategy commitStrategy_;
  // Records the writer operator creation time in ns. This is used to record
  // the running wall time of a writer operator. This can help to detect the
  // slow scaled writer scheduling in Prestissimo.
  const uint64_t createTimeNs_{0};

  std::unique_ptr<ColumnStatsCollector> statsCollector_;
  std::shared_ptr<connector::Connector> connector_;
  std::shared_ptr<connector::ConnectorQueryCtx> connectorQueryCtx_;
  std::unique_ptr<connector::DataSink> dataSink_;

  // Contains the mappings between input and output columns.
  std::vector<column_index_t> inputMapping_;

  // Stores the mapped input and output types. Note that input types must have
  // the same types as the types receing in addInput(), but they may be in a
  // different order. Output type may have different types to allow the writer
  // to convert them (for example, when writing structs as flap maps).
  std::shared_ptr<const RowType> mappedInputType_;
  std::shared_ptr<const RowType> mappedOutputType_;

  // The blocking future might be set when finish data sink.
  ContinueFuture blockingFuture_{ContinueFuture::makeEmpty()};
  BlockingReason blockingReason_{BlockingReason::kNotBlocked};

  bool finished_{false};
  bool closed_{false};
  vector_size_t numWrittenRows_{0};
};

// TODO: TableWriteTraits got moved to velox/core as it pertains to plan
// metadata, not execution. Maintaining the alias here in order not to break
// backward compatibility.
using core::TableWriteTraits;

} // namespace facebook::velox::exec
