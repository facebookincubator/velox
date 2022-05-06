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
#include "velox/exec/TableScan.h"
#include "velox/common/time/Timer.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

DEFINE_bool(enable_split_preload, true, "Prefetch split metadata");

namespace facebook::velox::exec {

TableScan::TableScan(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::TableScanNode> tableScanNode)
    : SourceOperator(
          driverCtx,
          tableScanNode->outputType(),
          operatorId,
          tableScanNode->id(),
          "TableScan"),
      tableHandle_(tableScanNode->tableHandle()),
      columnHandles_(tableScanNode->assignments()),
      driverCtx_(driverCtx),
      blockingFuture_(false) {}

RowVectorPtr TableScan::getOutput() {
  if (noMoreSplits_) {
    return nullptr;
  }

  for (;;) {
    if (needNewSplit_) {
      exec::Split split;
      auto reason = driverCtx_->task->getSplitOrFuture(
          driverCtx_->splitGroupId,
          planNodeId(),
          split,
          blockingFuture_,
          maxPreloadedSplits_,
          splitPreloader_);
      if (reason != BlockingReason::kNotBlocked) {
        return nullptr;
      }

      if (!split.hasConnectorSplit()) {
        noMoreSplits_ = true;
        if (dataSource_) {
          auto connectorStats = dataSource_->runtimeStats();
          for (const auto& [name, counter] : connectorStats) {
            if (UNLIKELY(stats_.runtimeStats.count(name) == 0)) {
              stats_.runtimeStats.insert(
                  std::make_pair(name, RuntimeMetric(counter.unit)));
            } else {
              VELOX_CHECK_EQ(stats_.runtimeStats.at(name).unit, counter.unit);
            }
            stats_.runtimeStats.at(name).addValue(counter.value);
          }
        }
        return nullptr;
      }

      const auto& connectorSplit = split.connectorSplit;
      needNewSplit_ = false;

      if (!connector_) {
        connector_ = connector::getConnector(connectorSplit->connectorId);
        connectorQueryCtx_ = operatorCtx_->createConnectorQueryCtx(
            connectorSplit->connectorId, planNodeId());
        dataSource_ = connector_->createDataSource(
            outputType_,
            tableHandle_,
            columnHandles_,
            connectorQueryCtx_.get());
        for (const auto& entry : pendingDynamicFilters_) {
          dataSource_->addDynamicFilter(entry.first, entry.second);
        }
        pendingDynamicFilters_.clear();
      } else {
        VELOX_CHECK(
            connector_->connectorId() == connectorSplit->connectorId,
            "Got splits with different connector IDs");
      }

      if (connectorSplit->dataSource) {
        // The AsyncSource returns a unique_ptr to a shared_ptr. The
        // unique_ptr will be nullptr if there was a cancellation.
        auto preparedPtr = connectorSplit->dataSource->move();
        if (!preparedPtr) {
          // There must be a cancellation.
          VELOX_CHECK(operatorCtx_->task()->isCancelled());
          return nullptr;
        }
        auto preparedDataSource = std::move(*preparedPtr);
        dataSource_->setFromDataSource(std::move(preparedDataSource));
      } else {
        dataSource_->addSplit(connectorSplit);
      }
      ++stats_.numSplits;
      setBatchSize();
    }

    const auto ioTimeStartMicros = getCurrentTimeMicro();
    // Check for  cancellation since scans that filter everything out will not
    // hit the check in Driver.
    if (operatorCtx_->task()->isCancelled()) {
      return nullptr;
    }
    auto data = dataSource_->next(readBatchSize_);
    checkPreload();
    stats().addRuntimeStat(
        "dataSourceWallNanos",
        RuntimeCounter(
            (getCurrentTimeMicro() - ioTimeStartMicros) * 1'000,
            RuntimeCounter::Unit::kNanos));
    stats_.rawInputPositions = dataSource_->getCompletedRows();
    stats_.rawInputBytes = dataSource_->getCompletedBytes();
    if (data) {
      if (data->size() > 0) {
        stats_.inputPositions += data->size();
        stats_.inputBytes += data->retainedSize();
        return data;
      }
      continue;
    }

    driverCtx_->task->splitFinished();
    needNewSplit_ = true;
  }
}

void TableScan::preload(std::shared_ptr<connector::ConnectorSplit> split) {
  // The AsyncSource returns a unique_ptr to the shared_ptr of the
  // DataSource. The callback may outlive the Task, hence it captures
  // a shared_ptr to it. This is required to keep memory pools live
  // for the duration. The callback checks for task cancellation to
  // avoid needless work.
  using DataSourcePtr = std::shared_ptr<connector::DataSource>;
  split->dataSource = std::make_shared<AsyncSource<DataSourcePtr>>(
      [type = outputType_,
       table = tableHandle_,
       columns = columnHandles_,
       connector = connector_,
       ctx = connectorQueryCtx_,
       task = operatorCtx_->task(),
       split]() -> std::unique_ptr<DataSourcePtr> {
        if (task->isCancelled()) {
          return nullptr;
        }
        auto ptr = std::make_unique<DataSourcePtr>();
        *ptr = connector->createDataSource(type, table, columns, ctx.get());
        if (task->isCancelled()) {
          return nullptr;
        }
        (*ptr)->addSplit(split);
        return ptr;
      });
}

void TableScan::checkPreload() {
  auto executor = connector_->executor();
  if (!FLAGS_enable_split_preload || !executor ||
      !connector_->supportsSplitPreload()) {
    return;
  }
  if (dataSource_->allPrefetchIssued()) {
    maxPreloadedSplits_ = driverCtx_->task->numDrivers(driverCtx_->driver);
    if (!splitPreloader_) {
      splitPreloader_ =
          [executor, this](std::shared_ptr<connector::ConnectorSplit> split) {
            preload(split);
            executor->add([split]() { split->dataSource->prepare(); });
          };
    }
  }
}

bool TableScan::isFinished() {
  return noMoreSplits_;
}

void TableScan::setBatchSize() {
  constexpr int64_t kMB = 1 << 20;
  auto estimate = dataSource_->estimatedRowSize();
  if (estimate == connector::DataSource::kUnknownRowSize) {
    readBatchSize_ = kDefaultBatchSize;
    return;
  }
  if (estimate < 1024) {
    readBatchSize_ = 10000; // No more than 10MB of data per batch.
    return;
  }
  readBatchSize_ = std::min<int64_t>(100, 10 * kMB / estimate);
}

void TableScan::addDynamicFilter(
    ChannelIndex outputChannel,
    const std::shared_ptr<common::Filter>& filter) {
  if (dataSource_) {
    dataSource_->addDynamicFilter(outputChannel, filter);
  } else {
    pendingDynamicFilters_.emplace(outputChannel, filter);
  }
}

} // namespace facebook::velox::exec
