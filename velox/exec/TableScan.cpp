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

DEFINE_bool(enable_split_preload, false, "Prefetch split metadata");

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
      planNodeId_(tableScanNode->id()),
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
          planNodeId_,
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
          for (const auto& entry : connectorStats) {
            stats_.runtimeStats[entry.first].addValue(entry.second);
          }
        }
        return nullptr;
      }

      const auto& connectorSplit = split.connectorSplit;
      currentSplitGroupId_ = split.groupId;
      needNewSplit_ = false;

      if (!connector_) {
        connector_ = connector::getConnector(connectorSplit->connectorId);
        connectorQueryCtx_ = driverCtx_->createConnectorQueryCtx(
            connectorSplit->connectorId, planNodeId_);
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
        auto prepared = *connectorSplit->dataSource->move().release();
        dataSource_->setFromDataSource(std::move(prepared));
      } else {
        dataSource_->addSplit(connectorSplit);
      }
      ++stats_.numSplits;
      setBatchSize();
    }

    const auto ioTimeStartMicros = getCurrentTimeMicro();
    auto data = dataSource_->next(readBatchSize_);
    checkPreload();
    stats().addRuntimeStat(
        "dataSourceWallNanos",
        (getCurrentTimeMicro() - ioTimeStartMicros) * 1'000);
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

    driverCtx_->task->splitFinished(planNodeId_, currentSplitGroupId_);
    currentSplitGroupId_ = -1;
    needNewSplit_ = true;
  }
}

void TableScan::preload(std::shared_ptr<connector::ConnectorSplit> split) {
  using DataSourcePtr = std::shared_ptr<connector::DataSource>;
  split->dataSource =
      std::make_shared<AsyncSource<DataSourcePtr>>([type = outputType_,
                                                    table = tableHandle_,
                                                    columns = columnHandles_,
                                                    connector = connector_,
                                                    ctx = connectorQueryCtx_,
                                                    split]() {
        auto value = std::make_unique<DataSourcePtr>();
        *value = connector->createDataSource(type, table, columns, ctx.get());
        (*value)->addSplit(split);
        return value;
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
