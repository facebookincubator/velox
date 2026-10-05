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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileConnectorSplit.h"

namespace facebook::velox::connector::hive {

struct ScanReadResult {
  enum class State { kData, kBlocked, kEnd };

  ScanReadResult(State state, uint64_t physicalRowsScanned = 0)
      : state(state), physicalRowsScanned(physicalRowsScanned) {}

  State state;
  // New physical progress, independent of the number of output rows, in all
  // three states. Cached output may have no new physical progress.
  uint64_t physicalRowsScanned{0};
};

/// Produces batches for one logical ConnectorSplit. A logical split need not
/// correspond to a single physical file.
class FileScanReader {
 public:
  virtual ~FileScanReader() = default;

  /// Performs bounded work. kData sets a non-null RowVector in 'output', which
  /// may be empty. kBlocked sets a valid future; empty output must not be used
  /// to poll asynchronous work. kEnd means all input and cached output have
  /// been consumed. 'output' is ignored for kBlocked and kEnd.
  ///
  /// The caller owns the reusable output slot. Readers must not retain an
  /// extra reference to it, and must respect shared vector/buffer ownership
  /// when reusing it. Lazy output follows the DWIO batch lifetime contract:
  /// consumers load it before advancing the reader. Readers that advance
  /// physical inputs internally must materialize any values they retain.
  virtual ScanReadResult
  next(uint64_t maxRows, VectorPtr& output, ContinueFuture& future) = 0;

  /// Returns a cumulative snapshot for this logical split, including active
  /// physical readers. Repeated calls must not change or consume the stats.
  /// Take the final snapshot before cancel(), which may release the counters.
  /// Shared IO counters are accounted for separately by FileDataSource.
  virtual std::unordered_map<std::string, RuntimeMetric> getRuntimeStats()
      const = 0;

  virtual void resetFilterCaches() = 0;
  virtual int64_t estimatedRowSize() const = 0;
  virtual bool allPrefetchIssued() const = 0;

  /// Rebinds the context during serial takeover of a preloaded data source.
  /// Readers that do not support takeover must reject it.
  virtual void setConnectorQueryCtx(const ConnectorQueryCtx* context) = 0;

  /// Optional, non-owning context for the current scan batch callback. Returns
  /// null when the batch cannot be attributed to one physical file.
  virtual const FileConnectorSplit* currentFileSplit() const {
    return nullptr;
  }

  /// Releases inputs and pending work after EOF, cancellation or failure.
  /// Must be idempotent and non-throwing. Destruction must also release them.
  /// FileDataSource takes the final statistics snapshot before calling this.
  virtual void cancel() noexcept = 0;
};

} // namespace facebook::velox::connector::hive
