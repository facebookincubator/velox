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
#include "velox/exec/Exchange.h"
#include "velox/exec/tests/utils/Cursor.h"
#include "velox/experimental/query/ExecutablePlan.h"
#include "velox/experimental/query/LocalSchema.h"
#include "velox/experimental/query/SchemaSource.h"

namespace facebook::velox::exec {

/// Iterator for obtaining splits for a scan. One is created for each table
/// scan.
class SplitSource {
 public:
  /// Returns a split for 'worker'. This may implement soft affinity or strict
  /// bucket to worker mapping.
  virtual Split next(int32_t worker) = 0;
};

/// A factory for getting a SplitSource for each TableScan. The splits produced
/// may depend on partition keys, buckets etc mentioned by each tableScan.
class SplitSourceFactory {
 public:
  /// Returns a splitSource for one TableScan across all Tasks of
  /// the fragment. The source will be invoked to produce splits for
  /// each individual worker runnin the scan.
  virtual std::unique_ptr<SplitSource> splitSourceForScan(
      const core::TableScanNode& scan) = 0;
};

class LocalRunner : public std::enable_shared_from_this<LocalRunner> {
 public:
  LocalRunner(
      std::vector<ExecutableFragment> plan,
      std::shared_ptr<core::QueryCtx> queryCtx,
      SplitSourceFactory* splitSourceFactory,
      ExecutablePlanOptions options)
      : plan_(std::move(plan)),
        splitSourceFactory_(splitSourceFactory),
        options_(options) {
    params_.queryCtx = queryCtx;
  }

  test::TaskCursor* cursor();

  std::vector<TaskStats> stats() const;

 private:
  // Propagates 'error_' to 'stages_' and 'cursor_' if set
  void terminate();

  // Serializes 'cursor_' and 'error_'.
  std::mutex mutex_;
  std::vector<std::shared_ptr<RemoteConnectorSplit>> makeStages();
  test::CursorParameters params_;
  std::vector<ExecutableFragment> plan_;
  SplitSourceFactory* splitSourceFactory_;
  ExecutablePlanOptions options_;
  std::unique_ptr<test::TaskCursor> cursor_;
  std::vector<std::vector<std::shared_ptr<Task>>> stages_;
  std::exception_ptr error_;
  bool tasksCreated_{false};
};
class LocalSplitSource : public SplitSource {
 public:
  LocalSplitSource(const verax::LocalTable* table, int32_t splitsPerFile)
      : table_(table), splitsPerFile_(splitsPerFile) {}

  Split next(int32_t worker) override;

 private:
  std::mutex mutex_;
  const verax::LocalTable* table_;
  std::vector<std::shared_ptr<connector::ConnectorSplit>> fileSplits_;
  const int32_t splitsPerFile_;
  int32_t currentFile_{-1};
  int32_t currentSplit_{0};
};

class LocalSplitSourceFactory : public SplitSourceFactory {
 public:
  LocalSplitSourceFactory(verax::LocalSchema& schema, int32_t splitsPerFile)
      : schema_(schema), splitsPerFile_(splitsPerFile) {}

  std::unique_ptr<SplitSource> splitSourceForScan(
      const core::TableScanNode& scan) override;

 private:
  verax::LocalSchema& schema_;
  const int32_t splitsPerFile_;
};

} // namespace facebook::velox::exec
