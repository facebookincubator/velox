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

#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Singleton store holding preloaded table data (CPU or GPU batches).
/// When populated, the PreloadedTableScanAdapter replaces TableScan operators
/// with PreloadedScanOperator instances that serve data from this store.
class PreloadedTableStore {
 public:
  static PreloadedTableStore& getInstance();

  void store(
      const std::string& tableName,
      std::vector<RowVectorPtr> batches);

  const std::vector<RowVectorPtr>* getBatches(
      const std::string& tableName) const;

  bool isPopulated() const {
    return !tables_.empty();
  }

  /// Returns true if stored batches are GPU-resident CudfVectors.
  bool isGpuData() const;

  void clear();

 private:
  PreloadedTableStore() = default;
  std::unordered_map<std::string, std::vector<RowVectorPtr>> tables_;
};

/// SourceOperator that serves preloaded table data with column pruning.
/// Replaces TableScan when data has been pre-loaded into PreloadedTableStore.
class PreloadedScanOperator : public exec::SourceOperator {
 public:
  PreloadedScanOperator(
      int32_t operatorId,
      exec::DriverCtx* ctx,
      const std::shared_ptr<const core::TableScanNode>& scanNode);

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

  void close() override {
    exec::Operator::close();
  }

 private:
  RowVectorPtr selectColumns(const RowVectorPtr& batch) const;

  const RowTypePtr scanOutputType_;
  const std::string tableName_;
  const std::vector<RowVectorPtr>* batches_;
  std::vector<column_index_t> columnMapping_;
  size_t currentBatch_ = 0;
  bool finished_ = false;
};

/// Registers the preloaded table scan adapter, overwriting the default
/// TableScanAdapter. When PreloadedTableStore is populated, TableScan
/// operators are replaced with PreloadedScanOperator.
void registerPreloadedTableScanAdapter();

} // namespace facebook::velox::cudf_velox
