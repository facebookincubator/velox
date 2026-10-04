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

#include "velox/connectors/hive/FileSplitReader.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::hive {

/// A logical split can produce buffered rows without scanning more physical
/// rows, or scan rows without producing output. Neither means end of split.
struct ScanReadResult {
  enum class State { kData, kBlocked, kEnd };

  State state;
  RowVectorPtr rows;
  uint64_t physicalRowsScanned{0};
};

/// Batch and ownership boundary between FileDataSource and a logical split.
/// kData owns a non-null RowVector (possibly empty); kBlocked supplies a valid
/// future; kEnd means that all input and buffered output have been consumed.
/// Readers retain physical resources until the consumer advances the reader.
class FileScanReader {
 public:
  virtual ~FileScanReader() = default;

  virtual ScanReadResult next(uint64_t maxRows, ContinueFuture& future) = 0;
  virtual void resetFilterCaches() = 0;
  virtual void addDynamicFilter(
      column_index_t /*channel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) {
    resetFilterCaches();
  }
  virtual void updateRuntimeStats(dwio::common::RuntimeStats& stats) const = 0;
  virtual int64_t estimatedRowSize() const = 0;
  virtual bool allPrefetchIssued() const = 0;
  virtual void resetSplit() = 0;
  virtual void cancel() = 0;
  virtual void setConnectorQueryCtx(const ConnectorQueryCtx* ctx) = 0;

  /// File that produced the last batch, when it came from one physical file.
  /// A multi-file merge must return nullptr rather than name an arbitrary file.
  virtual const FileConnectorSplit* currentFileSplit() const = 0;
};

/// Default adapter for the existing Hive / Iceberg physical reader factories.
class FileSplitReaderAdapter final : public FileScanReader {
 public:
  FileSplitReaderAdapter(
      std::unique_ptr<FileSplitReader> reader,
      RowTypePtr outputType,
      memory::MemoryPool* pool)
      : reader_(std::move(reader)),
        outputType_(std::move(outputType)),
        pool_(pool) {}

  ScanReadResult next(uint64_t maxRows, ContinueFuture& future) override;
  void resetFilterCaches() override;
  void updateRuntimeStats(dwio::common::RuntimeStats& stats) const override;
  int64_t estimatedRowSize() const override;
  bool allPrefetchIssued() const override;
  void resetSplit() override;
  void cancel() override;
  void setConnectorQueryCtx(const ConnectorQueryCtx* ctx) override;
  const FileConnectorSplit* currentFileSplit() const override;

 private:
  std::unique_ptr<FileSplitReader> reader_;
  const RowTypePtr outputType_;
  memory::MemoryPool* const pool_;
  VectorPtr output_;
  bool active_{true};
};

} // namespace facebook::velox::connector::hive
