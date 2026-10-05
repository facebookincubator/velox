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

#include "velox/connectors/hive/FileScanReader.h"
#include "velox/connectors/hive/FileSplitReader.h"
#include "velox/dwio/common/Statistics.h"

namespace facebook::velox::connector::hive {

class FileScanSpec;
struct FileScanState;

/// Adapts the existing synchronous file reader and its connector-specific
/// specializations to the logical scan interface.
class FileSplitReaderAdapter final : public FileScanReader {
 public:
  FileSplitReaderAdapter(
      std::unique_ptr<FileSplitReader> reader,
      std::shared_ptr<const FileConnectorSplit> split,
      memory::MemoryPool* pool,
      std::shared_ptr<const FileScanSpec> fileScanSpec = nullptr,
      std::shared_ptr<FileScanState> state = nullptr);

  void prepare(
      std::shared_ptr<random::RandomSkipTracker> randomSkip,
      const folly::F14FastSet<std::string>& remainingFilterColumns,
      const std::shared_ptr<common::MetadataFilter>& metadataFilter,
      const RowTypePtr& readerProducedType);

  ScanReadResult
  next(uint64_t maxRows, VectorPtr& output, ContinueFuture& future) override;

  dwio::common::RuntimeStats getRuntimeStats() const override;
  void resetFilterCaches() override;
  int64_t estimatedRowSize() const override;
  bool allPrefetchIssued() const override;
  void setConnectorQueryCtx(const ConnectorQueryCtx* context) override;
  const FileConnectorSplit* currentFileSplit() const override;
  void cancel() noexcept override;

  const RowTypePtr& readerOutputType() const {
    return reader_->readerOutputType();
  }

 private:
  // Keep preparation stats at a stable address across data source takeover.
  // Some existing reader specializations retain a reference to them.
  dwio::common::RuntimeStats preparationStats_;
  // Own the immutable inputs and physical filter storage referenced by the
  // reader, including after a preloaded data source has been destroyed.
  std::shared_ptr<const FileScanSpec> fileScanSpec_;
  std::shared_ptr<FileScanState> state_;
  std::unique_ptr<FileSplitReader> reader_;
  std::shared_ptr<const FileConnectorSplit> split_;
  memory::MemoryPool* const pool_;
  RowTypePtr outputType_;
  bool ended_{false};
};

} // namespace facebook::velox::connector::hive
