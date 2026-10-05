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

#include "velox/connectors/hive/FileSplitReaderAdapter.h"

namespace facebook::velox::connector::hive {

FileSplitReaderAdapter::FileSplitReaderAdapter(
    std::unique_ptr<FileSplitReader> reader,
    std::shared_ptr<const FileConnectorSplit> split,
    memory::MemoryPool* pool,
    std::shared_ptr<const FileScanSpec> fileScanSpec,
    std::shared_ptr<FileScanState> state)
    : fileScanSpec_(std::move(fileScanSpec)),
      state_(std::move(state)),
      reader_(std::move(reader)),
      split_(std::move(split)),
      pool_(pool) {
  VELOX_CHECK_NOT_NULL(reader_);
}

void FileSplitReaderAdapter::prepare(
    std::shared_ptr<random::RandomSkipTracker> randomSkip,
    const folly::F14FastSet<std::string>& remainingFilterColumns,
    const std::shared_ptr<common::MetadataFilter>& metadataFilter,
    const RowTypePtr& readerProducedType) {
  reader_->configureReaderOptions(std::move(randomSkip));
  reader_->setRemainingFilterColumns(remainingFilterColumns);
  reader_->prepareSplit(metadataFilter, preparationStats_);
  const auto& physicalOutputType = reader_->readerOutputType();
  outputType_ = readerProducedType ? readerProducedType : physicalOutputType;
  if (outputType_->size() < physicalOutputType->size()) {
    // The split reader may append unselected equality-delete or lineage
    // columns. Preserve extraction result types while adding those columns
    // to the batch type.
    auto names = outputType_->names();
    auto types = outputType_->children();
    for (auto i = outputType_->size(); i < physicalOutputType->size(); ++i) {
      names.push_back(physicalOutputType->nameOf(i));
      types.push_back(physicalOutputType->childAt(i));
    }
    outputType_ = ROW(std::move(names), std::move(types));
  }
}

ScanReadResult FileSplitReaderAdapter::next(
    uint64_t maxRows,
    VectorPtr& output,
    ContinueFuture& /*future*/) {
  VELOX_CHECK_GT(maxRows, 0);
  if (ended_ || reader_->emptySplit()) {
    ended_ = true;
    return {ScanReadResult::State::kEnd};
  }
  VELOX_CHECK_NOT_NULL(outputType_, "Reader has not been prepared");
  if (!output || !output->type()->equivalent(*outputType_)) {
    output = BaseVector::create(outputType_, 0, pool_);
  }
  const auto scanned = reader_->next(maxRows, output);
  ended_ = scanned == 0;
  return {
      ended_ ? ScanReadResult::State::kEnd : ScanReadResult::State::kData,
      scanned};
}

std::unordered_map<std::string, RuntimeMetric>
FileSplitReaderAdapter::getRuntimeStats() const {
  auto stats = preparationStats_;
  if (reader_) {
    reader_->updateRuntimeStats(stats);
  }
  return stats.toRuntimeMetricMap();
}

void FileSplitReaderAdapter::resetFilterCaches() {
  if (reader_) {
    reader_->resetFilterCaches();
  }
}

int64_t FileSplitReaderAdapter::estimatedRowSize() const {
  return reader_ ? reader_->estimatedRowSize() : DataSource::kUnknownRowSize;
}

bool FileSplitReaderAdapter::allPrefetchIssued() const {
  return reader_ && reader_->allPrefetchIssued();
}

void FileSplitReaderAdapter::setConnectorQueryCtx(
    const ConnectorQueryCtx* context) {
  VELOX_CHECK_NOT_NULL(reader_);
  reader_->setConnectorQueryCtx(context);
}

const FileConnectorSplit* FileSplitReaderAdapter::currentFileSplit() const {
  return split_.get();
}

void FileSplitReaderAdapter::cancel() noexcept {
  ended_ = true;
  reader_.reset();
  state_.reset();
  fileScanSpec_.reset();
  split_.reset();
}

} // namespace facebook::velox::connector::hive
