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
#include "velox/connectors/hive/FileScanReader.h"

namespace facebook::velox::connector::hive {

ScanReadResult FileSplitReaderAdapter::next(
    uint64_t maxRows,
    ContinueFuture& /*future*/) {
  if (reader_->emptySplit()) {
    return {ScanReadResult::State::kEnd, nullptr};
  }
  if (!output_) {
    output_ = BaseVector::create(outputType_, 0, pool_);
  }
  const auto scanned = reader_->next(maxRows, output_);
  if (scanned == 0) {
    return {ScanReadResult::State::kEnd, nullptr};
  }
  return {
      ScanReadResult::State::kData,
      std::static_pointer_cast<RowVector>(output_),
      scanned};
}

void FileSplitReaderAdapter::resetFilterCaches() {
  if (reader_) {
    reader_->resetFilterCaches();
  }
}

void FileSplitReaderAdapter::updateRuntimeStats(
    dwio::common::RuntimeStats& stats) const {
  if (active_ && reader_) {
    reader_->updateRuntimeStats(stats);
  }
}

int64_t FileSplitReaderAdapter::estimatedRowSize() const {
  return reader_ ? reader_->estimatedRowSize() : DataSource::kUnknownRowSize;
}

bool FileSplitReaderAdapter::allPrefetchIssued() const {
  return reader_ && reader_->allPrefetchIssued();
}

void FileSplitReaderAdapter::resetSplit() {
  active_ = false;
  reader_->resetSplit();
}

void FileSplitReaderAdapter::cancel() {
  active_ = false;
  output_.reset();
  reader_.reset();
}

void FileSplitReaderAdapter::setConnectorQueryCtx(
    const ConnectorQueryCtx* ctx) {
  reader_->setConnectorQueryCtx(ctx);
}

const FileConnectorSplit* FileSplitReaderAdapter::currentFileSplit() const {
  return reader_ ? reader_->fileSplit().get() : nullptr;
}

} // namespace facebook::velox::connector::hive
