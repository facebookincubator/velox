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

#include "velox/dwio/common/Writer.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/SortBuffer.h"

namespace facebook::velox::connector::hive {

/// Sorting Writer object is used to write sorted data into a single file.
class SortingWriter : public dwio::common::Writer {
 public:
  /// The list of runtime stats reported by the sort writer when it dedupes
  /// rows.
  static constexpr const char* kSortingWriterDedupedRows =
      "sortWriterDedupedRows";
  static constexpr const char* kSortingWriterDedupeNanos =
      "sortWriterDedupeNanos";

  SortingWriter(
      std::unique_ptr<dwio::common::Writer> writer,
      std::unique_ptr<exec::SortBuffer> sortBuffer,
      std::vector<column_index_t> dedupeKeyChannels,
      vector_size_t maxOutputRowsConfig,
      uint64_t maxOutputBytesConfig,
      uint64_t outputTimeSliceLimitMs);

  ~SortingWriter() override;

  void write(const VectorPtr& data) override;

  bool finish() override;

  /// No action because we need to accumulate all data and sort before data can
  /// be flushed.
  void flush() override;

  /// Closes the writer. Returns file metadata, or null if no metadata is
  /// available (e.g. for an empty file).
  std::unique_ptr<dwio::common::FileMetadata> close() override;

  /// Returns the stats of the underlying writer plus the dedupe stats when
  /// dedupe is enabled.
  folly::F14FastMap<std::string, RuntimeMetric> runtimeStats() const override;

  uint64_t numDuplicateRows() const override {
    return dedupedRows_;
  }

  void abort() override;

 private:
  class MemoryReclaimer : public exec::MemoryReclaimer {
   public:
    static std::unique_ptr<memory::MemoryReclaimer> create(
        SortingWriter* writer);

    bool reclaimableBytes(
        const memory::MemoryPool& pool,
        uint64_t& reclaimableBytes) const override;

    uint64_t reclaim(
        memory::MemoryPool* pool,
        uint64_t targetBytes,
        uint64_t maxWaitMs,
        memory::MemoryReclaimer::Stats& stats) override;

   private:
    MemoryReclaimer(SortingWriter* writer)
        : exec::MemoryReclaimer(0),
          writer_(writer),
          canReclaim_(writer_->sortBuffer_->canSpill()) {}

    SortingWriter* const writer_;
    const bool canReclaim_;
  };

  bool canReclaim() const;

  uint64_t reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats);

  vector_size_t outputBatchRows();

  // Keeps the first row of each run of equal keys as a dictionary over 'batch'.
  // Returns 'batch' as is when there is no dedupe key or no duplicate row.
  RowVectorPtr dedupe(const RowVectorPtr& batch);

  const std::unique_ptr<dwio::common::Writer> outputWriter_;
  const vector_size_t maxOutputRowsConfig_;
  const uint64_t maxOutputBytesConfig_;
  const uint64_t finishTimeSliceLimitMs_;
  memory::MemoryPool* const sortPool_;
  const bool canReclaim_;
  // Leading sort columns that form the dedupe key. Empty disables the dedupe.
  const std::vector<column_index_t> dedupeKeyChannels_;

  std::unique_ptr<exec::SortBuffer> sortBuffer_;

  // Dedupe state: last key of the previous output batch and counters.
  std::vector<VectorPtr> lastKeys_;
  uint64_t dedupedRows_{0};
  uint64_t dedupeNs_{0};
};

} // namespace facebook::velox::connector::hive
