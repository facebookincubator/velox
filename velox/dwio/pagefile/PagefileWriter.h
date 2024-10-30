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

#include <iterator>
#include <limits>

#include "velox/dwio/common/Writer.h"
#include "velox/dwio/common/WriterFactory.h"
#include "velox/dwio/dwrf/common/Encryption.h"
#include "velox/dwio/dwrf/writer/ColumnWriter.h"
#include "velox/dwio/dwrf/writer/FlushPolicy.h"
#include "velox/dwio/dwrf/writer/LayoutPlanner.h"
#include "velox/dwio/dwrf/writer/WriterBase.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::pagefile {

class PagefileWriter : public dwio::common::Writer {
 public:
  PagefileWriter(
      std::unique_ptr<dwio::common::FileSink> sink,
      const dwio::common::WriterOptions& options,
      std::shared_ptr<memory::MemoryPool> pool);

  ~PagefileWriter() override = default;

  void write(const VectorPtr& input) override;

  // Forces the writer to flush, does not close the writer.
  void flush() override;

  void close() override;

  void abort() override;

  /// True if we can reclaim memory from this writer by memory arbitration.
  bool canReclaim() const;

  tsan_atomic<bool>& testingNonReclaimableSection() {
    return *nonReclaimableSection_;
  }

 private:
  class MemoryReclaimer : public exec::MemoryReclaimer {
   public:
    static std::unique_ptr<memory::MemoryReclaimer> create(PagefileWriter* writer);

    bool reclaimableBytes(
        const memory::MemoryPool& pool,
        uint64_t& reclaimableBytes) const override;

    uint64_t reclaim(
        memory::MemoryPool* pool,
        uint64_t targetBytes,
        uint64_t maxWaitMs,
        memory::MemoryReclaimer::Stats& stats) override;

   private:
    explicit MemoryReclaimer(PagefileWriter* writer) : writer_(writer) {
      VELOX_CHECK_NOT_NULL(writer_);
    }

    PagefileWriter* const writer_;
  };

  // Sets the memory reclaimers for all the memory pools used by this writer.
  void setMemoryReclaimers(const std::shared_ptr<memory::MemoryPool>& pool);

  // Invoked to ensure sufficient memory to process the given size of input by
  // reserving memory from each of the leaf memory pool. This only applies if we
  // support memory reclaim on this writer. The memory reservation might trigger
  // stripe flush by memory arbitration if the query root memory pool doesn't
  // enough memory capacity.
  void ensureWriteFits(size_t appendBytes, size_t appendRows);

  // Grows the memory pool size by the specified ratio.
  bool maybeReserveMemory(double estimatedMemoryGrowthRatio);

  // Releases the unused memory reservations after we flush a stripe.
  void releaseMemory();

  // Create a new stripe. No-op if there is no data written.
  void flushInternal(bool close = false);

  std::unique_ptr<dwio::common::FileSink> sink_;
  const std::shared_ptr<const dwio::common::TypeWithId> schema_;
  const common::SpillConfig* const spillConfig_;
  // If not null, used by memory arbitration to track if this file writer is
  // under memory reclaimable section or not.
  tsan_atomic<bool>* const nonReclaimableSection_{nullptr};
  std::shared_ptr<memory::MemoryPool> pool_;

};

class PagefileWriterFactory : public dwio::common::WriterFactory {
 public:
  PagefileWriterFactory() : WriterFactory(dwio::common::FileFormat::PAGEFILE) {}

  std::unique_ptr<dwio::common::Writer> createWriter(
      std::unique_ptr<dwio::common::FileSink> sink,
      const dwio::common::WriterOptions& options) override;
};

void registerPagefileWriterFactory();

void unregisterPagefileWriterFactory();

} // namespace facebook::velox::pagefile
