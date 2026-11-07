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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/Operator.h"

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <atomic>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::serializer::presto {
class PrestoOutputStreamListener;
}

namespace facebook::velox::cudf_velox {

/// Serializes the rows of a task into Presto SerializedPages directly from
/// the GPU table and enqueues them into the task's DefaultOutputBufferManager,
/// which consumers drain as they drain the pages of exec::PartitionedOutput.
///
/// Replaces the CudfToVelox + exec::PartitionedOutput pair at the end of a GPU
/// pipeline when the output goes to one destination buffer (a single,
/// broadcast or arbitrary output without partition keys) through the Presto
/// serde without compression, and every output column has a flat Presto
/// encoding. An output whose input is already on the CPU keeps
/// exec::PartitionedOutput. Input is gathered until it fills a page of
/// cudf.presto_output_page_bytes or kMaxPageRows rows, unless the output
/// flushes eagerly. The pages carry the CRC32 checksum when the buffer
/// manager has a Presto output stream listener, as the pages of
/// exec::PartitionedOutput do.
///
/// The pages are copied into pinned memory from cuDF's pinned memory resource
/// in batches of at most kPagesPerHostBatch pages and max_output_buffer_size
/// bytes, or one page when a page is larger, so a driver holds at most that
/// much beyond the pages in the output buffer; the runtime stat
/// peakPinnedPageBytes reports the most such bytes alive at once, the pages
/// the consumer has not freed yet included.
class CudfPrestoOutput : public CudfOperatorBase {
 public:
  /// Rows a page holds at most. Bounds the pages of narrow rows and of rows
  /// without columns, which bytes do not bound.
  static constexpr int64_t kMaxPageRows = 1 << 20;

  /// The largest cudf.presto_output_page_bytes: a page exceeds the setting by
  /// one row and must stay within the int32 size of a Presto page.
  static constexpr int64_t kMaxPageBytes =
      std::numeric_limits<int32_t>::max() / 2;

  CudfPrestoOutput(
      int32_t operatorId,
      exec::DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
      bool eagerFlush,
      const std::shared_ptr<exec::DefaultOutputBufferManager>& bufferManager);

  ~CudfPrestoOutput() override;

  /// Returns why 'node' cannot be served by this operator under the query's
  /// configuration, or nullopt when it can.
  static std::optional<std::string> cpuFallbackReason(
      const core::PartitionedOutputNode& node,
      const core::QueryCtx& queryCtx);

  /// Whether cudf.presto_output_enabled allows the operator under 'config'.
  static bool isEnabled(const core::QueryConfig& config);

  /// Throws a user error unless 'pageBytes' is a valid
  /// cudf.presto_output_page_bytes: positive and at most kMaxPageBytes. The
  /// startup configuration and the operator share it.
  static void checkPageBytes(int64_t pageBytes);

  /// Always true: the driver checks isBlocked() before adding input, so the
  /// blocked state never accumulates input.
  bool needsInput() const override {
    return true;
  }

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override {
    return finished_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;

  /// Always returns nullptr. Serializes the buffered input until every page
  /// is enqueued or the consumer blocks.
  RowVectorPtr doGetOutput() override;

  void doClose() override;

 private:
  struct Column;
  struct Serializer;
  struct TaskLists;

  // One serialized page inside a wave buffer.
  struct PageRange {
    size_t offset;
    size_t size;
    int32_t numRows;
    // The page's host fields: a range of Wave::fields.
    size_t firstField;
    size_t numFields;
  };

  // Bytes a page gets on the host once it is there: 'size' bytes of
  // Wave::fieldBytes from 'source', at 'offset' into the page.
  struct HostField {
    size_t source;
    size_t offset;
    size_t size;
  };

  // A batch of pages that share one device buffer.
  struct Wave {
    rmm::device_buffer data;
    std::vector<PageRange> pages;
    // The small fields of the pages: page headers, encoding names, position
    // counts, has-nulls bytes and string byte counts.
    std::vector<uint8_t> fieldBytes;
    std::vector<HostField> fields;
    size_t next{0};
  };

  // A page copied to pinned host memory, ready to be enqueued.
  struct HostPage {
    void* data;
    size_t size;
    int32_t numRows;
  };

  // Turns the pending inputs into a Serializer over one table.
  void startSerializer();

  // Lays out the next wave of pages of the current Serializer on the device.
  void buildNextWave();

  // Copies a batch of pages of the current wave to pinned host memory.
  void copyPagesToHost();

  // Writes the host fields of 'range' into its host page.
  void
  writeHostFields(const Wave& wave, const PageRange& range, HostPage& page);

  // Sets the codec marker and the CRC32 of a host page.
  void checksumPage(HostPage& page);

  // Enqueues the ready host pages. Returns true when the consumer blocked.
  bool enqueueReadyPages(exec::DefaultOutputBufferManager& bufferManager);

  // Runs the pipeline until everything is enqueued or the consumer blocks.
  void drain();

  void releaseHostPage(HostPage& page);

  const std::weak_ptr<exec::DefaultOutputBufferManager> bufferManager_;
  const RowTypePtr inputType_;
  const std::vector<column_index_t> outputChannels_;
  const int64_t pageBytes_;
  // Device bytes laid out per wave: a bounded number of pages.
  const int64_t waveBytes_;
  // Bytes copied to the host between two stream synchronizations: at most
  // kPagesPerHostBatch pages, and no more than the output buffer accepts
  // before it blocks; a batch always holds one page.
  const int64_t hostBatchBytes_;
  // Serialize each input as it arrives instead of filling a page first.
  const bool eagerFlush_;
  // The buffer manager's listener, and the same when it computes the Presto
  // checksum; null otherwise.
  const std::unique_ptr<OutputStreamListener> listener_;
  serializer::presto::PrestoOutputStreamListener* const checksumListener_;
  // Pinned page bytes not freed yet, shared with the release callbacks of the
  // enqueued pages, which run on the consumer's threads.
  const std::shared_ptr<std::atomic<int64_t>> livePinnedBytes_;
  rmm::host_device_async_resource_ref pinnedMr_;

  std::vector<CudfVectorPtr> pendingInputs_;
  int64_t pendingRows_{0};
  int64_t pendingBytes_{0};

  std::unique_ptr<Serializer> serializer_;
  // The task lists of the wave being laid out, reused across waves.
  std::unique_ptr<TaskLists> tasks_;
  std::optional<Wave> wave_;
  std::deque<HostPage> ready_;

  exec::BlockingReason blockingReason_{exec::BlockingReason::kNotBlocked};
  ContinueFuture future_;
  bool finished_{false};

  int64_t peakPinnedBytes_{0};
  uint64_t layoutNanos_{0};
  uint64_t copyNanos_{0};
};

} // namespace facebook::velox::cudf_velox
