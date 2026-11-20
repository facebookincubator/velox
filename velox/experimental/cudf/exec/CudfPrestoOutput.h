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
/// pipeline when cudf.presto_output_enabled is set, which it is not by
/// default, and the output goes to one destination buffer (a single,
/// broadcast or arbitrary output without partition keys) through the Presto
/// serde without compression, and every output column has a flat Presto
/// encoding. An output whose input is already on the CPU keeps
/// exec::PartitionedOutput, and so does one whose buffer manager has an output
/// stream listener other than the Presto one. The pages carry the CRC32
/// checksum when the buffer manager has the Presto output stream listener, as
/// the pages of exec::PartitionedOutput do.
///
/// Input is gathered until the columns of the output fill a wave of pages of
/// cudf.presto_output_page_bytes, or kMaxPageRows rows, unless the output
/// flushes eagerly. The pages are laid out on the device a wave at a time.
/// Besides the batch being serialized, the device holds the pages of one wave
/// and, while that wave is laid out, the compacted and converted values of
/// its rows.
///
/// The pages are memory of the operator's pool, as the pages of
/// exec::PartitionedOutput are, so the memory limit of the query covers them
/// until the consumer frees them. Pinned memory from cuDF's pinned memory
/// resource only stages their copy from the device, a batch of at most
/// kPagesPerHostBatch pages and max_page_partitioning_buffer_size bytes at a
/// time, or one page when a page is larger. A driver holds at most that many
/// pages beyond those in the output buffer.
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
  /// configuration and with the output stream listener of 'bufferManager', or
  /// nullopt when it can.
  static std::optional<std::string> cpuFallbackReason(
      const core::PartitionedOutputNode& node,
      const core::QueryCtx& queryCtx,
      const exec::DefaultOutputBufferManager& bufferManager);

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
    // The compacted and converted values the layout of the wave reads. Empty
    // once the layout has run.
    std::vector<rmm::device_buffer> denseValues;
    std::vector<PageRange> pages;
    // The small fields of the pages: page headers, encoding names, position
    // counts, has-nulls bytes and string byte counts.
    std::vector<uint8_t> fieldBytes;
    std::vector<HostField> fields;
    size_t next{0};
  };

  // A page in memory of the operator's pool, ready to be enqueued.
  struct HostPage {
    void* data;
    size_t size;
    int32_t numRows;
  };

  // Turns the pending inputs into a Serializer over one table.
  void startSerializer();

  // Returns the values of a fixed-width column that the pages ['firstPage',
  // 'lastPage') of the current Serializer hold: those of the non-null rows,
  // converted to what the Presto block carries. They are the column's own
  // values unless the rows have nulls or the type needs conversion; the copy
  // made then goes into 'wave', so it is as large as the wave at most.
  const uint8_t*
  waveValues(size_t columnIndex, size_t firstPage, size_t lastPage, Wave& wave);

  // Lays out the next wave of pages of the current Serializer on the device.
  void buildNextWave();

  // Copies a batch of pages of the current wave to memory of the operator's
  // pool, through pinned memory that is held only meanwhile.
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

  const std::weak_ptr<exec::DefaultOutputBufferManager> bufferManager_;
  // The columns of the input the output has, in its order. Empty when the
  // output has every column of the input in the order of the input.
  const std::vector<column_index_t> outputChannels_;
  const int64_t pageBytes_;
  // Device bytes laid out per wave: a bounded number of pages.
  const int64_t waveBytes_;
  // Bytes of output columns to gather before they are serialized: a wave, so
  // that the layout, which waits for the device a fixed number of times, runs
  // once per wave rather than once per page.
  const int64_t gatherBytes_;
  // Device bytes of gathered inputs, the columns the output drops included,
  // at which the inputs are serialized although the columns of the output
  // fill less than 'gatherBytes_': the dropped columns stay on the device as
  // long as their input is gathered.
  const int64_t maxPendingInputBytes_;
  // Bytes copied to the host between two stream synchronizations, which the
  // operator holds besides the pages in the output buffer: at most
  // kPagesPerHostBatch pages, and no more than
  // max_page_partitioning_buffer_size, which bounds what
  // exec::PartitionedOutput holds; a batch always holds one page.
  const int64_t hostBatchBytes_;
  // Serialize each input as it arrives instead of gathering a wave first.
  const bool eagerFlush_;
  // The buffer manager's listener, which computes the Presto checksum; null
  // without a listener.
  const std::unique_ptr<OutputStreamListener> listener_;
  serializer::presto::PrestoOutputStreamListener* const checksumListener_;
  rmm::host_device_async_resource_ref pinnedMr_;

  std::vector<CudfVectorPtr> pendingInputs_;
  int64_t pendingRows_{0};
  // Device bytes of the columns of the output in 'pendingInputs_'.
  int64_t pendingBytes_{0};
  // Device bytes of every column of 'pendingInputs_'.
  int64_t pendingInputBytes_{0};

  std::unique_ptr<Serializer> serializer_;
  // The task lists of the wave being laid out, reused across waves.
  std::unique_ptr<TaskLists> tasks_;
  std::optional<Wave> wave_;
  std::deque<HostPage> ready_;

  exec::BlockingReason blockingReason_{exec::BlockingReason::kNotBlocked};
  ContinueFuture future_;
  bool finished_{false};

  uint64_t layoutNanos_{0};
  uint64_t copyNanos_{0};
};

} // namespace facebook::velox::cudf_velox
