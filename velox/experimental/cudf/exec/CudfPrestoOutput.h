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
/// encoding. Input is gathered until it fills a page of
/// cudf.presto_output_page_bytes or kMaxPageRows rows, unless the output
/// flushes eagerly. The pages carry the CRC32 checksum when the buffer
/// manager has a Presto output stream listener, as the pages of
/// exec::PartitionedOutput do.
class CudfPrestoOutput : public CudfOperatorBase {
 public:
  /// Rows a page holds at most. Bounds the pages of narrow rows and of rows
  /// without columns, which bytes do not bound.
  static constexpr int64_t kMaxPageRows = 1 << 20;

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

  /// Returns the DefaultOutputBufferManager of the output transport that
  /// 'node' names, or nullptr when that transport is not registered or does
  /// not use the default manager.
  static std::shared_ptr<exec::DefaultOutputBufferManager> bufferManagerFor(
      const core::PartitionedOutputNode& node,
      const core::QueryCtx& queryCtx);

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

  // One serialized page inside a wave buffer.
  struct PageRange {
    size_t offset;
    size_t size;
    int32_t numRows;
  };

  // A batch of pages that share one device buffer.
  struct Wave {
    rmm::device_buffer data;
    std::vector<PageRange> pages;
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
  // Device bytes laid out per wave, and bytes copied to the host between two
  // stream synchronizations: a bounded number of pages each.
  const int64_t waveBytes_;
  const int64_t hostBatchBytes_;
  // Serialize each input as it arrives instead of filling a page first.
  const bool eagerFlush_;
  // The buffer manager's listener, and the same when it computes the Presto
  // checksum; null otherwise.
  const std::unique_ptr<OutputStreamListener> listener_;
  serializer::presto::PrestoOutputStreamListener* const checksumListener_;
  rmm::host_device_async_resource_ref pinnedMr_;

  std::vector<CudfVectorPtr> pendingInputs_;
  int64_t pendingRows_{0};
  int64_t pendingBytes_{0};

  std::unique_ptr<Serializer> serializer_;
  std::optional<Wave> wave_;
  std::deque<HostPage> ready_;

  exec::BlockingReason blockingReason_{exec::BlockingReason::kNotBlocked};
  ContinueFuture future_;
  bool finished_{false};

  int64_t layoutNanos_{0};
  int64_t copyNanos_{0};
};

} // namespace facebook::velox::cudf_velox
