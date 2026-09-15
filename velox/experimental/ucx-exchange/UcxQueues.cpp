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
#include "velox/experimental/ucx-exchange/UcxQueues.h"

#include <algorithm>
#include <limits>
#include <sstream>

#include <glog/logging.h>

namespace facebook::velox::ucx_exchange {

namespace {

int64_t checkedBytes(std::size_t bytes) {
  VELOX_CHECK_LE(
      bytes,
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
      "UCX payload size exceeds signed accounting range");
  return static_cast<int64_t>(bytes);
}

int64_t checkedAdd(int64_t left, int64_t right, const char* counter) {
  VELOX_CHECK_GE(left, 0, "{} must be non-negative", counter);
  VELOX_CHECK_GE(right, 0, "{} increment must be non-negative", counter);
  VELOX_CHECK_LE(
      right,
      std::numeric_limits<int64_t>::max() - left,
      "{} overflow: {} + {}",
      counter,
      left,
      right);
  return left + right;
}

int64_t
checkedMultiply(int64_t value, int64_t multiplier, const char* counter) {
  VELOX_CHECK_GE(value, 0, "{} value must be non-negative", counter);
  VELOX_CHECK_GE(multiplier, 0, "{} multiplier must be non-negative", counter);
  if (value == 0) {
    return 0;
  }
  VELOX_CHECK_LE(
      multiplier,
      std::numeric_limits<int64_t>::max() / value,
      "{} overflow: {} * {}",
      counter,
      value,
      multiplier);
  return value * multiplier;
}

void checkedSubtract(int64_t& value, int64_t decrement, const char* counter) {
  VELOX_CHECK_GE(decrement, 0, "{} decrement must be non-negative", counter);
  VELOX_CHECK_GE(
      value, decrement, "{} underflow: {} - {}", counter, value, decrement);
  value -= decrement;
}

int64_t checkedConfiguredMax(uint64_t bytes) {
  VELOX_CHECK_LE(
      bytes,
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "maxOutputBufferSize exceeds signed accounting range");
  return static_cast<int64_t>(bytes);
}

int64_t percentageOf(int64_t value, int32_t percentage) {
  VELOX_CHECK_GE(value, 0);
  VELOX_CHECK_GE(percentage, 0);
  VELOX_CHECK_LE(percentage, 100);
  const auto whole =
      checkedMultiply(value / 100, percentage, "percentage whole component");
  const auto remainder =
      checkedMultiply(
          value % 100, percentage, "percentage remainder component") /
      100;
  return checkedAdd(whole, remainder, "percentage result");
}

} // namespace

void UcxDestinationQueue::Stats::recordEnqueue(
    const cudf::packed_columns* data) {
  if (data != nullptr) {
    bytesQueued = checkedAdd(
        bytesQueued, checkedBytes(data->gpu_data->size()), "bytesQueued");
    packedColumnsQueued =
        checkedAdd(packedColumnsQueued, 1, "packedColumnsQueued");
  }
}

void UcxDestinationQueue::Stats::recordDequeue(
    const cudf::packed_columns* data) {
  if (data != nullptr) {
    const int64_t size = checkedBytes(data->gpu_data->size());

    checkedSubtract(bytesQueued, size, "bytesQueued");
    checkedSubtract(packedColumnsQueued, 1, "packedColumnsQueued");

    bytesSent = checkedAdd(bytesSent, size, "bytesSent");
    packedColumnsSent = checkedAdd(packedColumnsSent, 1, "packedColumnsSent");
    bytesInFlight = checkedAdd(bytesInFlight, size, "bytesInFlight");
    packedColumnsInFlight =
        checkedAdd(packedColumnsInFlight, 1, "packedColumnsInFlight");
  }
}

void UcxDestinationQueue::enqueueBack(
    std::shared_ptr<cudf::packed_columns> data,
    vector_size_t numRows) {
  // drop duplicate end markers.
  if (data == nullptr && !queue_.empty() && queue_.back().data == nullptr) {
    return;
  }

  // Validate the accounting update before changing either the queue or its
  // counters. In particular, a deque allocation failure must leave this
  // operation wholly uncommitted so the producer still owns its reservation.
  auto committedStats = stats_;
  committedStats.recordEnqueue(data.get());
  queue_.push_back(QueuedPage{std::move(data), numRows});
  stats_ = committedStats;
}

void UcxDestinationQueue::enqueueFront(
    std::shared_ptr<cudf::packed_columns> data,
    vector_size_t numRows) {
  // ignore nullptr.
  if (data == nullptr) {
    return;
  }

  // insert at the front.
  queue_.push_front(QueuedPage{std::move(data), numRows});
}

void UcxDestinationQueue::rollbackEnqueueBack() {
  VELOX_CHECK(!queue_.empty(), "Cannot roll back an empty destination queue");
  const auto& data = queue_.back().data;
  VELOX_CHECK_NOT_NULL(data, "Cannot roll back an end marker as output data");
  auto committedStats = stats_;
  checkedSubtract(
      committedStats.bytesQueued,
      checkedBytes(data->gpu_data->size()),
      "bytesQueued");
  checkedSubtract(committedStats.packedColumnsQueued, 1, "packedColumnsQueued");
  queue_.pop_back();
  stats_ = committedStats;
}

UcxDestinationQueue::Data UcxDestinationQueue::getData(
    UcxDataAvailableCallback notify) {
  if (queue_.empty()) {
    // delay notification.
    notify_ = std::move(notify);
    return {};
  }

  // Build everything that can allocate, and validate the accounting update,
  // before removing the page. This gives callers a strong exception guarantee
  // around the queued-to-in-flight ownership transition.
  std::vector<int64_t> remainingBytes;
  remainingBytes.reserve(queue_.size() - 1);
  if (queue_.front().data == nullptr) {
    VELOX_CHECK_EQ(
        queue_.size(), 1, "null marker found before queued output data");
  }
  // fill in the remainingbytes vector.
  for (std::size_t i = 1; i < queue_.size(); ++i) {
    if (queue_[i].data == nullptr) {
      VELOX_CHECK_EQ(i, queue_.size() - 1, "null marker found in the middle");
      break;
    }
    remainingBytes.push_back(checkedBytes(queue_[i].data->gpu_data->size()));
  }

  auto committedStats = stats_;
  committedStats.recordDequeue(queue_.front().data.get());
  auto page = std::move(queue_.front());
  queue_.pop_front();
  stats_ = committedStats;
  return {std::move(page.data), page.numRows, std::move(remainingBytes), true};
}

void UcxDestinationQueue::deleteResults() {
  for (auto i = 0; i < queue_.size(); ++i) {
    if (queue_[i].data == nullptr) {
      VELOX_CHECK_EQ(i, queue_.size() - 1, "null marker found in the middle");
      break;
    }
  }
  queue_.clear();
}

UcxDataAvailable UcxDestinationQueue::getAndClearNotify() {
  if (notify_ == nullptr) {
    return UcxDataAvailable();
  }
  UcxDataAvailable result;
  // Move the waiter out before getData(). If deleteResults() cleared the
  // destination first, getData(nullptr) sees an empty queue and installs the
  // supplied null callback; extracting notify_ afterwards would therefore
  // silently discard the original waiter.
  result.callback = std::move(notify_);
  auto data = getData(nullptr);
  result.data = std::move(data.data);
  result.numRows = data.numRows;
  result.remainingBytes = std::move(data.remainingBytes);
  clearNotify();
  return result;
}

void UcxDestinationQueue::clearNotify() {
  notify_ = nullptr;
}

void UcxDestinationQueue::finish() {
  VELOX_CHECK_NULL(notify_, "notify must be cleared before finish");
  VELOX_CHECK(queue_.empty(), "data must be fetched before finish");
}

UcxDestinationQueue::Stats UcxDestinationQueue::stats() const {
  return stats_;
}

int64_t UcxDestinationQueue::transferBytes() const {
  return checkedAdd(stats_.bytesQueued, stats_.bytesInFlight, "transferBytes");
}

bool UcxDestinationQueue::waitingForData() const {
  return notify_ != nullptr;
}

void UcxDestinationQueue::releaseInFlight(
    int64_t bytes,
    int64_t numPackedColumns) {
  checkedSubtract(stats_.bytesInFlight, bytes, "destination bytesInFlight");
  checkedSubtract(
      stats_.packedColumnsInFlight,
      numPackedColumns,
      "destination packedColumnsInFlight");
}

std::string UcxDestinationQueue::toString() {
  std::stringstream out;
  out << "[available: " << queue_.size() << ", "
      << (notify_ ? "notify registered, " : "") << this << "]";
  return out.str();
}

// ---------- UcxOutputQueue ----------

UcxOutputQueue::UcxOutputQueue(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers,
    core::PartitionedOutputNode::Kind kind)
    : task_(task), kind_(kind), numDrivers_(numDrivers) {
  VELOX_CHECK_LE(
      numDestinations, static_cast<uint32_t>(std::numeric_limits<int>::max()));
  if (task_) {
    maxSize_ = checkedConfiguredMax(
        task_->queryCtx()->queryConfig().maxOutputBufferSize());
    continueSize_ = percentageOf(maxSize_, kContinuePct);
  } // else: maxSize_ and continueSize_ will be set once the task is created and
    // initialize called.
  if (numDestinations > 0) {
    addOutputBuffersLocked(static_cast<int>(numDestinations));
  }
  if (task_) {
    initialized_.store(true, std::memory_order_release);
  }
}

bool UcxOutputQueue::initialize(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers,
    core::PartitionedOutputNode::Kind kind) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_LE(
      numDestinations, static_cast<uint32_t>(std::numeric_limits<int>::max()));
  if (task_) {
    // already initialized!
    return false;
  }
  kind_ = kind;
  numDrivers_ = numDrivers;
  task_ = task;
  maxSize_ = checkedConfiguredMax(
      task_->queryCtx()->queryConfig().maxOutputBufferSize());
  continueSize_ = percentageOf(maxSize_, kContinuePct);
  // Publish task metadata before destination queue expansion. Acceptor only
  // needs task/kind to choose the intra-node path; getData() takes mutex_ and
  // waits for any queue expansion in this function to finish.
  initialized_.store(true, std::memory_order_release);
  if (static_cast<size_t>(numDestinations) > queues_.size()) {
    addOutputBuffersLocked(static_cast<int>(numDestinations));
  }
  VELOX_CHECK_EQ(transferReservedBytes_.size(), queues_.size());
  VELOX_CHECK_EQ(transferPromises_.size(), queues_.size());
  VELOX_CHECK_EQ(transferWindowBytes_.size(), queues_.size());
  return true;
}

void UcxOutputQueue::addOutputBuffersLocked(int numBuffers) {
  using Kind = core::PartitionedOutputNode::Kind;

  VELOX_CHECK_GE(numBuffers, 0);
  VELOX_CHECK_GT(static_cast<size_t>(numBuffers), queues_.size());
  VELOX_CHECK(!noMoreQueues_, "Cannot add destinations after noMoreBuffers");
  VELOX_CHECK_EQ(transferReservedBytes_.size(), queues_.size());
  VELOX_CHECK_EQ(transferPromises_.size(), queues_.size());
  VELOX_CHECK_EQ(transferWindowBytes_.size(), queues_.size());

  const auto numNewBuffers =
      static_cast<int64_t>(numBuffers) - static_cast<int64_t>(queues_.size());
  int64_t addedQueuedBytes = 0;
  int64_t addedQueuedPackedColumns = 0;
  if (kind_ == Kind::kBroadcast) {
    int64_t historyBytes = 0;
    for (const auto& [data, numRows] : dataToBroadcast_) {
      VELOX_CHECK_NOT_NULL(data);
      VELOX_CHECK_GE(numRows, 0);
      historyBytes = checkedAdd(
          historyBytes,
          checkedBytes(data->gpu_data->size()),
          "late broadcast history bytes");
    }
    addedQueuedBytes = checkedMultiply(
        historyBytes, numNewBuffers, "late broadcast queued bytes");
    addedQueuedPackedColumns = checkedMultiply(
        static_cast<int64_t>(dataToBroadcast_.size()),
        numNewBuffers,
        "late broadcast queued packed columns");
  }
  const auto committedQueuedBytes =
      checkedAdd(queuedBytes_, addedQueuedBytes, "queuedBytes");
  const auto committedQueuedPackedColumns = checkedAdd(
      queuedPackedColumns_, addedQueuedPackedColumns, "queuedPackedColumns");

  // Allocate and completely backfill every new destination before publishing
  // any of them. A host allocation failure therefore leaves both the visible
  // destination set and global accounting unchanged.
  std::vector<std::unique_ptr<UcxDestinationQueue>> newQueues;
  newQueues.reserve(numNewBuffers);
  for (int64_t i = 0; i < numNewBuffers; ++i) {
    auto buffer = std::make_unique<UcxDestinationQueue>();
    if (kind_ == Kind::kBroadcast) {
      for (const auto& [data, numRows] : dataToBroadcast_) {
        buffer->enqueueBack(data, numRows);
      }
    }
    // Retained broadcast payloads must precede EOS for a late destination.
    if (atEnd_) {
      buffer->enqueueBack(nullptr, /*numRows=*/0);
    }
    newQueues.emplace_back(std::move(buffer));
  }

  queues_.reserve(numBuffers);
  transferReservedBytes_.reserve(numBuffers);
  transferPromises_.reserve(numBuffers);
  transferWindowBytes_.reserve(numBuffers);
  if (addedQueuedBytes > 0) {
    updateTotalQueuedBytesMsLocked();
  }
  for (auto& buffer : newQueues) {
    queues_.emplace_back(std::move(buffer));
    transferReservedBytes_.push_back(0);
    transferPromises_.emplace_back();
    transferWindowBytes_.push_back(0);
  }
  queuedBytes_ = committedQueuedBytes;
  queuedPackedColumns_ = committedQueuedPackedColumns;
}

void UcxOutputQueue::updateNumDrivers(uint32_t newNumDrivers) {
  bool isNoMoreDrivers{false};
  {
    std::lock_guard<std::mutex> l(mutex_);
    numDrivers_ = newNumDrivers;
    // If we finished all drivers, ensure we register that we are 'done'.
    if (numDrivers_ == numFinished_) {
      isNoMoreDrivers = true;
    }
  }
  if (isNoMoreDrivers) {
    noMoreDrivers();
  }
}

void UcxOutputQueue::enqueue(
    int destination,
    std::unique_ptr<cudf::packed_columns> data,
    vector_size_t numRows,
    int64_t transferReservationBytes) {
  VELOX_CHECK_NOT_NULL(data);
  VELOX_CHECK_NOT_NULL(task_);
  VELOX_CHECK_GE(numRows, 0);
  VELOX_CHECK_GE(transferReservationBytes, 0);
  if (!task_->isRunning()) {
    std::vector<ContinuePromise> transferPromises;
    if (transferReservationBytes > 0) {
      std::lock_guard<std::mutex> l(mutex_);
      if (!terminated_ && destination >= 0 &&
          static_cast<size_t>(destination) < transferReservedBytes_.size() &&
          queues_[destination] != nullptr) {
        releaseTransferReservationLocked(destination, transferReservationBytes);
        collectTransferPromisesLocked(destination, transferPromises);
      }
    }
    for (auto& promise : transferPromises) {
      promise.setValue();
    }
    return;
  }
  std::vector<UcxDataAvailable> dataAvailableCallbacks;
  {
    std::lock_guard<std::mutex> l(mutex_);
    // Partitioned output can notify at most one destination; broadcast can
    // notify every destination. Reserve before transferring
    // queue ownership so callback-vector growth cannot split the commit.
    dataAvailableCallbacks.reserve(
        kind_ == core::PartitionedOutputNode::Kind::kBroadcast ? queues_.size()
                                                               : 1);
    const auto numBytes = checkedBytes(data->gpu_data->size());
    auto sharedData = std::shared_ptr<cudf::packed_columns>(std::move(data));

    bool success = false;
    if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast) {
      VELOX_CHECK_EQ(
          transferReservationBytes,
          0,
          "Broadcast output does not use transfer reservations");
      VELOX_CHECK_EQ(destination, 0, "Broadcast uses destination 0");
      enqueueBroadcastOutputLocked(
          std::move(sharedData), numRows, dataAvailableCallbacks);
      // For broadcast, count queuedBytes_ once per active destination so
      // that each destination's dequeue symmetrically decrements it. The
      // total sent stats count the logical data once.
      int64_t numActive = 0;
      for (auto& q : queues_) {
        if (q != nullptr) {
          numActive++;
        }
      }
      updateTotalQueuedBytesMsLocked();
      queuedBytes_ = checkedAdd(
          queuedBytes_,
          checkedMultiply(numBytes, numActive, "broadcast queuedBytes"),
          "queuedBytes");
      queuedPackedColumns_ =
          checkedAdd(queuedPackedColumns_, numActive, "queuedPackedColumns");
      if (!noMoreQueues_) {
        broadcastHistoryBytes_ = checkedAdd(
            broadcastHistoryBytes_, numBytes, "broadcastHistoryBytes");
        broadcastHistoryPackedColumns_ = checkedAdd(
            broadcastHistoryPackedColumns_, 1, "broadcastHistoryPackedColumns");
      }
      totalBytesSent_ = checkedAdd(totalBytesSent_, numBytes, "totalBytesSent");
      totalRowsSent_ = checkedAdd(totalRowsSent_, numRows, "totalRowsSent");
      totalPackedColumnsSent_ =
          checkedAdd(totalPackedColumnsSent_, 1, "totalPackedColumnsSent");
      success = true;
    } else {
      VELOX_CHECK_GE(destination, 0);
      VELOX_CHECK_LT(destination, queues_.size());
      success = enqueuePartitionedOutputLocked(
          destination, std::move(sharedData), numRows, dataAvailableCallbacks);
      if (success) {
        updateStatsWithEnqueuedLocked(numBytes, numRows);
      }
      // This is the reservation-to-queued/in-flight commit point. Everything
      // above that can allocate either completed successfully or left the
      // reservation untouched for the producer's scope guard to release.
      releaseTransferReservationLocked(destination, transferReservationBytes);
    }
  }
  // Now that data is enqueued, notify blocked readers (outside of mutex.)
  for (auto& callback : dataAvailableCallbacks) {
    callback.notify();
  }
}

bool UcxOutputQueue::checkBlocked(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  const auto blockedBytes = producerBlockedBytesLocked();
  if (!terminated_ && maxSize_ > 0 && blockedBytes >= maxSize_) {
    if (!future) {
      return true;
    }
    VLOG(2) << "[BACKPRESSURE] task=" << (task_ ? task_->taskId() : "n/a")
            << " BLOCKED producerBytes=" << blockedBytes
            << " retainedBytes=" << retainedBytesLocked()
            << " maxSize=" << maxSize_
            << " waitingProducers=" << (promises_.size() + 1);
    promises_.emplace_back("UcxOutputQueue::checkBlocked");
    *future = promises_.back().getSemiFuture();
    return true;
  }
  return false;
}

bool UcxOutputQueue::checkTransferCapacity(
    int destination,
    int64_t maxBytes,
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(maxBytes, 0);
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(static_cast<size_t>(destination), queues_.size());
  if (terminated_ || queues_[destination] == nullptr) {
    return false;
  }

  const auto* queue = queues_[destination].get();
  const auto destinationStats = queue->stats();
  const auto reservedBytes = transferReservedBytes_[destination];
  if (queue->waitingForData() && destinationStats.bytesQueued == 0 &&
      reservedBytes == 0) {
    return false;
  }

  if (transferBytesLocked(destination) >= maxBytes) {
    if (future) {
      transferPromises_[destination].emplace_back(
          "UcxOutputQueue::checkTransferCapacity");
      *future = transferPromises_[destination].back().getSemiFuture();
    }
    return true;
  }
  return false;
}

bool UcxOutputQueue::reserveTransferBytes(
    int destination,
    int64_t bytes,
    int64_t maxBytes,
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(bytes, 0);
  VELOX_CHECK_GT(maxBytes, 0);
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(static_cast<size_t>(destination), queues_.size());
  if (terminated_ || queues_[destination] == nullptr) {
    return false;
  }

  if (fullTransferCongested_) {
    const auto retainedBytes = retainedBytesWithTransferReservationsLocked();
    if (retainedBytes > 0) {
      maybeGrowFullTransferRetainedLimitLocked(retainedBytes);
      const auto retainedLimit = fullTransferRetainedLimitLocked();
      if (retainedLimit > 0 &&
          (bytes > retainedLimit || retainedBytes > retainedLimit - bytes)) {
        if (future) {
          promises_.emplace_back("UcxOutputQueue::reserveTransferBytes");
          *future = promises_.back().getSemiFuture();
        }
        return true;
      }
    }
  }

  auto* queue = queues_[destination].get();
  const auto destinationStats = queue->stats();
  auto& reservedBytes = transferReservedBytes_[destination];
  if (queue->waitingForData() && destinationStats.bytesQueued == 0 &&
      reservedBytes == 0) {
    reservedBytes = checkedAdd(reservedBytes, bytes, "transfer reservation");
    return false;
  }

  const auto transferBytes = transferBytesLocked(destination);
  if (bytes > maxBytes || transferBytes > maxBytes - bytes) {
    if (future) {
      transferPromises_[destination].emplace_back(
          "UcxOutputQueue::reserveTransferBytes");
      *future = transferPromises_[destination].back().getSemiFuture();
    }
    return true;
  }

  reservedBytes = checkedAdd(reservedBytes, bytes, "transfer reservation");
  return false;
}

bool UcxOutputQueue::waitForFullTransferCapacity(
    int64_t bytes,
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(bytes, 0);
  if (terminated_ || !fullTransferCongested_) {
    return false;
  }

  const auto retainedBytes = retainedBytesWithTransferReservationsLocked();
  if (retainedBytes == 0) {
    // No queue event could ever satisfy a promise installed here. The caller
    // must treat a repeated allocation failure as terminal or use an external
    // GPU-memory wake-up source.
    return false;
  }
  maybeGrowFullTransferRetainedLimitLocked(retainedBytes);
  const auto retainedLimit = fullTransferRetainedLimitLocked();
  if (retainedLimit > 0 &&
      (bytes > retainedLimit || retainedBytes > retainedLimit - bytes)) {
    if (future) {
      promises_.emplace_back("UcxOutputQueue::waitForFullTransferCapacity");
      *future = promises_.back().getSemiFuture();
    }
    return true;
  }
  return false;
}

bool UcxOutputQueue::reserveFullTransferBytes(
    int destination,
    int64_t bytes,
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(bytes, 0);
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(static_cast<size_t>(destination), queues_.size());
  if (terminated_ || queues_[destination] == nullptr) {
    return false;
  }

  const auto retainedBytes = retainedBytesWithTransferReservationsLocked();
  if (retainedBytes > 0) {
    maybeGrowFullTransferRetainedLimitLocked(retainedBytes);
    auto retainedLimit = fullTransferRetainedLimitLocked();
    const bool exceedsLimit = retainedLimit > 0 &&
        (bytes > retainedLimit || retainedBytes > retainedLimit - bytes);
    if (exceedsLimit && !fullTransferCongested_) {
      // A successful full contiguous split must be able to reserve its
      // already-materialized aggregate even when it is larger than the normal
      // per-destination default. Keep this one admission exact; subsequent
      // probes re-evaluate the explicit limit against the then-active
      // destination count.
      fullTransferRetainedLimit_ =
          checkedAdd(retainedBytes, bytes, "full transfer retained limit");
      retainedLimit = fullTransferRetainedLimit_;
    }
    if (retainedLimit > 0 &&
        (bytes > retainedLimit || retainedBytes > retainedLimit - bytes)) {
      if (future) {
        promises_.emplace_back("UcxOutputQueue::reserveFullTransferBytes");
        *future = promises_.back().getSemiFuture();
      }
      return true;
    }
  }

  if (maxSize_ > 0 && fullTransferCongested_) {
    const auto fairDestinationBudget = std::max<int64_t>(
        1, fullTransferRetainedLimitLocked() / activeDestinationCountLocked());
    const auto destinationBytes = transferBytesLocked(destination);
    const auto minimumDestinationBudget = checkedAdd(
        transferReservedBytes_[destination],
        bytes,
        "minimum destination budget");
    const auto destinationBudget =
        std::max(fairDestinationBudget, minimumDestinationBudget);
    if (destinationBytes > 0 &&
        (bytes > destinationBudget ||
         destinationBytes > destinationBudget - bytes)) {
      if (future) {
        transferPromises_[destination].emplace_back(
            "UcxOutputQueue::reserveFullTransferBytes");
        *future = transferPromises_[destination].back().getSemiFuture();
      }
      return true;
    }
  }

  auto& reservedBytes = transferReservedBytes_[destination];
  reservedBytes = checkedAdd(reservedBytes, bytes, "transfer reservation");
  return false;
}

void UcxOutputQueue::releaseTransferReservation(
    int destination,
    int64_t bytes) {
  std::vector<ContinuePromise> promises;
  std::vector<ContinuePromise> transferPromises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_GE(bytes, 0);
    VELOX_CHECK_GE(destination, 0);
    VELOX_CHECK_LT(static_cast<size_t>(destination), queues_.size());
    if (terminated_ || queues_[destination] == nullptr) {
      return;
    }
    releaseTransferReservationLocked(destination, bytes);
    maybeContinueProducersLocked(promises);
    collectTransferPromisesLocked(destination, transferPromises);
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
  for (auto& promise : transferPromises) {
    promise.setValue();
  }
}

int64_t UcxOutputQueue::transferWindowBytes(
    int destination,
    int64_t baseBytes,
    int64_t normalBytes,
    int64_t maxBytes) {
  std::lock_guard<std::mutex> l(mutex_);
  return transferWindowBytesLocked(
      destination, baseBytes, normalBytes, maxBytes);
}

void UcxOutputQueue::recordTransferCongestion(
    int destination,
    int64_t baseBytes) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(baseBytes, 0);
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(static_cast<size_t>(destination), transferWindowBytes_.size());
  if (terminated_ || queues_[destination] == nullptr) {
    return;
  }

  auto& window = transferWindowBytes_[destination];
  window = window <= 0 ? baseBytes : std::max(baseBytes, window / 2);
}

void UcxOutputQueue::recordTransferDemand(
    int destination,
    int64_t targetBytes,
    int64_t baseBytes,
    int64_t maxBytes) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_GT(targetBytes, 0);
  VELOX_CHECK_GT(baseBytes, 0);
  VELOX_CHECK_GE(maxBytes, baseBytes);
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(static_cast<size_t>(destination), transferWindowBytes_.size());
  if (terminated_ || queues_[destination] == nullptr) {
    return;
  }

  auto& window = transferWindowBytes_[destination];
  if (window <= 0) {
    window = baseBytes;
  }
  const auto requestedWindow = std::clamp(targetBytes, baseBytes, maxBytes);
  const auto growthWindow = window > maxBytes - baseBytes
      ? maxBytes
      : checkedAdd(window, baseBytes, "transfer window");
  window = std::max(
      window, std::min(maxBytes, std::max(growthWindow, requestedWindow)));
}

void UcxOutputQueue::recordFullTransferCongestion() {
  std::lock_guard<std::mutex> l(mutex_);
  if (terminated_ || maxSize_ == 0) {
    return;
  }
  fullTransferCongested_ = true;

  const auto defaultLimit = defaultFullTransferRetainedLimitLocked();
  auto retainedBytes = retainedBytesWithTransferReservationsLocked();
  if (retainedBytes == 0) {
    retainedBytes = fullTransferRetainedLimit_ > 0 ? fullTransferRetainedLimit_
                                                   : defaultLimit;
  }
  const auto reducedLimit = retainedBytes - retainedBytes / 8;
  fullTransferRetainedLimit_ = std::min(
      defaultLimit,
      std::max<int64_t>(std::max<int64_t>(1, maxSize_), reducedLimit));
}

void UcxOutputQueue::releaseInFlightBytes(
    int destination,
    int64_t bytes,
    int64_t numPackedColumns) {
  std::vector<ContinuePromise> promises;
  std::vector<ContinuePromise> transferPromises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_GE(bytes, 0);
    VELOX_CHECK_GT(numPackedColumns, 0);
    VELOX_CHECK_GE(destination, 0);
    VELOX_CHECK_LT(static_cast<size_t>(destination), queues_.size());
    if (queues_[destination] != nullptr) {
      queues_[destination]->releaseInFlight(bytes, numPackedColumns);
    }
    updateStatsWithSendCompleteLocked(bytes, numPackedColumns, promises);
    collectTransferPromisesLocked(destination, transferPromises);
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
  for (auto& promise : transferPromises) {
    promise.setValue();
  }
}

void UcxOutputQueue::getData(int destination, UcxDataAvailableCallback notify) {
  VELOX_CHECK_GE(destination, 0);
  UcxDestinationQueue::Data data;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (static_cast<size_t>(destination) >= queues_.size()) {
      VELOX_CHECK_LT(destination, std::numeric_limits<int>::max());
      VELOX_CHECK(
          !isInitialized() ||
              kind_ != core::PartitionedOutputNode::Kind::kPartitioned,
          "Cannot add destinations to initialized partitioned output");
      addOutputBuffersLocked(destination + 1);
    }
    auto* queue = queues_[destination].get();
    // queue can be nullptr here if the task has terminated and results
    // have been removed. In this case, no data is returned.
    if (queue) {
      // Capture weak_ptr instead of raw `this` to prevent use-after-free.
      // The callback fires outside the lock (from enqueue() or terminate()),
      // and concurrent removeTask() can destroy the UcxOutputQueue while
      // the callback is still executing.
      std::weak_ptr<UcxOutputQueue> weakSelf = shared_from_this();
      data = queue->getData([destination, notify, weakSelf](
                                std::shared_ptr<cudf::packed_columns> data,
                                vector_size_t numRows,
                                std::vector<int64_t> remainingBytes) {
        std::vector<ContinuePromise> promises;
        if (data) {
          const auto bytes = checkedBytes(data->gpu_data->size());
          auto self = weakSelf.lock();
          if (self) {
            {
              std::lock_guard<std::mutex> l(self->mutex_);
              self->updateStatsWithDequeuedLocked(bytes, 1, promises);
            }
            for (auto& promise : promises) {
              promise.setValue();
            }
          }
        }
        notify(std::move(data), numRows, std::move(remainingBytes));
      });
      if (data.data) {
        // This implies data.immediate and no notify upcall will be done.
        // Ownership is now retained by the server until transfer completion.
        updateStatsWithDequeuedLocked(
            checkedBytes(data.data->gpu_data->size()), 1, promises);
      }
    } else {
      data = UcxDestinationQueue::Data{nullptr, 0, {}, true};
    }
  }
  // outside lock: If we have data, then return it immediately.
  if (data.immediate) {
    notify(std::move(data.data), data.numRows, std::move(data.remainingBytes));
  } else {
    VLOG(2) << "[QUEUE] task=" << (task_ ? task_->taskId() : "n/a")
            << " dest=" << destination
            << " server waiting for data (callback installed)";
  }
  // wake up any producers that are waiting for queue to become less full.
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void UcxOutputQueue::noMoreData() {
  // Increment number of finished drivers.
  checkIfDone(true);
}

void UcxOutputQueue::noMoreDrivers() {
  // Do not increment number of finished drivers.
  checkIfDone(false);
}

void UcxOutputQueue::checkIfDone(bool oneDriverFinished) {
  std::vector<UcxDataAvailable> finished;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (oneDriverFinished) {
      ++numFinished_;
    }
    VELOX_CHECK_LE(
        numFinished_,
        numDrivers_,
        "Each driver should call noMoreData exactly once");
    atEnd_ = numFinished_ == numDrivers_;
    if (!atEnd_) {
      return;
    }
    {
      int64_t avgRows = totalPackedColumnsSent_ > 0
          ? totalRowsSent_ / totalPackedColumnsSent_
          : 0;
      VLOG(1) << "[OUTPUT-STATS] task=" << (task_ ? task_->taskId() : "n/a")
              << " totalRows=" << totalRowsSent_
              << " chunks=" << totalPackedColumnsSent_
              << " avgRowsPerChunk=" << avgRows
              << " totalBytes=" << totalBytesSent_;
    }
    for (auto& queue : queues_) {
      if (queue != nullptr) {
        queue->enqueueBack(nullptr, /*numRows=*/0);
        finished.push_back(queue->getAndClearNotify());
      }
    }
  }
  // Notify outside of mutex.
  for (auto& notification : finished) {
    notification.notify();
  }
}

bool UcxOutputQueue::enqueuePartitionedOutputLocked(
    int destination,
    std::shared_ptr<cudf::packed_columns> data,
    vector_size_t numRows,
    std::vector<UcxDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(static_cast<size_t>(destination), queues_.size());
  bool success = false;
  auto* queue = queues_[destination].get();
  if (queue != nullptr) {
    queue->enqueueBack(std::move(data), numRows);
    dataAvailableCbs.emplace_back(queue->getAndClearNotify());
    success = true;
  }
  return success;
}

void UcxOutputQueue::releaseTransferReservationLocked(
    int destination,
    int64_t bytes) {
  VELOX_CHECK_GE(bytes, 0);
  if (bytes == 0) {
    return;
  }
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(
      static_cast<size_t>(destination), transferReservedBytes_.size());
  if (terminated_ || queues_[destination] == nullptr) {
    return;
  }
  checkedSubtract(
      transferReservedBytes_[destination], bytes, "transfer reservation");
}

int64_t UcxOutputQueue::transferWindowBytesLocked(
    int destination,
    int64_t baseBytes,
    int64_t normalBytes,
    int64_t maxBytes) {
  VELOX_CHECK_GT(baseBytes, 0);
  VELOX_CHECK_GE(normalBytes, baseBytes);
  VELOX_CHECK_GE(maxBytes, normalBytes);
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(static_cast<size_t>(destination), queues_.size());
  VELOX_CHECK_EQ(transferWindowBytes_.size(), queues_.size());

  auto* queue = queues_[destination].get();
  if (terminated_ || queue == nullptr) {
    return baseBytes;
  }

  auto& window = transferWindowBytes_[destination];
  if (window <= 0) {
    window = normalBytes;
  }
  window = std::clamp(window, baseBytes, maxBytes);

  const auto destinationStats = queue->stats();
  const auto transferBytes = transferBytesLocked(destination);
  const auto retainedBytes = retainedBytesLocked();
  const bool retainedPressure = maxSize_ > 0 &&
      retainedBytes >= static_cast<int64_t>(maxSize_ - maxSize_ / 10);

  if (retainedPressure) {
    window = baseBytes;
  } else if (
      queue->waitingForData() && destinationStats.bytesQueued == 0 &&
      transferBytes == 0) {
    const auto grown = window > maxBytes - baseBytes
        ? maxBytes
        : checkedAdd(window, baseBytes, "transfer window");
    window = std::min(maxBytes, std::max(normalBytes, grown));
  } else if (transferBytes < window / 2 && window < normalBytes) {
    window = window > normalBytes - baseBytes
        ? normalBytes
        : checkedAdd(window, baseBytes, "transfer window");
  }
  return window;
}

void UcxOutputQueue::collectTransferPromisesLocked(
    int destination,
    std::vector<ContinuePromise>& promises) {
  if (destination < 0 ||
      static_cast<size_t>(destination) >= transferPromises_.size()) {
    return;
  }
  for (auto& promise : transferPromises_[destination]) {
    promises.push_back(std::move(promise));
  }
  transferPromises_[destination].clear();
}

void UcxOutputQueue::collectAllTransferPromisesLocked(
    std::vector<ContinuePromise>& promises) {
  for (auto& destinationPromises : transferPromises_) {
    for (auto& promise : destinationPromises) {
      promises.push_back(std::move(promise));
    }
    destinationPromises.clear();
  }
}

void UcxOutputQueue::enqueueBroadcastOutputLocked(
    std::shared_ptr<cudf::packed_columns> data,
    vector_size_t numRows,
    std::vector<UcxDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());

  // Allocate the late-destination history entry before changing any
  // destination. Then enqueue to every destination before extracting a
  // waiting callback. If a deque allocation fails, no callback has observed
  // the page and all preceding destination enqueues can be rolled back.
  const bool retainForBackfill = !noMoreQueues_;
  if (retainForBackfill) {
    dataToBroadcast_.emplace_back(data, numRows);
  }
  size_t destination = 0;
  try {
    for (; destination < queues_.size(); ++destination) {
      auto& queue = queues_[destination];
      if (queue != nullptr) {
        queue->enqueueBack(data, numRows);
      }
    }
  } catch (...) {
    for (size_t rollbackDestination = 0; rollbackDestination < destination;
         ++rollbackDestination) {
      auto& queue = queues_[rollbackDestination];
      if (queue != nullptr) {
        queue->rollbackEnqueueBack();
      }
    }
    if (retainForBackfill) {
      dataToBroadcast_.pop_back();
    }
    throw;
  }

  for (auto& queue : queues_) {
    if (queue != nullptr) {
      dataAvailableCbs.emplace_back(queue->getAndClearNotify());
    }
  }
}

void UcxOutputQueue::clearBroadcastHistoryLocked(
    std::vector<ContinuePromise>& promises) {
  if (dataToBroadcast_.empty()) {
    VELOX_CHECK_EQ(broadcastHistoryBytes_, 0);
    VELOX_CHECK_EQ(broadcastHistoryPackedColumns_, 0);
    return;
  }
  VELOX_CHECK_EQ(broadcastHistoryPackedColumns_, dataToBroadcast_.size());
  int64_t retainedBytes = 0;
  for (const auto& item : dataToBroadcast_) {
    const auto& data = item.first;
    VELOX_CHECK_NOT_NULL(data);
    retainedBytes = checkedAdd(
        retainedBytes,
        checkedBytes(data->gpu_data->size()),
        "broadcast history bytes");
  }
  VELOX_CHECK_EQ(broadcastHistoryBytes_, retainedBytes);
  dataToBroadcast_.clear();
  broadcastHistoryBytes_ = 0;
  broadcastHistoryPackedColumns_ = 0;
  maybeContinueProducersLocked(promises);
}

bool UcxOutputQueue::isFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  return isFinishedLocked();
}

bool UcxOutputQueue::isFinishedLocked() {
  // For broadcast, we can only be finished after receiving the no more
  // (destination) buffers signal, matching OutputBuffer::isFinishedLocked().
  if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast &&
      !noMoreQueues_) {
    return false;
  }
  for (auto& queue : queues_) {
    if (queue != nullptr) {
      return false;
    }
  }
  return true;
}

void UcxOutputQueue::updateOutputBuffers(int numBuffers, bool noMoreBuffers) {
  using Kind = core::PartitionedOutputNode::Kind;
  bool isFinished{false};
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_GE(numBuffers, 0);
    if (kind_ == Kind::kPartitioned) {
      VELOX_CHECK_EQ(queues_.size(), numBuffers);
      VELOX_CHECK(noMoreBuffers);
      noMoreQueues_ = true;
      return;
    }

    VELOX_CHECK_EQ(kind_, Kind::kBroadcast);
    if (static_cast<size_t>(numBuffers) > queues_.size()) {
      addOutputBuffersLocked(numBuffers);
    }

    if (!noMoreBuffers) {
      return;
    }

    noMoreQueues_ = true;
    clearBroadcastHistoryLocked(promises);
    isFinished = isFinishedLocked();
  }

  for (auto& promise : promises) {
    promise.setValue();
  }

  if (isFinished && task_) {
    task_->setAllOutputConsumed();
  }
}

void UcxOutputQueue::deleteResults(int destination) {
  bool isFinished{false};
  UcxDataAvailable dataAvailable;
  std::vector<ContinuePromise> promises;
  std::vector<ContinuePromise> transferPromises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (destination < 0 || static_cast<size_t>(destination) >= queues_.size()) {
      VLOG(1) << "deleteResults: destination " << destination
              << " out of range (size=" << queues_.size() << "), ignoring";
      return;
    }
    auto* queue = queues_[destination].get();
    if (queue == nullptr) {
      VLOG(1) << "Extra delete received for destination " << destination;
      return;
    }
    // remember destination queue fill stats
    int64_t bytes = queue->stats().bytesQueued;
    int64_t packedCols = queue->stats().packedColumnsQueued;
    queue->deleteResults();
    dataAvailable = queue->getAndClearNotify();
    queue->finish();
    queues_[destination] = nullptr;
    isFinished = isFinishedLocked();
    // update UcxOutputQueue stats
    updateStatsWithFreedLocked(bytes, packedCols, promises);
    transferReservedBytes_[destination] = 0;
    transferWindowBytes_[destination] = 0;

    // Destination deletion is terminal for producers targeting it. Wake every
    // waiter so each driver can observe task/destination cancellation instead
    // of depending on a transfer event that can no longer occur.
    for (auto& promise : promises_) {
      promises.push_back(std::move(promise));
    }
    promises_.clear();
    collectAllTransferPromisesLocked(transferPromises);
  }

  // Outside of mutex.
  dataAvailable.notify();
  // wake up any producers that are waiting for queue to become less full.
  for (auto& promise : promises) {
    promise.setValue();
  }
  for (auto& promise : transferPromises) {
    promise.setValue();
  }

  if (isFinished && task_) {
    task_->setAllOutputConsumed();
  }
}

void UcxOutputQueue::terminate() {
  std::vector<UcxDataAvailable> pendingCallbacks;
  std::vector<ContinuePromise> promises;
  std::vector<ContinuePromise> transferPromises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (task_ && task_->isRunning()) {
      LOG(WARNING) << "UcxOutputQueue::terminate() called while task "
                   << task_->taskId() << " is still running";
    }
    terminated_ = true;
    clearBroadcastHistoryLocked(promises);
    // Fire all pending getData callbacks with nullptr to signal end-of-stream.
    // This handles the case where a producer task fails or is cancelled before
    // noMoreData() is called, preventing consumers from being orphaned.
    for (auto& queue : queues_) {
      if (queue != nullptr) {
        queue->enqueueBack(nullptr, /*numRows=*/0);
        pendingCallbacks.push_back(queue->getAndClearNotify());
      }
    }
    // Release any outstanding producer-side promises (blocked on queue-full).
    std::fill(transferReservedBytes_.begin(), transferReservedBytes_.end(), 0);
    std::fill(transferWindowBytes_.begin(), transferWindowBytes_.end(), 0);
    for (auto& promise : promises_) {
      promises.push_back(std::move(promise));
    }
    promises_.clear();
    collectAllTransferPromisesLocked(transferPromises);
  }

  // Fire callbacks outside of mutex to avoid potential deadlocks.
  for (auto& callback : pendingCallbacks) {
    callback.notify();
  }
  // Unblock any blocked producers.
  for (auto& promise : promises) {
    promise.setValue();
  }
  for (auto& promise : transferPromises) {
    promise.setValue();
  }
}

std::string UcxOutputQueue::toString() {
  std::stringstream out;
  std::lock_guard<std::mutex> l(mutex_);
  out << "[UcxOutputQueue task="
      << (task_ ? task_->taskId() : "<uninitialized>") << " queues=";
  for (const auto& queue : queues_) {
    out << (queue ? queue->toString() : "<deleted>") << ", ";
  }
  out << "transferReservedBytes=" << transferReservedBytesLocked()
      << ", queuedBytes=" << queuedBytes_
      << ", queuedPackedColumns=" << queuedPackedColumns_
      << ", inFlightBytes=" << inFlightBytes_
      << ", inFlightPackedColumns=" << inFlightPackedColumns_
      << ", broadcastHistoryBytes=" << broadcastHistoryBytes_
      << ", broadcastHistoryPackedColumns=" << broadcastHistoryPackedColumns_
      << ", fullTransferRetainedLimit=" << fullTransferRetainedLimitLocked()
      << ", fullTransferCongested=" << fullTransferCongested_
      << ", waitingProducers=" << promises_.size()
      << ", numFinished=" << numFinished_ << "/" << numDrivers_
      << ", atEnd=" << atEnd_ << ", noMoreQueues=" << noMoreQueues_
      << ", terminated=" << terminated_ << "]";
  return out.str();
}

std::optional<double> UcxOutputQueue::getUtilization() {
  std::lock_guard<std::mutex> l(mutex_);
  if (maxSize_ == 0) {
    return std::nullopt;
  }
  return retainedBytesLocked() / static_cast<double>(maxSize_);
}

std::optional<bool> UcxOutputQueue::isOverutilized() {
  std::lock_guard<std::mutex> l(mutex_);
  if (maxSize_ == 0) {
    return std::nullopt;
  }
  return atEnd_ ||
      retainedBytesLocked() > (0.5 * static_cast<double>(maxSize_));
}

exec::OutputBuffer::Stats UcxOutputQueue::stats() {
  std::lock_guard<std::mutex> l(mutex_);
  std::vector<UcxDestinationQueue::Stats> queueStats;

  updateTotalQueuedBytesMsLocked();

  auto stats = exec::OutputBuffer::Stats(
      kind(),
      noMoreQueues_,
      atEnd_,
      isFinishedLocked(),
      retainedBytesLocked(),
      retainedPackedColumnsLocked(),
      totalBytesSent_,
      totalRowsSent_,
      totalPackedColumnsSent_,
      getAverageQueueTimeMsLocked(),
      0 /* FIXME: compute num top buffers. */,
      {/* FIXME: transition queueStats to exec::DestinationBuffer::Stats */});
  return stats;
}

void UcxOutputQueue::updateStatsWithEnqueuedLocked(
    int64_t bytes,
    int64_t rows) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ = checkedAdd(queuedBytes_, bytes, "queuedBytes");
  queuedPackedColumns_ =
      checkedAdd(queuedPackedColumns_, 1, "queuedPackedColumns");

  totalBytesSent_ = checkedAdd(totalBytesSent_, bytes, "totalBytesSent");
  totalRowsSent_ = checkedAdd(totalRowsSent_, rows, "totalRowsSent");
  totalPackedColumnsSent_ =
      checkedAdd(totalPackedColumnsSent_, 1, "totalPackedColumnsSent");
}

void UcxOutputQueue::updateStatsWithDequeuedLocked(
    int64_t bytes,
    int64_t numPackedColumns,
    std::vector<ContinuePromise>& promises) {
  updateTotalQueuedBytesMsLocked();

  checkedSubtract(queuedBytes_, bytes, "queuedBytes");
  checkedSubtract(
      queuedPackedColumns_, numPackedColumns, "queuedPackedColumns");
  inFlightBytes_ = checkedAdd(inFlightBytes_, bytes, "inFlightBytes");
  inFlightPackedColumns_ = checkedAdd(
      inFlightPackedColumns_, numPackedColumns, "inFlightPackedColumns");

  maybeContinueProducersLocked(promises);
}

void UcxOutputQueue::updateStatsWithFreedLocked(
    int64_t bytes,
    int64_t numPackedCols,
    std::vector<ContinuePromise>& promises) {
  updateTotalQueuedBytesMsLocked();

  checkedSubtract(queuedBytes_, bytes, "queuedBytes");
  checkedSubtract(queuedPackedColumns_, numPackedCols, "queuedPackedColumns");
  maybeContinueProducersLocked(promises);
}

void UcxOutputQueue::updateStatsWithSendCompleteLocked(
    int64_t bytes,
    int64_t numPackedColumns,
    std::vector<ContinuePromise>& promises) {
  checkedSubtract(inFlightBytes_, bytes, "inFlightBytes");
  checkedSubtract(
      inFlightPackedColumns_, numPackedColumns, "inFlightPackedColumns");
  maybeContinueProducersLocked(promises);
}

void UcxOutputQueue::updateTotalQueuedBytesMsLocked() {
  const auto deltaMs = queueWatch_.lap().count();
  if (queuedBytes_ > 0) {
    totalQueuedBytesMs_ +=
        static_cast<double>(queuedBytes_) * static_cast<double>(deltaMs);
  }
}

int64_t UcxOutputQueue::getAverageQueueTimeMsLocked() const {
  if (totalBytesSent_ > 0) {
    return totalQueuedBytesMs_ / totalBytesSent_;
  }

  return 0;
}

void UcxOutputQueue::maybeContinueProducersLocked(
    std::vector<ContinuePromise>& promises) {
  auto blockedBytes = producerBlockedBytesLocked();
  auto continueBytes = static_cast<int64_t>(continueSize_);
  if (fullTransferCongested_ && fullTransferRetainedLimit_ > 0) {
    blockedBytes = retainedBytesWithTransferReservationsLocked();
    maybeGrowFullTransferRetainedLimitLocked(blockedBytes);
    continueBytes = fullTransferRetainedLimitLocked();
  }
  if (blockedBytes > continueBytes || promises_.empty()) {
    return;
  }

  const auto count = blockedBytes == 0 ? promises_.size() : size_t{1};
  VLOG(2) << "[BACKPRESSURE] task=" << (task_ ? task_->taskId() : "n/a")
          << " unblocking=" << count << " waiting=" << promises_.size()
          << " retainedBytes=" << retainedBytesLocked()
          << " transferReservedBytes=" << transferReservedBytesLocked()
          << " continueBytes=" << continueBytes;
  for (size_t i = 0; i < count; ++i) {
    promises.push_back(std::move(promises_[i]));
  }
  promises_.erase(promises_.begin(), promises_.begin() + count);
}

int64_t UcxOutputQueue::retainedBytesLocked() const {
  return checkedAdd(
      checkedAdd(queuedBytes_, inFlightBytes_, "retainedBytes"),
      broadcastHistoryBytes_,
      "retainedBytes with broadcast history");
}

int64_t UcxOutputQueue::producerBlockedBytesLocked() const {
  // Includes in-flight sends. Broadcast output does not use destination
  // materialization reservations and can otherwise dequeue into an unbounded
  // UCXX backlog.
  return retainedBytesLocked();
}

int64_t UcxOutputQueue::retainedPackedColumnsLocked() const {
  return checkedAdd(
      checkedAdd(
          queuedPackedColumns_,
          inFlightPackedColumns_,
          "retainedPackedColumns"),
      broadcastHistoryPackedColumns_,
      "retainedPackedColumns with broadcast history");
}

int64_t UcxOutputQueue::transferBytesLocked(int destination) const {
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(static_cast<size_t>(destination), queues_.size());
  VELOX_CHECK_EQ(transferReservedBytes_.size(), queues_.size());
  const auto* queue = queues_[destination].get();
  return checkedAdd(
      queue ? queue->transferBytes() : 0,
      transferReservedBytes_[destination],
      "destination transfer bytes");
}

int64_t UcxOutputQueue::transferReservedBytesLocked() const {
  int64_t total = 0;
  for (const auto bytes : transferReservedBytes_) {
    total = checkedAdd(total, bytes, "total transfer reservations");
  }
  return total;
}

int64_t UcxOutputQueue::retainedBytesWithTransferReservationsLocked() const {
  return checkedAdd(
      retainedBytesLocked(),
      transferReservedBytesLocked(),
      "retained and reserved bytes");
}

int64_t UcxOutputQueue::activeDestinationCountLocked() const {
  int64_t count = 0;
  for (const auto& queue : queues_) {
    if (queue) {
      count = checkedAdd(count, 1, "active destination count");
    }
  }
  return std::max<int64_t>(count, 1);
}

int64_t UcxOutputQueue::defaultFullTransferRetainedLimitLocked() const {
  if (maxSize_ == 0) {
    return 0;
  }
  return checkedMultiply(
      static_cast<int64_t>(maxSize_),
      activeDestinationCountLocked(),
      "default full transfer retained limit");
}

int64_t UcxOutputQueue::fullTransferRetainedLimitLocked() const {
  const auto defaultLimit = defaultFullTransferRetainedLimitLocked();
  if (defaultLimit == 0) {
    return 0;
  }
  if (fullTransferRetainedLimit_ <= 0) {
    return defaultLimit;
  }
  // Destination deletion lowers the default immediately. Do not let a window
  // learned with more active destinations continue admitting against that
  // obsolete aggregate capacity while retained sends drain.
  return std::min(
      defaultLimit, std::max<int64_t>(1, fullTransferRetainedLimit_));
}

void UcxOutputQueue::maybeGrowFullTransferRetainedLimitLocked(
    int64_t retainedBytes) {
  VELOX_CHECK_GE(retainedBytes, 0);
  if (maxSize_ == 0 || !fullTransferCongested_ ||
      fullTransferRetainedLimit_ <= 0) {
    return;
  }
  if (retainedBytes <= fullTransferRetainedLimit_ / 2) {
    const auto defaultLimit = defaultFullTransferRetainedLimitLocked();
    if (fullTransferRetainedLimit_ >= defaultLimit) {
      fullTransferRetainedLimit_ = 0;
      fullTransferCongested_ = false;
      return;
    }
    const auto increment = std::min<int64_t>(
        std::max<int64_t>(1, static_cast<int64_t>(maxSize_)),
        defaultLimit - fullTransferRetainedLimit_);
    fullTransferRetainedLimit_ = checkedAdd(
        fullTransferRetainedLimit_, increment, "full transfer retained limit");
    if (fullTransferRetainedLimit_ == defaultLimit) {
      // The learned window has fully recovered. Returning to the default mode
      // prevents repeated capacity probes from ratcheting it without bound.
      fullTransferRetainedLimit_ = 0;
      fullTransferCongested_ = false;
    }
  }
}

} // namespace facebook::velox::ucx_exchange
