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

#include <atomic>

namespace facebook::velox::ucx_exchange {

void UcxDestinationQueue::Stats::recordEnqueue(
    const cudf::packed_columns* data) {
  if (data != nullptr) {
    bytesQueued += data->gpu_data->size();
    packedColumnsQueued++;
  }
}

void UcxDestinationQueue::Stats::recordDequeue(
    const cudf::packed_columns* data) {
  if (data != nullptr) {
    const int64_t size = data->gpu_data->size();

    bytesQueued -= size;
    VELOX_DCHECK_GE(bytesQueued, 0, "bytesQueued must be non-negative");
    --packedColumnsQueued;
    VELOX_DCHECK_GE(
        packedColumnsQueued, 0, "packedColumnsQueued must be non-negative");

    bytesSent += size;
    packedColumnsSent++;
  }
}

void UcxDestinationQueue::enqueueBack(
    std::shared_ptr<cudf::packed_columns> data) {
  // drop duplicate end markers.
  if (data == nullptr && !queue_.empty() && queue_.back() == nullptr) {
    return;
  }

  if (data != nullptr) {
    stats_.recordEnqueue(data.get());
  }
  queue_.push_back(std::move(data));
}

void UcxDestinationQueue::enqueueFront(
    std::shared_ptr<cudf::packed_columns> data) {
  // ignore nullptr.
  if (data == nullptr) {
    return;
  }

  // insert at the front.
  queue_.push_front(std::move(data));
}

UcxDestinationQueue::Data UcxDestinationQueue::getData(
    UcxDataAvailableCallback notify) {
  if (queue_.empty()) {
    // delay notification.
    notify_ = std::move(notify);
    return {};
  }

  // queue is not empty.
  auto data = std::move(queue_.front());
  queue_.pop_front();
  stats_.recordDequeue(data.get());

  std::vector<int64_t> remainingBytes;
  remainingBytes.reserve(queue_.size());
  // fill in the remainingbytes vector.
  for (std::size_t i = 0; i < queue_.size(); ++i) {
    if (queue_[i] == nullptr) {
      VELOX_CHECK_EQ(i, queue_.size() - 1, "null marker found in the middle");
      break;
    }
    remainingBytes.push_back(queue_[i]->gpu_data->size());
  }
  return {std::move(data), std::move(remainingBytes), true};
}

void UcxDestinationQueue::deleteResults() {
  for (auto i = 0; i < queue_.size(); ++i) {
    if (queue_[i] == nullptr) {
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
  result.callback = notify_;
  auto data = getData(nullptr);
  result.data = std::move(data.data);
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
  if (task_) {
    maxSize_ = task_->queryCtx()->queryConfig().maxOutputBufferSize();
    continueSize_ = (maxSize_ * kContinuePct) / 100;
  } // else: maxSize_ and continueSize_ will be set once the task is created and
    // initialize called.
  // create a queue for each destination.
  queues_.reserve(numDestinations);
  for (int i = 0; i < numDestinations; ++i) {
    // create the destination queues inside the vector using emplace_back.
    queues_.emplace_back(std::make_unique<UcxDestinationQueue>());
  }
}

bool UcxOutputQueue::initialize(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers,
    core::PartitionedOutputNode::Kind kind) {
  std::lock_guard<std::mutex> l(mutex_);
  if (task_) {
    // already initialized!
    return false;
  }
  kind_ = kind;
  numDrivers_ = numDrivers;
  task_ = task;
  maxSize_ = task_->queryCtx()->queryConfig().maxOutputBufferSize();
  continueSize_ = (maxSize_ * kContinuePct) / 100;
  // create additional queues if there are more destinations.
  for (int i = queues_.size(); i < numDestinations; ++i) {
    // create the destination queues inside the vector using emplace_back.
    queues_.emplace_back(std::make_unique<UcxDestinationQueue>());
  }
  // Publish the initialized flag last with release semantics. Lock-free
  // readers (canUseIntraNode → isInitialized) use an acquire load, so
  // all writes above (kind_, task_, queues_, etc.) are guaranteed visible
  // once they observe initialized_ == true.
  initialized_.store(true, std::memory_order_release);
  return true;
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
    int32_t numRows) {
  VELOX_CHECK_NOT_NULL(data);
  VELOX_CHECK_NOT_NULL(task_);
  VELOX_CHECK(
      task_->isRunning(), "Task is terminated, cannot add data to output.");
  std::vector<UcxDataAvailable> dataAvailableCallbacks;
  {
    std::lock_guard<std::mutex> l(mutex_);
    auto numBytes = data->gpu_data->size();
    auto sharedData = std::shared_ptr<cudf::packed_columns>(std::move(data));

    bool success = false;
    if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast) {
      VELOX_CHECK_EQ(destination, 0, "Broadcast uses destination 0");
      enqueueBroadcastOutputLocked(
          std::move(sharedData), dataAvailableCallbacks);
      // For broadcast, count queuedBytes_ once per active destination so
      // that each destination's dequeue symmetrically decrements it. The
      // total sent stats count the logical data once.
      int numActive = 0;
      for (auto& q : queues_) {
        if (q != nullptr) {
          numActive++;
        }
      }
      updateTotalQueuedBytesMsLocked();
      queuedBytes_ += numBytes * numActive;
      queuedPackedColumns_ += numActive;
      totalBytesSent_ += numBytes;
      totalRowsSent_ += numRows;
      totalPackedColumnsSent_++;
      success = true;
    } else {
      VELOX_CHECK_LT(destination, queues_.size());
      success = enqueuePartitionedOutputLocked(
          destination, std::move(sharedData), dataAvailableCallbacks);
      if (success) {
        updateStatsWithEnqueuedLocked(numBytes, numRows);
      }
    }
  }
  // Now that data is enqueued, notify blocked readers (outside of mutex.)
  for (auto& callback : dataAvailableCallbacks) {
    callback.notify();
  }
}

bool UcxOutputQueue::checkBlocked(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  if (queuedBytes_ >= maxSize_ && future) {
    VLOG(2) << "[BACKPRESSURE] task=" << (task_ ? task_->taskId() : "n/a")
            << " BLOCKED queuedBytes=" << queuedBytes_
            << " maxSize=" << maxSize_
            << " waitingProducers=" << (promises_.size() + 1);
    promises_.emplace_back("UcxOutputQueue::checkBlocked");
    *future = promises_.back().getSemiFuture();
    return true;
  }
  return false;
}

void UcxOutputQueue::getData(int destination, UcxDataAvailableCallback notify) {
  UcxDestinationQueue::Data data;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    // If the queue doesn't exist yet, create an empty queue to store
    // the notify callback. The queue will eventually be initialized when
    // the task is being created.
    for (int i = queues_.size(); i <= destination; ++i) {
      // create the destination queues inside the vector using emplace_back.
      queues_.emplace_back(std::make_unique<UcxDestinationQueue>());
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
      data = queue->getData([notify, weakSelf](
                                std::shared_ptr<cudf::packed_columns> data,
                                std::vector<int64_t> remainingBytes) {
        std::vector<ContinuePromise> promises;
        int64_t bytes = data ? data->gpu_data->size() : -1L;
        notify(std::move(data), std::move(remainingBytes));
        if (bytes >= 0L) {
          auto self = weakSelf.lock();
          if (!self) {
            // Queue was destroyed by removeTask(), safe to skip stats update.
            return;
          }
          std::lock_guard<std::mutex> l(self->mutex_);
          self->updateStatsWithFreedLocked(bytes, 1L, promises);
        }
        // outside of lock:
        // wake up any producers that are waiting for queue to become less full.
        for (auto& promise : promises) {
          promise.setValue();
        }
      });
      if (data.data) {
        // This implies data.immediate and no notify upcall will be done.
        // Need to update the stats here.
        updateStatsWithFreedLocked(data.data->gpu_data->size(), 1L, promises);
      }
    } else {
      data = UcxDestinationQueue::Data{nullptr, {}, true};
    }
  }
  // outside lock: If we have data, then return it immediately.
  if (data.immediate) {
    notify(std::move(data.data), std::move(data.remainingBytes));
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
        queue->enqueueBack(nullptr);
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
    std::vector<UcxDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());
  VELOX_CHECK_LT(destination, queues_.size());
  bool success = false;
  auto* queue = queues_[destination].get();
  if (queue != nullptr) {
    queue->enqueueBack(std::move(data));
    dataAvailableCbs.emplace_back(queue->getAndClearNotify());
    success = true;
  }
  return success;
}

void UcxOutputQueue::enqueueBroadcastOutputLocked(
    std::shared_ptr<cudf::packed_columns> data,
    std::vector<UcxDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());

  for (auto& queue : queues_) {
    if (queue != nullptr) {
      queue->enqueueBack(data);
      dataAvailableCbs.emplace_back(queue->getAndClearNotify());
    }
  }

  // Store for late-arriving destinations (backfill).
  if (!noMoreQueues_) {
    dataToBroadcast_.emplace_back(std::move(data));
  }
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
  if (kind_ == Kind::kPartitioned) {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_EQ(queues_.size(), numBuffers);
    VELOX_CHECK(noMoreBuffers);
    noMoreQueues_ = true;
    return;
  }

  VELOX_CHECK_EQ(kind_, Kind::kBroadcast);
  bool isFinished;
  {
    std::lock_guard<std::mutex> l(mutex_);

    if (numBuffers > queues_.size()) {
      // Add new destination queues and backfill with broadcast data.
      int32_t numNewBuffers = numBuffers - queues_.size();
      queues_.reserve(numBuffers);
      for (int32_t i = 0; i < numNewBuffers; ++i) {
        auto buffer = std::make_unique<UcxDestinationQueue>();
        for (const auto& data : dataToBroadcast_) {
          buffer->enqueueBack(data);
          // Account for backfilled data in queuedBytes_ so that dequeue
          // decrements don't drive it negative.
          queuedBytes_ += data->gpu_data->size();
          queuedPackedColumns_++;
        }
        if (atEnd_) {
          buffer->enqueueBack(nullptr);
        }
        queues_.emplace_back(std::move(buffer));
      }
    }

    if (!noMoreBuffers) {
      return;
    }

    noMoreQueues_ = true;
    dataToBroadcast_.clear();
    isFinished = isFinishedLocked();
  }

  if (isFinished && task_) {
    task_->setAllOutputConsumed();
  }
}

void UcxOutputQueue::deleteResults(int destination) {
  bool isFinished;
  UcxDataAvailable dataAvailable;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (destination >= queues_.size()) {
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
  }

  // Outside of mutex.
  dataAvailable.notify();
  // wake up any producers that are waiting for queue to become less full.
  for (auto& promise : promises) {
    promise.setValue();
  }

  if (isFinished && task_) {
    task_->setAllOutputConsumed();
  }
}

void UcxOutputQueue::terminate() {
  std::vector<UcxDataAvailable> pendingCallbacks;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (task_ && task_->isRunning()) {
      LOG(WARNING) << "UcxOutputQueue::terminate() called while task "
                   << task_->taskId() << " is still running";
    }
    // Fire all pending getData callbacks with nullptr to signal end-of-stream.
    // This handles the case where a producer task fails or is cancelled before
    // noMoreData() is called, preventing consumers from being orphaned.
    for (auto& queue : queues_) {
      if (queue != nullptr) {
        queue->enqueueBack(nullptr);
        pendingCallbacks.push_back(queue->getAndClearNotify());
      }
    }
    // Release any outstanding producer-side promises (blocked on queue-full).
    promises = std::move(promises_);
  }

  // Fire callbacks outside of mutex to avoid potential deadlocks.
  for (auto& callback : pendingCallbacks) {
    callback.notify();
  }
  // Unblock any blocked producers.
  for (auto& promise : promises) {
    promise.setValue();
  }
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
      queuedBytes_,
      queuedPackedColumns_,
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

  queuedBytes_ += bytes;
  queuedPackedColumns_++;

  totalBytesSent_ += bytes;
  totalRowsSent_ += rows;
  totalPackedColumnsSent_++;
}

void UcxOutputQueue::updateStatsWithFreedLocked(
    int64_t bytes,
    int64_t numPackedCols,
    std::vector<ContinuePromise>& promises) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ -= bytes;
  queuedPackedColumns_ -= numPackedCols;

  VELOX_CHECK_GE(queuedBytes_, 0);
  VELOX_CHECK_GE(queuedPackedColumns_, 0);

  // Check whether queue is below low-water mark and return outstanding
  // promises
  if (queuedBytes_ <= continueSize_ && !promises_.empty()) {
    VLOG(2) << "[BACKPRESSURE] task=" << (task_ ? task_->taskId() : "n/a")
            << " UNBLOCKING " << promises_.size() << " producers"
            << " queuedBytes=" << queuedBytes_
            << " continueSize=" << continueSize_;
    promises = std::move(promises_);
  }
}

void UcxOutputQueue::updateTotalQueuedBytesMsLocked() {
  const auto nowMs = getCurrentTimeMs();
  if (queuedBytes_ > 0) {
    const auto deltaMs = nowMs - queueStartMs_;
    totalQueuedBytesMs_ += queuedBytes_ * deltaMs;
  }

  queueStartMs_ = nowMs;
}

int64_t UcxOutputQueue::getAverageQueueTimeMsLocked() const {
  if (totalBytesSent_ > 0) {
    return totalQueuedBytesMs_ / totalBytesSent_;
  }

  return 0;
}

} // namespace facebook::velox::ucx_exchange
