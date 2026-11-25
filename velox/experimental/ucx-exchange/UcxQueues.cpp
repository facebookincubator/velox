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
    std::shared_ptr<cudf::packed_columns> data,
    vector_size_t numRows) {
  // drop duplicate end markers.
  if (data == nullptr && !queue_.empty() && queue_.back().data == nullptr) {
    return;
  }

  if (data != nullptr) {
    stats_.recordEnqueue(data.get());
  }
  queue_.push_back(QueuedPage{std::move(data), numRows});
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

UcxDestinationQueue::Data UcxDestinationQueue::getData(
    UcxDataAvailableCallback notify) {
  if (queue_.empty()) {
    // delay notification.
    notify_ = std::move(notify);
    return {};
  }

  // queue is not empty.
  auto page = std::move(queue_.front());
  queue_.pop_front();
  stats_.recordDequeue(page.data.get());

  std::vector<int64_t> remainingBytes;
  remainingBytes.reserve(queue_.size());
  // fill in the remainingbytes vector.
  for (std::size_t i = 0; i < queue_.size(); ++i) {
    if (queue_[i].data == nullptr) {
      VELOX_CHECK_EQ(i, queue_.size() - 1, "null marker found in the middle");
      break;
    }
    remainingBytes.push_back(queue_[i].data->gpu_data->size());
  }
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

std::vector<std::shared_ptr<cudf::packed_columns>>
UcxDestinationQueue::takeQueuedData() {
  std::vector<std::shared_ptr<cudf::packed_columns>> data;
  data.reserve(queue_.size());
  for (auto& page : queue_) {
    if (page.data != nullptr) {
      stats_.bytesQueued -= page.data->gpu_data->size();
      --stats_.packedColumnsQueued;
      data.push_back(std::move(page.data));
    }
  }
  const bool atEnd = !queue_.empty() && queue_.back().data == nullptr;
  queue_.clear();
  if (atEnd) {
    queue_.push_back(QueuedPage{nullptr, 0});
  }
  return data;
}

UcxDataAvailable UcxDestinationQueue::getAndClearNotify() {
  if (notify_ == nullptr) {
    return UcxDataAvailable();
  }
  UcxDataAvailable result;
  result.callback = notify_;
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

std::string UcxDestinationQueue::toString() {
  std::stringstream out;
  out << "[available: " << queue_.size() << ", "
      << (notify_ ? "notify registered, " : "") << this << "]";
  return out.str();
}

// ---------- UcxOutputQueue ----------

UcxOutputQueue::UcxOutputQueue(std::string taskId)
    : taskId_(std::move(taskId)) {}

bool UcxOutputQueue::initialize(
    std::shared_ptr<exec::Task> task,
    int32_t numDestinations,
    uint32_t numDrivers,
    core::PartitionedOutputNode::Kind kind,
    std::vector<UcxDataAvailable>& notifications) {
  // Checked before any member changes, so that a failed initialization leaves
  // the entry Pending for removeTask() to find.
  VELOX_CHECK_GE(
      numDestinations, 0, "Number of destinations must be non-negative");
  VELOX_CHECK_NOT_NULL(task);
  // Read before taking mutex_, which no Task method may run under.
  const uint64_t maxSize =
      task->queryCtx()->queryConfig().maxOutputBufferSize();
  std::lock_guard<std::mutex> l(mutex_);
  if (state_ != TaskEntryState::kPending) {
    return false;
  }
  kind_ = kind;
  numDrivers_ = numDrivers;
  task_ = std::move(task);
  removedTask_.reset();
  maxSize_ = maxSize;
  continueSize_ = (maxSize_ * kContinuePct) / 100;
  state_.store(TaskEntryState::kActive, std::memory_order_release);
  // Publish task metadata before destination queue expansion. Acceptor only
  // needs task/kind to choose the intra-node path; getData() takes mutex_ and
  // waits for any queue expansion in this function to finish.
  initialized_.store(true, std::memory_order_release);
  addDestinationQueuesLocked(numDestinations, notifications);
  if (kind_ == core::PartitionedOutputNode::Kind::kPartitioned) {
    clearPendingReadersLocked(notifications);
  }
  return true;
}

std::shared_ptr<exec::Task> UcxOutputQueue::task() {
  std::lock_guard<std::mutex> l(mutex_);
  // task_ is only set while Active; the check states the contract.
  return state_ == TaskEntryState::kActive ? task_ : nullptr;
}

void UcxOutputQueue::addBoundReader() {
  std::lock_guard<std::mutex> l(mutex_);
  ++numBoundReaders_;
}

bool UcxOutputQueue::removeBoundReader() {
  std::lock_guard<std::mutex> l(mutex_);
  // A server's close() may run inside a UCX callback, which must not throw.
  if (numBoundReaders_ == 0) {
    LOG(ERROR) << "UCX output queue has no bound reader to release, task "
               << taskId_;
    return false;
  }
  --numBoundReaders_;
  return state_ == TaskEntryState::kPending && numBoundReaders_ == 0 &&
      !failedInitialization_ && !hasReleasedDestinationLocked();
}

bool UcxOutputQueue::hasReleasedDestinationLocked() const {
  for (const auto& [destination, queue] : pendingQueues_) {
    if (queue == nullptr) {
      return true;
    }
  }
  return false;
}

void UcxOutputQueue::recordFailedInitialization(
    std::weak_ptr<exec::Task> task) {
  std::lock_guard<std::mutex> l(mutex_);
  removedTask_ = std::move(task);
  failedInitialization_ = true;
}

void UcxOutputQueue::markRemoved() {
  std::shared_ptr<exec::Task> task;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (state_ == TaskEntryState::kRemoved) {
      return;
    }
    if (task_ != nullptr) {
      removedTask_ = task_;
    }
    // Moved out and released after unlocking, so that ~Task never runs under
    // mutex_.
    task = std::move(task_);
    state_.store(TaskEntryState::kRemoved, std::memory_order_release);
  }
}

bool UcxOutputQueue::isDroppable() {
  std::lock_guard<std::mutex> l(mutex_);
  return state_ == TaskEntryState::kRemoved && removedTask_.expired() &&
      numBoundReaders_ == 0;
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
    vector_size_t numRows) {
  VELOX_CHECK_NOT_NULL(data);
  std::vector<UcxDataAvailable> dataAvailableCallbacks;
  {
    std::lock_guard<std::mutex> l(mutex_);
    // removeTask() terminates the output of a task that failed or was aborted.
    // The entry's state tells, since no Task method may run under mutex_.
    VELOX_CHECK(
        state_ == TaskEntryState::kActive,
        "Cannot add data to output that is not initialized or was removed.");
    VELOX_CHECK(!terminated_, "Task is terminated, cannot add data to output.");
    auto numBytes = data->gpu_data->size();
    auto sharedData = std::shared_ptr<cudf::packed_columns>(std::move(data));

    bool success = false;
    if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast) {
      VELOX_CHECK_EQ(destination, 0, "Broadcast uses destination 0");
      enqueueBroadcastOutputLocked(
          std::move(sharedData), numRows, dataAvailableCallbacks);
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
          destination, std::move(sharedData), numRows, dataAvailableCallbacks);
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
    VLOG(2) << "[BACKPRESSURE] task=" << taskId_
            << " BLOCKED queuedBytes=" << queuedBytes_
            << " maxSize=" << maxSize_
            << " waitingProducers=" << (promises_.size() + 1);
    promises_.emplace_back("UcxOutputQueue::checkBlocked");
    *future = promises_.back().getSemiFuture();
    return true;
  }
  return false;
}

bool UcxOutputQueue::isDestinationCountFinalLocked() const {
  return terminated_ || noMoreQueues_ ||
      (state_ == TaskEntryState::kActive &&
       kind_ == core::PartitionedOutputNode::Kind::kPartitioned);
}

void UcxOutputQueue::getData(int destination, UcxDataAvailableCallback notify) {
  UcxDestinationQueue::Data data;
  std::vector<ContinuePromise> promises;
  {
    std::unique_lock<std::mutex> l(mutex_);
    if (terminated_) {
      l.unlock();
      notify(nullptr, /*numRows=*/0, {});
      return;
    }
    if (destination < 0 ||
        (static_cast<size_t>(destination) >= queues_.size() &&
         isDestinationCountFinalLocked())) {
      const auto numQueues = queues_.size();
      const auto kind = kind_;
      const bool noMoreQueues = noMoreQueues_;
      l.unlock();
      VLOG(1) << "UCX reader outside published destinations: task=" << taskId_
              << " destination=" << destination << " queues=" << numQueues
              << " kind=" << kind << " noMoreQueues=" << noMoreQueues;
      notify(nullptr, /*numRows=*/0, {});
      return;
    }
    UcxDestinationQueue* queue;
    if (static_cast<size_t>(destination) < queues_.size()) {
      queue = queues_[destination].get();
    } else {
      // A reader's ID is not authority to allocate or backfill intermediate
      // destinations. Wait for initialize() or updateOutputBuffers().
      auto [it, inserted] = pendingQueues_.try_emplace(destination);
      if (inserted) {
        it->second = std::make_unique<UcxDestinationQueue>();
      }
      queue = it->second.get();
    }
    // A null queue means deleteResults() has already removed this destination.
    if (queue) {
      // Capture weak_ptr instead of raw `this` to prevent use-after-free.
      // The callback fires outside the lock (from enqueue() or terminate()),
      // and concurrent removeTask() can destroy the UcxOutputQueue while
      // the callback is still executing.
      std::weak_ptr<UcxOutputQueue> weakSelf = shared_from_this();
      data = queue->getData([notify, weakSelf](
                                std::shared_ptr<cudf::packed_columns> data,
                                vector_size_t numRows,
                                std::vector<int64_t> remainingBytes) {
        std::vector<ContinuePromise> promises;
        int64_t bytes = data ? data->gpu_data->size() : -1L;
        notify(std::move(data), numRows, std::move(remainingBytes));
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
      data = UcxDestinationQueue::Data{nullptr, 0, {}, true};
    }
  }
  // outside lock: If we have data, then return it immediately.
  if (data.immediate) {
    notify(std::move(data.data), data.numRows, std::move(data.remainingBytes));
  } else {
    VLOG(2) << "[QUEUE] task=" << taskId_ << " dest=" << destination
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
      VLOG(1) << "[OUTPUT-STATS] task=" << taskId_
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
  VELOX_CHECK_LT(destination, queues_.size());
  bool success = false;
  auto* queue = queues_[destination].get();
  if (queue != nullptr) {
    queue->enqueueBack(std::move(data), numRows);
    dataAvailableCbs.emplace_back(queue->getAndClearNotify());
    success = true;
  }
  return success;
}

void UcxOutputQueue::enqueueBroadcastOutputLocked(
    std::shared_ptr<cudf::packed_columns> data,
    vector_size_t numRows,
    std::vector<UcxDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());

  for (auto& queue : queues_) {
    if (queue != nullptr) {
      queue->enqueueBack(data, numRows);
      dataAvailableCbs.emplace_back(queue->getAndClearNotify());
    }
  }

  // Store for late-arriving destinations (backfill).
  if (!noMoreQueues_) {
    dataToBroadcast_.emplace_back(std::move(data), numRows);
  }
}

bool UcxOutputQueue::isFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  return isFinishedLocked();
}

bool UcxOutputQueue::isFinishedLocked() {
  if (state_ != TaskEntryState::kActive || !pendingQueues_.empty()) {
    return false;
  }
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

void UcxOutputQueue::addDestinationQueuesLocked(
    int32_t numDestinations,
    std::vector<UcxDataAvailable>& notifications) {
  VELOX_CHECK_GE(
      numDestinations, 0, "Number of destinations must be non-negative");
  if (static_cast<size_t>(numDestinations) <= queues_.size()) {
    return;
  }

  const bool isBroadcast =
      kind_ == core::PartitionedOutputNode::Kind::kBroadcast;
  VELOX_CHECK(
      !isBroadcast || !noMoreQueues_,
      "Cannot add broadcast destinations after no more output buffers");

  if (isBroadcast && !dataToBroadcast_.empty()) {
    updateTotalQueuedBytesMsLocked();
  }
  queues_.reserve(numDestinations);
  while (queues_.size() < static_cast<size_t>(numDestinations)) {
    std::unique_ptr<UcxDestinationQueue> queue;
    auto it = pendingQueues_.find(queues_.size());
    if (it != pendingQueues_.end()) {
      queue = std::move(it->second);
      pendingQueues_.erase(it);
    } else {
      queue = std::make_unique<UcxDestinationQueue>();
    }
    if (queue && isBroadcast) {
      for (const auto& [data, numRows] : dataToBroadcast_) {
        queue->enqueueBack(data, numRows);
        // Each destination's dequeue must free the bytes counted by backfill.
        queuedBytes_ += data->gpu_data->size();
        ++queuedPackedColumns_;
      }
      if (atEnd_) {
        queue->enqueueBack(nullptr, /*numRows=*/0);
      }
      if (!dataToBroadcast_.empty() || atEnd_) {
        notifications.push_back(queue->getAndClearNotify());
      }
    }
    queues_.emplace_back(std::move(queue));
  }
}

void UcxOutputQueue::clearPendingReadersLocked(
    std::vector<UcxDataAvailable>& notifications) {
  for (auto& [destination, queue] : pendingQueues_) {
    if (queue) {
      notifications.push_back(queue->getAndClearNotify());
    }
  }
  pendingQueues_.clear();
}

void UcxOutputQueue::updateOutputBuffers(int numBuffers, bool noMoreBuffers) {
  using Kind = core::PartitionedOutputNode::Kind;
  std::vector<UcxDataAvailable> notifications;
  std::shared_ptr<exec::Task> finishedTask;
  size_t numRejected{0};
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (terminated_) {
      return;
    }
    if (kind_ == Kind::kPartitioned) {
      VELOX_CHECK_EQ(queues_.size(), numBuffers);
      VELOX_CHECK(noMoreBuffers);
      noMoreQueues_ = true;
      return;
    }

    VELOX_CHECK_EQ(kind_, Kind::kBroadcast);
    addDestinationQueuesLocked(numBuffers, notifications);

    if (noMoreBuffers) {
      noMoreQueues_ = true;
      dataToBroadcast_.clear();
      const auto priorNotifications = notifications.size();
      clearPendingReadersLocked(notifications);
      numRejected = notifications.size() - priorNotifications;
    }
    if (isFinishedLocked()) {
      finishedTask = task_;
    }
  }
  if (numRejected > 0) {
    VLOG(1) << "UCX reader requests outside final destinations: task="
            << taskId_ << " finalBuffers=" << numBuffers
            << " rejected=" << numRejected;
  }
  for (auto& notification : notifications) {
    notification.notify();
  }
  if (finishedTask) {
    finishedTask->setAllOutputConsumed();
  }
}

void UcxOutputQueue::deleteResults(int destination) {
  std::shared_ptr<exec::Task> finishedTask;
  UcxDataAvailable dataAvailable;
  std::vector<ContinuePromise> promises;
  {
    std::unique_lock<std::mutex> l(mutex_);
    if (destination < 0) {
      return;
    }
    if (static_cast<size_t>(destination) >= queues_.size()) {
      // Only tombstone while the destination can still be published; once the
      // count is final the reader already got EOS and a stray entry would
      // wedge isFinishedLocked().
      const bool publishable = !isDestinationCountFinalLocked();
      auto it = pendingQueues_.find(destination);
      if (it != pendingQueues_.end()) {
        if (it->second) {
          dataAvailable = it->second->getAndClearNotify();
        }
        if (publishable) {
          // Keep a tombstone so publication cannot resurrect deleted results.
          it->second.reset();
        } else {
          pendingQueues_.erase(it);
        }
      } else if (publishable) {
        pendingQueues_.emplace(destination, nullptr);
      }
      l.unlock();
      dataAvailable.notify();
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
    if (isFinishedLocked()) {
      finishedTask = task_;
    }
    // update UcxOutputQueue stats
    updateStatsWithFreedLocked(bytes, packedCols, promises);
  }

  // Outside of mutex.
  dataAvailable.notify();
  // wake up any producers that are waiting for queue to become less full.
  for (auto& promise : promises) {
    promise.setValue();
  }

  if (finishedTask) {
    finishedTask->setAllOutputConsumed();
  }
}

void UcxOutputQueue::terminate() {
  std::vector<UcxDataAvailable> pendingCallbacks;
  std::vector<ContinuePromise> promises;
  std::shared_ptr<exec::Task> task;
  // Chunks no reader will fetch once terminated_ is set. Destroyed after
  // mutex_ is released.
  std::vector<std::shared_ptr<cudf::packed_columns>> discardedData;
  std::vector<std::pair<std::shared_ptr<cudf::packed_columns>, vector_size_t>>
      discardedBroadcastData;
  {
    std::lock_guard<std::mutex> l(mutex_);
    task = task_;
    terminated_ = true;
    // Pending queues hold readers, never data.
    clearPendingReadersLocked(pendingCallbacks);
    int64_t discardedBytes{0};
    int64_t numDiscarded{0};
    // Fire all pending getData callbacks with nullptr to signal end-of-stream.
    // This handles the case where a producer task fails or is cancelled before
    // noMoreData() is called, preventing consumers from being orphaned.
    for (auto& queue : queues_) {
      if (queue != nullptr) {
        for (auto& data : queue->takeQueuedData()) {
          discardedBytes += data->gpu_data->size();
          ++numDiscarded;
          discardedData.push_back(std::move(data));
        }
        queue->enqueueBack(nullptr, /*numRows=*/0);
        pendingCallbacks.push_back(queue->getAndClearNotify());
      }
    }
    // Broadcast keeps these for destinations not added yet; none will be.
    // They are not counted in queuedBytes_.
    discardedBroadcastData = std::move(dataToBroadcast_);
    dataToBroadcast_.clear();
    // Subtracts what was discarded rather than resetting the counters: a
    // chunk that a reader took but whose callback has not run yet is still
    // counted, and that callback subtracts it.
    updateStatsWithFreedLocked(discardedBytes, numDiscarded, promises);
    // Release any outstanding producer-side promises (blocked on queue-full).
    for (auto& promise : promises_) {
      promises.push_back(std::move(promise));
    }
    promises_.clear();
  }

  if (task && task->isRunning()) {
    LOG(WARNING) << "UcxOutputQueue::terminate() called while task " << taskId_
                 << " is still running";
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

std::optional<double> UcxOutputQueue::getUtilization() {
  std::lock_guard<std::mutex> l(mutex_);
  if (maxSize_ == 0) {
    return std::nullopt;
  }
  return queuedBytes_ / static_cast<double>(maxSize_);
}

std::optional<bool> UcxOutputQueue::isOverutilized() {
  std::lock_guard<std::mutex> l(mutex_);
  if (maxSize_ == 0) {
    return std::nullopt;
  }
  return (queuedBytes_ > (0.5 * static_cast<double>(maxSize_))) || atEnd_;
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
    VLOG(2) << "[BACKPRESSURE] task=" << taskId_ << " UNBLOCKING "
            << promises_.size() << " producers"
            << " queuedBytes=" << queuedBytes_
            << " continueSize=" << continueSize_;
    promises = std::move(promises_);
  }
}

void UcxOutputQueue::updateTotalQueuedBytesMsLocked() {
  const auto deltaMs = queueWatch_.lap().count();
  if (queuedBytes_ > 0) {
    totalQueuedBytesMs_ += queuedBytes_ * deltaMs;
  }
}

int64_t UcxOutputQueue::getAverageQueueTimeMsLocked() const {
  if (totalBytesSent_ > 0) {
    return totalQueuedBytesMs_ / totalBytesSent_;
  }

  return 0;
}

} // namespace facebook::velox::ucx_exchange
