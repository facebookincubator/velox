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
#include "velox/experimental/cudf-exchange/CudfQueues.h"

namespace facebook::velox::cudf_exchange {

void CudfDestinationQueue::Stats::recordEnqueue(
    const cudf::packed_columns* data) {
  if (data != nullptr) {
    bytesQueued += data->gpu_data->size();
    packedColumnsQueued++;
  }
}

void CudfDestinationQueue::Stats::recordDequeue(
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

void CudfDestinationQueue::enqueueBack(
    std::unique_ptr<cudf::packed_columns> data) {
  // drop duplicate end markers.
  if (data == nullptr && !queue_.empty() && queue_.back() == nullptr) {
    return;
  }

  if (data != nullptr) {
    stats_.recordEnqueue(data.get());
  }
  queue_.push_back(std::move(data));
}

void CudfDestinationQueue::enqueueFront(
    std::unique_ptr<cudf::packed_columns> data) {
  // ignore nullptr.
  if (data == nullptr) {
    return;
  }

  // insert at the front.
  queue_.push_front(std::move(data));
}

CudfDestinationQueue::Data CudfDestinationQueue::getData(
    CudfDataAvailableCallback notify) {
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

void CudfDestinationQueue::deleteResults() {
  for (auto i = 0; i < queue_.size(); ++i) {
    if (queue_[i] == nullptr) {
      VELOX_CHECK_EQ(i, queue_.size() - 1, "null marker found in the middle");
      break;
    }
  }
  queue_.clear();
}

CudfDataAvailable CudfDestinationQueue::getAndClearNotify() {
  if (notify_ == nullptr) {
    return CudfDataAvailable();
  }
  CudfDataAvailable result;
  result.callback = notify_;
  auto data = getData(nullptr);
  result.data = std::move(data.data);
  result.remainingBytes = std::move(data.remainingBytes);
  clearNotify();
  return result;
}

void CudfDestinationQueue::clearNotify() {
  notify_ = nullptr;
}

void CudfDestinationQueue::finish() {
  VELOX_CHECK_NULL(notify_, "notify must be cleared before finish");
  VELOX_CHECK(queue_.empty(), "data must be fetched before finish");
}

CudfDestinationQueue::Stats CudfDestinationQueue::stats() const {
  return stats_;
}

std::string CudfDestinationQueue::toString() {
  std::stringstream out;
  out << "[available: " << queue_.size() << ", "
      << (notify_ ? "notify registered, " : "") << this << "]";
  return out.str();
}

// ---------- CudfOutputQueue ----------

CudfOutputQueue::CudfOutputQueue(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers)
    : task_(task), numDrivers_(numDrivers) {
  if (task_) {
    maxSize_ = task_->queryCtx()->queryConfig().maxOutputBufferSize();
    continueSize_ = (maxSize_ * kContinuePct) / 100;
  } // else: maxSize_ and continueSize_ will be set once the task is created and
    // initialize called.
  // create a queue for each destination.
  queues_.reserve(numDestinations);
  for (int i = 0; i < numDestinations; ++i) {
    // create the destination queues inside the vector using emplace_back.
    queues_.emplace_back(std::make_unique<CudfDestinationQueue>());
  }
}

bool CudfOutputQueue::initialize(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers) {
  std::lock_guard<std::mutex> l(mutex_);
  if (task_) {
    // already initialized!
    return false;
  }
  task_ = task;
  numDrivers_ = numDrivers;
  maxSize_ = task_->queryCtx()->queryConfig().maxOutputBufferSize();
  continueSize_ = (maxSize_ * kContinuePct) / 100;
  // create additional queues if there are more destinations.
  for (int i = queues_.size(); i < numDestinations; ++i) {
    // create the destination queues inside the vector using emplace_back.
    queues_.emplace_back(std::make_unique<CudfDestinationQueue>());
  }
  return true;
}

void CudfOutputQueue::updateNumDrivers(uint32_t newNumDrivers) {
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

void CudfOutputQueue::enqueue(
    int destination,
    std::unique_ptr<cudf::packed_columns> data,
    int32_t numRows) {
  VELOX_CHECK_NOT_NULL(data);
  VELOX_CHECK_NOT_NULL(task_);
  VELOX_CHECK(
      task_->isRunning(), "Task is terminated, cannot add data to output.");
  std::vector<CudfDataAvailable> dataAvailableCallbacks;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_LT(destination, queues_.size());

    // TODO: Support other output modes as well. This is only for partitioned.
    auto numBytes = data->gpu_data->size();
    if (enqueuePartitionedOutputLocked(
            destination, std::move(data), dataAvailableCallbacks)) {
      // enqueueing was successful - update the stats.
      updateStatsWithEnqueuedLocked(numBytes, numRows);
    }
  }
  // Now that data is enqueued, notify blocked readers (outside of mutex.)
  for (auto& callback : dataAvailableCallbacks) {
    callback.notify();
  }
}

bool CudfOutputQueue::checkBlocked(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  if (queuedBytes_ >= maxSize_ && future) {
    promises_.emplace_back("CudfOutputQueue::checkBlocked");
    *future = promises_.back().getSemiFuture();
    return true;
  }
  return false;
}

void CudfOutputQueue::getData(
    int destination,
    CudfDataAvailableCallback notify) {
  CudfDestinationQueue::Data data;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    // If the queue doesn't exist yet, create an empty queue to store
    // the notify callback. The queue will eventually be initialized when
    // the task is being created.
    for (int i = queues_.size(); i <= destination; ++i) {
      // create the destination queues inside the vector using emplace_back.
      queues_.emplace_back(std::make_unique<CudfDestinationQueue>());
    }
    auto* queue = queues_[destination].get();
    // queue can be nullptr here if the task has terminated and results
    // have been removed. In this case, no data is returned.
    if (queue) {
      data = queue->getData([notify, this](
                                std::unique_ptr<cudf::packed_columns> data,
                                std::vector<int64_t> remainingBytes) {
        std::vector<ContinuePromise> promises;
        int64_t bytes = data ? data->gpu_data->size() : -1L;
        notify(std::move(data), std::move(remainingBytes));
        if (bytes >= 0L) {
          std::lock_guard<std::mutex> l(mutex_);
          this->updateStatsWithFreedLocked(bytes, 1L, promises);
        }
        // outside of lock:
        // wake up any producers that are waiting for queue to become less full.
        if (promises.empty()) {
          VLOG(3) << "No waiting producers in task: " << task_->taskId();
        } else {
          VLOG(3) << "Waking up producers in task: " << task_->taskId();
        }
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
      data = CudfDestinationQueue::Data{nullptr, {}, true};
    }
  }
  // outside lock: If we have data, then return it immediately.
  if (data.immediate) {
    notify(std::move(data.data), std::move(data.remainingBytes));
  }
  // wake up any producers that are waiting for queue to become less full.
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void CudfOutputQueue::noMoreData() {
  // Increment number of finished drivers.
  checkIfDone(true);
}

void CudfOutputQueue::noMoreDrivers() {
  // Do not increment number of finished drivers.
  checkIfDone(false);
}

void CudfOutputQueue::checkIfDone(bool oneDriverFinished) {
  std::vector<CudfDataAvailable> finished;
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

bool CudfOutputQueue::enqueuePartitionedOutputLocked(
    int destination,
    std::unique_ptr<cudf::packed_columns> data,
    std::vector<CudfDataAvailable>& dataAvailableCbs) {
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

bool CudfOutputQueue::isFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  return isFinishedLocked();
}

bool CudfOutputQueue::isFinishedLocked() {
  for (auto& queue : queues_) {
    if (queue != nullptr) {
      return false;
    }
  }
  return true;
}

void CudfOutputQueue::deleteResults(int destination) {
  bool isFinished;
  CudfDataAvailable dataAvailable;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_LT(destination, queues_.size());
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
    // update CudfOutputQueue stats
    updateStatsWithFreedLocked(bytes, packedCols, promises);
  }

  // Outside of mutex.
  dataAvailable.notify();
  // wake up any producers that are waiting for queue to become less full.
  for (auto& promise : promises) {
    promise.setValue();
  }

  if (isFinished) {
    task_->setAllOutputConsumed();
  }
}

void CudfOutputQueue::terminate() {
  if (task_) {
    VELOX_CHECK(!task_->isRunning());
    // TODO: When support for queue-full is added, this must
    // release the outstanding promises.
  }
}

exec::OutputBuffer::Stats CudfOutputQueue::stats() {
  std::lock_guard<std::mutex> l(mutex_);
  std::vector<CudfDestinationQueue::Stats> queueStats;

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

void CudfOutputQueue::updateStatsWithEnqueuedLocked(
    int64_t bytes,
    int64_t rows) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ += bytes;
  queuedPackedColumns_++;

  totalBytesSent_ += bytes;
  totalRowsSent_ += rows;
  totalPackedColumnsSent_++;
}

void CudfOutputQueue::updateStatsWithFreedLocked(
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
  if (queuedBytes_ <= continueSize_) {
    promises = std::move(promises_);
  }
}

void CudfOutputQueue::updateTotalQueuedBytesMsLocked() {
  const auto nowMs = getCurrentTimeMs();
  if (queuedBytes_ > 0) {
    const auto deltaMs = nowMs - queueStartMs_;
    totalQueuedBytesMs_ += queuedBytes_ * deltaMs;
  }

  queueStartMs_ = nowMs;
}

int64_t CudfOutputQueue::getAverageQueueTimeMsLocked() const {
  if (totalBytesSent_ > 0) {
    return totalQueuedBytesMs_ / totalBytesSent_;
  }

  return 0;
}

} // namespace facebook::velox::cudf_exchange
