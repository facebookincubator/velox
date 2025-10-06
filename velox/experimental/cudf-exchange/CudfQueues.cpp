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
    rowsQueued++; // FIXME: Add #rows as input param
    packedColumnsQueued++;
  }
}

void CudfDestinationQueue::Stats::recordDequeue(
    const cudf::packed_columns* data) {
  if (data != nullptr) {
    bytesSent += data->gpu_data->size();
    rowsSent++; // FIXME: Add #rows as input param
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

std::vector<std::unique_ptr<cudf::packed_columns>>
CudfDestinationQueue::deleteResults() {
  std::vector<std::unique_ptr<cudf::packed_columns>> freed;
  for (auto i = 0; i < queue_.size(); ++i) {
    if (queue_[i] == nullptr) {
      VELOX_CHECK_EQ(i, queue_.size() - 1, "null marker found in the middle");
      break;
    }
    freed.push_back(std::move(queue_[i]));
  }
  queue_.clear();
  return freed;
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
  stats_.finished = true;
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
    int numDestinations,
    uint32_t numDrivers)
    : task_(task), numDrivers_(numDrivers) {
  // create a queue for each destination.
  queues_.reserve(numDestinations);
  for (int i = 0; i < numDestinations; ++i) {
    // create the destination queues inside the vector using emplace_back.
    queues_.emplace_back(std::make_unique<CudfDestinationQueue>());
  }
}

bool CudfOutputQueue::initialize(
    std::shared_ptr<exec::Task> task,
    int numDestinations,
    uint32_t numDrivers) {
  std::lock_guard<std::mutex> l(mutex_);
  if (task_) {
    // already initialized!
    return false;
  }
  task_ = task;
  numDrivers_ = numDrivers;
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

bool CudfOutputQueue::enqueue(
    int destination,
    std::unique_ptr<cudf::packed_columns> data,
    ContinueFuture* future) {
  VELOX_CHECK_NOT_NULL(data);
  VELOX_CHECK(
      task_->isRunning(), "Task is terminated, cannot add data to output.");
  std::vector<CudfDataAvailable> dataAvailableCallbacks;
  bool blocked = false;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_LT(destination, queues_.size());

    // TODO: Support other output modes as well. This is only for partitioned.
    enqueuePartitionedOutputLocked(
        destination, std::move(data), dataAvailableCallbacks);

    // FIXME: Add a check whether queue exceeds its limit.
  }
  // Now that data is enqueued, notify blocked readers (outside of mutex.)

  for (auto& callback : dataAvailableCallbacks) {
    callback.notify();
  }
  return blocked;
}

void CudfOutputQueue::getData(
    int destination,
    CudfDataAvailableCallback notify) {
  CudfDestinationQueue::Data data;
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
    data = queue ? queue->getData(notify)
                 : CudfDestinationQueue::Data{nullptr, {}, true};
  }
  // outside lock: If we have data, then return it immediately.
  if (data.immediate) {
    notify(std::move(data.data), std::move(data.remainingBytes));
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

void CudfOutputQueue::enqueuePartitionedOutputLocked(
    int destination,
    std::unique_ptr<cudf::packed_columns> data,
    std::vector<CudfDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());
  VELOX_CHECK_LT(destination, queues_.size());
  auto* queue = queues_[destination].get();
  if (queue != nullptr) {
    queue->enqueueBack(std::move(data));
    dataAvailableCbs.emplace_back(queue->getAndClearNotify());
  } else {
    // Some downstream tasks may finish early and delete the corresponding
    // queues. Further data for these queues is dropped.
    // updateStatsWithFreedPagesLocked(1, data->size());
  }
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

bool CudfOutputQueue::deleteResults(int destination) {
  bool isFinished;
  CudfDataAvailable dataAvailable;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_LT(destination, queues_.size());
    auto* queue = queues_[destination].get();
    if (queue == nullptr) {
      VLOG(1) << "Extra delete received for destination " << destination;
      return false;
    }
    std::move(queue->deleteResults());
    dataAvailable = queue->getAndClearNotify();
    queue->finish();
    queues_[destination] = nullptr;
    isFinished = isFinishedLocked();
  }

  // Outside of mutex.
  dataAvailable.notify();

  if (isFinished) {
    task_->setAllOutputConsumed();
  }
  return isFinished;
}

void CudfOutputQueue::terminate() {
  if (task_) {
    VELOX_CHECK(!task_->isRunning());
    // TODO: When support for queue-full is added, this must
    // release the outstanding promises.
  }
}

} // namespace facebook::velox::cudf_exchange
