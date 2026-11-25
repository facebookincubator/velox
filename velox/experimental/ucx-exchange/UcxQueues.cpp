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
#include <utility>

namespace facebook::velox::ucx_exchange {

ReaderStep ReaderStep::chunk(
    std::shared_ptr<cudf::packed_columns> chunk,
    vector_size_t numRows,
    std::vector<int64_t> remainingBytes) {
  ReaderStep step{Kind::kChunk};
  step.data = std::move(chunk);
  step.numRows = numRows;
  step.remainingBytes = std::move(remainingBytes);
  return step;
}

ReaderStep ReaderStep::end(ChannelOutcome outcome) {
  ReaderStep step{Kind::kEnd};
  step.outcome = std::move(outcome);
  return step;
}

ReaderStep ReaderStep::stop() {
  return ReaderStep{Kind::kStop};
}

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

std::optional<ReaderStep> UcxDestinationQueue::getData(ReaderCallback notify) {
  if (queue_.empty()) {
    // delay notification.
    notify_ = std::move(notify);
    return std::nullopt;
  }

  // queue is not empty.
  auto page = std::move(queue_.front());
  queue_.pop_front();
  stats_.recordDequeue(page.data.get());

  if (page.data == nullptr) {
    VELOX_CHECK(queue_.empty(), "null marker found in the middle");
    return ReaderStep::end(ChannelOutcome::clean());
  }
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
  return ReaderStep::chunk(
      std::move(page.data), page.numRows, std::move(remainingBytes));
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
  queue_.clear();
  return data;
}

UcxDataAvailable UcxDestinationQueue::takeReadyReader() {
  if (notify_ == nullptr || queue_.empty()) {
    return UcxDataAvailable();
  }
  UcxDataAvailable result;
  result.callback = takeReader();
  result.step = getData(nullptr);
  return result;
}

ReaderCallback UcxDestinationQueue::takeReader() {
  return std::exchange(notify_, nullptr);
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

void UcxDataAvailable::notify() {
  if (callback) {
    VELOX_DCHECK(step.has_value());
    callback(std::move(*step));
  }
}

// ---------- UcxOutputQueue ----------

namespace {
const std::shared_ptr<const ChannelOutcome>& cleanOutcome() {
  static const auto kClean =
      std::make_shared<const ChannelOutcome>(ChannelOutcome::clean());
  return kClean;
}

bool hasEnded(DestinationState state) {
  return state == DestinationState::kEnded ||
      state == DestinationState::kReleased;
}
} // namespace

void UcxOutputQueue::Deferred::run() {
  for (auto& notification : notifications) {
    notification.notify();
  }
  for (auto& callback : onReleased) {
    callback();
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
  if (finishedTask != nullptr) {
    finishedTask->setAllOutputConsumed();
  }
}

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
  Deferred deferred;
  {
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
    // needs task/kind to choose the intra-node path; getData() takes mutex_
    // and waits for any queue expansion in this function to finish.
    initialized_.store(true, std::memory_order_release);
    addDestinationQueuesLocked(numDestinations, deferred);
    if (kind_ == core::PartitionedOutputNode::Kind::kPartitioned) {
      clearPendingReadersLocked(deferred);
    }
  }
  // The manager notifies outside its own lock. Publication releases no
  // destination and frees no chunk, so there is nothing else to defer.
  VELOX_DCHECK(deferred.onReleased.empty());
  VELOX_DCHECK(deferred.promises.empty());
  VELOX_DCHECK(deferred.freedChunks.empty());
  for (auto& notification : deferred.notifications) {
    notifications.push_back(std::move(notification));
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

bool UcxOutputQueue::removeBoundReader(int destination) {
  // Destroyed after the lock is released.
  std::function<void()> onReleased;
  std::lock_guard<std::mutex> l(mutex_);
  // A server's close() may run inside a UCX callback, which must not throw.
  if (numBoundReaders_ == 0) {
    LOG(ERROR) << "UCX output queue has no bound reader to release, task "
               << taskId_;
    return false;
  }
  --numBoundReaders_;
  if (auto it = onReleased_.find(destination); it != onReleased_.end()) {
    onReleased = std::move(it->second);
    onReleased_.erase(it);
  }
  return state_ == TaskEntryState::kPending && numBoundReaders_ == 0 &&
      !failedInitialization_ && !hasReleasedDestinationLocked();
}

bool UcxOutputQueue::hasReleasedDestinationLocked() const {
  for (const auto& [index, destination] : pendingQueues_) {
    if (destination.state == DestinationState::kReleased) {
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

bool UcxOutputQueue::isDroppable() {
  std::lock_guard<std::mutex> l(mutex_);
  return state_ == TaskEntryState::kRemoved && removedTask_.expired() &&
      numBoundReaders_ == 0;
}

DestinationDecision UcxOutputQueue::applyLocked(
    Destination& destination,
    int index,
    DestinationEvent event,
    const std::shared_ptr<const ChannelOutcome>& candidate,
    Deferred& deferred) {
  auto decision = decideDestination(destination.state, event, *candidate);
  if (decision.outcome.has_value()) {
    destination.outcome = *decision.outcome == *candidate
        ? candidate
        : std::make_shared<const ChannelOutcome>(*decision.outcome);
  }
  if (decision.freeChunks && destination.queue != nullptr) {
    auto chunks = destination.queue->takeQueuedData();
    int64_t bytes{0};
    for (const auto& chunk : chunks) {
      bytes += chunk->gpu_data->size();
    }
    updateStatsWithFreedLocked(
        bytes, static_cast<int64_t>(chunks.size()), deferred.promises);
    for (auto& chunk : chunks) {
      deferred.freedChunks.push_back(std::move(chunk));
    }
  }
  const bool released = decision.next == DestinationState::kReleased &&
      destination.state != DestinationState::kReleased;
  destination.state = decision.next;
  if (hasEnded(destination.state) && destination.queue != nullptr) {
    if (auto reader = destination.queue->takeReader()) {
      deferred.notifications.push_back(
          UcxDataAvailable{std::move(reader), endedStep(destination)});
    }
    destination.queue.reset();
  }
  if (released) {
    if (auto it = onReleased_.find(index); it != onReleased_.end()) {
      deferred.onReleased.push_back(std::move(it->second));
      onReleased_.erase(it);
    }
  }
  return decision;
}

ReaderStep UcxOutputQueue::endedStep(const Destination& destination) {
  VELOX_DCHECK(hasEnded(destination.state));
  if (destination.outcome != nullptr &&
      (destination.state == DestinationState::kEnded ||
       !destination.outcome->isClean())) {
    return ReaderStep::end(*destination.outcome);
  }
  // Released without an error: the reader stops, rather than get a clean end
  // that no event decided for it.
  VELOX_DCHECK_EQ(destination.state, DestinationState::kReleased);
  return ReaderStep::stop();
}

void UcxOutputQueue::notifyReadyReaderLocked(
    Destination& destination,
    int index,
    Deferred& deferred) {
  auto ready = destination.queue->takeReadyReader();
  if (ready.callback == nullptr) {
    return;
  }
  const bool endMarkerTaken = ready.step->kind == ReaderStep::Kind::kEnd;
  deferred.notifications.push_back(std::move(ready));
  if (endMarkerTaken) {
    applyLocked(
        destination,
        index,
        DestinationEvent::kEndMarkerTaken,
        cleanOutcome(),
        deferred);
  }
}

UcxOutputQueue::Destination* UcxOutputQueue::findDestinationLocked(int index) {
  if (index < 0) {
    return nullptr;
  }
  if (static_cast<size_t>(index) < queues_.size()) {
    return &queues_[index];
  }
  auto it = pendingQueues_.find(index);
  return it != pendingQueues_.end() ? &it->second : nullptr;
}

std::shared_ptr<const ChannelOutcome> UcxOutputQueue::discardOutcomeLocked(
    const Destination& destination) const {
  // Servers delete a destination's results after it ended. A deletion before
  // that drops the destination's output, as UcxPartitionedOutput does when it
  // runs out of GPU memory, unless every driver finished and nothing that the
  // destination would get is left.
  const bool holdsChunks = destination.state == DestinationState::kUnpublished
      ? !dataToBroadcast_.empty()
      : destination.queue != nullptr &&
          destination.queue->stats().packedColumnsQueued > 0;
  if (atEnd_ && !holdsChunks) {
    return cleanOutcome();
  }
  auto error = fmt::format(
      "Producer task discarded its output before sending all of it. Task: {}",
      taskId_);
  return std::make_shared<const ChannelOutcome>(
      ChannelOutcome::producerError(truncateProducerError(std::move(error))));
}

ChannelOutcome UcxOutputQueue::removalOutcomeLocked(
    const RemovalFacts& facts) const {
  if (state_ == TaskEntryState::kPending) {
    return ChannelOutcome::producerError(
        truncateProducerError(std::string{kProducerNotInitializedError}));
  }
  if (!atEnd_) {
    return ChannelOutcome::producerError(truncateProducerError(
        fmt::format(
            "Producer task ended before sending all its output. Task: {}, state: {}, error: {}",
            facts.taskId,
            facts.state,
            facts.error)));
  }
  // A chunk counts as delivered once a server took it from its queue: the
  // send keeps it alive until it completes.
  int64_t numUndelivered{0};
  for (const auto& destination : queues_) {
    if (destination.queue != nullptr) {
      numUndelivered += destination.queue->stats().packedColumnsQueued;
    }
  }
  if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast &&
      !noMoreQueues_) {
    numUndelivered += dataToBroadcast_.size();
  }
  if (numUndelivered == 0) {
    return ChannelOutcome::clean();
  }
  return ChannelOutcome::producerError(truncateProducerError(
      fmt::format(
          "Producer task ended before its consumers received all its output. Task: {}, state: {}, undelivered packed tables: {}, error: {}",
          facts.taskId,
          facts.state,
          numUndelivered,
          facts.error)));
}

void UcxOutputQueue::remove(const RemovalFacts& facts) {
  // Released after the deferred work, outside the lock, so that ~Task never
  // runs under mutex_.
  std::shared_ptr<exec::Task> task;
  Deferred deferred;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (state_ == TaskEntryState::kRemoved) {
      return;
    }
    removedOutcome_ =
        std::make_shared<const ChannelOutcome>(removalOutcomeLocked(facts));
    for (size_t index = 0; index < queues_.size(); ++index) {
      applyLocked(
          queues_[index],
          index,
          DestinationEvent::kRemoved,
          removedOutcome_,
          deferred);
    }
    for (auto& [index, destination] : pendingQueues_) {
      applyLocked(
          destination,
          index,
          DestinationEvent::kRemoved,
          removedOutcome_,
          deferred);
    }
    // Kept for destinations not added yet; none will be. Not counted in
    // queuedBytes_.
    for (auto& [data, numRows] : dataToBroadcast_) {
      deferred.freedChunks.push_back(std::move(data));
    }
    dataToBroadcast_.clear();
    for (auto& promise : promises_) {
      deferred.promises.push_back(std::move(promise));
    }
    promises_.clear();
    if (task_ != nullptr) {
      removedTask_ = task_;
    }
    task = std::move(task_);
    state_.store(TaskEntryState::kRemoved, std::memory_order_release);
  }
  deferred.run();
}

void UcxOutputQueue::consumerGone(int destination) {
  Deferred deferred;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (destination < 0) {
      return;
    }
    auto* entry = findDestinationLocked(destination);
    if (entry == nullptr) {
      // Once the count is final, the reader of a destination beyond it already
      // ended.
      if (state_ == TaskEntryState::kRemoved ||
          isDestinationCountFinalLocked()) {
        return;
      }
      entry = &pendingQueues_[destination];
    }
    const auto previous = entry->state;
    const auto decision = applyLocked(
        *entry,
        destination,
        DestinationEvent::kConsumerGone,
        cleanOutcome(),
        deferred);
    if (decision.next != previous && isFinishedLocked()) {
      deferred.finishedTask = task_;
    }
  }
  deferred.run();
}

std::optional<std::string> UcxOutputQueue::sendFailed(
    int destination,
    std::string_view status) {
  Deferred deferred;
  bool failTask{false};
  {
    std::lock_guard<std::mutex> l(mutex_);
    auto* entry = findDestinationLocked(destination);
    if (entry == nullptr) {
      return std::nullopt;
    }
    // No setAllOutputConsumed(): the caller fails the task, and a finished
    // task would drop that error.
    failTask = applyLocked(
                   *entry,
                   destination,
                   DestinationEvent::kSendFailed,
                   cleanOutcome(),
                   deferred)
                   .failTask;
  }
  deferred.run();
  if (!failTask) {
    return std::nullopt;
  }
  return fmt::format(
      "Failed to send to consumer over UCX, task {}: {}", taskId_, status);
}

void UcxOutputQueue::setOnReleased(
    int destination,
    std::function<void()> onReleased) {
  // The callback replaced, if any, is destroyed after the lock is released.
  std::lock_guard<std::mutex> l(mutex_);
  std::swap(onReleased_[destination], onReleased);
}

DestinationState UcxOutputQueue::destinationState(int destination) {
  std::lock_guard<std::mutex> l(mutex_);
  const auto* entry = findDestinationLocked(destination);
  return entry != nullptr ? entry->state : DestinationState::kUnpublished;
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
  Deferred deferred;
  {
    std::lock_guard<std::mutex> l(mutex_);
    // The entry's state tells, since no Task method may run under mutex_.
    VELOX_CHECK(
        state_ != TaskEntryState::kPending,
        "Cannot add data to output that is not initialized.");
    // removeTask() ends the output of a task that failed or was aborted.
    VELOX_CHECK(
        state_ != TaskEntryState::kRemoved,
        "Task is terminated, cannot add data to output.");
    auto numBytes = data->gpu_data->size();
    auto sharedData = std::shared_ptr<cudf::packed_columns>(std::move(data));

    if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast) {
      VELOX_CHECK_EQ(destination, 0, "Broadcast uses destination 0");
      // Count queuedBytes_ once per open destination, so that each
      // destination's dequeue symmetrically decrements it. The total sent
      // stats count the logical data once.
      int numOpen{0};
      for (size_t index = 0; index < queues_.size(); ++index) {
        auto& entry = queues_[index];
        if (entry.state == DestinationState::kOpen) {
          entry.queue->enqueueBack(sharedData, numRows);
          notifyReadyReaderLocked(entry, index, deferred);
          ++numOpen;
        }
      }
      updateTotalQueuedBytesMsLocked();
      queuedBytes_ += numBytes * numOpen;
      queuedPackedColumns_ += numOpen;
      totalBytesSent_ += numBytes;
      totalRowsSent_ += numRows;
      totalPackedColumnsSent_++;
      // Stored for late-arriving destinations (backfill).
      if (!noMoreQueues_) {
        dataToBroadcast_.emplace_back(std::move(sharedData), numRows);
      } else {
        deferred.freedChunks.push_back(std::move(sharedData));
      }
    } else {
      VELOX_CHECK_LT(destination, queues_.size());
      auto& entry = queues_[destination];
      if (entry.state == DestinationState::kOpen) {
        entry.queue->enqueueBack(std::move(sharedData), numRows);
        notifyReadyReaderLocked(entry, destination, deferred);
        updateStatsWithEnqueuedLocked(numBytes, numRows);
      } else {
        // The destination ended or was released; nobody reads the chunk.
        deferred.freedChunks.push_back(std::move(sharedData));
      }
    }
  }
  // Now that data is enqueued, notify blocked readers (outside of mutex.)
  deferred.run();
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
  return state_ == TaskEntryState::kRemoved || noMoreQueues_ ||
      (state_ == TaskEntryState::kActive &&
       kind_ == core::PartitionedOutputNode::Kind::kPartitioned);
}

void UcxOutputQueue::getData(int destination, ReaderCallback notify) {
  std::optional<ReaderStep> step;
  Deferred deferred;
  {
    std::unique_lock<std::mutex> l(mutex_);
    if (state_ == TaskEntryState::kRemoved && destination >= 0) {
      const auto* entry = findDestinationLocked(destination);
      step = entry != nullptr && hasEnded(entry->state)
          ? endedStep(*entry)
          : ReaderStep::end(*removedOutcome_);
    } else if (
        destination < 0 ||
        (static_cast<size_t>(destination) >= queues_.size() &&
         isDestinationCountFinalLocked())) {
      // No rows are ever produced for such a destination.
      VLOG(1) << "UCX reader outside published destinations: task=" << taskId_
              << " destination=" << destination << " queues=" << queues_.size()
              << " kind=" << kind_ << " noMoreQueues=" << noMoreQueues_;
      step = ReaderStep::end(ChannelOutcome::clean());
    } else {
      Destination* entry;
      if (static_cast<size_t>(destination) < queues_.size()) {
        entry = &queues_[destination];
      } else {
        // A reader's ID is not authority to allocate or backfill intermediate
        // destinations. Wait for initialize() or updateOutputBuffers().
        entry = &pendingQueues_[destination];
        if (entry->state == DestinationState::kUnpublished &&
            entry->queue == nullptr) {
          entry->queue = std::make_unique<UcxDestinationQueue>();
        }
      }
      if (hasEnded(entry->state)) {
        step = endedStep(*entry);
      } else {
        // Capture weak_ptr instead of raw `this` to prevent use-after-free.
        // The callback fires outside the lock, and the last reference to the
        // UcxOutputQueue can go while the callback is still executing.
        std::weak_ptr<UcxOutputQueue> weakSelf = shared_from_this();
        step = entry->queue->getData([notify, weakSelf](ReaderStep received) {
          std::vector<ContinuePromise> promises;
          int64_t bytes = received.data ? received.data->gpu_data->size() : -1L;
          notify(std::move(received));
          if (bytes >= 0L) {
            auto self = weakSelf.lock();
            if (!self) {
              // The entry is gone, safe to skip the stats update.
              return;
            }
            std::lock_guard<std::mutex> l(self->mutex_);
            self->updateStatsWithFreedLocked(bytes, 1L, promises);
          }
          // outside of lock:
          // wake up any producers that are waiting for queue to become less
          // full.
          for (auto& promise : promises) {
            promise.setValue();
          }
        });
        if (!step.has_value()) {
          VLOG(2) << "[QUEUE] task=" << taskId_ << " dest=" << destination
                  << " server waiting for data (callback installed)";
        } else if (step->kind == ReaderStep::Kind::kChunk) {
          // No notify upcall will be done. Need to update the stats here.
          updateStatsWithFreedLocked(
              step->data->gpu_data->size(), 1L, deferred.promises);
        } else {
          applyLocked(
              *entry,
              destination,
              DestinationEvent::kEndMarkerTaken,
              cleanOutcome(),
              deferred);
        }
      }
    }
  }
  // outside lock: If we have data, then return it immediately.
  if (step.has_value()) {
    notify(std::move(*step));
  }
  deferred.run();
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
  Deferred deferred;
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
    for (size_t index = 0; index < queues_.size(); ++index) {
      auto& entry = queues_[index];
      if (entry.state == DestinationState::kOpen) {
        entry.queue->enqueueBack(nullptr, /*numRows=*/0);
        notifyReadyReaderLocked(entry, index, deferred);
      }
    }
    // Every destination may have been released before the last driver
    // finished, for instance before the task initialized. A driver calls
    // noMoreData() without holding the task's lock; updateNumDrivers() runs
    // under it, so its path must not report the output consumed.
    if (oneDriverFinished && isFinishedLocked()) {
      deferred.finishedTask = task_;
    }
  }
  // Notify outside of mutex.
  deferred.run();
}

bool UcxOutputQueue::isFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  return isFinishedLocked();
}

bool UcxOutputQueue::isFinishedLocked() {
  if (state_ != TaskEntryState::kActive) {
    return false;
  }
  for (const auto& [index, destination] : pendingQueues_) {
    if (destination.state == DestinationState::kUnpublished) {
      return false;
    }
  }
  // For broadcast, we can only be finished after receiving the no more
  // (destination) buffers signal, matching OutputBuffer::isFinishedLocked().
  if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast &&
      !noMoreQueues_) {
    return false;
  }
  for (const auto& destination : queues_) {
    if (destination.state != DestinationState::kReleased) {
      return false;
    }
  }
  return true;
}

void UcxOutputQueue::addDestinationQueuesLocked(
    int32_t numDestinations,
    Deferred& deferred) {
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
    const int index = queues_.size();
    Destination destination;
    auto it = pendingQueues_.find(index);
    if (it != pendingQueues_.end()) {
      destination = std::move(it->second);
      pendingQueues_.erase(it);
    }
    if (destination.state == DestinationState::kUnpublished &&
        destination.queue == nullptr) {
      destination.queue = std::make_unique<UcxDestinationQueue>();
    }
    // A destination released before publication stays Released.
    applyLocked(
        destination,
        index,
        DestinationEvent::kPublished,
        cleanOutcome(),
        deferred);
    if (destination.state == DestinationState::kOpen && isBroadcast) {
      for (const auto& [data, numRows] : dataToBroadcast_) {
        destination.queue->enqueueBack(data, numRows);
        // Each destination's dequeue must free the bytes counted by backfill.
        queuedBytes_ += data->gpu_data->size();
        ++queuedPackedColumns_;
      }
      if (atEnd_) {
        destination.queue->enqueueBack(nullptr, /*numRows=*/0);
      }
      notifyReadyReaderLocked(destination, index, deferred);
    }
    queues_.emplace_back(std::move(destination));
  }
}

void UcxOutputQueue::clearPendingReadersLocked(Deferred& deferred) {
  for (auto& [index, destination] : pendingQueues_) {
    if (destination.state != DestinationState::kUnpublished) {
      continue;
    }
    if (destination.queue == nullptr) {
      continue;
    }
    if (auto reader = destination.queue->takeReader()) {
      // A reader for a destination beyond the final count ends cleanly at
      // once: no rows are ever produced for it.
      deferred.notifications.push_back(
          UcxDataAvailable{
              std::move(reader), ReaderStep::end(ChannelOutcome::clean())});
    }
  }
  pendingQueues_.clear();
}

void UcxOutputQueue::updateOutputBuffers(int numBuffers, bool noMoreBuffers) {
  using Kind = core::PartitionedOutputNode::Kind;
  Deferred deferred;
  size_t numRejected{0};
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (state_ == TaskEntryState::kRemoved) {
      return;
    }
    if (kind_ == Kind::kPartitioned) {
      VELOX_CHECK_EQ(queues_.size(), numBuffers);
      VELOX_CHECK(noMoreBuffers);
      noMoreQueues_ = true;
      return;
    }

    VELOX_CHECK_EQ(kind_, Kind::kBroadcast);
    addDestinationQueuesLocked(numBuffers, deferred);

    if (noMoreBuffers) {
      noMoreQueues_ = true;
      for (auto& [data, numRows] : dataToBroadcast_) {
        deferred.freedChunks.push_back(std::move(data));
      }
      dataToBroadcast_.clear();
      const auto priorNotifications = deferred.notifications.size();
      clearPendingReadersLocked(deferred);
      numRejected = deferred.notifications.size() - priorNotifications;
    }
    if (isFinishedLocked()) {
      deferred.finishedTask = task_;
    }
  }
  if (numRejected > 0) {
    VLOG(1) << "UCX reader requests outside final destinations: task="
            << taskId_ << " finalBuffers=" << numBuffers
            << " rejected=" << numRejected;
  }
  deferred.run();
}

void UcxOutputQueue::deleteResults(int destination) {
  Deferred deferred;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (destination < 0) {
      return;
    }
    auto* entry = findDestinationLocked(destination);
    if (entry == nullptr) {
      // Once the count is final, the reader of a destination beyond it already
      // ended, and an entry for it would wedge isFinishedLocked().
      if (state_ == TaskEntryState::kRemoved ||
          isDestinationCountFinalLocked()) {
        return;
      }
      // Kept, so that publication cannot resurrect deleted results.
      entry = &pendingQueues_[destination];
    }
    const auto previous = entry->state;
    const auto decision = applyLocked(
        *entry,
        destination,
        DestinationEvent::kResultsDeleted,
        discardOutcomeLocked(*entry),
        deferred);
    if (decision.next == previous) {
      VLOG(1) << "Extra delete received for destination " << destination;
      return;
    }
    if (isFinishedLocked()) {
      deferred.finishedTask = task_;
    }
  }
  deferred.run();
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
    for (auto& promise : promises_) {
      promises.push_back(std::move(promise));
    }
    promises_.clear();
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
