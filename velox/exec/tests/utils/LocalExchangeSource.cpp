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
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include <fmt/format.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <optional>
#include <string_view>
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec::test {
namespace {

constexpr std::string_view kParentSeparator{"?parent="};

class LocalExchangeSourceImpl : public LocalExchangeSource {
 public:
  // Local pages arrive from the producer's output buffers. The source does
  // not allocate from the consumer's pool.
  LocalExchangeSourceImpl(
      const std::string& taskId,
      int destination,
      std::shared_ptr<exec::ExchangeQueue> queue)
      : LocalExchangeSource(
            taskId,
            destination,
            std::move(queue),
            std::shared_ptr<memory::MemoryPool>{}) {
    if (const auto separator = taskId.find(kParentSeparator);
        separator != std::string::npos) {
      producerTaskId_ = taskId.substr(0, separator);
      parentTaskId_ = taskId.substr(separator + kParentSeparator.size());
    } else {
      producerTaskId_ = taskId;
    }
  }

  bool supportsMetrics() const override {
    return true;
  }

  bool shouldRequestLocked() override {
    if (closed_ || atEnd_) {
      return false;
    }
    return !requestPending_.exchange(true);
  }

  folly::SemiFuture<Response> request(
      uint32_t maxBytes,
      std::chrono::microseconds maxWait) override {
    // A timeout can complete the promise before getData() returns. Serialize
    // calls to the buffer manager so an older request cannot replace a newer
    // request's notification. close() and the callbacks do not take this lock.
    std::lock_guard<std::mutex> requestLock(requestMutex_);
    auto promise = VeloxPromise<Response>("LocalExchangeSource::request");
    auto future = promise.getSemiFuture();
    // Shared by the timeout and the output buffer's notification. Only the
    // first result for this request can remove its timeout and publish data.
    auto active = std::make_shared<std::atomic_bool>(true);
    int64_t requestedSequence;
    DataAvailableCallback resultCallback;
    {
      std::lock_guard<std::mutex> l(queue_->mutex());
      if (closed_) {
        promise.setValue(Response{0, false, {}});
        return future;
      }
      VELOX_CHECK(requestPending_);
      requestedSequence = sequence_;
      resultCallback = [weak = weak_from_this(), requestedSequence, active](
                           std::vector<std::unique_ptr<folly::IOBuf>> data,
                           int64_t sequence,
                           std::vector<int64_t> remainingBytes) {
        {
          std::lock_guard<std::mutex> l(mutex_);
          if (timeouts_.erase(active) == 0) {
            return;
          }
          *active = false;
        }
        if (auto self = std::static_pointer_cast<LocalExchangeSourceImpl>(
                weak.lock())) {
          self->processResult(
              requestedSequence,
              std::move(data),
              sequence,
              std::move(remainingBytes));
        }
      };
      registerTimeout(resultCallback, active, maxWait);
      activeRequest_ = active;
      promise_ = std::move(promise);
    }

    common::testutil::TestValue::adjust(
        "facebook::velox::exec::test::LocalExchangeSource::request", this);

    if (!active->load()) {
      return future;
    }

    const auto& buffers = DefaultOutputBufferManager::getInstanceRef();
    auto activeCheck = [active]() { return active->load(); };
    if (!parentTaskId_.has_value()) {
      // A producer that does not exist yet answers nothing, and the timeout
      // completes the request.
      buffers->getData(
          producerTaskId_,
          destination_,
          maxBytes,
          requestedSequence,
          std::move(resultCallback),
          std::move(activeCheck));
      return future;
    }

    buffers->getDataWhenCreated(
        *parentTaskId_,
        producerTaskId_,
        destination_,
        maxBytes,
        requestedSequence,
        std::move(resultCallback),
        std::move(activeCheck));

    // Remove a request registered after close() cleared the pending reads.
    bool deleteResults;
    {
      std::lock_guard<std::mutex> l(queue_->mutex());
      deleteResults = closed_ && !atEnd_;
    }
    if (deleteResults) {
      buffers->deleteResultsWhenCreated(
          *parentTaskId_, producerTaskId_, destination_);
    }

    return future;
  }

  folly::SemiFuture<Response> requestDataSizes(
      std::chrono::microseconds maxWait) override {
    return request(0, maxWait);
  }

  void pause() override {
    common::testutil::TestValue::adjust(
        "facebook::velox::exec::test::LocalExchangeSource::pause", nullptr);
    auto buffers = DefaultOutputBufferManager::getInstanceRef();
    VELOX_CHECK_NOT_NULL(buffers, "invalid DefaultOutputBufferManager");
    int64_t ackSequence;
    {
      std::lock_guard<std::mutex> l(queue_->mutex());
      if (closed_) {
        return;
      }
      ackSequence = sequence_;
    }
    buffers->acknowledge(producerTaskId_, destination_, ackSequence);
  }

  void close() override {
    auto promise = VeloxPromise<Response>::makeEmpty();
    std::shared_ptr<std::atomic_bool> active;
    bool atEnd;
    {
      std::lock_guard<std::mutex> l(queue_->mutex());
      if (closed_) {
        return;
      }
      closed_ = true;
      atEnd = atEnd_;
      requestPending_ = false;
      promise = std::move(promise_);
      active = std::move(activeRequest_);
    }
    if (active) {
      std::lock_guard<std::mutex> l(mutex_);
      *active = false;
      timeouts_.erase(active);
    }
    if (promise.valid()) {
      promise.setValue(Response{0, false, {}});
    }
    // At end, the results are already deleted, and the producer may be gone:
    // deleting them again would record a delete that no producer consumes.
    if (atEnd) {
      return;
    }
    auto buffers = DefaultOutputBufferManager::getInstanceRef();
    if (parentTaskId_.has_value()) {
      buffers->deleteResultsWhenCreated(
          *parentTaskId_, producerTaskId_, destination_);
    } else {
      buffers->deleteResults(producerTaskId_, destination_);
    }
  }

  folly::F14FastMap<std::string, RuntimeMetric> metrics() const override {
    return {
        {"localExchangeSource.numPages", RuntimeMetric(numPages_)},
        {"localExchangeSource.totalBytes",
         RuntimeMetric(totalBytes_, RuntimeCounter::Unit::kBytes)},
        {std::string(Operator::kBackgroundCpuTimeNanos),
         RuntimeMetric(123 * 1000000, RuntimeCounter::Unit::kNanos)},
    };
  }

  // Invoked to start exchange source to run. It clears 'stop_' which allows
  // the background timeout executor to run on the first exchange source
  // request.
  static void start() {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_NULL(timerThread_);
    VELOX_CHECK(timeouts_.empty());
    stop_ = false;
  }

  // Invoked to stop the exchange source. It sets 'stop_', clears the current
  // pending request 'timeouts_' and join/destroy 'timeoutCheckExecutor_' if
  // created.
  static void stop() {
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (stop_) {
        VELOX_CHECK(timeouts_.empty());
        return;
      }
      stop_ = true;
      if (timerThread_ == nullptr) {
        VELOX_CHECK(timeouts_.empty());
        return;
      }
      timeouts_.clear();
    }
    timerWakeup_.notify_one();
    timerThread_->join();
    timerThread_.reset();
  }

 private:
  struct Timeout {
    DataAvailableCallback callback;
    std::chrono::system_clock::time_point deadline;
  };

  // Called with the queue mutex held so close() cannot race registration.
  void registerTimeout(
      DataAvailableCallback callback,
      const std::shared_ptr<std::atomic_bool>& active,
      std::chrono::microseconds maxWait) {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(!stop_, "Local exchange source has stopped");

    if (timerThread_ == nullptr) {
      // The lambda references only static state, so it does not capture 'this'
      // and safely outlives any single source on the detached timer thread.
      timerThread_ = std::make_unique<std::thread>([]() {
        std::unique_lock<std::mutex> l(mutex_);
        while (!stop_) {
          const auto now = std::chrono::system_clock::now();
          DataAvailableCallback callback = nullptr;
          std::optional<std::chrono::system_clock::time_point> nextDeadline;
          for (const auto& [active, entry] : timeouts_) {
            if (entry.deadline <= now) {
              callback = entry.callback;
              break;
            }
            if (!nextDeadline.has_value() || entry.deadline < *nextDeadline) {
              nextDeadline = entry.deadline;
            }
          }
          if (callback) {
            // Outside of mutex. The callback removes its entry from
            // 'timeouts_'.
            l.unlock();
            callback({}, 0, {});
            l.lock();
            continue;
          }
          if (nextDeadline.has_value()) {
            timerWakeup_.wait_until(l, *nextDeadline);
          } else {
            timerWakeup_.wait(l);
          }
        }
      });
      if (!exitInitialized_) {
        exitInitialized_ = true;
        atexit([]() { stop(); });
      }
    }
    timeouts_.emplace(
        active,
        Timeout{
            std::move(callback), std::chrono::system_clock::now() + maxWait});
    timerWakeup_.notify_one();
  }

  void processResult(
      int64_t requestedSequence,
      std::vector<std::unique_ptr<folly::IOBuf>> data,
      int64_t sequence,
      std::vector<int64_t> remainingBytes) {
    if (requestedSequence > sequence && !data.empty()) {
      VLOG(2) << "Receives earlier sequence than requested: task "
              << remoteTaskId_ << ", destination " << destination_
              << ", requested " << sequence << ", received "
              << requestedSequence;
      int64_t nExtra = requestedSequence - sequence;
      VELOX_CHECK(nExtra < data.size());
      data.erase(data.cbegin(), data.cbegin() + nExtra);
      sequence = requestedSequence;
    }
    if (data.empty()) {
      sequence = requestedSequence;
    }
    std::vector<std::unique_ptr<SerializedPageBase>> pages;
    bool atEnd = false;
    int64_t totalBytes = 0;
    for (auto& inputPage : data) {
      if (!inputPage) {
        atEnd = true;
        // Keep looping, there could be extra end markers.
        continue;
      }
      totalBytes += inputPage->length();
      inputPage->unshare();
      pages.push_back(
          std::make_unique<PrestoSerializedPage>(std::move(inputPage)));
      inputPage = nullptr;
    }
    if (data.empty()) {
      common::testutil::TestValue::adjust(
          "facebook::velox::exec::test::LocalExchangeSource::timeout", this);
    }

    try {
      common::testutil::TestValue::adjust(
          "facebook::velox::exec::test::LocalExchangeSource", this);
    } catch (const std::exception& e) {
      setRequestError(e.what(), std::current_exception());
      return;
    }

    auto requestPromise = VeloxPromise<Response>::makeEmpty();
    std::vector<ContinuePromise> queuePromises;
    {
      std::lock_guard<std::mutex> l(queue_->mutex());
      if (closed_) {
        return;
      }
      requestPending_ = false;
      requestPromise = std::move(promise_);
      numPages_ += pages.size();
      totalBytes_ += totalBytes;
      for (auto& page : pages) {
        queue_->enqueueLocked(std::move(page), queuePromises);
      }
      if (atEnd) {
        queue_->enqueueLocked(nullptr, queuePromises);
        atEnd_ = true;
      }
      if (!data.empty()) {
        sequence_ = sequence + pages.size();
      }
    }
    for (auto& promise : queuePromises) {
      promise.setValue();
    }
    if (atEnd) {
      DefaultOutputBufferManager::getInstanceRef()->deleteResults(
          producerTaskId_, destination_);
    }

    if (requestPromise.valid()) {
      requestPromise.setValue(
          Response{totalBytes, atEnd, std::move(remainingBytes)});
    }
  }

  void setRequestError(const std::string& error, std::exception_ptr exception) {
    auto requestPromise = VeloxPromise<Response>::makeEmpty();
    std::vector<ContinuePromise> queuePromises;
    {
      std::lock_guard<std::mutex> l(queue_->mutex());
      if (closed_) {
        return;
      }
      requestPromise = std::move(promise_);
      queuePromises = queue_->setErrorLocked(error);
    }
    for (auto& promise : queuePromises) {
      promise.setValue();
    }
    if (requestPromise.valid()) {
      requestPromise.setException(
          folly::exception_wrapper(std::move(exception)));
    }
  }

  static inline std::mutex mutex_;
  static inline folly::F14FastMap<std::shared_ptr<std::atomic_bool>, Timeout>
      timeouts_;
  static inline std::unique_ptr<std::thread> timerThread_;
  // Wakes the timer thread, which waits until the earliest deadline in
  // 'timeouts_', when a timeout is registered or stop() is called.
  static inline std::condition_variable timerWakeup_;
  static inline std::atomic_bool stop_{false};
  static inline bool exitInitialized_{false};

  // Records the total number of pages fetched from sources.
  std::atomic<int64_t> numPages_{0};
  std::atomic<uint64_t> totalBytes_{0};
  std::mutex requestMutex_;
  // Guarded by the queue mutex.
  VeloxPromise<Response> promise_{VeloxPromise<Response>::makeEmpty()};
  std::shared_ptr<std::atomic_bool> activeRequest_;
  bool closed_{false};
  // The task whose output buffer this source reads.
  std::string producerTaskId_;
  // The existing task that creates 'producerTaskId_', when this source may read
  // it before it exists. See subTaskLocation().
  std::optional<std::string> parentTaskId_;
};
} // namespace

// static
std::string LocalExchangeSource::subTaskLocation(
    std::string_view taskId,
    std::string_view parentTaskId) {
  return fmt::format("{}{}{}", taskId, kParentSeparator, parentTaskId);
}

std::unique_ptr<ExchangeSource> createLocalExchangeSource(
    const std::string& taskId,
    int destination,
    std::shared_ptr<ExchangeQueue> queue,
    memory::MemoryPool* /*pool*/) {
  if (strncmp(taskId.c_str(), "local://", 8) == 0) {
    return std::make_unique<LocalExchangeSourceImpl>(
        taskId, destination, std::move(queue));
  } else if (strncmp(taskId.c_str(), "bad://", 6) == 0) {
    throw std::runtime_error("Testing error");
  }
  return nullptr;
}

void testingStartLocalExchangeSource() {
  LocalExchangeSourceImpl::start();
}

void testingShutdownLocalExchangeSource() {
  LocalExchangeSourceImpl::stop();
}

} // namespace facebook::velox::exec::test
