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
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/PartitionedOutputBufferManager.h"

namespace facebook::velox::exec::test {
namespace {

class LocalExchangeSource : public exec::ExchangeSource {
 public:
  LocalExchangeSource(
      const std::string& taskId,
      int destination,
      std::shared_ptr<exec::ExchangeQueue> queue,
      memory::MemoryPool* pool)
      : ExchangeSource(taskId, destination, queue, pool) {}

  bool supportsFlowControlV2() const override {
    return true;
  }

  bool shouldRequestLocked() override {
    if (atEnd_) {
      return false;
    }
    return !requestPending_.exchange(true);
  }

  folly::SemiFuture<Response> request(
      uint32_t maxBytes,
      uint32_t /*maxWaitSeconds*/) override {
    ++numRequests_;

    auto promise = VeloxPromise<Response>("LocalExchangeSource::request");
    auto future = promise.getSemiFuture();

    if (numRequests_ % 2 == 0) {
      {
        std::lock_guard<std::mutex> l(queue_->mutex());
        requestPending_ = false;
      }
      // Simulate no-data.
      promise.setValue(Response{0, false});
      return future;
    }

    promise_ = std::move(promise);

    auto buffers = PartitionedOutputBufferManager::getInstance().lock();
    VELOX_CHECK_NOT_NULL(buffers, "invalid PartitionedOutputBufferManager");
    VELOX_CHECK(requestPending_);
    auto requestedSequence = sequence_;
    auto self = shared_from_this();
    buffers->getData(
        taskId_,
        destination_,
        maxBytes,
        sequence_,
        // Since this lambda may outlive 'this', we need to capture a
        // shared_ptr to the current object (self).
        [self, requestedSequence, buffers, this](
            std::vector<std::unique_ptr<folly::IOBuf>> data, int64_t sequence) {
          if (requestedSequence > sequence) {
            VLOG(2) << "Receives earlier sequence than requested: task "
                    << taskId_ << ", destination " << destination_
                    << ", requested " << sequence << ", received "
                    << requestedSequence;
            int64_t nExtra = requestedSequence - sequence;
            VELOX_CHECK(nExtra < data.size());
            data.erase(data.begin(), data.begin() + nExtra);
            sequence = requestedSequence;
          }
          std::vector<std::unique_ptr<SerializedPage>> pages;
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
                std::make_unique<SerializedPage>(std::move(inputPage)));
            inputPage = nullptr;
          }
          numPages_ += pages.size();
          totalBytes_ += totalBytes;

          try {
            common::testutil::TestValue::adjust(
                "facebook::velox::exec::test::LocalExchangeSource", &numPages_);
          } catch (const std::exception& e) {
            queue_->setError(e.what());
            checkSetRequestPromise();
            return;
          }

          int64_t ackSequence;
          VeloxPromise<Response> requestPromise;
          {
            std::vector<ContinuePromise> queuePromises;
            {
              std::lock_guard<std::mutex> l(queue_->mutex());
              requestPending_ = false;
              requestPromise = std::move(promise_);
              for (auto& page : pages) {
                queue_->enqueueLocked(std::move(page), queuePromises);
              }
              if (atEnd) {
                queue_->enqueueLocked(nullptr, queuePromises);
                atEnd_ = true;
              }
              ackSequence = sequence_ = sequence + pages.size();
            }
            for (auto& promise : queuePromises) {
              promise.setValue();
            }
          }
          // Outside of queue mutex.
          if (atEnd_) {
            buffers->deleteResults(taskId_, destination_);
          } else {
            buffers->acknowledge(taskId_, destination_, ackSequence);
          }

          if (!requestPromise.isFulfilled()) {
            requestPromise.setValue(Response{totalBytes, atEnd_});
          }
        });

    return future;
  }

  void close() override {
    checkSetRequestPromise();

    auto buffers = PartitionedOutputBufferManager::getInstance().lock();
    buffers->deleteResults(taskId_, destination_);
  }

  folly::F14FastMap<std::string, int64_t> stats() const override {
    return {
        {"localExchangeSource.numPages", numPages_},
        {"localExchangeSource.totalBytes", totalBytes_},
        {ExchangeClient::kBackgroundCpuTimeMs, 123},
    };
  }

 private:
  bool checkSetRequestPromise() {
    VeloxPromise<Response> promise;
    {
      std::lock_guard<std::mutex> l(queue_->mutex());
      promise = std::move(promise_);
    }
    if (promise.valid() && !promise.isFulfilled()) {
      promise.setValue(Response{0, false});
      return true;
    }

    return false;
  }

  // Records the total number of pages fetched from sources.
  std::atomic<int64_t> numPages_{0};
  std::atomic<uint64_t> totalBytes_{0};
  VeloxPromise<Response> promise_{VeloxPromise<Response>::makeEmpty()};
  int32_t numRequests_{0};
};

} // namespace

std::unique_ptr<ExchangeSource> createLocalExchangeSource(
    const std::string& taskId,
    int destination,
    std::shared_ptr<ExchangeQueue> queue,
    memory::MemoryPool* pool) {
  if (strncmp(taskId.c_str(), "local://", 8) == 0) {
    return std::make_unique<LocalExchangeSource>(
        taskId, destination, std::move(queue), pool);
  } else if (strncmp(taskId.c_str(), "bad://", 6) == 0) {
    throw std::runtime_error("Testing error");
  }
  return nullptr;
}

} // namespace facebook::velox::exec::test
