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
#include <folly/executors/IOThreadPoolExecutor.h>
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/Exchange.h"

namespace facebook::velox::exec {

namespace {

class TestingExchangeSource : public ExchangeSource {
 public:
  TestingExchangeSource(
      const std::string& taskId,
      int destination,
      std::shared_ptr<ExchangeQueue> queue,
      memory::MemoryPool* pool)
      : ExchangeSource(taskId, destination, queue, pool) {}

  bool shouldRequestLocked() override {
    if (atEnd_) {
      return false;
    }
    return !requestPending_.exchange(true);
  }

  void request() override {
    request(2);
  }

  void request(uint64_t bytes) override {
    VELOX_CHECK(requestPending_);
    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(queue_->mutex());
      requestPending_ = false;
      if (sentBytes_ == 10) {
        queue_->enqueueLocked(nullptr, promises);
        atEnd_ = true;
      } else {
        auto page =
            std::make_unique<SerializedPage>(folly::IOBuf::copyBuffer("a", 1));
        sentBytes_ += page->size();
        queue_->recordReplyLocked(page->size());
        queue_->enqueueLocked(std::move(page), promises);
      }
    }
    for (auto& promise : promises) {
      promise.setValue();
    }
  }

  void close() override {}

  folly::F14FastMap<std::string, int64_t> stats() const override {
    return {{"sentBytes", sentBytes_}};
  }

  int sentBytes_{0};
};

TEST(ExchangeClientTest, nonVeloxCreateExchangeSourceException) {
  std::shared_ptr<memory::MemoryPool> rootPool{
      memory::defaultMemoryManager().addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool{rootPool->addLeafChild("leaf")};

  ExchangeSource::registerFactory(
      [](const auto& taskId, auto destination, auto queue, auto pool)
          -> std::shared_ptr<ExchangeSource> {
        if (taskId.find("$$") != std::string::npos) {
          throw std::runtime_error("Testing error");
        }
        return nullptr;
      });

  ExchangeClient client(1, pool.get());
  VELOX_ASSERT_THROW(
      client.addRemoteTaskId("$$task.1.2.3"),
      "Failed to create ExchangeSource: Testing error. Task ID: $$task.1.2.3.");

  // Test with a very long task ID. Make sure it is truncated.
  VELOX_ASSERT_THROW(
      client.addRemoteTaskId(std::string(1024, 'x') + "$$"),
      "Failed to create ExchangeSource: Testing error. "
      "Task ID: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.");
}

TEST(ExchangeClientTest, basic) {
  auto executor = std::make_unique<folly::IOThreadPoolExecutor>(10);
  std::shared_ptr<memory::MemoryPool> rootPool{
      memory::defaultMemoryManager().addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool{rootPool->addLeafChild("leaf")};

  ExchangeSource::registerFactory(
      [](const std::string& taskId,
         int destination,
         std::shared_ptr<velox::exec::ExchangeQueue> queue,
         velox::memory::MemoryPool* pool) {
        return std::make_unique<TestingExchangeSource>(
            taskId, destination, queue, pool);
      });
  int bufferSize = 32 * 1024 * 1024;
  auto client = ExchangeClient::create(1, pool.get(), bufferSize);
  client->addRemoteTaskId("0");
  client->addRemoteTaskId("1");
  client->noMoreRemoteTasks();
  VELOX_CHECK_EQ(client->queue()->totalBytes(), 2);

  bool atEnd = false;
  ContinueFuture dataFuture;
  auto page = client->next(&atEnd, &dataFuture);
  // next() will add 2 pages, and dequeue 1 page.
  VELOX_CHECK_EQ(client->queue()->totalBytes(), 3);

  for (int i = 0; i < 19; ++i) {
    auto page = client->next(&atEnd, &dataFuture);
    VELOX_CHECK_EQ(page->size(), 1);
  }
  VELOX_CHECK_NULL(client->next(&atEnd, &dataFuture));
  VELOX_CHECK_EQ(client->queue()->numCompleted(), 2);
  int expectedSize = (bufferSize + 20) / 22;
  VELOX_CHECK_EQ(client->queue()->expectedSerializedPageSize(), expectedSize);
  VELOX_CHECK_EQ(client->stats()["peakBytes"].max, 11);
  VELOX_CHECK(atEnd);
}

} // namespace
} // namespace facebook::velox::exec
