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

#include <folly/ScopeGuard.h>
#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#include "velox/common/base/Exceptions.h"
#include "velox/experimental/ucx-exchange/CommElement.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/tests/FakeUcxPeer.h"
#include "velox/experimental/ucx-exchange/tests/UcxExchangeTestBase.h"

namespace facebook::velox::ucx_exchange {
namespace {

// Requeues itself on every process() until stopped, like an intra-node server
// in WaitingForIntraNodeRetrieve or an intra-node source in
// WaitingForIntraNodeData whose poll finds nothing.
class SelfRequeueingElement final
    : public CommElement,
      public std::enable_shared_from_this<SelfRequeueingElement> {
 public:
  explicit SelfRequeueingElement(std::shared_ptr<Communicator> communicator)
      : CommElement(std::move(communicator)) {}

  void process() override {
    numProcessed_.fetch_add(1, std::memory_order_relaxed);
    if (stop_.load(std::memory_order_acquire)) {
      communicator_->unregister(shared_from_this());
      stopped_.post();
      return;
    }
    communicator_->addToWorkQueue(shared_from_this());
  }

  void onEndpointClosed(std::string_view /*reason*/) override {}

  void onElementFailed(std::string_view /*error*/) override {}

  // Asks the element to stop requeueing and waits until it has unregistered.
  bool stopAndWait(std::chrono::milliseconds timeout) {
    stop_.store(true, std::memory_order_release);
    return stopped_.try_wait_for(timeout);
  }

  uint64_t numProcessed() const {
    return numProcessed_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> numProcessed_{0};
  folly::Baton<> stopped_;
};

// Records when the Communicator tells it that its endpoint closed.
class EndpointClosedProbe final : public CommElement {
 public:
  explicit EndpointClosedProbe(std::shared_ptr<Communicator> communicator)
      : CommElement(std::move(communicator)) {}

  void process() override {}

  void onEndpointClosed(std::string_view /*reason*/) override {
    closed_.post();
  }

  void onElementFailed(std::string_view /*error*/) override {}

  bool waitForClosed(std::chrono::milliseconds timeout) {
    return closed_.try_wait_for(timeout);
  }

 private:
  folly::Baton<> closed_;
};

// Throws from process(), as an element does when a post or a serialization
// fails, and records how the Communicator ends it.
class ThrowingElement final
    : public CommElement,
      public std::enable_shared_from_this<ThrowingElement> {
 public:
  explicit ThrowingElement(std::shared_ptr<Communicator> communicator)
      : CommElement(std::move(communicator)) {}

  void process() override {
    VELOX_FAIL("Injected process() failure");
  }

  void onEndpointClosed(std::string_view /*reason*/) override {
    endpointClosed_.store(true, std::memory_order_release);
  }

  void onElementFailed(std::string_view error) override {
    error_ = std::string(error);
    communicator_->unregister(shared_from_this());
    failed_.post();
  }

  // Returns the error the element was failed with, or nullopt if it was not
  // failed within 'timeout'.
  std::optional<std::string> waitForFailure(std::chrono::milliseconds timeout) {
    if (!failed_.try_wait_for(timeout)) {
      return std::nullopt;
    }
    return error_;
  }

  bool endpointClosed() const {
    return endpointClosed_.load(std::memory_order_acquire);
  }

 private:
  // Written on the Communicator thread before 'failed_' is posted.
  std::string error_;
  folly::Baton<> failed_;
  std::atomic<bool> endpointClosed_{false};
};

// Starts a self-requeueing element and waits until the Communicator is
// spinning on it.
std::shared_ptr<SelfRequeueingElement> startSpinningElement(
    const std::shared_ptr<Communicator>& communicator) {
  auto element = std::make_shared<SelfRequeueingElement>(communicator);
  communicator->registerCommElement(element);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (element->numProcessed() < 1'000 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return element;
}

// Posts an empty action and returns how long it took to run, or nullopt if it
// did not run within 'timeout'.
std::optional<std::chrono::milliseconds> postedActionLatency(
    const std::shared_ptr<Communicator>& communicator,
    std::chrono::milliseconds timeout) {
  auto ran = std::make_shared<std::promise<void>>();
  auto future = ran->get_future();
  const auto start = std::chrono::steady_clock::now();
  communicator->postToLoop([ran]() { ran->set_value(); });
  if (future.wait_for(timeout) != std::future_status::ready) {
    return std::nullopt;
  }
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
}

class CommunicatorStallTest : public UcxExchangeTestBase {};

// Without a self-requeueing element, a posted action runs at once. This is the
// control case for the tests below.
TEST_F(CommunicatorStallTest, postedActionRunsPromptlyWhenIdle) {
  auto communicator = Communicator::getInstance();
  auto latency =
      postedActionLatency(communicator, std::chrono::milliseconds(2'000));
  ASSERT_TRUE(latency.has_value());
  LOG(INFO) << "Posted action latency while idle: " << latency->count()
            << " ms";
  EXPECT_LT(latency->count(), 200);
}

// While an element requeues itself on every process(), a posted action must
// still run.
TEST_F(CommunicatorStallTest, postedActionRunsWhileAnElementRequeuesItself) {
  auto communicator = Communicator::getInstance();
  auto element = startSpinningElement(communicator);
  SCOPE_EXIT {
    EXPECT_TRUE(element->stopAndWait(std::chrono::seconds(5)))
        << "The spinning element did not stop";
  };
  ASSERT_GE(element->numProcessed(), 1'000)
      << "The element never started spinning";

  auto latency =
      postedActionLatency(communicator, std::chrono::milliseconds(2'000));
  EXPECT_TRUE(latency.has_value())
      << "A posted action did not run within 2 s while an element requeued "
         "itself";
}

// The same applies to deferred endpoint cleanup: an element on a closed
// endpoint is told while another element spins.
TEST_F(CommunicatorStallTest, endpointCleanupRunsWhileAnElementRequeuesItself) {
  auto communicator = Communicator::getInstance();
  auto element = startSpinningElement(communicator);
  auto probe = std::make_shared<EndpointClosedProbe>(communicator);
  // A Baton may be waited on only once after it is posted, so the guard waits
  // only if the body did not see the post.
  bool closed = false;
  bool cleanupDeferred = false;
  SCOPE_EXIT {
    EXPECT_TRUE(element->stopAndWait(std::chrono::seconds(5)))
        << "The spinning element did not stop";
    // The cleanup may still be queued, so wait for it before the probe goes.
    if (cleanupDeferred && !closed) {
      EXPECT_TRUE(probe->waitForClosed(std::chrono::seconds(5)));
    }
  };
  ASSERT_GE(element->numProcessed(), 1'000)
      << "The element never started spinning";

  auto endpoint = std::make_shared<EndpointRef>(nullptr);
  endpoint->addCommElem(probe);
  communicator->deferEndpointCleanup(endpoint);
  cleanupDeferred = true;
  closed = probe->waitForClosed(std::chrono::milliseconds(2'000));
  EXPECT_TRUE(closed)
      << "Deferred endpoint cleanup did not run within 2 s while an element "
         "requeued itself";
}

// The same applies to the close receive: a consumer's close message releases
// its destination while another element spins.
TEST_F(CommunicatorStallTest, closeMessageHandledWhileAnElementRequeuesItself) {
  auto communicator = Communicator::getInstance();
  auto element = startSpinningElement(communicator);
  SCOPE_EXIT {
    EXPECT_TRUE(element->stopAndWait(std::chrono::seconds(5)))
        << "The spinning element did not stop";
  };
  ASSERT_GE(element->numProcessed(), 1'000)
      << "The element never started spinning";

  // Binding a reader creates a Pending output queue for a task that has not
  // started.
  const PartitionKey key{getUniqueTaskPrefix() + "closeWhileSpinning", 0};
  auto outputQueue = queueManager()->bindReader(
      key.taskId, key.destination, /*onReleased=*/nullptr);
  SCOPE_EXIT {
    queueManager()->releaseReader(outputQueue, key.destination);
    // A Released destination keeps the Pending output queue, so remove the
    // task to drop it.
    queueManager()->removeTask(key.taskId);
  };
  ASSERT_EQ(outputQueue->destinationState(0), DestinationState::kUnpublished);

  auto peer =
      FakeUcxPeer::connect("127.0.0.1", communicator->getListenerPort());
  const auto message = makeCloseMsg(key);
  ASSERT_EQ(
      peer->sendTag(
          getCloseTag(fnv1a_32(key.toString())),
          &message,
          sizeof(message),
          std::chrono::seconds(3)),
      UCS_OK);

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (outputQueue->destinationState(0) != DestinationState::kReleased &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(outputQueue->destinationState(0), DestinationState::kReleased)
      << "The close message was not handled within 2 s while an element "
         "requeued itself";
}

// An element whose process() throws is failed with the error, not told that
// its endpoint closed: the failure is local, and its peer is still there. The
// Communicator keeps serving every other element on this worker.
TEST_F(CommunicatorStallTest, throwingElementFailsAndLoopContinues) {
  auto communicator = Communicator::getInstance();
  auto element = std::make_shared<ThrowingElement>(communicator);
  communicator->registerCommElement(element);
  EXPECT_EQ(
      element->waitForFailure(std::chrono::milliseconds(2'000)),
      "Injected process() failure");
  EXPECT_FALSE(element->endpointClosed());
  EXPECT_TRUE(
      postedActionLatency(communicator, std::chrono::milliseconds(2'000))
          .has_value())
      << "The Communicator stopped after an element threw";
}

} // namespace
} // namespace facebook::velox::ucx_exchange
