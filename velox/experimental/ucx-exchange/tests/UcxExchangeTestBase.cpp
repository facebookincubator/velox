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
#include "velox/experimental/ucx-exchange/tests/UcxExchangeTestBase.h"

#include <fmt/format.h>

#include "velox/common/memory/Memory.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/tests/utils/Ports.h"

namespace facebook::velox::ucx_exchange {

void UcxExchangeTestEnvironment::SetUp() {
  initializeExchangeState();
}

void UcxExchangeTestEnvironment::TearDown() {
  destroyExchangeState();
}

void UcxExchangeTestEnvironment::initializeExchangeState() {
  VLOG(0) << "setting up UCX exchange test environment";
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});

  // UcxExchangeSource derives the UCX port as the split URL's port + 3, and
  // remoteSplit() advertises splitUrlPort() to match, so the offset must stay
  // below the chosen port.
  communicatorPort_ = exec::test::Ports::findFreePort();
  ASSERT_GT(communicatorPort_, kSplitUrlPortOffset);

  queueManager_ = UcxOutputQueueManager::getInstanceRef();
  ContinueFuture future;
  communicator_ = facebook::velox::ucx_exchange::Communicator::initAndGet(
      communicatorPort_, std::string(kUnusedCoordinatorUrl), &future);
  if (communicator_) {
    communicatorThread_ = std::make_shared<std::thread>(
        &facebook::velox::ucx_exchange::Communicator::run, communicator_.get());
  } else {
    ADD_FAILURE() << "Communicator initialization failed";
  }
  future.wait();
}

void UcxExchangeTestEnvironment::destroyExchangeState() {
  if (communicator_ == nullptr) {
    return;
  }
  communicator_->stop();
  if (communicatorThread_ != nullptr && communicatorThread_->joinable()) {
    communicatorThread_->join();
  }
  // run() cancels the close receive before it returns, so that UCX does not
  // write into the buffer after the worker is gone.
  EXPECT_FALSE(communicator_->hasPendingCloseReceive());
  communicatorThread_.reset();
  communicator_.reset();
}

std::shared_ptr<UcxOutputQueueManager>
    UcxExchangeTestEnvironment::queueManager_;
std::shared_ptr<std::thread> UcxExchangeTestEnvironment::communicatorThread_;
std::shared_ptr<Communicator> UcxExchangeTestEnvironment::communicator_;
std::atomic<uint32_t> UcxExchangeTestEnvironment::testCounter_{0};
uint16_t UcxExchangeTestEnvironment::communicatorPort_{0};

namespace {

[[maybe_unused]] const auto* kUcxExchangeTestEnvironment =
    testing::AddGlobalTestEnvironment(new UcxExchangeTestEnvironment);

} // namespace

void UcxExchangeTestBase::SetUp() {
  VLOG(0) << "creating pool";
  pool_ = facebook::velox::memory::memoryManager()->addLeafPool(
      "UcxTestMemoryPool");
}

exec::Split UcxExchangeTestBase::remoteSplit(
    std::string_view taskId,
    int partitionId) {
  std::string remoteUrl = fmt::format(
      "http://127.0.0.1:{}/v1/task/{}/results/{}",
      UcxExchangeTestEnvironment::splitUrlPort(),
      taskId,
      partitionId);
  return exec::Split(
      std::make_shared<facebook::velox::exec::RemoteConnectorSplit>(remoteUrl));
}

} // namespace facebook::velox::ucx_exchange
