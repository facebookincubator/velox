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

#include <cudf/contiguous_split.hpp>
#include <fmt/format.h>
#include <folly/ScopeGuard.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/Ports.h"
#include "velox/exec/tests/utils/TaskMonitor.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/PartitionKey.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/tests/FakeUcxPeer.h"
#include "velox/experimental/ucx-exchange/tests/UcxExchangeTestBase.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"

using namespace std::chrono_literals;

namespace facebook::velox::ucx_exchange {
namespace {

// UcxExchangeSource connects to the split URL's port + 3.
constexpr uint16_t kSplitUrlPortOffset{3};

// Returns the tag 'taskId's destination 0 gets its handshake response on.
uint64_t responseTag(std::string_view taskId) {
  return getHandshakeResponseTag(
      fnv1a_32(PartitionKey{std::string(taskId), 0}.toString()));
}

// Returns the first 'size' bytes of a handshake for destination 0 of 'taskId'
// that says it runs 'protocolVersion'.
std::vector<uint8_t>
makeHandshake(std::string_view taskId, size_t size, uint32_t protocolVersion) {
  HandshakeMsg handshake{};
  std::memcpy(
      handshake.taskId,
      taskId.data(),
      std::min(taskId.size(), sizeof(handshake.taskId) - 1));
  handshake.destination = 0;
  handshake.protocolVersion = protocolVersion;
  const auto* bytes = reinterpret_cast<const uint8_t*>(&handshake);
  return std::vector<uint8_t>(bytes, bytes + size);
}

class UcxProtocolVersionTest : public UcxExchangeTestBase {
 protected:
  // Returns the port the Communicator listens on.
  static uint16_t listenerPort() {
    return Communicator::getInstance()->getListenerPort();
  }

  // Returns a client that reads destination 0 of 'taskId' from the producer
  // listening on 'ucxPort'.
  static std::shared_ptr<UcxExchangeClient> makeClient(
      uint16_t ucxPort,
      std::string_view taskId) {
    auto client = std::make_shared<UcxExchangeClient>("consumer", 0, 1);
    client->addRemoteTaskId(
        fmt::format(
            "http://127.0.0.1:{}/v1/task/{}/results/0",
            ucxPort - kSplitUrlPortOffset,
            taskId));
    client->noMoreRemoteTasks();
    return client;
  }

  // Reads 'client' to its end and returns the number of rows. Throws the
  // client's error, or an error of its own if the end does not come in time.
  static int64_t drain(UcxExchangeClient& client) {
    int64_t numRows{0};
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    for (;;) {
      bool atEnd{false};
      ContinueFuture future = ContinueFuture::makeEmpty();
      auto data = client.next(0, &atEnd, &future);
      if (data != nullptr) {
        numRows += data->numRows;
        continue;
      }
      if (atEnd) {
        return numRows;
      }
      VELOX_CHECK(
          std::move(future).wait(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  deadline - std::chrono::steady_clock::now())),
          "The UCX exchange client did not end in time");
    }
  }

  // Sends one chunk from a new producer task to a client of this worker and
  // checks that it arrives: the worker still serves queries after meeting a
  // bad peer.
  void expectOneChunkExchange() {
    constexpr vector_size_t kNumRows{100};
    const auto taskId = getUniqueTaskPrefix() + "afterBadPeer";
    auto task = makeProducerTask(taskId, pool_.get());
    queueManager()->initializeTask(
        task, core::PartitionedOutputNode::Kind::kPartitioned, 1, 1);
    SCOPE_EXIT {
      queueManager()->removeTask(taskId);
    };
    cuda::stream_ref stream{cudaStream_t{cudaStreamDefault}};
    auto table = makeTable(kNumRows, UcxTestData::kTestRowType, stream);
    auto data = std::make_unique<cudf::packed_columns>(
        cudf::pack(table->view(), stream));
    stream.sync();
    queueManager()->enqueue(taskId, 0, std::move(data), kNumRows);
    queueManager()->noMoreData(taskId);

    auto client = makeClient(listenerPort(), taskId);
    EXPECT_EQ(drain(*client), kNumRows);
    client->close();
  }
};

TEST_F(UcxProtocolVersionTest, olderProducerFailsConsumer) {
  const std::string taskId{"producer.0.0.0.0"};
  const auto port = exec::test::Ports::findFreePort();
  ASSERT_GT(port, kSplitUrlPortOffset);
  auto peer = FakeUcxPeer::listen(port);
  auto client = makeClient(port, taskId);

  const auto handshake = peer->waitForHandshake(3s);
  ASSERT_EQ(handshake.size(), sizeof(HandshakeMsg));
  const std::string handshakeTaskId(
      reinterpret_cast<const char*>(handshake.data()),
      strnlen(
          reinterpret_cast<const char*>(handshake.data()),
          sizeof(HandshakeMsg::taskId)));
  EXPECT_EQ(handshakeTaskId, taskId);

  // A producer before version 1 responds with 8 bytes, which hold no version.
  const std::array<uint8_t, 8> legacyResponse{};
  ASSERT_EQ(
      peer->sendTag(
          responseTag(taskId),
          legacyResponse.data(),
          legacyResponse.size(),
          3s),
      UCS_OK);

  VELOX_ASSERT_THROW(
      drain(*client),
      fmt::format(
          "UCX exchange peers must run the same protocol version: this worker "
          "runs 1, producer worker 127.0.0.1:{} runs 0",
          port));
  // The source sends nothing more, not even a close message.
  EXPECT_EQ(
      peer->receiveTag(
              getCloseTag(fnv1a_32(PartitionKey{taskId, 0}.toString())),
              sizeof(CloseMsg),
              1s)
          .first,
      UCS_ERR_TIMED_OUT);
  client->close();
}

TEST_F(UcxProtocolVersionTest, olderConsumerGetsTruncatedResponse) {
  const auto taskId = getUniqueTaskPrefix() + "olderConsumer";
  auto task = makeProducerTask(taskId, pool_.get());
  queueManager()->initializeTask(
      task, core::PartitionedOutputNode::Kind::kPartitioned, 1, 1);
  SCOPE_EXIT {
    queueManager()->removeTask(taskId);
  };
  {
    auto peer = FakeUcxPeer::connect("127.0.0.1", listenerPort());
    const auto handshake = makeHandshake(taskId, kLegacyHandshakeSize, 0);
    peer->sendAm(handshake.data(), handshake.size());

    // A consumer before version 1 receives 8 bytes. UCX reports the longer
    // response as a truncation, which fails that consumer.
    EXPECT_EQ(
        peer->receiveTag(responseTag(taskId), 8, 3s).first,
        UCS_ERR_MESSAGE_TRUNCATED);
  }

  ASSERT_TRUE(exec::test::TaskMonitor::waitForFailure(*task, 3'000'000));
  EXPECT_THAT(
      task->errorMessage(),
      testing::HasSubstr("this worker runs 1, consumer worker "));
  EXPECT_THAT(task->errorMessage(), testing::HasSubstr(" runs 0"));

  expectOneChunkExchange();
}

TEST_F(UcxProtocolVersionTest, otherVersionIsRejected) {
  const auto taskId = getUniqueTaskPrefix() + "otherVersion";
  auto task = makeProducerTask(taskId, pool_.get());
  queueManager()->initializeTask(
      task, core::PartitionedOutputNode::Kind::kPartitioned, 1, 1);
  SCOPE_EXIT {
    queueManager()->removeTask(taskId);
  };
  {
    auto peer = FakeUcxPeer::connect("127.0.0.1", listenerPort());
    const auto handshake = makeHandshake(taskId, sizeof(HandshakeMsg), 2);
    peer->sendAm(handshake.data(), handshake.size());

    const auto [status, bytes] =
        peer->receiveTag(responseTag(taskId), sizeof(HandshakeResponse), 3s);
    ASSERT_EQ(status, UCS_OK);
    std::array<uint8_t, sizeof(HandshakeResponse)> responseBytes{};
    std::memcpy(responseBytes.data(), bytes.data(), responseBytes.size());
    const auto response = decodeHandshakeResponse(responseBytes);
    EXPECT_EQ(response.status, HandshakeStatus::kVersionMismatch);
    EXPECT_EQ(response.protocolVersion, 1u);
  }

  ASSERT_TRUE(exec::test::TaskMonitor::waitForFailure(*task, 3'000'000));
  EXPECT_THAT(task->errorMessage(), testing::HasSubstr(" runs 2"));
}

TEST_F(UcxProtocolVersionTest, shortHandshakeIsDropped) {
  const auto taskId = getUniqueTaskPrefix() + "shortHandshake";
  {
    auto peer = FakeUcxPeer::connect("127.0.0.1", listenerPort());
    const auto handshake = makeHandshake(taskId, 100, kProtocolVersion);
    peer->sendAm(handshake.data(), handshake.size());

    // The handshake is too short to name its channel, so nothing answers it.
    EXPECT_EQ(
        peer->receiveTag(responseTag(taskId), sizeof(HandshakeResponse), 1s)
            .first,
        UCS_ERR_TIMED_OUT);
  }

  expectOneChunkExchange();
}

TEST_F(UcxProtocolVersionTest, undecodableMessageFailsConsumer) {
  const auto taskId = getUniqueTaskPrefix() + "undecodable";
  const auto port = exec::test::Ports::findFreePort();
  ASSERT_GT(port, kSplitUrlPortOffset);
  auto peer = FakeUcxPeer::listen(port);
  auto client = makeClient(port, taskId);

  ASSERT_EQ(peer->waitForHandshake(3s).size(), sizeof(HandshakeMsg));
  const HandshakeResponse response{};
  ASSERT_EQ(
      peer->sendTag(responseTag(taskId), &response, sizeof(response), 3s),
      UCS_OK);
  std::array<uint8_t, 16> garbage;
  garbage.fill(0xFF);
  ASSERT_EQ(
      peer->sendTag(
          getMetadataTag(fnv1a_32(PartitionKey{taskId, 0}.toString()), 0),
          garbage.data(),
          garbage.size(),
          3s),
      UCS_OK);

  // The queue keeps its error, so each drain reports it again.
  VELOX_ASSERT_THROW(
      drain(*client),
      fmt::format(
          "Cannot decode message from producer, host 127.0.0.1:{}, task {}/0: ",
          port,
          taskId));
  VELOX_ASSERT_THROW(drain(*client), "Unexpected magic number: 4294967295");
  client->close();
}

} // namespace
} // namespace facebook::velox::ucx_exchange
