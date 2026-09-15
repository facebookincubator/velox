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

#include "velox/experimental/ucx-exchange/UcxExchangeQueue.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"

#include <cudf/table/table.hpp>
#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>
#include <cuda/stream>

#include <limits>

namespace facebook::velox::ucx_exchange {
namespace {

TEST(UcxExchangeQueueTest, packedBufferUsesReceiveStream) {
  cudaStream_t producerHandle;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&producerHandle, cudaStreamNonBlocking),
      cudaSuccess);
  auto releaseProducer =
      folly::makeGuard([&] { cudaStreamDestroy(producerHandle); });
  cuda::stream_ref producerStream{producerHandle};
  cudaStream_t receiveHandle;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&receiveHandle, cudaStreamNonBlocking),
      cudaSuccess);
  auto releaseReceive =
      folly::makeGuard([&] { cudaStreamDestroy(receiveHandle); });
  cuda::stream_ref receiveStream{receiveHandle};
  cudf::table table;
  auto packedColumns = cudf::pack(table.view(), producerStream);
  producerStream.sync();
  auto tableView = cudf::unpack(packedColumns);
  auto packedTable = std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedColumns)});

  // Dropping a received page must free its buffer on the stream used by the
  // receive credit callback, even when the producer used a different stream.
  PackedTableWithStream received{
      std::move(packedTable), receiveStream, /*numRows=*/5};
  EXPECT_EQ(received.packedTable->data.gpu_data->stream(), receiveStream);
}

TEST(UcxExchangeQueueTest, receiveReservationsAreByteBounded) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1,
                         /*receiveHighWaterBytes=*/100};
  std::lock_guard<std::mutex> lock(queue.mutex());

  // The adaptive default provides two payloads plus one high-water unit of
  // slack per pipeline lane: (100 + 100) * 2 = 400.
  EXPECT_EQ(queue.receivePrefetchByteLimitLocked(), 400);
  EXPECT_TRUE(queue.tryReserveReceiveBytesLocked(250));
  EXPECT_EQ(queue.reservedReceiveBytesLocked(), 250);
  EXPECT_FALSE(queue.tryReserveReceiveBytesLocked(151));
  EXPECT_TRUE(queue.tryReserveReceiveBytesLocked(150));
  EXPECT_EQ(queue.queuedReceiveBytesLocked(), 400);
  EXPECT_FALSE(queue.receiveCanPrefetchLocked());

  queue.releaseReceiveBytesLocked(400);
  EXPECT_EQ(queue.retainedReceiveBytesLocked(), 0);
  EXPECT_TRUE(queue.receiveCanPrefetchLocked());
}

TEST(UcxExchangeQueueTest, oversizedPayloadMakesProgressWhenEmpty) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1,
                         /*receiveHighWaterBytes=*/100};
  std::lock_guard<std::mutex> lock(queue.mutex());

  EXPECT_TRUE(queue.tryReserveReceiveBytesLocked(1'000));
  EXPECT_FALSE(queue.tryReserveReceiveBytesLocked(1));
  queue.releaseReceiveBytesLocked(1'000);
}

TEST(UcxExchangeQueueTest, allocationPressureShrinksAndRecoversWindow) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1,
                         /*receiveHighWaterBytes=*/100};
  std::lock_guard<std::mutex> lock(queue.mutex());

  ASSERT_TRUE(queue.tryReserveReceiveBytesLocked(200));
  ASSERT_TRUE(queue.recordReceiveAllocationPressureLocked(200));
  EXPECT_EQ(queue.receivePrefetchByteLimitLocked(), 350);
  EXPECT_FALSE(queue.tryReserveReceiveBytesLocked(151));
  EXPECT_TRUE(queue.tryReserveReceiveBytesLocked(150));

  queue.releaseReceiveBytesLocked(350);
  EXPECT_EQ(queue.receivePrefetchByteLimitLocked(), 400);
}

TEST(UcxExchangeQueueTest, reservationTransferDoesNotGrowWindow) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1,
                         /*receiveHighWaterBytes=*/100};
  std::lock_guard<std::mutex> lock(queue.mutex());

  ASSERT_TRUE(queue.tryReserveReceiveBytesLocked(100));
  ASSERT_TRUE(queue.recordReceiveAllocationPressureLocked(300));
  const auto reducedLimit = queue.receivePrefetchByteLimitLocked();
  ASSERT_EQ(reducedLimit, 350);

  // Moving ownership from a posted receive to the queue does not free memory,
  // so consuming the reservation must not be treated as window recovery.
  queue.consumeReceiveReservationLocked(100);
  EXPECT_EQ(queue.receivePrefetchByteLimitLocked(), reducedLimit);
}

TEST(UcxExchangeQueueTest, activeAccountingRejectsDoubleRelease) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1,
                         /*receiveHighWaterBytes=*/100};
  std::lock_guard<std::mutex> lock(queue.mutex());

  ASSERT_TRUE(queue.tryReserveReceiveBytesLocked(25));
  queue.releaseReceiveBytesLocked(25);
  EXPECT_ANY_THROW(queue.releaseReceiveBytesLocked(1));
  EXPECT_ANY_THROW(queue.releaseInFlightReceiveBytesLocked(1));
}

TEST(UcxExchangeQueueTest, activeReservationAccountingRejectsOverflow) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  std::lock_guard<std::mutex> lock(queue.mutex());

  ASSERT_TRUE(
      queue.tryReserveReceiveBytesLocked(std::numeric_limits<uint64_t>::max()));
  EXPECT_ANY_THROW(queue.tryReserveReceiveBytesLocked(1));
}

TEST(UcxExchangeQueueTest, terminalAccountingAllowsLateCallbackRelease) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1,
                         /*receiveHighWaterBytes=*/100};
  {
    std::lock_guard<std::mutex> lock(queue.mutex());
    ASSERT_TRUE(queue.tryReserveReceiveBytesLocked(25));
  }

  queue.close();

  std::lock_guard<std::mutex> lock(queue.mutex());
  EXPECT_EQ(queue.retainedReceiveBytesLocked(), 0);
  EXPECT_NO_THROW(queue.releaseReceiveBytesLocked(25));
  EXPECT_NO_THROW(queue.releaseInFlightReceiveBytesLocked(25));
  EXPECT_FALSE(queue.tryReserveReceiveBytesLocked(1));
}

TEST(UcxExchangeQueueTest, terminalClientErrorIsNotMaskedByClose) {
  auto client = std::make_shared<UcxExchangeClient>(
      "test-task", /*destination=*/0, /*numberOfConsumers=*/1, 100);
  client->failReceive("downstream CUDA stream synchronization failed");

  bool atEnd = false;
  ContinueFuture future;
  EXPECT_ANY_THROW(client->next(/*consumerId=*/0, &atEnd, &future));
}

} // namespace
} // namespace facebook::velox::ucx_exchange
