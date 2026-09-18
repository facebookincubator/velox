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

#include <thread>
#include <vector>

#include <fmt/format.h>
#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/MaterializedOutputBuffer.h"
#include "velox/exec/MaterializedOutputBufferManager.h"

namespace facebook::velox::exec {
namespace {

class RecordingExchangeSink : public ExchangeSink {
 public:
  explicit RecordingExchangeSink(int32_t numPartitions = 0)
      : numPartitions_(numPartitions) {}

  void append(int32_t /*partition*/, std::string_view data) override {
    stringAppendBytes_ += data.size();
  }

  void append(int32_t /*partition*/, std::unique_ptr<folly::IOBuf> data)
      override {
    ++chainAppendCount_;
    chainElementCount_ += data->countChainElements();
    chainAppendBytes_ += data->computeChainDataLength();
  }

  CommittedExchangeOutput finish() override {
    finished_ = true;
    CommittedExchangeOutput output;
    for (int32_t partition = 0; partition < numPartitions_; ++partition) {
      output.locations.emplace(
          partition, fmt::format("partition:{}", partition));
    }
    return output;
  }

  void abort() override {
    aborted_ = true;
  }

  folly::F14FastMap<std::string, int64_t> stats() const override {
    return {{"totalBytesWritten", stringAppendBytes_ + chainAppendBytes_}};
  }

  const int32_t numPartitions_;
  std::atomic_int64_t stringAppendBytes_{0};
  std::atomic_int64_t chainAppendCount_{0};
  std::atomic_int64_t chainElementCount_{0};
  std::atomic_int64_t chainAppendBytes_{0};
  std::atomic_bool finished_{false};
  std::atomic_bool aborted_{false};
};

class BlockingExchangeSink : public RecordingExchangeSink {
 public:
  void append(int32_t partition, std::unique_ptr<folly::IOBuf> data) override {
    if (!appendStartedPosted_.exchange(true)) {
      appendStarted_.post();
      releaseAppend_.wait();
    }
    RecordingExchangeSink::append(partition, std::move(data));
  }

  std::atomic_bool appendStartedPosted_{false};
  folly::Baton<> appendStarted_;
  folly::Baton<> releaseAppend_;
};

class MaterializedOutputBufferTest : public ::testing::Test {
 protected:
  std::shared_ptr<RecordingExchangeSink> createSink(int32_t numPartitions) {
    return std::make_shared<RecordingExchangeSink>(numPartitions);
  }

  std::unique_ptr<folly::IOBuf> makeData(std::string_view content) {
    return folly::IOBuf::copyBuffer(content.data(), content.size());
  }
};

TEST_F(MaterializedOutputBufferTest, singleDriverSinglePartition) {
  auto sink = createSink(1);
  MaterializedOutputBuffer buffer(1, sink, 1 << 20);
  buffer.setNumDrivers(1);

  buffer.enqueue(0, makeData("hello"));
  buffer.enqueue(0, makeData("world"));
  ContinueFuture future;
  EXPECT_EQ(buffer.isBlocked(&future), BlockingReason::kNotBlocked);

  EXPECT_TRUE(buffer.noMoreDrivers());
  EXPECT_TRUE(sink->finished_);
  EXPECT_EQ(buffer.outputLocations().size(), 1);
}

TEST_F(MaterializedOutputBufferTest, multiplePartitions) {
  auto sink = createSink(3);
  MaterializedOutputBuffer buffer(3, sink, 1 << 20);
  buffer.setNumDrivers(1);

  buffer.enqueue(0, makeData("partition_0"));
  buffer.enqueue(1, makeData("partition_1"));
  buffer.enqueue(2, makeData("partition_2"));

  EXPECT_TRUE(buffer.noMoreDrivers());
  EXPECT_GT(buffer.stats().at("totalBytesWritten"), 0);
  EXPECT_EQ(buffer.outputLocations().size(), 3);
}

TEST_F(MaterializedOutputBufferTest, multipleDrivers) {
  auto sink = createSink(2);
  auto buffer = std::make_shared<MaterializedOutputBuffer>(2, sink, 1 << 20);
  buffer->setNumDrivers(4);

  std::vector<std::thread> threads;
  for (int32_t driver = 0; driver < 4; ++driver) {
    threads.emplace_back([buffer, driver]() {
      for (int32_t row = 0; row < 10; ++row) {
        const auto data = fmt::format("driver_{}_row_{}", driver, row);
        buffer->enqueue(
            row % 2, folly::IOBuf::copyBuffer(data.data(), data.size()));
      }
      buffer->noMoreDrivers();
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(sink->finished_);
  EXPECT_EQ(buffer->outputLocations().size(), 2);
}

TEST_F(MaterializedOutputBufferTest, iobufChainAppend) {
  auto sink = createSink(1);
  MaterializedOutputBuffer buffer(1, sink, 100, 10);

  buffer.enqueue(0, makeData("hello"));
  buffer.enqueue(0, makeData("world!"));

  EXPECT_EQ(sink->chainAppendCount_, 1);
  EXPECT_EQ(sink->chainElementCount_, 2);
  EXPECT_EQ(sink->chainAppendBytes_, 11);
  EXPECT_EQ(sink->stringAppendBytes_, 0);
}

TEST_F(MaterializedOutputBufferTest, backpressureWhileFlusherIsBusy) {
  auto sink = std::make_shared<BlockingExchangeSink>();
  MaterializedOutputBuffer buffer(1, sink, 100, 100);

  buffer.enqueue(0, makeData(std::string(80, 'a')));
  std::thread flusher(
      [&]() { buffer.enqueue(0, makeData(std::string(30, 'b'))); });
  sink->appendStarted_.wait();

  buffer.enqueue(0, makeData(std::string(80, 'c')));
  ContinueFuture future;
  EXPECT_EQ(buffer.isBlocked(&future), BlockingReason::kWaitForConsumer);
  ASSERT_TRUE(future.valid());

  sink->releaseAppend_.post();
  flusher.join();
  EXPECT_TRUE(future.isReady());
  EXPECT_EQ(buffer.bufferedBytes(), 0);
}

TEST_F(MaterializedOutputBufferTest, fullAtHighWatermark) {
  auto sink = createSink(1);
  MaterializedOutputBuffer buffer(1, sink, 100, 100);

  EXPECT_FALSE(buffer.isBufferFull());
  buffer.enqueue(0, makeData(std::string(89, 'a')));
  EXPECT_FALSE(buffer.isBufferFull());
  buffer.enqueue(0, makeData("b"));
  EXPECT_TRUE(buffer.isBufferFull());
}

TEST_F(MaterializedOutputBufferTest, drainAll) {
  auto sink = createSink(3);
  MaterializedOutputBuffer buffer(3, sink, 1 << 20);

  buffer.enqueue(0, makeData("data_0"));
  buffer.enqueue(1, makeData("data_1"));
  buffer.enqueue(2, makeData("data_2"));

  EXPECT_GT(buffer.drainAll(), 0);
  EXPECT_EQ(buffer.bufferedBytes(), 0);
  EXPECT_EQ(sink->chainAppendCount_, 3);
}

TEST_F(MaterializedOutputBufferTest, abort) {
  auto sink = createSink(1);
  MaterializedOutputBuffer buffer(1, sink, 1 << 20);

  buffer.enqueue(0, makeData("data"));
  buffer.abort();

  EXPECT_TRUE(sink->aborted_);
  EXPECT_EQ(buffer.state(), MaterializedOutputBuffer::State::kAborted);
  EXPECT_EQ(buffer.bufferedBytes(), 0);
}

TEST_F(MaterializedOutputBufferTest, noMoreDataIsIdempotent) {
  auto sink = createSink(1);
  MaterializedOutputBuffer buffer(1, sink, 1 << 20);

  buffer.enqueue(0, makeData("data"));
  buffer.noMoreData();
  buffer.noMoreData();

  EXPECT_TRUE(sink->finished_);
  EXPECT_EQ(sink->chainAppendCount_, 1);
}

TEST_F(MaterializedOutputBufferTest, rejectsEnqueueAfterClose) {
  auto sink = createSink(1);
  MaterializedOutputBuffer buffer(1, sink, 1 << 20);
  buffer.noMoreData();

  VELOX_ASSERT_THROW(
      buffer.enqueue(0, makeData("late")), "enqueue called after noMoreData()");
}

TEST_F(MaterializedOutputBufferTest, configurableOutputBatchSize) {
  ExchangeSink::Factory sinkFactory =
      [](const std::string&,
         const std::string&,
         memory::MemoryPool*) -> std::shared_ptr<ExchangeSink> {
    return std::make_shared<RecordingExchangeSink>();
  };
  MaterializedOutputBatchConfig config{
      .minOutputBatchBytes = 100,
      .maxOutputBatchBytes = 300,
      .estimatedRowBytes = 40,
  };
  MaterializedOutputBufferManager manager(
      std::move(sinkFactory), 1'024, config);

  EXPECT_EQ(manager.outputBatchSizeBytes(1), 100);
  EXPECT_EQ(manager.outputBatchSizeBytes(5), 200);
  EXPECT_EQ(manager.outputBatchSizeBytes(10), 300);
}

} // namespace
} // namespace facebook::velox::exec
