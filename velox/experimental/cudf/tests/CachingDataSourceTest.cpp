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

#include "velox/experimental/cudf/connectors/hive/CachingDataSource.h"

#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/memory/MallocAllocator.h"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {
using namespace std::chrono_literals;

struct ReadState {
  using Requests = std::vector<std::pair<size_t, size_t>>;

  void recordRead(size_t offset, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    requests.emplace_back(offset, bytes);
    ++reads;
  }

  Requests recordedReads() {
    std::lock_guard<std::mutex> lock(mutex);
    return requests;
  }

  std::string data = std::string(8192, 'x');
  std::atomic<size_t> reads{0};
  std::atomic<bool> shortRead{false};
  std::atomic<bool> failRead{false};
  std::atomic<bool> destroyed{false};
  std::shared_future<void> gate;
  std::mutex mutex;
  Requests requests;
};

class MemorySource : public cudf::io::datasource {
 public:
  explicit MemorySource(std::shared_ptr<ReadState> state)
      : state_(std::move(state)) {}
  ~MemorySource() override {
    state_->destroyed = true;
  }
  size_t size() const override {
    return state_->data.size();
  }
  bool supports_device_read() const override {
    return true;
  }
  bool is_device_read_preferred(size_t) const override {
    return true;
  }
  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t bytes)
      override {
    std::vector<uint8_t> result(
        offset < size() ? std::min(bytes, size() - offset) : 0);
    result.resize(host_read(offset, result.size(), result.data()));
    return datasource::buffer::create(std::move(result));
  }
  size_t host_read(size_t offset, size_t bytes, uint8_t* dst) override {
    state_->recordRead(offset, bytes);
    if (state_->gate.valid()) {
      state_->gate.wait();
    }
    VELOX_CHECK(!state_->failRead.exchange(false), "Injected remote failure");
    bytes = offset < size() ? std::min(bytes, size() - offset) : 0;
    if (state_->shortRead.exchange(false) && bytes > 0) {
      --bytes;
    }
    if (bytes > 0) {
      std::memcpy(dst, state_->data.data() + offset, bytes);
    }
    return bytes;
  }

 private:
  std::shared_ptr<ReadState> state_;
};

// External completion, like RemoteHandle::pread: no thread blocks on each
// pending read, and destroying the std::future does not stop writes to dst.
struct AsyncReadControl {
  std::mutex mutex;
  std::vector<std::function<void()>> pending;
  bool open{false};
  std::atomic<size_t> submitted{0};
  std::atomic<size_t> synchronous{0};
  std::atomic<bool> failSubmission{false};
  std::atomic<bool> invalidFuture{false};

  void add(std::function<void()> fill) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!open) {
        pending.push_back(std::move(fill));
        ++submitted;
        return;
      }
    }
    ++submitted;
    fill();
  }

  void completeNewest() {
    std::function<void()> fill;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (pending.empty()) {
        return;
      }
      fill = std::move(pending.back());
      pending.pop_back();
    }
    fill();
  }

  void releaseAll() {
    std::vector<std::function<void()>> fills;
    {
      std::lock_guard<std::mutex> lock(mutex);
      open = true;
      fills.swap(pending);
    }
    for (auto& fill : fills) {
      fill();
    }
  }
};

class AsyncMemorySource : public MemorySource {
 public:
  AsyncMemorySource(
      std::shared_ptr<ReadState> state,
      std::shared_ptr<AsyncReadControl> control)
      : MemorySource(state), state_(state), control_(std::move(control)) {}

  size_t host_read(size_t offset, size_t bytes, uint8_t* dst) override {
    ++control_->synchronous;
    return MemorySource::host_read(offset, bytes, dst);
  }

  std::future<size_t> host_read_async(size_t offset, size_t bytes, uint8_t* dst)
      override {
    VELOX_CHECK(
        !control_->failSubmission.exchange(false),
        "Injected submission failure");
    if (control_->invalidFuture.exchange(false)) {
      return {};
    }
    auto promise = std::make_shared<std::promise<size_t>>();
    auto future = promise->get_future();
    state_->recordRead(offset, bytes);
    control_->add([state = state_, offset, bytes, dst, promise]() mutable {
      try {
        VELOX_CHECK(
            !state->failRead.exchange(false), "Injected remote failure");
        if (state->shortRead.exchange(false) && bytes != 0) {
          --bytes;
        }
        std::memcpy(dst, state->data.data() + offset, bytes);
        promise->set_value(bytes);
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    });
    return future;
  }

 private:
  std::shared_ptr<ReadState> state_;
  std::shared_ptr<AsyncReadControl> control_;
};

bool waitUntil(const std::function<bool()>& condition) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!condition() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return condition();
}

struct StreamGate {
  std::promise<void> entered;
  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();
};

void CUDART_CB waitForStreamGate(void* argument) {
  auto* gate = static_cast<StreamGate*>(argument);
  gate->entered.set_value();
  gate->released.wait();
}

class CachingDataSourceTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // This test executable owns its remote executor.
    setenv("KVIKIO_NTHREADS", "2", 1);
    setenv("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS", "8", 1);
  }

  std::unique_ptr<cudf::io::datasource> source(bool cacheable = true) {
    return maybeCacheKvikioDataSource(
        std::make_unique<MemorySource>(read_),
        path_,
        cache_.get(),
        cacheable,
        stats_);
  }
  std::string fromDevice(const uint8_t* device, size_t bytes) {
    std::string result(bytes, '\0');
    CUDF_CUDA_TRY(
        cudaMemcpy(result.data(), device, bytes, cudaMemcpyDeviceToHost));
    return result;
  }

  std::unique_ptr<cudf::io::datasource> asyncSource(
      const std::shared_ptr<AsyncReadControl>& control) {
    return maybeCacheKvikioDataSource(
        std::make_unique<AsyncMemorySource>(read_, control),
        path_,
        cache_.get(),
        true,
        stats_);
  }

  void fillPattern(size_t bytes) {
    read_->data.resize(bytes);
    for (size_t i = 0; i < bytes; ++i) {
      read_->data[i] = static_cast<char>(i % 251);
    }
  }

  std::shared_ptr<memory::MallocAllocator> allocator_ =
      std::make_shared<memory::MallocAllocator>(
          memory::MemoryAllocator::Options{
              .capacity = 16 << 20,
              .reservationByteLimit = 0});
  std::shared_ptr<cache::AsyncDataCache> cache_ =
      cache::AsyncDataCache::create(allocator_.get());
  std::shared_ptr<ReadState> read_ = std::make_shared<ReadState>();
  std::shared_ptr<IoStats> stats_ = std::make_shared<IoStats>();
  const std::string path_ = "s3://test/caching-datasource";
};

TEST_F(CachingDataSourceTest, asyncFillsExceedExecutorThreadsAndStayBounded) {
  constexpr size_t kRequests = 24;
  constexpr size_t kBytes = 4096;
  fillPattern(kRequests * kBytes);
  auto control = std::make_shared<AsyncReadControl>();
  auto input = asyncSource(control);
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      read_->data.size(),
      stream.view(),
      cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  std::vector<std::future<size_t>> reads;
  // Release before the future destructors, including on an ASSERT failure.
  auto release = folly::makeGuard([&] { control->releaseAll(); });
  for (size_t i = 0; i < kRequests; ++i) {
    reads.push_back(input->device_read_async(
        i * kBytes, kBytes, dst + i * kBytes, stream.view()));
  }
  ASSERT_TRUE(waitUntil([&] { return control->submitted.load() >= 8; }));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(control->submitted, 8)
      << "The host-read window must exceed two executor threads but stay bounded";
  EXPECT_EQ(control->synchronous, 0);
  control->releaseAll();
  for (auto& read : reads) {
    EXPECT_EQ(read.get(), kBytes);
  }
  EXPECT_EQ(fromDevice(dst, read_->data.size()), read_->data);
  EXPECT_EQ(read_->reads, kRequests);
  const auto metrics = stats_->stats();
  EXPECT_EQ(metrics.at("cudfKvikioCacheAsyncFillSubmitted").sum, kRequests);
  EXPECT_EQ(metrics.at("cudfKvikioCacheAsyncFillInFlightSamples").max, 8);
  EXPECT_EQ(metrics.count("cudfKvikioCacheDeferredHostReads"), 0);
  EXPECT_EQ(metrics.count("cudfKvikioCacheAsyncReadFailures"), 0);
}

class CachingDataSourceFragmentTest : public CachingDataSourceTest,
                                      public testing::WithParamInterface<bool> {
};

TEST_P(CachingDataSourceFragmentTest, deviceReadsReuseCachedFragments) {
  const bool nonContiguous = GetParam();
  // More fragments than the per-key retry limit, with nonuniform data to
  // catch incorrect source offsets or overwriting an earlier destination.
  constexpr size_t kFragments = 40;
  constexpr size_t kFragmentBytes = 4096;
  constexpr size_t kTailBytes = 4096 + 731;
  constexpr size_t kPrefixBytes = kFragments * kFragmentBytes;
  constexpr size_t kBytes = kPrefixBytes + kTailBytes;
  fillPattern(kBytes);
  StringIdLease file(fileIds(), path_);
  for (size_t offset = 0; offset < kPrefixBytes; offset += kFragmentBytes) {
    auto pin = cache_->findOrCreate(
        {file.id(), offset}, kFragmentBytes, !nonContiguous);
    auto* entry = pin.checkedEntry();
    ASSERT_TRUE(entry->isExclusive());
    ASSERT_EQ(entry->hasContiguousData(), !nonContiguous);
    size_t copied = 0;
    for (const auto& range : entry->dataRanges(kFragmentBytes)) {
      std::memcpy(
          range.data(), read_->data.data() + offset + copied, range.size());
      copied += range.size();
    }
    entry->setExclusiveToShared();
  }
  auto control = std::make_shared<AsyncReadControl>();
  control->releaseAll();
  auto input = asyncSource(control);
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      kBytes, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  EXPECT_EQ(
      input->device_read_async(0, kBytes, dst, stream.view()).get(), kBytes);
  EXPECT_EQ(fromDevice(dst, kBytes), read_->data);
  EXPECT_EQ(
      read_->recordedReads(),
      (ReadState::Requests{{kPrefixBytes, kTailBytes}}));
  EXPECT_EQ(control->synchronous, 0);
  auto metrics = stats_->stats();
  EXPECT_EQ(metrics.at("cudfKvikioCacheHitBytes").sum, kPrefixBytes);
  EXPECT_EQ(metrics.at("cudfKvikioCacheMissBytes").sum, kTailBytes);
  EXPECT_EQ(metrics.at("cudfKvikioCacheReusedPrefixes").sum, kFragments);
  EXPECT_EQ(metrics.at("cudfKvikioCacheReusedPrefixBytes").sum, kPrefixBytes);
  EXPECT_EQ(metrics.count("cudfKvikioCacheAsyncReadSynchronousFallbacks"), 0);

  // Synchronous overloads and a smaller final fragment use the same chain.
  EXPECT_EQ(
      input->device_read(0, kBytes - 17, dst, stream.view()), kBytes - 17);
  auto owned = input->device_read(0, kBytes, stream.view());
  EXPECT_EQ(fromDevice(owned->data(), owned->size()), read_->data);
  EXPECT_EQ(fromDevice(dst, kBytes - 17), read_->data.substr(0, kBytes - 17));
  EXPECT_EQ(read_->reads, 1);
}

INSTANTIATE_TEST_SUITE_P(
    CachingDataSourceFragmentTest,
    CachingDataSourceFragmentTest,
    testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "nonContiguous" : "contiguous";
    });

TEST_F(
    CachingDataSourceTest,
    longerAsyncReadWaitsForPrefixThenFillsOnlySuffix) {
  auto control = std::make_shared<AsyncReadControl>();
  auto input = asyncSource(control);
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      5120, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  std::vector<std::future<size_t>> reads;
  auto release = folly::makeGuard([&] { control->releaseAll(); });
  reads.push_back(input->device_read_async(0, 1024, dst, stream.view()));
  ASSERT_TRUE(waitUntil([&] { return control->submitted.load() == 1; }));
  reads.push_back(input->device_read_async(0, 4096, dst + 1024, stream.view()));
  ASSERT_TRUE(
      waitUntil([&] { return cache_->refreshStats().numWaitExclusive > 0; }));
  control->completeNewest();
  ASSERT_TRUE(waitUntil([&] { return control->submitted.load() == 2; }));
  EXPECT_EQ(reads.front().get(), 1024);
  EXPECT_EQ(
      read_->recordedReads(), (ReadState::Requests{{0, 1024}, {1024, 3072}}));
  control->releaseAll();
  EXPECT_EQ(reads.back().get(), 4096);
  EXPECT_EQ(fromDevice(dst, 5120), std::string(5120, 'x'));
  EXPECT_EQ(stats_->stats().at("cudfKvikioCacheHitBytes").sum, 1024);
  EXPECT_EQ(stats_->stats().at("cudfKvikioCacheMissBytes").sum, 4096);
  EXPECT_EQ(control->synchronous, 0);
}

TEST_F(CachingDataSourceTest, asyncSharedFillAndOutOfOrderCompletion) {
  read_->data.assign(16384, 'a');
  auto control = std::make_shared<AsyncReadControl>();
  auto input = asyncSource(control);
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      16384, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  std::vector<std::future<size_t>> reads;
  auto release = folly::makeGuard([&] { control->releaseAll(); });
  reads.push_back(input->device_read_async(0, 4096, dst, stream.view()));
  ASSERT_TRUE(waitUntil([&] { return control->submitted.load() == 1; }));
  // Two followers of the pending fill must not occupy the two executor
  // threads and prevent a later, independent fill from being submitted.
  reads.push_back(input->device_read_async(0, 4096, dst + 4096, stream.view()));
  reads.push_back(input->device_read_async(0, 4096, dst + 8192, stream.view()));
  reads.push_back(
      input->device_read_async(4096, 4096, dst + 12288, stream.view()));
  ASSERT_TRUE(waitUntil([&] { return control->submitted.load() == 2; }));
  control->completeNewest();
  ASSERT_TRUE(waitUntil([&] {
    return stats_->stats().contains("cudfKvikioCacheHostToDeviceBytes");
  })) << "A ready range must complete while the oldest range is still pending";
  EXPECT_EQ(reads.back().get(), 4096);
  EXPECT_EQ(fromDevice(dst + 12288, 4096), std::string(4096, 'a'));
  control->releaseAll();
  for (size_t i = 0; i < reads.size() - 1; ++i) {
    EXPECT_EQ(reads[i].get(), 4096);
  }
  EXPECT_EQ(control->submitted, 2);
  EXPECT_EQ(stats_->stats().at("cudfKvikioCacheHitBytes").sum, 8192);
  EXPECT_EQ(fromDevice(dst, 16384), std::string(16384, 'a'));
}

TEST_F(CachingDataSourceTest, asyncFailedAndShortFillsAreNotPublished) {
  auto control = std::make_shared<AsyncReadControl>();
  control->releaseAll();
  auto input = asyncSource(control);
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      4096, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  read_->failRead = true;
  EXPECT_THROW(
      input->device_read_async(0, 4096, dst, stream.view()).get(),
      VeloxRuntimeError);
  read_->shortRead = true;
  EXPECT_THROW(
      input->device_read_async(0, 4096, dst, stream.view()).get(),
      VeloxRuntimeError);
  control->failSubmission = true;
  EXPECT_THROW(
      input->device_read_async(0, 4096, dst, stream.view()).get(),
      VeloxRuntimeError);
  control->invalidFuture = true;
  EXPECT_THROW(
      input->device_read_async(0, 4096, dst, stream.view()).get(),
      VeloxRuntimeError);
  EXPECT_EQ(input->device_read_async(0, 4096, dst, stream.view()).get(), 4096);
  EXPECT_EQ(input->device_read_async(0, 4096, dst, stream.view()).get(), 4096);
  EXPECT_EQ(control->submitted, 3);
  EXPECT_EQ(stats_->stats().at("cudfKvikioCacheAsyncReadFailures").sum, 4);
  EXPECT_EQ(fromDevice(dst, 4096), read_->data.substr(0, 4096));
}

TEST_F(
    CachingDataSourceTest,
    discardedAsyncReadDrainsBeforeReleasingCacheStorage) {
  auto control = std::make_shared<AsyncReadControl>();
  auto input = asyncSource(control);
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      4096, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  auto read = input->device_read_async(0, 4096, dst, stream.view());
  std::future<void> discard;
  auto release = folly::makeGuard([&] { control->releaseAll(); });
  ASSERT_TRUE(waitUntil([&] { return control->submitted.load() == 1; }));
  input.reset();
  discard = std::async(
      std::launch::async, [read = std::move(read)]() mutable { read = {}; });
  cache_->clear();
  // numEntries counts published entries, not this still-exclusive fill.
  EXPECT_EQ(cache_->refreshStats().numExclusive, 1);
  EXPECT_FALSE(read_->destroyed);
  EXPECT_EQ(discard.wait_for(20ms), std::future_status::timeout);
  control->releaseAll();
  discard.get();
  EXPECT_EQ(fromDevice(dst, 4096), read_->data.substr(0, 4096));
  cache_->clear();
  EXPECT_EQ(cache_->refreshStats().numEntries, 0);
}

TEST_F(CachingDataSourceTest, asyncCapacityFailureUsesUncachedFallback) {
  read_->data.assign(32 << 20, 'b');
  auto control = std::make_shared<AsyncReadControl>();
  auto input = asyncSource(control);
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      read_->data.size(),
      stream.view(),
      cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  EXPECT_EQ(
      input->device_read_async(0, read_->data.size(), dst, stream.view()).get(),
      read_->data.size());
  EXPECT_EQ(control->submitted, 0);
  EXPECT_EQ(control->synchronous, 1);
  EXPECT_EQ(
      stats_->stats().at("cudfKvikioCacheAsyncReadSynchronousFallbacks").sum,
      1);
  EXPECT_EQ(fromDevice(dst, read_->data.size()), read_->data);
  EXPECT_EQ(cache_->refreshStats().numEntries, 0);
}

TEST_F(CachingDataSourceTest, missesHitsAndSynchronousOverloads) {
  fillPattern(128 << 10);
  auto input = source();
  rmm::cuda_stream stream;
  constexpr size_t kSize = 17 * 4096 + 37;
  rmm::device_buffer destination(
      kSize, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  EXPECT_EQ(
      input->device_read_async(19, kSize, dst, stream.view()).get(), kSize);
  EXPECT_EQ(input->device_read(19, kSize, dst, stream.view()), kSize);
  auto owned = input->device_read(19, kSize, stream.view());
  EXPECT_EQ(fromDevice(dst, kSize), read_->data.substr(19, kSize));
  EXPECT_EQ(
      fromDevice(owned->data(), owned->size()), read_->data.substr(19, kSize));
  EXPECT_EQ(read_->reads, 1);
  const auto metrics = stats_->stats();
  EXPECT_EQ(metrics.at("cudfKvikioCacheMissBytes").sum, kSize);
  EXPECT_EQ(metrics.at("cudfKvikioCacheHitBytes").sum, 2 * kSize);
  EXPECT_EQ(metrics.at("cudfKvikioCacheHostToDeviceBytes").sum, 3 * kSize);
  EXPECT_EQ(input->host_read(19, kSize)->size(), kSize);
  EXPECT_EQ(read_->reads, 1);
}

TEST_F(CachingDataSourceTest, cacheOffAndNonCacheablePreserveDelegateIdentity) {
  for (const bool cacheable : {false, true}) {
    auto delegate = std::make_unique<MemorySource>(read_);
    auto* original = delegate.get();
    auto actual = maybeCacheKvikioDataSource(
        std::move(delegate),
        path_,
        cacheable ? nullptr : cache_.get(),
        cacheable);
    EXPECT_EQ(actual.get(), original);
  }
}

TEST_F(CachingDataSourceTest, hostMissHitAndFileIdentity) {
  auto input = source();
  auto first = input->host_read(17, 512);
  auto second = input->host_read(17, 512);
  EXPECT_EQ(read_->reads, 1);
  EXPECT_EQ(std::memcmp(first->data(), second->data(), 512), 0);
  const auto metrics = stats_->stats();
  EXPECT_EQ(metrics.at("cudfKvikioCacheMissBytes").sum, 512);
  EXPECT_EQ(metrics.at("cudfKvikioCacheHitBytes").sum, 512);
  auto anotherFile = maybeCacheKvikioDataSource(
      std::make_unique<MemorySource>(read_),
      "s3://test/another-file",
      cache_.get(),
      true);
  anotherFile->host_read(17, 512);
  EXPECT_EQ(read_->reads, 2);
}

TEST_F(CachingDataSourceTest, failedAndShortReadsAreNotPublished) {
  auto input = source();
  read_->shortRead = true;
  EXPECT_THROW(input->host_read(31, 100), VeloxRuntimeError);
  read_->failRead = true;
  EXPECT_THROW(input->host_read(31, 100), VeloxRuntimeError);
  EXPECT_EQ(input->host_read(31, 100)->size(), 100);
  EXPECT_EQ(read_->reads, 3);
  input->host_read(31, 100);
  EXPECT_EQ(read_->reads, 3);
}

TEST_F(CachingDataSourceTest, nonContiguousCacheHit) {
  StringIdLease file(fileIds(), path_);
  auto pin = cache_->findOrCreate({file.id(), 0}, read_->data.size(), false);
  auto* entry = pin.checkedEntry();
  ASSERT_FALSE(entry->hasContiguousData());
  for (const auto& range : entry->dataRanges(read_->data.size())) {
    std::memset(range.data(), 'p', range.size());
  }
  entry->setExclusiveToShared();
  auto input = source();
  auto bytes = input->host_read(0, read_->data.size());
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(bytes->data()), bytes->size()),
      std::string(read_->data.size(), 'p'));
  EXPECT_EQ(read_->reads, 0);
}

TEST_F(CachingDataSourceTest, emptyEofAndOverflowingEnd) {
  auto input = source();
  EXPECT_EQ(input->host_read(0, 0)->size(), 0);
  EXPECT_EQ(input->host_read(input->size(), 1)->size(), 0);
  EXPECT_EQ(
      input->host_read(std::numeric_limits<size_t>::max(), 1, nullptr), 0);
  EXPECT_EQ(read_->reads, 0);
  EXPECT_EQ(
      input->host_read(input->size() - 3, std::numeric_limits<size_t>::max())
          ->size(),
      3);
  EXPECT_EQ(read_->reads, 1);
}

TEST_F(CachingDataSourceTest, longerRangeReusesShortPrefix) {
  auto input = source();
  input->host_read(1, 100);
  EXPECT_EQ(input->host_read(1, 200)->size(), 200);
  EXPECT_EQ(read_->reads, 2);
  input->host_read(1, 50);
  EXPECT_EQ(read_->reads, 2);
  EXPECT_EQ(
      read_->recordedReads(), (ReadState::Requests{{1, 100}, {101, 100}}));
  StringIdLease file(fileIds(), path_);
  auto pin = cache_->findOrCreate({file.id(), 1}, 100, true);
  EXPECT_EQ(pin.checkedEntry()->size(), 100);
}

TEST_F(
    CachingDataSourceTest,
    hostReadsReuseFragmentChainAndClampFinalFragment) {
  fillPattern(read_->data.size());
  auto input = source();
  input->host_read(0, 1024);
  input->host_read(1024, 512);
  input->host_read(1536, 1024);
  std::vector<uint8_t> dst(4096);
  EXPECT_EQ(input->host_read(0, dst.size(), dst.data()), dst.size());
  EXPECT_EQ(std::memcmp(dst.data(), read_->data.data(), dst.size()), 0);
  EXPECT_EQ(
      read_->recordedReads(),
      (ReadState::Requests{
          {0, 1024}, {1024, 512}, {1536, 1024}, {2560, 1536}}));
  auto metrics = stats_->stats();
  EXPECT_EQ(metrics.at("cudfKvikioCacheMissBytes").sum, 4096);
  EXPECT_EQ(metrics.at("cudfKvikioCacheHitBytes").sum, 2560);
  EXPECT_EQ(metrics.at("cudfKvikioCacheReusedPrefixes").sum, 3);
  EXPECT_EQ(metrics.at("cudfKvikioCacheReusedPrefixBytes").sum, 2560);
  auto owned = input->host_read_async(0, 1800).get();
  EXPECT_EQ(owned->size(), 1800);
  EXPECT_EQ(std::memcmp(owned->data(), read_->data.data(), 1800), 0);
  EXPECT_EQ(
      input->host_read_async(0, dst.size(), dst.data()).get(), dst.size());
  EXPECT_EQ(std::memcmp(dst.data(), read_->data.data(), dst.size()), 0);
  EXPECT_EQ(read_->reads, 4);
}

TEST_F(CachingDataSourceTest, failedHostSuffixPreservesCachedPrefix) {
  auto input = source();
  input->host_read(0, 1024);
  read_->shortRead = true;
  EXPECT_THROW(input->host_read(0, 4096), VeloxRuntimeError);
  read_->failRead = true;
  EXPECT_THROW(input->host_read(0, 4096), VeloxRuntimeError);
  EXPECT_EQ(input->host_read(0, 4096)->size(), 4096);
  EXPECT_EQ(
      read_->recordedReads(),
      (ReadState::Requests{
          {0, 1024}, {1024, 3072}, {1024, 3072}, {1024, 3072}}));
}

TEST_F(CachingDataSourceTest, uncachedFallbackReadsOnlyMissingSuffix) {
  read_->data.assign(32 << 20, 'b');
  auto control = std::make_shared<AsyncReadControl>();
  auto input = asyncSource(control);
  input->host_read(0, 4096);
  // Hold the prefix so an unsuccessful large cache allocation cannot evict it.
  StringIdLease file(fileIds(), path_);
  auto pin = cache_->findOrCreate({file.id(), 0}, 4096, true);
  auto host = input->host_read(0, read_->data.size());
  EXPECT_EQ(host->size(), read_->data.size());
  EXPECT_EQ(std::memcmp(host->data(), read_->data.data(), host->size()), 0);
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      read_->data.size(),
      stream.view(),
      cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  EXPECT_EQ(
      input->device_read_async(0, read_->data.size(), dst, stream.view()).get(),
      read_->data.size());
  EXPECT_EQ(fromDevice(dst, read_->data.size()), read_->data);
  // Short uncached reads return the prefix plus the actual suffix length.
  read_->shortRead = true;
  EXPECT_EQ(
      input->device_read(0, read_->data.size(), dst, stream.view()),
      read_->data.size() - 1);
  EXPECT_EQ(
      read_->recordedReads(),
      (ReadState::Requests{
          {0, 4096},
          {4096, read_->data.size() - 4096},
          {4096, read_->data.size() - 4096},
          {4096, read_->data.size() - 4096}}));
  EXPECT_EQ(control->submitted, 0);
}

TEST_F(CachingDataSourceTest, cacheCapacityFailureFallsBackWithoutPublishing) {
  read_->data.assign(32 << 20, 'b');
  auto input = source();
  auto result = input->host_read(0, read_->data.size());
  EXPECT_EQ(result->size(), read_->data.size());
  EXPECT_EQ(result->data()[result->size() - 1], 'b');
  EXPECT_EQ(read_->reads, 1);
  EXPECT_EQ(
      stats_->stats().at("cudfKvikioCacheBypassBytes").sum, result->size());
}

TEST_F(CachingDataSourceTest, concurrentReadersShareOneFill) {
  auto input = source();
  std::promise<void> start;
  auto started = start.get_future().share();
  std::vector<std::future<void>> readers;
  for (int i = 0; i < 8; ++i) {
    readers.push_back(std::async(std::launch::async, [&input, started] {
      started.wait();
      auto result = input->host_read(0, 4096);
      EXPECT_EQ(result->size(), 4096);
      EXPECT_EQ(result->data()[4095], 'x');
    }));
  }
  start.set_value();
  for (auto& reader : readers) {
    reader.get();
  }
  EXPECT_EQ(read_->reads, 1);
}

TEST_F(CachingDataSourceTest, hostFutureRetainsDelegate) {
  auto input = source();
  auto future = input->host_read_async(0, 100);
  input.reset();
  EXPECT_FALSE(read_->destroyed);
  EXPECT_EQ(future.get()->size(), 100);
  EXPECT_TRUE(read_->destroyed);
}

TEST_F(CachingDataSourceTest, deviceReadServesSecondReadFromCache) {
  auto input = source();
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      1024, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  EXPECT_EQ(input->device_read_async(0, 1024, dst, stream.view()).get(), 1024);
  EXPECT_EQ(fromDevice(dst, 1024), read_->data.substr(0, 1024));
  EXPECT_EQ(input->device_read_async(0, 1024, dst, stream.view()).get(), 1024);
  EXPECT_EQ(read_->reads, 1);
  EXPECT_EQ(fromDevice(dst, 1024), read_->data.substr(0, 1024));
}

TEST_F(CachingDataSourceTest, owningDeviceReadUsesCache) {
  auto input = source();
  rmm::cuda_stream stream;
  auto first = input->device_read(9, 200, stream.view());
  auto second = input->device_read(9, 200, stream.view());
  EXPECT_EQ(read_->reads, 1);
  EXPECT_EQ(
      fromDevice(first->data(), first->size()), read_->data.substr(9, 200));
  EXPECT_EQ(
      fromDevice(second->data(), second->size()), read_->data.substr(9, 200));
}

TEST_F(CachingDataSourceTest, synchronousReadsUseCacheAndClampRanges) {
  auto input = source();
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      200, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  // Exercise a miss and both synchronous cache-hit overloads.
  EXPECT_EQ(input->device_read(9, 200, dst, stream.view()), 200);
  EXPECT_EQ(input->device_read(9, 200, dst, stream.view()), 200);
  auto owned = input->device_read(9, 200, stream.view());
  EXPECT_EQ(read_->reads, 1);
  EXPECT_EQ(fromDevice(dst, 200), read_->data.substr(9, 200));
  EXPECT_EQ(
      fromDevice(owned->data(), owned->size()), read_->data.substr(9, 200));

  EXPECT_EQ(input->device_read(0, 0, nullptr, stream.view()), 0);
  EXPECT_EQ(input->device_read(input->size(), 1, nullptr, stream.view()), 0);
  EXPECT_EQ(
      input->device_read(
          std::numeric_limits<size_t>::max(), 1, nullptr, stream.view()),
      0);
  EXPECT_EQ(
      input->device_read(
          input->size() - 3,
          std::numeric_limits<size_t>::max(),
          dst,
          stream.view()),
      3);
  EXPECT_EQ(fromDevice(dst, 3), read_->data.substr(input->size() - 3));
  EXPECT_THROW(
      input->device_read(0, 1, nullptr, stream.view()), VeloxRuntimeError);
}

TEST_F(CachingDataSourceTest, synchronousReadsFromSingleThreadCaller) {
  for (const bool owning : {false, true}) {
    SCOPED_TRACE(owning);
    rmm::cuda_stream stream;
    rmm::device_buffer destination(
        100, stream.view(), cudf::get_current_device_resource_ref());
    auto* dst = static_cast<uint8_t*>(destination.data());
    folly::CPUThreadPoolExecutor pool(1);
    auto input = maybeCacheKvikioDataSource(
        std::make_unique<MemorySource>(read_),
        path_,
        cache_.get(),
        true,
        stats_);
    // Exercise each overload on a cache hit from a single-threaded caller.
    ASSERT_EQ(input->host_read(0, 100)->size(), 100);
    std::unique_ptr<cudf::io::datasource::buffer> owned;
    std::promise<size_t> completed;
    auto result = completed.get_future();
    pool.add([&] {
      try {
        if (owning) {
          owned = input->device_read(0, 100, stream.view());
          completed.set_value(owned->size());
        } else {
          completed.set_value(input->device_read(0, 100, dst, stream.view()));
        }
      } catch (...) {
        completed.set_exception(std::current_exception());
      }
    });
    const auto initial = result.wait_for(5s);
    EXPECT_EQ(initial, std::future_status::ready)
        << "Synchronous read did not complete from the caller executor";
    EXPECT_EQ(result.get(), 100);
    EXPECT_EQ(
        fromDevice(owning ? owned->data() : dst, 100), std::string(100, 'x'));
    pool.join();
  }
}

TEST_F(CachingDataSourceTest, synchronousReadWaitsForH2D) {
  auto input = source();
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      100, stream.view(), cudf::get_current_device_resource_ref());
  stream.synchronize();
  auto* dst = static_cast<uint8_t*>(destination.data());
  StreamGate gate;
  CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.value(), waitForStreamGate, &gate));
  gate.entered.get_future().wait();
  auto waiter = std::async(std::launch::async, [&] {
    return input->device_read(10, 100, dst, stream.view());
  });
  EXPECT_EQ(waiter.wait_for(50ms), std::future_status::timeout);
  gate.release.set_value();
  EXPECT_EQ(waiter.get(), 100);
  EXPECT_EQ(fromDevice(dst, 100), read_->data.substr(10, 100));
}

TEST_F(CachingDataSourceTest, deviceFutureRetainsDelegate) {
  std::promise<void> release;
  read_->gate = release.get_future().share();
  auto input = source();
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      100, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  auto future = input->device_read_async(10, 100, dst, stream.view());
  input.reset();
  EXPECT_FALSE(read_->destroyed);
  release.set_value();
  EXPECT_EQ(future.get(), 100);
  EXPECT_EQ(fromDevice(dst, 100), read_->data.substr(10, 100));
}

TEST_F(CachingDataSourceTest, usesTheDestinationStreamsDevice) {
  int count = 0;
  CUDF_CUDA_TRY(cudaGetDeviceCount(&count));
  if (count < 2) {
    GTEST_SKIP() << "Requires two CUDA devices";
  }
  auto input = source();
  for (int device : {0, 1, 0}) {
    const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
    rmm::cuda_stream stream;
    rmm::device_buffer destination(
        100, stream.view(), cudf::get_current_device_resource_ref());
    auto* dst = static_cast<uint8_t*>(destination.data());
    EXPECT_EQ(input->device_read_async(0, 100, dst, stream.view()).get(), 100);
    EXPECT_EQ(fromDevice(dst, 100), std::string(100, 'x'));
  }
  EXPECT_EQ(read_->reads, 1);
}

TEST_F(CachingDataSourceTest, completionAndDiscardWaitForH2D) {
  for (bool discard : {false, true}) {
    auto input = source();
    rmm::cuda_stream stream;
    rmm::device_buffer destination(
        100, stream.view(), cudf::get_current_device_resource_ref());
    stream.synchronize();
    auto* dst = static_cast<uint8_t*>(destination.data());
    StreamGate gate;
    CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.value(), waitForStreamGate, &gate));
    gate.entered.get_future().wait();
    // Use distinct ranges so both passes exercise asynchronous cache fills.
    auto future =
        input->device_read_async(discard ? 101 : 1, 100, dst, stream.view());
    auto waiter = std::async(
        std::launch::async, [f = std::move(future), discard]() mutable {
          if (discard) {
            f = {};
          } else {
            EXPECT_EQ(f.get(), 100);
          }
        });
    EXPECT_EQ(waiter.wait_for(50ms), std::future_status::timeout);
    gate.release.set_value();
    waiter.get();
    EXPECT_EQ(fromDevice(dst, 100), std::string(100, 'x'));
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
