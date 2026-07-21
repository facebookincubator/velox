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

#include "velox/dwio/common/DirectBufferedInput.h"

#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/init/Init.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "velox/common/caching/FileIds.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"

namespace facebook::velox::dwio::common {
namespace {

using namespace std::chrono_literals;

constexpr uint64_t kRegionBytes = 64 << 10;
constexpr uint64_t kRegionStride = 2 << 20;
constexpr auto kReadLatency = 10ms;
constexpr int32_t kRounds = 3;

struct Scenario {
  const char* name;
  int32_t readCalls;
  int32_t nativeMaxInFlight;
};

// Shapes observed in the downstream seekexp2 benchmark. This benchmark
// isolates the IO-overlap component; it does not model Parquet decode or Azure
// service behavior and therefore must not be reported as end-to-end speedup.
constexpr Scenario kScenarios[] = {
    {"full_scan", 45, 21},
    {"sparse_every_40th_column", 125, 22},
    {"filter_bytes_bound", 345, 6},
    {"full_outer_join", 604, 33},
};

uint8_t expectedByte(uint64_t offset) {
  return static_cast<uint8_t>(offset % 251);
}

uint64_t fillBuffers(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers) {
  uint64_t position = offset;
  uint64_t total = 0;
  for (const auto& buffer : buffers) {
    if (buffer.data() != nullptr) {
      for (uint64_t index = 0; index < buffer.size(); ++index) {
        buffer.data()[index] =
            static_cast<char>(expectedByte(position + index));
      }
    }
    position += buffer.size();
    total += buffer.size();
  }
  return total;
}

class SyntheticReadFile final : public ReadFile {
 public:
  SyntheticReadFile(
      uint64_t fileSize,
      bool nativeAsync,
      int32_t maxInFlight,
      std::chrono::milliseconds latency)
      : fileSize_(fileSize),
        nativeAsync_(nativeAsync),
        maxInFlight_(maxInFlight),
        latency_(latency) {
    VELOX_CHECK_GT(maxInFlight_, 0);
    completionThread_ = std::thread([this] { completionLoop(); });
  }

  ~SyntheticReadFile() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_one();
    completionThread_.join();
  }

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buffer,
      const FileIoContext& /*context*/) const override {
    std::this_thread::sleep_for(latency_);
    auto* bytes = static_cast<char*>(buffer);
    for (uint64_t index = 0; index < length; ++index) {
      bytes[index] = static_cast<char>(expectedByte(offset + index));
    }
    syncCalls_.fetch_add(1, std::memory_order_relaxed);
    return {bytes, length};
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& /*context*/) const override {
    std::this_thread::sleep_for(latency_);
    syncCalls_.fetch_add(1, std::memory_order_relaxed);
    return fillBuffers(offset, buffers);
  }

  folly::SemiFuture<uint64_t> preadvAsync(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& /*context*/) const override {
    VELOX_CHECK(nativeAsync_);
    auto request = std::make_shared<Request>();
    request->offset = offset;
    request->buffers = buffers;
    auto future = request->promise.getSemiFuture();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_.push_back(std::move(request));
      asyncCalls_.fetch_add(1, std::memory_order_relaxed);
    }
    condition_.notify_one();
    return future;
  }

  bool hasPreadvAsync() const override {
    return nativeAsync_;
  }

  uint64_t size() const override {
    return fileSize_;
  }

  uint64_t memoryUsage() const override {
    return sizeof(*this);
  }

  bool shouldCoalesce() const override {
    return true;
  }

  std::string getName() const override {
    return nativeAsync_ ? "SyntheticNativeAsyncReadFile"
                        : "SyntheticBlockingReadFile";
  }

  uint64_t getNaturalReadSize() const override {
    return kRegionBytes;
  }

  uint64_t syncCalls() const {
    return syncCalls_.load(std::memory_order_relaxed);
  }

  uint64_t asyncCalls() const {
    return asyncCalls_.load(std::memory_order_relaxed);
  }

  uint64_t peakInFlight() const {
    return peakInFlight_.load(std::memory_order_relaxed);
  }

 private:
  struct Request {
    uint64_t offset;
    std::vector<folly::Range<char*>> buffers;
    folly::Promise<uint64_t> promise;
    std::chrono::steady_clock::time_point readyAt;
    uint64_t sequence;
  };

  struct ReadySooner {
    bool operator()(
        const std::shared_ptr<Request>& left,
        const std::shared_ptr<Request>& right) const {
      if (left->readyAt != right->readyAt) {
        return left->readyAt > right->readyAt;
      }
      return left->sequence > right->sequence;
    }
  };

  void admitPending() const {
    const auto now = std::chrono::steady_clock::now();
    while (!pending_.empty() && active_.size() < maxInFlight_) {
      auto request = std::move(pending_.front());
      pending_.pop_front();
      request->readyAt = now + latency_;
      request->sequence = nextSequence_++;
      active_.push(std::move(request));
      const auto inFlight = active_.size();
      auto peak = peakInFlight_.load(std::memory_order_relaxed);
      while (inFlight > peak &&
             !peakInFlight_.compare_exchange_weak(
                 peak, inFlight, std::memory_order_relaxed)) {
      }
    }
  }

  void completionLoop() const {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      admitPending();
      if (stopping_ && pending_.empty() && active_.empty()) {
        return;
      }
      if (active_.empty()) {
        condition_.wait(
            lock, [this] { return stopping_ || !pending_.empty(); });
        continue;
      }

      const auto readyAt = active_.top()->readyAt;
      if (condition_.wait_until(lock, readyAt) != std::cv_status::timeout) {
        continue;
      }

      std::vector<std::shared_ptr<Request>> completed;
      const auto now = std::chrono::steady_clock::now();
      while (!active_.empty() && active_.top()->readyAt <= now) {
        completed.push_back(active_.top());
        active_.pop();
      }
      admitPending();
      lock.unlock();
      for (const auto& request : completed) {
        try {
          request->promise.setValue(
              fillBuffers(request->offset, request->buffers));
        } catch (...) {
          request->promise.setException(
              folly::exception_wrapper{std::current_exception()});
        }
      }
      lock.lock();
    }
  }

  const uint64_t fileSize_;
  const bool nativeAsync_;
  const size_t maxInFlight_;
  const std::chrono::milliseconds latency_;

  mutable std::mutex mutex_;
  mutable std::condition_variable condition_;
  mutable std::deque<std::shared_ptr<Request>> pending_;
  mutable std::priority_queue<
      std::shared_ptr<Request>,
      std::vector<std::shared_ptr<Request>>,
      ReadySooner>
      active_;
  mutable uint64_t nextSequence_{0};
  mutable bool stopping_{false};
  mutable std::atomic<uint64_t> syncCalls_{0};
  mutable std::atomic<uint64_t> asyncCalls_{0};
  mutable std::atomic<uint64_t> peakInFlight_{0};
  std::thread completionThread_;
};

struct RunResult {
  int64_t elapsedUs;
  uint64_t syncCalls;
  uint64_t asyncCalls;
  uint64_t peakInFlight;
  uint64_t bytesRead;
};

RunResult
runOnce(const Scenario& scenario, bool nativeAsync, int32_t executorThreads) {
  const uint64_t fileSize =
      static_cast<uint64_t>(scenario.readCalls) * kRegionStride + kRegionBytes;
  auto readFile = std::make_shared<SyntheticReadFile>(
      fileSize,
      nativeAsync,
      nativeAsync ? scenario.nativeMaxInFlight : 1,
      kReadLatency);

  std::unique_ptr<folly::IOThreadPoolExecutor> executor;
  if (!nativeAsync) {
    executor = std::make_unique<folly::IOThreadPoolExecutor>(executorThreads);
  }

  auto rootPool = memory::memoryManager()->addRootPool();
  auto pool = rootPool->addLeafChild("AsyncPrefetchBenchmark");
  auto ioStatistics = std::make_shared<IoStatistics>();
  auto tracker = std::make_shared<cache::ScanTracker>(
      "AsyncPrefetchBenchmark", nullptr, 256 << 10);
  io::ReaderOptions options(pool.get());
  options.setDataIoStats(ioStatistics);
  options.setMetadataIoStats(ioStatistics);
  options.setLoadQuantum(kRegionBytes);
  options.setMaxCoalesceDistance(1 << 20);
  options.setMaxCoalesceBytes(kRegionBytes);
  options.setMaxOutstandingPrefetchBytes(256LL << 20);

  StringIdLease fileId(
      fileIds(),
      std::string("AsyncPrefetchBenchmark-") + scenario.name +
          (nativeAsync ? "-native" : "-blocking"));
  StringIdLease groupId(fileIds(), "AsyncPrefetchBenchmarkGroup");
  DirectBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      tracker,
      std::move(groupId),
      ioStatistics,
      nullptr,
      executor.get(),
      options);

  std::vector<std::unique_ptr<SeekableInputStream>> streams;
  streams.reserve(scenario.readCalls);
  for (int32_t index = 0; index < scenario.readCalls; ++index) {
    streams.push_back(input.enqueue(
        ::facebook::velox::common::Region{
            static_cast<uint64_t>(index) * kRegionStride, kRegionBytes},
        nullptr));
  }

  const auto start = std::chrono::steady_clock::now();
  input.load(LogType::TEST);
  uint64_t bytesRead = 0;
  for (int32_t index = 0; index < scenario.readCalls; ++index) {
    const void* buffer = nullptr;
    int32_t size = 0;
    while (streams[index]->Next(&buffer, &size)) {
      VELOX_CHECK_NOT_NULL(buffer);
      VELOX_CHECK_GT(size, 0);
      const auto* bytes = static_cast<const uint8_t*>(buffer);
      const uint64_t absoluteOffset =
          static_cast<uint64_t>(index) * kRegionStride +
          bytesRead % kRegionBytes;
      VELOX_CHECK_EQ(bytes[0], expectedByte(absoluteOffset));
      VELOX_CHECK_EQ(bytes[size - 1], expectedByte(absoluteOffset + size - 1));
      bytesRead += size;
    }
  }
  const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
  VELOX_CHECK_EQ(
      bytesRead, static_cast<uint64_t>(scenario.readCalls) * kRegionBytes);

  return {
      elapsedUs,
      readFile->syncCalls(),
      readFile->asyncCalls(),
      readFile->peakInFlight(),
      bytesRead};
}

void printResult(
    const Scenario& scenario,
    const std::string& mode,
    const std::string& round,
    const RunResult& result) {
  std::cout << scenario.name << ',' << mode << ',' << round << ','
            << result.elapsedUs << ',' << result.syncCalls << ','
            << result.asyncCalls << ',' << result.peakInFlight << ','
            << result.bytesRead << '\n';
}

} // namespace
} // namespace facebook::velox::dwio::common

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});

  using namespace facebook::velox::dwio::common;
  std::cout
      << "scenario,mode,round,elapsed_us,sync_calls,async_calls,peak_in_flight,bytes_read\n";
  for (const auto& scenario : kScenarios) {
    for (const int32_t executorThreads : {1, 2, 8}) {
      const std::string mode =
          "blocking_executor_" + std::to_string(executorThreads);
      std::vector<RunResult> results;
      results.reserve(kRounds);
      for (int32_t round = 1; round <= kRounds; ++round) {
        results.push_back(runOnce(scenario, false, executorThreads));
        printResult(scenario, mode, std::to_string(round), results.back());
      }
      printResult(
          scenario,
          mode,
          "best",
          *std::min_element(
              results.begin(),
              results.end(),
              [](const auto& left, const auto& right) {
                return left.elapsedUs < right.elapsedUs;
              }));
    }
    std::vector<RunResult> results;
    results.reserve(kRounds);
    for (int32_t round = 1; round <= kRounds; ++round) {
      results.push_back(runOnce(scenario, true, 0));
      printResult(
          scenario,
          "native_async_no_executor",
          std::to_string(round),
          results.back());
    }
    printResult(
        scenario,
        "native_async_no_executor",
        "best",
        *std::min_element(
            results.begin(),
            results.end(),
            [](const auto& left, const auto& right) {
              return left.elapsedUs < right.elapsedUs;
            }));
  }
  return 0;
}
