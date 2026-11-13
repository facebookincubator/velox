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

#include "velox/experimental/cudf/memory/PinnedMemoryAllocator.h"

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <folly/ScopeGuard.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

using namespace facebook::velox;
using cudf_velox::PinnedMemoryAllocator;
namespace {
constexpr uint64_t kMiB = 1 << 20;
using Clock = std::chrono::steady_clock;

template <typename Allocate, typename Free>
void allocationCase(
    const char* name,
    uint64_t bytes,
    int iterations,
    int threads,
    Allocate allocate,
    Free free) {
  std::vector<std::vector<double>> samples(threads);
  std::vector<std::thread> workers;
  auto begin = Clock::now();
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      samples[t].reserve(iterations);
      for (int i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        auto* ptr = allocate(bytes);
        VELOX_CHECK_NOT_NULL(ptr);
        free(ptr, bytes);
        samples[t].push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - start)
                .count());
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  const auto seconds =
      std::chrono::duration<double>(Clock::now() - begin).count();
  std::vector<double> merged;
  for (auto& sample : samples) {
    merged.insert(merged.end(), sample.begin(), sample.end());
  }
  std::sort(merged.begin(), merged.end());
  std::cout << name << ',' << bytes << ',' << threads << ',' << merged.size()
            << ',' << merged[merged.size() / 2] << ','
            << merged[merged.size() * 95 / 100] << ','
            << merged.size() / seconds << '\n';
}

void cacheCase(int cycles, uint64_t spillBytes) {
  auto mr = std::make_shared<PinnedMemoryAllocator>(
      PinnedMemoryAllocator::Options{64 * kMiB, 0});
  cache::AsyncDataCache::Options options;
  options.numShards = 1;
  auto cache = cache::AsyncDataCache::create(mr.get(), nullptr, options);
  auto shutdown = folly::makeGuard([&] { cache->shutdown(); });
  StringIdLease file(fileIds(), std::string_view("pinned-memory-benchmark"));
  int misses = 0, hits = 0;
  uint64_t peakCharged = 0;
  std::vector<double> spillUs;
  auto begin = Clock::now();
  for (int cycle = 0; cycle < cycles; ++cycle) {
    for (int entry = 0; entry < 48; ++entry) {
      auto pin =
          cache->findOrCreate({file.id(), entry * kMiB}, kMiB, true, nullptr);
      if (pin.checkedEntry()->isExclusive()) {
        ++misses;
        std::memset(pin.checkedEntry()->contiguousData(), entry, kMiB);
        pin.checkedEntry()->setExclusiveToShared();
      } else {
        ++hits;
        VELOX_CHECK_EQ(pin.checkedEntry()->contiguousData()[0], entry);
      }
    }
    auto start = Clock::now();
    auto* spill = mr->allocateBytes(spillBytes);
    VELOX_CHECK_NOT_NULL(spill);
    spillUs.push_back(
        std::chrono::duration<double, std::micro>(Clock::now() - start)
            .count());
    peakCharged = std::max(
        peakCharged, memory::AllocationTraits::pageBytes(mr->numAllocated()));
    std::memset(spill, 0x49, spillBytes);
    mr->freeBytes(spill, spillBytes);
  }
  auto stats = cache->refreshStats();
  std::sort(spillUs.begin(), spillUs.end());
  std::cout << "cache,spill_bytes=" << spillBytes << ",cycles=" << cycles
            << ",seconds="
            << std::chrono::duration<double>(Clock::now() - begin).count()
            << ",hits=" << hits << ",misses=" << misses
            << ",evictions=" << stats.numEvict
            << ",spill_p50_us=" << spillUs[spillUs.size() / 2]
            << ",spill_p95_us=" << spillUs[spillUs.size() * 95 / 100]
            << ",peak_charged=" << peakCharged
            << ",retained=" << mr->retainedBytes() << '\n';
}

void transferCase(const char* name, void* host, size_t bytes, int iterations) {
  void* device;
  VELOX_CHECK(cudaMalloc(&device, bytes) == cudaSuccess);
  auto freeDevice = folly::makeGuard([&] { cudaFree(device); });
  cudaStream_t stream;
  VELOX_CHECK(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  auto freeStream = folly::makeGuard([&] { cudaStreamDestroy(stream); });
  cudaEvent_t start, stop;
  VELOX_CHECK(cudaEventCreate(&start) == cudaSuccess);
  auto freeStart = folly::makeGuard([&] { cudaEventDestroy(start); });
  VELOX_CHECK(cudaEventCreate(&stop) == cudaSuccess);
  auto freeStop = folly::makeGuard([&] { cudaEventDestroy(stop); });
  std::memset(host, 0x35, bytes);
  VELOX_CHECK(cudaMemsetAsync(device, 0x35, bytes, stream) == cudaSuccess);
  VELOX_CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  for (auto direction : {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost}) {
    void* destination = direction == cudaMemcpyHostToDevice ? device : host;
    const void* source = direction == cudaMemcpyHostToDevice ? host : device;
    VELOX_CHECK(cudaEventRecord(start, stream) == cudaSuccess);
    for (int i = 0; i < iterations; ++i) {
      VELOX_CHECK(
          cudaMemcpyAsync(destination, source, bytes, direction, stream) ==
          cudaSuccess);
    }
    VELOX_CHECK(cudaEventRecord(stop, stream) == cudaSuccess);
    VELOX_CHECK(cudaEventSynchronize(stop) == cudaSuccess);
    float milliseconds;
    VELOX_CHECK(
        cudaEventElapsedTime(&milliseconds, start, stop) == cudaSuccess);
    const double gib = static_cast<double>(bytes) * iterations / (1ULL << 30);
    std::cout << "transfer,allocator=" << name << ",bytes=" << bytes
              << ",direction="
              << (direction == cudaMemcpyHostToDevice ? "H2D" : "D2H")
              << ",GiB_per_second=" << gib / (milliseconds / 1000.0) << '\n';
  }
  const auto* data = static_cast<const uint8_t*>(host);
  VELOX_CHECK(
      std::all_of(data, data + bytes, [](auto byte) { return byte == 0x35; }));
}
} // namespace

// Usage: velox_cudf_pinned_memory_benchmark [iterations=1000] [cache_cycles=10]
int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::stoi(argv[1]) : 1000;
  const int cycles = argc > 2 ? std::stoi(argv[2]) : 10;
  VELOX_CHECK_GT(iterations, 0);
  VELOX_CHECK_GT(cycles, 0);
  int device;
  VELOX_CHECK(cudaGetDevice(&device) == cudaSuccess);
  // Exclude context initialization and lazy driver setup from cold pool growth.
  {
    cuda::pinned_memory_pool primer(0);
    auto* ptr = primer.allocate_sync(4096);
    primer.deallocate_sync(ptr, 4096);
    VELOX_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  }
  std::cout << "allocator,bytes,threads,samples,p50_us,p95_us,ops_per_second\n";
  for (auto bytes : {4096ULL, 65536ULL, 1048576ULL, 8388608ULL}) {
    for (int threads : {1, 4}) {
      cuda::memory_pool_properties properties;
      properties.max_pool_size = 64 * kMiB;
      cuda::pinned_memory_pool pool(0, properties);
      PinnedMemoryAllocator mr({64 * kMiB, 0});
      // First allocation includes backing growth; report it separately.
      allocationCase(
          "cccl_cold",
          bytes,
          1,
          1,
          [&](auto size) { return pool.allocate_sync(size); },
          [&](auto* ptr, auto size) { pool.deallocate_sync(ptr, size); });
      allocationCase(
          "adapter_cold",
          bytes,
          1,
          1,
          [&](auto size) { return mr.allocateBytes(size); },
          [&](auto* ptr, auto size) { mr.freeBytes(ptr, size); });
      allocationCase(
          "cccl_warm",
          bytes,
          iterations,
          threads,
          [&](auto size) { return pool.allocate_sync(size); },
          [&](auto* ptr, auto size) { pool.deallocate_sync(ptr, size); });
      allocationCase(
          "adapter_warm",
          bytes,
          iterations,
          threads,
          [&](auto size) { return mr.allocateBytes(size); },
          [&](auto* ptr, auto size) { mr.freeBytes(ptr, size); });
    }
  }
  cacheCase(cycles, 8 * kMiB);
  cacheCase(cycles, 48 * kMiB);
  for (auto bytes : {kMiB, 8 * kMiB}) {
    cuda::pinned_memory_pool pool(0);
    auto* direct = pool.allocate_sync(bytes);
    auto freeDirect =
        folly::makeGuard([&] { pool.deallocate_sync(direct, bytes); });
    PinnedMemoryAllocator mr({64 * kMiB, 0});
    auto* adapted = mr.allocateBytes(bytes);
    VELOX_CHECK_NOT_NULL(adapted);
    auto freeAdapted = folly::makeGuard([&] { mr.freeBytes(adapted, bytes); });
    transferCase("cccl", direct, bytes, iterations);
    transferCase("adapter", adapted, bytes, iterations);
  }
}
