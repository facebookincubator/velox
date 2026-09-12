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

// Measures the cost the thread-local temporary memory resource dispatcher adds
// to cuDF's implicit allocation path. The dispatcher keeps a pointer-keyed map
// so a temporary freed on another thread still routes back to the resource
// that allocated it, and that map is what this benchmark prices.
//
// Run against the memory resource modes that matter: "async" (the default) has
// no host-side lock of its own, so the dispatcher is the only serialization
// point, whereas "pool" already serializes every allocation on RMM's own
// mutex.

#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cuda/stream_ref>

#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

DEFINE_string(memory_resource, "async", "RMM resource mode to benchmark.");
DEFINE_int32(memory_percent, 50, "Percent of device memory for pooled modes.");
DEFINE_int32(allocations_per_thread, 20000, "Allocate/free pairs per thread.");
DEFINE_int32(allocation_bytes, 4096, "Size of each temporary allocation.");

namespace facebook::velox::cudf_velox {
namespace {

// Nanoseconds per allocate/free pair, averaged over all threads.
double runTrial(
    rmm::device_async_resource_ref resource,
    int32_t numThreads,
    int32_t allocationsPerThread,
    std::size_t bytes) {
  constexpr std::size_t kAlignment = 256;
  const cuda::stream_ref stream{cudaStream_t{0}};

  std::atomic<int32_t> ready{0};
  std::atomic<bool> start{false};

  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for (int32_t i = 0; i < numThreads; ++i) {
    threads.emplace_back([&]() {
      ready.fetch_add(1);
      while (!start.load(std::memory_order_acquire)) {
      }
      for (int32_t j = 0; j < allocationsPerThread; ++j) {
        auto* pointer = resource.allocate(stream, bytes, kAlignment);
        resource.deallocate(stream, pointer, bytes, kAlignment);
      }
    });
  }

  while (ready.load() < numThreads) {
  }
  const auto began = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  const auto elapsed = std::chrono::steady_clock::now() - began;

  const auto nanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  return static_cast<double>(nanos) /
      (static_cast<double>(numThreads) * allocationsPerThread);
}

void runBenchmark() {
  auto upstream =
      createMemoryResource(FLAGS_memory_resource, FLAGS_memory_percent);
  auto dispatcher = createThreadLocalTemporaryMemoryResource(upstream);

  const std::size_t bytes = FLAGS_allocation_bytes;
  std::printf(
      "mode=%s bytes=%zu allocations/thread=%d\n\n",
      FLAGS_memory_resource.c_str(),
      bytes,
      FLAGS_allocations_per_thread);
  std::printf(
      "%8s %14s %14s %10s\n", "threads", "direct(ns)", "dispatch(ns)", "ratio");

  for (const int32_t numThreads : {1, 2, 4, 8, 16, 32}) {
    // Warm the pool so the first trial does not pay for growth.
    runTrial(upstream, numThreads, 1000, bytes);

    const auto direct =
        runTrial(upstream, numThreads, FLAGS_allocations_per_thread, bytes);
    const auto dispatched =
        runTrial(dispatcher, numThreads, FLAGS_allocations_per_thread, bytes);
    std::printf(
        "%8d %14.1f %14.1f %10.2fx\n",
        numThreads,
        direct,
        dispatched,
        dispatched / direct);
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::cudf_velox::runBenchmark();
  return 0;
}
