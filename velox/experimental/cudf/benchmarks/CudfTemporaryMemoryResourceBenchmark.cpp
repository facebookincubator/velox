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

// Measures what charging cuDF's implicit temporary allocations to Velox pools
// costs, on the resource stack registerCudf() installs. Each thread allocates
// and frees on its own stream from the cuDF stream pool, as operators do. The
// columns are:
//   - direct: the statistics adaptor over the configured RMM resource, which
//     is what cuDF allocated from before the dispatcher;
//   - dispatch: the dispatcher outside any operator call, which falls back to
//     the same adaptor;
//   - scoped: the dispatcher inside an operator call, which charges a
//     thread-safe leaf of a `gpu` custom root and keeps a copy of the
//     resource per allocation.

#include "velox/experimental/cudf/exec/CudfMemoryResource.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream_ref>

#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <vector>

DEFINE_string(memory_resource, "async", "RMM resource mode to benchmark.");
DEFINE_int32(memory_percent, 50, "Percent of device memory for pooled modes.");
DEFINE_int32(allocations_per_thread, 20000, "Allocate/free pairs per thread.");
DEFINE_int32(allocation_bytes, 4096, "Size of each temporary allocation.");

namespace facebook::velox::cudf_velox {
namespace {

using DeviceResource = cuda::mr::any_resource<cuda::mr::device_accessible>;

// Nanoseconds per allocate/free pair, averaged over all threads. Thread i
// allocates from 'resource' with 'operatorResources[i]' installed as the
// operator's resources, or with none when 'operatorResources' is empty.
double runTrial(
    rmm::device_async_resource_ref resource,
    std::vector<CudfMemoryResource>& operatorResources,
    int32_t numThreads,
    int32_t allocationsPerThread,
    std::size_t bytes) {
  constexpr std::size_t kAlignment = 256;

  std::atomic<int32_t> ready{0};
  std::atomic<bool> start{false};

  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for (int32_t i = 0; i < numThreads; ++i) {
    threads.emplace_back([&, i]() {
      const auto stream = cudfGlobalStreamPool().get_stream();
      std::optional<ScopedCudfMemoryResources> scope;
      if (!operatorResources.empty()) {
        rmm::device_async_resource_ref operatorResource{operatorResources[i]};
        scope.emplace(operatorResource, operatorResource);
      }
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

// Returns one resource per thread, each charging its own leaf of 'gpuRoot',
// like the operators of concurrent drivers.
std::vector<CudfMemoryResource> makeOperatorResources(
    const DeviceResource& upstream,
    const std::shared_ptr<memory::MemoryPool>& gpuRoot,
    const std::shared_ptr<memory::CustomMemoryResource>& resourceOwner,
    int32_t numThreads) {
  std::vector<CudfMemoryResource> resources;
  resources.reserve(numThreads);
  for (int32_t i = 0; i < numThreads; ++i) {
    resources.emplace_back(
        upstream,
        gpuRoot->addLeafChild("operator." + std::to_string(i)),
        resourceOwner);
  }
  return resources;
}

void runBenchmark() {
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  constexpr int64_t kCapacity = 1L << 40;
  memory::MemoryAllocator::Options allocatorOptions;
  allocatorOptions.capacity = kCapacity;
  auto resourceOwner = std::make_shared<memory::CustomMemoryResource>(
      std::string{kCudfMemoryResourceTag},
      std::make_shared<memory::MallocAllocator>(allocatorOptions),
      memory::MemoryArbitrator::create({}),
      []() { return memory::MemoryReclaimer::create(0); },
      kCapacity);
  auto gpuRoot = memory::memoryManager()->addCustomRootPool(
      "temporaryMemoryResourceBenchmark.gpu", resourceOwner);

  DeviceResource upstream = rmm::mr::statistics_resource_adaptor{
      createMemoryResource(FLAGS_memory_resource, FLAGS_memory_percent)};
  auto dispatcher = createThreadLocalTemporaryMemoryResource(upstream);

  const std::size_t bytes = FLAGS_allocation_bytes;
  std::printf(
      "mode=%s bytes=%zu allocations/thread=%d\n\n",
      FLAGS_memory_resource.c_str(),
      bytes,
      FLAGS_allocations_per_thread);
  std::printf(
      "%8s %12s %12s %12s %10s %10s\n",
      "threads",
      "direct(ns)",
      "dispatch(ns)",
      "scoped(ns)",
      "dispatch/x",
      "scoped/x");

  std::vector<CudfMemoryResource> noOperatorResources;
  for (const int32_t numThreads : {1, 2, 4, 8, 16, 32}) {
    auto operatorResources =
        makeOperatorResources(upstream, gpuRoot, resourceOwner, numThreads);

    // Warm the pool so the first trial does not pay for growth.
    runTrial(upstream, noOperatorResources, numThreads, 1000, bytes);

    const auto direct = runTrial(
        upstream,
        noOperatorResources,
        numThreads,
        FLAGS_allocations_per_thread,
        bytes);
    const auto dispatched = runTrial(
        dispatcher,
        noOperatorResources,
        numThreads,
        FLAGS_allocations_per_thread,
        bytes);
    const auto scoped = runTrial(
        dispatcher,
        operatorResources,
        numThreads,
        FLAGS_allocations_per_thread,
        bytes);
    std::printf(
        "%8d %12.1f %12.1f %12.1f %9.2fx %9.2fx\n",
        numThreads,
        direct,
        dispatched,
        scoped,
        dispatched / direct,
        scoped / direct);
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::cudf_velox::runBenchmark();
  return 0;
}
