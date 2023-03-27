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

#include <deque>

#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/common/time/Timer.h"

DEFINE_uint64(
    max_memory_bytes,
    1'000'000'000,
    "The cap of memory size in bytes");
DEFINE_uint64(allocation_size, 4096, "The memory allocation size in bytes");
DEFINE_uint32(num_memory_threads, 64, "The number of memory thread");
DEFINE_uint32(
    num_allocations_per_thread,
    50'000,
    "The number of allocations per each memory thread");
DEFINE_uint32(
    memory_allocator_type,
    0,
    "The type of memory allocator. 0 is malloc allocator, 1 is mmap allocator");
DEFINE_uint32(
    num_runs,
    32,
    "The number of benchmark runs and reports the average results");

using namespace facebook::velox;
using namespace facebook::velox::memory;

namespace {
class MemoryOperator {
 public:
  MemoryOperator(
      MemoryManager* memoryManager,
      std::shared_ptr<MemoryUsageTracker> tracker,
      uint64_t maxMemory,
      uint64_t allocationSize,
      uint32_t maxOps)
      : maxMemory_(maxMemory),
        allocationBytes_(allocationSize),
        maxOps_(maxOps),
        pool_(memoryManager->getPool(
            fmt::format("MemoryOperator{}", poolId_++),
            MemoryPool::Kind::kLeaf)) {
    rng_.seed(1234);
  }

  ~MemoryOperator() = default;

  void run();

  uint64_t clockCount() const {
    return clockCount_;
  }

 private:
  struct Allocation {
    void* ptr;

    explicit Allocation(void* _ptr) : ptr(_ptr) {}
  };

  bool full() const;

  void allocate();

  void free();

  void cleanup();

  static inline int32_t poolId_{0};

  const uint64_t maxMemory_;
  const size_t allocationBytes_;
  const uint32_t maxOps_;
  const std::shared_ptr<MemoryPool> pool_;

  folly::Random::DefaultGenerator rng_;
  uint64_t allocatedBytes_{0};
  std::deque<Allocation> allocations_;
  uint64_t clockCount_{0};
};

void MemoryOperator::run() {
  for (int i = 0; i < maxOps_; ++i) {
    allocate();
  }
  cleanup();
}

void MemoryOperator::allocate() {
  while (full()) {
    free();
  }
  {
    ClockTimer cpuTimer(clockCount_);
    allocations_.emplace_back(pool_->allocate(allocationBytes_));
  }
  allocatedBytes_ += allocationBytes_;
}

void MemoryOperator::free() {
  const int freeIdx = folly::Random::rand32(rng_) % allocations_.size();
  Allocation freeAllocation = allocations_[freeIdx];
  allocations_[freeIdx] = allocations_.back();
  allocations_.pop_back();
  {
    ClockTimer cpuTimer(clockCount_);
    pool_->free(freeAllocation.ptr, allocationBytes_);
  }
  allocatedBytes_ -= allocationBytes_;
}

void MemoryOperator::cleanup() {
  for (const auto& allocation : allocations_) {
    {
      ClockTimer cpuTimer(clockCount_);
      pool_->free(allocation.ptr, allocationBytes_);
    }
  }
}

bool MemoryOperator::full() const {
  return allocatedBytes_ + allocationBytes_ >= maxMemory_;
}

class MemoryAllocationBenchMark {
 public:
  enum class Type {
    kMalloc = 0,
    kMmap = 1,
  };

  struct Options {
    Type allocatorType;
    uint64_t maxMemory;
    uint64_t allocationBytes;
    uint32_t numThreads;
    uint32_t numOpsPerThread;
  };

  explicit MemoryAllocationBenchMark(const Options& options)
      : options_(options), tracker_(MemoryUsageTracker::create()) {
    const int64_t maxMemory = options_.maxMemory + (256 << 20);
    switch (options_.allocatorType) {
      case Type::kMmap: {
        memory::MmapAllocator::Options mmapOptions;
        mmapOptions.capacity = maxMemory;
        allocator_ = std::make_shared<MmapAllocator>(mmapOptions);
        manager_ = std::make_shared<MemoryManager>(IMemoryManager::Options{
            .capacity = maxMemory, .allocator = allocator_.get()});
      } break;
      case Type::kMalloc:
        manager_ = std::make_shared<MemoryManager>(
            IMemoryManager::Options{.capacity = maxMemory});
        break;
      default:
        VELOX_USER_FAIL(
            "Unknown allocator type: {}",
            static_cast<int>(options_.allocatorType));
        break;
    }
  }

  ~MemoryAllocationBenchMark() = default;

  void run();

  void printStats();

 private:
  struct Result {
    uint64_t runTimeUs;
    uint64_t clockCount;
  };

  const Options options_;
  const std::shared_ptr<MemoryUsageTracker> tracker_;
  std::shared_ptr<MmapAllocator> allocator_;
  std::shared_ptr<MemoryManager> manager_;
  std::vector<Result> results_;
};

void MemoryAllocationBenchMark::run() {
  std::vector<std::thread> memThreads;
  memThreads.reserve(options_.numThreads);
  std::vector<std::unique_ptr<MemoryOperator>> operators;
  operators.reserve(options_.numThreads);
  uint64_t runTimeUs{0};
  uint64_t clockCount{0};
  {
    MicrosecondTimer clock(&runTimeUs);
    for (int i = 0; i < options_.numThreads; ++i) {
      auto memOp = std::make_unique<MemoryOperator>(
          manager_.get(),
          tracker_,
          options_.maxMemory / options_.numThreads,
          options_.allocationBytes,
          options_.numOpsPerThread);
      memThreads.push_back(
          std::thread([rawOp = memOp.get()]() { rawOp->run(); }));
      operators.push_back(std::move(memOp));
    }
    for (int i = 0; i < options_.numThreads; ++i) {
      memThreads[i].join();
    }
  }
  for (const auto& op : operators) {
    clockCount += op->clockCount();
  }

  results_.push_back({runTimeUs, clockCount});
}

void MemoryAllocationBenchMark::printStats() {
  double sumRunTumeMs{0};
  double sumClockCount{0};
  for (const auto& result : results_) {
    sumRunTumeMs += result.runTimeUs / 1000;
    sumClockCount += result.clockCount;
  }
  const uint64_t avgRunTimeMs = sumRunTumeMs / results_.size();
  const uint64_t avgClockCount = sumClockCount / results_.size();
  LOG(INFO) << "\n\t\tSIZE\t\tTIME\t\tCLOCK\n\t\t"
            << succinctBytes(options_.allocationBytes) << "\t\t"
            << succinctMillis(avgRunTimeMs) << "\t\t" << avgClockCount;
}
} // namespace

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  MemoryAllocationBenchMark::Options options;
  options.numThreads = FLAGS_num_memory_threads;
  options.maxMemory = FLAGS_max_memory_bytes;
  options.allocationBytes = FLAGS_allocation_size;
  options.allocatorType = FLAGS_memory_allocator_type == 0
      ? MemoryAllocationBenchMark::Type::kMalloc
      : MemoryAllocationBenchMark::Type::kMmap;
  options.numOpsPerThread = FLAGS_num_allocations_per_thread;
  auto benchmark = std::make_unique<MemoryAllocationBenchMark>(options);
  for (int i = 0; i < FLAGS_num_runs; ++i) {
    benchmark->run();
  }
  benchmark->printStats();
  return 0;
}
