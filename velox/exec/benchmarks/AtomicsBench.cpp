/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <atomic>
#include <thread>
#include <vector>
#include "velox/common/base/Portability.h"
#include "velox/exec/OneWayStatusFlag.h"

namespace {

using facebook::velox::exec::OneWayStatusFlag;
constexpr size_t kNumThreads = 88;
constexpr size_t kNumIterations = 10000;

#if defined(__x86_64__) && !defined(TSAN_BUILD)

class OneWayStatusFlagUnsafe {
 public:
  bool check() const {
    return fastStatus_ || atomicStatus_.load();
  }

  void set() {
    if (!fastStatus_) {
      atomicStatus_.store(true);
      fastStatus_ = true;
    }
  }

 private:
  bool fastStatus_{false};
  std::atomic_bool atomicStatus_{false};
};

#endif

void runParallelUpdates(
    std::function<void(size_t iter)> callback,
    int threadCount,
    int updateCount) {
  std::vector<std::thread> threads;

  for (size_t i = 0; i < threadCount; ++i) {
    threads.emplace_back([&]() {
      for (size_t j = 0; j < updateCount; ++j) {
        callback(j);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

BENCHMARK(std_atomic_bool_write_seq_cst) {
  std::atomic_bool flag{false};
  runParallelUpdates(
      [&](size_t iters) {
        for (size_t i = 0; i < iters; ++i) {
          flag.store(true);
          bool dummy{};
          folly::doNotOptimizeAway(dummy);
        }
      },
      kNumThreads, // Threads
      kNumIterations); // Iterations per thread
}

BENCHMARK(std_atomic_bool_write_release) {
  std::atomic_bool flag{false};
  runParallelUpdates(
      [&](size_t iters) {
        for (size_t i = 0; i < iters; ++i) {
          flag.store(true, std::memory_order_release);
          bool dummy{};
          folly::doNotOptimizeAway(dummy);
        }
      },
      kNumThreads, // Threads
      kNumIterations); // Iterations per thread
}

BENCHMARK(std_atomic_bool_write_relaxed) {
  std::atomic_bool flag{false};
  runParallelUpdates(
      [&](size_t iters) {
        for (size_t i = 0; i < iters; ++i) {
          flag.store(true, std::memory_order_relaxed);
          bool dummy{};
          folly::doNotOptimizeAway(dummy);
        }
      },
      kNumThreads, // Threads
      kNumIterations); // Iterations per thread
}

BENCHMARK(one_way_flag_write) {
  OneWayStatusFlag flag;
  runParallelUpdates(
      [&](size_t iters) {
        for (size_t i = 0; i < iters; ++i) {
          flag.set();
          bool dummy{};
          folly::doNotOptimizeAway(dummy);
        }
      },
      kNumThreads, // Threads
      kNumIterations); // Iterations per thread
}

#if defined(__x86_64__) && !defined(TSAN_BUILD)

BENCHMARK(one_way_flag_unsafe_write) {
  OneWayStatusFlagUnsafe flag;
  runParallelUpdates(
      [&](size_t iters) {
        for (size_t i = 0; i < iters; ++i) {
          flag.set();
          bool dummy{};
          folly::doNotOptimizeAway(dummy);
        }
      },
      kNumThreads, // Threads
      kNumIterations); // Iterations per thread
}

#endif

// Read Benchmarks
BENCHMARK(std_atomic_bool_read_seq_cst) {
  std::atomic_bool flag{false};
  runParallelUpdates(
      [&](size_t iters) {
        for (size_t i = 0; i < iters; ++i) {
          folly::doNotOptimizeAway(flag.load(std::memory_order_seq_cst));
        }
      },
      kNumThreads, // Threads
      kNumIterations); // Iterations per thread
}

BENCHMARK(std_atomic_bool_read_acquire) {
  std::atomic_bool flag{false};
  runParallelUpdates(
      [&](size_t iters) {
        for (size_t i = 0; i < iters; ++i) {
          folly::doNotOptimizeAway(flag.load(std::memory_order_acquire));
        }
      },
      kNumThreads, // Threads
      kNumIterations); // Iterations per thread
}

BENCHMARK(std_atomic_bool_read_relaxed) {
  std::atomic_bool flag{false};
  runParallelUpdates(
      [&](size_t iters) {
        for (size_t i = 0; i < iters; ++i) {
          folly::doNotOptimizeAway(flag.load(std::memory_order_relaxed));
        }
      },
      kNumThreads, // Threads
      kNumIterations); // Iterations per thread
}

BENCHMARK(one_way_flag_read) {
  OneWayStatusFlag flag;
  runParallelUpdates(
      [&](size_t iters) {
        for (size_t i = 0; i < iters; ++i) {
          folly::doNotOptimizeAway(flag.check());
        }
      },
      kNumThreads, // Threads
      kNumIterations); // Iterations per thread
}

#if defined(__x86_64__) && !defined(TSAN_BUILD)

BENCHMARK(one_way_flag_unsafe_read) {
  OneWayStatusFlagUnsafe flag;
  runParallelUpdates(
      [&](size_t iters) {
        for (size_t i = 0; i < iters; ++i) {
          folly::doNotOptimizeAway(flag.check());
        }
      },
      kNumThreads, // Threads
      kNumIterations); // Iterations per thread
}

#endif

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
}
