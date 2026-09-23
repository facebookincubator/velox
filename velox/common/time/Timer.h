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
#pragma once

#include <folly/CppAttributes.h>
#include <folly/chrono/Hardware.h>
#include <atomic>
#include <chrono>
#include <limits>
#include <optional>

#include "velox/common/process/ProcessBase.h"

namespace facebook::velox {

struct CpuWallTiming;

/// Measures wall time (steady_clock) between construction and destruction,
/// incrementing a user-supplied counter in microseconds.
class MicrosecondWallTimer {
 public:
  explicit MicrosecondWallTimer(uint64_t* timer) : timer_(timer) {
    start_ = std::chrono::steady_clock::now();
  }

  ~MicrosecondWallTimer() {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_);

    (*timer_) += duration.count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
  uint64_t* timer_;
};

/// Measures wall time (steady_clock) between construction and destruction,
/// incrementing a user-supplied counter in nanoseconds.
class NanosecondWallTimer {
 public:
  explicit NanosecondWallTimer(uint64_t* timer) : timer_(timer) {
    start_ = std::chrono::steady_clock::now();
  }

  ~NanosecondWallTimer() {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start_);

    (*timer_) += duration.count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
  uint64_t* timer_;
};

/// Measures per-thread CPU time (CLOCK_THREAD_CPUTIME_ID) between construction
/// and destruction, incrementing a user-supplied counter in microseconds.
/// Excludes time spent sleeping or blocked on I/O.
class MicrosecondCPUTimer {
 public:
  explicit MicrosecondCPUTimer(uint64_t* timer)
      : timer_(timer), start_(process::threadCpuNanos()) {}

  ~MicrosecondCPUTimer() {
    (*timer_) += (process::threadCpuNanos() - start_) / 1'000;
  }

 private:
  uint64_t* timer_;
  uint64_t start_;
};

/// Measures per-thread CPU time (CLOCK_THREAD_CPUTIME_ID) between construction
/// and destruction, incrementing a user-supplied counter in nanoseconds.
/// Excludes time spent sleeping or blocked on I/O.
class NanosecondCPUTimer {
 public:
  explicit NanosecondCPUTimer(uint64_t* timer)
      : timer_(timer), start_(process::threadCpuNanos()) {}

  ~NanosecondCPUTimer() {
    (*timer_) += process::threadCpuNanos() - start_;
  }

 private:
  uint64_t* timer_;
  uint64_t start_;
};

/// Adds process-wide CPU and wall time to a CpuWallTiming. Unlike
/// CpuWallTimer, which measures CPU consumed by the calling thread, this timer
/// includes user and system CPU consumed by all process threads. CPU time can
/// exceed wall time when background thread pools run work concurrently.
class ProcessCpuWallTimer {
 public:
  explicit ProcessCpuWallTimer(CpuWallTiming& timing);
  ~ProcessCpuWallTimer();

  ProcessCpuWallTimer(const ProcessCpuWallTimer&) = delete;
  ProcessCpuWallTimer& operator=(const ProcessCpuWallTimer&) = delete;
  ProcessCpuWallTimer(ProcessCpuWallTimer&&) = delete;
  ProcessCpuWallTimer& operator=(ProcessCpuWallTimer&&) = delete;

 private:
  // Returns user and system CPU consumed across all process threads.
  static uint64_t processCpuNanos() noexcept;

  static constexpr uint64_t kUnavailableCpuTime{
      std::numeric_limits<uint64_t>::max()};

  const std::chrono::steady_clock::time_point wallTimeStart_;
  const uint64_t cpuTimeStart_;
  CpuWallTiming& timing_;
};

using MicrosecondTimer = MicrosecondWallTimer;
using NanosecondTimer = NanosecondWallTimer;

/// Measures the time between construction and destruction with CPU clock
/// counter (rdtsc on X86) and increments a user-supplied counter with the cycle
/// count.
class ClockTimer {
 public:
  explicit ClockTimer(uint64_t& total)
      : total_(&total), start_(folly::hardware_timestamp()) {}

  explicit ClockTimer(std::atomic<uint64_t>& total)
      : atomicTotal_(&total), start_(folly::hardware_timestamp()) {}

  ~ClockTimer() {
    auto elapsed = folly::hardware_timestamp() - start_;
    if (total_) {
      *total_ += elapsed;
    } else {
      *atomicTotal_ += elapsed;
    }
  }

 private:
  uint64_t* total_{nullptr};
  std::atomic<uint64_t>* atomicTotal_{nullptr};
  uint64_t start_;
};

/// Returns the current epoch time in seconds.
uint64_t getCurrentTimeSec();

/// Returns the current epoch time in milliseconds.
uint64_t getCurrentTimeMs();

/// Returns the current epoch time in microseconds.
uint64_t getCurrentTimeMicro();

/// Returns the current epoch time in nanoseconds.
uint64_t getCurrentTimeNano();
} // namespace facebook::velox
