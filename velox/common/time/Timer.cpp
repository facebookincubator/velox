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

#include "velox/common/time/Timer.h"

#include <sys/resource.h>

#include "velox/common/testutil/ScopedTestTime.h"
#include "velox/common/time/CpuWallTimer.h"

namespace facebook::velox {

using namespace std::chrono;
using common::testutil::ScopedTestTime;

ProcessCpuWallTimer::ProcessCpuWallTimer(CpuWallTiming& timing)
    : wallTimeStart_{steady_clock::now()},
      cpuTimeStart_{processCpuNanos()},
      timing_{timing} {
  ++timing_.count;
}

ProcessCpuWallTimer::~ProcessCpuWallTimer() {
  const auto cpuTimeEnd = processCpuNanos();
  if (cpuTimeStart_ != kUnavailableCpuTime &&
      cpuTimeEnd != kUnavailableCpuTime) {
    timing_.cpuNanos += cpuTimeEnd - cpuTimeStart_;
  }
  timing_.wallNanos +=
      duration_cast<nanoseconds>(steady_clock::now() - wallTimeStart_).count();
}

uint64_t ProcessCpuWallTimer::processCpuNanos() noexcept {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return kUnavailableCpuTime;
  }
  const auto toNanos = [](const timeval& value) {
    return static_cast<uint64_t>(value.tv_sec) * 1'000'000'000 +
        static_cast<uint64_t>(value.tv_usec) * 1'000;
  };
  return toNanos(usage.ru_utime) + toNanos(usage.ru_stime);
}

uint64_t EpochTime::nowNanos() {
  const auto systemNanos =
      duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
          .count();
#ifndef NDEBUG
  return ScopedTestTime::getCurrentTestTimeNano().value_or(systemNanos);
#else
  return systemNanos;
#endif
}

} // namespace facebook::velox
