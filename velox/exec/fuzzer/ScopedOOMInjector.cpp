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
#include "velox/exec/fuzzer/ScopedOOMInjector.h"

#include <glog/logging.h>
#include <chrono>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/testutil/TestValue.h"

namespace facebook::velox::exec::test {
namespace {
const std::string kInjectionPoint =
    "facebook::velox::memory::MemoryPoolImpl::reserveThreadSafe";

size_t now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
} // namespace

ScopedOOMInjector::ScopedOOMInjector(
    const std::function<bool()>& oomConditionChecker,
    uint64_t oomCheckIntervalMs)
    : oomConditionChecker_(oomConditionChecker),
      oomCheckIntervalMs_(oomCheckIntervalMs) {}

ScopedOOMInjector::~ScopedOOMInjector() {
  common::testutil::TestValue::clear(kInjectionPoint);
  enabled_ = false;
}

void ScopedOOMInjector::enable() {
#ifdef NDEBUG
  VELOX_FAIL("OOM injection can only be used in debug builds");
#endif

  if (enabled_.exchange(true)) {
    VELOX_FAIL("Already enabled");
  }

  common::testutil::TestValue::enable();

  common::testutil::TestValue::set(
      kInjectionPoint,
      std::function<void(memory::MemoryPool*)>([&](memory::MemoryPool*) {
        const auto currentTime = now();
        if (currentTime - lastOomCheckTime_ >= oomCheckIntervalMs_) {
          lastOomCheckTime_ = currentTime;
          if (oomConditionChecker_()) {
            LOG(INFO) << "<-- Triggering OOM --";
            VELOX_MEM_POOL_CAP_EXCEEDED(kErrorMessage);
          }
        }
      }));
}

} // namespace facebook::velox::exec::test
