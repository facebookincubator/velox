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

#include "velox/common/process/Profiler.h"
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <thread>
#include "velox/common/process/TraceContext.h"

using namespace facebook::velox::process;
using namespace facebook::velox;

namespace {
int32_t fi(int32_t x) {
  return x < 2 ? x : fi(x - 1) + fi(x - 2);
}
} // namespace

TEST(ProfilerTest, basic) {
  constexpr int32_t kNumThreads = 10;
#if !defined(linux)
  return;
#endif
  filesystems::registerLocalFileSystem();
  Profiler::start("profilertest");
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  std::atomic<int32_t> sum = 0;
  for (int32_t i = 0; i < kNumThreads; ++i) {
    threads.push_back(std::thread([&]() {
      sum += fi(40);
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  for (auto& thread : threads) {
    thread.join();
  }
  LOG(INFO) << "Sum " << sum;
  // The test exits during the measurement interval. We expect no
  // crash on exit if the threads are properly joined.
}
