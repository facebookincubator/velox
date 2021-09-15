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

#include <mutex>
#include <string>
#include <unordered_map>

#include <pthread.h>
#include <sys/types.h>
#include <string>
#include <vector>

namespace facebook {
namespace velox {
namespace process {

/**
 * Current executable's name.
 */
std::string getAppName();

/**
 * This machine'a name.
 */
std::string getHostName();

/**
 * Process identifier.
 */
pid_t getProcessId();

/**
 * Current thread's identifier.
 */
pthread_t getThreadId();

/**
 * Get current working directory.
 */
std::string getCurrentDirectory();

/**
 * Returns elapsed CPU nanoseconds on the calling thread
 */
uint64_t threadCpuNanos();

// True if the machine has Intel AVX2 instructions and these are not disabled by
// flag.
bool hasAvx2();

// True if the machine has Intel BMI2 instructions and these are not disabled by
// flag.
bool hasBmi2();

struct TraceData {
  int32_t numThreads{0};
  int32_t numEnters{0};
  uint64_t totalMs{0};
  uint64_t maxMs{0};
};

class Context {
 public:
  Context(const std::string& label);

  ~Context();

  static std::string statusLine();

 private:
  std::string label_;
  std::chrono::steady_clock::time_point enterTime_;

  static std::mutex mutex_;
  static std::unordered_map<std::string, TraceData> counts_;
};

} // namespace process
} // namespace velox
} // namespace facebook
