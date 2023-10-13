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

#include <folly/futures/Future.h>
#include <folly/futures/Promise.h>
#include <string>
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::process {

class Profiler {
 public:
  /// Starts periodic production of perf reports.
  static void start(const std::string& path);

  // Stops profiling background associated threads. Threads are stopped on
  // return.
  static void stop();

  static bool isRunning();

 private:
  static void copyToResult(int32_t counter, const std::string& path);
  static void makeProfileDir(std::string path);
  static void threadFunction(std::string path);

  static bool profileStarted_;
  static std::thread profileThread_;
  static std::mutex profileMutex_;
  static std::shared_ptr<velox::filesystems::FileSystem> fileSystem_;
  static bool isSleeping_;
  static bool shouldStop_;
  static folly::Promise<bool> sleepPromise_;
};

} // namespace facebook::velox::process
