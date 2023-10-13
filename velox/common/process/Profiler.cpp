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
#include "velox/common/file/File.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

namespace facebook::velox::process {

bool Profiler::profileStarted_;
std::thread Profiler::profileThread_;
std::mutex Profiler::profileMutex_;
std::shared_ptr<velox::filesystems::FileSystem> Profiler::fileSystem_;
bool Profiler::isSleeping_;
bool Profiler::shouldStop_;
folly::Promise<bool> Profiler::sleepPromise_;

void Profiler::copyToResult(int32_t counter, const std::string& path) {
  int32_t fd = open("/tmp/perf", O_RDONLY);
  if (fd < 0) {
    return;
  }
  auto bufferSize = 400000;
  char* buffer = reinterpret_cast<char*>(malloc(bufferSize));
  auto readSize = ::read(fd, buffer, bufferSize);
  close(fd);
  auto target = fmt::format("{}/prof-{}", path, counter);
  try {
    try {
      fileSystem_->remove(target);
    } catch (const std::exception& e) {
      // ignore
    }
    auto out = fileSystem_->openFileForWrite(target);
    out->append(std::string_view(buffer, readSize));
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error opening/writing " << target << ":" << e.what();
  }
  ::free(buffer);
}

void Profiler::makeProfileDir(std::string path) {
  try {
    fileSystem_->mkdir(path);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to create directory " << path << ":" << e.what();
  }
}

void Profiler::threadFunction(std::string path) {
  const int32_t pid = getpid();
  makeProfileDir(path);
  for (int32_t counter = 0;; ++counter) {
    std::thread systemThread([&]() {
      system(
          fmt::format(
              "cd /tmp; perf record --pid {};"
              "perf report --sort symbol > /tmp/perf;"
              "sed --in-place 's/      / /' /tmp/perf; sed --in-place 's/      / /' /tmp/perf; ",
              pid)
              .c_str());
      copyToResult(counter, path);
    });
    folly::SemiFuture<bool> sleepFuture(false);
    {
      std::lock_guard<std::mutex> l(profileMutex_);
      isSleeping_ = true;
      sleepPromise_ = folly::Promise<bool>();
      sleepFuture = sleepPromise_.getSemiFuture();
    }
    if (!shouldStop_) {
      try {
        auto& executor = folly::QueuedImmediateExecutor::instance();
        std::move(sleepFuture)
            .via(&executor)
            .wait((std::chrono::seconds(counter < 2 ? 60 : 300)));
      } catch (std::exception& e) {
      }
    }
    {
      std::lock_guard<std::mutex> l(profileMutex_);
      isSleeping_ = false;
    }
    system("killall -2 perf");
    systemThread.join();
    if (shouldStop_) {
      return;
    }
  }
}

bool Profiler::isRunning() {
  std::lock_guard<std::mutex> l(profileMutex_);
  return profileStarted_;
}

void Profiler::start(const std::string& path) {
  {
#if !defined(linux)
    VELOX_FAIL("Profiler is only available for Linux");
#endif
    std::lock_guard<std::mutex> l(profileMutex_);
    if (profileStarted_) {
      return;
    }
    profileStarted_ = true;
  }
  fileSystem_ = velox::filesystems::getFileSystem(path, nullptr);
  if (!fileSystem_) {
    LOG(ERROR) << "Failed to find file system for " << path
               << ". Profiler not started.";
    return;
  }
  makeProfileDir(path);
  atexit(Profiler::stop);
  profileThread_ = std::thread([path]() { threadFunction(path); });
}

void Profiler::stop() {
  {
    std::lock_guard<std::mutex> l(profileMutex_);
    shouldStop_ = true;
    if (isSleeping_) {
      sleepPromise_.setValue(true);
    }
  }
  profileThread_.join();
  {
    std::lock_guard<std::mutex> l(profileMutex_);
    profileStarted_ = false;
  }
  LOG(INFO) << "Stopped profiling";
}

} // namespace facebook::velox::process
