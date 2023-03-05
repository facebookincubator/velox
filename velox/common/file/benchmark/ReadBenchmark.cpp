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

#include "velox/common/file/benchmark/ReadBenchmark.h"
#include <fstream>
#include <iostream>

#include <fcntl.h>

#include <folly/Random.h>
#include <folly/Synchronized.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/futures/Future.h>

#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/time/Timer.h"
#include "velox/core/Context.h"

DEFINE_string(path, "", "Path of test file");
DEFINE_int64(
    file_size_gb,
    0,
    "Limits the test to the first --file_size_gb "
    "of --path. 0 means use the whole file");
DEFINE_int32(num_threads, 16, "Test parallelism");
DEFINE_int32(seed, 0, "Random seed, 0 means no seed");
DEFINE_bool(odirect, false, "Use O_DIRECT");

DEFINE_int32(
    request_bytes,
    0,
    "If 0, runs through a set of predefined read patterns. "
    "If non-0, this is the size of a single read request. These requests are "
    "read in --num_in_run consecutive batches with --gap bytes between each request."
    "A single file read call resolves the above requests.");
DEFINE_int32(
    gap,
    0,
    "Gap between consecutive reads if --request_bytes is non-0");
DEFINE_int32(
    num_in_run,
    10,
    "Number of consecutive requests of --request_bytes separated by --gap bytes");
DEFINE_int32(
    measurement_size,
    100 << 20,
    "Total reads per thread throughput for a --request_bytes/--gap/--num_in_run combination");
DEFINE_string(config, "", "Path of the file-system config file if any");

namespace {
static bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}
} // namespace

DEFINE_validator(path, &notEmpty);

namespace facebook::velox {

class ReadBenchmark::Impl {
 private:
  enum class Mode { Pread = 0, Preadv = 1, MultiplePread = 2 };

  // Struct to read data into. If we read contiguous and then copy to
  // non-contiguous buffers, we read to 'buffer' and copy to
  // 'bufferCopy'.
  struct Scratch {
    std::string buffer;
    std::string bufferCopy;
  };

  // Read the specified config file.
  void readConfig() {
    if (!FLAGS_config.empty()) {
      std::ifstream configFile(FLAGS_config);
      if (!configFile.is_open()) {
        throw std::runtime_error(fmt::format(
            "Couldn't open config file {} for reading.", FLAGS_config));
      }

      std::unordered_map<std::string, std::string> properties;
      std::string line;
      while (getline(configFile, line)) {
        line.erase(
            std::remove_if(line.begin(), line.end(), isspace), line.end());
        if (line[0] == '#' || line.empty()) {
          continue;
        }
        auto delimiterPos = line.find('=');
        auto name = line.substr(0, delimiterPos);
        auto value = line.substr(delimiterPos + 1);
        properties.emplace(name, value);
      }

      config_ = std::make_shared<facebook::velox::core::MemConfig>(properties);
    }
  }

  void clearCache() {
#ifdef linux
    // system("echo 3 >/proc/sys/vm/drop_caches");
    bool success = false;
    auto fd = open("/proc//sys/vm/drop_caches", O_WRONLY);
    if (fd > 0) {
      success = write(fd, "3", 1) == 1;
      close(fd);
    }
    if (!success) {
      LOG(ERROR) << "Failed to clear OS disk cache: errno=" << errno;
    }
#endif
  }

  Scratch& getScratch(int32_t size) {
    auto scratch = scratch_.withWLock([&](auto& table) {
      auto& ptr = table[std::this_thread::get_id()];
      if (!ptr) {
        ptr = std::make_unique<Scratch>();
      }
      ptr->buffer.resize(size);
      ptr->bufferCopy.resize(size);
      return ptr.get();
    });
    return *scratch;
  }

  // Measures the throughput for various ReadFile APIs(modes).
  void randomReads(
      int32_t requestSize,
      int32_t gap,
      int32_t numInRun,
      int32_t repeats,
      Mode mode,
      bool parallel) {
    clearCache();
    std::vector<folly::Promise<bool>> promises;
    std::vector<folly::SemiFuture<bool>> futures;
    uint64_t usec = 0;
    std::string label;
    {
      MicrosecondTimer timer(&usec);
      // Size to read from a single read call.
      int32_t rangeReadSize = requestSize * numInRun + gap * (numInRun - 1);
      auto& globalScratch = getScratch(rangeReadSize);
      globalScratch.buffer.resize(rangeReadSize);
      globalScratch.bufferCopy.resize(rangeReadSize);
      for (auto repeat = 0; repeat < repeats; ++repeat) {
        std::unique_ptr<folly::Promise<bool>> promise;
        if (parallel) {
          auto [tempPromise, future] = folly::makePromiseContract<bool>();
          promise = std::make_unique<folly::Promise<bool>>();
          *promise = std::move(tempPromise);
          futures.push_back(std::move(future));
        }
        int64_t offset =
            folly::Random::rand64(rng_) % (fileSize_ - rangeReadSize);
        switch (mode) {
          case Mode::Pread:
            label = "1 pread";
            if (parallel) {
              executor_->add([offset,
                              gap,
                              requestSize,
                              numInRun,
                              rangeReadSize,
                              this,
                              capturedPromise = std::move(promise)]() {
                auto& scratch = getScratch(rangeReadSize);
                readFile_->pread(offset, rangeReadSize, scratch.buffer.data());
                for (auto i = 0; i < numInRun; ++i) {
                  memcpy(
                      scratch.bufferCopy.data() + i * requestSize,
                      scratch.buffer.data() + i * (requestSize + gap),
                      requestSize);
                }
                capturedPromise->setValue(true);
              }

              );
            } else {
              readFile_->pread(
                  offset, rangeReadSize, globalScratch.buffer.data());
              for (auto i = 0; i < numInRun; ++i) {
                memcpy(
                    globalScratch.bufferCopy.data() + i * requestSize,
                    globalScratch.buffer.data() + i * (requestSize + gap),
                    requestSize);
              }
            }
            break;
          case Mode::Preadv: {
            label = "1 preadv";
            if (parallel) {
              executor_->add([offset,
                              gap,
                              requestSize,
                              rangeReadSize,
                              this,
                              capturedPromise = std::move(promise)]() {
                auto& scratch = getScratch(rangeReadSize);
                std::vector<folly::Range<char*>> ranges;
                for (auto start = 0; start < rangeReadSize;
                     start += requestSize + gap) {
                  ranges.push_back(folly::Range<char*>(
                      scratch.buffer.data() + start, requestSize));
                  if (gap && start + gap < rangeReadSize) {
                    ranges.push_back(folly::Range<char*>(nullptr, gap));
                  }
                }
                readFile_->preadv(offset, ranges);
                capturedPromise->setValue(true);
              });
            } else {
              std::vector<folly::Range<char*>> ranges;
              for (auto start = 0; start < rangeReadSize;
                   start += requestSize + gap) {
                ranges.push_back(folly::Range<char*>(
                    globalScratch.buffer.data() + start, requestSize));
                if (gap && start + gap < rangeReadSize) {
                  ranges.push_back(folly::Range<char*>(nullptr, gap));
                }
              }
              readFile_->preadv(offset, ranges);
            }

            break;
          }
          case Mode::MultiplePread: {
            label = "multiple pread";
            if (parallel) {
              executor_->add([offset,
                              gap,
                              requestSize,
                              numInRun,
                              rangeReadSize,
                              this,
                              capturedPromise = std::move(promise)]() {
                auto& scratch = getScratch(rangeReadSize);
                for (auto counter = 0; counter < numInRun; ++counter) {
                  readFile_->pread(
                      offset + counter * (requestSize + gap),
                      requestSize,
                      scratch.buffer.data() + counter * requestSize);
                }
                capturedPromise->setValue(true);
              });
            } else {
              for (auto counter = 0; counter < numInRun; ++counter) {
                readFile_->pread(
                    offset + counter * (requestSize + gap),
                    requestSize,
                    globalScratch.buffer.data() + counter * requestSize);
              }
            }
            break;
          }
        }
      }
      if (parallel) {
        auto& exec = folly::QueuedImmediateExecutor::instance();
        for (int32_t i = futures.size() - 1; i >= 0; --i) {
          std::move(futures[i]).via(&exec).wait();
        }
      }
    }
    out_ << fmt::format(
                "{} MB/s {}{}",
                (static_cast<float>(numInRun) * requestSize * repeats) / usec,
                label,
                parallel ? " mt" : "")
         << std::endl;
  }

 public:
  void initialize() {
    executor_ =
        std::make_unique<folly::IOThreadPoolExecutor>(FLAGS_num_threads);
    if (FLAGS_odirect) {
      int32_t o_direct =
#ifdef linux
          O_DIRECT;
#else
          0;
#endif
      fd_ = open(
          FLAGS_path.c_str(),
          O_CREAT | O_RDWR | (FLAGS_odirect ? o_direct : 0),
          S_IRUSR | S_IWUSR);
      if (fd_ < 0) {
        LOG(ERROR) << "Could not open " << FLAGS_path;
        exit(1);
      }
      readFile_ = std::make_unique<LocalReadFile>(fd_);

    } else {
      filesystems::registerLocalFileSystem();
      readConfig();
      auto lfs = filesystems::getFileSystem(FLAGS_path, config_);
      readFile_ = lfs->openFileForRead(FLAGS_path);
    }
    fileSize_ = readFile_->size();
    if (FLAGS_file_size_gb) {
      fileSize_ = std::min<uint64_t>(FLAGS_file_size_gb << 30, fileSize_);
    }

    if (fileSize_ <= FLAGS_measurement_size) {
      LOG(ERROR) << "File size " << fileSize_
                 << " is <= then --measurement_size " << FLAGS_measurement_size;
      exit(1);
    }
    if (FLAGS_seed) {
      rng_.seed(FLAGS_seed);
    }
  }

  void modes(int32_t requestSize, int32_t gap, int32_t numInRun) {
    int repeats = std::max<int32_t>(
        3, (FLAGS_measurement_size) / (requestSize * numInRun));
    out_ << fmt::format(
                "Request Size: {} Gap: {} Num in Run: {} Repeats: {}",
                requestSize,
                gap,
                numInRun,
                repeats)
         << std::endl;
    randomReads(requestSize, gap, numInRun, repeats, Mode::Pread, false);
    randomReads(requestSize, gap, numInRun, repeats, Mode::Preadv, false);
    randomReads(
        requestSize, gap, numInRun, repeats, Mode::MultiplePread, false);
    randomReads(requestSize, gap, numInRun, repeats, Mode::Pread, true);
    randomReads(requestSize, gap, numInRun, repeats, Mode::Preadv, true);
    randomReads(requestSize, gap, numInRun, repeats, Mode::MultiplePread, true);
  }

  Impl(std::ostream& out) : out_(out) {}

 private:
  static constexpr int64_t kRegionSize = 64 << 20; // 64MB
  static constexpr int32_t kWrite = -10000;
  // 0 means no op, kWrite means being written, other numbers are reader counts.
  std::string writeBatch_;
  int32_t fd_;
  std::unique_ptr<folly::IOThreadPoolExecutor> executor_;
  std::unique_ptr<ReadFile> readFile_;
  folly::Random::DefaultGenerator rng_;
  int64_t fileSize_;
  std::shared_ptr<Config> config_;
  std::ostream& out_;

  folly::Synchronized<
      std::unordered_map<std::thread::id, std::unique_ptr<Scratch>>>
      scratch_;
};

ReadBenchmark::ReadBenchmark(std::ostream& out) {
  impl_ = std::make_shared<Impl>(out);
}

void ReadBenchmark::initialize() {
  impl_->initialize();
}

void ReadBenchmark::run() {
  if (FLAGS_request_bytes) {
    impl_->modes(FLAGS_request_bytes, FLAGS_gap, FLAGS_num_in_run);
    return;
  }
  impl_->modes(1100, 0, 10);
  impl_->modes(1100, 1200, 10);
  impl_->modes(16 * 1024, 0, 10);
  impl_->modes(16 * 1024, 10000, 10);
  impl_->modes(1000000, 0, 8);
  impl_->modes(1000000, 100000, 8);
}

} // namespace facebook::velox
