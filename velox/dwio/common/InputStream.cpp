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

#include "velox/dwio/common/InputStream.h"

#include <fcntl.h>
#include <folly/container/F14Map.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <functional>
#include <istream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "velox/common/time/Timer.h"
#include "velox/dwio/common/exception/Exception.h"

using ::facebook::velox::common::Region;

namespace facebook::velox::dwio::common {

namespace {
int64_t totalBufferSize(const std::vector<folly::Range<char*>>& buffers) {
  int64_t bufferSize = 0;
  for (auto& buffer : buffers) {
    bufferSize += buffer.size();
  }
  return bufferSize;
}

class MeasureTimeCloser {
 public:
  MeasureTimeCloser(IoStatistics* stats, uint64_t length)
      : stats_{stats}, length_{length}, startTime_{getCurrentTimeMicro()} {}

  ~MeasureTimeCloser() {
    if (!stats_) {
      return;
    }

    stats_->incRawBytesRead(length_);
    stats_->incTotalScanTime((getCurrentTimeMicro() - startTime_) * 1000);
  }

 private:
  IoStatistics* stats_;
  uint64_t length_;
  size_t startTime_;
};
} // namespace

folly::SemiFuture<uint64_t> InputStream::readAsync(
    const std::vector<folly::Range<char*>>& buffers,
    uint64_t offset,
    LogType logType) {
  try {
    read(buffers, offset, logType);
    uint64_t size = 0;
    for (auto& range : buffers) {
      size += range.size();
    }
    return folly::SemiFuture<uint64_t>(size);
  } catch (const std::exception& e) {
    return folly::makeSemiFuture<uint64_t>(e);
  }
}

void InputStream::logRead(uint64_t offset, uint64_t length, LogType purpose)
    const {
  metricsLog_->logRead(
      0, "readFully", getLength(), 0, 0, offset, length, purpose, 1, 0);
}

ReadFileInputStream::ReadFileInputStream(
    std::shared_ptr<velox::ReadFile> readFile,
    const MetricsLogPtr& metricsLog,
    IoStatistics* stats)
    : InputStream(readFile->getName(), metricsLog, stats),
      readFile_(std::move(readFile)) {}

void ReadFileInputStream::read(
    void* buf,
    uint64_t length,
    uint64_t offset,
    MetricsLog::MetricsType purpose) {
  if (!buf) {
    throw std::invalid_argument("Buffer is null");
  }
  logRead(offset, length, purpose);
  std::string_view data_read;
  {
    const MeasureTimeCloser closer(getStats(), length);
    data_read = readFile_->pread(offset, length, buf);
  }

  DWIO_ENSURE_EQ(
      data_read.size(),
      length,
      "Should read exactly as requested. File name: ",
      getName(),
      ", offset: ",
      offset,
      ", length: ",
      length,
      ", read: ",
      data_read.size());
}

void ReadFileInputStream::read(
    const std::vector<folly::Range<char*>>& buffers,
    uint64_t offset,
    LogType logType) {
  const int64_t bufferSize = totalBufferSize(buffers);
  logRead(offset, bufferSize, logType);
  auto size = readFile_->preadv(offset, buffers);
  DWIO_ENSURE_EQ(
      size,
      bufferSize,
      "Should read exactly as requested. File name: ",
      getName(),
      ", offset: ",
      offset,
      ", length: ",
      bufferSize,
      ", read: ",
      size);
}

folly::SemiFuture<uint64_t> ReadFileInputStream::readAsync(
    const std::vector<folly::Range<char*>>& buffers,
    uint64_t offset,
    LogType logType) {
  const int64_t bufferSize = totalBufferSize(buffers);
  logRead(offset, bufferSize, logType);
  return readFile_->preadvAsync(offset, buffers);
}

bool ReadFileInputStream::hasReadAsync() const {
  return readFile_->hasPreadvAsync();
}

void ReadFileInputStream::vread(
    folly::Range<const velox::common::Region*> regions,
    folly::Range<folly::IOBuf*> iobufs,
    const LogType purpose) {
  DWIO_ENSURE_GT(regions.size(), 0, "regions to read can't be empty");
  const size_t length = std::accumulate(
      regions.cbegin(),
      regions.cend(),
      size_t(0),
      [&](size_t acc, const auto& r) { return acc + r.length; });
  logRead(regions[0].offset, length, purpose);
  {
    const MeasureTimeCloser closer(getStats(), length);
    readFile_->preadv(regions, iobufs);
  }
}

} // namespace facebook::velox::dwio::common
