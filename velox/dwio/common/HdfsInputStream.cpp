/*
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

#include <fcntl.h>
#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <hdfs/FileStatus.h>

#include "velox/dwio/common/InputStream.h"

using namespace facebook::dwio::common;

namespace facebook {
namespace dwio {
namespace common {

HdfsInputStream::HdfsInputStream(
    const std::string& path,
    const MetricsLogPtr& metricsLog,
    common::IoStatistics* stats)
    : InputStream(path, metricsLog, stats) {
  //todo: pass nanme node ip and port from config file
  std::string host ("10.188.183.76");
  int port = 65212;

  // create hdfs instance
  struct hdfsBuilder *builder = hdfsNewBuilder();
  hdfsBuilderConfSetStr(builder, "input.read.timeout", "60000"); // 1 min
  hdfsBuilderConfSetStr(builder, "input.connect.timeout", "60000"); // 1 min
  hdfsBuilderSetNameNode(builder, host.c_str());
  hdfsBuilderSetNameNodePort(builder, port);
  fs = hdfsBuilderConnect(builder);
  if (fs == nullptr) {
    throw std::runtime_error("Unable to connect to HDFS, get error: " + std::string(hdfsGetLastError()));
  }
  file = hdfsOpenFile(fs, path.c_str(), O_RDONLY, 0, 0, 0);
  if (file == nullptr) {
    throw std::runtime_error("Unable to open file" + path + ".get error: " + std::string(hdfsGetLastError()));
  }
}

void HdfsInputStream::read(
    void* buf,
    uint64_t length,
    uint64_t offset,
    MetricsLog::MetricsType purpose) {
  if (!buf) {
    throw std::invalid_argument("Buffer is null");
  }

  // log the metric
  logRead(offset, length, purpose);

  auto dest = static_cast<char*>(buf);
  if(offset > 0) {
    hdfsSeek(fs, file, offset);
  }
  uint64_t totalBytesRead = 0;
  while (totalBytesRead < length) {
    auto bytesRead = hdfsRead(fs, file, dest, length - totalBytesRead);
    DWIO_ENSURE_GE(
        bytesRead,
        0,
        "pread failure . File name: ",
        path_,
        ", offset: ",
        offset,
        ", length: ",
        length,
        ", read: ",
        totalBytesRead,
        ", errno: ",
        errno)

    DWIO_ENSURE_NE(
        bytesRead,
        0,
        "Unexepected EOF. File name: ",
        path_,
        ", offset: ",
        offset,
        ", length: ",
        length,
        ", read: ",
        totalBytesRead)

    totalBytesRead += bytesRead;
    dest += bytesRead;
  }

  DWIO_ENSURE_EQ(
      totalBytesRead,
      length,
      "Should read exactly as requested. File name: ",
      path_,
      ", offset: ",
      offset,
      ", length: ",
      length,
      ", read: ",
      totalBytesRead)

  if (stats_) {
    stats_->incRawBytesRead(length);
  }
}

HdfsInputStream::~HdfsInputStream() {
  hdfsCloseFile(fs, file);
}

uint64_t HdfsInputStream::getLength() const {
  return hdfsAvailable(fs, file);
}

uint64_t HdfsInputStream::getNaturalReadSize() const {
  return hdfsAvailable(fs, file);;
}
void HdfsInputStream::logRead(uint64_t offset, uint64_t length, LogType purpose) {
  metricsLog_->logRead(
      0, "readFully", getLength(), 0, 0, offset, length, purpose, 1, 0);
}

} // namespace common
} // namespace dwio
} // namespace facebook
