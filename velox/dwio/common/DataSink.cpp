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

#include "velox/dwio/common/DataSink.h"
#include "velox/common/file/FileSystems.h"
#include "velox/core/Context.h"
#include "velox/dwio/common/exception/Exception.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace facebook::velox::dwio::common {

FileSink::FileSink(
    const std::string& name,
    const MetricsLogPtr& metricLogger,
    IoStatistics* stats)
    : DataSink{name, metricLogger, stats} {
  file_ = open(name_.c_str(), O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
  if (file_ == -1) {
    markClosed();
    DWIO_RAISE("Can't open ", name_, " ErrorNo ", errno, ": ", strerror(errno));
  }
}

void FileSink::write(std::vector<DataBuffer<char>>& buffers) {
  writeImpl(buffers, [&](auto& buffer) {
    size_t size = buffer.size();
    size_t offset = 0;
    while (offset < size) {
      // Write system call can write fewer bytes than requested.
      auto bytesWritten = ::write(file_, buffer.data() + offset, size - offset);

      // errno should only be accessed when the return value is -1.
      DWIO_ENSURE_NE(
          bytesWritten,
          -1,
          "Bad write of ",
          name_,
          " ErrorNo ",
          errno,
          " Remaining ",
          size - offset);

      // ensure the file is making some forward progress in each loop.
      DWIO_ENSURE_GT(
          bytesWritten,
          0,
          "No bytes transferred ",
          name_,
          " Size: ",
          size,
          " Offset: ",
          offset);

      offset += bytesWritten;
    }
    return size;
  });
}

static std::vector<DataSink::Factory>& factories() {
  static std::vector<DataSink::Factory> factories;
  return factories;
}

// static
bool DataSink::registerFactory(const DataSink::Factory& factory) {
  factories().push_back(factory);
  return true;
}

std::unique_ptr<DataSink> DataSink::create(
    const std::string& path,
    const MetricsLogPtr& metricsLog,
    IoStatistics* stats) {
  DWIO_ENSURE_NOT_NULL(metricsLog.get());
  for (auto& factory : factories()) {
    auto result = factory(path, metricsLog, stats);
    if (result) {
      return result;
    }
  }
  return std::make_unique<FileSink>(path, metricsLog, stats);
}

static std::unique_ptr<DataSink> fileSink(
    const std::string& filename,
    const MetricsLogPtr& metricsLog,
    IoStatistics* stats = nullptr) {
  if (strncmp(filename.c_str(), "file:", 5) == 0) {
    return std::make_unique<FileSink>(filename.substr(5), metricsLog, stats);
  }
  return nullptr;
}

VELOX_REGISTER_DATA_SINK_METHOD_DEFINITION(FileSink, fileSink);

WriteFileSink::WriteFileSink(
    std::unique_ptr<WriteFile> writeFile,
    const MetricsLogPtr& metricLogger,
    IoStatistics* stats)
    : DataSink{"WriteFileSink", metricLogger, stats} {
  if (writeFile == nullptr) {
    markClosed();
    DWIO_RAISE("Can't open Write file is NULL.");
  }
  writeFile_ = std::move(writeFile);
}

void WriteFileSink::write(std::vector<DataBuffer<char>>& buffers) {
  int cnt = 0;
  uint64_t writeSizePerFlush = 0;
  // Write the data buffer to hdfs.
  writeImpl(buffers, [&](auto& buffer) {
    size_t size = buffer.size();
    std::string data(buffer.data(), size);
    writeFile_->append(data);
    cnt++;
    writeSizePerFlush += size;
    // Flush the data if it's last buffer or write size greater than configured
    // flush size.
    if (cnt == buffers.size() || writeSizePerFlush >= flushSize_) {
      writeFile_->flush();
      writeSizePerFlush = 0;
    }
    return size;
  });
}

void WriteFileSink::setFlushSize(uint64_t flushSize) {
  flushSize_ = flushSize;
}

std::unique_ptr<DataSink> WriteFileSink::createWriteFileSink(
    const std::string& path_,
    const std::unordered_map<std::string, std::string>* props,
    const MetricsLogPtr& metricsLog,
    IoStatistics* stats) {
  std::shared_ptr<const facebook::velox::core::MemConfig> config = nullptr;
  // Create the write file sink, if props is nullptr, will read the default
  // config of hdfs-site.xml in LIBHDFS3_CONF environment variable, by set the
  // 'hive.hdfs.host' as 'default'
  if (props == nullptr) {
    static const std::unordered_map<std::string, std::string> configMap(
        {{"hive.hdfs.host", "default"}, {"hive.hdfs.port", "0"}});
    config =
        std::make_shared<const facebook::velox::core::MemConfig>(configMap);
  } else {
    config = std::make_shared<const facebook::velox::core::MemConfig>(*props);
  }
  std::shared_ptr<facebook::velox::filesystems::FileSystem> fs =
      filesystems::getFileSystem(path_, config);
  VELOX_CHECK(fs != nullptr, "Failed to connector file system");
  std::string filePath_ = path_;
  // Remove hdfs prefix from path. If the file path like: 'hdfs://a/b/c',  then
  // we will remove the hdfs prefix, make the path to be '/a/b/c'
  if (strncmp(filePath_.c_str(), "hdfs:", 5) == 0) {
    std::string hdfsPrefix("hdfs://");
    filePath_ = filePath_.replace(0, hdfsPrefix.size(), "");
    int pathBeginIndex = filePath_.find("/", 0);
    filePath_ = filePath_.substr(pathBeginIndex);
  }
  std::unique_ptr<WriteFile> writeFile = fs->openFileForWrite(filePath_);
  VELOX_CHECK(writeFile != nullptr, "Failed to open write file");
  return std::make_unique<WriteFileSink>(
      std::move(writeFile), metricsLog, stats);
}

static std::unique_ptr<DataSink> writeFileSink(
    const std::string& file_path_,
    const MetricsLogPtr& metricsLog,
    IoStatistics* stats) {
  return WriteFileSink::createWriteFileSink(
      file_path_, nullptr, metricsLog, stats);
}

VELOX_REGISTER_DATA_SINK_METHOD_DEFINITION(WriteFileSink, writeFileSink);

} // namespace facebook::velox::dwio::common
