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

#include <chrono>

#include "velox/common/file/File.h"
#include "velox/dwio/common/Closeable.h"
#include "velox/dwio/common/DataBuffer.h"
#include "velox/dwio/common/IoStatistics.h"
#include "velox/dwio/common/MetricsLog.h"

namespace facebook::velox::dwio::common {

/**
 * An abstract interface for providing DWIO writer a data sink
 */
class DataSink : public Closeable {
 public:
  explicit DataSink(
      std::string name,
      MetricsLogPtr metricLogger,
      IoStatistics* stats = nullptr)
      : name_{std::move(name)},
        size_{0},
        metricLogger_{std::move(metricLogger)},
        stats_{stats} {}

  ~DataSink() override {
    destroy();
  }

  /**
   * Total number of bytes written.
   */
  virtual uint64_t size() const {
    return size_;
  }

  /**
   * Returns true if data sink supports buffering. In the case when buffering is
   * not supported, caller need to buffer data to yield optimal write size.
   */
  virtual bool isBuffered() const {
    return true;
  }

  /**
   * Write single data buffer.
   */
  void write(DataBuffer<char> buffer) {
    std::vector<DataBuffer<char>> vec;
    vec.push_back(std::move(buffer));
    writeWithLogging(vec);
  }

  /**
   * General write wrapper with logging. All concrete subclasses gets logging
   * for free if they call a public method that goes through this method.
   */
  void writeWithLogging(std::vector<DataBuffer<char>>& buffers) {
    auto start = std::chrono::steady_clock::now();
    write(buffers);
    auto end = std::chrono::steady_clock::now();
    metricLogger_->logWrite(
        size_,
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count());
  }

  /**
   * Write data buffers.
   */
  virtual void write(std::vector<DataBuffer<char>>& buffers) = 0;

  /**
   * Get the name of the data sink for error messages.
   */
  const std::string& getName() const {
    return name_;
  }

  const MetricsLogPtr& getMetricsLog() const {
    return metricLogger_;
  }

  using Factory = std::function<std::unique_ptr<DataSink>(
      const std::string&,
      const common::MetricsLogPtr&,
      IoStatistics* stats)>;

  static std::unique_ptr<DataSink> create(
      const std::string&,
      const common::MetricsLogPtr& = common::MetricsLog::voidLog(),
      IoStatistics* stats = nullptr);

  static bool registerFactory(const Factory& factory);

 protected:
  std::string name_;
  uint64_t size_;
  MetricsLogPtr metricLogger_;
  IoStatistics* stats_;

  void writeImpl(
      std::vector<DataBuffer<char>>& buffers,
      const std::function<uint64_t(const DataBuffer<char>&)>& callback) {
    DWIO_ENSURE(!isClosed(), "Cannot write to closed sink.");
    uint64_t size = 0;
    for (auto& buf : buffers) {
      size += callback(buf);
    }
    size_ += size;
    if (stats_) {
      stats_->incRawBytesWritten(size);
    }
    // Writing buffer is treated as transferring ownership. So clearing the
    // buffers after all buffers are written.
    buffers.clear();
  }
};

class FileSink : public DataSink {
 public:
  explicit FileSink(
      const std::string& name,
      const MetricsLogPtr& metricLogger = MetricsLog::voidLog(),
      IoStatistics* stats = nullptr);

  ~FileSink() override {
    destroy();
  }

  using DataSink::write;

  void write(std::vector<DataBuffer<char>>& buffers) override;

  static void registerFactory();

 protected:
  void doClose() override {
    ::close(file_);
  }

 private:
  int file_;
};

class MemorySink : public DataSink {
 public:
  MemorySink(
      velox::memory::MemoryPool& pool,
      size_t capacity,
      const MetricsLogPtr& metricLogger = MetricsLog::voidLog(),
      IoStatistics* stats = nullptr)
      : DataSink{"MemorySink", metricLogger, stats}, data_{pool, capacity} {}

  ~MemorySink() override {
    markClosed();
  }

  using DataSink::write;

  void write(std::vector<DataBuffer<char>>& buffers) override {
    writeImpl(buffers, [&](auto& buffer) {
      auto size = buffer.size();
      DWIO_ENSURE_LE(size_ + size, data_.capacity());
      memcpy(data_.data() + size_, buffer.data(), size);
      return size;
    });
  }

  const char* getData() const {
    return data_.data();
  }

  void reset() {
    size_ = 0;
  }

 private:
  DataBuffer<char> data_;
};

/// A file sink support to write to hdfs.
class WriteFileSink : public DataSink {
 public:
  /// The constructor.
  /// @param writeFile The configured write handle for hdfs.
  /// @param metricLogger The metric logger.
  /// @param stats The io statictics.
  explicit WriteFileSink(
      std::unique_ptr<WriteFile> writeFile,
      const MetricsLogPtr& metricLogger = MetricsLog::voidLog(),
      IoStatistics* stats = nullptr);

  ~WriteFileSink() override {
    destroy();
  }

  using DataSink::write;

  /// Write the data.
  /// @param buffers The data buffers to write.
  void write(std::vector<DataBuffer<char>>& buffers) override;

  /// Set flush size.
  /// @param flushSize data flush size, when write data size greater than this
  /// value, will flush the data.
  void setFlushSize(uint64_t flushSize);

  static void registerFactory();

  /// Create a write file sink.
  /// @param filePath The file path to write.
  /// @param props The custimized configuration, if it set as nullptr, will use
  /// the default config from LIB3HDFS_CONF variable for hdfs file write sink.
  /// @param metricsLog The metrics log.
  /// @param stats The io statistics.
  static std::unique_ptr<DataSink> createWriteFileSink(
      const std::string& filePath,
      const std::unordered_map<std::string, std::string>* props = nullptr,
      const MetricsLogPtr& metricsLog = MetricsLog::voidLog(),
      IoStatistics* stats = nullptr);

 protected:
  /// Close the write file handle.
  void doClose() override {
    if (writeFile_ != nullptr) {
      writeFile_->close();
    }
  }

 private:
  /// The write file handle.
  std::unique_ptr<WriteFile> writeFile_;
  /// The data flush size, default value is 128*1024*1024.
  uint64_t flushSize_ = 128 * 1024 * 1024;
};

} // namespace facebook::velox::dwio::common

#define VELOX_STATIC_REGISTER_DATA_SINK(function)                           \
  namespace {                                                               \
  static bool FB_ANONYMOUS_VARIABLE(g_DataSinkFunction) =                   \
      facebook::velox::dwio::common::DataSink::registerFactory((function)); \
  }

#define VELOX_REGISTER_DATA_SINK_METHOD_DEFINITION(class, function)       \
  void class ::registerFactory() {                                        \
    facebook::velox::dwio::common::DataSink::registerFactory((function)); \
  }
