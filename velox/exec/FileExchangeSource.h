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

#include <folly/container/F14Map.h>
#include <folly/dynamic.h>
#include <atomic>
#include <string>
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/ExchangeSource.h"
#include "velox/exec/FileExchangeFormat.h"

namespace facebook::velox::exec {

/// ExchangeSource that reads from file-based exchange storage.
///
/// Used by the existing Velox Exchange operator to read durable exchange
/// data in fault-tolerant mode. The serialized location identifies one
/// successful task attempt's paged partition file.
class FileExchangeSource : public ExchangeSource {
 public:
  /// Creates a source for one task/partition paged file.
  FileExchangeSource(
      const std::string& taskId,
      int32_t destination,
      std::shared_ptr<ExchangeQueue> queue,
      memory::MemoryPool* pool);

  /// Claims the next request while the source remains open.
  bool shouldRequestLocked() override;

  /// Reads and enqueues pages up to the requested byte budget.
  folly::SemiFuture<Response> request(
      uint32_t maxBytes,
      std::chrono::microseconds maxWait) override;

  void pause() override {}

  /// Prevents new requests from starting.
  void close() override;

  folly::SemiFuture<Response> requestDataSizes(
      std::chrono::microseconds maxWait) override {
    return request(0, maxWait);
  }

  folly::F14FastMap<std::string, RuntimeMetric> metrics() const override {
    return {};
  }

  bool supportsMetrics() const override {
    return true;
  }

  folly::dynamic toJson() override {
    return folly::dynamic::object("filePath", filePath_);
  }

 private:
  // Opens the paged input file on first request.
  void openFile();

  // Reads the next page, or returns null at end of file.
  std::unique_ptr<SerializedPageBase> readPage();

  // Enqueues a page and wakes waiting consumers.
  void enqueuePage(std::unique_ptr<SerializedPageBase> page);

  // Verifies the complete file before publishing successful end of stream.
  void enqueueEnd();

  // Verifies the completed file against its committed checksum.
  void verifyChecksum() const;

  // Exact path of the task/partition paged file.
  std::string filePath_;
  // Committed physical file size in bytes.
  uint64_t expectedFileSize_{0};
  // Committed CRC32C of all physical file bytes.
  uint32_t expectedChecksum_{0};
  // Number of physical file bytes consumed so far.
  uint64_t fileBytesRead_{0};
  // Incremental CRC32C of physical file bytes consumed so far.
  uint32_t fileChecksum_{~0U};
  // Filesystem that owns the input file implementation.
  std::shared_ptr<filesystems::FileSystem> fileSystem_;
  // Lazily opened input file retained until this source is destroyed.
  std::unique_ptr<ReadFile> file_;
  // True after close prevents further reads.
  std::atomic_bool closed_{false};
  // True after the terminal queue marker has been emitted. Guarded by the
  // exchange queue mutex together with ExchangeSource::atEnd_.
  bool endEnqueued_{false};
};

/// Factory that creates FileExchangeSource instances for serialized file
/// exchange locations.
class FileExchangeSourceFactory {
 public:
  std::shared_ptr<ExchangeSource> operator()(
      const std::string& taskId,
      int32_t destination,
      std::shared_ptr<ExchangeQueue> queue,
      memory::MemoryPool* pool);

  /// Returns a factory function for registration with
  /// velox::exec::ExchangeSource::registerFactory().
  static ExchangeSource::Factory toFactory() {
    return [](const std::string& taskId,
              int32_t destination,
              std::shared_ptr<ExchangeQueue> queue,
              memory::MemoryPool* pool) -> std::shared_ptr<ExchangeSource> {
      FileExchangeSourceFactory f;
      return f(taskId, destination, std::move(queue), pool);
    };
  }
};

} // namespace facebook::velox::exec
