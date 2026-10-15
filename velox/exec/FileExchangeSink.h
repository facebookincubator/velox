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
#include <optional>

#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/ExchangeSink.h"
#include "velox/exec/FileExchangeFormat.h"

namespace facebook::velox::exec {

/// Writes one task attempt's partitioned output to durable files.
class FileExchangeSink final : public ExchangeSink {
 public:
  /// Creates a sink from serialized transport options.
  static std::shared_ptr<ExchangeSink> create(
      const std::string& config,
      const std::string& taskId,
      memory::MemoryPool* pool);

  /// Creates paged output files lazily for non-empty destination partitions.
  FileExchangeSink(
      std::string rootDir,
      std::string exchangeId,
      std::string taskId,
      int32_t numPartitions);

  /// Appends one length-delimited serialized page to a partition file.
  void append(int32_t partition, std::string_view page) override;

  /// Appends one length-delimited serialized page without coalescing its
  /// IOBuf chain.
  void append(int32_t partition, std::unique_ptr<folly::IOBuf> page) override;

  /// Closes all partition files and returns non-empty committed locations.
  CommittedExchangeOutput finish() override;

  /// Closes all open partition files without committing output.
  void abort() override;

  /// Returns payload bytes and page count written by this sink.
  folly::F14FastMap<std::string, int64_t> stats() const override;

 private:
  struct PartitionWriter {
    // Writes bytes while maintaining the committed size and checksum.
    void append(const void* data, size_t numBytes);

    std::unique_ptr<WriteFile> file;
    uint64_t size{0};
    uint32_t checksum{~0U};
  };

  // Returns the directory containing one partition's task files.
  std::string partitionDir(int32_t partition) const;

  // Returns the single paged file written for one partition.
  std::string partitionFile(int32_t partition) const;

  // Returns the partition writer state, opening its file lazily.
  PartitionWriter& partitionWriter(int32_t partition);

  // Closes one partition writer and returns its output file, if non-empty.
  std::optional<file_exchange::ExchangeOutputFile> closeWriter(
      int32_t partition);

  // Root directory for all file exchange data.
  const std::string rootDir_;
  // Identifier shared by all tasks writing one exchange.
  const std::string exchangeId_;
  // Identifier of the task attempt owned by this sink.
  const std::string taskId_;
  // Fixed number of destination partitions.
  const int32_t numPartitions_;
  // Filesystem selected from the root directory URI.
  const std::shared_ptr<filesystems::FileSystem> fileSystem_;

  // Serializes writer lifecycle and append calls.
  mutable std::mutex mutex_;
  // Lazily opened output and integrity state for each partition.
  std::vector<PartitionWriter> writers_;
  // Total payload bytes, excluding page-size headers.
  int64_t totalBytesWritten_{0};
  // Total number of pages written.
  int64_t totalPages_{0};
  // True after finish or abort closes this sink.
  bool closed_{false};
};

} // namespace facebook::velox::exec
