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

#include "velox/common/base/SpillStats.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileInputStream.h"
#include "velox/exec/ExchangeQueue.h"

namespace facebook::velox::exec {
namespace test {
class SerializedPageSpillerTest;
}

/// Used for spilling a sequence of 'SerializedPage'. The spiller preserves the
/// order of the pages.
class SerializedPageSpiller {
 public:
  struct Result {
    std::deque<std::string> spillFilePaths;
    uint64_t totalPages;
    uint64_t totalBytes;
  };

  SerializedPageSpiller(
      const std::string& filePrefix,
      const std::string& fileCreateConfig,
      uint64_t writeBufferSize,
      memory::MemoryPool* pool,
      folly::Synchronized<common::SpillStats>* spillStats)
      : filePrefix_(filePrefix),
        fileCreateConfig_(fileCreateConfig),
        writeBufferSize_(writeBufferSize),
        spillStats_(spillStats),
        pool_(pool) {
    VELOX_CHECK_NOT_NULL(pool_);
  }

  /// Spills all the 'pages' to a single file. The method does not free the
  /// original in-memory structure of 'pages'. It is caller's responsibility to
  /// free them.
  void spill(const std::vector<std::shared_ptr<SerializedPage>>* pages);

  /// Finishes the spilling and return the spilled result.
  Result finishSpill();

 private:
  // Creates and returns the next spill file name and file to write
  // to.
  std::tuple<std::string, std::unique_ptr<WriteFile>> nextSpillWriteFile();

  const std::string filePrefix_;

  const std::string fileCreateConfig_;

  const uint64_t writeBufferSize_;

  folly::Synchronized<common::SpillStats>* const spillStats_;

  memory::MemoryPool* const pool_;

  // Each spilled file represents a series of 'SerializedPage'.
  std::deque<std::string> spillFilePaths_;

  uint64_t totalBytes_{0};

  uint64_t totalPages_{0};

  uint32_t nextFileId_{0};

  friend class test::SerializedPageSpillerTest;
};

/// Used for reading a sequence of 'SerializedPage' that were spilled by
/// 'SerializedPageSpiller'. The reading preserves the page order. It is used by
/// 'DestinationBuffer' and provides convenient APIs for reading and deleting
/// the pages, with unspilling handled transparently.
class SerializedPageSpillReader {
 public:
  SerializedPageSpillReader(
      SerializedPageSpiller::Result&& spillResult,
      uint64_t readBufferSize,
      memory::MemoryPool* pool,
      folly::Synchronized<common::SpillStats>* spillStats)
      : readBufferSize_(readBufferSize),
        pool_(pool),
        spillStats_(spillStats),
        spillFilePaths_(std::move(spillResult.spillFilePaths)),
        remainingPages_(spillResult.totalPages),
        remainingBytes_(spillResult.totalBytes) {}

  /// Returns true if there are any remaining pages from the reader.
  bool empty() const;

  /// Returns the number of remaining pages from the reader.
  uint64_t remainingPages() const;

  /// Returns the total bytes of the remaining pages from the reader.
  uint64_t remainingBytes() const;

  std::shared_ptr<SerializedPage> at(uint64_t index);

  /// Returns true if the page at 'index' is null.
  bool isEmptyAt(uint64_t index);

  uint64_t sizeAt(uint64_t index);

  /// Delete 'numPages' from the front.
  void deleteFront(uint64_t numPages);

  /// Delete all pages from the reader.
  void deleteAll();

 private:
  // Ensures the current file stream is open. If not opens the next file.
  void ensureFileStream();

  // Ensures all pages up to 'index' are loaded in memory.
  void ensurePages(uint64_t index);

  // Unspills one serialized page and returns it.
  std::shared_ptr<SerializedPage> unspillNextPage();

  const uint64_t readBufferSize_;

  memory::MemoryPool* const pool_;

  folly::Synchronized<common::SpillStats>* const spillStats_;

  // The current file stream.
  std::unique_ptr<common::FileInputStream> curFileStream_;

  // A small number of front pages buffered in memory from spilled pages.
  // These pages will be kept in memory and won't be spilled again.
  std::vector<std::shared_ptr<SerializedPage>> bufferedPages_;

  std::deque<std::string> spillFilePaths_;

  uint64_t remainingPages_;

  uint64_t remainingBytes_;

  friend class test::SerializedPageSpillerTest;
};
} // namespace facebook::velox::exec
