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

#include <folly/container/F14Set.h>
#include <optional>

#include "velox/common/base/SpillConfig.h"
#include "velox/common/base/SpillStats.h"
#include "velox/common/base/TreeOfLosers.h"
#include "velox/common/compression/Compression.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileInputStream.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/SerializedPageFile.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::exec {
using SpillSortKey = std::pair<column_index_t, CompareFlags>;

/// Records info of a finished spill file which is used for read.
struct SpillFileInfo {
  uint32_t id;
  RowTypePtr type;
  std::string path;
  /// The file size in bytes.
  uint64_t size;
  std::vector<SpillSortKey> sortingKeys;
  common::CompressionKind compressionKind;
};

using SpillFiles = std::vector<SpillFileInfo>;

/// Used to write the spilled data to a sequence of files for one partition. If
/// data is sorted, each file is sorted. The globally sorted order is produced
/// by merging the constituent files.
class SpillWriter : public serializer::SerializedPageFileWriter {
 public:
  /// 'type' is a RowType describing the content. 'numSortKeys' is the number
  /// of leading columns on which the data is sorted. 'path' is a file path
  /// prefix. ' 'targetFileSize' is the target byte size of a single file.
  /// 'writeBufferSize' specifies the size limit of the buffered data before
  /// write to file. 'fileOptions' specifies the file layout on remote storage
  /// which is storage system specific. 'pool' is used for buffering and
  /// constructing the result data read from 'this'. 'stats' is used to collect
  /// the spill write stats. 'fsStats' is used to collect filesystem
  /// internal stats.
  ///
  /// When writing sorted spill runs, the caller is responsible for buffering
  /// and sorting the data. write is called multiple times, followed by flush().
  SpillWriter(
      const RowTypePtr& type,
      const std::vector<SpillSortKey>& sortingKeys,
      common::CompressionKind compressionKind,
      const std::string& pathPrefix,
      uint64_t targetFileSize,
      uint64_t writeBufferSize,
      const std::string& fileCreateConfig,
      const common::UpdateAndCheckSpillLimitCB& updateAndCheckSpillLimitCb,
      memory::MemoryPool* pool,
      folly::Synchronized<common::SpillStats>* stats,
      filesystems::File::IoStats* fsStats);

  /// Finishes this file writer and returns the written spill files info.
  ///
  /// NOTE: we don't allow write to a spill writer after finish
  SpillFiles finish();

  std::vector<std::string> testingSpilledFilePaths() const;

  std::vector<uint32_t> testingSpilledFileIds() const;

 private:
  // Invoked to increment the number of spilled files and the file size.
  void updateFileStats(
      const serializer::SerializedPageFile::FileInfo& fileInfo) override;

  // Invoked to update the number of spilled rows.
  void updateAppendStats(uint64_t numRows, uint64_t serializationTimeUs)
      override;

  // Invoked to update the disk write stats.
  void updateWriteStats(
      uint64_t spilledBytes,
      uint64_t flushTimeUs,
      uint64_t writeTimeUs) override;

  const RowTypePtr type_;

  const std::vector<SpillSortKey> sortingKeys_;

  folly::Synchronized<common::SpillStats>* const stats_;

  // Updates the aggregated bytes of this query, and throws if exceeds
  // the max bytes limit.
  const common::UpdateAndCheckSpillLimitCB updateAndCheckLimitCb_;
};

/// Represents a spill file for read which turns the serialized spilled data
/// on disk back into a sequence of spilled row vectors.
///
/// NOTE: The class will not delete spill file upon destruction, so the user
/// needs to remove the unused spill files at some point later. For example, a
/// query Task deletes all the generated spill files in one operation using
/// rmdir() call.
class SpillReadFile : public serializer::SerializedPageFileReader {
 public:
  static std::unique_ptr<SpillReadFile> create(
      const SpillFileInfo& fileInfo,
      uint64_t bufferSize,
      memory::MemoryPool* pool,
      folly::Synchronized<common::SpillStats>* stats,
      filesystems::File::IoStats* fsStats = nullptr);

  uint32_t id() const {
    return id_;
  }

  const std::vector<SpillSortKey>& sortingKeys() const {
    return sortingKeys_;
  }

  /// Returns the file size in bytes.
  uint64_t size() const {
    return size_;
  }

  const std::string& testingFilePath() const {
    return path_;
  }

 private:
  SpillReadFile(
      uint32_t id,
      const std::string& path,
      uint64_t size,
      uint64_t bufferSize,
      const RowTypePtr& type,
      const std::vector<SpillSortKey>& sortingKeys,
      common::CompressionKind compressionKind,
      memory::MemoryPool* pool,
      folly::Synchronized<common::SpillStats>* stats,
      filesystems::File::IoStats* fsStats);

  // Records spill read stats at the end of read input.
  void updateFinalStats() override;

  void updateSerializationTimeStats(uint64_t timeNs) override;

  // The spill file id which is monotonically increasing and unique for each
  // associated spill partition.
  const uint32_t id_;

  const std::string path_;

  // The file size in bytes.
  const uint64_t size_;

  const std::vector<SpillSortKey> sortingKeys_;

  folly::Synchronized<common::SpillStats>* const stats_;
};

} // namespace facebook::velox::exec
