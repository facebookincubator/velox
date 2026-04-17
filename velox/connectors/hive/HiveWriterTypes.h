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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Portability.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/exec/SpillStats.h"

namespace facebook::velox::connector::hive {

/// Identifies a hive writer.
struct HiveWriterId {
  std::optional<uint32_t> partitionId{std::nullopt};
  std::optional<uint32_t> bucketId{std::nullopt};

  HiveWriterId() = default;

  HiveWriterId(
      std::optional<uint32_t> _partitionId,
      std::optional<uint32_t> _bucketId = std::nullopt)
      : partitionId(_partitionId), bucketId(_bucketId) {}

  /// Returns the special writer id for the un-partitioned (and non-bucketed)
  /// table.
  static const HiveWriterId& unpartitionedId();

  std::string toString() const;

  bool operator==(const HiveWriterId& other) const {
    return std::tie(partitionId, bucketId) ==
        std::tie(other.partitionId, other.bucketId);
  }
};

struct HiveWriterIdHasher {
  std::size_t operator()(const HiveWriterId& id) const {
    return bits::hashMix(
        id.partitionId.value_or(std::numeric_limits<uint32_t>::max()),
        id.bucketId.value_or(std::numeric_limits<uint32_t>::max()));
  }
};

struct HiveWriterIdEq {
  bool operator()(const HiveWriterId& lhs, const HiveWriterId& rhs) const {
    return lhs == rhs;
  }
};

/// Parameters for Hive writers.
class HiveWriterParameters {
 public:
  enum class UpdateMode {
    kNew, // Write files to a new directory.
    kOverwrite, // Overwrite an existing directory.
    // Append mode is currently only supported for unpartitioned tables.
    kAppend, // Append to an unpartitioned table.
  };

  /// @param updateMode Write the files to a new directory, or append to an
  /// existing directory or overwrite an existing directory.
  /// @param partitionName Partition name in the typical Hive style, which is
  /// also the partition subdirectory part of the partition path.
  /// @param targetFileName The final name of a file after committing.
  /// @param targetDirectory The final directory that a file should be in after
  /// committing.
  /// @param writeFileName The temporary name of the file that a running writer
  /// writes to. If a running writer writes directory to the target file, set
  /// writeFileName to targetFileName by default.
  /// @param writeDirectory The temporary directory that a running writer writes
  /// to. If a running writer writes directory to the target directory, set
  /// writeDirectory to targetDirectory by default.
  HiveWriterParameters(
      UpdateMode updateMode,
      std::optional<std::string> partitionName,
      std::string targetFileName,
      std::string targetDirectory,
      std::optional<std::string> writeFileName = std::nullopt,
      std::optional<std::string> writeDirectory = std::nullopt)
      : updateMode_(updateMode),
        partitionName_(std::move(partitionName)),
        targetFileName_(std::move(targetFileName)),
        targetDirectory_(std::move(targetDirectory)),
        writeFileName_(writeFileName.value_or(targetFileName_)),
        writeDirectory_(writeDirectory.value_or(targetDirectory_)) {}

  UpdateMode updateMode() const {
    return updateMode_;
  }

  static std::string updateModeToString(UpdateMode updateMode) {
    switch (updateMode) {
      case UpdateMode::kNew:
        return "NEW";
      case UpdateMode::kOverwrite:
        return "OVERWRITE";
      case UpdateMode::kAppend:
        return "APPEND";
      default:
        VELOX_UNSUPPORTED("Unsupported update mode.");
    }
  }

  const std::optional<std::string>& partitionName() const {
    return partitionName_;
  }

  const std::string& targetFileName() const {
    return targetFileName_;
  }

  const std::string& writeFileName() const {
    return writeFileName_;
  }

  const std::string& targetDirectory() const {
    return targetDirectory_;
  }

  const std::string& writeDirectory() const {
    return writeDirectory_;
  }

 private:
  const UpdateMode updateMode_;
  const std::optional<std::string> partitionName_;
  const std::string targetFileName_;
  const std::string targetDirectory_;
  const std::string writeFileName_;
  const std::string writeDirectory_;
};

/// Information about a single file written as part of a writer's output.
/// When file rotation occurs, multiple HiveFileInfo entries are created.
struct HiveFileInfo {
  /// The temporary file name used during writing (in the staging directory).
  std::string writeFileName;
  /// The final file name after commit (in the target directory).
  std::string targetFileName;
  /// Size of the file in bytes.
  uint64_t fileSize{0};
  /// Number of rows in the file.
  uint64_t numRows{0};
};

struct HiveWriterInfo {
  HiveWriterInfo(
      HiveWriterParameters parameters,
      std::shared_ptr<memory::MemoryPool> _writerPool,
      std::shared_ptr<memory::MemoryPool> _sinkPool,
      std::shared_ptr<memory::MemoryPool> _sortPool)
      : writerParameters(std::move(parameters)),
        nonReclaimableSectionHolder(new tsan_atomic<bool>(false)),
        spillStats(std::make_unique<exec::SpillStats>()),
        writerPool(std::move(_writerPool)),
        sinkPool(std::move(_sinkPool)),
        sortPool(std::move(_sortPool)) {}

  const HiveWriterParameters writerParameters;
  const std::unique_ptr<tsan_atomic<bool>> nonReclaimableSectionHolder;
  /// Collects the spill stats from sort writer if the spilling has been
  /// triggered.
  const std::unique_ptr<exec::SpillStats> spillStats;
  const std::shared_ptr<memory::MemoryPool> writerPool;
  const std::shared_ptr<memory::MemoryPool> sinkPool;
  const std::shared_ptr<memory::MemoryPool> sortPool;
  /// Total rows written by this writer across all files.
  uint64_t numWrittenRows = 0;
  /// Rows written to the current file; reset to 0 when the file is finalized.
  uint64_t currentFileWrittenRows{0};
  uint64_t inputSizeInBytes = 0;
  /// File sequence number for tracking multiple files written due to size-based
  /// splitting. Incremented each time the writer rotates to a new file.
  /// Used to generate sequenced file names (e.g., file_1.orc, file_2.orc).
  /// Invariant during write: fileSequenceNumber == writtenFiles.size()
  /// After close: fileSequenceNumber + 1 == writtenFiles.size() (final file
  /// added)
  uint32_t fileSequenceNumber{0};
  /// Tracks all files written by this writer.
  /// During write: contains only rotated (completed) files.
  /// After close: contains all files including the final one (via
  /// finalizeWriterFile).
  std::vector<HiveFileInfo> writtenFiles;
  /// Snapshot of total bytes written at the start of the current file.
  /// Used as baseline to calculate current file size: rawBytesWritten() - this.
  /// Updated to ioStats->rawBytesWritten() after each rotation.
  uint64_t cumulativeWrittenBytes{0};
  /// Current file's write filename (set when file is created/rotated).
  /// This avoids recomputing makeSequencedFileName() in commitMessage().
  std::string currentWriteFileName;
  /// Current file's target filename (set when file is created/rotated).
  std::string currentTargetFileName;
};

} // namespace facebook::velox::connector::hive
