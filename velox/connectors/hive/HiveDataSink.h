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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/PartitionIdGenerator.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Writer.h"

namespace facebook::velox::dwrf {
class Writer;
}

namespace facebook::velox::connector::hive {
class HiveColumnHandle;

class LocationHandle;
using LocationHandlePtr = std::shared_ptr<const LocationHandle>;

/// Location related properties of the Hive table to be written.
class LocationHandle : public ISerializable {
 public:
  enum class TableType {
    /// Write to a new table to be created.
    kNew,
    /// Write to an existing table.
    kExisting,
  };

  LocationHandle(
      std::string targetPath,
      std::string writePath,
      TableType tableType)
      : targetPath_(std::move(targetPath)),
        writePath_(std::move(writePath)),
        tableType_(tableType) {}

  const std::string& targetPath() const {
    return targetPath_;
  }

  const std::string& writePath() const {
    return writePath_;
  }

  TableType tableType() const {
    return tableType_;
  }

  std::string toString() const;

  static void registerSerDe();

  folly::dynamic serialize() const override;

  static LocationHandlePtr create(const folly::dynamic& obj);

  static const std::string tableTypeName(LocationHandle::TableType type);

  static LocationHandle::TableType tableTypeFromName(const std::string& name);

 private:
  // Target directory path.
  const std::string targetPath_;
  // Staging directory path.
  const std::string writePath_;
  // Whether the table to be written is new, already existing or temporary.
  const TableType tableType_;
};

class HiveSortingColumn : public ISerializable {
 public:
  HiveSortingColumn(
      const std::string& sortColumn,
      const core::SortOrder& sortOrder);

  const std::string& sortColumn() const {
    return sortColumn_;
  }

  core::SortOrder sortOrder() const {
    return sortOrder_;
  }

  folly::dynamic serialize() const override;

  static std::shared_ptr<HiveSortingColumn> deserialize(
      const folly::dynamic& obj,
      void* context);

  std::string toString() const;

  static void registerSerDe();

 private:
  const std::string sortColumn_;
  const core::SortOrder sortOrder_;
};

class HiveBucketProperty : public ISerializable {
 public:
  enum class Kind { kHiveCompatible, kPrestoNative };

  HiveBucketProperty(
      Kind kind,
      int32_t bucketCount,
      const std::vector<std::string>& bucketedBy,
      const std::vector<TypePtr>& bucketedTypes,
      const std::vector<std::shared_ptr<const HiveSortingColumn>>& sortedBy);

  Kind kind() const {
    return kind_;
  }

  static std::string kindString(Kind kind);

  /// Returns the number of bucket count.
  int32_t bucketCount() const {
    return bucketCount_;
  }

  /// Returns the bucketed by column names.
  const std::vector<std::string>& bucketedBy() const {
    return bucketedBy_;
  }

  /// Returns the bucketed by column types.
  const std::vector<TypePtr>& bucketedTypes() const {
    return bucketTypes_;
  }

  /// Returns the hive sorting columns if not empty.
  const std::vector<std::shared_ptr<const HiveSortingColumn>>& sortedBy()
      const {
    return sortedBy_;
  }

  folly::dynamic serialize() const override;

  static std::shared_ptr<HiveBucketProperty> deserialize(
      const folly::dynamic& obj,
      void* context);

  bool operator==(const HiveBucketProperty& other) const {
    return true;
  }

  static void registerSerDe();

  std::string toString() const;

 private:
  void validate() const;

  const Kind kind_;
  const int32_t bucketCount_;
  const std::vector<std::string> bucketedBy_;
  const std::vector<TypePtr> bucketTypes_;
  const std::vector<std::shared_ptr<const HiveSortingColumn>> sortedBy_;
};

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    HiveBucketProperty::Kind kind) {
  os << HiveBucketProperty::kindString(kind);
  return os;
}

class HiveInsertTableHandle;
using HiveInsertTableHandlePtr = std::shared_ptr<HiveInsertTableHandle>;

/**
 * Represents a request for Hive write.
 */
class HiveInsertTableHandle : public ConnectorInsertTableHandle {
 public:
  HiveInsertTableHandle(
      std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
      std::shared_ptr<const LocationHandle> locationHandle,
      const dwio::common::FileFormat tableStorageFormat =
          dwio::common::FileFormat::DWRF,
      std::shared_ptr<HiveBucketProperty> bucketProperty = nullptr)
      : inputColumns_(std::move(inputColumns)),
        locationHandle_(std::move(locationHandle)),
        tableStorageFormat_(tableStorageFormat),
        bucketProperty_(std::move(bucketProperty)) {}

  virtual ~HiveInsertTableHandle() = default;

  const std::vector<std::shared_ptr<const HiveColumnHandle>>& inputColumns()
      const {
    return inputColumns_;
  }

  const std::shared_ptr<const LocationHandle>& locationHandle() const {
    return locationHandle_;
  }

  dwio::common::FileFormat tableStorageFormat() const {
    return tableStorageFormat_;
  }

  bool supportsMultiThreading() const override {
    return true;
  }

  bool isPartitioned() const;

  bool isBucketed() const;

  const HiveBucketProperty* bucketProperty() const;

  bool isInsertTable() const;

  folly::dynamic serialize() const override;

  static HiveInsertTableHandlePtr create(const folly::dynamic& obj);

  static void registerSerDe();

  std::string toString() const;

 private:
  const std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns_;
  const std::shared_ptr<const LocationHandle> locationHandle_;
  const dwio::common::FileFormat tableStorageFormat_;
  const std::shared_ptr<HiveBucketProperty> bucketProperty_;
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

struct HiveWriterInfo {
  explicit HiveWriterInfo(HiveWriterParameters parameters)
      : writerParameters(std::move(parameters)) {}

  const HiveWriterParameters writerParameters;
  vector_size_t numWrittenRows = 0;
};

/// Identifies a hive writer.
struct HiveWriterId {
  std::optional<uint32_t> partitionId{std::nullopt};
  std::optional<uint32_t> bucketId{std::nullopt};

  HiveWriterId() = default;

  explicit HiveWriterId(uint32_t _partitionId)
      : HiveWriterId(_partitionId, std::nullopt) {}

  HiveWriterId(uint32_t _partitionId, std::optional<uint32_t> _bucketId)
      : partitionId(_partitionId), bucketId(_bucketId) {}

  /// Returns the special writer id for the un-partitioned table.
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

class HiveDataSink : public DataSink {
 public:
  HiveDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy);

  static uint32_t maxBucketCount() {
    static const uint32_t kMaxBucketCount = 100'000;
    return kMaxBucketCount;
  }

  void appendData(RowVectorPtr input) override;

  std::vector<std::string> finish() const override;

  void close() override;

 private:
  // Returns true if the table is partitioned.
  FOLLY_ALWAYS_INLINE bool isPartitioned() const {
    return partitionIdGenerator_ != nullptr;
  }

  // Returns true if the table is bucketed.
  FOLLY_ALWAYS_INLINE bool isBucketed() const {
    return bucketCount_ != 0;
  }

  FOLLY_ALWAYS_INLINE bool isCommitRequired() const {
    return commitStrategy_ != CommitStrategy::kNoCommit;
  }

  // Compute the partition id and bucket id for each row in 'input'.
  void computePartitionAndBucketIds(const RowVectorPtr& input);

  // Computes the number of input rows as well as the actual input row indices
  // to each corresponding (bucketed) partition based on the partition and
  // bucket ids calculated by 'computePartitionAndBucketIds'. The function also
  // ensures that there is a writer created for each (bucketed) partition.
  void splitInputRowsAndEnsureWriters();

  // Makes sure to create one writer for the given writer id. The function
  // returns the corresponding index in 'writers_'.
  uint32_t ensureWriter(const HiveWriterId& id);

  // Appends a new writer for the given 'id'. The function returns the index of
  // the newly created writer in 'writers_'.
  uint32_t appendWriter(const HiveWriterId& id);

  HiveWriterParameters getWriterParameters(
      const std::optional<std::string>& partition,
      std::optional<uint32_t> bucketId) const;

  // Gets write and target file names for a writer based on the table commit
  // strategy as well as table partitioned type. If commit is not required, the
  // write file and target file has the same name. If not, add a temp file
  // prefix to the target file for write file name. The coordinator (or driver
  // for Presto on spark) will rename the write file to target file to commit
  // the table write when update the metadata store. If it is a bucketed table,
  // the file name encodes the corresponding bucket id.
  std::pair<std::string, std::string> getWriterFileNames(
      std::optional<uint32_t> bucketId) const;

  HiveWriterParameters::UpdateMode getUpdateMode() const;

  const RowTypePtr inputType_;
  const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle_;
  const ConnectorQueryCtx* connectorQueryCtx_;
  const CommitStrategy commitStrategy_;
  const uint32_t maxOpenWriters_;
  const std::vector<column_index_t> partitionChannels_;
  const std::unique_ptr<PartitionIdGenerator> partitionIdGenerator_;
  const int32_t bucketCount_{0};
  const std::unique_ptr<core::PartitionFunction> bucketFunction_;

  // The map from writer id to the writer index in 'writers_' and 'writerInfo_'.
  folly::F14FastMap<HiveWriterId, uint32_t, HiveWriterIdHasher, HiveWriterIdEq>
      writerIndexMap_;

  // Below are structures for partitions from all inputs. writerInfo_ and
  // writers_ are both indexed by partitionId.
  std::vector<std::shared_ptr<HiveWriterInfo>> writerInfo_;
  std::vector<std::unique_ptr<dwio::common::Writer>> writers_;

  // Below are structures updated when processing current input. partitionIds_
  // are indexed by the row of input_. partitionRows_, rawPartitionRows_ and
  // partitionSizes_ are indexed by partitionId.
  raw_vector<uint64_t> partitionIds_;
  std::vector<BufferPtr> partitionRows_;
  std::vector<vector_size_t*> rawPartitionRows_;
  std::vector<vector_size_t> partitionSizes_;

  // Reusable buffers for bucket id calculations.
  std::vector<uint32_t> bucketIds_;
};

} // namespace facebook::velox::connector::hive
