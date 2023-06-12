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

#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/HiveConnector.h"

#include "velox/common/base/Fs.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HivePartitionUtil.h"
#include "velox/exec/HashPartitionFunction.h"

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace facebook::velox::connector::hive {

namespace {

// Returns a subset of column indices corresponding to partition keys.
std::vector<column_index_t> getPartitionChannels(
    const std::shared_ptr<const HiveInsertTableHandle>& insertTableHandle) {
  std::vector<column_index_t> channels;

  for (column_index_t i = 0; i < insertTableHandle->inputColumns().size();
       i++) {
    if (insertTableHandle->inputColumns()[i]->isPartitionKey()) {
      channels.push_back(i);
    }
  }

  return channels;
}

std::string makePartitionDirectory(
    const std::string& tableDirectory,
    const std::optional<std::string>& partitionSubdirectory) {
  if (partitionSubdirectory.has_value()) {
    return fs::path(tableDirectory) / partitionSubdirectory.value();
  }
  return tableDirectory;
}

std::string makeUuid() {
  return boost::lexical_cast<std::string>(boost::uuids::random_generator()());
}

std::unordered_map<LocationHandle::TableType, std::string> tableTypeNames() {
  return {
      {LocationHandle::TableType::kNew, "kNew"},
      {LocationHandle::TableType::kExisting, "kExisting"},
  };
}

template <typename K, typename V>
std::unordered_map<V, K> invertMap(const std::unordered_map<K, V>& mapping) {
  std::unordered_map<V, K> inverted;
  for (const auto& [key, value] : mapping) {
    inverted.emplace(value, key);
  }
  return inverted;
}

std::unique_ptr<core::PartitionFunction> createBucketFunction(
    HiveBucketProperty* bucketProperty,
    const RowTypePtr inputType) {
  if (bucketProperty == nullptr) {
    return nullptr;
  }
  std::vector<column_index_t> bucketedByChannels;
  for (const auto& column : bucketProperty->bucketedBy()) {
    for (column_index_t i = 0; i < inputType->size(); ++i) {
      if (inputType->childAt(i)->name() == column) {
        bucketedByChannels.push_back(i);
        break;
      }
    }
  }
  VELOX_CHECK_EQ(
      bucketedByChannels.size(), bucketProperty->bucketedBy().size());
  return std::make_unique<exec::HashPartitionFunction>(
      bucketProperty->bucketCount(), inputType, bucketedByChannels);
}

#if 0
std::string computeBucketedFileName(
    const std::string& queryId,
    int32_t bucket) {
  // std::string paddedBucket = Strings.padStart(Integer.toString(bucket),
  // BUCKET_NUMBER_PADDING, '0'); return format("0%s_0_%s", paddedBucket,
  // queryId);
}
#endif

HiveWriterId getWriterId(
    const std::optional<uint64_t>& partitionId,
    std::optional<uint32_t> bucketId) {
  return HiveWriterId{partitionId.value_or(0), bucketId.value_or(0)};
}

HiveWriterId unpartitionedWriterId() {
  static const HiveWriterId writerId{0};
  return writerId;
}
} // namespace

HiveSortingColumn::HiveSortingColumn(
    const std::string& sortColumn,
    core::SortOrder sortOrder)
    : sortColumn_(sortColumn), sortOrder_(sortOrder) {
  if (FOLLY_UNLIKELY(
          sortColumn_.empty() ||
          (sortOrder_.isAscending() && !sortOrder_.isNullsFirst()) ||
          (!sortOrder_.isAscending() && sortOrder_.isNullsFirst()))) {
    VELOX_USER_FAIL("Bad HiveSortingColumn: {}", toString());
  }
}

std::string HiveSortingColumn::toString() const {
  return fmt::format(
      "[COLUMN{{}} ORDER[{}]]", sortColumn_, sortOrder_.toString());
}

HiveBucketProperty::HiveBucketProperty(
    Kind kind,
    int32_t bucketCount,
    const std::vector<std::string>& bucketedBy,
    const std::vector<TypePtr>& bucketTypes,
    const std::vector<HiveSortingColumn>& sortedBy)
    : kind_(kind),
      bucketedBy_(bucketedBy),
      bucketCount_(bucketCount),
      bucketTypes_(bucketTypes),
      sortedBy_(sortedBy) {
  validate();
}

void HiveBucketProperty::validate() const {
  VELOX_USER_CHECK_GT(bucketCount_, 0, "Bucket count can't be zero");
  VELOX_USER_CHECK(!bucketedBy_.empty(), "Bucket columns can't be empty");
  VELOX_USER_CHECK_EQ(
      bucketedBy_.size(),
      bucketTypes_.size(),
      "The number of bucket column and type do not match {}",
      toString());
}

const std::string LocationHandle::tableTypeName(
    LocationHandle::TableType type) {
  static const auto tableTypes = tableTypeNames();
  return tableTypes.at(type);
}

LocationHandle::TableType LocationHandle::tableTypeFromName(
    const std::string& name) {
  static const auto nameTableTypes = invertMap(tableTypeNames());
  return nameTableTypes.at(name);
}

HiveDataSink::HiveDataSink(
    RowTypePtr inputType,
    std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy)
    : inputType_(std::move(inputType)),
      insertTableHandle_(std::move(insertTableHandle)),
      connectorQueryCtx_(connectorQueryCtx),
      maxOpenWriters_(
          HiveConfig::maxPartitionsPerWriter(connectorQueryCtx_->config())),
      commitStrategy_(commitStrategy),
      partitionChannels_(getPartitionChannels(insertTableHandle_)),
      partitionIdGenerator_(
          !partitionChannels_.empty() ? std::make_unique<PartitionIdGenerator>(
                                            inputType_,
                                            partitionChannels_,
                                            maxOpenWriters_,
                                            connectorQueryCtx_->memoryPool())
                                      : nullptr),
      bucketFunction_(createBucketFunction(
          insertTableHandle_->bucketProperty(),
          inputType_)) {
  // TODO: remove this hack after Prestissimo adds to register dwrf writer.
  facebook::velox::dwrf::registerDwrfWriterFactory();
}

void HiveDataSink::appendData(RowVectorPtr input) {
  // Write to unpartitioned table.
  if (partitionChannels_.empty()) {
    ensureSingleWriter();

    const auto writerId = unpartitionedWriterId();
    writers_[writerId]->write(input);
    writerInfo_[writerId]->numWrittenRows += input->size();
    return;
  }

  maybeCalculateBucketIdVector(input);

  // Write to partitioned table.
  partitionIdGenerator_->run(input, partitionIds_);

  for (column_index_t i = 0; i < input->childrenSize(); ++i) {
    input->childAt(i)->loadedVector();
  }

  const auto numPartitions = partitionIdGenerator_->numPartitions();

  // All inputs belong to a single partition.
  if (numPartitions == 1) {
    const HiveWriterId writerId{partitionIds_[0], bucketIds_[0]};
    writers_[writerId]->write(input);
    writerInfo_[writerId]->numWrittenRows += input->size();
    return;
  }

  computePartitionRowsAndEnsureWriters();

  for (auto entry : partitionSizes_) {
    const vector_size_t partitionSize = entry.second;
    if (partitionSize == 0) {
      continue;
    }
    const HiveWriterId& id = entry.first;
    RowVectorPtr writerInput = partitionSize == input->size()
        ? input
        : exec::wrap(partitionSize, partitionRows_[id], input);
    writers_[id]->write(writerInput);
    writerInfo_[id]->numWrittenRows += partitionSize;
  }
}

void HiveDataSink::maybeCalculateBucketIdVector(RowVectorPtr input) {
  if (bucketFunction_ == nullptr) {
    return;
  }
  bucketFunction_->partition(*input, bucketIds_);
}

std::vector<std::string> HiveDataSink::finish() const {
  std::vector<std::string> partitionUpdates;
  partitionUpdates.reserve(writerInfo_.size());

  for (const auto& entry : writerInfo_) {
    if (entry.second != nullptr) {
      auto& info = entry.second;
      // clang-format off
      auto partitionUpdateJson = folly::toJson(
       folly::dynamic::object
          ("name", info->writerParameters.partitionName().value_or(""))
          ("updateMode",
            HiveWriterParameters::updateModeToString(
              info->writerParameters.updateMode()))
          ("writePath", info->writerParameters.writeDirectory())
          ("targetPath", info->writerParameters.targetDirectory())
          ("fileWriteInfos", folly::dynamic::array(
            folly::dynamic::object
              ("writeFileName", info->writerParameters.writeFileName())
              ("targetFileName", info->writerParameters.targetFileName())
              ("fileSize", 0)))
          ("rowCount", info->numWrittenRows)
         // TODO(gaoge): track and send the fields when inMemoryDataSizeInBytes, onDiskDataSizeInBytes
         // and containsNumberedFileNames are needed at coordinator when file_renaming_enabled are turned on.
          ("inMemoryDataSizeInBytes", 0)
          ("onDiskDataSizeInBytes", 0)
          ("containsNumberedFileNames", true));
      // clang-format on
      partitionUpdates.push_back(partitionUpdateJson);
    }
  }
  return partitionUpdates;
}

void HiveDataSink::close() {
  for (const auto& entry : writers_) {
    entry.second->close();
  }
}

void HiveDataSink::ensureSingleWriter() {
  if (writers_.empty()) {
    appendWriter(unpartitionedWriterId());
  }
}

void HiveDataSink::ensurePartitionWriter(const HiveWriterId& id) {
  if (writers_.count(id) != 0) {
    return;
  }
  appendWriter(id);
}

void HiveDataSink::appendWriter(const HiveWriterId& id) {
  // Check max open writers.
  VELOX_USER_CHECK_LE(
      writers_.size(),
      maxOpenWriters_,
      "Exceeded limit of {} open writers for partitions/buckets",
      maxOpenWriters_);
  VELOX_CHECK_EQ(writers_.size(), writerInfo_.size());

  std::optional<std::string> partitionName;
  if (partitionIdGenerator_ != nullptr) {
    partitionName = partitionIdGenerator_->partitionName(id.partitionId);
  }
  // Without explicitly setting flush policy, the default memory based flush
  // policy is used.
  auto writerParameters = getWriterParameters(partitionName, id.bucketId);
  const auto writePath = fs::path(writerParameters.writeDirectory()) /
      writerParameters.writeFileName();
  writerInfo_.emplace(
      id, std::make_shared<HiveWriterInfo>(std::move(writerParameters)));

  auto writerFactory =
      dwio::common::getWriterFactory(insertTableHandle_->tableStorageFormat());
  dwio::common::WriterOptions options;
  options.schema = inputType_;
  options.memoryPool = connectorQueryCtx_->connectorMemoryPool();
  writers_.emplace(
      id,
      writerFactory->createWriter(
          dwio::common::DataSink::create(writePath), options));
}

void HiveDataSink::computePartitionRowsAndEnsureWriters() {
  VELOX_CHECK_EQ(bucketIds_.size(), partitionIds_.size());

  for (auto entry : partitionSizes_) {
    entry.second = 0;
  }

  const auto numRows = partitionIds_.size();
  for (auto row = 0; row < numRows; ++row) {
    const HiveWriterId id{partitionIds_[row], bucketIds_[row]};
    if (FOLLY_UNLIKELY(partitionRows_[id] == nullptr) ||
        (partitionRows_[id]->capacity() < numRows * sizeof(vector_size_t))) {
      ensurePartitionWriter(id);
      partitionRows_[id] =
          allocateIndices(numRows, connectorQueryCtx_->memoryPool());
      rawPartitionRows_[id] = partitionRows_[id]->asMutable<vector_size_t>();
    }
    rawPartitionRows_[id][partitionSizes_[id]] = row;
    ++partitionSizes_[id];
  }

  for (auto& entry : partitionRows_) {
    entry.second->setSize(partitionSizes_[entry.first] * sizeof(vector_size_t));
  }
}

HiveWriterParameters HiveDataSink::getWriterParameters(
    const std::optional<std::string>& partition,
    std::optional<uint32_t> bucketId) const {
  const auto updateMode = getUpdateMode();

  std::string targetFileName;
  std::string writeFileName;
  switch (commitStrategy_) {
    case CommitStrategy::kNoCommit: {
      if (bucketId.has_value()) {
        targetFileName = fmt::format(
            "{}_{}_{}_{}",
            connectorQueryCtx_->taskId(),
            connectorQueryCtx_->driverId(),
            bucketId.value(),
            makeUuid());
      } else {
        targetFileName = fmt::format(
            "{}_{}_{}",
            connectorQueryCtx_->taskId(),
            connectorQueryCtx_->driverId(),
            makeUuid());
      }
      writeFileName = targetFileName;
      break;
    }
    case CommitStrategy::kTaskCommit: {
      if (bucketId.has_value()) {
        targetFileName = fmt::format(
            "{}_{}_{}_{}",
            connectorQueryCtx_->taskId(),
            connectorQueryCtx_->driverId(),
            bucketId.value(),
            0);
      } else {
        targetFileName = fmt::format(
            "{}_{}_{}",
            connectorQueryCtx_->taskId(),
            connectorQueryCtx_->driverId(),
            0);
      }
      writeFileName =
          fmt::format(".tmp.velox.{}_{}", targetFileName, makeUuid());
      break;
    }
    default:
      VELOX_UNREACHABLE(commitStrategyToString(commitStrategy_));
  }

  return HiveWriterParameters{
      updateMode,
      partition,
      targetFileName,
      makePartitionDirectory(
          insertTableHandle_->locationHandle()->targetPath(), partition),
      writeFileName,
      makePartitionDirectory(
          insertTableHandle_->locationHandle()->writePath(), partition)};
}

HiveWriterParameters::UpdateMode HiveDataSink::getUpdateMode() const {
  if (insertTableHandle_->isInsertTable()) {
    if (insertTableHandle_->isPartitioned()) {
      const auto insertBehavior = HiveConfig::insertExistingPartitionsBehavior(
          connectorQueryCtx_->config());
      switch (insertBehavior) {
        case HiveConfig::InsertExistingPartitionsBehavior::kOverwrite:
          return HiveWriterParameters::UpdateMode::kOverwrite;
        case HiveConfig::InsertExistingPartitionsBehavior::kError:
          return HiveWriterParameters::UpdateMode::kNew;
        default:
          VELOX_UNSUPPORTED(
              "Unsupported insert existing partitions behavior: {}",
              HiveConfig::insertExistingPartitionsBehaviorString(
                  insertBehavior));
      }
    } else {
      if (insertTableHandle_->isBucketed()) {
        VELOX_USER_FAIL("Cannot insert into bucketed unpartitioned Hive table");
      }
      if (HiveConfig::immutablePartitions(connectorQueryCtx_->config())) {
        VELOX_USER_FAIL("Unpartitioned Hive tables are immutable.");
      }
      return HiveWriterParameters::UpdateMode::kAppend;
    }
  } else {
    return HiveWriterParameters::UpdateMode::kNew;
  }
}

bool HiveInsertTableHandle::isPartitioned() const {
  return std::any_of(
      inputColumns_.begin(), inputColumns_.end(), [](auto column) {
        return column->isPartitionKey();
      });
}

bool HiveInsertTableHandle::isBucketed() const {
  return bucketProperty_.has_value();
}

bool HiveInsertTableHandle::isInsertTable() const {
  return locationHandle_->tableType() == LocationHandle::TableType::kExisting;
}

folly::dynamic HiveInsertTableHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "HiveInsertTableHandle";
  folly::dynamic arr = folly::dynamic::array;
  for (const auto& ic : inputColumns_) {
    arr.push_back(ic->serialize());
  }

  obj["inputColumns"] = arr;
  obj["locationHandle"] = locationHandle_->serialize();
  return obj;
}

HiveInsertTableHandlePtr HiveInsertTableHandle::create(
    const folly::dynamic& obj) {
  auto inputColumns = ISerializable::deserialize<std::vector<HiveColumnHandle>>(
      obj["inputColumns"]);
  auto locationHandle =
      ISerializable::deserialize<LocationHandle>(obj["locationHandle"]);
  return std::make_shared<HiveInsertTableHandle>(inputColumns, locationHandle);
}

void HiveInsertTableHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("HiveInsertTableHandle", HiveInsertTableHandle::create);
}

std::string HiveInsertTableHandle::toString() const {
  std::ostringstream out;
  out << "HiveInsertTableHandle [inputColumns: [";
  for (const auto& i : inputColumns_) {
    out << " " << i->toString();
  }
  out << " ], locationHandle: " << locationHandle_->toString() << "]";
  return out.str();
}

std::string LocationHandle::toString() const {
  return fmt::format(
      "LocationHandle [targetPath: {}, writePath: {}, tableType: {},",
      targetPath_,
      writePath_,
      tableTypeName(tableType_));
}

void LocationHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("LocationHandle", LocationHandle::create);
}

folly::dynamic LocationHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "LocationHandle";
  obj["targetPath"] = targetPath_;
  obj["writePath"] = writePath_;
  obj["tableType"] = tableTypeName(tableType_);
  return obj;
}

LocationHandlePtr LocationHandle::create(const folly::dynamic& obj) {
  auto targetPath = obj["targetPath"].asString();
  auto writePath = obj["writePath"].asString();
  auto tableType = tableTypeFromName(obj["tableType"].asString());
  return std::make_shared<LocationHandle>(targetPath, writePath, tableType);
}

} // namespace facebook::velox::connector::hive
