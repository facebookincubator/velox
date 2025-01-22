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

#include "velox/common/base/Counters.h"
#include "velox/common/base/Fs.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/core/ITypedExpr.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/SortingWriter.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/SortBuffer.h"

#include "velox/experimental/cudf/connectors/parquet/ParquetConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnectorSplit.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSink.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::cudf_velox::connector::parquet {

/*
// CUDF STRUCT for sortingColumn
struct sorting_column {
  int column_idx{}; //!< leaf column index within the row group
  bool is_descending{false}; //!< true if sort order is descending
  bool is_nulls_first{true}; //!< true if nulls come before non-null values
};
*/

namespace {
#define WRITER_NON_RECLAIMABLE_SECTION_GUARD(index)       \
  memory::NonReclaimableSectionGuard nonReclaimableGuard( \
      writerInfo_[(index)]->nonReclaimableSectionHolder.get())

// Returns the type of non-partition data columns.
RowTypePtr getNonPartitionTypes(
    const std::vector<column_index_t>& dataCols,
    const RowTypePtr& inputType) {
  std::vector<std::string> childNames;
  std::vector<TypePtr> childTypes;
  const auto& dataSize = dataCols.size();
  childNames.reserve(dataSize);
  childTypes.reserve(dataSize);
  for (int dataCol : dataCols) {
    childNames.push_back(inputType->nameOf(dataCol));
    childTypes.push_back(inputType->childAt(dataCol));
  }

  return ROW(std::move(childNames), std::move(childTypes));
}

// Filters out partition columns if there is any.
RowVectorPtr makeDataInput(
    const std::vector<column_index_t>& dataCols,
    const RowVectorPtr& input) {
  std::vector<VectorPtr> childVectors;
  childVectors.reserve(dataCols.size());
  for (int dataCol : dataCols) {
    childVectors.push_back(input->childAt(dataCol));
  }

  return std::make_shared<RowVector>(
      input->pool(),
      getNonPartitionTypes(dataCols, asRowType(input->type())),
      input->nulls(),
      input->size(),
      std::move(childVectors),
      input->getNullCount());
}

// Returns a subset of column indices corresponding to partition keys.
std::vector<column_index_t> getPartitionChannels(
    const std::shared_ptr<const ParquetInsertTableHandle>& insertTableHandle) {
  std::vector<column_index_t> channels;

  for (column_index_t i = 0; i < insertTableHandle->inputColumns().size();
       i++) {
    if (insertTableHandle->inputColumns()[i]->isPartitionKey()) {
      channels.push_back(i);
    }
  }

  return channels;
}

// Returns the column indices of non-partition data columns.
std::vector<column_index_t> getNonPartitionChannels(
    const std::vector<column_index_t>& partitionChannels,
    const column_index_t childrenSize) {
  std::vector<column_index_t> dataChannels;
  dataChannels.reserve(childrenSize - partitionChannels.size());

  for (column_index_t i = 0; i < childrenSize; i++) {
    if (std::find(partitionChannels.cbegin(), partitionChannels.cend(), i) ==
        partitionChannels.cend()) {
      dataChannels.push_back(i);
    }
  }

  return dataChannels;
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
    const ParquetBucketProperty& bucketProperty,
    const RowTypePtr& inputType) {
  const auto& bucketedBy = bucketProperty.bucketedBy();
  const auto& bucketedTypes = bucketProperty.bucketedTypes();
  std::vector<column_index_t> bucketedByChannels;
  bucketedByChannels.reserve(bucketedBy.size());
  for (int32_t i = 0; i < bucketedBy.size(); ++i) {
    const auto& bucketColumn = bucketedBy[i];
    const auto& bucketType = bucketedTypes[i];
    const auto inputChannel = inputType->getChildIdx(bucketColumn);
    if (FOLLY_UNLIKELY(
            !inputType->childAt(inputChannel)->equivalent(*bucketType))) {
      VELOX_USER_FAIL(
          "Input column {} type {} doesn't match bucket type {}",
          inputType->nameOf(inputChannel),
          inputType->childAt(inputChannel)->toString(),
          bucketType->toString());
    }
    bucketedByChannels.push_back(inputChannel);
  }
  return std::make_unique<ParquetPartitionFunction>(
      bucketProperty.bucketCount(), bucketedByChannels);
}

std::string computeBucketedFileName(
    const std::string& queryId,
    int32_t bucket) {
  static const uint32_t kMaxBucketCountPadding =
      std::to_string(ParquetDataSink::maxBucketCount() - 1).size();
  const std::string bucketValueStr = std::to_string(bucket);
  return fmt::format(
      "0{:0>{}}_0_{}", bucketValueStr, kMaxBucketCountPadding, queryId);
}

std::shared_ptr<memory::MemoryPool> createSinkPool(
    const std::shared_ptr<memory::MemoryPool>& writerPool) {
  return writerPool->addLeafChild(fmt::format("{}.sink", writerPool->name()));
}

std::shared_ptr<memory::MemoryPool> createSortPool(
    const std::shared_ptr<memory::MemoryPool>& writerPool) {
  return writerPool->addLeafChild(fmt::format("{}.sort", writerPool->name()));
}

uint64_t getFinishTimeSliceLimitMsFromParquetConfig(
    const std::shared_ptr<const ParquetConfig>& config,
    const config::ConfigBase* sessions) {
  const uint64_t flushTimeSliceLimitMsFromConfig =
      config->sortWriterFinishTimeSliceLimitMs(sessions);
  // NOTE: if the flush time slice limit is set to 0, then we treat it as no
  // limit.
  return flushTimeSliceLimitMsFromConfig == 0
      ? std::numeric_limits<uint64_t>::max()
      : flushTimeSliceLimitMsFromConfig;
}

cudf::io::column_encoding getEncodingType(arrow::Encoding::type encoding) {
  using encoding_type = cudf::io::column_encoding;

  static std::unordered_map<arrow::Encoding::type, encoding_type> const map = {
      {arrow::Encoding::type::PLAIN_DICTIONARY, encoding_type::DICTIONARY},
      {arrow::Encoding::type::PLAIN, encoding_type::PLAIN},
      {arrow::Encoding::type::DELTA_BINARY_PACKED,
       encoding_type::DELTA_BINARY_PACKED},
      {arrow::Encoding::type::DELTA_LENGTH_BYTE_ARRAY,
       encoding_type::DELTA_LENGTH_BYTE_ARRAY},
      {arrow::Encoding::type::DELTA_BYTE_ARRAY,
       encoding_type::DELTA_BYTE_ARRAY},
  };

  VELOX_CHECK(
      map.find(encoding) != map.end(),
      "Unsupported encoding type requested. Supported encoding types are: "
      "PLAIN_DICTIONARY, PLAIN, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, "
      "DELTA_BYTE_ARRAY");

  return map.at(encoding);
}

cudf::io::compression_type getCompressionType(
    facebook::velox::common::CompressionKind name) {
  using compression_type = cudf::io::compression_type;

  static std::unordered_map<
      facebook::velox::common::CompressionKind,
      compression_type> const map = {
      {facebook::velox::common::CompressionKind::CompressionKind_NONE,
       compression_type::NONE},
      {facebook::velox::common::CompressionKind::CompressionKind_SNAPPY,
       compression_type::SNAPPY},
      {facebook::velox::common::CompressionKind::CompressionKind_LZ4,
       compression_type::LZ4},
      {facebook::velox::common::CompressionKind::CompressionKind_ZSTD,
       compression_type::ZSTD}};

  VELOX_CHECK(
      map.find(name) != map.end(),
      "Unsupported compression type requested. Supported compression types are: "
      "NONE, SNAPPY, LZ4, ZSTD");

  return map.at(name);
}

} // namespace

const ParquetWriterId& ParquetWriterId::unpartitionedId() {
  static const ParquetWriterId writerId{0};
  return writerId;
}

std::string ParquetWriterId::toString() const {
  if (partitionId.has_value() && bucketId.has_value()) {
    return fmt::format("part[{}.{}]", partitionId.value(), bucketId.value());
  }

  if (partitionId.has_value() && !bucketId.has_value()) {
    return fmt::format("part[{}]", partitionId.value());
  }

  // This WriterId is used to add an identifier in the MemoryPools. This could
  // indicate unpart, but the bucket number needs to be disambiguated. So
  // creating a new label using bucket.
  if (!partitionId.has_value() && bucketId.has_value()) {
    return fmt::format("bucket[{}]", bucketId.value());
  }

  return "unpart";
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

ParquetSortingColumn::ParquetSortingColumn(
    const std::string& sortColumn,
    const core::SortOrder& sortOrder)
    : sortColumn_(sortColumn), sortOrder_(sortOrder) {
  VELOX_USER_CHECK(!sortColumn_.empty(), "parquet sort column must be set");

  if (FOLLY_UNLIKELY(
          (sortOrder_.isAscending() && !sortOrder_.isNullsFirst()) ||
          (!sortOrder_.isAscending() && sortOrder_.isNullsFirst()))) {
    VELOX_USER_FAIL("Bad parquet sort order: {}", toString());
  }
}

folly::dynamic ParquetSortingColumn::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "ParquetSortingColumn";
  obj["columnName"] = sortColumn_;
  obj["sortOrder"] = sortOrder_.serialize();
  return obj;
}

std::shared_ptr<ParquetSortingColumn> ParquetSortingColumn::deserialize(
    const folly::dynamic& obj,
    void* context) {
  const std::string columnName = obj["columnName"].asString();
  const auto sortOrder = core::SortOrder::deserialize(obj["sortOrder"]);
  return std::make_shared<ParquetSortingColumn>(columnName, sortOrder);
}

std::string ParquetSortingColumn::toString() const {
  return fmt::format(
      "[COLUMN[{}] ORDER[{}]]", sortColumn_, sortOrder_.toString());
}

void HiveSortingColumn::registerSerDe() {
  auto& registry =
      facebook::velox::DeserializationWithContextRegistryForSharedPtr();
  registry.Register("ParquetSortingColumn", ParquetSortingColumn::deserialize);
}

ParquetDataSink::ParquetDataSink(
    RowTypePtr inputType,
    std::shared_ptr<const ParquetInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const ParquetConfig>& parquetConfig)
    : inputType_(std::move(inputType)),
      insertTableHandle_(std::move(insertTableHandle)),
      connectorQueryCtx_(connectorQueryCtx),
      commitStrategy_(commitStrategy),
      parquetConfig_(parquetConfig),
      updateMode_(getUpdateMode()),
      maxOpenWriters_(parquetConfig_->maxPartitionsPerWriters(
          connectorQueryCtx->sessionProperties())),
      partitionChannels_(getPartitionChannels(insertTableHandle_)),
      partitionIdGenerator_(
          !partitionChannels_.empty()
              ? std::make_unique<PartitionIdGenerator>(
                    inputType_,
                    partitionChannels_,
                    maxOpenWriters_,
                    connectorQueryCtx_->memoryPool(),
                    parquetConfig_->isPartitionPathAsLowerCase(
                        connectorQueryCtx->sessionProperties()))
              : nullptr),
      dataChannels_(
          getNonPartitionChannels(partitionChannels_, inputType_->size())),
      bucketCount_(
          insertTableHandle_->bucketProperty() == nullptr
              ? 0
              : insertTableHandle_->bucketProperty()->bucketCount()),
      bucketFunction_(
          isBucketed() ? createBucketFunction(
                             *insertTableHandle_->bucketProperty(),
                             inputType_)
                       : nullptr),
      writerFactory_(
          dwio::common::getWriterFactory(insertTableHandle_->storageFormat())),
      spillConfig_(connectorQueryCtx->spillConfig()),
      sortWriterFinishTimeSliceLimitMs_(
          getFinishTimeSliceLimitMsFromParquetConfig(
              parquetConfig_,
              connectorQueryCtx->sessionProperties())) {
  if (isBucketed()) {
    VELOX_USER_CHECK_LT(
        bucketCount_, maxBucketCount(), "bucketCount exceeds the limit");
  }
  VELOX_USER_CHECK(
      (commitStrategy_ == CommitStrategy::kNoCommit) ||
          (commitStrategy_ == CommitStrategy::kTaskCommit),
      "Unsupported commit strategy: {}",
      commitStrategyToString(commitStrategy_));

  if (!isBucketed()) {
    return;
  }
  const auto& sortedProperty = insertTableHandle_->bucketProperty()->sortedBy();
  if (!sortedProperty.empty()) {
    sortColumnIndices_.reserve(sortedProperty.size());
    sortCompareFlags_.reserve(sortedProperty.size());
    for (int i = 0; i < sortedProperty.size(); ++i) {
      auto columnIndex =
          getNonPartitionTypes(dataChannels_, inputType_)
              ->getChildIdxIfExists(sortedProperty.at(i)->sortColumn());
      if (columnIndex.has_value()) {
        sortColumnIndices_.push_back(columnIndex.value());
        sortCompareFlags_.push_back(
            {sortedProperty.at(i)->sortOrder().isNullsFirst(),
             sortedProperty.at(i)->sortOrder().isAscending(),
             false,
             CompareFlags::NullHandlingMode::kNullAsValue});
      }
    }
  }
}

void ParquetDataSink::appendData(RowVectorPtr input) {
  checkRunning();

  // Write to unpartitioned (and unbucketed) table.
  if (!isPartitioned() && !isBucketed()) {
    const auto index = ensureWriter(ParquetWriterId::unpartitionedId());
    write(index, input);
    return;
  }

  // Compute partition and bucket numbers.
  computePartitionAndBucketIds(input);

  // Lazy load all the input columns.
  for (column_index_t i = 0; i < input->childrenSize(); ++i) {
    input->childAt(i)->loadedVector();
  }

  // All inputs belong to a single non-bucketed partition. The partition id
  // must be zero.
  if (!isBucketed() && partitionIdGenerator_->numPartitions() == 1) {
    const auto index = ensureWriter(ParquetWriterId{0});
    write(index, input);
    return;
  }

  splitInputRowsAndEnsureWriters();

  for (auto index = 0; index < writers_.size(); ++index) {
    const vector_size_t partitionSize = partitionSizes_[index];
    if (partitionSize == 0) {
      continue;
    }

    RowVectorPtr writerInput = partitionSize == input->size()
        ? input
        : exec::wrap(partitionSize, partitionRows_[index], input);
    write(index, writerInput);
  }
}

void ParquetDataSink::write(size_t index, RowVectorPtr input) {
  WRITER_NON_RECLAIMABLE_SECTION_GUARD(index);
  auto dataInput = makeDataInput(dataChannels_, input);

  writers_[index]->write(dataInput);
  writerInfo_[index]->inputSizeInBytes += dataInput->estimateFlatSize();
  writerInfo_[index]->numWrittenRows += dataInput->size();
}

std::string ParquetDataSink::stateString(State state) {
  switch (state) {
    case State::kRunning:
      return "RUNNING";
    case State::kFinishing:
      return "FLUSHING";
    case State::kClosed:
      return "CLOSED";
    case State::kAborted:
      return "ABORTED";
    default:
      VELOX_UNREACHABLE("BAD STATE: {}", static_cast<int>(state));
  }
}

void ParquetDataSink::computePartitionAndBucketIds(const RowVectorPtr& input) {
  VELOX_CHECK(isPartitioned() || isBucketed());
  if (isPartitioned()) {
    if (!parquetConfig_->allowNullPartitionKeys(
            connectorQueryCtx_->sessionProperties())) {
      // Check that there are no nulls in the partition keys.
      for (auto& partitionIdx : partitionChannels_) {
        auto col = input->childAt(partitionIdx);
        if (col->mayHaveNulls()) {
          for (auto i = 0; i < col->size(); ++i) {
            VELOX_USER_CHECK(
                !col->isNullAt(i),
                "Partition key must not be null: {}",
                input->type()->asRow().nameOf(partitionIdx));
          }
        }
      }
    }
    partitionIdGenerator_->run(input, partitionIds_);
  }

  if (isBucketed()) {
    bucketFunction_->partition(*input, bucketIds_);
  }
}

DataSink::Stats ParquetDataSink::stats() const {
  Stats stats;
  if (state_ == State::kAborted) {
    return stats;
  }

  int64_t numWrittenBytes{0};
  int64_t writeIOTimeUs{0};
  for (const auto& ioStats : ioStats_) {
    numWrittenBytes += ioStats->rawBytesWritten();
    writeIOTimeUs += ioStats->writeIOTimeUs();
  }
  stats.numWrittenBytes = numWrittenBytes;
  stats.writeIOTimeUs = writeIOTimeUs;

  if (state_ != State::kClosed) {
    return stats;
  }

  stats.numWrittenFiles = writers_.size();
  for (int i = 0; i < writerInfo_.size(); ++i) {
    const auto& info = writerInfo_.at(i);
    VELOX_CHECK_NOT_NULL(info);
    const auto spillStats = info->spillStats->rlock();
    if (!spillStats->empty()) {
      stats.spillStats += *spillStats;
    }
  }
  return stats;
}

std::shared_ptr<memory::MemoryPool> ParquetDataSink::createWriterPool(
    const ParquetWriterId& writerId) {
  auto* connectorPool = connectorQueryCtx_->connectorMemoryPool();
  return connectorPool->addAggregateChild(
      fmt::format("{}.{}", connectorPool->name(), writerId.toString()));
}

void ParquetDataSink::setMemoryReclaimers(
    ParquetWriterInfo* writerInfo,
    io::IoStatistics* ioStats) {
  auto* connectorPool = connectorQueryCtx_->connectorMemoryPool();
  if (connectorPool->reclaimer() == nullptr) {
    return;
  }
  writerInfo->writerPool->setReclaimer(
      WriterReclaimer::create(this, writerInfo, ioStats));
  writerInfo->sinkPool->setReclaimer(exec::MemoryReclaimer::create());
  // NOTE: we set the memory reclaimer for sort pool when we construct the sort
  // writer.
}

void ParquetDataSink::setState(State newState) {
  checkStateTransition(state_, newState);
  state_ = newState;
}

/// Validates the state transition from 'oldState' to 'newState'.
void ParquetDataSink::checkStateTransition(State oldState, State newState) {
  switch (oldState) {
    case State::kRunning:
      if (newState == State::kAborted || newState == State::kFinishing) {
        return;
      }
      break;
    case State::kFinishing:
      if (newState == State::kAborted || newState == State::kClosed ||
          // The finishing state is reentry state if we yield in the middle of
          // finish processing if a single run takes too long.
          newState == State::kFinishing) {
        return;
      }
      [[fallthrough]];
    case State::kAborted:
    case State::kClosed:
    default:
      break;
  }
  VELOX_FAIL("Unexpected state transition from {} to {}", oldState, newState);
}

bool ParquetDataSink::finish() {
  // Flush is reentry state.
  setState(State::kFinishing);

  // As for now, only sorted writer needs flush buffered data. For non-sorted
  // writer, data is directly written to the underlying file writer.
  if (!sortWrite()) {
    return true;
  }

  // TODO: we might refactor to move the data sorting logic into parquet data
  // sink.
  const uint64_t startTimeMs = getCurrentTimeMs();
  for (auto i = 0; i < writers_.size(); ++i) {
    WRITER_NON_RECLAIMABLE_SECTION_GUARD(i);
    if (!writers_[i]->finish()) {
      return false;
    }
    if (getCurrentTimeMs() - startTimeMs > sortWriterFinishTimeSliceLimitMs_) {
      return false;
    }
  }
  return true;
}

std::vector<std::string> ParquetDataSink::close() {
  setState(State::kClosed);
  closeInternal();

  std::vector<std::string> partitionUpdates;
  partitionUpdates.reserve(writerInfo_.size());
  for (int i = 0; i < writerInfo_.size(); ++i) {
    const auto& info = writerInfo_.at(i);
    VELOX_CHECK_NOT_NULL(info);
    // clang-format off
      auto partitionUpdateJson = folly::toJson(
       folly::dynamic::object
          ("name", info->writerParameters.partitionName().value_or(""))
          ("updateMode",
            ParquetWriterParameters::updateModeToString(
              info->writerParameters.updateMode()))
          ("writePath", info->writerParameters.writeDirectory())
          ("targetPath", info->writerParameters.targetDirectory())
          ("fileWriteInfos", folly::dynamic::array(
            folly::dynamic::object
              ("writeFileName", info->writerParameters.writeFileName())
              ("targetFileName", info->writerParameters.targetFileName())
              ("fileSize", ioStats_.at(i)->rawBytesWritten())))
          ("rowCount", info->numWrittenRows)
          ("inMemoryDataSizeInBytes", info->inputSizeInBytes)
          ("onDiskDataSizeInBytes", ioStats_.at(i)->rawBytesWritten())
          ("containsNumberedFileNames", true));
    // clang-format on
    partitionUpdates.push_back(partitionUpdateJson);
  }
  return partitionUpdates;
}

void ParquetDataSink::abort() {
  setState(State::kAborted);
  closeInternal();
}

void ParquetDataSink::closeInternal() {
  VELOX_CHECK_NE(state_, State::kRunning);
  VELOX_CHECK_NE(state_, State::kFinishing);

  TestValue::adjust(
      "facebook::velox::connector::parquet::ParquetDataSink::closeInternal",
      this);

  if (state_ == State::kClosed) {
    for (int i = 0; i < writers_.size(); ++i) {
      WRITER_NON_RECLAIMABLE_SECTION_GUARD(i);
      writers_[i]->close();
    }
  } else {
    for (int i = 0; i < writers_.size(); ++i) {
      WRITER_NON_RECLAIMABLE_SECTION_GUARD(i);
      writers_[i]->abort();
    }
  }
}

uint32_t ParquetDataSink::ensureWriter(const ParquetWriterId& id) {
  auto it = writerIndexMap_.find(id);
  if (it != writerIndexMap_.end()) {
    return it->second;
  }
  return appendWriter(id);
}

uint32_t ParquetDataSink::appendWriter(const ParquetWriterId& id) {
  // Check max open writers.
  VELOX_USER_CHECK_LE(
      writers_.size(), maxOpenWriters_, "Exceeded open writer limit");
  VELOX_CHECK_EQ(writers_.size(), writerInfo_.size());
  VELOX_CHECK_EQ(writerIndexMap_.size(), writerInfo_.size());

  std::optional<std::string> partitionName;
  if (isPartitioned()) {
    partitionName =
        partitionIdGenerator_->partitionName(id.partitionId.value());
  }

  // Without explicitly setting flush policy, the default memory based flush
  // policy is used.
  auto writerParameters = getWriterParameters(partitionName, id.bucketId);
  const auto writePath = fs::path(writerParameters.writeDirectory()) /
      writerParameters.writeFileName();
  auto writerPool = createWriterPool(id);
  auto sinkPool = createSinkPool(writerPool);
  std::shared_ptr<memory::MemoryPool> sortPool{nullptr};
  if (sortWrite()) {
    sortPool = createSortPool(writerPool);
  }
  writerInfo_.emplace_back(std::make_shared<ParquetWriterInfo>(
      std::move(writerParameters),
      std::move(writerPool),
      std::move(sinkPool),
      std::move(sortPool)));
  ioStats_.emplace_back(std::make_shared<io::IoStatistics>());
  setMemoryReclaimers(writerInfo_.back().get(), ioStats_.back().get());

  // Take the writer options provided by the user as a starting point, or
  // allocate a new one.
  auto options = insertTableHandle_->writerOptions();
  if (!options) {
    options = writerFactory_->createWriterOptions();
  }

  const auto* connectorSessionProperties =
      connectorQueryCtx_->sessionProperties();

  // Only overwrite options in case they were not already provided.
  if (options->schema == nullptr) {
    options->schema = getNonPartitionTypes(dataChannels_, inputType_);
  }

  if (options->memoryPool == nullptr) {
    options->memoryPool = writerInfo_.back()->writerPool.get();
  }

  if (!options->compressionKind) {
    options->compressionKind = insertTableHandle_->compressionKind();
  }

  if (options->spillConfig == nullptr && canReclaim()) {
    options->spillConfig = spillConfig_;
  }

  if (options->nonReclaimableSection == nullptr) {
    options->nonReclaimableSection =
        writerInfo_.back()->nonReclaimableSectionHolder.get();
  }

  if (options->memoryReclaimerFactory == nullptr ||
      options->memoryReclaimerFactory() == nullptr) {
    options->memoryReclaimerFactory = []() {
      return exec::MemoryReclaimer::create();
    };
  }

  updateWriterOptionsFromParquetConfig(
      insertTableHandle_->storageFormat(),
      parquetConfig_,
      connectorSessionProperties,
      options);

  const auto& sessionTimeZoneName = connectorQueryCtx_->sessionTimezone();
  if (!sessionTimeZoneName.empty()) {
    options->sessionTimezone = tz::locateZone(sessionTimeZoneName);
  }
  options->adjustTimestampToTimezone =
      connectorQueryCtx_->adjustTimestampToTimezone();

  // Prevents the memory allocation during the writer creation.
  WRITER_NON_RECLAIMABLE_SECTION_GUARD(writerInfo_.size() - 1);
  auto writer = writerFactory_->createWriter(
      dwio::common::FileSink::create(
          writePath,
          {
              .bufferWrite = false,
              .connectorProperties = parquetConfig_->config(),
              .fileCreateConfig = parquetConfig_->writeFileCreateConfig(),
              .pool = writerInfo_.back()->sinkPool.get(),
              .metricLogger = dwio::common::MetricsLog::voidLog(),
              .stats = ioStats_.back().get(),
          }),
      options);
  writer = maybeCreateBucketSortWriter(std::move(writer));
  writers_.emplace_back(std::move(writer));
  // Extends the buffer used for partition rows calculations.
  partitionSizes_.emplace_back(0);
  partitionRows_.emplace_back(nullptr);
  rawPartitionRows_.emplace_back(nullptr);

  writerIndexMap_.emplace(id, writers_.size() - 1);
  return writerIndexMap_[id];
}

std::unique_ptr<facebook::velox::dwio::common::Writer>
ParquetDataSink::maybeCreateBucketSortWriter(
    std::unique_ptr<facebook::velox::dwio::common::Writer> writer) {
  if (!sortWrite()) {
    return writer;
  }
  auto* sortPool = writerInfo_.back()->sortPool.get();
  VELOX_CHECK_NOT_NULL(sortPool);
  auto sortBuffer = std::make_unique<exec::SortBuffer>(
      getNonPartitionTypes(dataChannels_, inputType_),
      sortColumnIndices_,
      sortCompareFlags_,
      sortPool,
      writerInfo_.back()->nonReclaimableSectionHolder.get(),
      connectorQueryCtx_->prefixSortConfig(),
      spillConfig_,
      writerInfo_.back()->spillStats.get());
  return std::make_unique<dwio::common::SortingWriter>(
      std::move(writer),
      std::move(sortBuffer),
      parquetConfig_->sortWriterMaxOutputRows(
          connectorQueryCtx_->sessionProperties()),
      parquetConfig_->sortWriterMaxOutputBytes(
          connectorQueryCtx_->sessionProperties()),
      sortWriterFinishTimeSliceLimitMs_);
}

ParquetWriterId ParquetDataSink::getWriterId(size_t row) const {
  std::optional<int32_t> partitionId;
  if (isPartitioned()) {
    VELOX_CHECK_LT(partitionIds_[row], std::numeric_limits<uint32_t>::max());
    partitionId = static_cast<uint32_t>(partitionIds_[row]);
  }

  std::optional<int32_t> bucketId;
  if (isBucketed()) {
    bucketId = bucketIds_[row];
  }
  return ParquetWriterId{partitionId, bucketId};
}

void ParquetDataSink::splitInputRowsAndEnsureWriters() {
  VELOX_CHECK(isPartitioned() || isBucketed());
  if (isBucketed() && isPartitioned()) {
    VELOX_CHECK_EQ(bucketIds_.size(), partitionIds_.size());
  }

  std::fill(partitionSizes_.begin(), partitionSizes_.end(), 0);

  const auto numRows =
      isPartitioned() ? partitionIds_.size() : bucketIds_.size();
  for (auto row = 0; row < numRows; ++row) {
    auto id = getWriterId(row);
    uint32_t index = ensureWriter(id);

    VELOX_DCHECK_LT(index, partitionSizes_.size());
    VELOX_DCHECK_EQ(partitionSizes_.size(), partitionRows_.size());
    VELOX_DCHECK_EQ(partitionRows_.size(), rawPartitionRows_.size());
    if (FOLLY_UNLIKELY(partitionRows_[index] == nullptr) ||
        (partitionRows_[index]->capacity() < numRows * sizeof(vector_size_t))) {
      partitionRows_[index] =
          allocateIndices(numRows, connectorQueryCtx_->memoryPool());
      rawPartitionRows_[index] =
          partitionRows_[index]->asMutable<vector_size_t>();
    }
    rawPartitionRows_[index][partitionSizes_[index]] = row;
    ++partitionSizes_[index];
  }

  for (uint32_t i = 0; i < partitionSizes_.size(); ++i) {
    if (partitionSizes_[i] != 0) {
      VELOX_CHECK_NOT_NULL(partitionRows_[i]);
      partitionRows_[i]->setSize(partitionSizes_[i] * sizeof(vector_size_t));
    }
  }
}

ParquetWriterParameters ParquetDataSink::getWriterParameters(
    const std::optional<std::string>& partition,
    std::optional<uint32_t> bucketId) const {
  auto [targetFileName, writeFileName] = getWriterFileNames(bucketId);

  return ParquetWriterParameters{
      updateMode_,
      partition,
      targetFileName,
      makePartitionDirectory(
          insertTableHandle_->locationHandle()->targetPath(), partition),
      writeFileName,
      makePartitionDirectory(
          insertTableHandle_->locationHandle()->writePath(), partition)};
}

std::pair<std::string, std::string> ParquetDataSink::getWriterFileNames(
    std::optional<uint32_t> bucketId) const {
  auto targetFileName = insertTableHandle_->locationHandle()->targetFileName();
  const bool generateFileName = targetFileName.empty();
  if (bucketId.has_value()) {
    VELOX_CHECK(generateFileName);
    // TODO: add parquet.file_renaming_enabled support.
    targetFileName = computeBucketedFileName(
        connectorQueryCtx_->queryId(), bucketId.value());
  } else if (generateFileName) {
    // targetFileName includes planNodeId and Uuid. As a result, different
    // table writers run by the same task driver or the same table writer
    // run in different task tries would have different targetFileNames.
    targetFileName = fmt::format(
        "{}_{}_{}_{}",
        connectorQueryCtx_->taskId(),
        connectorQueryCtx_->driverId(),
        connectorQueryCtx_->planNodeId(),
        makeUuid());
  }
  VELOX_CHECK(!targetFileName.empty());
  const std::string writeFileName = isCommitRequired()
      ? fmt::format(".tmp.velox.{}_{}", targetFileName, makeUuid())
      : targetFileName;
  if (generateFileName &&
      insertTableHandle_->storageFormat() ==
          dwio::common::FileFormat::PARQUET) {
    return {
        fmt::format("{}{}", targetFileName, ".parquet"),
        fmt::format("{}{}", writeFileName, ".parquet")};
  }
  return {targetFileName, writeFileName};
}

ParquetWriterParameters::UpdateMode ParquetDataSink::getUpdateMode() const {
  if (insertTableHandle_->isExistingTable()) {
    if (insertTableHandle_->isPartitioned()) {
      const auto insertBehavior =
          parquetConfig_->insertExistingPartitionsBehavior(
              connectorQueryCtx_->sessionProperties());
      switch (insertBehavior) {
        case ParquetConfig::InsertExistingPartitionsBehavior::kOverwrite:
          return ParquetWriterParameters::UpdateMode::kOverwrite;
        case ParquetConfig::InsertExistingPartitionsBehavior::kError:
          return ParquetWriterParameters::UpdateMode::kNew;
        default:
          VELOX_UNSUPPORTED(
              "Unsupported insert existing partitions behavior: {}",
              ParquetConfig::insertExistingPartitionsBehaviorString(
                  insertBehavior));
      }
    } else {
      if (insertTableHandle_->isBucketed()) {
        VELOX_USER_FAIL(
            "Cannot insert into bucketed unpartitioned Parquet table");
      }
      if (parquetConfig_->immutablePartitions()) {
        VELOX_USER_FAIL("Unpartitioned Parquet tables are immutable.");
      }
      return ParquetWriterParameters::UpdateMode::kAppend;
    }
  } else {
    return ParquetWriterParameters::UpdateMode::kNew;
  }
}

bool ParquetInsertTableHandle::isPartitioned() const {
  return std::any_of(
      inputColumns_.begin(), inputColumns_.end(), [](auto column) {
        return column->isPartitionKey();
      });
}

const ParquetBucketProperty* ParquetInsertTableHandle::bucketProperty() const {
  return bucketProperty_.get();
}

bool ParquetInsertTableHandle::isBucketed() const {
  return bucketProperty() != nullptr;
}

bool ParquetInsertTableHandle::isExistingTable() const {
  return locationHandle_->tableType() == LocationHandle::TableType::kExisting;
}

folly::dynamic ParquetInsertTableHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "ParquetInsertTableHandle";
  folly::dynamic arr = folly::dynamic::array;
  for (const auto& ic : inputColumns_) {
    arr.push_back(ic->serialize());
  }

  obj["inputColumns"] = arr;
  obj["locationHandle"] = locationHandle_->serialize();
  obj["tableStorageFormat"] = dwio::common::toString(storageFormat_);

  if (bucketProperty_) {
    obj["bucketProperty"] = bucketProperty_->serialize();
  }

  if (compressionKind_.has_value()) {
    obj["compressionKind"] = common::compressionKindToString(*compressionKind_);
  }

  return obj;
}

ParquetInsertTableHandlePtr ParquetInsertTableHandle::create(
    const folly::dynamic& obj) {
  auto inputColumns =
      ISerializable::deserialize<std::vector<ParquetColumnHandle>>(
          obj["inputColumns"]);
  auto locationHandle =
      ISerializable::deserialize<LocationHandle>(obj["locationHandle"]);
  auto storageFormat =
      dwio::common::toFileFormat(obj["tableStorageFormat"].asString());

  std::optional<common::CompressionKind> compressionKind = std::nullopt;
  if (obj.count("compressionKind") > 0) {
    compressionKind =
        common::stringToCompressionKind(obj["compressionKind"].asString());
  }

  std::shared_ptr<const ParquetBucketProperty> bucketProperty;
  if (obj.count("bucketProperty") > 0) {
    bucketProperty = ISerializable::deserialize<ParquetBucketProperty>(
        obj["bucketProperty"]);
  }

  return std::make_shared<ParquetInsertTableHandle>(
      inputColumns, locationHandle, storageFormat, compressionKind);
}

std::string ParquetInsertTableHandle::toString() const {
  std::ostringstream out;
  out << "ParquetInsertTableHandle [" << dwio::common::toString(storageFormat_);
  if (compressionKind_.has_value()) {
    out << " " << common::compressionKindToString(compressionKind_.value());
  } else {
    out << " none";
  }
  out << "], [inputColumns: [";
  for (const auto& i : inputColumns_) {
    out << " " << i->toString();
  }
  out << " ], locationHandle: " << locationHandle_->toString();
  if (bucketProperty_) {
    out << ", bucketProperty: " << bucketProperty_->toString();
  }

  out << "]";
  return out.str();
}

std::string LocationHandle::toString() const {
  return fmt::format(
      "LocationHandle [targetPath: {}, writePath: {}, tableType: {},",
      targetPath_,
      writePath_,
      tableTypeName(tableType_));
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

std::unique_ptr<memory::MemoryReclaimer>
ParquetDataSink::WriterReclaimer::create(
    ParquetDataSink* dataSink,
    ParquetWriterInfo* writerInfo,
    io::IoStatistics* ioStats) {
  return std::unique_ptr<memory::MemoryReclaimer>(
      new ParquetDataSink::WriterReclaimer(dataSink, writerInfo, ioStats));
}

bool ParquetDataSink::WriterReclaimer::reclaimableBytes(
    const memory::MemoryPool& pool,
    uint64_t& reclaimableBytes) const {
  VELOX_CHECK_EQ(pool.name(), writerInfo_->writerPool->name());
  reclaimableBytes = 0;
  if (!dataSink_->canReclaim()) {
    return false;
  }
  return exec::MemoryReclaimer::reclaimableBytes(pool, reclaimableBytes);
}

uint64_t ParquetDataSink::WriterReclaimer::reclaim(
    memory::MemoryPool* pool,
    uint64_t targetBytes,
    uint64_t maxWaitMs,
    memory::MemoryReclaimer::Stats& stats) {
  VELOX_CHECK_EQ(pool->name(), writerInfo_->writerPool->name());
  if (!dataSink_->canReclaim()) {
    return 0;
  }

  if (*writerInfo_->nonReclaimableSectionHolder.get()) {
    RECORD_METRIC_VALUE(kMetricMemoryNonReclaimableCount);
    LOG(WARNING) << "Can't reclaim from parquet writer pool " << pool->name()
                 << " which is under non-reclaimable section, "
                 << " reserved memory: "
                 << succinctBytes(pool->reservedBytes());
    ++stats.numNonReclaimableAttempts;
    return 0;
  }

  const uint64_t memoryUsageBeforeReclaim = pool->reservedBytes();
  const std::string memoryUsageTreeBeforeReclaim = pool->treeMemoryUsage();
  const auto writtenBytesBeforeReclaim = ioStats_->rawBytesWritten();
  const auto reclaimedBytes =
      exec::MemoryReclaimer::reclaim(pool, targetBytes, maxWaitMs, stats);
  const auto earlyFlushedRawBytes =
      ioStats_->rawBytesWritten() - writtenBytesBeforeReclaim;
  addThreadLocalRuntimeStat(
      kEarlyFlushedRawBytes,
      RuntimeCounter(earlyFlushedRawBytes, RuntimeCounter::Unit::kBytes));
  if (earlyFlushedRawBytes > 0) {
    RECORD_METRIC_VALUE(
        kMetricFileWriterEarlyFlushedRawBytes, earlyFlushedRawBytes);
  }
  const uint64_t memoryUsageAfterReclaim = pool->reservedBytes();
  if (memoryUsageAfterReclaim > memoryUsageBeforeReclaim) {
    VELOX_FAIL(
        "Unexpected memory growth after memory reclaim from {}, the memory usage before reclaim: {}, after reclaim: {}\nThe memory tree usage before reclaim:\n{}\nThe memory tree usage after reclaim:\n{}",
        pool->name(),
        succinctBytes(memoryUsageBeforeReclaim),
        succinctBytes(memoryUsageAfterReclaim),
        memoryUsageTreeBeforeReclaim,
        pool->treeMemoryUsage());
  }
  return reclaimedBytes;
}
} // namespace facebook::velox::cudf_velox::connector::parquet
