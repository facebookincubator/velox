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

#include "velox/connectors/hive/FileDataSink.h"

#include "velox/common/testutil/TestValue.h"
#include "velox/exec/OperatorUtils.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::connector::hive {
namespace {
#define WRITER_NON_RECLAIMABLE_SECTION_GUARD(index)       \
  memory::NonReclaimableSectionGuard nonReclaimableGuard( \
      writerInfo_[(index)]->nonReclaimableSectionHolder.get())

std::shared_ptr<memory::MemoryPool> createSinkPool(
    const std::shared_ptr<memory::MemoryPool>& writerPool) {
  return writerPool->addLeafChild(fmt::format("{}.sink", writerPool->name()));
}

std::shared_ptr<memory::MemoryPool> createSortPool(
    const std::shared_ptr<memory::MemoryPool>& writerPool) {
  return writerPool->addLeafChild(fmt::format("{}.sort", writerPool->name()));
}
} // namespace

const WriterId& WriterId::unpartitionedId() {
  static const WriterId writerId{0};
  return writerId;
}

std::string WriterId::toString() const {
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

RowTypePtr FileDataSink::getNonPartitionTypes(
    const std::vector<column_index_t>& dataCols,
    const RowTypePtr& inputType) {
  std::vector<std::string> childNames;
  std::vector<TypePtr> childTypes;
  const auto& dataSize = dataCols.size();
  childNames.reserve(dataSize);
  childTypes.reserve(dataSize);
  for (auto dataCol : dataCols) {
    childNames.push_back(inputType->nameOf(dataCol));
    childTypes.push_back(inputType->childAt(dataCol));
  }

  return ROW(std::move(childNames), std::move(childTypes));
}

RowVectorPtr FileDataSink::makeDataInput(
    const std::vector<column_index_t>& dataCols,
    const RowVectorPtr& input) {
  std::vector<VectorPtr> childVectors;
  childVectors.reserve(dataCols.size());
  for (auto dataCol : dataCols) {
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

FileDataSink::FileDataSink(
    RowTypePtr inputType,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    dwio::common::FileFormat storageFormat,
    uint32_t maxOpenWriters,
    std::vector<column_index_t> partitionChannels,
    std::vector<column_index_t> dataChannels,
    int32_t bucketCount,
    std::unique_ptr<core::PartitionFunction> bucketFunction,
    std::unique_ptr<PartitionIdGenerator> partitionIdGenerator,
    std::shared_ptr<dwio::common::WriterFactory> writerFactory,
    uint64_t maxTargetFileBytes,
    bool partitionKeyAsLowerCase,
    const common::SpillConfig* spillConfig,
    uint64_t sortWriterFinishTimeSliceLimitMs)
    : inputType_(std::move(inputType)),
      connectorQueryCtx_(connectorQueryCtx),
      commitStrategy_(commitStrategy),
      storageFormat_(storageFormat),
      maxOpenWriters_(maxOpenWriters),
      partitionChannels_(std::move(partitionChannels)),
      partitionIdGenerator_(std::move(partitionIdGenerator)),
      dataChannels_(std::move(dataChannels)),
      bucketCount_(bucketCount),
      bucketFunction_(std::move(bucketFunction)),
      writerFactory_(std::move(writerFactory)),
      spillConfig_(spillConfig),
      sortWriterFinishTimeSliceLimitMs_(sortWriterFinishTimeSliceLimitMs),
      maxTargetFileBytes_(maxTargetFileBytes),
      partitionKeyAsLowerCase_(partitionKeyAsLowerCase) {
  fileSystemStats_ = std::make_unique<IoStats>();
}

void FileDataSink::appendData(RowVectorPtr input) {
  checkRunning();

  // Lazy load all the input columns.
  input->loadedVector();

  // Write to unpartitioned (and unbucketed) table.
  if (!isPartitioned() && !isBucketed()) {
    const auto index = ensureWriter(WriterId::unpartitionedId());
    write(index, input);
    return;
  }

  // Compute partition and bucket numbers.
  computePartitionAndBucketIds(input);

  // All inputs belong to a single non-bucketed partition. The partition id
  // must be zero.
  if (!isBucketed() && partitionIdGenerator_->numPartitions() == 1) {
    const auto index = ensureWriter(WriterId{0});
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

void FileDataSink::write(size_t index, RowVectorPtr input) {
  WRITER_NON_RECLAIMABLE_SECTION_GUARD(index);
  auto dataInput = makeDataInput(dataChannels_, input);

  if (writers_[index] == nullptr) {
    writers_[index] = createWriterForIndex(index);
  }

  writers_[index]->write(dataInput);
  writerInfo_[index]->inputSizeInBytes += dataInput->estimateFlatSize();
  writerInfo_[index]->numWrittenRows += dataInput->size();
  writerInfo_[index]->currentFileWrittenRows += dataInput->size();

  // File rotation is not supported for bucketed tables (require one file per
  // bucket with predictable name) or sorted writes (SortingWriter not
  // recreated).
  if (maxTargetFileBytes_ == 0 || isBucketed() || sortWrite()) {
    return;
  }

  const auto currentFileBytes = getCurrentFileBytes(index);
  if (currentFileBytes >= maxTargetFileBytes_) {
    rotateWriter(index);
  }
}

uint64_t FileDataSink::getCurrentFileBytes(size_t writerIndex) const {
  VELOX_CHECK_LT(writerIndex, ioStats_.size());
  VELOX_CHECK_LT(writerIndex, writerInfo_.size());
  const auto totalBytes = ioStats_[writerIndex]->rawBytesWritten();
  const auto baselineBytes = writerInfo_[writerIndex]->cumulativeWrittenBytes;
  // Sanity check: total should always be >= baseline since ioStats is
  // never reset and cumulative is a snapshot of rawBytesWritten at rotation.
  VELOX_DCHECK_GE(totalBytes, baselineBytes);
  return totalBytes - baselineBytes;
}

void FileDataSink::finalizeWriterFile(size_t index) {
  VELOX_CHECK_LT(index, writerInfo_.size());
  VELOX_CHECK_LT(index, ioStats_.size());

  auto& info = writerInfo_[index];

  // Capture current file stats AFTER close to include footer bytes.
  const auto currentFileBytes = getCurrentFileBytes(index);

  // Finalize the current file into writtenFiles using the stored names.
  if (currentFileBytes > 0) {
    FileInfo fileInfo;
    fileInfo.writeFileName = info->currentWriteFileName;
    fileInfo.targetFileName = info->currentTargetFileName;
    fileInfo.fileSize = currentFileBytes;
    fileInfo.numRows = info->currentFileWrittenRows;
    // Reset for next file.
    info->currentFileWrittenRows = 0;
    info->writtenFiles.push_back(std::move(fileInfo));
  }

  // Update cumulative stats as a snapshot of total stats so far.
  // This becomes the baseline for the next file.
  info->cumulativeWrittenBytes = ioStats_[index]->rawBytesWritten();
}

void FileDataSink::rotateWriter(size_t index) {
  VELOX_CHECK_LT(index, writers_.size());
  VELOX_CHECK_LT(index, writerInfo_.size());

  auto& info = writerInfo_[index];

  // Close the writer first to flush all data including footer.
  writers_[index]->close();

  // Finalize the current file state.
  finalizeWriterFile(index);

  // Release old writer's memory pools. The new writer will be created lazily
  // on the next write to avoid creating empty files.
  writers_[index].reset();

  ++info->fileSequenceNumber;
}

std::string FileDataSink::stateString(State state) {
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

DataSink::Stats FileDataSink::stats() const {
  Stats stats;
  if (state_ == State::kAborted) {
    return stats;
  }

  for (const auto& ioStats : ioStats_) {
    stats.numWrittenBytes += ioStats->rawBytesWritten();
    stats.writeIOTimeUs += ioStats->writeIOTimeUs();
  }

  if (state_ != State::kClosed) {
    return stats;
  }

  // Count total files written, including rotated files.
  stats.numWrittenFiles = 0;
  for (size_t i = 0; i < writerInfo_.size(); ++i) {
    const auto& info = writerInfo_.at(i);
    VELOX_CHECK_NOT_NULL(info);
    stats.numWrittenFiles += info->writtenFiles.size();
    if (!info->spillStats->empty()) {
      stats.spillStats += *info->spillStats;
    }
  }
  return stats;
}

std::unordered_map<std::string, RuntimeCounter> FileDataSink::runtimeStats()
    const {
  std::unordered_map<std::string, RuntimeCounter> runtimeStats;

  const auto fsStatsMap = fileSystemStats_->stats();
  for (const auto& [statName, statValue] : fsStatsMap) {
    runtimeStats.emplace(
        statName, RuntimeCounter(statValue.sum, statValue.unit));
  }

  return runtimeStats;
}

std::shared_ptr<memory::MemoryPool> FileDataSink::createWriterPool(
    const WriterId& writerId) {
  auto* connectorPool = connectorQueryCtx_->connectorMemoryPool();
  return connectorPool->addAggregateChild(
      fmt::format("{}.{}", connectorPool->name(), writerId.toString()));
}

void FileDataSink::setMemoryReclaimers(
    WriterInfo* /*writerInfo*/,
    io::IoStatistics* /*ioStats*/) {
  // Default no-op. Subclasses override to set up format-specific reclaimers.
}

void FileDataSink::setState(State newState) {
  checkStateTransition(state_, newState);
  state_ = newState;
}

void FileDataSink::checkStateTransition(State oldState, State newState) {
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

bool FileDataSink::finish() {
  // Flush is reentry state.
  setState(State::kFinishing);

  // As for now, only sorted writer needs flush buffered data. For non-sorted
  // writer, data is directly written to the underlying file writer.
  if (!sortWrite()) {
    return true;
  }

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

std::vector<std::string> FileDataSink::close() {
  setState(State::kClosed);
  closeInternal();
  return commitMessage();
}

void FileDataSink::abort() {
  setState(State::kAborted);
  closeInternal();
}

void FileDataSink::closeInternal() {
  VELOX_CHECK_NE(state_, State::kRunning);
  VELOX_CHECK_NE(state_, State::kFinishing);

  TestValue::adjust(
      "facebook::velox::connector::hive::FileDataSink::closeInternal", this);

  // NOTE: writers_[i] can be nullptr during file rotation. In rotateWriter(),
  // we call writers_[index].reset() to release the old writer before creating
  // a new one. If an error occurs during new writer creation, or if abort is
  // called during this window, the writer slot may be empty.
  if (state_ == State::kClosed) {
    for (int i = 0; i < writers_.size(); ++i) {
      if (writers_[i] == nullptr) {
        continue;
      }
      WRITER_NON_RECLAIMABLE_SECTION_GUARD(i);
      writers_[i]->close();
      finalizeWriterFile(i);
    }
  } else {
    for (int i = 0; i < writers_.size(); ++i) {
      if (writers_[i] == nullptr) {
        continue;
      }
      WRITER_NON_RECLAIMABLE_SECTION_GUARD(i);
      writers_[i]->abort();
    }
  }
}

uint32_t FileDataSink::ensureWriter(const WriterId& id) {
  auto it = writerIndexMap_.find(id);
  if (it != writerIndexMap_.end()) {
    return it->second;
  }
  return appendWriter(id);
}

uint32_t FileDataSink::appendWriter(const WriterId& id) {
  // Check max open writers.
  VELOX_USER_CHECK_LE(
      writers_.size(), maxOpenWriters_, "Exceeded open writer limit");
  VELOX_CHECK_EQ(writers_.size(), writerInfo_.size());
  VELOX_CHECK_EQ(writerIndexMap_.size(), writerInfo_.size());

  std::optional<std::string> partitionName;
  if (isPartitioned()) {
    partitionName = getPartitionName(id.partitionId.value());
  }

  // Without explicitly setting flush policy, the default memory based flush
  // policy is used.
  auto writerParameters = getWriterParameters(partitionName, id.bucketId);
  auto writerPool = createWriterPool(id);
  auto sinkPool = createSinkPool(writerPool);
  std::shared_ptr<memory::MemoryPool> sortPool{nullptr};
  if (sortWrite()) {
    sortPool = createSortPool(writerPool);
  }
  writerInfo_.emplace_back(
      std::make_shared<WriterInfo>(
          std::move(writerParameters),
          std::move(writerPool),
          std::move(sinkPool),
          std::move(sortPool)));
  ioStats_.emplace_back(std::make_unique<io::IoStatistics>());

  setMemoryReclaimers(writerInfo_.back().get(), ioStats_.back().get());
  writers_.emplace_back(createWriterForIndex(writerInfo_.size() - 1));
  addThreadLocalRuntimeStat(
      fmt::format("{}WriterCount", dwio::common::toString(storageFormat_)),
      RuntimeCounter(1));
  // Extends the buffer used for partition rows calculations.
  partitionSizes_.emplace_back(0);
  partitionRows_.emplace_back(nullptr);
  rawPartitionRows_.emplace_back(nullptr);

  writerIndexMap_.emplace(id, writers_.size() - 1);
  return writerIndexMap_[id];
}

WriterId FileDataSink::getWriterId(vector_size_t row) const {
  std::optional<uint32_t> partitionId;
  if (isPartitioned()) {
    VELOX_CHECK_LT(partitionIds_[row], std::numeric_limits<uint32_t>::max());
    partitionId = static_cast<uint32_t>(partitionIds_[row]);
  }

  std::optional<uint32_t> bucketId;
  if (isBucketed()) {
    bucketId = bucketIds_[row];
  }
  return WriterId{partitionId, bucketId};
}

void FileDataSink::updatePartitionRows(
    uint32_t index,
    vector_size_t numRows,
    vector_size_t row) {
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

void FileDataSink::splitInputRowsAndEnsureWriters() {
  VELOX_CHECK(isPartitioned() || isBucketed());
  if (isBucketed() && isPartitioned()) {
    VELOX_CHECK_EQ(bucketIds_.size(), partitionIds_.size());
  }

  std::fill(partitionSizes_.begin(), partitionSizes_.end(), 0);

  const auto numRows = static_cast<vector_size_t>(
      isPartitioned() ? partitionIds_.size() : bucketIds_.size());
  for (vector_size_t row = 0; row < numRows; ++row) {
    const auto id = getWriterId(row);
    const uint32_t index = ensureWriter(id);
    updatePartitionRows(index, numRows, row);
  }

  for (uint32_t i = 0; i < partitionSizes_.size(); ++i) {
    if (partitionSizes_[i] != 0) {
      VELOX_CHECK_NOT_NULL(partitionRows_[i]);
      partitionRows_[i]->setSize(partitionSizes_[i] * sizeof(vector_size_t));
    }
  }
}

} // namespace facebook::velox::connector::hive
