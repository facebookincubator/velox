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

#include "velox/exec/Spill.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/serializers/PrestoSerializer.h"

namespace facebook::velox::exec {
namespace {
// Spilling currently uses the default PrestoSerializer which by default
// serializes timestamp with millisecond precision to maintain compatibility
// with presto. Since velox's native timestamp implementation supports
// nanosecond precision, we use this serde option to ensure the serializer
// preserves precision.
static const bool kDefaultUseLosslessTimestamp = true;

std::vector<folly::Synchronized<SpillStats>>& allSpillStats() {
  static std::vector<folly::Synchronized<SpillStats>> spillStatsList(
      std::thread::hardware_concurrency());
  return spillStatsList;
}

folly::Synchronized<SpillStats>& localSpillStats() {
  const auto idx = std::hash<std::thread::id>{}(std::this_thread::get_id());
  auto& spillStatsVector = allSpillStats();
  return spillStatsVector[idx % spillStatsVector.size()];
}
} // namespace

std::atomic<int32_t> SpillFile::ordinalCounter_;

void SpillInput::next(bool /*throwIfPastEnd*/) {
  int32_t readBytes = std::min(input_->size() - offset_, buffer_->capacity());
  VELOX_CHECK_LT(0, readBytes, "Reading past end of spill file");
  setRange({buffer_->asMutable<uint8_t>(), readBytes, 0});
  input_->pread(offset_, readBytes, buffer_->asMutable<char>());
  offset_ += readBytes;
}

void SpillMergeStream::pop() {
  if (++index_ >= size_) {
    setNextBatch();
  }
}

int32_t SpillMergeStream::compare(const MergeStream& other) const {
  auto& otherStream = static_cast<const SpillMergeStream&>(other);
  auto& children = rowVector_->children();
  auto& otherChildren = otherStream.current().children();
  int32_t key = 0;
  if (sortCompareFlags().empty()) {
    do {
      auto result = children[key]
                        ->compare(
                            otherChildren[key].get(),
                            index_,
                            otherStream.index_,
                            CompareFlags())
                        .value();
      if (result != 0) {
        return result;
      }
    } while (++key < numSortingKeys());
  } else {
    do {
      auto result = children[key]
                        ->compare(
                            otherChildren[key].get(),
                            index_,
                            otherStream.index_,
                            sortCompareFlags()[key])
                        .value();
      if (result != 0) {
        return result;
      }
    } while (++key < numSortingKeys());
  }
  return 0;
}

SpillFile::SpillFile(
    RowTypePtr type,
    int32_t numSortingKeys,
    const std::vector<CompareFlags>& sortCompareFlags,
    const std::string& path,
    common::CompressionKind compressionKind,
    memory::MemoryPool* pool)
    : type_(std::move(type)),
      numSortingKeys_(numSortingKeys),
      sortCompareFlags_(sortCompareFlags),
      ordinal_(ordinalCounter_++),
      path_(fmt::format("{}-{}", path, ordinal_)),
      compressionKind_(compressionKind),
      pool_(pool) {
  // NOTE: if the spilling operator has specified the sort comparison flags,
  // then it must match the number of sorting keys.
  VELOX_CHECK(
      sortCompareFlags_.empty() || sortCompareFlags_.size() == numSortingKeys_);
}

WriteFile& SpillFile::output() {
  if (!output_) {
    auto fs = filesystems::getFileSystem(path_, nullptr);
    output_ = fs->openFileForWrite(path_);
  }
  return *output_;
}

void SpillFile::startRead() {
  constexpr uint64_t kMaxReadBufferSize =
      (1 << 20) - AlignedBuffer::kPaddedSize; // 1MB - padding.
  VELOX_CHECK(!output_);
  VELOX_CHECK(!input_);
  auto fs = filesystems::getFileSystem(path_, nullptr);
  auto file = fs->openFileForRead(path_);
  auto buffer = AlignedBuffer::allocate<char>(
      std::min<uint64_t>(fileSize_, kMaxReadBufferSize), pool_);
  input_ = std::make_unique<SpillInput>(std::move(file), std::move(buffer));
}

bool SpillFile::nextBatch(RowVectorPtr& rowVector) {
  if (input_->atEnd()) {
    return false;
  }
  serializer::presto::PrestoVectorSerde::PrestoOptions options = {
      kDefaultUseLosslessTimestamp, compressionKind_};
  VectorStreamGroup::read(input_.get(), pool_, type_, &rowVector, &options);
  return true;
}

SpillFileList::SpillFileList(
    const RowTypePtr& type,
    int32_t numSortingKeys,
    const std::vector<CompareFlags>& sortCompareFlags,
    const std::string& path,
    uint64_t targetFileSize,
    common::CompressionKind compressionKind,
    memory::MemoryPool* pool,
    folly::Synchronized<SpillStats>* stats)
    : type_(type),
      numSortingKeys_(numSortingKeys),
      sortCompareFlags_(sortCompareFlags),
      path_(path),
      targetFileSize_(targetFileSize),
      compressionKind_(compressionKind),
      pool_(pool),
      stats_(stats) {
  // NOTE: if the associated spilling operator has specified the sort
  // comparison flags, then it must match the number of sorting keys.
  VELOX_CHECK(
      sortCompareFlags_.empty() || sortCompareFlags_.size() == numSortingKeys_);
}

WriteFile& SpillFileList::currentOutput() {
  if (files_.empty() || !files_.back()->isWritable() ||
      files_.back()->size() > targetFileSize_) {
    if (!files_.empty() && files_.back()->isWritable()) {
      files_.back()->finishWrite();
      updateSpilledFiles(files_.back()->size());
    }
    files_.push_back(std::make_unique<SpillFile>(
        type_,
        numSortingKeys_,
        sortCompareFlags_,
        fmt::format("{}-{}", path_, files_.size()),
        compressionKind_,
        pool_));
  }
  return files_.back()->output();
}

uint64_t SpillFileList::flush() {
  uint64_t writtenBytes = 0;
  if (batch_) {
    IOBufOutputStream out(
        *pool_, nullptr, std::max<int64_t>(64 * 1024, batch_->size()));
    uint64_t flushTimeUs{0};
    {
      MicrosecondTimer timer(&flushTimeUs);
      batch_->flush(&out);
    }

    batch_.reset();
    auto iobuf = out.getIOBuf();
    auto& file = currentOutput();
    uint64_t writeTimeUs{0};
    uint32_t numDiskWrites{0};
    {
      MicrosecondTimer timer(&writeTimeUs);
      for (auto& range : *iobuf) {
        ++numDiskWrites;
        file.append(std::string_view(
            reinterpret_cast<const char*>(range.data()), range.size()));
        writtenBytes += range.size();
      }
    }
    updateWriteStats(numDiskWrites, writtenBytes, flushTimeUs, writeTimeUs);
  }
  return writtenBytes;
}

uint64_t SpillFileList::write(
    const RowVectorPtr& rows,
    const folly::Range<IndexRange*>& indices) {
  uint64_t timeUs{0};
  {
    MicrosecondTimer timer(&timeUs);
    if (batch_ == nullptr) {
      serializer::presto::PrestoVectorSerde::PrestoOptions options = {
          kDefaultUseLosslessTimestamp, compressionKind_};
      batch_ = std::make_unique<VectorStreamGroup>(pool_);
      batch_->createStreamTree(
          std::static_pointer_cast<const RowType>(rows->type()),
          1000,
          &options);
    }
    batch_->append(rows, indices);
  }
  updateAppendStats(rows->size(), timeUs);
  return flush();
}

void SpillFileList::updateAppendStats(
    uint64_t numRows,
    uint64_t serializationTimeUs) {
  auto statsLocked = stats_->wlock();
  statsLocked->spilledRows += numRows;
  statsLocked->spillSerializationTimeUs += serializationTimeUs;
  updateGlobalSpillAppendStats(numRows, serializationTimeUs);
}

void SpillFileList::updateWriteStats(
    uint32_t numDiskWrites,
    uint64_t spilledBytes,
    uint64_t flushTimeUs,
    uint64_t fileWriteTimeUs) {
  auto statsLocked = stats_->wlock();
  statsLocked->spilledBytes += spilledBytes;
  statsLocked->spillFlushTimeUs += flushTimeUs;
  statsLocked->spillWriteTimeUs += fileWriteTimeUs;
  statsLocked->spillDiskWrites += numDiskWrites;
  updateGlobalSpillWriteStats(
      numDiskWrites, spilledBytes, flushTimeUs, fileWriteTimeUs);
}

void SpillFileList::updateSpilledFiles(uint64_t fileSize) {
  ++stats_->wlock()->spilledFiles;
  addThreadLocalRuntimeStat(
      "spillFileSize", RuntimeCounter(fileSize, RuntimeCounter::Unit::kBytes));
  incrementGlobalSpilledFiles();
}

void SpillFileList::finishFile() {
  flush();
  if (files_.empty()) {
    return;
  }
  if (files_.back()->isWritable()) {
    files_.back()->finishWrite();
    updateSpilledFiles(files_.back()->size());
  }
}

std::vector<std::string> SpillFileList::testingSpilledFilePaths() const {
  std::vector<std::string> spilledFiles;
  for (auto& file : files_) {
    spilledFiles.push_back(file->testingFilePath());
  }
  return spilledFiles;
}

SpillState::SpillState(
    const std::string& path,
    int32_t maxPartitions,
    int32_t numSortingKeys,
    const std::vector<CompareFlags>& sortCompareFlags,
    uint64_t targetFileSize,
    common::CompressionKind compressionKind,
    memory::MemoryPool* pool,
    folly::Synchronized<SpillStats>* stats)
    : path_(path),
      maxPartitions_(maxPartitions),
      numSortingKeys_(numSortingKeys),
      sortCompareFlags_(sortCompareFlags),
      targetFileSize_(targetFileSize),
      compressionKind_(compressionKind),
      pool_(pool),
      stats_(stats),
      files_(maxPartitions_) {}

void SpillState::setPartitionSpilled(int32_t partition) {
  VELOX_DCHECK_LT(partition, maxPartitions_);
  VELOX_DCHECK_LT(spilledPartitionSet_.size(), maxPartitions_);
  VELOX_DCHECK(!spilledPartitionSet_.contains(partition));
  spilledPartitionSet_.insert(partition);
  ++stats_->wlock()->spilledPartitions;
  incrementGlobalSpilledPartitionStats();
}

void SpillState::updateSpilledInputBytes(uint64_t bytes) {
  auto statsLocked = stats_->wlock();
  statsLocked->spilledInputBytes += bytes;
  updateGlobalSpillMemoryBytes(bytes);
}

uint64_t SpillState::appendToPartition(
    int32_t partition,
    const RowVectorPtr& rows) {
  VELOX_CHECK(
      isPartitionSpilled(partition), "Partition {} is not spilled", partition);
  // Ensure that partition exist before writing.
  if (!files_.at(partition)) {
    files_[partition] = std::make_unique<SpillFileList>(
        std::static_pointer_cast<const RowType>(rows->type()),
        numSortingKeys_,
        sortCompareFlags_,
        fmt::format("{}-spill-{}", path_, partition),
        targetFileSize_,
        compressionKind_,
        pool_,
        stats_);
  }
  updateSpilledInputBytes(rows->estimateFlatSize());

  IndexRange range{0, rows->size()};
  return files_[partition]->write(rows, folly::Range<IndexRange*>(&range, 1));
}

std::unique_ptr<TreeOfLosers<SpillMergeStream>> SpillState::startMerge(
    int32_t partition,
    std::unique_ptr<SpillMergeStream>&& extra) {
  VELOX_CHECK_LT(partition, files_.size());
  std::vector<std::unique_ptr<SpillMergeStream>> result;
  auto list = std::move(files_[partition]);
  if (list != nullptr) {
    for (auto& file : list->files()) {
      result.push_back(FileSpillMergeStream::create(std::move(file)));
    }
  }
  VELOX_DCHECK_EQ(!result.empty(), isPartitionSpilled(partition));
  if (extra != nullptr) {
    result.push_back(std::move(extra));
  }
  // Check if the partition is empty or not.
  if (FOLLY_UNLIKELY(result.empty())) {
    return nullptr;
  }
  return std::make_unique<TreeOfLosers<SpillMergeStream>>(std::move(result));
}

SpillFiles SpillState::files(int32_t partition) {
  VELOX_CHECK_LT(partition, files_.size());

  auto list = std::move(files_[partition]);
  if (list == nullptr) {
    return {};
  }
  return list->files();
}

const SpillPartitionNumSet& SpillState::spilledPartitionSet() const {
  return spilledPartitionSet_;
}

std::vector<std::string> SpillState::testingSpilledFilePaths() const {
  std::vector<std::string> spilledFiles;
  for (const auto& list : files_) {
    if (list != nullptr) {
      const auto spilledFilesFromList = list->testingSpilledFilePaths();
      spilledFiles.insert(
          spilledFiles.end(),
          spilledFilesFromList.begin(),
          spilledFilesFromList.end());
    }
  }
  return spilledFiles;
}

std::vector<std::unique_ptr<SpillPartition>> SpillPartition::split(
    int numShards) {
  const int32_t numFilesPerShard = bits::roundUp(files_.size(), numShards);
  std::vector<std::unique_ptr<SpillPartition>> shards(numShards);

  for (int shard = 0, fileIdx = 0; shard < numShards; ++shard) {
    SpillFiles shardFiles;
    shardFiles.reserve(numFilesPerShard);
    while (shardFiles.size() < numFilesPerShard && fileIdx < files_.size()) {
      shardFiles.push_back(std::move(files_[fileIdx++]));
    }
    shards[shard] =
        std::make_unique<SpillPartition>(id_, std::move(shardFiles));
  }
  files_.clear();
  return shards;
}

std::unique_ptr<UnorderedStreamReader<BatchStream>>
SpillPartition::createReader() {
  std::vector<std::unique_ptr<BatchStream>> streams;
  streams.reserve(files_.size());
  for (auto& file : files_) {
    streams.push_back(FileSpillBatchStream::create(std::move(file)));
  }
  files_.clear();
  return std::make_unique<UnorderedStreamReader<BatchStream>>(
      std::move(streams));
}

SpillStats::SpillStats(
    uint64_t _spillRuns,
    uint64_t _spilledInputBytes,
    uint64_t _spilledBytes,
    uint64_t _spilledRows,
    uint32_t _spilledPartitions,
    uint64_t _spilledFiles,
    uint64_t _spillFillTimeUs,
    uint64_t _spillSortTimeUs,
    uint64_t _spillSerializationTimeUs,
    uint64_t _spillDiskWrites,
    uint64_t _spillFlushTimeUs,
    uint64_t _spillWriteTimeUs)
    : spillRuns(_spillRuns),
      spilledInputBytes(_spilledInputBytes),
      spilledBytes(_spilledBytes),
      spilledRows(_spilledRows),
      spilledPartitions(_spilledPartitions),
      spilledFiles(_spilledFiles),
      spillFillTimeUs(_spillFillTimeUs),
      spillSortTimeUs(_spillSortTimeUs),
      spillSerializationTimeUs(_spillSerializationTimeUs),
      spillDiskWrites(_spillDiskWrites),
      spillFlushTimeUs(_spillFlushTimeUs),
      spillWriteTimeUs(_spillWriteTimeUs) {}

SpillStats& SpillStats::operator+=(const SpillStats& other) {
  spillRuns += other.spillRuns;
  spilledInputBytes += other.spilledInputBytes;
  spilledBytes += other.spilledBytes;
  spilledRows += other.spilledRows;
  spilledPartitions += other.spilledPartitions;
  spilledFiles += other.spilledFiles;
  spillFillTimeUs += other.spillFillTimeUs;
  spillSortTimeUs += other.spillSortTimeUs;
  spillSerializationTimeUs += other.spillSerializationTimeUs;
  spillDiskWrites += other.spillDiskWrites;
  spillFlushTimeUs += other.spillFlushTimeUs;
  spillWriteTimeUs += other.spillWriteTimeUs;
  return *this;
}

SpillStats SpillStats::operator-(const SpillStats& other) const {
  SpillStats result;
  result.spillRuns = spillRuns - other.spillRuns;
  result.spilledInputBytes = spilledInputBytes - other.spilledInputBytes;
  result.spilledBytes = spilledBytes - other.spilledBytes;
  result.spilledRows = spilledRows - other.spilledRows;
  result.spilledPartitions = spilledPartitions - other.spilledPartitions;
  result.spilledFiles = spilledFiles - other.spilledFiles;
  result.spillFillTimeUs = spillFillTimeUs - other.spillFillTimeUs;
  result.spillSortTimeUs = spillSortTimeUs - other.spillSortTimeUs;
  result.spillSerializationTimeUs =
      spillSerializationTimeUs - other.spillSerializationTimeUs;
  result.spillDiskWrites = spillDiskWrites - other.spillDiskWrites;
  result.spillFlushTimeUs = spillFlushTimeUs - other.spillFlushTimeUs;
  result.spillWriteTimeUs = spillWriteTimeUs - other.spillWriteTimeUs;
  return result;
}

bool SpillStats::operator<(const SpillStats& other) const {
  uint32_t eqCount{0};
  uint32_t gtCount{0};
  uint32_t ltCount{0};
#define UPDATE_COUNTER(counter)           \
  do {                                    \
    if (counter < other.counter) {        \
      ++ltCount;                          \
    } else if (counter > other.counter) { \
      ++gtCount;                          \
    } else {                              \
      ++eqCount;                          \
    }                                     \
  } while (0);

  UPDATE_COUNTER(spillRuns);
  UPDATE_COUNTER(spilledInputBytes);
  UPDATE_COUNTER(spilledBytes);
  UPDATE_COUNTER(spilledRows);
  UPDATE_COUNTER(spilledPartitions);
  UPDATE_COUNTER(spilledFiles);
  UPDATE_COUNTER(spillFillTimeUs);
  UPDATE_COUNTER(spillSortTimeUs);
  UPDATE_COUNTER(spillSerializationTimeUs);
  UPDATE_COUNTER(spillDiskWrites);
  UPDATE_COUNTER(spillFlushTimeUs);
  UPDATE_COUNTER(spillWriteTimeUs);
#undef UPDATE_COUNTER
  VELOX_CHECK(
      !((gtCount > 0) && (ltCount > 0)),
      "gtCount {} ltCount {}",
      gtCount,
      ltCount);
  return ltCount > 0;
}

bool SpillStats::operator>(const SpillStats& other) const {
  return !(*this < other) && (*this != other);
}

bool SpillStats::operator>=(const SpillStats& other) const {
  return !(*this < other);
}

bool SpillStats::operator<=(const SpillStats& other) const {
  return !(*this > other);
}

bool SpillStats::operator==(const SpillStats& other) const {
  return std::tie(
             spillRuns,
             spilledInputBytes,
             spilledBytes,
             spilledRows,
             spilledPartitions,
             spilledFiles,
             spillFillTimeUs,
             spillSortTimeUs,
             spillSerializationTimeUs,
             spillDiskWrites,
             spillFlushTimeUs,
             spillWriteTimeUs) ==
      std::tie(
             other.spillRuns,
             other.spilledInputBytes,
             other.spilledBytes,
             other.spilledRows,
             other.spilledPartitions,
             other.spilledFiles,
             other.spillFillTimeUs,
             other.spillSortTimeUs,
             other.spillSerializationTimeUs,
             other.spillDiskWrites,
             other.spillFlushTimeUs,
             other.spillWriteTimeUs);
}

void SpillStats::reset() {
  spillRuns = 0;
  spilledInputBytes = 0;
  spilledBytes = 0;
  spilledRows = 0;
  spilledPartitions = 0;
  spilledFiles = 0;
  spillFillTimeUs = 0;
  spillSortTimeUs = 0;
  spillSerializationTimeUs = 0;
  spillDiskWrites = 0;
  spillFlushTimeUs = 0;
  spillWriteTimeUs = 0;
}

std::string SpillStats::toString() const {
  return fmt::format(
      "spillRuns[{}] spilledInputBytes[{}] spilledBytes[{}] spilledRows[{}] spilledPartitions[{}] spilledFiles[{}] spillFillTimeUs[{}] spillSortTime[{}] spillSerializationTime[{}] spillDiskWrites[{}] spillFlushTime[{}] spillWriteTime[{}]",
      spillRuns,
      succinctBytes(spilledInputBytes),
      succinctBytes(spilledBytes),
      spilledRows,
      spilledPartitions,
      spilledFiles,
      succinctMicros(spillFillTimeUs),
      succinctMicros(spillSortTimeUs),
      succinctMicros(spillSerializationTimeUs),
      spillDiskWrites,
      succinctMicros(spillFlushTimeUs),
      succinctMicros(spillWriteTimeUs));
}

SpillPartitionIdSet toSpillPartitionIdSet(
    const SpillPartitionSet& partitionSet) {
  SpillPartitionIdSet partitionIdSet;
  partitionIdSet.reserve(partitionSet.size());
  for (auto& partitionEntry : partitionSet) {
    partitionIdSet.insert(partitionEntry.first);
  }
  return partitionIdSet;
}

void updateGlobalSpillRunStats(uint64_t numRuns) {
  auto statsLocked = localSpillStats().wlock();
  statsLocked->spillRuns += numRuns;
}

void updateGlobalSpillAppendStats(
    uint64_t numRows,
    uint64_t serializationTimeUs) {
  auto statsLocked = localSpillStats().wlock();
  statsLocked->spilledRows += numRows;
  statsLocked->spillSerializationTimeUs += serializationTimeUs;
}

void incrementGlobalSpilledPartitionStats() {
  ++localSpillStats().wlock()->spilledPartitions;
}

void updateGlobalSpillFillTime(uint64_t timeUs) {
  localSpillStats().wlock()->spillFillTimeUs += timeUs;
}

void updateGlobalSpillSortTime(uint64_t timeUs) {
  localSpillStats().wlock()->spillSortTimeUs += timeUs;
}

void updateGlobalSpillWriteStats(
    uint32_t numDiskWrites,
    uint64_t spilledBytes,
    uint64_t flushTimeUs,
    uint64_t writeTimeUs) {
  auto statsLocked = localSpillStats().wlock();
  statsLocked->spillDiskWrites += numDiskWrites;
  statsLocked->spilledBytes += spilledBytes;
  statsLocked->spillFlushTimeUs += flushTimeUs;
  statsLocked->spillWriteTimeUs += writeTimeUs;
}

void updateGlobalSpillMemoryBytes(uint64_t spilledInputBytes) {
  auto statsLocked = localSpillStats().wlock();
  statsLocked->spilledInputBytes += spilledInputBytes;
}

void incrementGlobalSpilledFiles() {
  ++localSpillStats().wlock()->spilledFiles;
}

SpillStats globalSpillStats() {
  SpillStats gSpillStats;
  for (auto& spillStats : allSpillStats()) {
    gSpillStats += spillStats.copy();
  }
  return gSpillStats;
}
} // namespace facebook::velox::exec
