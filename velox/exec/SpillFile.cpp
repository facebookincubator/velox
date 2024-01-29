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

#include "velox/exec/SpillFile.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::exec {
namespace {
// Spilling currently uses the default PrestoSerializer which by default
// serializes timestamp with millisecond precision to maintain compatibility
// with presto. Since velox's native timestamp implementation supports
// nanosecond precision, we use this serde option to ensure the serializer
// preserves precision.
static const bool kDefaultUseLosslessTimestamp = true;

static int64_t chainBytes(folly::IOBuf& iobuf) {
  int64_t size = 0;
  for (auto& range : iobuf) {
    size += range.size();
  }
  return size;
}
} // namespace

void SpillInputStream::next(bool /*throwIfPastEnd*/) {
  const int32_t readBytes = std::min(size_ - offset_, buffer_->capacity());
  VELOX_CHECK_LT(0, readBytes, "Reading past end of spill file");
  setRange({buffer_->asMutable<uint8_t>(), readBytes, 0});

  uint64_t readTimeUs{0};
  {
    MicrosecondTimer timer(&readTimeUs);
    file_->pread(offset_, readBytes, buffer_->asMutable<char>());
    offset_ += readBytes;
  }
  updateStats(readTimeUs);
}

void SpillInputStream::updateStats(
    uint64_t readTimeUs) {
  auto statsLocked = stats_->wlock();
  ++statsLocked->spillFileReads;
  statsLocked->spillFileReadTimeUs += readTimeUs;
}

std::unique_ptr<SpillWriteFile> SpillWriteFile::create(
    uint32_t id,
    const std::string& pathPrefix,
    const std::string& fileCreateConfig) {
  return std::unique_ptr<SpillWriteFile>(
      new SpillWriteFile(id, pathPrefix, fileCreateConfig));
}

SpillWriteFile::SpillWriteFile(
    uint32_t id,
    const std::string& pathPrefix,
    const std::string& fileCreateConfig)
    : id_(id), path_(fmt::format("{}-{}", pathPrefix, ordinalCounter_++)) {
  auto fs = filesystems::getFileSystem(path_, nullptr);
  file_ = fs->openFileForWrite(
      path_,
      filesystems::FileOptions{
          {{filesystems::FileOptions::kFileCreateConfig.toString(),
            fileCreateConfig}},
          nullptr});
}

void SpillWriteFile::finish() {
  VELOX_CHECK_NOT_NULL(file_);
  size_ = file_->size();
  file_->close();
  file_ = nullptr;
}

uint64_t SpillWriteFile::size() const {
  if (file_ != nullptr) {
    return file_->size();
  }
  return size_;
}

uint64_t SpillWriteFile::write(std::unique_ptr<folly::IOBuf> iobuf) {
  auto writtenBytes = iobuf->computeChainDataLength();
  file_->append(std::move(iobuf));
  return writtenBytes;
}

SpillWriter::SpillWriter(
    const RowTypePtr& type,
    const uint32_t numSortKeys,
    const std::vector<CompareFlags>& sortCompareFlags,
    common::CompressionKind compressionKind,
    const std::string& pathPrefix,
    uint64_t targetFileSize,
    uint64_t writeBufferSize,
    const std::string& fileCreateConfig,
    common::UpdateAndCheckSpillLimitCB& updateAndCheckSpillLimitCb,
    folly::Synchronized<common::SpillStats>* stats)
    : type_(type),
      numSortKeys_(numSortKeys),
      sortCompareFlags_(sortCompareFlags),
      compressionKind_(compressionKind),
      pathPrefix_(pathPrefix),
      targetFileSize_(targetFileSize),
      writeBufferSize_(writeBufferSize),
      fileCreateConfig_(fileCreateConfig),
      updateAndCheckSpillLimitCb_(updateAndCheckSpillLimitCb),
      stats_(stats) {
  // NOTE: if the associated spilling operator has specified the sort
  // comparison flags, then it must match the number of sorting keys.
  VELOX_CHECK(
      sortCompareFlags_.empty() || sortCompareFlags_.size() == numSortKeys_);
}

SpillWriteFile* SpillWriter::ensureFile() {
  if ((currentFile_ != nullptr) && (currentFile_->size() > targetFileSize_)) {
    closeFile();
  }
  if (currentFile_ == nullptr) {
    currentFile_ = SpillWriteFile::create(
        nextFileId_++,
        fmt::format("{}-{}", pathPrefix_, finishedFiles_.size()),
        fileCreateConfig_);
  }
  return currentFile_.get();
}

void SpillWriter::closeFile() {
  if (currentFile_ == nullptr) {
    return;
  }
  currentFile_->finish();
  updateSpilledFileStats(currentFile_->size());
  finishedFiles_.push_back(SpillFileInfo{
      .id = currentFile_->id(),
      .type = type_,
      .path = currentFile_->path(),
      .size = currentFile_->size(),
      .numSortKeys = numSortKeys_,
      .sortFlags = sortCompareFlags_,
      .compressionKind = compressionKind_});
  currentFile_.reset();
}

uint64_t SpillWriter::flush(bool force, memory::MemoryPool* pool) {
  if (batch_ != nullptr) {
    uint64_t flushTimeUS{0};
    uint64_t flushedBytes{0};
    {
      MicrosecondTimer timer(&flushTimeUS);
      IOBufOutputStream out(
          *pool, nullptr, std::max<int64_t>(64 * 1024, batch_->size()));
      batch_->flush(&out);
      auto iobuf = out.getIOBuf();
      flushedBytes = chainBytes(*iobuf);
      bufferBytes_ += flushedBytes;
      if (buffers_ == nullptr) {
        buffers_ = std::move(iobuf);
      } else {
        buffers_->insertAfterThisOne(std::move(iobuf));
      }
    }
    batch_.reset();
  }

  if (bufferBytes_ == 0) {
    return 0;
  }

  if (!force && ((writeBufferSize_ == 0) || (bufferBytes_ < writeBufferSize_))) {
    return 0;
  }

  auto* file = ensureFile();
  VELOX_CHECK_NOT_NULL(file);

  uint64_t writeTimeUs{0};
  uint64_t writtenBytes{0};
  {
    MicrosecondTimer timer(&writeTimeUs);
    writtenBytes = file->write(std::move(buffers_));
    VELOX_CHECK_EQ(bufferBytes_, writtenBytes);
    buffers_ = nullptr;
    bufferBytes_ = 0;
  }
  updateWriteStats(writtenBytes, writeTimeUs);
  updateAndCheckSpillLimitCb_(writtenBytes);
  return writtenBytes;
}

uint64_t SpillWriter::write(
    const RowVectorPtr& rows,
    const folly::Range<IndexRange*>& indices,
    memory::MemoryPool* pool) {
  checkNotFinished();

  uint64_t appendTimeUS{0};
  {
    MicrosecondTimer timer(&appendTimeUS);
    if (batch_ == nullptr) {
      serializer::presto::PrestoVectorSerde::PrestoOptions options = {
          kDefaultUseLosslessTimestamp, compressionKind_};
      batch_ = std::make_unique<VectorStreamGroup>(pool);
      batch_->createStreamTree(
          std::static_pointer_cast<const RowType>(rows->type()),
          1'000,
          &options);
    }
    batch_->append(rows, indices);
  }
  updateAppendStats(rows->size(), appendTimeUS);

  if (batch_->size() < 1 << 20) {
    return 0;
  }
  return flush(false, pool);
}

void SpillWriter::updateAppendStats(
    uint64_t numRows,
    uint64_t serializationTimeUs) {
  auto statsLocked = stats_->wlock();
  statsLocked->spilledRows += numRows;
  statsLocked->spillSerializationTimeUs += serializationTimeUs;
  common::updateGlobalSpillAppendStats(numRows, serializationTimeUs);
}

void SpillWriter::updateWriteStats(
    uint64_t spilledBytes,
    uint64_t fileWriteTimeUs) {
  auto statsLocked = stats_->wlock();
  statsLocked->spilledBytes += spilledBytes;
  statsLocked->spillWriteTimeUs += fileWriteTimeUs;
  ++statsLocked->spillWrites;
  common::updateGlobalSpillWriteStats(spilledBytes, fileWriteTimeUs);
}

void SpillWriter::updateSpilledFileStats(uint64_t fileSize) {
  ++stats_->wlock()->spilledFiles;
  addThreadLocalRuntimeStat(
      "spillFileSize", RuntimeCounter(fileSize, RuntimeCounter::Unit::kBytes));
  common::incrementGlobalSpilledFiles();
}

void SpillWriter::finishFile(memory::MemoryPool* pool) {
  checkNotFinished();
  flush(true, pool);
  closeFile();
  VELOX_CHECK_NULL(currentFile_);
}

SpillFiles SpillWriter::finish(memory::MemoryPool* pool) {
  checkNotFinished();
  auto finishGuard = folly::makeGuard([this]() { finished_ = true; });

  finishFile(pool);
  return std::move(finishedFiles_);
}

std::vector<std::string> SpillWriter::testingSpilledFilePaths() const {
  checkNotFinished();

  std::vector<std::string> spilledFilePaths;
  for (auto& file : finishedFiles_) {
    spilledFilePaths.push_back(file.path);
  }
  if (currentFile_ != nullptr) {
    spilledFilePaths.push_back(currentFile_->path());
  }
  return spilledFilePaths;
}

std::vector<uint32_t> SpillWriter::testingSpilledFileIds() const {
  checkNotFinished();

  std::vector<uint32_t> fileIds;
  for (auto& file : finishedFiles_) {
    fileIds.push_back(file.id);
  }
  if (currentFile_ != nullptr) {
    fileIds.push_back(currentFile_->id());
  }
  return fileIds;
}

std::unique_ptr<SpillReadFile> SpillReadFile::create(
    const SpillFileInfo& fileInfo,
    memory::MemoryPool* pool,
    folly::Synchronized<common::SpillStats>* stats) {
  return std::unique_ptr<SpillReadFile>(new SpillReadFile(
      fileInfo.id,
      fileInfo.path,
      fileInfo.size,
      fileInfo.type,
      fileInfo.numSortKeys,
      fileInfo.sortFlags,
      fileInfo.compressionKind,
      pool,
      stats));
}

SpillReadFile::SpillReadFile(
    uint32_t id,
    const std::string& path,
    uint64_t size,
    const RowTypePtr& type,
    uint32_t numSortKeys,
    const std::vector<CompareFlags>& sortCompareFlags,
    common::CompressionKind compressionKind,
    memory::MemoryPool* pool,
    folly::Synchronized<common::SpillStats>* stats)
    : id_(id),
      path_(path),
      size_(size),
      type_(type),
      numSortKeys_(numSortKeys),
      sortCompareFlags_(sortCompareFlags),
      compressionKind_(compressionKind),
      readOptions_{kDefaultUseLosslessTimestamp, compressionKind_},
      pool_(pool),
      stats_(stats) {
  constexpr uint64_t kMaxReadBufferSize =
      (1 << 20) - AlignedBuffer::kPaddedSize; // 1MB - padding.
  auto fs = filesystems::getFileSystem(path_, nullptr);
  auto file = fs->openFileForRead(path_);
  auto buffer = AlignedBuffer::allocate<char>(
      std::min<uint64_t>(size_, kMaxReadBufferSize), pool_);
  input_ =
      std::make_unique<SpillInputStream>(std::move(file), std::move(buffer), stats_);
}

bool SpillReadFile::nextBatch(RowVectorPtr& rowVector) {
  if (input_->atEnd()) {
    return false;
  }
  uint64_t spillReadTimeUs{0};
  {
    MicrosecondTimer timer(&spillReadTimeUs);
    VectorStreamGroup::read(
        input_.get(), pool_, type_, &rowVector, &readOptions_);
  }
  updateStats();
  return true;
}

void SpillReadFile::updateStats(uint64_t readTimeUs) {
  auto statsLocked = stats_->wlock();
  ++statsLocked->spillBatchReads;
  statsLocked->spillBatchReadTimeUs += readTimeUs;
}
} // namespace facebook::velox::exec
