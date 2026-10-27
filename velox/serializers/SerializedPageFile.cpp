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

#include "velox/serializers/SerializedPageFile.h"
#include <cstdint>
#include <memory>
#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TestValue.h"

namespace facebook::velox::serializer {
std::unique_ptr<SerializedPageFile> SerializedPageFile::create(
    uint32_t id,
    const std::string& pathPrefix,
    const std::string& fileCreateConfig,
    IoStats* ioStats) {
  return std::unique_ptr<SerializedPageFile>(
      new SerializedPageFile(id, pathPrefix, fileCreateConfig, ioStats));
}

SerializedPageFile::SerializedPageFile(
    uint32_t id,
    const std::string& pathPrefix,
    const std::string& fileCreateConfig,
    IoStats* ioStats)
    : id_(id), path_(fmt::format("{}-{}", pathPrefix, ordinalCounter_++)) {
  auto fs = filesystems::getFileSystem(path_, nullptr);
  file_ = fs->openFileForWrite(
      path_,
      filesystems::FileOptions{
          .values =
              {{filesystems::FileOptions::kFileCreateConfig.toString(),
                fileCreateConfig}},
          .stats = ioStats});
}

void SerializedPageFile::finish() {
  VELOX_CHECK_NOT_NULL(file_);
  size_ = file_->size();
  file_->close();
  file_ = nullptr;
}

uint64_t SerializedPageFile::size() {
  if (file_ != nullptr) {
    return file_->size();
  }
  return size_;
}

uint64_t SerializedPageFile::write(std::unique_ptr<folly::IOBuf> iobuf) {
  auto writtenBytes = iobuf->computeChainDataLength();
  file_->append(std::move(iobuf));
  return writtenBytes;
}

SerializedPageFileWriter::SerializedPageFileWriter(
    const std::string& pathPrefix,
    uint64_t targetFileSize,
    uint64_t writeBufferSize,
    const std::string& fileCreateConfig,
    std::unique_ptr<VectorSerde::Options> serdeOptions,
    VectorSerde* serde,
    memory::MemoryPool* pool,
    IoStats* ioStats)
    : pathPrefix_(pathPrefix),
      targetFileSize_(targetFileSize),
      writeBufferSize_(writeBufferSize),
      fileCreateConfig_(fileCreateConfig),
      serdeOptions_(std::move(serdeOptions)),
      pool_(pool),
      serde_(serde),
      ioStats_(ioStats) {}

SerializedPageFile* SerializedPageFileWriter::ensureFile() {
  if ((currentFile_ != nullptr) && (currentFile_->size() > targetFileSize_)) {
    closeFile();
  }
  if (currentFile_ == nullptr) {
    currentFile_ = SerializedPageFile::create(
        nextFileId_++,
        fmt::format("{}-{}", pathPrefix_, finishedFiles_.size()),
        fileCreateConfig_,
        ioStats_);
  }
  return currentFile_.get();
}

void SerializedPageFileWriter::closeFile() {
  if (currentFile_ == nullptr) {
    return;
  }
  currentFile_->finish();
  const auto& fileInfo = currentFile_->fileInfo();
  updateFileStats(fileInfo);
  finishedFiles_.push_back(fileInfo);
  currentFile_.reset();
}

size_t SerializedPageFileWriter::numFinishedFiles() const {
  return finishedFiles_.size();
}

uint64_t SerializedPageFileWriter::flush() {
  if (batch_ == nullptr) {
    return 0;
  }

  auto* file = ensureFile();
  VELOX_CHECK_NOT_NULL(file);

  IOBufOutputStream out(
      *pool_, nullptr, std::max<int64_t>(64 * 1024, batch_->size()));
  uint64_t flushTimeNs{0};
  {
    NanosecondWallTimer timer(&flushTimeNs);
    batch_->flush(&out);
  }
  batch_.reset();

  uint64_t writeTimeNs{0};
  uint64_t writtenBytes{0};
  auto iobuf = out.getIOBuf();
  {
    NanosecondWallTimer timer(&writeTimeNs);
    writtenBytes = file->write(std::move(iobuf));
  }
  updateWriteStats(writtenBytes, flushTimeNs, writeTimeNs);
  return writtenBytes;
}

folly::Range<const vector_size_t*> SerializedPageFileWriter::estimateRowSizes(
    const RowVectorPtr& rows,
    const folly::Range<IndexRange*>& indices) {
  vector_size_t numRows = 0;
  for (const auto& range : indices) {
    numRows += range.size;
  }
  if (numRows > rowSizes_.size()) {
    rowNumbers_.resize(numRows);
    rowSizes_.resize(numRows);
    sizePointers_.resize(numRows);
    std::iota(sizePointers_.begin(), sizePointers_.end(), rowSizes_.data());
  }

  vector_size_t rowIndex = 0;
  for (const auto& range : indices) {
    for (vector_size_t offset = 0; offset < range.size; ++offset) {
      rowNumbers_[rowIndex++] = range.begin + offset;
    }
  }
  std::fill(rowSizes_.begin(), rowSizes_.begin() + numRows, 0);
  Scratch scratch;
  serde_->estimateSerializedSize(
      rows.get(),
      folly::Range(rowNumbers_.data(), numRows),
      sizePointers_.data(),
      scratch);
  common::testutil::TestValue::adjust(
      "facebook::velox::serializer::SerializedPageFileWriter::estimateRowSizes",
      &rowSizes_);
  return folly::Range<const vector_size_t*>(rowSizes_.data(), numRows);
}

namespace {
// Returns true if 'type' or any type nested in it is OPAQUE, whose serialized
// size the serde can't estimate.
bool containsOpaque(const Type& type) {
  if (type.isOpaque()) {
    return true;
  }
  for (uint32_t i = 0; i < type.size(); ++i) {
    if (containsOpaque(*type.childAt(i))) {
      return true;
    }
  }
  return false;
}
} // namespace

bool SerializedPageFileWriter::canEstimateWrites(const Type& type) const {
  return writeBufferSize_ > 0 && !containsOpaque(type);
}

uint64_t SerializedPageFileWriter::write(
    const RowVectorPtr& rows,
    const folly::Range<IndexRange*>& indices) {
  if (rows == nullptr || rows->size() == 0 || indices.size() == 0) {
    return 0;
  }

  checkNotFinished();

  uint64_t timeNs{0};
  auto append = [&](vector_size_t begin, vector_size_t size) {
    IndexRange range{begin, size};
    NanosecondWallTimer timer(&timeNs);
    if (batch_ == nullptr) {
      batch_ = std::make_unique<VectorStreamGroup>(pool_, serde_);
      batch_->createStreamTree(
          std::static_pointer_cast<const RowType>(rows->type()),
          1'000,
          serdeOptions_.get());
    }
    batch_->append(rows, folly::Range(&range, 1));
  };

  uint64_t writtenBytes{0};
  if (!canEstimateWrites(*rows->type())) {
    for (const auto& range : indices) {
      append(range.begin, range.size);
    }
    if (batch_->size() >= writeBufferSize_) {
      writtenBytes += flush();
    }
  } else {
    const auto rowSizes = estimateRowSizes(rows, indices);
    // The serializer can't write a row of 2GB or more, and appending one can
    // overflow its offsets. The serde caps a row estimate at INT32_MAX.
    for (const auto rowSize : rowSizes) {
      VELOX_CHECK_LT(
          rowSize,
          std::numeric_limits<int32_t>::max(),
          "Serialized row exceeds 2GB");
    }
    uint64_t bufferedSize = batch_ == nullptr ? 0 : batch_->size();
    vector_size_t rowIndex = 0;
    for (const auto& range : indices) {
      vector_size_t chunkBegin = range.begin;
      vector_size_t chunkSize = 0;
      uint64_t chunkBytes = 0;
      if (range.size > 0 && bufferedSize > 0 &&
          bufferedSize + rowSizes[rowIndex] > writeBufferSize_) {
        writtenBytes += flush();
        bufferedSize = 0;
      }
      for (vector_size_t offset = 0; offset < range.size; ++offset) {
        const auto rowSize = rowSizes[rowIndex++];
        if (chunkSize > 0 &&
            bufferedSize + chunkBytes + rowSize > writeBufferSize_) {
          append(chunkBegin, chunkSize);
          writtenBytes += flush();
          bufferedSize = 0;
          chunkBegin += chunkSize;
          chunkSize = 0;
          chunkBytes = 0;
        }

        ++chunkSize;
        chunkBytes += rowSize;
      }
      if (chunkSize > 0) {
        append(chunkBegin, chunkSize);
        bufferedSize = batch_->size();
        if (bufferedSize >= writeBufferSize_) {
          writtenBytes += flush();
          bufferedSize = 0;
        }
      }
    }
  }

  updateAppendStats(rows->size(), timeNs);
  return writtenBytes;
}

void SerializedPageFileWriter::finishFile() {
  checkNotFinished();
  flush();
  closeFile();
  VELOX_CHECK_NULL(currentFile_);
}

std::vector<SerializedPageFile::FileInfo> SerializedPageFileWriter::finish() {
  checkNotFinished();
  auto finishGuard = folly::makeGuard([this]() { finished_ = true; });

  finishFile();
  return std::move(finishedFiles_);
}

SerializedPageFileReader::SerializedPageFileReader(
    const std::string& path,
    uint64_t bufferSize,
    const RowTypePtr& type,
    VectorSerde* serde,
    std::unique_ptr<VectorSerde::Options> readOptions,
    memory::MemoryPool* pool,
    IoStats* ioStats)
    : readOptions_(std::move(readOptions)),
      pool_(pool),
      serde_(serde),
      type_(type) {
  auto fs = filesystems::getFileSystem(path, nullptr);
  auto file =
      fs->openFileForRead(path, filesystems::FileOptions{.stats = ioStats});
  input_ = std::make_unique<common::FileInputStream>(
      std::move(file), bufferSize, pool_);
}

bool SerializedPageFileReader::nextBatch(RowVectorPtr& rowVector) {
  if (input_->atEnd()) {
    updateFinalStats();
    return false;
  }

  uint64_t timeNs{0};
  {
    NanosecondWallTimer timer{&timeNs};
    VectorStreamGroup::read(
        input_.get(), pool_, type_, serde_, &rowVector, readOptions_.get());
  }
  updateSerializationTimeStats(timeNs);
  return true;
}
} // namespace facebook::velox::serializer
