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

namespace facebook::velox::serializer {
std::unique_ptr<SerializedPageFile> SerializedPageFile::create(
    uint32_t id,
    const std::string& pathPrefix,
    const std::string& fileCreateConfig,
    filesystems::File::IoStats* fsStats) {
  return std::unique_ptr<SerializedPageFile>(
      new SerializedPageFile(id, pathPrefix, fileCreateConfig, fsStats));
}

SerializedPageFile::SerializedPageFile(
    uint32_t id,
    const std::string& pathPrefix,
    const std::string& fileCreateConfig,
    filesystems::File::IoStats* fsStats)
    : id_(id), path_(fmt::format("{}-{}", pathPrefix, ordinalCounter_++)) {
  auto fs = filesystems::getFileSystem(path_, nullptr);
  file_ = fs->openFileForWrite(
      path_,
      filesystems::FileOptions{
          {{filesystems::FileOptions::kFileCreateConfig.toString(),
            fileCreateConfig}},
          nullptr,
          std::nullopt,
          false,
          true,
          true,
          std::nullopt,
          fsStats});
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
    filesystems::File::IoStats* fsStats)
    : pathPrefix_(pathPrefix),
      targetFileSize_(targetFileSize),
      writeBufferSize_(writeBufferSize),
      fileCreateConfig_(fileCreateConfig),
      serdeOptions_(std::move(serdeOptions)),
      pool_(pool),
      serde_(serde),
      fsStats_(fsStats) {}

SerializedPageFile* SerializedPageFileWriter::ensureFile() {
  if ((currentFile_ != nullptr) && (currentFile_->size() > targetFileSize_)) {
    closeFile();
  }
  if (currentFile_ == nullptr) {
    currentFile_ = SerializedPageFile::create(
        nextFileId_++,
        fmt::format("{}-{}", pathPrefix_, finishedFiles_.size()),
        fileCreateConfig_,
        fsStats_);
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
    NanosecondTimer timer(&flushTimeNs);
    batch_->flush(&out);
  }
  batch_.reset();

  uint64_t writeTimeNs{0};
  uint64_t writtenBytes{0};
  auto iobuf = out.getIOBuf();
  {
    NanosecondTimer timer(&writeTimeNs);
    writtenBytes = file->write(std::move(iobuf));
  }
  updateWriteStats(writtenBytes, flushTimeNs, writeTimeNs);
  return writtenBytes;
}

uint64_t SerializedPageFileWriter::write(
    const RowVectorPtr& rows,
    const folly::Range<IndexRange*>& indices) {
  if (rows == nullptr || rows->size() == 0 || indices.size() == 0) {
    return 0;
  }

  checkNotFinished();

  uint64_t timeNs{0};
  {
    NanosecondTimer timer(&timeNs);
    if (batch_ == nullptr) {
      batch_ = std::make_unique<VectorStreamGroup>(pool_, serde_);
      batch_->createStreamTree(
          std::static_pointer_cast<const RowType>(rows->type()),
          1'000,
          serdeOptions_.get());
    }
    batch_->append(rows, indices);
  }
  updateAppendStats(rows->size(), timeNs);
  if (batch_->size() < writeBufferSize_) {
    return 0;
  }
  return flush();
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
    filesystems::File::IoStats* fsStats)
    : readOptions_(std::move(readOptions)),
      pool_(pool),
      serde_(serde),
      type_(type) {
  auto fs = filesystems::getFileSystem(path, nullptr);
  auto file =
      fs->openFileForRead(path, filesystems::FileOptions{.stats = fsStats});
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
    NanosecondTimer timer{&timeNs};
    VectorStreamGroup::read(
        input_.get(), pool_, type_, serde_, &rowVector, readOptions_.get());
  }
  updateSerializationTimeStats(timeNs);
  return true;
}
} // namespace facebook::velox::serializer
