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

#include "velox/common/file/FileStream.h"

namespace facebook::velox::common {

FileReadStream::FileReadStream(
    std::unique_ptr<ReadFile>&& file,
    memory::MemoryPool* pool)
    : file_(std::move(file)), fileSize_(file_->size()), pool_(pool) {
  VELOX_CHECK_NOT_NULL(pool_);
  VELOX_CHECK_GT(fileSize_, 0, "Empty FileInputStream");
}

size_t FileReadStream::size() const {
  return fileSize_;
}

bool FileReadStream::atEnd() const {
  return tellp() >= fileSize_;
}

std::streampos FileReadStream::tellp() const {
  return fileOffset_;
}

void FileReadStream::seekp(std::streampos position) {
  static_assert(sizeof(std::streamsize) <= sizeof(uint64_t));
  VELOX_CHECK_GE(
      fileSize_,
      static_cast<uint64_t>(position),
      "Seek past the end of input file");
  fileOffset_ = position;
}

void FileReadStream::skip(int32_t size) {
  VELOX_CHECK_GE(size, 0, "Attempting to skip negative number of bytes");
  VELOX_CHECK_LE(
      size,
      remainingSize(),
      "Skip past the end of FileReadStream: {}",
      fileSize_);
  fileOffset_ += size;
}

size_t FileReadStream::remainingSize() const {
  VELOX_CHECK_LE(fileSize_ - fileOffset_, std::numeric_limits<size_t>::max());
  return fileSize_ - fileOffset_;
}

uint8_t FileReadStream::readByte() {
  VELOX_CHECK_GT(
      remainingSize(), 0, "Read past the end of input file {}", fileSize_);
  uint8_t byte;
  readBytes(&byte, 1);
  return byte;
}

void FileReadStream::readBytes(uint8_t* bytes, int32_t size) {
  VELOX_CHECK_GE(size, 0, "Attempting to read negative number of bytes");
  VELOX_CHECK_LE(
      size, remainingSize(), "Read past the end of input file {}", fileSize_);
  uint64_t readTimeNs{0};
  {
    NanosecondTimer timer{&readTimeNs};
    file_->pread(fileOffset_, size, bytes);
  }
  updateStats(size, readTimeNs);
  fileOffset_ += size;
}

std::string_view FileReadStream::nextView(int32_t /*size*/) {
  VELOX_UNSUPPORTED("nextView is not supported by FileReadStream");
}

std::string FileReadStream::toString() const {
  return fmt::format(
      "file (offset {}/size {})",
      succinctBytes(fileOffset_),
      succinctBytes(fileSize_));
}

FileReadStream::Stats FileReadStream::stats() const {
  return stats_;
}

void FileReadStream::updateStats(uint64_t readBytes, uint64_t readTimeNs) {
  stats_.readBytes += readBytes;
  stats_.readTimeNs += readTimeNs;
  ++stats_.numReads;
}

bool FileReadStream::Stats::operator==(
    const FileReadStream::Stats& other) const {
  return std::tie(numReads, readBytes, readTimeNs) ==
      std::tie(other.numReads, other.readBytes, other.readTimeNs);
}

std::string FileReadStream::Stats::toString() const {
  return fmt::format(
      "numReads: {}, readBytes: {}, readTimeNs: {}",
      numReads,
      succinctBytes(readBytes),
      succinctMicros(readTimeNs));
}

FileInputStream::FileInputStream(
    std::unique_ptr<ReadFile>&& file,
    uint64_t bufferSize,
    memory::MemoryPool* pool)
    : FileReadStream(std::move(file), pool),
      bufferSize_(std::min(fileSize_, bufferSize)),
      readAheadEnabled_((bufferSize_ < fileSize_) && file_->hasPreadvAsync()) {
  VELOX_CHECK_NOT_NULL(pool_);
  VELOX_CHECK_GT(fileSize_, 0, "Empty FileInputStream");

  buffers_.push_back(AlignedBuffer::allocate<char>(bufferSize_, pool_));
  if (readAheadEnabled_) {
    buffers_.push_back(AlignedBuffer::allocate<char>(bufferSize_, pool_));
  }
  readNextRange();
}

FileInputStream::~FileInputStream() {
  if (!readAheadWait_.valid()) {
    return;
  }
  try {
    readAheadWait_.wait();
  } catch (const std::exception& ex) {
    // ignore any prefetch error when query has failed.
    LOG(WARNING) << "FileInputStream read-ahead failed on destruction "
                 << ex.what();
  }
}

void FileInputStream::readNextRange() {
  VELOX_CHECK(current_ == nullptr || current_->availableBytes() == 0);
  ranges_.clear();
  current_ = nullptr;

  int32_t readBytes{0};
  uint64_t readTimeNs{0};
  {
    NanosecondTimer timer{&readTimeNs};
    if (readAheadWait_.valid()) {
      readBytes = std::move(readAheadWait_)
                      .via(&folly::QueuedImmediateExecutor::instance())
                      .wait()
                      .value();
      VELOX_CHECK(!readAheadWait_.valid());
      VELOX_CHECK_LT(
          0, readBytes, "Read past end of FileInputStream {}", fileSize_);
      advanceBuffer();
    } else {
      readBytes = readSize();
      VELOX_CHECK_LT(
          0, readBytes, "Read past end of FileInputStream {}", fileSize_);
      NanosecondTimer timer_2{&readTimeNs};
      file_->pread(fileOffset_, readBytes, buffer()->asMutable<char>());
    }
  }

  ranges_.resize(1);
  ranges_[0] = {buffer()->asMutable<uint8_t>(), readBytes, 0};
  current_ = ranges_.data();
  fileOffset_ += readBytes;

  updateStats(readBytes, readTimeNs);

  maybeIssueReadahead();
}

size_t FileInputStream::size() const {
  return fileSize_;
}

bool FileInputStream::atEnd() const {
  return tellp() >= fileSize_;
}

std::streampos FileInputStream::tellp() const {
  if (current_ == nullptr) {
    VELOX_CHECK_EQ(fileOffset_, fileSize_);
    return fileOffset_;
  }
  return fileOffset_ - current_->availableBytes();
}

void FileInputStream::seekp(std::streampos position) {
  static_assert(sizeof(std::streamsize) <= sizeof(int64_t));
  const int64_t seekPos = position;
  const int64_t curPos = tellp();
  VELOX_CHECK_GE(
      seekPos, curPos, "Backward seek is not supported by FileInputStream");

  const int64_t toSkip = seekPos - curPos;
  if (toSkip == 0) {
    return;
  }
  doSeek(toSkip);
}

void FileInputStream::skip(int32_t size) {
  doSeek(size);
}

void FileInputStream::doSeek(int64_t skipBytes) {
  VELOX_CHECK_GE(skipBytes, 0, "Attempting to skip negative number of bytes");
  if (skipBytes == 0) {
    return;
  }

  VELOX_CHECK_LE(
      skipBytes,
      remainingSize(),
      "Skip past the end of FileInputStream: {}",
      fileSize_);

  for (;;) {
    const int64_t skippedBytes =
        std::min<int64_t>(current_->availableBytes(), skipBytes);
    skipBytes -= skippedBytes;
    current_->position += skippedBytes;
    if (skipBytes == 0) {
      return;
    }
    readNextRange();
  }
}

size_t FileInputStream::remainingSize() const {
  return fileSize_ - tellp();
}

uint8_t FileInputStream::readByte() {
  VELOX_CHECK_GT(
      remainingSize(), 0, "Read past the end of input file {}", fileSize_);

  if (current_->availableBytes() > 0) {
    return current_->buffer[current_->position++];
  }
  readNextRange();
  return readByte();
}

void FileInputStream::readBytes(uint8_t* bytes, int32_t size) {
  VELOX_CHECK_GE(size, 0, "Attempting to read negative number of bytes");
  if (size == 0) {
    return;
  }

  VELOX_CHECK_LE(
      size, remainingSize(), "Read past the end of input file {}", fileSize_);

  int32_t offset{0};
  for (;;) {
    const int32_t readBytes =
        std::min<int64_t>(current_->availableBytes(), size);
    simd::memcpy(
        bytes + offset, current_->buffer + current_->position, readBytes);
    offset += readBytes;
    size -= readBytes;
    current_->position += readBytes;
    if (size == 0) {
      return;
    }
    readNextRange();
  }
}

std::string_view FileInputStream::nextView(int32_t size) {
  VELOX_CHECK_GE(size, 0, "Attempting to view negative number of bytes");
  if (remainingSize() == 0) {
    return std::string_view(nullptr, 0);
  }

  if (current_->availableBytes() == 0) {
    readNextRange();
  }

  VELOX_CHECK_GT(current_->availableBytes(), 0);
  const auto position = current_->position;
  const auto viewSize = std::min<int64_t>(current_->availableBytes(), size);
  current_->position += viewSize;
  return std::string_view(
      reinterpret_cast<char*>(current_->buffer) + position, viewSize);
}

uint64_t FileInputStream::readSize() const {
  return std::min(fileSize_ - fileOffset_, bufferSize_);
}

void FileInputStream::maybeIssueReadahead() {
  VELOX_CHECK(!readAheadWait_.valid());
  if (!readAheadEnabled_) {
    return;
  }
  const auto size = readSize();
  if (size == 0) {
    return;
  }
  std::vector<folly::Range<char*>> ranges;
  ranges.emplace_back(nextBuffer()->asMutable<char>(), size);
  readAheadWait_ = file_->preadvAsync(fileOffset_, ranges);
  VELOX_CHECK(readAheadWait_.valid());
}

std::string FileInputStream::toString() const {
  return fmt::format(
      "file (offset {}/size {}) current (position {}/ size {})",
      succinctBytes(fileOffset_),
      succinctBytes(fileSize_),
      current_ == nullptr ? "NULL" : succinctBytes(current_->position),
      current_ == nullptr ? "NULL" : succinctBytes(current_->size));
}
} // namespace facebook::velox::common
