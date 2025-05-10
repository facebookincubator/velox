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

#include "velox/exec/SerializedPageSpiller.h"

namespace facebook::velox::exec {
namespace {
// A wrapper around a write file that provides buffer capability.
class BufferedWriteFile {
 public:
  BufferedWriteFile(
      std::unique_ptr<WriteFile> writeFile,
      uint64_t flushThresholdBytes,
      memory::MemoryPool* pool)
      : flushThresholdBytes_(flushThresholdBytes),
        pool_(pool),
        writeFile_(std::move(writeFile)),
        bufferStream_(std::make_unique<IOBufOutputStream>(*pool_)) {}

  ~BufferedWriteFile() {
    close();
  }

  void append(char* payload, int64_t bytes) {
    VELOX_CHECK_NOT_NULL(bufferStream_);
    if (flushThresholdBytes_ == 0) {
      // Bypass the copy if no buffer is intended.
      writeFile_->append(std::string_view(payload, bytes));
      return;
    }
    bufferStream_->write(payload, bytes);
    bufferBytes_ += bytes;
    checkFlush();
  }

  void append(std::unique_ptr<folly::IOBuf>&& iobuf) {
    VELOX_CHECK_NOT_NULL(bufferStream_);
    if (flushThresholdBytes_ == 0) {
      // Bypass the copy if no buffer is intended.
      writeFile_->append(std::move(iobuf));
      return;
    }
    for (auto range = iobuf->begin(); range != iobuf->end(); range++) {
      bufferStream_->write(
          reinterpret_cast<const char*>(range->data()), range->size());
    }
    bufferBytes_ += iobuf->computeChainDataLength();
    checkFlush();
  }

  void close() {
    auto iobuf = bufferStream_->getIOBuf();
    if (iobuf->computeChainDataLength() != 0) {
      writeFile_->append(std::move(iobuf));
    }
    writeFile_->close();
    bufferStream_.reset();
  }

 private:
  void checkFlush() {
    if (bufferBytes_ < flushThresholdBytes_) {
      return;
    }
    writeFile_->append(bufferStream_->getIOBuf());
    bufferStream_ = std::make_unique<IOBufOutputStream>(*pool_);
    bufferBytes_ = 0;
  }

  const uint64_t flushThresholdBytes_;
  memory::MemoryPool* const pool_;

  std::unique_ptr<WriteFile> writeFile_;
  std::unique_ptr<IOBufOutputStream> bufferStream_;
  uint64_t bufferBytes_{0};
};
} // namespace

void SerializedPageSpiller::spill(
    const std::vector<std::shared_ptr<SerializedPage>>* pages) {
  if (pages->empty()) {
    return;
  }

  auto [path, writeFile] = nextSpillWriteFile();
  BufferedWriteFile bufferedWriteFile(
      std::move(writeFile), writeBufferSize_, pool_);
  spillFilePaths_.push_back(path);

  // Spill file layout:
  //  --- Page 0 ---
  //
  //  (1 Byte) is null page
  //  (8 Bytes) payload size
  //  (1 Bytes) has num rows
  //  (8 Bytes) num rows
  //  (x Bytes) payload
  //
  //  --- Page 1 ---
  //        ...
  //  --- Page n ---
  //        ...
  const auto totalBytesBeforeSpill = totalBytes_;
  uint64_t totalRows{0};
  uint64_t spilledBytes{0};
  uint64_t spillWriteTimeNs{0};
  {
    NanosecondTimer timer(&spillWriteTimeNs);
    for (auto& page : *pages) {
      if (page != nullptr) {
        const auto pageSize = page->size();
        totalBytes_ += pageSize;
        totalRows += page->numRows().value_or(0);
      }

      // Spill payload headers.
      uint8_t isNull = (page == nullptr) ? 1 : 0;
      bufferedWriteFile.append(
          reinterpret_cast<char*>(&isNull), sizeof(uint8_t));
      totalBytes_ += sizeof(uint8_t);
      if (page == nullptr) {
        continue;
      }

      int64_t pageBytes = 0;
      if (page != nullptr) {
        pageBytes = page->size();
      }
      bufferedWriteFile.append(
          reinterpret_cast<char*>(&pageBytes), sizeof(int64_t));
      totalBytes_ += sizeof(int64_t);

      auto numRowsOpt = page->numRows();
      uint8_t hasNumRows = numRowsOpt.has_value() ? 1 : 0;
      bufferedWriteFile.append(reinterpret_cast<char*>(&hasNumRows), 1);
      totalBytes_ += 1;
      if (numRowsOpt.has_value()) {
        int64_t numRows = numRowsOpt.value();
        bufferedWriteFile.append(
            reinterpret_cast<char*>(&numRows), sizeof(int64_t));
        totalBytes_ += sizeof(int64_t);
      }

      // Spill payload.
      bufferedWriteFile.append(page->getIOBuf());
    }
  }
  VELOX_CHECK_GE(totalBytes_, totalBytesBeforeSpill);
  spilledBytes += totalBytes_ - totalBytesBeforeSpill;
  totalPages_ += pages->size();

  auto spillStatsWPtr = spillStats_->wlock();
  spillStatsWPtr->spillRuns++;
  spillStatsWPtr->spilledInputBytes += spilledBytes;
  spillStatsWPtr->spilledBytes += spilledBytes;
  spillStatsWPtr->spilledRows += totalRows;
  spillStatsWPtr->spilledFiles++;
  spillStatsWPtr->spillWrites++;
  spillStatsWPtr->spillWriteTimeNanos += spillWriteTimeNs;
}

SerializedPageSpiller::Result SerializedPageSpiller::finishSpill() {
  return {std::move(spillFilePaths_), totalPages_, totalBytes_};
}

std::tuple<std::string, std::unique_ptr<WriteFile>>
SerializedPageSpiller::nextSpillWriteFile() {
  std::string path = fmt::format("{}-{}", filePrefix_, nextFileId_++);
  auto fs = filesystems::getFileSystem(path, nullptr);
  return {
      path,
      fs->openFileForWrite(
          path,
          filesystems::FileOptions{
              {{filesystems::FileOptions::kFileCreateConfig.toString(),
                fileCreateConfig_}},
              nullptr,
              std::nullopt})};
}

bool SerializedPageSpillReader::empty() const {
  return remainingPages_ == 0;
}

uint64_t SerializedPageSpillReader::remainingPages() const {
  return remainingPages_;
}

uint64_t SerializedPageSpillReader::remainingBytes() const {
  return remainingBytes_;
}

bool SerializedPageSpillReader::isEmptyAt(uint64_t index) {
  ensurePages(index);
  return bufferedPages_[index] == nullptr;
}

uint64_t SerializedPageSpillReader::sizeAt(uint64_t index) {
  ensurePages(index);
  const auto& page = bufferedPages_[index];
  VELOX_CHECK_NOT_NULL(page);
  return page->size();
}

std::shared_ptr<SerializedPage> SerializedPageSpillReader::at(uint64_t index) {
  ensurePages(index);
  return bufferedPages_[index];
}

void SerializedPageSpillReader::deleteAll() {
  spillFilePaths_.clear();
  curFileStream_.reset();
  bufferedPages_.clear();
  remainingPages_ = 0;
  remainingBytes_ = 0;
}

void SerializedPageSpillReader::deleteFront(uint64_t numPages) {
  if (numPages == 0) {
    return;
  }
  ensurePages(numPages - 1);
  for (uint32_t i = 0; i < numPages; ++i) {
    const auto& page = bufferedPages_[i];
    remainingBytes_ -= (page == nullptr ? 0 : page->size());
  }

  bufferedPages_.erase(
      bufferedPages_.begin(), bufferedPages_.begin() + numPages);
  remainingPages_ -= numPages;
}

void SerializedPageSpillReader::ensureFileStream() {
  if (curFileStream_ != nullptr) {
    return;
  }
  VELOX_CHECK(!spillFilePaths_.empty());
  auto filePath = spillFilePaths_.front();
  auto fs = filesystems::getFileSystem(filePath, nullptr);
  auto file = fs->openFileForRead(filePath);
  curFileStream_ = std::make_unique<common::FileInputStream>(
      std::move(file), readBufferSize_, pool_);
  spillFilePaths_.pop_front();
}

void SerializedPageSpillReader::ensurePages(uint64_t index) {
  VELOX_CHECK_LT(index, remainingPages_);
  if (index < bufferedPages_.size()) {
    return;
  }

  while (index >= bufferedPages_.size()) {
    ensureFileStream();
    bufferedPages_.push_back(unspillNextPage());
  }
}

namespace {
struct FreeData {
  std::shared_ptr<memory::MemoryPool> pool;
  int64_t bytesToFree;
};

void freeFunc(void* data, void* userData) {
  auto* freeData = reinterpret_cast<FreeData*>(userData);
  freeData->pool->free(data, freeData->bytesToFree);
  delete freeData;
}
} // namespace

std::shared_ptr<SerializedPage> SerializedPageSpillReader::unspillNextPage() {
  VELOX_CHECK(!empty());
  ensureFileStream();

  // Read payload headers
  auto isNull = !!(curFileStream_->read<uint8_t>());
  if (isNull) {
    if (curFileStream_->atEnd()) {
      curFileStream_.reset();
    }
    return nullptr;
  }
  auto iobufBytes = curFileStream_->read<int64_t>();
  auto hasNumRows = curFileStream_->read<uint8_t>() == 0 ? false : true;
  int64_t numRows{0};
  if (hasNumRows) {
    numRows = curFileStream_->read<int64_t>();
  }

  // Read payload
  VELOX_CHECK_GE(curFileStream_->remainingSize(), iobufBytes);
  void* rawBuf = pool_->allocate(iobufBytes);
  curFileStream_->readBytes(reinterpret_cast<uint8_t*>(rawBuf), iobufBytes);
  if (curFileStream_->atEnd()) {
    curFileStream_.reset();
  }

  auto* userData = new FreeData();
  userData->pool = pool_->shared_from_this();
  userData->bytesToFree = iobufBytes;
  auto iobuf =
      folly::IOBuf::takeOwnership(rawBuf, iobufBytes, freeFunc, userData, true);

  return std::make_shared<SerializedPage>(
      std::move(iobuf),
      nullptr,
      hasNumRows ? std::optional(numRows) : std::nullopt);
}
} // namespace facebook::velox::exec
