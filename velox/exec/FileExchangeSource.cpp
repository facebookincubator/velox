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

#include "velox/exec/FileExchangeSource.h"

#include <mutex>

#include <folly/hash/Checksum.h>
#include <folly/io/IOBuf.h>
#include "velox/common/base/Exceptions.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/exec/FileExchangeFormat.h"
#include "velox/serializers/PrestoSerializer.h"

namespace facebook::velox::exec {

FileExchangeSource::FileExchangeSource(
    const std::string& taskId,
    int32_t destination,
    std::shared_ptr<ExchangeQueue> queue,
    memory::MemoryPool* pool)
    : ExchangeSource(taskId, destination, queue, pool) {
  auto outputFile = file_exchange::ExchangeOutputFile::deserialize(taskId);
  filePath_ = std::move(outputFile.path);
  expectedFileSize_ = outputFile.size;
  expectedChecksum_ = outputFile.checksum;
}

bool FileExchangeSource::shouldRequestLocked() {
  if (atEnd_ || closed_) {
    return false;
  }
  return !requestPending_.exchange(true);
}

void FileExchangeSource::openFile() {
  if (file_ != nullptr) {
    return;
  }
  fileSystem_ = filesystems::getFileSystem(filePath_, nullptr);
  file_ = fileSystem_->openFileForRead(filePath_);
  const auto fileSize = file_->size();
  VELOX_CHECK_EQ(
      fileSize,
      expectedFileSize_,
      "File exchange size mismatch for {}",
      filePath_);
}

std::unique_ptr<SerializedPageBase> FileExchangeSource::readPage() {
  if (fileBytesRead_ == expectedFileSize_) {
    return nullptr;
  }
  VELOX_CHECK_GE(
      expectedFileSize_ - fileBytesRead_,
      sizeof(file_exchange::PageSize),
      "Incomplete page-size header in file: {}",
      filePath_);

  file_exchange::PageSize encodedPageSize;
  const auto encodedPageSizeData = file_->pread(
      fileBytesRead_, sizeof(file_exchange::PageSize), &encodedPageSize);
  VELOX_CHECK_EQ(
      encodedPageSizeData.size(),
      sizeof(file_exchange::PageSize),
      "Incomplete page-size header in file: {}",
      filePath_);
  fileChecksum_ = folly::crc32c(
      reinterpret_cast<const uint8_t*>(&encodedPageSize),
      sizeof(file_exchange::PageSize),
      fileChecksum_);
  fileBytesRead_ += sizeof(file_exchange::PageSize);

  const auto pageSize = file_exchange::decodePageSize(encodedPageSize);
  VELOX_CHECK_LE(
      pageSize,
      expectedFileSize_ - fileBytesRead_,
      "Incomplete page payload in file: {}",
      filePath_);
  auto buffer = folly::IOBuf::create(pageSize);
  const auto pageData =
      file_->pread(fileBytesRead_, pageSize, buffer->writableData());
  VELOX_CHECK_EQ(
      pageData.size(),
      pageSize,
      "Incomplete page payload in file: {}",
      filePath_);
  fileChecksum_ =
      folly::crc32c(buffer->writableData(), pageSize, fileChecksum_);
  fileBytesRead_ += pageSize;
  buffer->append(pageSize);
  return std::make_unique<PrestoSerializedPage>(std::move(buffer));
}

void FileExchangeSource::verifyChecksum() const {
  VELOX_CHECK_EQ(
      fileChecksum_,
      expectedChecksum_,
      "File exchange checksum mismatch for {}",
      filePath_);
}

void FileExchangeSource::enqueuePage(std::unique_ptr<SerializedPageBase> page) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> lock(queue_->mutex());
    queue_->enqueueLocked(std::move(page), promises);
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void FileExchangeSource::enqueueEnd() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> lock(queue_->mutex());
    if (endEnqueued_) {
      return;
    }
    verifyChecksum();
    endEnqueued_ = true;
    atEnd_ = true;
    queue_->enqueueLocked(nullptr, promises);
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

folly::SemiFuture<ExchangeSource::Response> FileExchangeSource::request(
    uint32_t maxBytes,
    std::chrono::microseconds /*maxWait*/) {
  if (closed_) {
    requestPending_ = false;
    return folly::makeSemiFuture(
        Response{.bytes = 0, .atEnd = false, .remainingBytes = {}});
  }

  openFile();
  int64_t bytes{0};
  bool atEnd{false};
  while (bytes == 0 || bytes < maxBytes) {
    auto page = readPage();
    if (page == nullptr) {
      enqueueEnd();
      atEnd = true;
      break;
    }
    bytes += page->size();
    enqueuePage(std::move(page));
  }
  requestPending_ = false;
  return folly::makeSemiFuture(
      Response{.bytes = bytes, .atEnd = atEnd, .remainingBytes = {}});
}

void FileExchangeSource::close() {
  closed_ = true;
}

// --- Factory ---

std::shared_ptr<ExchangeSource> FileExchangeSourceFactory::operator()(
    const std::string& taskId,
    int32_t destination,
    std::shared_ptr<ExchangeQueue> queue,
    memory::MemoryPool* pool) {
  if (file_exchange::ExchangeOutputFile::serialized(taskId)) {
    return std::make_shared<FileExchangeSource>(
        taskId, destination, std::move(queue), pool);
  }
  return nullptr;
}

} // namespace facebook::velox::exec
