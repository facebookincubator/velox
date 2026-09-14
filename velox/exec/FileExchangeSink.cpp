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

#include "velox/exec/FileExchangeSink.h"

#include <folly/dynamic.h>
#include <folly/hash/Checksum.h>
#include <folly/json.h>

#include "velox/common/base/Exceptions.h"
#include "velox/exec/FileExchangeFormat.h"

namespace facebook::velox::exec {
namespace {

std::string partitionPath(
    const std::string& rootDir,
    const std::string& exchangeId,
    int32_t partition) {
  return fmt::format("{}/{}/partition_{}", rootDir, exchangeId, partition);
}

} // namespace

void FileExchangeSink::PartitionWriter::append(
    const void* data,
    size_t numBytes) {
  VELOX_CHECK_NOT_NULL(file);
  file->append(std::string_view{reinterpret_cast<const char*>(data), numBytes});
  size += numBytes;
  checksum =
      folly::crc32c(reinterpret_cast<const uint8_t*>(data), numBytes, checksum);
}

FileExchangeSink::FileExchangeSink(
    std::string rootDir,
    std::string exchangeId,
    std::string taskId,
    int32_t numPartitions)
    : rootDir_(std::move(rootDir)),
      exchangeId_(std::move(exchangeId)),
      taskId_(std::move(taskId)),
      numPartitions_(numPartitions),
      fileSystem_(filesystems::getFileSystem(rootDir_, nullptr)),
      writers_(numPartitions) {
  for (int32_t partition = 0; partition < numPartitions_; ++partition) {
    fileSystem_->mkdir(partitionDir(partition));
  }
}

std::string FileExchangeSink::partitionDir(int32_t partition) const {
  return partitionPath(rootDir_, exchangeId_, partition);
}

std::string FileExchangeSink::partitionFile(int32_t partition) const {
  return fmt::format("{}/{}.bin", partitionDir(partition), taskId_);
}

FileExchangeSink::PartitionWriter& FileExchangeSink::partitionWriter(
    int32_t partition) {
  auto& partitionWriter = writers_[partition];
  if (partitionWriter.file == nullptr) {
    const auto filePath = partitionFile(partition);
    partitionWriter.file = fileSystem_->openFileForWrite(filePath);
  }
  return partitionWriter;
}

std::optional<file_exchange::ExchangeOutputFile> FileExchangeSink::closeWriter(
    int32_t partition) {
  auto& partitionWriter = writers_[partition];
  if (partitionWriter.file == nullptr) {
    return std::nullopt;
  }
  partitionWriter.file->close();
  VELOX_CHECK_EQ(partitionWriter.file->size(), partitionWriter.size);
  return file_exchange::ExchangeOutputFile{
      .path = partitionFile(partition),
      .size = partitionWriter.size,
      .checksum = partitionWriter.checksum,
  };
}

void FileExchangeSink::append(int32_t partition, std::string_view page) {
  append(partition, folly::IOBuf::copyBuffer(page.data(), page.size()));
}

void FileExchangeSink::append(
    int32_t partition,
    std::unique_ptr<folly::IOBuf> page) {
  VELOX_CHECK_GE(partition, 0);
  VELOX_CHECK_LT(partition, numPartitions_);
  VELOX_CHECK_NOT_NULL(page);

  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK(!closed_);
  auto& partitionWriter = this->partitionWriter(partition);
  const auto pageSize = page->computeChainDataLength();
  const auto encodedPageSize = file_exchange::encodePageSize(pageSize);
  partitionWriter.append(&encodedPageSize, sizeof(file_exchange::PageSize));
  for (const auto& range : *page) {
    partitionWriter.append(range.data(), range.size());
  }
  totalBytesWritten_ += pageSize;
  ++totalPages_;
}

CommittedExchangeOutput FileExchangeSink::finish() {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK(!closed_, "File exchange sink is already closed");

  CommittedExchangeOutput output;
  for (int32_t partition = 0; partition < numPartitions_; ++partition) {
    if (auto outputFile = closeWriter(partition)) {
      output.locations[partition] = outputFile->serialize();
    }
  }
  closed_ = true;
  return output;
}

void FileExchangeSink::abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return;
  }
  for (int32_t partition = 0; partition < numPartitions_; ++partition) {
    closeWriter(partition);
  }
  closed_ = true;
}

folly::F14FastMap<std::string, int64_t> FileExchangeSink::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {
      {"totalBytesWritten", totalBytesWritten_},
      {"totalPages", totalPages_},
  };
}

std::shared_ptr<ExchangeSink> FileExchangeSink::create(
    const std::string& config,
    const std::string& taskId,
    memory::MemoryPool* /*pool*/) {
  auto parsed = folly::parseJson(config);
  return std::make_shared<FileExchangeSink>(
      parsed["rootDir"].asString(),
      parsed["exchangeId"].asString(),
      taskId,
      parsed["numPartitions"].asInt());
}

} // namespace facebook::velox::exec
