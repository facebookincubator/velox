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

#include "velox/common/file/File.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <memory>
#include <stdexcept>

#include <fcntl.h>
#include <folly/portability/SysUio.h>

namespace facebook::velox {

std::string ReadFile::pread(uint64_t offset, uint64_t length) const {
  std::string buf;
  buf.resize(length);
  auto res = pread(offset, length, buf.data());
  buf.resize(res.size());
  return buf;
}

uint64_t ReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers) const {
  auto fileSize = size();
  uint64_t numRead = 0;
  if (offset >= fileSize) {
    return 0;
  }
  for (auto& range : buffers) {
    auto copySize = std::min<size_t>(range.size(), fileSize - offset);
    if (range.data()) {
      pread(offset, copySize, range.data());
    }
    offset += copySize;
    numRead += copySize;
  }
  return numRead;
}

std::string_view
InMemoryReadFile::pread(uint64_t offset, uint64_t length, void* buf) const {
  bytesRead_ += length;
  memcpy(buf, file_.data() + offset, length);
  return {static_cast<char*>(buf), length};
}

std::string InMemoryReadFile::pread(uint64_t offset, uint64_t length) const {
  bytesRead_ += length;
  return std::string(file_.data() + offset, length);
}

void InMemoryWriteFile::append(std::string_view data) {
  file_->append(data);
}

uint64_t InMemoryWriteFile::size() const {
  return file_->size();
}

LocalReadFile::LocalReadFile(std::string_view path) : path_(path) {
  fd_ = open(path_.c_str(), O_RDONLY);
  VELOX_CHECK_GE(
      fd_,
      0,
      "open failure in LocalReadFile constructor, {} {} {}.",
      fd_,
      path,
      folly::errnoStr(errno));
  const off_t rc = lseek(fd_, 0, SEEK_END);
  VELOX_CHECK_GE(
      rc,
      0,
      "fseek failure in LocalReadFile constructor, {} {} {}.",
      rc,
      path,
      folly::errnoStr(errno));
  size_ = rc;
}

LocalReadFile::LocalReadFile(int32_t fd) : fd_(fd) {}

LocalReadFile::~LocalReadFile() {
  const int ret = close(fd_);
  if (ret < 0) {
    LOG(WARNING) << "close failure in LocalReadFile destructor: " << ret << ", "
                 << folly::errnoStr(errno);
  }
}

void LocalReadFile::preadInternal(uint64_t offset, uint64_t length, char* pos)
    const {
  bytesRead_ += length;
  auto bytesRead = ::pread(fd_, pos, length, offset);
  VELOX_CHECK_EQ(
      bytesRead,
      length,
      "fread failure in LocalReadFile::PReadInternal, {} vs {}.",
      bytesRead,
      length);
}

std::string_view
LocalReadFile::pread(uint64_t offset, uint64_t length, void* buf) const {
  preadInternal(offset, length, static_cast<char*>(buf));
  return {static_cast<char*>(buf), length};
}

uint64_t LocalReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers) const {
  // Dropped bytes sized so that a typical dropped range of 50K is not
  // too many iovecs.
  static thread_local std::vector<char> droppedBytes(16 * 1024);
  std::vector<struct iovec> iovecs;
  iovecs.reserve(buffers.size());
  for (auto& range : buffers) {
    if (!range.data()) {
      auto skipSize = range.size();
      while (skipSize) {
        auto bytes = std::min<size_t>(droppedBytes.size(), skipSize);
        iovecs.push_back({droppedBytes.data(), bytes});
        skipSize -= bytes;
      }
    } else {
      iovecs.push_back({range.data(), range.size()});
    }
  }
  return folly::preadv(fd_, iovecs.data(), iovecs.size(), offset);
}

uint64_t LocalReadFile::size() const {
  return size_;
}

uint64_t LocalReadFile::memoryUsage() const {
  // TODO: does FILE really not use any more memory? From the stdio.h
  // source code it looks like it has only a single integer? Probably
  // we need to go deeper and see how much system memory is being taken
  // by the file descriptor the integer refers to?
  return sizeof(FILE);
}

LocalWriteFile::LocalWriteFile(std::string_view path) {
  std::unique_ptr<char[]> buf(new char[path.size() + 1]);
  buf[path.size()] = 0;
  memcpy(buf.get(), path.data(), path.size());
  {
    FILE* exists = fopen(buf.get(), "rb");
    VELOX_CHECK(
        !exists, "Failure in LocalWriteFile: path '{}' already exists.", path);
  }
  auto file = fopen(buf.get(), "ab");
  VELOX_CHECK(
      file,
      "fopen failure in LocalWriteFile constructor, {} {}.",
      path,
      folly::errnoStr(errno));
  file_ = file;
}

LocalWriteFile::~LocalWriteFile() {
  try {
    close();
  } catch (const std::exception& ex) {
    // We cannot throw an exception from the destructor. Warn instead.
    LOG(WARNING) << "fclose failure in LocalWriteFile destructor: "
                 << ex.what();
  }
}

void LocalWriteFile::append(std::string_view data) {
  VELOX_CHECK(!closed_, "file is closed");
  const uint64_t bytes_written = fwrite(data.data(), 1, data.size(), file_);
  VELOX_CHECK_EQ(
      bytes_written,
      data.size(),
      "fwrite failure in LocalWriteFile::append, {} vs {}.",
      bytes_written,
      data.size());
}

void LocalWriteFile::flush() {
  VELOX_CHECK(!closed_, "file is closed");
  auto ret = fflush(file_);
  VELOX_CHECK_EQ(
      ret,
      0,
      "fflush failed in LocalWriteFile::flush: {}.",
      folly::errnoStr(errno));
}

void LocalWriteFile::close() {
  if (!closed_) {
    auto ret = fclose(file_);
    VELOX_CHECK_EQ(
        ret,
        0,
        "fwrite failure in LocalWriteFile::close: {}.",
        folly::errnoStr(errno));
    closed_ = true;
  }
}

uint64_t LocalWriteFile::size() const {
  return ftell(file_);
}
} // namespace facebook::velox
