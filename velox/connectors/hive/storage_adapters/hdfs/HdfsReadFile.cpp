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

#include "HdfsReadFile.h"

#include <chrono>
#include <thread>

#include "velox/external/hdfs/ArrowHdfsInternal.h"

namespace facebook::velox {

namespace {
// Transient HDFS read failures (DataNode hiccups, dropped connections) surface
// as a negative return from libhdfs3's hdfsRead. Retry a bounded number of
// times with exponential backoff before giving up, instead of failing the whole
// Spark task (and therefore the whole query) on the first transient error.
constexpr int kMaxReadRetries = 3;
constexpr int kReadRetryBaseDelayMs = 200;
} // namespace

struct HdfsFile {
  filesystems::arrow::io::internal::LibHdfsShim* driver_;
  hdfsFS client_;
  hdfsFile handle_;

  HdfsFile() : driver_(nullptr), client_(nullptr), handle_(nullptr) {}
  ~HdfsFile() {
    if (handle_ && driver_->CloseFile(client_, handle_) == -1) {
      LOG(ERROR) << "Unable to close file, errno: " << errno;
    }
  }

  void open(
      filesystems::arrow::io::internal::LibHdfsShim* driver,
      hdfsFS client,
      const std::string& path) {
    driver_ = driver;
    client_ = client;
    handle_ = driver->OpenFile(client, path.data(), O_RDONLY, 0, 0, 0);
    VELOX_CHECK_NOT_NULL(
        handle_,
        "Unable to open file {}. got error: {}",
        path,
        driver_->GetLastExceptionRootCause());
  }

  void seek(uint64_t offset) const {
    VELOX_CHECK_EQ(
        driver_->Seek(client_, handle_, offset),
        0,
        "Cannot seek through HDFS file, error is : {}",
        driver_->GetLastExceptionRootCause());
  }

  // Close the current handle (if any) and reopen the file. Used by
  // preadInternal to recover a thread-local handle whose stream went bad after
  // a transient read failure. file_ is a folly::ThreadLocal in the owning Impl,
  // so each thread reopens its own handle; the shared HDFS client is untouched.
  void reopen(const std::string& path) {
    if (handle_) {
      // Ignore the close result: the stream is already in a bad state and is
      // being discarded regardless.
      driver_->CloseFile(client_, handle_);
      handle_ = nullptr;
    }
    handle_ = driver_->OpenFile(client_, path.data(), O_RDONLY, 0, 0, 0);
    VELOX_CHECK_NOT_NULL(
        handle_,
        "Unable to reopen file {}. got error: {}",
        path,
        driver_->GetLastExceptionRootCause());
  }

  // Returns the raw libhdfs3 result, including non-positive values on failure.
  // The caller (preadInternal) decides whether a non-positive result is
  // retriable.
  int32_t read(char* pos, uint64_t length) const {
    return driver_->Read(client_, handle_, pos, length);
  }
};

class HdfsReadFile::Impl {
 public:
  Impl(
      filesystems::arrow::io::internal::LibHdfsShim* driver,
      hdfsFS hdfs,
      const std::string_view path)
      : driver_(driver), hdfsClient_(hdfs), filePath_(path) {
    fileInfo_ = driver_->GetPathInfo(hdfsClient_, filePath_.data());
    if (fileInfo_ == nullptr) {
      auto error = fmt::format(
          "FileNotFoundException: Path {} does not exist.", filePath_);
      auto errMsg = fmt::format(
          "Unable to get file path info for file: {}. got error: {}",
          filePath_,
          error);
      if (error.find("FileNotFoundException") != std::string::npos) {
        VELOX_FILE_NOT_FOUND_ERROR(errMsg);
      }
      VELOX_FAIL(errMsg);
    }
  }

  ~Impl() {
    // Should call hdfsFreeFileInfo to avoid memory leak
    if (fileInfo_) {
      driver_->FreeFileInfo(fileInfo_, 1);
    }
  }

  void preadInternal(uint64_t offset, uint64_t length, char* pos) const {
    checkFileReadParameters(offset, length);
    if (!file_->handle_) {
      file_->open(driver_, hdfsClient_, filePath_);
    }
    file_->seek(offset);
    uint64_t totalBytesRead = 0;
    int retries = 0;
    while (totalBytesRead < length) {
      auto bytesRead = file_->read(pos, length - totalBytesRead);
      // checkFileReadParameters guarantees offset + length stays within the
      // file, so we never legitimately hit EOF here. A non-positive result is
      // therefore always a failure to make progress: a negative value is an
      // explicit libhdfs3 error, and a zero means the read stalled without
      // advancing. Both are treated as transient and retried; leaving the zero
      // case out would spin this loop forever.
      if (bytesRead <= 0) {
        VELOX_CHECK_LT(
            retries,
            kMaxReadRetries,
            "Read failure in HDFSReadFile::preadInternal after {} retries, "
            "file: {}, offset: {}, length: {}, root cause: {}",
            kMaxReadRetries,
            filePath_,
            offset,
            length,
            driver_->GetLastExceptionRootCause());
        ++retries;
        LOG(WARNING) << "Transient HDFS read failure on " << filePath_
                     << " (offset=" << offset + totalBytesRead << ", retry "
                     << retries << "/" << kMaxReadRetries << "), root cause: "
                     << driver_->GetLastExceptionRootCause();
        std::this_thread::sleep_for(
            std::chrono::milliseconds(
                kReadRetryBaseDelayMs * (1 << (retries - 1))));
        // Rebuild the handle and reposition to the first unread byte; already
        // read bytes are kept.
        file_->reopen(filePath_);
        file_->seek(offset + totalBytesRead);
        continue;
      }
      totalBytesRead += bytesRead;
      pos += bytesRead;
    }
  }

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& context) const {
    preadInternal(offset, length, static_cast<char*>(buf));
    return {static_cast<char*>(buf), length};
  }

  std::string
  pread(uint64_t offset, uint64_t length, const FileIoContext& context) const {
    std::string result(length, 0);
    char* pos = result.data();
    preadInternal(offset, length, pos);
    return result;
  }

  uint64_t size() const {
    return fileInfo_->mSize;
  }

  uint64_t memoryUsage() const {
    return fileInfo_->mBlockSize;
  }

  bool shouldCoalesce() const {
    return false;
  }

  std::string getName() const {
    return filePath_;
  }

  void checkFileReadParameters(uint64_t offset, uint64_t length) const {
    auto fileSize = size();
    auto endPoint = offset + length;
    VELOX_CHECK_GE(
        fileSize,
        endPoint,
        "Cannot read HDFS file beyond its size: {}, offset: {}, end point: {}",
        fileSize,
        offset,
        endPoint);
  }

 private:
  filesystems::arrow::io::internal::LibHdfsShim* driver_;
  hdfsFS hdfsClient_;
  std::string filePath_;
  hdfsFileInfo* fileInfo_;
  folly::ThreadLocal<HdfsFile> file_;
};

HdfsReadFile::HdfsReadFile(
    filesystems::arrow::io::internal::LibHdfsShim* driver,
    hdfsFS hdfs,
    const std::string_view path)
    : pImpl(std::make_unique<Impl>(driver, hdfs, path)) {}

HdfsReadFile::~HdfsReadFile() = default;

std::string_view HdfsReadFile::pread(
    uint64_t offset,
    uint64_t length,
    void* buf,
    const FileIoContext& context) const {
  return pImpl->pread(offset, length, buf, context);
}

std::string HdfsReadFile::pread(
    uint64_t offset,
    uint64_t length,
    const FileIoContext& context) const {
  return pImpl->pread(offset, length, context);
}

uint64_t HdfsReadFile::size() const {
  return pImpl->size();
}

uint64_t HdfsReadFile::memoryUsage() const {
  return pImpl->memoryUsage();
}

bool HdfsReadFile::shouldCoalesce() const {
  return pImpl->shouldCoalesce();
}

std::string HdfsReadFile::getName() const {
  return pImpl->getName();
}

void HdfsReadFile::checkFileReadParameters(uint64_t offset, uint64_t length)
    const {
  pImpl->checkFileReadParameters(offset, length);
}

} // namespace facebook::velox
