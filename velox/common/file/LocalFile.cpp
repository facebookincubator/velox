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

#include "velox/common/file/LocalFile.h"

#include "velox/common/base/Fs.h"
#ifndef _WIN32
// io_uring is a Linux kernel feature; the reader is unavailable on Windows.
#include "velox/common/file/IoUringReader.h"
#endif // _WIN32

#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>
#endif // _WIN32
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <glog/logging.h>

#include <folly/Conv.h>
#include <folly/portability/SysUio.h>

#include "velox/common/memory/Allocation.h"

#ifdef linux
#include <linux/fs.h>
#include <sys/ioctl.h>
#endif // linux
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <algorithm>
#include <crtdbg.h>
#include <stdlib.h>
#include <windows.h>

// Suppress MSVC CRT invalid parameter handler (which calls abort())
// when calling CRT functions with invalid file descriptors.
static void _silentInvalidParameterHandler(
    const wchar_t*,
    const wchar_t*,
    const wchar_t*,
    unsigned int,
    uintptr_t) {}

// Get Windows HANDLE from fd, returning INVALID_HANDLE_VALUE on failure
// without triggering MSVC's default invalid parameter handler (abort).
inline HANDLE _safeGetOsfHandle(int fd) {
  auto old = _set_invalid_parameter_handler(_silentInvalidParameterHandler);
  int oldMode = _CrtSetReportMode(_CRT_ASSERT, 0);
  HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  _CrtSetReportMode(_CRT_ASSERT, oldMode);
  _set_invalid_parameter_handler(old);
  return h;
}

// Windows wrappers for Unix file functions
inline int ftruncate(int fd, int64_t length) {
  return _chsize_s(fd, length);
}

inline int fsync(int fd) {
  return _commit(fd);
}

// Thread-safe pread using Win32 ReadFile with OVERLAPPED (like POSIX pread).
// Does NOT modify the file pointer, so concurrent calls on the same fd are safe.
// Use int64_t instead of off_t because off_t is 32-bit on Windows,
// which would truncate offsets for files larger than 2GB.
inline ssize_t pread(int fd, void* buf, size_t count, int64_t offset) {
  HANDLE h = _safeGetOsfHandle(fd);
  if (h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }

  size_t totalRead = 0;
  auto* dest = static_cast<char*>(buf);
  while (totalRead < count) {
    DWORD toRead = static_cast<DWORD>(
        std::min<size_t>(count - totalRead, 1u << 30)); // 1GB max per call
    OVERLAPPED ov = {};
    int64_t pos = static_cast<int64_t>(offset) + totalRead;
    ov.Offset = static_cast<DWORD>(pos & 0xFFFFFFFF);
    ov.OffsetHigh = static_cast<DWORD>((pos >> 32) & 0xFFFFFFFF);
    DWORD bytesRead = 0;
    if (!ReadFile(h, dest + totalRead, toRead, &bytesRead, &ov)) {
      DWORD err = GetLastError();
      if (err == ERROR_HANDLE_EOF) break;
      errno = EIO;
      return -1;
    }
    if (bytesRead == 0) break; // EOF
    totalRead += bytesRead;
  }
  return static_cast<ssize_t>(totalRead);
}

// Thread-safe pwritev using Win32 WriteFile with OVERLAPPED.
// Use int64_t instead of off_t because off_t is 32-bit on Windows.
inline ssize_t pwritev(int fd, const struct iovec* iov, int iovcnt, int64_t offset) {
  HANDLE h = _safeGetOsfHandle(fd);
  if (h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }

  ssize_t totalWritten = 0;
  int64_t pos = static_cast<int64_t>(offset);
  for (int i = 0; i < iovcnt; ++i) {
    size_t remaining = iov[i].iov_len;
    auto* src = static_cast<const char*>(iov[i].iov_base);
    while (remaining > 0) {
      DWORD toWrite = static_cast<DWORD>(
          std::min<size_t>(remaining, 1u << 30));
      OVERLAPPED ov = {};
      ov.Offset = static_cast<DWORD>(pos & 0xFFFFFFFF);
      ov.OffsetHigh = static_cast<DWORD>((pos >> 32) & 0xFFFFFFFF);
      DWORD bytesWritten = 0;
      if (!WriteFile(h, src, toWrite, &bytesWritten, &ov)) return -1;
      totalWritten += bytesWritten;
      src += bytesWritten;
      remaining -= bytesWritten;
      pos += bytesWritten;
      if (bytesWritten < toWrite) break;
    }
  }
  return totalWritten;
}

// Windows file permission constants
#define S_IRUSR _S_IREAD
#define S_IWUSR _S_IWRITE
#endif // _WIN32

namespace facebook::velox {

#define RETURN_IF_ERROR(func, result) \
  result = func;                      \
  if (result < 0) {                   \
    return result;                    \
  }

namespace {

int32_t openReadFile(const std::string& path, bool directIo) {
  int32_t flags = O_RDONLY;
#ifdef _WIN32
  flags |= O_BINARY; // Windows: prevent text-mode translation.
#endif // _WIN32
#ifdef linux
  if (directIo) {
    flags |= O_DIRECT;
  }
#endif // linux
  const auto fd = open(path.c_str(), flags);
  if (fd < 0) {
    if (errno == ENOENT) {
      VELOX_FILE_NOT_FOUND_ERROR("No such file or directory: {}", path);
    }
    VELOX_FAIL(
        "open failure in LocalReadFile constructor, {} {} {}.",
        fd,
        path,
        folly::errnoStr(errno));
  }
  return fd;
}

long fileSize(int32_t fd, std::string_view path) {
#ifdef _WIN32
  // Use Win32 GetFileSizeEx to reliably get the 64-bit file size. CRT lseek
  // uses 32-bit off_t on Windows and can fail/truncate on files > 2GB.
  HANDLE h = _safeGetOsfHandle(fd);
  LARGE_INTEGER fileSizeResult;
  VELOX_CHECK(
      h != INVALID_HANDLE_VALUE && GetFileSizeEx(h, &fileSizeResult),
      "GetFileSizeEx failure in LocalReadFile constructor, {} {}.",
      path,
      GetLastError());
  return static_cast<long>(fileSizeResult.QuadPart);
#else
  const off_t ret = lseek(fd, 0, SEEK_END);
  VELOX_CHECK_GE(
      ret,
      0,
      "fseek failure in LocalReadFile constructor, {} {} {}.",
      ret,
      path,
      folly::errnoStr(errno));
  return ret;
#endif // _WIN32
}

FOLLY_ALWAYS_INLINE void checkNotClosed(bool closed) {
  VELOX_CHECK(!closed, "file is closed");
}

#ifdef STATX_DIOALIGN
bool isPowerOfTwo(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}
#endif

uint64_t checkBatchRead(
    folly::Range<const common::Region*> regions,
    folly::Range<const folly::Range<char*>*> buffers) {
  VELOX_CHECK(!regions.empty(), "preadv requires at least one region");
  VELOX_CHECK_EQ(
      regions.size(), buffers.size(), "preadv requires one buffer per region");

  uint64_t length{0};
  for (size_t i = 0; i < regions.size(); ++i) {
    const auto& region = regions[i];
    const auto& buffer = buffers[i];
    VELOX_CHECK_NOT_NULL(buffer.data(), "preadv buffer must not be null");
    VELOX_CHECK_EQ(
        buffer.size(),
        region.length,
        "preadv destination buffer length must match region length");
    length += region.length;
  }
  return length;
}

uint64_t getReadAlignment(int32_t fd, bool directIo) {
#ifndef STATX_DIOALIGN
  (void)fd;
#endif

  if (!directIo) {
    return 1;
  }

#ifdef STATX_DIOALIGN
  struct statx statxBuffer{};
  const int ret = ::statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &statxBuffer);
  if (ret == 0 && (statxBuffer.stx_mask & STATX_DIOALIGN) != 0) {
    const auto alignment = std::max<uint64_t>(
        statxBuffer.stx_dio_mem_align, statxBuffer.stx_dio_offset_align);
    if (isPowerOfTwo(alignment)) {
      return alignment;
    }
  }
#endif

  // Page size is the conservative direct-I/O alignment fallback when the
  // filesystem or kernel does not report a more specific requirement.
  return memory::AllocationTraits::kPageSize;
}

bool validateUseIoUringConfig(bool directIo, bool useIoUring) {
  if (!useIoUring) {
    return false;
  }

  VELOX_CHECK(
      directIo, "LocalReadFile useIoUring requested but direct IO is disabled");
#ifndef _WIN32
  VELOX_CHECK(
      IoUringReader::available(),
      "LocalReadFile useIoUring requested but io_uring is unavailable");
  return true;
#else
  VELOX_FAIL("LocalReadFile useIoUring requested but io_uring is unavailable");
#endif // _WIN32
}

template <typename T>
T getAttribute(
    const std::unordered_map<std::string, std::string>& attributes,
    const std::string_view& key,
    const T& defaultValue) {
  if (attributes.count(std::string(key)) > 0) {
    try {
      return folly::to<T>(attributes.at(std::string(key)));
    } catch (const std::exception& e) {
      VELOX_FAIL("Failed while parsing File attributes: {}", e.what());
    }
  }
  return defaultValue;
}

} // namespace

LocalReadFile::LocalReadFile(
    std::string_view path,
    folly::Executor* executor,
    bool bufferIo,
    bool useIoUring)
    : executor_(executor),
      directIo_{!bufferIo},
      useIoUring_{validateUseIoUringConfig(directIo_, useIoUring)},
      path_(path),
      fd_{openReadFile(path_, directIo_)},
      size_{fileSize(fd_, path_)},
      readAlignment_{getReadAlignment(fd_, directIo_)} {}

LocalReadFile::~LocalReadFile() {
  const int ret = close(fd_);
  if (ret < 0) {
    LOG(WARNING) << "close failure in LocalReadFile destructor: " << ret << ", "
                 << folly::errnoStr(errno);
  }
}

bool LocalReadFile::directIo(uint64_t& alignment) const {
  if (!directIo_) {
    alignment = 1;
    return false;
  }
  alignment = readAlignment_;
  return true;
}

void LocalReadFile::preadInternal(uint64_t offset, uint64_t length, char* pos)
    const {
  bytesRead_ += length;
  auto bytesRead = ::pread(fd_, pos, length, offset);
  const int savedErrno = errno;
  VELOX_CHECK_EQ(
      bytesRead,
      length,
      "fread failure in LocalReadFile::PReadInternal, {} vs {}: {}, path={}, fd={}, offset={}, length={}, buffer={}, directIo={}, readAlignment={}",
      bytesRead,
      length,
      folly::errnoStr(savedErrno),
      path_,
      fd_,
      offset,
      length,
      static_cast<void*>(pos),
      directIo_,
      readAlignment_);
}

std::string_view LocalReadFile::pread(
    uint64_t offset,
    uint64_t length,
    void* buf,
    const FileIoContext& /*context*/) const {
  preadInternal(offset, length, static_cast<char*>(buf));
  return {static_cast<char*>(buf), length};
}

uint64_t LocalReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers,
    const FileIoContext& context) const {
#ifdef _WIN32
  // On Windows, folly::preadv uses lseek+readv with 32-bit off_t, which is
  // not thread-safe and truncates offsets > 2GB. Fall back to the base class
  // implementation which uses our thread-safe pread (ReadFile + OVERLAPPED).
  return ReadFile::preadv(offset, buffers, context);
#else
  // Dropped bytes sized so that a typical dropped range of 50K is not
  // too many iovecs.
  static thread_local std::vector<char> droppedBytes(16 * 1024);
  uint64_t totalBytesRead = 0;
  std::vector<struct iovec> iovecs;
  iovecs.reserve(buffers.size());

  auto readvFunc = [&]() -> ssize_t {
    const auto bytesRead =
        folly::preadv(fd_, iovecs.data(), iovecs.size(), offset);
    if (bytesRead < 0) {
      LOG(ERROR) << "preadv failed with error: " << folly::errnoStr(errno);
    } else {
      totalBytesRead += bytesRead;
      offset += bytesRead;
    }
    iovecs.clear();
    return bytesRead;
  };

  for (auto& range : buffers) {
    if (!range.data()) {
      auto skipSize = range.size();
      while (skipSize) {
        auto bytes = std::min<size_t>(droppedBytes.size(), skipSize);

        if (iovecs.size() >= IOV_MAX) {
          ssize_t bytesRead{0};
          RETURN_IF_ERROR(readvFunc(), bytesRead);
        }

        iovecs.push_back({droppedBytes.data(), bytes});
        skipSize -= bytes;
      }
    } else {
      if (iovecs.size() >= IOV_MAX) {
        ssize_t bytesRead{0};
        RETURN_IF_ERROR(readvFunc(), bytesRead);
      }

      iovecs.push_back({range.data(), range.size()});
    }
  }

  // Perform any remaining preadv calls
  if (!iovecs.empty()) {
    ssize_t bytesRead{0};
    RETURN_IF_ERROR(readvFunc(), bytesRead);
  }

  return totalBytesRead;
#endif // _WIN32
}

uint64_t LocalReadFile::preadv(
    folly::Range<const common::Region*> regions,
    folly::Range<const folly::Range<char*>*> buffers,
    const FileIoContext& context) const {
  if (!useIoUring_) {
    return ReadFile::preadv(regions, buffers, context);
  }

#ifndef _WIN32
  const auto length = checkBatchRead(regions, buffers);
  // TODO: Extend io_uring support to the other read APIs.
  bytesRead_ += length;
  return ThreadLocalIoUringReader::get().read(fd_, regions, buffers);
#else
  VELOX_FAIL("io_uring is not supported on Windows");
#endif // _WIN32
}

folly::SemiFuture<uint64_t> LocalReadFile::preadvAsync(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers,
    const FileIoContext& context) const {
  if (!executor_) {
    return ReadFile::preadvAsync(offset, buffers, context);
  }
  auto [promise, future] = folly::makePromiseContract<uint64_t>();
  executor_->add([this,
                  _promise = std::move(promise),
                  _offset = offset,
                  _buffers = buffers,
                  _context = context]() mutable {
    auto delegateFuture = ReadFile::preadvAsync(_offset, _buffers, _context);
    _promise.setTry(std::move(delegateFuture).getTry());
  });
  return std::move(future);
}

uint64_t LocalReadFile::size() const {
#ifdef _MSC_VER
  // On Windows, the file may be extended by another handle (e.g., SsdFile
  // opens separate read/write handles). Query the actual file size so that
  // ReadFile::preadv (the Windows fallback) sees the current size.
  const auto actual = _filelengthi64(fd_);
  if (actual >= 0) {
    return static_cast<uint64_t>(actual);
  }
#endif // _MSC_VER
  return size_;
}

uint64_t LocalReadFile::memoryUsage() const {
  // TODO: does FILE really not use any more memory? From the stdio.h
  // source code it looks like it has only a single integer? Probably
  // we need to go deeper and see how much system memory is being taken
  // by the file descriptor the integer refers to?
  return sizeof(FILE);
}

bool LocalWriteFile::Attributes::cowDisabled(
    const std::unordered_map<std::string, std::string>& attrs) {
  return getAttribute<bool>(attrs, kNoCow, kDefaultNoCow);
}

LocalWriteFile::LocalWriteFile(
    std::string_view path,
    bool shouldCreateParentDirectories,
    bool shouldThrowOnFileAlreadyExists,
    bool bufferIo)
    : path_(path) {
  const auto dir = fs::path(path_).parent_path();
  if (shouldCreateParentDirectories && !fs::exists(dir)) {
    VELOX_CHECK(
        common::generateFileDirectory(dir.string().c_str()),
        "Failed to generate file directory");
  }

  // File open flags: write-only, create the file if it doesn't exist.
  int32_t flags = O_WRONLY | O_CREAT;
#ifdef _WIN32
  flags |= O_BINARY; // Windows: prevent text-mode translation.
#endif // _WIN32
  if (shouldThrowOnFileAlreadyExists) {
    flags |= O_EXCL;
  }
#ifdef linux
  if (!bufferIo) {
    flags |= O_DIRECT;
  }
#endif // linux

  // The file mode bits to be applied when a new file is created. By default
  // user has read and write access to the file.
  // NOTE: The mode argument must be supplied if O_CREAT or O_TMPFILE is
  // specified in flags; if it is not supplied, some arbitrary bytes from the
  // stack will be applied as the file mode.
  const int32_t mode = S_IRUSR | S_IWUSR;

  std::unique_ptr<char[]> buf(new char[path_.size() + 1]);
  buf[path_.size()] = 0;
  ::memcpy(buf.get(), path_.data(), path_.size());
  fd_ = open(buf.get(), flags, mode);
  VELOX_CHECK_GE(
      fd_,
      0,
      "Cannot open or create {}. Error: {}",
      path_,
      folly::errnoStr(errno));

#ifdef _WIN32
  // Use _lseeki64 on Windows because off_t/lseek are 32-bit (long), which
  // fails for files > 2GB with EINVAL.
  const int64_t ret = _lseeki64(fd_, 0, SEEK_END);
#else
  const off_t ret = lseek(fd_, 0, SEEK_END);
#endif // _WIN32
  VELOX_CHECK_GE(
      ret,
      0,
      "fseek failure in LocalWriteFile constructor, {} {} {}.",
      ret,
      path_,
      folly::errnoStr(errno));
  size_ = ret;
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
  checkNotClosed(closed_);
  const uint64_t bytesWritten = ::write(fd_, data.data(), data.size());
  VELOX_CHECK_EQ(
      bytesWritten,
      data.size(),
      "fwrite failure in LocalWriteFile::append, {} vs {}: {}",
      bytesWritten,
      data.size(),
      folly::errnoStr(errno));
  size_ += bytesWritten;
}

void LocalWriteFile::append(std::unique_ptr<folly::IOBuf> data) {
  checkNotClosed(closed_);
  uint64_t totalBytesWritten{0};
  for (auto rangeIter = data->begin(); rangeIter != data->end(); ++rangeIter) {
    const auto bytesToWrite = rangeIter->size();
    const uint64_t bytesWritten =
        ::write(fd_, rangeIter->data(), rangeIter->size());
    totalBytesWritten += bytesWritten;
    if (bytesWritten != bytesToWrite) {
      VELOX_FAIL(
          "fwrite failure in LocalWriteFile::append, {} vs {}: {}",
          bytesWritten,
          bytesToWrite,
          folly::errnoStr(errno));
    }
  }
  const auto totalBytesToWrite = data->computeChainDataLength();
  VELOX_CHECK_EQ(
      totalBytesWritten,
      totalBytesToWrite,
      "Failure in LocalWriteFile::append, {} vs {}",
      totalBytesWritten,
      totalBytesToWrite);
  size_ += totalBytesWritten;
}

void LocalWriteFile::write(
    const std::vector<iovec>& iovecs,
    int64_t offset,
    int64_t length) {
  checkNotClosed(closed_);
  VELOX_CHECK_GE(offset, 0, "Offset cannot be negative.");
  const auto bytesWritten = ::pwritev(
      fd_, iovecs.data(), static_cast<ssize_t>(iovecs.size()), offset);
  VELOX_CHECK_EQ(
      bytesWritten,
      length,
      "Failure in LocalWriteFile::write, {} vs {}",
      bytesWritten,
      length);
  size_ = std::max<uint64_t>(size_, offset + bytesWritten);
}

void LocalWriteFile::truncate(int64_t newSize) {
  checkNotClosed(closed_);
  VELOX_CHECK_GE(newSize, 0, "New size cannot be negative.");
  const auto ret = ::ftruncate(fd_, newSize);
  VELOX_CHECK_EQ(
      ret,
      0,
      "ftruncate failed in LocalWriteFile::truncate: {}.",
      folly::errnoStr(errno));
  // Reposition the file offset to the end of the file for append().
#ifdef _WIN32
  _lseeki64(fd_, newSize, SEEK_SET);
#else
  ::lseek(fd_, newSize, SEEK_SET);
#endif // _WIN32
  size_ = newSize;
}

void LocalWriteFile::flush() {
  checkNotClosed(closed_);
  const auto ret = ::fsync(fd_);
  VELOX_CHECK_EQ(
      ret,
      0,
      "fsync failed in LocalWriteFile::flush: {}.",
      folly::errnoStr(errno));
}

void LocalWriteFile::setAttributes(
    const std::unordered_map<std::string, std::string>& attributes) {
  checkNotClosed(closed_);
  attributes_ = attributes;
#ifdef linux
  if (Attributes::cowDisabled(attributes_)) {
    int attr{0};
    auto ret = ioctl(fd_, FS_IOC_GETFLAGS, &attr);
    VELOX_CHECK_EQ(
        0,
        ret,
        "ioctl(FS_IOC_GETFLAGS) failed: {}, {}",
        ret,
        folly::errnoStr(errno));
    attr |= FS_NOCOW_FL;
    ret = ioctl(fd_, FS_IOC_SETFLAGS, &attr);
    VELOX_CHECK_EQ(
        0,
        ret,
        "ioctl(FS_IOC_SETFLAGS, FS_NOCOW_FL) failed: {}, {}",
        ret,
        folly::errnoStr(errno));
  }
#endif // linux
}

std::unordered_map<std::string, std::string> LocalWriteFile::getAttributes()
    const {
  checkNotClosed(closed_);
  return attributes_;
}

void LocalWriteFile::close() {
  if (!closed_) {
    const auto ret = ::close(fd_);
    VELOX_CHECK_EQ(
        ret,
        0,
        "close failed in LocalWriteFile::close: {}.",
        folly::errnoStr(errno));
    closed_ = true;
  }
}

} // namespace facebook::velox
