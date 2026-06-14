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

// Abstraction of a simplified file interface.
//
// Implementations are available for local disk and in-memory.
//
// We implement only a small subset of the normal file operations, namely
// Append for writing data and PRead for reading data.
//
// All functions are not threadsafe -- external locking is required, even
// for const member functions.

#pragma once

#include <fcntl.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <string_view>

#include <folly/Executor.h>
#include <folly/Range.h>
#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <folly/futures/Future.h>
#include <folly/io/IOBuf.h>

#include "velox/buffer/Buffer.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/file/FileIoTracer.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/Region.h"
#include "velox/common/io/IoStatistics.h"

namespace facebook::velox {

/// Free form statistics for file I/O operations. The keys are arbitrary
/// strings, and values are RuntimeMetric. This class can be used to record
/// observability about filesystem operations.
class IoStats {
 public:
  IoStats() = default;

  void addCounter(const std::string& name, RuntimeCounter counter);

  void merge(const IoStats& other);

  folly::F14FastMap<std::string, RuntimeMetric> stats() const;

 private:
  folly::Synchronized<folly::F14FastMap<std::string, RuntimeMetric>> stats_;
};

struct FileIoContext {
  /// Stats for IO operations.
  IoStats* ioStats{nullptr};

  /// Options for file read operations.
  folly::F14FastMap<std::string, std::string> fileOpts;

  /// Tracer for IO operations, providing call stack context.
  std::shared_ptr<FileIoTracer> ioTracer;

  /// When false, hints to the storage layer that this read should not be cached
  /// or should be evicted soon after reading. This is useful for one-time reads
  /// where caching would waste resources.
  bool cacheable{false};

  FileIoContext() = default;

  explicit FileIoContext(
      IoStats* stats,
      folly::F14FastMap<std::string, std::string> fileOpts = {},
      std::shared_ptr<FileIoTracer> tracer = nullptr,
      bool cacheable = false)
      : ioStats(stats),
        fileOpts(std::move(fileOpts)),
        ioTracer(std::move(tracer)),
        cacheable(cacheable) {}
};

// A read-only file.  All methods in this object should be thread safe.
class ReadFile {
 public:
  virtual ~ReadFile() = default;

  // Reads the data at [offset, offset + length) into the provided pre-allocated
  // buffer 'buf'. The bytes are returned as a string_view pointing to 'buf'.
  // This method should be thread safe.
  virtual std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& context = {}) const = 0;

  // Same as above, but returns owned data directly.
  //
  // This method should be thread safe.
  virtual std::string pread(
      uint64_t offset,
      uint64_t length,
      const FileIoContext& context = {}) const;

  // Reads starting at 'offset' into the memory referenced by the
  // Ranges in 'buffers'. The buffers are filled left to right. A
  // buffer with nullptr data will cause its size worth of bytes to be skipped.
  // This method should be thread safe.
  virtual uint64_t preadv(
      uint64_t /*offset*/,
      const std::vector<folly::Range<char*>>& /*buffers*/,
      const FileIoContext& context = {}) const;

  // Vectorized read API. Implementations can coalesce and parallelize.
  // The offsets don't need to be sorted.
  // `iobufs` is a range of IOBufs to store the read data. They
  // will be stored in the same order as the input `regions` vector. So the
  // array must be pre-allocated by the caller, with the same size as `regions`,
  // but don't need to be initialized, since each iobuf will be copy-constructed
  // by the preadv.
  // Returns the total number of bytes read, which might be different than the
  // sum of all buffer sizes (for example, if coalescing was used).
  // This method should be thread safe.
  virtual uint64_t preadv(
      folly::Range<const common::Region*> regions,
      folly::Range<folly::IOBuf*> iobufs,
      const FileIoContext& context = {}) const;

  // Positioned batch read API. Implementations can submit multiple independent
  // reads together. 'buffers' must have one caller-owned destination per input
  // region, and each destination length must match its region length.
  // The default implementation reads one region at a time.
  // Returns the total number of bytes read.
  // This method should be thread safe.
  virtual uint64_t preadv(
      folly::Range<const common::Region*> regions,
      folly::Range<const folly::Range<char*>*> buffers,
      const FileIoContext& context = {}) const;

  // Returns true if preadvAsync has a native implementation that is
  // asynchronous. The default implementation is synchronous.
  virtual bool hasPreadvAsync() const {
    return false;
  }

  /// Like preadv but may execute asynchronously and returns the read size or
  /// exception via SemiFuture. Use hasPreadvAsync() to check if the
  /// implementation is in fact asynchronous.
  /// This method should be thread safe.
  virtual folly::SemiFuture<uint64_t> preadvAsync(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const {
    try {
      return folly::SemiFuture<uint64_t>(preadv(offset, buffers, context));
    } catch (const std::exception& e) {
      return folly::makeSemiFuture<uint64_t>(e);
    }
  }

  /// Returns true if this file requires direct-I/O alignment. Sets alignment to
  /// the byte alignment required for positioned reads, or 1 otherwise.
  virtual bool directIo(uint64_t& alignment) const {
    alignment = 1;
    return false;
  }

  // Whether preads should be coalesced where possible. E.g. remote disk would
  // set to true, in-memory to false.
  virtual bool shouldCoalesce() const = 0;

  // Number of bytes in the file.
  virtual uint64_t size() const = 0;

  // An estimate for the total amount of memory *this uses.
  virtual uint64_t memoryUsage() const = 0;

  // The total number of bytes *this had been used to read since creation or
  // the last resetBytesRead. We sum all the |length| variables passed to
  // preads, not the actual amount of bytes read (which might be less).
  virtual uint64_t bytesRead() const {
    return bytesRead_;
  }

  virtual void resetBytesRead() {
    bytesRead_ = 0;
  }

  virtual std::string getName() const = 0;

  /// Gets the natural size for reads. Returns the number of bytes that should
  /// be read at once.
  virtual uint64_t getNaturalReadSize() const = 0;

 protected:
  mutable std::atomic_uint64_t bytesRead_{0};
};

// A write-only file. Nothing written to the file should be read back until it
// is closed.
class WriteFile {
 public:
  virtual ~WriteFile() = default;

  /// Appends data to the end of the file.
  virtual void append(std::string_view data) = 0;

  /// Appends data to the end of the file.
  virtual void append(std::unique_ptr<folly::IOBuf> /* data */) {
    VELOX_NYI("IOBuf appending is not implemented");
  }

  /// Writes data at the given offset of the file.
  ///
  /// NOTE: this is only supported on local file system and used by SSD cache
  /// for now. For filesystem like S3, it is not supported.
  virtual void write(
      const std::vector<iovec>& /* iovecs */,
      int64_t /* offset */,
      int64_t /* length */
  ) {
    VELOX_NYI("{} is not implemented", __FUNCTION__);
  }

  /// Truncates file to a new size.
  ///
  /// NOTE: this is only supported on local file system and used by SSD cache
  /// for now. For filesystem like S3, it is not supported.
  virtual void truncate(int64_t /* newSize */) {
    VELOX_NYI("{} is not implemented", __FUNCTION__);
  }

  /// Flushes any write buffers, i.e. ensures the remote storage backend or
  /// local storage medium received all the written data.
  virtual void flush() = 0;

  /// Sets the file attributes, which are file implementation specific.
  virtual void setAttributes(
      const std::unordered_map<std::string, std::string>& /* attributes */) {
    VELOX_NYI("{} is not implemented", __FUNCTION__);
  }

  /// Gets the file attributes, which are file implementation specific.
  virtual std::unordered_map<std::string, std::string> getAttributes() const {
    VELOX_NYI("{} is not implemented", __FUNCTION__);
  }

  /// Closes the file. Any cleanup (disk flush, etc.) will be done here.
  virtual void close() = 0;

  /// Current file size, i.e. the sum of all previous Appends.  No flush should
  /// be needed to get the exact size written, and this should be able to be
  /// called after the file close.
  virtual uint64_t size() const = 0;

  virtual const std::string getName() const {
    VELOX_NYI("{} is not implemented", __FUNCTION__);
  }
};

// We currently do a simple implementation for the in-memory files
// that simply resizes a string as needed. If there ever gets used in
// a performance sensitive path we'd probably want to move to a Cord-like
// implementation for underlying storage.

// We don't provide registration functions for the in-memory files, as they
// aren't intended for any robust use needing a filesystem.

class InMemoryReadFile : public ReadFile {
 public:
  explicit InMemoryReadFile(std::string_view file) : file_(file) {}

  explicit InMemoryReadFile(std::string file)
      : ownedFile_(std::move(file)), file_(ownedFile_) {}

  explicit InMemoryReadFile(BufferPtr buffer)
      : ownedBuffer_{std::move(buffer)},
        file_{std::string_view{
            ownedBuffer_->as<char>(),
            static_cast<size_t>(ownedBuffer_->size())}} {}

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& context = {}) const override;

  std::string pread(
      uint64_t offset,
      uint64_t length,
      const FileIoContext& context = {}) const override;

  uint64_t size() const final {
    return file_.size();
  }

  uint64_t memoryUsage() const final {
    return size();
  }

  // Mainly for testing. Coalescing isn't helpful for in memory data.
  void setShouldCoalesce(bool shouldCoalesce) {
    shouldCoalesce_ = shouldCoalesce;
  }

  bool shouldCoalesce() const final {
    return shouldCoalesce_;
  }

  std::string getName() const override {
    return "<InMemoryReadFile>";
  }

  uint64_t getNaturalReadSize() const override {
    return 1024;
  }

 private:
  const BufferPtr ownedBuffer_;
  const std::string ownedFile_;
  const std::string_view file_;
  bool shouldCoalesce_ = false;
};

class InMemoryWriteFile final : public WriteFile {
 public:
  explicit InMemoryWriteFile(std::string* file) : file_(file) {}

  void append(std::string_view data) final;
  void append(std::unique_ptr<folly::IOBuf> data) final;
  void flush() final {}
  void close() final {}
  uint64_t size() const final;

 private:
  std::string* file_;
};

} // namespace facebook::velox

#include "velox/common/file/LocalFile.h"
