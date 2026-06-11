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

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <folly/Range.h>
#include <folly/SharedMutex.h>
#include <folly/container/F14Map.h>
#include <folly/container/IntrusiveList.h>
#include <folly/io/async/Liburing.h>

#include "velox/common/file/Region.h"

#if FOLLY_HAS_LIBURING
#include <liburing.h>
#endif

namespace facebook::velox {

/// Synchronous positioned-read helper backed by one io_uring instance.
///
/// The reader owns the ring and is intended for batched reads from a single
/// submitting thread. It is not thread-safe.
class IoUringReader {
 public:
  static constexpr int32_t kDefaultQueueDepth{4'096};
  static constexpr int32_t kDefaultRegisteredFileSlots{8'192};

  struct Options {
    /// Number of submission queue entries requested for the io_uring instance.
    /// This controls how many read SQEs this reader prepares in one batch.
    /// Initialization fails if the kernel rejects the configured depth.
    int32_t submissionQueueDepth{kDefaultQueueDepth};

    /// Number of completion queue entries requested for the io_uring instance.
    /// Initialization fails if the kernel rejects the configured depth.
    int32_t completionQueueDepth{kDefaultQueueDepth};

    /// Number of registered-file slots to register for this io_uring instance.
    /// This bounds how many distinct fds can be reused without updating the
    /// kernel registered-file table.
    int32_t registeredFileSlots{kDefaultRegisteredFileSlots};

    /// Requests IORING_SETUP_SINGLE_ISSUER when supported. This tells the
    /// kernel one task submits to the ring, which matches per-thread readers.
    /// Set to false to initialize the ring without this kernel optimization.
    bool singleIssuer{true};

    /// Requests IORING_SETUP_COOP_TASKRUN when supported to let the submitter
    /// run deferred completion work cooperatively.
    bool coopTaskRun{true};

    /// Requests IORING_SETUP_DEFER_TASKRUN when supported. This requires
    /// singleIssuer=true and defers task work until the submitter enters the
    /// kernel. Configuring deferTaskRun=true with singleIssuer=false is
    /// invalid.
    bool deferTaskRun{true};
  };

  struct Stats {
    /// Number of read() calls issued through this reader.
    uint64_t readCalls{0};

    /// Total number of regions submitted across all read() calls.
    uint64_t regions{0};

    /// Smallest region count seen in one read() call.
    uint64_t minRegionsPerRead{std::numeric_limits<uint64_t>::max()};

    /// Largest region count seen in one read() call.
    uint64_t maxRegionsPerRead{0};

    /// Number of batches submitted to the ring.
    uint64_t batches{0};

    /// Smallest number of SQEs submitted in one batch.
    uint64_t minBatchSize{std::numeric_limits<uint64_t>::max()};

    /// Largest number of SQEs submitted in one batch.
    uint64_t maxBatchSize{0};

    /// Number of successful registered-file slot updates.
    uint64_t registeredFileUpdates{0};

    /// Adds counters from another stats snapshot.
    void merge(const Stats& other);

    /// Returns a compact string representation for logging.
    std::string toString() const;
  };

  explicit IoUringReader(Options options);
  ~IoUringReader();

  IoUringReader(const IoUringReader&) = delete;
  IoUringReader& operator=(const IoUringReader&) = delete;

  static bool available();

  /// Sets process-wide options used by subsequently constructed readers and
  /// threads that have not initialized their thread-local reader yet.
  static void setOptions(const Options& options);

  /// Returns a snapshot of this reader's stats.
  Stats stats() const;

  uint64_t read(
      int fd,
      folly::Range<const common::Region*> regions,
      folly::Range<const folly::Range<char*>*> buffers);

 private:
  friend Stats getIoUringReaderStats(uint64_t& numReaders);

  // Hook of system-wide reader stats list.
  struct IoUringReaderListEntry {
    IoUringReader* reader{nullptr};
    folly::IntrusiveListHook listHook;
  };
  using IoUringReaderList = folly::
      IntrusiveList<IoUringReaderListEntry, &IoUringReaderListEntry::listHook>;

  // Returns the system-wide reader stats list.
  static IoUringReaderList& readerList();

  // Returns the lock that protects the system-wide reader stats list.
  static folly::SharedMutex& readerListLock();

  // Options used to initialize this reader.
  const Options options_;

  // Read counters for this reader.
  Stats stats_;

  void registerStats();
  void unregisterStats();

  // Hook in the system-wide reader stats list.
  IoUringReaderListEntry readerListEntry_;

#if FOLLY_HAS_LIBURING
  void initializeRing();
  std::optional<int> registerFileForRead(int fd);
  void recordRead(size_t count);
  void recordBatch(size_t count);

  size_t prepareReads(
      int fd,
      folly::Range<const common::Region*> regions,
      folly::Range<const folly::Range<char*>*> buffers,
      size_t regionOffset,
      std::optional<int> registeredFileSlot);

  uint64_t submitAndWaitForReads(size_t count);

  struct RegisteredFile {
    // Raw fd number last registered in this registered-file slot.
    int fd{-1};

    // Device id from fstat() used to detect fd number reuse.
    uint64_t device{0};

    // Inode number from fstat() used to detect fd number reuse.
    uint64_t inode{0};

    // Open-file status flags from fcntl(F_GETFL).
    int statusFlags{0};
  };

  // Kernel ring owned by this reader.
  io_uring ring_{};

  // Actual submission queue depth returned by io_uring initialization.
  uint32_t submissionQueueDepth_{0};

  // Actual completion queue depth returned by io_uring initialization.
  uint32_t completionQueueDepth_{0};

  // Effective io_uring setup flags used by this reader.
  uint32_t setupFlags_{0};

  // Registered-file slots registered with the kernel for this ring.
  std::vector<RegisteredFile> registeredFiles_;

  // Map from raw fd number to the registered-file slot that currently holds it.
  folly::F14FastMap<int, size_t> fdToRegisteredFileSlot_;

  // Next registered-file slot to replace after all slots have been used.
  size_t nextRegisteredFileSlot_{0};

  // Whether the registered-file table has been registered with the kernel.
  bool fileTableInitialized_{false};
#endif
};

/// Returns best-effort merged stats across all live IoUringReader instances and
/// writes the number of live readers to 'numReaders'.
IoUringReader::Stats getIoUringReaderStats(uint64_t& numReaders);

/// Accessor for a process-wide facade that performs reads through a
/// thread-local IoUringReader.
class ThreadLocalIoUringReader {
 public:
  static ThreadLocalIoUringReader& get();

  /// Clears this thread's IoUringReader instance. Intended for tests that need
  /// isolated reader stats.
  static void testingClear();

  uint64_t read(
      int fd,
      folly::Range<const common::Region*> regions,
      folly::Range<const folly::Range<char*>*> buffers) const;
};

} // namespace facebook::velox
