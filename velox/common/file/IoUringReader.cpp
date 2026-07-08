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

#include "velox/common/file/IoUringReader.h"

#include "velox/common/base/Exceptions.h"

#ifdef _MSC_VER
// When liburing is unavailable (e.g. on Windows) the method bodies below are
// unreachable stubs that always throw via VELOX_UNSUPPORTED. MSVC does not treat
// it as [[noreturn]] here and emits C4716 ("must return a value").
#pragma warning(disable : 4716)
#endif

#include <algorithm>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

#include <folly/Conv.h>

#if FOLLY_HAS_LIBURING

#include <fcntl.h>
#include <sys/stat.h>
#include <cerrno>
#include <vector>

#include <glog/logging.h>

#include <folly/io/async/IoUringBackend.h>

#endif

namespace facebook::velox {

namespace {

void validateOptions(const IoUringReader::Options& options) {
  VELOX_CHECK_GT(
      options.submissionQueueDepth,
      0,
      "io_uring submissionQueueDepth must be positive");
  VELOX_CHECK_GT(
      options.completionQueueDepth,
      0,
      "io_uring completionQueueDepth must be positive");
  VELOX_CHECK_GT(
      options.registeredFileSlots,
      0,
      "io_uring registeredFileSlots must be positive");
  VELOX_CHECK(
      options.singleIssuer || !options.deferTaskRun,
      "io_uring deferTaskRun option requires singleIssuer option");
}

std::mutex& globalIoUringOptionsMutex() {
  static std::mutex mutex;
  return mutex;
}

IoUringReader::Options& globalIoUringOptions() {
  static IoUringReader::Options options;
  return options;
}

IoUringReader::Options getGlobalIoUringOptions() {
  std::lock_guard<std::mutex> l(globalIoUringOptionsMutex());
  return globalIoUringOptions();
}

std::unique_ptr<IoUringReader>& threadLocalIoUringReader() {
  static thread_local std::unique_ptr<IoUringReader> reader;
  return reader;
}

#if FOLLY_HAS_LIBURING

uint32_t singleIssuerIoUringSetupFlag() {
#ifdef IORING_SETUP_SINGLE_ISSUER
  return IORING_SETUP_SINGLE_ISSUER;
#else
  return 0;
#endif
}

uint32_t coopTaskRunIoUringSetupFlag() {
#ifdef IORING_SETUP_COOP_TASKRUN
  return IORING_SETUP_COOP_TASKRUN;
#else
  return 0;
#endif
}

uint32_t deferTaskRunIoUringSetupFlag() {
#ifdef IORING_SETUP_DEFER_TASKRUN
  return IORING_SETUP_DEFER_TASKRUN;
#else
  return 0;
#endif
}

uint32_t ioUringSetupFlagsFromOptions(const IoUringReader::Options& options) {
  uint32_t flags{0};
  if (options.singleIssuer) {
    flags |= singleIssuerIoUringSetupFlag();
  }
  if (options.coopTaskRun) {
    flags |= coopTaskRunIoUringSetupFlag();
  }
  if (options.deferTaskRun) {
    flags |= deferTaskRunIoUringSetupFlag();
  }
  return flags;
}

std::string ioUringSetupFlagsToString(uint32_t flags) {
  std::string text;
  auto append = [&](const char* name) {
    if (!text.empty()) {
      text += "|";
    }
    text += name;
  };

#ifdef IORING_SETUP_SINGLE_ISSUER
  if ((flags & IORING_SETUP_SINGLE_ISSUER) != 0) {
    append("SINGLE_ISSUER");
  }
#endif
#ifdef IORING_SETUP_COOP_TASKRUN
  if ((flags & IORING_SETUP_COOP_TASKRUN) != 0) {
    append("COOP_TASKRUN");
  }
#endif
#ifdef IORING_SETUP_DEFER_TASKRUN
  if ((flags & IORING_SETUP_DEFER_TASKRUN) != 0) {
    append("DEFER_TASKRUN");
  }
#endif
#ifdef IORING_SETUP_CQSIZE
  if ((flags & IORING_SETUP_CQSIZE) != 0) {
    append("CQSIZE");
  }
#endif
  if (text.empty()) {
    text = "none";
  }
  return folly::to<std::string>(text, " (raw=", flags, ")");
}

#ifdef IOSQE_FIXED_FILE
struct FileIdentity {
  uint64_t device{0};
  uint64_t inode{0};
  int statusFlags{0};
};

FileIdentity getFileIdentity(int fd) {
  VELOX_CHECK_GE(fd, 0, "io_uring read fd must be non-negative: {}", fd);

  FileIdentity identity;
  identity.statusFlags = ::fcntl(fd, F_GETFL);
  VELOX_CHECK_GE(
      identity.statusFlags,
      0,
      "fcntl(F_GETFL) failed for io_uring read fd {}: {}",
      fd,
      folly::errnoStr(errno));

  struct stat fileStat{};
  VELOX_CHECK_EQ(
      ::fstat(fd, &fileStat),
      0,
      "fstat failed for io_uring read fd {}: {}",
      fd,
      folly::errnoStr(errno));
  identity.device = static_cast<uint64_t>(fileStat.st_dev);
  identity.inode = static_cast<uint64_t>(fileStat.st_ino);
  return identity;
}
#endif

#endif

} // namespace

void IoUringReader::Stats::merge(const Stats& other) {
  readCalls += other.readCalls;
  regions += other.regions;
  if (other.readCalls > 0) {
    minRegionsPerRead = std::min(minRegionsPerRead, other.minRegionsPerRead);
    maxRegionsPerRead = std::max(maxRegionsPerRead, other.maxRegionsPerRead);
  }

  batches += other.batches;
  if (other.batches > 0) {
    minBatchSize = std::min(minBatchSize, other.minBatchSize);
    maxBatchSize = std::max(maxBatchSize, other.maxBatchSize);
  }

  registeredFileUpdates += other.registeredFileUpdates;
}

std::string IoUringReader::Stats::toString() const {
  const auto avgRegionsPerRead = readCalls == 0 ? 0 : regions / readCalls;
  const auto avgBatchSize = batches == 0 ? 0 : regions / batches;
  return folly::to<std::string>(
      "readCalls=",
      readCalls,
      ", regions=",
      regions,
      ", regionsPerReadAvg=",
      avgRegionsPerRead,
      ", regionsPerReadMin=",
      readCalls == 0 ? 0 : minRegionsPerRead,
      ", regionsPerReadMax=",
      maxRegionsPerRead,
      ", batches=",
      batches,
      ", batchSizeAvg=",
      avgBatchSize,
      ", batchSizeMin=",
      batches == 0 ? 0 : minBatchSize,
      ", batchSizeMax=",
      maxBatchSize,
      ", registeredFileUpdates=",
      registeredFileUpdates);
}

IoUringReader::IoUringReaderList& IoUringReader::readerList() {
  static IoUringReaderList readerList;
  return readerList;
}

folly::SharedMutex& IoUringReader::readerListLock() {
  static folly::SharedMutex lock;
  return lock;
}

void IoUringReader::registerStats() {
  VELOX_CHECK(!readerListEntry_.listHook.is_linked());
  readerListEntry_.reader = this;

  std::unique_lock guard{readerListLock()};
  readerList().push_back(readerListEntry_);
}

void IoUringReader::unregisterStats() {
  std::unique_lock guard{readerListLock()};
  if (readerListEntry_.listHook.is_linked()) {
    readerListEntry_.listHook.unlink();
  }
  readerListEntry_.reader = nullptr;
}

IoUringReader::Stats getIoUringReaderStats(uint64_t& numReaders) {
  std::shared_lock guard{IoUringReader::readerListLock()};
  numReaders = 0;
  IoUringReader::Stats stats;
  for (const auto& readerEntry : IoUringReader::readerList()) {
    ++numReaders;
    stats.merge(readerEntry.reader->stats());
  }
  return stats;
}

#if FOLLY_HAS_LIBURING

bool IoUringReader::available() {
  return folly::IoUringBackend::isAvailable();
}

#else

bool IoUringReader::available() {
  return false;
}

#endif

void IoUringReader::setOptions(const Options& options) {
  validateOptions(options);
  {
    std::lock_guard<std::mutex> l(globalIoUringOptionsMutex());
    globalIoUringOptions() = options;
  }
#if FOLLY_HAS_LIBURING
  LOG(INFO) << "IoUringReader options set: submissionQueueDepth="
            << options.submissionQueueDepth
            << ", completionQueueDepth=" << options.completionQueueDepth
            << ", registeredFileSlots=" << options.registeredFileSlots
            << ", setupFlags="
            << ioUringSetupFlagsToString(ioUringSetupFlagsFromOptions(options));
#endif
}

IoUringReader::IoUringReader(Options options) : options_{options} {
  validateOptions(options_);
#if FOLLY_HAS_LIBURING
  initializeRing();
  registerStats();
#else
  VELOX_UNSUPPORTED("io_uring is unavailable");
#endif
}

IoUringReader::~IoUringReader() {
  unregisterStats();
#if FOLLY_HAS_LIBURING
  if (fileTableInitialized_) {
    ::io_uring_unregister_files(&ring_);
  }
  ::io_uring_queue_exit(&ring_);
#endif
}

IoUringReader::Stats IoUringReader::stats() const {
  return stats_;
}

uint64_t IoUringReader::read(
    int fd,
    folly::Range<const common::Region*> regions,
    folly::Range<const folly::Range<char*>*> buffers) {
#if FOLLY_HAS_LIBURING
  recordRead(regions.size());
  const auto registeredFileSlot = registerFileForRead(fd);

  uint64_t totalBytes{0};
  size_t next{0};
  while (next < regions.size()) {
    const auto prepared =
        prepareReads(fd, regions, buffers, next, registeredFileSlot);
    totalBytes += submitAndWaitForReads(prepared);
    next += prepared;
  }
  return totalBytes;
#else
  (void)fd;
  (void)regions;
  (void)buffers;
  VELOX_UNSUPPORTED("io_uring is unavailable");
#endif
}

#if FOLLY_HAS_LIBURING

void IoUringReader::initializeRing() {
  registeredFiles_.resize(static_cast<size_t>(options_.registeredFileSlots));
  fdToRegisteredFileSlot_.reserve(
      static_cast<size_t>(options_.registeredFileSlots));

  io_uring_params params{};
  params.flags = ioUringSetupFlagsFromOptions(options_);
#ifdef IORING_SETUP_CQSIZE
  params.flags |= IORING_SETUP_CQSIZE;
  params.cq_entries = static_cast<uint32_t>(options_.completionQueueDepth);
#endif

  const int ret = ::io_uring_queue_init_params(
      static_cast<unsigned int>(options_.submissionQueueDepth),
      &ring_,
      &params);
  if (ret < 0) {
    VELOX_FAIL("io_uring_queue_init failed: {}", folly::errnoStr(-ret));
  }

  submissionQueueDepth_ = params.sq_entries;
  completionQueueDepth_ = params.cq_entries;
  setupFlags_ = params.flags;
  if (submissionQueueDepth_ == 0 || completionQueueDepth_ == 0) {
    ::io_uring_queue_exit(&ring_);
    VELOX_FAIL(
        "io_uring initialized with invalid queue depths: submissionQueueDepth={}, completionQueueDepth={}",
        submissionQueueDepth_,
        completionQueueDepth_);
  }

#ifdef IOSQE_FIXED_FILE
  std::vector<int> files(registeredFiles_.size(), -1);
  const int registerRet = ::io_uring_register_files(
      &ring_, files.data(), static_cast<unsigned int>(files.size()));
  if (registerRet < 0) {
    ::io_uring_queue_exit(&ring_);
    VELOX_FAIL(
        "io_uring_register_files failed: {}", folly::errnoStr(-registerRet));
  }
  fileTableInitialized_ = true;
#endif

  LOG(INFO) << "IoUringReader initialized: submissionQueueDepth="
            << submissionQueueDepth_
            << ", completionQueueDepth=" << completionQueueDepth_
            << ", registeredFileSlots=" << registeredFiles_.size()
            << ", setupFlags=" << ioUringSetupFlagsToString(setupFlags_);
}

std::optional<int> IoUringReader::registerFileForRead(int fd) {
#ifdef IOSQE_FIXED_FILE
  VELOX_CHECK_GE(fd, 0, "io_uring read fd must be non-negative: {}", fd);
  VELOX_CHECK(
      fileTableInitialized_,
      "io_uring registered-file table is not initialized");

  const auto identity = getFileIdentity(fd);

  size_t slot;
  auto it = fdToRegisteredFileSlot_.find(fd);
  if (it != fdToRegisteredFileSlot_.end()) {
    slot = it->second;
    const auto& registeredFile = registeredFiles_[slot];
    if (registeredFile.fd == fd && registeredFile.device == identity.device &&
        registeredFile.inode == identity.inode &&
        registeredFile.statusFlags == identity.statusFlags) {
      return static_cast<int>(slot);
    }
    fdToRegisteredFileSlot_.erase(it);
  } else {
    slot = nextRegisteredFileSlot_;
  }

  // The selected slot may still have a reverse lookup for the fd it is about
  // to evict. Remove that stale mapping before storing the new fd in the slot.
  if (registeredFiles_[slot].fd >= 0) {
    auto oldIt = fdToRegisteredFileSlot_.find(registeredFiles_[slot].fd);
    if (oldIt != fdToRegisteredFileSlot_.end() && oldIt->second == slot) {
      fdToRegisteredFileSlot_.erase(oldIt);
    }
  }

  const int ret = ::io_uring_register_files_update(
      &ring_, static_cast<unsigned int>(slot), &fd, 1);
  if (ret != 1) {
    if (ret < 0) {
      VELOX_FAIL(
          "io_uring_register_files_update failed for fd {} slot {}: {}",
          fd,
          slot,
          folly::errnoStr(-ret));
    }
    VELOX_FAIL(
        "io_uring_register_files_update unexpectedly updated {} entries for fd {} slot {}",
        ret,
        fd,
        slot);
  }

  registeredFiles_[slot].fd = fd;
  registeredFiles_[slot].device = identity.device;
  registeredFiles_[slot].inode = identity.inode;
  registeredFiles_[slot].statusFlags = identity.statusFlags;
  fdToRegisteredFileSlot_[fd] = slot;
  // Wrap around and evict slots in round-robin order once the table is full.
  nextRegisteredFileSlot_ = (slot + 1) % registeredFiles_.size();
  ++stats_.registeredFileUpdates;
  return static_cast<int>(slot);
#else
  return std::nullopt;
#endif
}

void IoUringReader::recordRead(size_t count) {
  ++stats_.readCalls;
  stats_.regions += count;
  stats_.minRegionsPerRead =
      std::min<uint64_t>(stats_.minRegionsPerRead, count);
  stats_.maxRegionsPerRead =
      std::max<uint64_t>(stats_.maxRegionsPerRead, count);
}

void IoUringReader::recordBatch(size_t count) {
  ++stats_.batches;
  stats_.minBatchSize = std::min<uint64_t>(stats_.minBatchSize, count);
  stats_.maxBatchSize = std::max<uint64_t>(stats_.maxBatchSize, count);
}

size_t IoUringReader::prepareReads(
    int fd,
    folly::Range<const common::Region*> regions,
    folly::Range<const folly::Range<char*>*> buffers,
    size_t regionOffset,
    std::optional<int> registeredFileSlot) {
  const auto count =
      std::min<size_t>(submissionQueueDepth_, regions.size() - regionOffset);
  VELOX_CHECK_GT(count, 0, "io_uring read batch must not be empty");
  recordBatch(count);
  for (size_t i = 0; i < count; ++i) {
    const auto index = regionOffset + i;
    const auto& region = regions[index];
    const auto& buffer = buffers[index];
    VELOX_CHECK_LE(
        buffer.size(),
        static_cast<size_t>(std::numeric_limits<int>::max()),
        "preadv read length exceeds io_uring result range");

    auto* sqe = ::io_uring_get_sqe(&ring_);
    VELOX_CHECK_NOT_NULL(sqe, "io_uring_get_sqe failed");
    ::io_uring_prep_read(
        sqe,
        registeredFileSlot.has_value() ? *registeredFileSlot : fd,
        buffer.data(),
        static_cast<unsigned int>(buffer.size()),
        region.offset);
#ifdef IOSQE_FIXED_FILE
    if (registeredFileSlot.has_value()) {
      sqe->flags |= IOSQE_FIXED_FILE;
    }
#endif
  }
  return count;
}

uint64_t IoUringReader::submitAndWaitForReads(size_t count) {
  // submit_and_wait() returns the number of SQEs submitted. wait_nr only
  // controls how many CQEs to wait for before returning; the loop below drains
  // and validates all completions for the submitted batch.
  const auto waitCount = std::min<size_t>(count, completionQueueDepth_);
  const int ret =
      ::io_uring_submit_and_wait(&ring_, static_cast<unsigned int>(waitCount));
  VELOX_CHECK_GE(
      ret, 0, "io_uring_submit_and_wait failed: {}", folly::errnoStr(-ret));
  VELOX_CHECK_EQ(
      ret, count, "io_uring_submit_and_wait submitted partial batch");

  uint64_t totalBytes{0};
  int error{0};
  size_t completed{0};
  while (completed < count) {
    io_uring_cqe* cqe{nullptr};
    int ret = ::io_uring_peek_cqe(&ring_, &cqe);
    if (ret == -EAGAIN) {
      const auto waitCount =
          std::min<size_t>(count - completed, completionQueueDepth_);
      ret = ::io_uring_wait_cqes(
          &ring_, &cqe, static_cast<unsigned>(waitCount), nullptr, nullptr);
    }
    VELOX_CHECK_GE(
        ret, 0, "io_uring completion wait failed: {}", folly::errnoStr(-ret));

    unsigned head;
    unsigned seen{0};
    io_uring_for_each_cqe(&ring_, head, cqe) {
      ++seen;
      if (cqe->res < 0 && error == 0) {
        error = cqe->res;
      } else if (cqe->res >= 0) {
        totalBytes += static_cast<uint64_t>(cqe->res);
      }
    }
    VELOX_CHECK_GT(seen, 0, "io_uring_wait_cqe returned no completions");
    ::io_uring_cq_advance(&ring_, seen);
    completed += seen;
    VELOX_CHECK_LE(completed, count, "io_uring completed too many reads");
  }

  VELOX_CHECK_GE(
      error, 0, "io_uring batch pread failed: {}", folly::errnoStr(-error));
  return totalBytes;
}

#endif

ThreadLocalIoUringReader& ThreadLocalIoUringReader::get() {
  VELOX_CHECK(
      IoUringReader::available(),
      "io_uring batch reads requested but io_uring is unavailable");
  static ThreadLocalIoUringReader reader;
  return reader;
}

void ThreadLocalIoUringReader::testingClear() {
  threadLocalIoUringReader().reset();
}

uint64_t ThreadLocalIoUringReader::read(
    int fd,
    folly::Range<const common::Region*> regions,
    folly::Range<const folly::Range<char*>*> buffers) const {
  auto& reader = threadLocalIoUringReader();
  if (reader == nullptr) {
    reader = std::make_unique<IoUringReader>(getGlobalIoUringOptions());
  }
  return reader->read(fd, regions, buffers);
}

} // namespace facebook::velox
