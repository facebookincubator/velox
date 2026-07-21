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

// Tests for the async-prefetch code path in ParquetReader
// (scheduleRowGroups + per-RG InputSlot promises + budget admission).
//
// These tests do not depend on a native-async storage backend: they drive
// the parquet scheduler through a plain BufferedInput that returns an IO
// executor from executor(), which is enough to exercise the cross-thread
// promise/await machinery, the budget admission gate, slot erase, and
// fault propagation. FailingReadFile additionally reports
// hasPreadvAsync()=true so the wiring can be extended to exercise the
// native-async submission path end-to-end.

#include <atomic>
#include <filesystem>
#include <thread>

#include <folly/executors/IOThreadPoolExecutor.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "velox/common/io/IoExecutorThreadRegistry.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"

namespace facebook::velox::parquet {
namespace {

using dwio::common::BufferedInput;
using dwio::common::ReaderOptions;
using dwio::common::RowReaderOptions;

// BufferedInput that returns a caller-provided executor from executor().
// A plain BufferedInput returns nullptr, so its scheduleRowGroups Phase 2
// continuation runs inline on the consumer thread. Returning a non-null
// executor makes the scheduler post the continuation onto that executor,
// exercising the cross-thread promise-fulfill / consumer-await path.
class AsyncBufferedInput : public BufferedInput {
 public:
  AsyncBufferedInput(
      std::shared_ptr<ReadFile> readFile,
      memory::MemoryPool& pool,
      folly::Executor* executor)
      : BufferedInput(std::move(readFile), pool), executor_(executor) {}

  folly::Executor* executor() const override {
    return executor_;
  }

 private:
  folly::Executor* const executor_;
};

// Wraps a real LocalReadFile. While `armed_`, any read whose byte range
// intersects [failOffset_, failOffset_ + failLength_) throws. armed_
// defaults to false so reader-construction reads (footer/metadata) are
// undisturbed.
//
// hasPreadvAsync() returns true so this file can be wired through
// CachedBufferedInput / DirectBufferedInput to exercise the native-async
// submission path; the default (inline) preadvAsync completes on the
// calling thread.
class FailingReadFile : public ReadFile {
 public:
  explicit FailingReadFile(const std::string& path)
      : inner_(std::make_shared<LocalReadFile>(path)) {}

  void failOnRange(int64_t offset, int64_t length, std::string message) {
    failOffset_ = offset;
    failLength_ = length;
    failMessage_ = std::move(message);
  }

  void arm() {
    armed_.store(true, std::memory_order_release);
  }

  int64_t armedReadCount() const {
    return armedReadCount_.load(std::memory_order_relaxed);
  }

  int64_t armedThrowCount() const {
    return armedThrowCount_.load(std::memory_order_relaxed);
  }

  int64_t preadvAsyncCallCount() const {
    return preadvAsyncCallCount_.load(std::memory_order_relaxed);
  }

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& ctx = {}) const override {
    maybeThrow(offset, length);
    return inner_->pread(offset, length, buf, ctx);
  }

  std::string pread(
      uint64_t offset,
      uint64_t length,
      const FileIoContext& ctx = {}) const override {
    maybeThrow(offset, length);
    return inner_->pread(offset, length, ctx);
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& ctx = {}) const override {
    uint64_t total = 0;
    for (const auto& b : buffers) {
      total += b.size();
    }
    maybeThrow(offset, total);
    return inner_->preadv(offset, buffers, ctx);
  }

  folly::SemiFuture<uint64_t> preadvAsync(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& ctx = {}) const override {
    preadvAsyncCallCount_.fetch_add(1, std::memory_order_relaxed);
    try {
      return folly::makeSemiFuture<uint64_t>(preadv(offset, buffers, ctx));
    } catch (const std::exception& e) {
      return folly::makeSemiFuture<uint64_t>(
          folly::exception_wrapper{std::current_exception()});
    }
  }

  bool hasPreadvAsync() const override {
    return true;
  }

  bool shouldCoalesce() const override {
    return inner_->shouldCoalesce();
  }

  uint64_t size() const override {
    return inner_->size();
  }

  uint64_t memoryUsage() const override {
    return inner_->memoryUsage();
  }

  std::string getName() const override {
    return "FailingReadFile(" + inner_->getName() + ")";
  }

  uint64_t getNaturalReadSize() const override {
    return inner_->getNaturalReadSize();
  }

 private:
  bool intersects(int64_t offset, int64_t length) const {
    if (!armed_.load(std::memory_order_acquire) || failLength_ <= 0) {
      return false;
    }
    armedReadCount_.fetch_add(1, std::memory_order_relaxed);
    const int64_t aEnd = offset + length;
    const int64_t bEnd = failOffset_ + failLength_;
    return offset < bEnd && failOffset_ < aEnd;
  }

  void maybeThrow(int64_t offset, int64_t length) const {
    if (intersects(offset, length)) {
      armedThrowCount_.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error(failMessage_);
    }
  }

  std::shared_ptr<ReadFile> inner_;
  int64_t failOffset_{0};
  int64_t failLength_{0};
  std::string failMessage_;
  std::atomic<bool> armed_{false};
  mutable std::atomic<int64_t> armedReadCount_{0};
  mutable std::atomic<int64_t> armedThrowCount_{0};
  mutable std::atomic<int64_t> preadvAsyncCallCount_{0};
};

class ParquetReaderAsyncTest : public ParquetTestBase {
 protected:
  void SetUp() override {
    ParquetTestBase::SetUp();
    // Single-thread executor for deterministic Phase 2 continuation
    // ordering. Registered so awaitRowGroupReady's isOnIoExecutorThread
    // DCHECK is meaningful.
    ioExecutor_ = std::make_shared<folly::IOThreadPoolExecutor>(1);
    io::IoExecutorThreadRegistry::instance().registerExecutor(
        ioExecutor_.get(), /*tasksPerThread=*/4);
  }

  void TearDown() override {
    ioExecutor_->stop();
    ioExecutor_.reset();
  }

  struct AsyncReaderHandle {
    std::unique_ptr<ParquetReader> reader;
    BufferedInput* input; // non-owning; lifetime tied to `reader`
  };

  AsyncReaderHandle makeAsyncReader(
      std::shared_ptr<ReadFile> readFile,
      const ReaderOptions& opts,
      folly::Executor* executor = nullptr) {
    auto* exec = executor ? executor : ioExecutor_.get();
    auto input =
        std::make_unique<AsyncBufferedInput>(readFile, opts.memoryPool(), exec);
    auto* inputPtr = input.get();
    auto reader = std::make_unique<ParquetReader>(std::move(input), opts);
    return {std::move(reader), inputPtr};
  }

  AsyncReaderHandle makeAsyncReader(
      const std::string& path,
      const ReaderOptions& opts,
      folly::Executor* executor = nullptr) {
    return makeAsyncReader(
        std::make_shared<LocalReadFile>(path), opts, executor);
  }

  uint64_t readAllRows(
      ParquetReader& reader,
      const RowTypePtr& rowType,
      uint64_t batchSize = 1000) {
    RowReaderOptions rowReaderOpts;
    rowReaderOpts.setScanSpec(makeScanSpec(rowType));
    auto rowReader = reader.createRowReader(rowReaderOpts);
    uint64_t total = 0;
    auto result = BaseVector::create(rowType, batchSize, pool_);
    while (true) {
      const auto got = rowReader->next(batchSize, result);
      if (got == 0) {
        break;
      }
      total += got;
    }
    return total;
  }

  // Writes a parquet file large enough to bypass the whole-file preload
  // (fileLength_ <= max(filePreloadThreshold, footerSpeculativeIoSize),
  // where footerSpeculativeIoSize defaults to 256 KB). Returns the path.
  // A large file forces the scheduler down the batched clone+load path
  // rather than the in-place cache-hit path.
  std::string writeLargeParquetFile(
      const RowTypePtr& rowType,
      uint64_t numBatches,
      uint64_t rowsPerBatch,
      uint64_t rowsPerRowGroup,
      const std::string& nameTag) {
    auto batches = createBatches(rowType, numBatches, rowsPerBatch);
    const std::string path = tempPath_->getPath() + "/" + nameTag + ".parquet";
    auto sink = std::make_unique<dwio::common::LocalFileSink>(
        path, dwio::common::FileSink::Options{.pool = leafPool_.get()});
    WriterOptions options;
    options.memoryPool = rootPool_.get();
    options.flushPolicyFactory = [rowsPerRowGroup]() {
      return std::make_unique<DefaultFlushPolicy>(
          rowsPerRowGroup, /*bytesInRowGroup=*/1L << 30);
    };
    auto writer = std::make_unique<Writer>(std::move(sink), options, rowType);
    for (const auto& b : batches) {
      writer->write(b);
    }
    writer->close();
    return path;
  }

  std::shared_ptr<folly::IOThreadPoolExecutor> ioExecutor_;
};

// Async scheduler path reads all rows from a multi-RG sample file.
TEST_F(ParquetReaderAsyncTest, smokeReadsAllRows) {
  const std::string sample = getExampleFilePath("multiple_row_groups.parquet");
  auto rowType = ROW({"id"}, {BIGINT()});

  ReaderOptions opts{leafPool_.get()};
  opts.setFilePreloadThreshold(0);
  opts.setPrefetchRowGroups(4);
  opts.setIOExecutor(ioExecutor_.get());

  auto h = makeAsyncReader(sample, opts);
  EXPECT_EQ(h.reader->fileMetaData().numRowGroups(), 4);
  const uint64_t total = readAllRows(*h.reader, rowType);
  EXPECT_EQ(total, h.reader->numberOfRows().value());
  EXPECT_GT(total, 0u);
}

// Budget cap = 1 byte: the required-now RG is admitted; speculative RGs are
// refused by the admission gate; the read still completes (the window keeps
// advancing as the consumer drains earlier RGs).
TEST_F(ParquetReaderAsyncTest, budgetCapAdmitsRequiredRowGroup) {
  const std::string sample = getExampleFilePath("multiple_row_groups.parquet");
  auto rowType = ROW({"id"}, {BIGINT()});

  ReaderOptions opts{leafPool_.get()};
  opts.setFilePreloadThreshold(0);
  opts.setPrefetchRowGroups(4);
  opts.setIOExecutor(ioExecutor_.get());

  auto h = makeAsyncReader(sample, opts);
  // A plain BufferedInput does not auto-propagate the cap from
  // ReaderOptions (only Cached/Direct do); set it directly.
  h.input->setMaxOutstandingPrefetchBytes(1);
  const uint64_t total = readAllRows(*h.reader, rowType);
  EXPECT_EQ(total, h.reader->numberOfRows().value());
}

// Small prefetch window forces repeated scheduleRowGroups calls whose
// windows overlap the prior call. Each subsequent call must hit the
// try_emplace skip path for already-scheduled RGs without double-fulfilling
// a promise or losing an RG.
TEST_F(ParquetReaderAsyncTest, overlappingWindowsStable) {
  const std::string sample = getExampleFilePath("multiple_row_groups.parquet");
  auto rowType = ROW({"id"}, {BIGINT()});

  ReaderOptions opts{leafPool_.get()};
  opts.setFilePreloadThreshold(0);
  opts.setPrefetchRowGroups(2);
  opts.setIOExecutor(ioExecutor_.get());

  auto h = makeAsyncReader(sample, opts);
  const uint64_t expected = h.reader->numberOfRows().value();
  const uint64_t total = readAllRows(*h.reader, rowType, /*batchSize=*/100);
  EXPECT_EQ(total, expected);
}

// As the consumer advances, the current RG is always buffered and old slots
// are erased. A regression in the erase ordering (or the isRowGroupBuffered
// readiness semantic) surfaces as a failed EXPECT here.
TEST_F(ParquetReaderAsyncTest, slotEraseTracksConsumerAdvance) {
  const std::string sample = getExampleFilePath("multiple_row_groups.parquet");
  auto rowType = ROW({"id"}, {BIGINT()});

  ReaderOptions opts{leafPool_.get()};
  opts.setFilePreloadThreshold(0);
  opts.setPrefetchRowGroups(2);
  opts.setIOExecutor(ioExecutor_.get());

  auto h = makeAsyncReader(sample, opts);
  const int numRowGroups = h.reader->fileMetaData().numRowGroups();
  ASSERT_EQ(numRowGroups, 4);

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  auto rowReader = h.reader->createRowReader(rowReaderOpts);
  auto* parquetRowReader = dynamic_cast<ParquetRowReader*>(rowReader.get());
  ASSERT_NE(parquetRowReader, nullptr);

  constexpr int kBatchSize = 1000;
  auto result = BaseVector::create(rowType, kBatchSize, pool_);
  for (int k = 0; k < numRowGroups; ++k) {
    EXPECT_TRUE(parquetRowReader->isRowGroupBuffered(k))
        << "RG " << k << " not buffered when consumer is about to read it";
    parquetRowReader->next(kBatchSize, result);
    parquetRowReader->nextRowNumber();
  }
}

// A file too large to preload forces the scheduler down the batched
// clone+load path (cache-miss). Verifies the batched path reads all rows.
TEST_F(ParquetReaderAsyncTest, batchedPathReadsAllRows) {
  auto rowType = ROW({"id"}, {BIGINT()});
  const std::string path = writeLargeParquetFile(
      rowType,
      /*numBatches=*/8,
      /*rowsPerBatch=*/64'000,
      /*rowsPerRowGroup=*/64'000,
      "batched_path");

  ReaderOptions opts{leafPool_.get()};
  opts.setFilePreloadThreshold(0);
  opts.setPrefetchRowGroups(4);
  opts.setIOExecutor(ioExecutor_.get());

  auto h = makeAsyncReader(path, opts);
  ASSERT_GT(h.reader->fileMetaData().numRowGroups(), 1);
  const uint64_t total = readAllRows(*h.reader, rowType);
  EXPECT_EQ(total, h.reader->numberOfRows().value());
  EXPECT_GT(total, 0u);
}

// A backend that throws on data reads (armed after construction) must
// propagate the failure to the consumer rather than hang or silently
// return short. Exercises the batched-load shared-fate + the per-slot
// setException wake-up path.
TEST_F(ParquetReaderAsyncTest, loadFailurePropagatesToConsumer) {
  auto rowType = ROW({"id"}, {BIGINT()});
  const std::string path = writeLargeParquetFile(
      rowType,
      /*numBatches=*/8,
      /*rowsPerBatch=*/64'000,
      /*rowsPerRowGroup=*/64'000,
      "load_failure");

  auto failing = std::make_shared<FailingReadFile>(path);
  const int64_t fileSize = static_cast<int64_t>(failing->size());
  // Fail any read in the first 80% of the file (the data region); the
  // footer sits at the end and is read during construction while
  // armed_==false.
  failing->failOnRange(0, (fileSize * 4) / 5, "injected backend failure");

  ReaderOptions opts{leafPool_.get()};
  opts.setFilePreloadThreshold(0);
  opts.setPrefetchRowGroups(4);
  opts.setIOExecutor(ioExecutor_.get());

  auto h = makeAsyncReader(std::static_pointer_cast<ReadFile>(failing), opts);
  failing->arm();
  bool threw = false;
  try {
    (void)readAllRows(*h.reader, rowType);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_GT(failing->armedReadCount(), 0)
      << "no reads observed after arm -- file likely preloaded; increase "
         "the row count in writeLargeParquetFile";
  EXPECT_TRUE(threw) << "backend threw " << failing->armedThrowCount()
                     << " times but the consumer did not observe an exception";
}

// Multi-iteration smoke under a 4-thread IO executor. Guards against the
// cache-hit-vs-batched-clone race: any data race on the shared ParquetData
// vectors tends to surface within tens of iterations on x64.
TEST_F(ParquetReaderAsyncTest, concurrentStressIterations) {
  const std::string sample = getExampleFilePath("multiple_row_groups.parquet");
  auto rowType = ROW({"id"}, {BIGINT()});

  auto multiExec = std::make_shared<folly::IOThreadPoolExecutor>(4);
  io::IoExecutorThreadRegistry::instance().registerExecutor(
      multiExec.get(), /*tasksPerThread=*/4);

  ReaderOptions opts{leafPool_.get()};
  opts.setFilePreloadThreshold(0);
  opts.setPrefetchRowGroups(4);
  opts.setIOExecutor(multiExec.get());

  constexpr int kIterations = 20;
  uint64_t expected = 0;
  for (int i = 0; i < kIterations; ++i) {
    auto h = makeAsyncReader(sample, opts, multiExec.get());
    if (i == 0) {
      expected = h.reader->numberOfRows().value();
    }
    const uint64_t total = readAllRows(*h.reader, rowType);
    ASSERT_EQ(total, expected) << "iteration " << i;
  }
  multiExec->stop();
}

} // namespace
} // namespace facebook::velox::parquet
