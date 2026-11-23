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
#include "velox/common/memory/ByteStream.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileInputStream.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/common/testutil/TempDirectoryPath.h"

#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>
#include <atomic>
#include <deque>
#include <functional>
#include <future>
#include <stdexcept>

#include <chrono>
#include <thread>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::memory;

namespace {
// Models a file whose read starts when its deferred future is activated.
struct DeferredReadState {
  std::atomic_uint32_t readExecutions{0};
  folly::Baton<> readStarted;
  folly::Baton<> continueRead;
  folly::Baton<> streamDestructionStarted;
  folly::Baton<> fileDestroyed;
  std::promise<void> nativeReadQueued;
  std::atomic_bool nativeFileDestroyed{false};
  std::thread::id readThreadId;
};

class DeferredReadFile : public InMemoryReadFile {
 public:
  DeferredReadFile(
      std::string data,
      bool blockRead,
      bool failRead = false,
      std::shared_ptr<DeferredReadState> state =
          std::make_shared<DeferredReadState>())
      : InMemoryReadFile(std::move(data)),
        blockRead_(blockRead),
        failRead_(failRead),
        state_(std::move(state)) {}

  ~DeferredReadFile() override {
    state_->fileDestroyed.post();
  }

  bool hasPreadvAsync() const override {
    return true;
  }

  folly::SemiFuture<uint64_t> preadvAsync(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const override {
    return folly::makeSemiFuture().deferValue(
        [this, offset, buffers, context](auto&&) -> uint64_t {
          const auto execution =
              state_->readExecutions.fetch_add(1, std::memory_order_relaxed);
          if (execution == 0) {
            state_->readThreadId = std::this_thread::get_id();
            state_->readStarted.post();
          }
          if (blockRead_) {
            state_->continueRead.wait();
          }
          if (failRead_) {
            VELOX_FAIL("deferred read failed");
          }
          uint64_t readBytes{0};
          for (const auto& buffer : buffers) {
            InMemoryReadFile::pread(
                offset + readBytes, buffer.size(), buffer.data(), context);
            readBytes += buffer.size();
          }
          return readBytes;
        });
  }

  bool waitForReadStart() const {
    return state_->readStarted.try_wait_for(std::chrono::seconds(5));
  }

  std::thread::id readThreadId() const {
    return state_->readThreadId;
  }

 private:
  const bool blockRead_;
  const bool failRead_;
  const std::shared_ptr<DeferredReadState> state_;
};

// Models native asynchronous I/O queued to the same executor as read-ahead.
class NativeAsyncReadFile : public InMemoryReadFile {
 public:
  enum class ReadBehavior { kRead, kFail, kCompleteWithoutRead };

  NativeAsyncReadFile(
      std::string data,
      folly::Executor* executor,
      std::shared_ptr<DeferredReadState> state,
      ReadBehavior behavior = ReadBehavior::kRead,
      bool blockFutureReturn = false)
      : InMemoryReadFile(std::move(data)),
        executor_(executor),
        state_(std::move(state)),
        behavior_(behavior),
        blockFutureReturn_(blockFutureReturn) {}

  ~NativeAsyncReadFile() override {
    state_->nativeFileDestroyed = true;
    state_->fileDestroyed.post();
  }

  bool hasPreadvAsync() const override {
    return true;
  }

  folly::SemiFuture<uint64_t> preadvAsync(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const override {
    auto [readPromise, future] = folly::makePromiseContract<uint64_t>();
    if (behavior_ == ReadBehavior::kCompleteWithoutRead) {
      uint64_t readBytes{0};
      for (const auto& buffer : buffers) {
        readBytes += buffer.size();
      }
      // Keep the lifetime test safe even if it detects a released file or
      // buffer.
      executor_->add([state = state_,
                      readBytes,
                      promise = std::move(readPromise)]() mutable {
        state->readExecutions.fetch_add(1, std::memory_order_relaxed);
        promise.setValue(readBytes);
      });
    } else {
      executor_->add([this,
                      offset,
                      buffers = buffers,
                      context = context,
                      promise = std::move(readPromise)]() mutable {
        state_->readExecutions.fetch_add(1, std::memory_order_relaxed);
        if (behavior_ == ReadBehavior::kFail) {
          promise.setException(std::runtime_error("native read failed"));
          return;
        }
        uint64_t readBytes{0};
        for (const auto& buffer : buffers) {
          InMemoryReadFile::pread(
              offset + readBytes, buffer.size(), buffer.data(), context);
          readBytes += buffer.size();
        }
        promise.setValue(readBytes);
      });
    }
    // Synchronizes access to the test executor's queue after add() completes.
    state_->nativeReadQueued.set_value();
    if (blockFutureReturn_) {
      state_->continueRead.wait();
    }
    return std::move(future);
  }

 private:
  folly::Executor* const executor_;
  const std::shared_ptr<DeferredReadState> state_;
  const ReadBehavior behavior_;
  const bool blockFutureReturn_;
};

// Signals destruction before the base stream drains pending reads.
class DestructionSignalingFileInputStream : public common::FileInputStream {
 public:
  DestructionSignalingFileInputStream(
      std::unique_ptr<ReadFile>&& file,
      uint64_t bufferSize,
      memory::MemoryPool* pool,
      folly::Executor* readAheadExecutor,
      std::shared_ptr<DeferredReadState> state)
      : FileInputStream(std::move(file), bufferSize, pool, readAheadExecutor),
        state_(std::move(state)) {}

  ~DestructionSignalingFileInputStream() override {
    state_->streamDestructionStarted.post();
  }

 private:
  const std::shared_ptr<DeferredReadState> state_;
};

// Lets tests decide when queued work makes progress.
class QueuedExecutor : public folly::Executor {
 public:
  void add(folly::Func function) override {
    functions_.wlock()->push_back(std::move(function));
  }

  size_t size() const {
    return functions_.rlock()->size();
  }

  void runNext() {
    popNext()();
  }

 private:
  // Release the queue lock before executing or destroying the task.
  folly::Func popNext() {
    auto functions = functions_.wlock();
    VELOX_CHECK(!functions->empty());
    auto function = std::move(functions->front());
    functions->pop_front();
    return function;
  }

  folly::Synchronized<std::deque<folly::Func>> functions_;
};

// Rejects scheduling so tests can verify constructor cleanup.
class ThrowingExecutor : public folly::Executor {
 public:
  void add(folly::Func /* unused */) override {
    VELOX_FAIL("executor rejected read-ahead");
  }
};

class JoiningThread {
 public:
  explicit JoiningThread(std::function<void()> func)
      : thread_(std::move(func)) {}

  ~JoiningThread() {
    join();
  }

  void join() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  std::thread thread_;
};
} // namespace

class FileInputStreamTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    filesystems::registerLocalFileSystem();
  }

  void SetUp() override {
    constexpr uint64_t kMaxMappedMemory = 64 << 20;
    MemoryManager::Options options;
    options.useMmapAllocator = true;
    options.allocatorCapacity = kMaxMappedMemory;
    options.arbitratorCapacity = kMaxMappedMemory;
    options.trackDefaultUsage = true;
    memoryManager_ = std::make_unique<MemoryManager>(options);
    mmapAllocator_ = static_cast<MmapAllocator*>(memoryManager_->allocator());
    pool_ = memoryManager_->addLeafPool("ByteStreamTest");
    rng_.seed(124);
    tempDirPath_ = TempDirectoryPath::create();
    fs_ = filesystems::getFileSystem(tempDirPath_->getPath(), nullptr);
  }

  void TearDown() override {}

  std::unique_ptr<common::FileInputStream> createStream(
      uint64_t streamSize,
      uint32_t bufferSize = 1024) {
    const auto filePath =
        fmt::format("{}/{}", tempDirPath_->getPath(), fileId_++);
    auto writeFile = fs_->openFileForWrite(filePath);
    std::vector<std::uint8_t> buffer(streamSize);
    for (int i = 0; i < streamSize; ++i) {
      buffer[i] = i % 256;
    }
    writeFile->append(
        std::string_view(reinterpret_cast<char*>(buffer.data()), streamSize));
    writeFile->close();
    return std::make_unique<common::FileInputStream>(
        fs_->openFileForRead(filePath), bufferSize, pool_.get());
  }

  folly::Random::DefaultGenerator rng_;
  std::unique_ptr<MemoryManager> memoryManager_;
  MmapAllocator* mmapAllocator_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::atomic_uint64_t fileId_{0};
  std::shared_ptr<TempDirectoryPath> tempDirPath_;
  std::shared_ptr<filesystems::FileSystem> fs_;
};

TEST_F(FileInputStreamTest, stats) {
  struct {
    size_t streamSize;
    size_t bufferSize;

    std::string debugString() const {
      return fmt::format(
          "streamSize {}, bufferSize {}", streamSize, bufferSize);
    }
  } testSettings[] = {
      {4096, 1024}, {4096, 4096}, {4096, 8192}, {4096, 4096 + 1024}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    auto byteStream = createStream(testData.streamSize, testData.bufferSize);
    ASSERT_EQ(byteStream->stats().numReads, 1);
    ASSERT_EQ(
        byteStream->stats().readBytes,
        std::min(testData.streamSize, testData.bufferSize));
    ASSERT_GT(byteStream->stats().readTimeNs, 0);
    std::vector<uint8_t> buffer(testData.streamSize / 8);
    for (int offset = 0; offset < testData.streamSize;) {
      byteStream->readBytes(buffer.data(), testData.streamSize / 8);
      for (int i = 0; i < testData.streamSize / 8; ++i, ++offset) {
        ASSERT_EQ(buffer[i], offset % 256);
      }
    }
    ASSERT_TRUE(byteStream->atEnd());
    ASSERT_EQ(
        byteStream->stats().numReads,
        bits::roundUp(testData.streamSize, testData.bufferSize) /
            testData.bufferSize);
    ASSERT_EQ(byteStream->stats().readBytes, testData.streamSize);
    ASSERT_GT(byteStream->stats().readTimeNs, 0);
  }
}

TEST_F(FileInputStreamTest, readTimeCountsEachReadOnce) {
  tests::utils::registerFaultyFileSystem();
  auto faultyFs = tests::utils::faultyFileSystem();
  constexpr auto kReadDelay = std::chrono::milliseconds(100);
  faultyFs->setFileInjectionHook([&](tests::utils::FaultFileOperation* op) {
    if (op->type == tests::utils::FaultFileOperation::Type::kRead) {
      std::this_thread::sleep_for(kReadDelay);
    }
  });
  SCOPE_EXIT {
    faultyFs->setFileInjectionHook(nullptr);
  };

  const auto filePath =
      fmt::format("{}/{}", tempDirPath_->getPath(), fileId_++);
  auto writeFile = fs_->openFileForWrite(filePath);
  writeFile->append(std::string(1024, 'x'));
  writeFile->close();

  auto readFile = faultyFs->openFileForRead(fmt::format("faulty:{}", filePath));
  // The constructor reads the first range synchronously.
  const auto start = std::chrono::steady_clock::now();
  common::FileInputStream stream(std::move(readFile), 1024, pool_.get());
  const uint64_t constructorNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
  ASSERT_EQ(stream.stats().numReads, 1);
  ASSERT_GE(
      stream.stats().readTimeNs,
      std::chrono::duration_cast<std::chrono::nanoseconds>(kReadDelay).count());
  // Counting the read twice would exceed the time the whole constructor took.
  ASSERT_LE(stream.stats().readTimeNs, constructorNs);
}

TEST_F(FileInputStreamTest, synchronousFileKeepsSingleBufferWithExecutor) {
  constexpr uint64_t kBufferSize = 1024;
  const std::string input(2 * kBufferSize + 17, 'x');
  uint64_t singleBufferBytes{0};
  {
    common::FileInputStream stream(
        std::make_unique<InMemoryReadFile>(input), kBufferSize, pool_.get());
    singleBufferBytes = pool_->usedBytes();
    ASSERT_GT(singleBufferBytes, 0);
  }
  EXPECT_EQ(pool_->usedBytes(), 0);

  QueuedExecutor executor;
  {
    common::FileInputStream stream(
        std::make_unique<InMemoryReadFile>(input),
        kBufferSize,
        pool_.get(),
        &executor);
    EXPECT_EQ(pool_->usedBytes(), singleBufferBytes);
    EXPECT_EQ(executor.size(), 0);
    std::string output(input.size(), '\0');
    stream.readBytes(
        reinterpret_cast<uint8_t*>(output.data()),
        static_cast<int32_t>(output.size()));
    EXPECT_EQ(output, input);
    EXPECT_EQ(executor.size(), 0);
    EXPECT_EQ(stream.stats().numReads, 3);
    EXPECT_EQ(stream.stats().readBytes, input.size());
  }
  EXPECT_EQ(pool_->usedBytes(), 0);
}

TEST_F(FileInputStreamTest, nativeReadAheadDoesNotBlockProducerExecutor) {
  constexpr uint64_t kBufferSize = 1024;
  const std::string input(2 * kBufferSize, 'x');
  QueuedExecutor executor;
  auto state = std::make_shared<DeferredReadState>();
  common::FileInputStream stream(
      std::make_unique<NativeAsyncReadFile>(input, &executor, state),
      kBufferSize,
      pool_.get(),
      &executor);
  ASSERT_EQ(executor.size(), 1);

  auto readQueued = state->nativeReadQueued.get_future();
  std::promise<void> producerFinished;
  auto producerCompletion = producerFinished.get_future();
  JoiningThread producer([&]() {
    executor.runNext();
    producerFinished.set_value();
  });
  EXPECT_EQ(
      readQueued.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  // A one-thread executor must be free to run its queued native I/O. Drain that
  // job even on failure so the regression test cannot leave a blocked thread.
  EXPECT_EQ(
      producerCompletion.wait_for(std::chrono::seconds(5)),
      std::future_status::ready);
  readQueued.wait();
  executor.runNext();
  producer.join();

  std::string output(input.size(), '\0');
  stream.readBytes(
      reinterpret_cast<uint8_t*>(output.data()),
      static_cast<int32_t>(output.size()));
  EXPECT_EQ(output, input);
  EXPECT_EQ(state->readExecutions, 1);
  EXPECT_EQ(stream.stats().numReads, 2);
  EXPECT_EQ(stream.stats().readBytes, input.size());
  EXPECT_EQ(executor.size(), 0);
}

TEST_F(FileInputStreamTest, deferredReadAheadWithoutExecutorRunsOnReader) {
  constexpr uint64_t kBufferSize = 1024;
  std::string input(2 * kBufferSize + 17, '\0');
  for (auto i = 0; i < input.size(); ++i) {
    input[i] = static_cast<char>(i % 251);
  }
  auto state = std::make_shared<DeferredReadState>();
  auto file = std::make_unique<DeferredReadFile>(input, false, false, state);
  auto* filePtr = file.get();
  common::FileInputStream stream(std::move(file), kBufferSize, pool_.get());
  EXPECT_EQ(state->readExecutions, 0);

  std::string output(input.size(), '\0');
  stream.readBytes(
      reinterpret_cast<uint8_t*>(output.data()),
      static_cast<int32_t>(output.size()));
  EXPECT_EQ(output, input);
  EXPECT_EQ(state->readExecutions, 2);
  EXPECT_EQ(filePtr->readThreadId(), std::this_thread::get_id());
  EXPECT_EQ(stream.stats().numReads, 3);
  EXPECT_EQ(stream.stats().readBytes, input.size());
}

TEST_F(
    FileInputStreamTest,
    deferredReadAheadWithoutExecutorSkipsPendingReadOnDestruction) {
  constexpr uint64_t kBufferSize = 1024;
  const std::string input(2 * kBufferSize, 'x');
  auto state = std::make_shared<DeferredReadState>();
  {
    common::FileInputStream stream(
        std::make_unique<DeferredReadFile>(input, false, false, state),
        kBufferSize,
        pool_.get());
  }
  EXPECT_EQ(state->readExecutions, 0);
}

TEST_F(FileInputStreamTest, deferredReadAheadRunsOnExecutorAndPreservesTail) {
  constexpr uint64_t kBufferSize = 1024;
  std::string input(3 * kBufferSize + 17, '\0');
  for (auto i = 0; i < input.size(); ++i) {
    input[i] = static_cast<char>(i % 251);
  }
  folly::CPUThreadPoolExecutor executor(1);
  auto file = std::make_unique<DeferredReadFile>(input, false);
  auto* filePtr = file.get();

  common::FileInputStream stream(
      std::move(file), kBufferSize, pool_.get(), &executor);

  ASSERT_TRUE(filePtr->waitForReadStart());
  EXPECT_NE(filePtr->readThreadId(), std::this_thread::get_id());

  std::string output(input.size(), '\0');
  stream.readBytes(
      reinterpret_cast<uint8_t*>(output.data()),
      static_cast<int32_t>(output.size()));
  EXPECT_EQ(output, input);
  EXPECT_EQ(stream.stats().numReads, 4);
  EXPECT_EQ(stream.stats().readBytes, input.size());
}

TEST_F(FileInputStreamTest, deferredReadAheadPropagatesError) {
  constexpr uint64_t kBufferSize = 1024;
  const std::string input(2 * kBufferSize, 'x');
  folly::CPUThreadPoolExecutor executor(1);
  common::FileInputStream stream(
      std::make_unique<DeferredReadFile>(input, false, true),
      kBufferSize,
      pool_.get(),
      &executor);

  std::string output(input.size(), '\0');
  VELOX_ASSERT_THROW(
      stream.readBytes(
          reinterpret_cast<uint8_t*>(output.data()),
          static_cast<int32_t>(output.size())),
      "deferred read failed");
}

TEST_F(FileInputStreamTest, nativeReadAheadPropagatesErrorAfterPublication) {
  constexpr uint64_t kBufferSize = 1024;
  const std::string input(2 * kBufferSize, 'x');
  QueuedExecutor readAheadExecutor;
  QueuedExecutor nativeExecutor;
  auto state = std::make_shared<DeferredReadState>();
  common::FileInputStream stream(
      std::make_unique<NativeAsyncReadFile>(
          input,
          &nativeExecutor,
          state,
          NativeAsyncReadFile::ReadBehavior::kFail),
      kBufferSize,
      pool_.get(),
      &readAheadExecutor);

  readAheadExecutor.runNext();
  ASSERT_EQ(readAheadExecutor.size(), 0);
  ASSERT_EQ(nativeExecutor.size(), 1);
  ASSERT_EQ(state->readExecutions, 0);
  nativeExecutor.runNext();

  std::string output(input.size(), '\0');
  try {
    stream.readBytes(
        reinterpret_cast<uint8_t*>(output.data()),
        static_cast<int32_t>(output.size()));
    ADD_FAILURE() << "Expected a native read error";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "native read failed");
  }
  EXPECT_EQ(state->readExecutions, 1);
  EXPECT_EQ(nativeExecutor.size(), 0);
}

TEST_F(FileInputStreamTest, deferredReadAheadStealsQueuedRead) {
  constexpr uint64_t kBufferSize = 1024;
  const std::string input(2 * kBufferSize, 'x');
  QueuedExecutor executor;
  auto file = std::make_unique<DeferredReadFile>(input, false);
  auto* filePtr = file.get();
  common::FileInputStream stream(
      std::move(file), kBufferSize, pool_.get(), &executor);
  ASSERT_EQ(executor.size(), 1);

  std::string output(input.size(), '\0');
  stream.readBytes(
      reinterpret_cast<uint8_t*>(output.data()),
      static_cast<int32_t>(output.size()));
  EXPECT_EQ(output, input);
  EXPECT_EQ(filePtr->readThreadId(), std::this_thread::get_id());

  // The queued prepare owns only the completed AsyncSource and is safe to run.
  executor.runNext();
}

TEST_F(FileInputStreamTest, deferredReadAheadCancelsQueuedReadOnDestruction) {
  constexpr uint64_t kBufferSize = 1024;
  const std::string input(2 * kBufferSize, 'x');
  QueuedExecutor executor;
  auto state = std::make_shared<DeferredReadState>();
  auto file = std::make_unique<DeferredReadFile>(input, false, false, state);
  {
    common::FileInputStream stream(
        std::move(file), kBufferSize, pool_.get(), &executor);
    ASSERT_EQ(executor.size(), 1);
  }

  // Destruction cancels an unstarted read. Running the queued prepare later
  // must be a no-op and must not touch the destroyed ReadFile or its buffer.
  executor.runNext();
  EXPECT_EQ(state->readExecutions, 0);
}

TEST_F(FileInputStreamTest, destructorWaitsForRunningReadAhead) {
  constexpr uint64_t kBufferSize = 1024;
  const std::string input(2 * kBufferSize, 'x');
  folly::CPUThreadPoolExecutor executor(1);
  auto state = std::make_shared<DeferredReadState>();
  auto stream = std::make_unique<DestructionSignalingFileInputStream>(
      std::make_unique<DeferredReadFile>(input, true, false, state),
      kBufferSize,
      pool_.get(),
      &executor,
      state);

  const bool readStarted =
      state->readStarted.try_wait_for(std::chrono::seconds(5));
  EXPECT_TRUE(readStarted);
  if (!readStarted) {
    state->continueRead.post();
    return;
  }

  JoiningThread destroyer([&stream]() { stream.reset(); });
  const bool destructionStarted =
      state->streamDestructionStarted.try_wait_for(std::chrono::seconds(5));
  EXPECT_TRUE(destructionStarted);
  // The fake file owns the state used by the blocked callback. It must remain
  // alive until close() has waited for the callback to finish.
  const bool destroyedWhileReadBlocked =
      state->fileDestroyed.try_wait_for(std::chrono::milliseconds(100));
  state->continueRead.post();
  EXPECT_FALSE(destroyedWhileReadBlocked);
  destroyer.join();
}

TEST_F(
    FileInputStreamTest,
    destructorDrainsNativeReadBeforeAndAfterPublication) {
  constexpr uint64_t kBufferSize = 1024;
  const std::string input(2 * kBufferSize, 'x');
  for (const bool blockFutureReturn : {false, true}) {
    SCOPED_TRACE(blockFutureReturn);
    QueuedExecutor readAheadExecutor;
    QueuedExecutor nativeExecutor;
    auto state = std::make_shared<DeferredReadState>();
    auto stream = std::make_unique<DestructionSignalingFileInputStream>(
        std::make_unique<NativeAsyncReadFile>(
            input,
            &nativeExecutor,
            state,
            NativeAsyncReadFile::ReadBehavior::kCompleteWithoutRead,
            blockFutureReturn),
        kBufferSize,
        pool_.get(),
        &readAheadExecutor,
        state);
    const auto pendingBufferBytes = pool_->usedBytes();
    EXPECT_GT(pendingBufferBytes, 0);

    auto readQueued = state->nativeReadQueued.get_future();
    std::promise<void> producerFinished;
    auto producerCompletion = producerFinished.get_future();
    JoiningThread producer([&]() {
      readAheadExecutor.runNext();
      producerFinished.set_value();
    });
    EXPECT_EQ(
        readQueued.wait_for(std::chrono::seconds(5)),
        std::future_status::ready);
    if (!blockFutureReturn) {
      EXPECT_EQ(
          producerCompletion.wait_for(std::chrono::seconds(5)),
          std::future_status::ready);
    }

    JoiningThread destroyer([&]() { stream.reset(); });
    EXPECT_TRUE(
        state->streamDestructionStarted.try_wait_for(std::chrono::seconds(5)));
    const bool fileDestroyedBeforeCompletion =
        state->fileDestroyed.try_wait_for(std::chrono::milliseconds(100));
    const bool buffersReleasedBeforeCompletion =
        pool_->usedBytes() != pendingBufferBytes;
    EXPECT_FALSE(fileDestroyedBeforeCompletion);
    EXPECT_FALSE(buffersReleasedBeforeCompletion);
    EXPECT_EQ(state->readExecutions, 0);

    state->continueRead.post();
    readQueued.wait();
    nativeExecutor.runNext();
    producer.join();
    destroyer.join();
    EXPECT_TRUE(state->nativeFileDestroyed);
    EXPECT_EQ(state->readExecutions, 1);
    EXPECT_EQ(pool_->usedBytes(), 0);
    EXPECT_EQ(nativeExecutor.size(), 0);
  }
}

TEST_F(FileInputStreamTest, executorRejectionUnwindsReadAheadSafely) {
  constexpr uint64_t kBufferSize = 1024;
  const std::string input(2 * kBufferSize, 'x');
  ThrowingExecutor executor;
  auto state = std::make_shared<DeferredReadState>();

  VELOX_ASSERT_THROW(
      common::FileInputStream(
          std::make_unique<DeferredReadFile>(input, false, false, state),
          kBufferSize,
          pool_.get(),
          &executor),
      "executor rejected read-ahead");
  EXPECT_EQ(state->readExecutions, 0);
  EXPECT_TRUE(state->fileDestroyed.try_wait_for(std::chrono::seconds(5)));
}
