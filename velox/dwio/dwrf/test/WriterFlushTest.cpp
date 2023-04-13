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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/dwio/type/fbhive/HiveTypeParser.h"

using namespace ::testing;
using facebook::velox::dwrf::MemoryUsageCategory;
using facebook::velox::dwrf::WriterContext;
using facebook::velox::dwrf::WriterOptions;

namespace {
constexpr size_t kSizeKB = 1024;
constexpr size_t kSizeMB = 1024 * 1024;
} // namespace

namespace facebook::velox::dwrf {
class MockMemoryPool : public velox::memory::MemoryPool {
 public:
  explicit MockMemoryPool(
      const std::string& name,
      MemoryPool::Kind kind,
      std::shared_ptr<MemoryPool> parent,
      int64_t cap = std::numeric_limits<int64_t>::max())
      : MemoryPool{name, kind, parent, {.alignment = velox::memory::MemoryAllocator::kMinAlignment}},
        memoryUsageTracker_{velox::memory::MemoryUsageTracker::create(cap)} {}

  // Methods not usually exposed by MemoryPool interface to
  // allow for manipulation.
  void updateLocalMemoryUsage(int64_t size) {
    localMemoryUsage_ += size;
  }

  void setLocalMemoryUsage(int64_t size) {
    localMemoryUsage_ = size;
  }

  void zeroOutMemoryUsage() {
    localMemoryUsage_ = 0;
  }

  static std::shared_ptr<MockMemoryPool> create() {
    return std::make_shared<MockMemoryPool>(
        "standalone_pool", MemoryPool::Kind::kAggregate, nullptr);
  }

  void* allocate(int64_t size) override {
    updateLocalMemoryUsage(size);
    return allocator_->allocateBytes(size);
  }

  void* allocateZeroFilled(int64_t numEntries, int64_t sizeEach) override {
    updateLocalMemoryUsage(numEntries * sizeEach);
    return allocator_->allocateZeroFilled(numEntries * sizeEach);
  }

  // No-op for attempts to shrink buffer.
  void* reallocate(void* p, int64_t size, int64_t newSize) override {
    void* newP = allocate(newSize);
    VELOX_CHECK_NOT_NULL(newP);
    ::memcpy(newP, p, std::min(size, newSize));
    free(p, size);
    return newP;
  }

  void free(void* p, int64_t size) override {
    allocator_->freeBytes(p, size);
    updateLocalMemoryUsage(-size);
  }

  void allocateNonContiguous(
      velox::memory::MachinePageCount /*unused*/,
      velox::memory::Allocation& /*unused*/,
      velox::memory::MachinePageCount /*unused*/) override {
    VELOX_UNSUPPORTED("allocateNonContiguous unsupported");
  }

  void freeNonContiguous(velox::memory::Allocation& /*unused*/) override {
    VELOX_UNSUPPORTED("freeNonContiguous unsupported");
  }

  velox::memory::MachinePageCount largestSizeClass() const override {
    VELOX_UNSUPPORTED("largestSizeClass unsupported");
  }

  const std::vector<velox::memory::MachinePageCount>& sizeClasses()
      const override {
    VELOX_UNSUPPORTED("sizeClasses unsupported");
  }

  void allocateContiguous(
      velox::memory::MachinePageCount /*unused*/,
      velox::memory::ContiguousAllocation& /*unused*/) override {
    VELOX_UNSUPPORTED("allocateContiguous unsupported");
  }

  void freeContiguous(velox::memory::ContiguousAllocation&
                      /*unused*/) override {
    VELOX_UNSUPPORTED("freeContiguous unsupported");
  }

  int64_t getCurrentBytes() const override {
    return localMemoryUsage_;
  }

  std::shared_ptr<MemoryPool> genChild(
      std::shared_ptr<MemoryPool> parent,
      const std::string& name,
      MemoryPool::Kind kind,
      std::shared_ptr<memory::MemoryReclaimer> /*unused*/) override {
    return std::make_shared<MockMemoryPool>(
        name, kind, parent, memoryUsageTracker_->maxMemory());
  }

  const std::shared_ptr<velox::memory::MemoryUsageTracker>&
  getMemoryUsageTracker() const override {
    return memoryUsageTracker_;
  }

  MOCK_CONST_METHOD0(getMaxBytes, int64_t());

  MOCK_METHOD1(updateSubtreeMemoryUsage, int64_t(int64_t));

  MOCK_CONST_METHOD0(getAlignment, uint16_t());

  uint64_t freeBytes() const override {
    VELOX_NYI("{} unsupported", __FUNCTION__);
  }

  uint64_t shrink(uint64_t /*unused*/) override {
    VELOX_NYI("{} unsupported", __FUNCTION__);
  }

  uint64_t grow(uint64_t /*unused*/) override {
    VELOX_NYI("{} unsupported", __FUNCTION__);
  }

  std::string toString() const override {
    return fmt::format(
        "Mock Memory Pool[{}]",
        velox::memory::MemoryAllocator::kindString(allocator_->kind()));
  }

 private:
  velox::memory::MemoryAllocator* const FOLLY_NONNULL allocator_{
      velox::memory::MemoryAllocator::getInstance()};
  int64_t localMemoryUsage_{0};
  std::shared_ptr<velox::memory::MemoryUsageTracker> memoryUsageTracker_;
};

// For testing functionality of Writer we need to instantiate
// it.
class DummyWriter : public velox::dwrf::Writer {
 public:
  DummyWriter(
      WriterOptions& options,
      std::unique_ptr<dwio::common::DataSink> sink,
      std::shared_ptr<memory::MemoryPool> pool)
      : Writer{options, std::move(sink), std::move(pool)} {}

  MOCK_METHOD1(
      flushImpl,
      void(std::function<proto::ColumnEncoding&(uint32_t)>));
  MOCK_METHOD0(createIndexEntryImpl, void());
  MOCK_METHOD1(
      writeFileStatsImpl,
      void(std::function<proto::ColumnStatistics&(uint32_t)>));
  MOCK_METHOD0(abandonDictionariesImpl, void());
  MOCK_METHOD0(resetImpl, void());

  friend class WriterFlushTestHelper;
  VELOX_FRIEND_TEST(TestWriterFlush, CheckAgainstMemoryBudget);
};

// Big idea is to directly manipulate context states (num rows) + memory pool
// and just call writer.write() to trigger the flush?

// The most elegant solution would be to mock column writers, which then
// updates the memory pool stats. The point is to control the memory footprint
// while ideally just calling writer.write() and make sure it takes all
// these into account.

struct SimulatedWrite {
  SimulatedWrite(
      uint64_t numRows,
      uint64_t outputStreamMemoryUsage,
      uint64_t generalMemoryUsage)
      : numRows{numRows},
        outputStreamMemoryUsage{outputStreamMemoryUsage},
        generalMemoryUsage{generalMemoryUsage} {}

  void apply(WriterContext& context) const {
    context.incRowCount(numRows);
    // Not the most accurate semantically, but suffices for testing
    // purposes.
    dynamic_cast<MockMemoryPool&>(
        context.getMemoryPool(MemoryUsageCategory::OUTPUT_STREAM))
        .updateLocalMemoryUsage(outputStreamMemoryUsage);
    dynamic_cast<MockMemoryPool&>(
        context.getMemoryPool(MemoryUsageCategory::GENERAL))
        .updateLocalMemoryUsage(generalMemoryUsage);
  }

  uint64_t numRows;
  uint64_t outputStreamMemoryUsage;
  uint64_t generalMemoryUsage;
};

struct SimulatedFlush {
  SimulatedFlush(
      uint64_t flushOverhead,
      uint64_t stripeRowCount,
      uint64_t stripeRawSize,
      uint64_t compressedSize,
      uint64_t dictMemoryUsage,
      uint64_t outputStreamMemoryUsage,
      uint64_t generalMemoryUsage)
      : flushOverhead{flushOverhead},
        stripeRowCount{stripeRowCount},
        stripeRawSize{stripeRawSize},
        compressedSize{compressedSize},
        dictMemoryUsage{dictMemoryUsage},
        outputStreamMemoryUsage{outputStreamMemoryUsage},
        generalMemoryUsage{generalMemoryUsage} {}

  void apply(WriterContext& context) const {
    context.stripeRawSize = stripeRawSize;
    ASSERT_EQ(stripeRowCount, context.stripeRowCount);
    auto& dictPool = dynamic_cast<MockMemoryPool&>(
        context.getMemoryPool(MemoryUsageCategory::DICTIONARY));
    auto& outputPool = dynamic_cast<MockMemoryPool&>(
        context.getMemoryPool(MemoryUsageCategory::OUTPUT_STREAM));
    auto& generalPool = dynamic_cast<MockMemoryPool&>(
        context.getMemoryPool(MemoryUsageCategory::GENERAL));
    dictPool.setLocalMemoryUsage(dictMemoryUsage);
    ASSERT_EQ(outputStreamMemoryUsage, outputPool.getCurrentBytes());
    outputPool.updateLocalMemoryUsage(flushOverhead);
    generalPool.setLocalMemoryUsage(generalMemoryUsage);

    context.recordAverageRowSize();
    context.recordFlushOverhead(flushOverhead);
    context.recordCompressionRatio(compressedSize);

    ++context.stripeIndex;
    // Clear context
    context.stripeRawSize = 0;
    context.stripeRowCount = 0;
    // Simplified memory footprint modeling for testing.
    dictPool.zeroOutMemoryUsage();
    outputPool.zeroOutMemoryUsage();
    generalPool.zeroOutMemoryUsage();
  }

  uint64_t flushOverhead;
  uint64_t stripeRowCount;
  uint64_t stripeRawSize;
  uint64_t compressedSize;
  // Memory footprint can change drastically at flush time
  // esp for dictionary encoding.
  uint64_t dictMemoryUsage;
  uint64_t outputStreamMemoryUsage;
  uint64_t generalMemoryUsage;
};

class WriterFlushTestHelper {
 public:
  static std::unique_ptr<DummyWriter> prepWriter(
      const std::shared_ptr<MockMemoryPool>& sinkPool,
      int64_t writerMemoryBudget) {
    WriterOptions options;
    options.config = std::make_shared<Config>();
    options.schema = dwio::type::fbhive::HiveTypeParser().parse(
        "struct<int_val:int,string_val:string>");
    // A completely memory pressure based flush policy.
    options.flushPolicyFactory = []() {
      return std::make_unique<LambdaFlushPolicy>([]() { return false; });
    };
    auto writer = std::make_unique<DummyWriter>(
        options,
        // Unused sink.
        std::make_unique<dwio::common::MemorySink>(*sinkPool, kSizeKB),
        std::make_shared<MockMemoryPool>(
            "writer_root_pool",
            memory::MemoryPool::Kind::kAggregate,
            nullptr,
            writerMemoryBudget));
    auto& context = writer->getContext();
    zeroOutMemoryUsage(context);
    return writer;
  }

  static void testRandomSequence(
      std::unique_ptr<DummyWriter> writer,
      int64_t numStripes,
      uint32_t seed,
      uint32_t averageOutputStreamMemoryUsage,
      uint32_t generalMemoryUsageVariation) {
    constexpr size_t kSequenceLength = 1000;
    std::mt19937 gen{};
    gen.seed(seed);
    auto sequence = generateSimulatedWrites(
        gen,
        averageOutputStreamMemoryUsage,
        generalMemoryUsageVariation,
        kSequenceLength);
    testRandomSequence(std::move(writer), numStripes, sequence, gen);
  }

 private:
  static void zeroOutMemoryUsage(WriterContext& context) {
    dynamic_cast<MockMemoryPool&>(
        context.getMemoryPool(MemoryUsageCategory::DICTIONARY))
        .zeroOutMemoryUsage();
    dynamic_cast<MockMemoryPool&>(
        context.getMemoryPool(MemoryUsageCategory::OUTPUT_STREAM))
        .zeroOutMemoryUsage();
    dynamic_cast<MockMemoryPool&>(
        context.getMemoryPool(MemoryUsageCategory::GENERAL))
        .zeroOutMemoryUsage();
  }

  static void testRandomSequence(
      std::unique_ptr<DummyWriter> writer,
      int64_t numStripes,
      const std::vector<SimulatedWrite>& writeSequence,
      std::mt19937& gen) {
    auto& context = writer->getContext();
    for (const auto& write : writeSequence) {
      if (writer->shouldFlush(context, write.numRows)) {
        ASSERT_EQ(
            0,
            context.getMemoryPool(MemoryUsageCategory::DICTIONARY)
                .getCurrentBytes());
        auto outputStreamMemoryUsage =
            context.getMemoryPool(MemoryUsageCategory::OUTPUT_STREAM)
                .getCurrentBytes();
        auto generalMemoryUsage =
            context.getMemoryPool(MemoryUsageCategory::GENERAL)
                .getCurrentBytes();

        uint64_t flushOverhead =
            folly::Random::rand32(0, context.stripeRawSize, gen);
        uint64_t compressedSize =
            folly::Random::rand32(0, context.stripeRawSize, gen);
        uint64_t dictMemoryUsage =
            folly::Random::rand32(0, flushOverhead / 3, gen);
        SimulatedFlush{
            flushOverhead,
            context.stripeRowCount,
            context.stripeRawSize,
            compressedSize,
            folly::to<uint64_t>(dictMemoryUsage),
            // Flush overhead is the delta of output stream memory
            // usage before and after flush. Peak memory footprint
            // happens when we finished writing dictionary encoded to
            // streams and before we can clear the dictionary encoders.
            folly::to<uint64_t>(outputStreamMemoryUsage) + flushOverhead -
                dictMemoryUsage,
            // For simplicy, general pool usage is held constant.
            folly::to<uint64_t>(generalMemoryUsage)}
            .apply(context);
      }
      write.apply(context);
    }
    EXPECT_EQ(numStripes, context.stripeIndex);
  }

  static std::vector<SimulatedWrite> generateSimulatedWrites(
      std::mt19937& gen,
      uint32_t averageOutputStreamMemoryUsage,
      uint32_t generalMemoryUsageVariation,
      size_t sequenceLength) {
    std::vector<SimulatedWrite> sequence;
    for (size_t i = 0; i < sequenceLength; ++i) {
      sequence.emplace_back(
          1000,
          folly::Random::rand32(
              averageOutputStreamMemoryUsage / 2,
              averageOutputStreamMemoryUsage,
              gen),
          // For simplicity, general pool memory footprint is monotonically
          // increasing from 0. It's equivalent to removing the base
          // footprint from the budget anyway.
          folly::Random::rand32(0, generalMemoryUsageVariation, gen));
    }
    return sequence;
  }
};

// This test checks against constructed test cases.
TEST(TestWriterFlush, CheckAgainstMemoryBudget) {
  auto pool = MockMemoryPool::create();
  {
    auto writer = WriterFlushTestHelper::prepWriter(pool, 1024);
    auto& context = writer->getContext();

    SimulatedWrite simWrite{10, 500, 300};
    simWrite.apply(context);
    // Writer has no data point in the first stripe and uses a static
    // (though configurable) flush overhead ratio to determine whether
    // we need to flush.
    EXPECT_FALSE(writer->shouldFlush(context, 10));
    EXPECT_FALSE(writer->shouldFlush(context, 20));
    EXPECT_FALSE(writer->shouldFlush(context, 200));
  }
  {
    auto writer = WriterFlushTestHelper::prepWriter(pool, 1024);
    auto& context = writer->getContext();

    SimulatedWrite simWrite{10, 500, 300};
    simWrite.apply(context);
    // The flush produces 0 overhead for miraculous reasons.
    SimulatedFlush simFlush{
        0 /* flushOverhead */,
        10 /* stripeRowCount */,
        1000 /* stripeRawSize */,
        450 /* compressedSize */,
        0 /* dictMemoryUsage */,
        500 /* outputStreamMemoryUsage */,
        300 /* generalMemoryUsage */};

    simFlush.apply(context);
    // Aborting write based on whether the write would exceed budget.
    // Ideally logic should be added to further break up batches like bbio.
    EXPECT_FALSE(writer->shouldFlush(context, 10));
    EXPECT_FALSE(writer->shouldFlush(context, 20));
    EXPECT_TRUE(writer->shouldFlush(context, 25));
    EXPECT_TRUE(writer->shouldFlush(context, 200));
  }
  {
    auto writer = WriterFlushTestHelper::prepWriter(pool, 1024);
    auto& context = writer->getContext();

    SimulatedWrite{10, 500, 300}.apply(context);
    SimulatedFlush simFlush{
        0 /* flushOverhead */,
        10 /* stripeRowCount */,
        1000 /* stripeRawSize */,
        450 /* compressedSize */,
        0 /* dictMemoryUsage */,
        500 /* outputStreamMemoryUsage */,
        300 /* generalMemoryUsage */};
    simFlush.apply(context);
    // Aborting write based on whether the write would exceed budget.
    SimulatedWrite{10, 500, 300}.apply(context);

    EXPECT_FALSE(writer->shouldFlush(context, 4));
    EXPECT_TRUE(writer->shouldFlush(context, 5));
    EXPECT_TRUE(writer->shouldFlush(context, 15));
    EXPECT_TRUE(writer->shouldFlush(context, 200));
  }
  {
    auto writer = WriterFlushTestHelper::prepWriter(pool, 1024);
    auto& context = writer->getContext();

    // 0 overhead flush but with raw size per row variance.
    SimulatedWrite{10, 500, 300}.apply(context);
    SimulatedFlush{
        0 /* flushOverhead */,
        10 /* stripeRowCount */,
        1000 /* stripeRawSize */,
        500 /* compressedSize */,
        0 /* dictMemoryUsage */,
        500 /* outputStreamMemoryUsage */,
        300 /* generalMemoryUsage */}
        .apply(context);
    SimulatedWrite{10, 500, 300}.apply(context);
    SimulatedFlush{
        0 /* flushOverhead */,
        10 /* stripeRowCount */,
        600 /* stripeRawSize */,
        300 /* compressedSize */,
        0 /* dictMemoryUsage */,
        500 /* outputStreamMemoryUsage */,
        300 /* generalMemoryUsage */}
        .apply(context);

    EXPECT_FALSE(writer->shouldFlush(context, 10));
    EXPECT_FALSE(writer->shouldFlush(context, 25));
    EXPECT_TRUE(writer->shouldFlush(context, 26));
    EXPECT_TRUE(writer->shouldFlush(context, 200));
  }
  {
    auto writer = WriterFlushTestHelper::prepWriter(pool, 1024);
    auto& context = writer->getContext();

    // 0 overhead flush but with raw size per row variance.
    SimulatedWrite{10, 500, 300}.apply(context);
    SimulatedFlush{
        200 /* flushOverhead */,
        10 /* stripeRowCount */,
        1000 /* stripeRawSize */,
        500 /* compressedSize */,
        0 /* dictMemoryUsage */,
        500 /* outputStreamMemoryUsage */,
        300 /* generalMemoryUsage */}
        .apply(context);

    SimulatedWrite{5, 250, 150}.apply(context);

    EXPECT_FALSE(writer->shouldFlush(context, 5));
    EXPECT_FALSE(writer->shouldFlush(context, 6));
    EXPECT_TRUE(writer->shouldFlush(context, 10));
    EXPECT_TRUE(writer->shouldFlush(context, 200));
  }
  {
    auto writer = WriterFlushTestHelper::prepWriter(pool, 1024);
    auto& context = writer->getContext();

    // 0 overhead flush but with flush overhead variance.
    SimulatedWrite{10, 500, 300}.apply(context);
    SimulatedFlush{
        200 /* flushOverhead */,
        10 /* stripeRowCount */,
        1000 /* stripeRawSize */,
        500 /* compressedSize */,
        0 /* dictMemoryUsage */,
        500 /* outputStreamMemoryUsage */,
        300 /* generalMemoryUsage */}
        .apply(context);
    SimulatedWrite{10, 500, 300}.apply(context);
    SimulatedFlush{
        100 /* flushOverhead */,
        10 /* stripeRowCount */,
        1000 /* stripeRawSize */,
        500 /* compressedSize */,
        0 /* dictMemoryUsage */,
        500 /* outputStreamMemoryUsage */,
        300 /* generalMemoryUsage */}
        .apply(context);

    SimulatedWrite{5, 250, 150}.apply(context);
    EXPECT_FALSE(writer->shouldFlush(context, 5));
    EXPECT_FALSE(writer->shouldFlush(context, 7));
    EXPECT_TRUE(writer->shouldFlush(context, 10));
    EXPECT_TRUE(writer->shouldFlush(context, 200));
  }
}

// Tests the number of stripes produced based on random results.
TEST(TestWriterFlush, MemoryBasedFlushRandom) {
  struct TestCase {
    TestCase(
        uint32_t seed,
        int64_t averageOutputStreamMemoryUsage,
        size_t numStripes)
        : seed{seed},
          averageOutputStreamMemoryUsage{averageOutputStreamMemoryUsage},
          numStripes{numStripes} {}

    uint32_t seed;
    int64_t averageOutputStreamMemoryUsage;
    size_t numStripes;
  };

  auto pool = MockMemoryPool::create();
  std::vector<TestCase> testCases{
      {10237629, 20 * kSizeMB, 29},
      // TODO: investigate why this fails on linux specifically.
      // {30227679, 20 * kSizeMB, 30},
      {10237629, 10 * kSizeMB, 15},
      {30227679, 10 * kSizeMB, 15},
      {10237629, 49 * kSizeMB, 69},
      {30227679, 70 * kSizeMB, 98}};

  for (auto& testCase : testCases) {
    WriterFlushTestHelper::testRandomSequence(
        WriterFlushTestHelper::prepWriter(pool, 512 * kSizeMB),
        testCase.numStripes,
        testCase.seed,
        testCase.averageOutputStreamMemoryUsage,
        kSizeMB);
  }
}
} // namespace facebook::velox::dwrf
