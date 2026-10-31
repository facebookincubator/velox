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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfPrestoOutput.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/compression/Compression.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/SerializedPage.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/serializers/PrestoHeader.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/PrestoSerializerSerializationUtils.h"
#include "velox/vector/VectorStream.h"

#include <cudf/types.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/aligned.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <fmt/format.h>
#include <folly/ScopeGuard.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <set>
#include <thread>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using facebook::velox::test::assertEqualVectors;

namespace {

// Converts GPU batches per input vector instead of accumulating 100k rows.
const std::unordered_map<std::string, std::string> kPerBatchGpuInput{
    {cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "1"}};

/// A pinned memory resource that counts the live allocations of the resource
/// it forwards to. Copies share the count, as a resource_ref may copy it.
class CountingPinnedResource {
 public:
  explicit CountingPinnedResource(rmm::host_device_async_resource_ref upstream)
      : upstream_(upstream), live_(std::make_shared<std::atomic<int64_t>>(0)) {}

  void* allocate_sync(
      size_t bytes,
      size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    ++*live_;
    return upstream_.allocate_sync(bytes, alignment);
  }

  void deallocate_sync(
      void* ptr,
      size_t bytes,
      size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    --*live_;
    upstream_.deallocate_sync(ptr, bytes, alignment);
  }

  void* allocate(
      cuda::stream_ref stream,
      size_t bytes,
      size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    ++*live_;
    return upstream_.allocate(stream, bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref stream,
      void* ptr,
      size_t bytes,
      size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    --*live_;
    upstream_.deallocate(stream, ptr, bytes, alignment);
  }

  bool operator==(const CountingPinnedResource& other) const noexcept {
    return live_ == other.live_;
  }

  bool operator!=(const CountingPinnedResource& other) const noexcept {
    return live_ != other.live_;
  }

  int64_t live() const {
    return live_->load();
  }

  friend void get_property(
      const CountingPinnedResource&,
      cuda::mr::device_accessible) noexcept {}

  friend void get_property(
      const CountingPinnedResource&,
      cuda::mr::host_accessible) noexcept {}

 private:
  rmm::host_device_async_resource_ref upstream_;
  std::shared_ptr<std::atomic<int64_t>> live_;
};

class PrestoOutputTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    savedCpuFallback_ = cudf_velox::CudfConfig::getInstance().allowCpuFallback;
    savedTimestampUnit_ = cudf_velox::CudfConfig::getInstance().timestampUnit;
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    bufferManager_->setListenerFactory(nullptr);
    cudf_velox::unregisterCudf();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = savedCpuFallback_;
    cudf_velox::CudfConfig::getInstance().timestampUnit = savedTimestampUnit_;
    OperatorTestBase::TearDown();
  }

  struct Destination {
    // The pages of the destination, deserialized with the Presto serde.
    std::vector<RowVectorPtr> pages;
    // The bytes of every page.
    std::vector<std::string> pageBytes;

    size_t numRows() const {
      size_t rows = 0;
      for (const auto& page : pages) {
        rows += page->size();
      }
      return rows;
    }

    serializer::presto::detail::PrestoHeader header(size_t page) const {
      std::string_view bytes(pageBytes.at(page));
      auto header = serializer::presto::detail::PrestoHeader::read(&bytes);
      VELOX_CHECK(header.has_value());
      return *header;
    }
  };

  struct Run {
    std::shared_ptr<Task> task;
    std::vector<Destination> destinations;

    // Destination 0.
    const std::vector<RowVectorPtr>& pages() const {
      return destinations.at(0).pages;
    }

    size_t numRows() const {
      return destinations.at(0).numRows();
    }
  };

  /// Task ids are unique within the process: Task::terminate() removes the
  /// output buffer of its id after the task finished, which would remove the
  /// buffer of a later task that reused the id.
  std::shared_ptr<Task> makeTask(
      const core::PlanNodePtr& plan,
      std::unordered_map<std::string, std::string> config) {
    static std::atomic<int> taskCounter{0};
    return Task::create(
        fmt::format("local://presto-output-test-{}", ++taskCounter),
        core::PlanFragment{plan},
        0,
        core::QueryCtx::create(
            executor_.get(), core::QueryConfig(std::move(config))),
        Task::ExecutionMode::kParallel);
  }

  /// Runs 'plan' as a task with 'numDrivers' drivers, then drains the pages
  /// of every destination and deserializes them when the node uses the Presto
  /// serde. A broadcast output gets 'numDestinations' destinations. Draining
  /// starts after 'drainDelay_'.
  Run execute(
      const core::PlanNodePtr& plan,
      std::unordered_map<std::string, std::string> config = {},
      int numDestinations = 1,
      int numDrivers = 1) {
    const auto compression = common::stringToCompressionKind(
        config.count(core::QueryConfig::kShuffleCompressionKind)
            ? config.at(core::QueryConfig::kShuffleCompressionKind)
            : "none");
    Run run;
    run.task = makeTask(plan, std::move(config));
    const auto& taskId = run.task->taskId();
    run.task->start(numDrivers);
    const auto outputNode =
        std::dynamic_pointer_cast<const core::PartitionedOutputNode>(plan);
    if (outputNode->isBroadcast()) {
      bufferManager_->updateOutputBuffers(taskId, numDestinations, true);
    }
    const bool isPresto = outputNode->serdeKind() ==
        VectorSerde::kindName(VectorSerde::Kind::kPresto);
    std::this_thread::sleep_for(drainDelay_);

    serializer::presto::PrestoVectorSerde::PrestoOptions options;
    options.compressionKind = compression;
    const auto rowType = asRowType(plan->outputType());
    for (int destination = 0; destination < numDestinations; ++destination) {
      Destination result;
      for (auto& iobuf : drain(*run.task, destination)) {
        iobuf->coalesce();
        result.pageBytes.emplace_back(
            reinterpret_cast<const char*>(iobuf->data()), iobuf->length());
        if (!isPresto) {
          continue;
        }
        result.pages.push_back(deserialize(std::move(iobuf), rowType, options));
      }
      run.destinations.push_back(std::move(result));
    }
    EXPECT_TRUE(waitForTaskCompletion(run.task.get(), 10'000'000))
        << run.task->taskId();
    return run;
  }

  RowVectorPtr deserialize(
      std::unique_ptr<folly::IOBuf> iobuf,
      const RowTypePtr& rowType,
      const serializer::presto::PrestoVectorSerde::PrestoOptions& options) {
    PrestoSerializedPage page(std::move(iobuf));
    auto stream = page.prepareStreamForDeserialize();
    RowVectorPtr vector;
    getNamedVectorSerde("Presto")->deserialize(
        stream.get(), pool(), rowType, &vector, &options);
    return vector;
  }

  std::vector<std::unique_ptr<folly::IOBuf>>
  getData(const Task& task, int destination, int64_t sequence) {
    auto [promise, semiFuture] = folly::makePromiseContract<
        std::vector<std::unique_ptr<folly::IOBuf>>>();
    const bool found = bufferManager_->getData(
        task.taskId(),
        destination,
        kMaxBytesPerFetch,
        sequence,
        [result = std::make_shared<
             folly::Promise<std::vector<std::unique_ptr<folly::IOBuf>>>>(
             std::move(promise))](
            std::vector<std::unique_ptr<folly::IOBuf>> pages,
            int64_t /*inSequence*/,
            std::vector<int64_t> /*remainingBytes*/) {
          result->setValue(std::move(pages));
        });
    VELOX_CHECK(
        found,
        "No output buffer for {} in state {}: {}",
        task.taskId(),
        taskStateString(task.state()),
        task.errorMessage());
    auto future = std::move(semiFuture).via(executor_.get());
    future.wait(std::chrono::seconds{30});
    VELOX_CHECK(future.isReady(), "Timed out waiting for output pages");
    return std::move(future).value();
  }

  std::vector<std::unique_ptr<folly::IOBuf>> drain(
      const Task& task,
      int destination) {
    std::vector<std::unique_ptr<folly::IOBuf>> result;
    for (;;) {
      auto pages = getData(task, destination, result.size());
      for (auto& page : pages) {
        if (page == nullptr) {
          bufferManager_->deleteResults(task.taskId(), destination);
          return result;
        }
        result.push_back(std::move(page));
      }
    }
  }

  RowVectorPtr concat(const std::vector<RowVectorPtr>& vectors) {
    VELOX_CHECK(!vectors.empty());
    vector_size_t total = 0;
    for (const auto& vector : vectors) {
      total += vector->size();
    }
    auto result =
        BaseVector::create<RowVector>(vectors.front()->type(), total, pool());
    vector_size_t offset = 0;
    for (const auto& vector : vectors) {
      result->copy(vector.get(), offset, 0, vector->size());
      offset += vector->size();
    }
    return result;
  }

  std::set<std::string> operatorTypes(
      const Task& task,
      const core::PlanNodeId& nodeId) {
    const auto planStats = toPlanStats(task.taskStats());
    std::set<std::string> types;
    for (const auto& [type, stats] : planStats.at(nodeId).operatorStats) {
      types.insert(type);
    }
    return types;
  }

  /// All input columns projected as they are: a GPU operator that keeps the
  /// rows, so that the CPU baseline also converts the batches through cuDF.
  static PlanBuilder& passThrough(PlanBuilder& builder) {
    return builder.project(builder.planNode()->outputType()->names());
  }

  /// Rows of every flat Presto type, with nulls in every column when
  /// 'withNulls'. Timestamps are whole milliseconds, which the Presto serde
  /// carries exactly.
  RowVectorPtr makeFlatRows(vector_size_t size, int seed, bool withNulls) {
    auto nulls = [&](int n) -> std::function<bool(vector_size_t)> {
      if (!withNulls) {
        return nullptr;
      }
      return nullEvery(n, seed % n);
    };
    return makeRowVector(
        {"b",
         "t",
         "s",
         "i",
         "l",
         "r",
         "d",
         "v",
         "vb",
         "ts",
         "dt",
         "sd",
         "ld",
         "iv"},
        {
            makeFlatVector<bool>(
                size,
                [&](auto row) { return (row + seed) % 3 == 0; },
                nulls(2)),
            makeFlatVector<int8_t>(
                size, [&](auto row) { return row * 7 + seed; }, nulls(3)),
            makeFlatVector<int16_t>(
                size, [&](auto row) { return row * 31 - seed; }, nulls(5)),
            makeFlatVector<int32_t>(
                size,
                [&](auto row) { return row * 1001 - 500 * seed; },
                nulls(7)),
            makeFlatVector<int64_t>(
                size,
                [&](auto row) { return row * 1'000'000'007LL - seed; },
                nulls(11)),
            makeFlatVector<float>(
                size, [&](auto row) { return row * 0.5f - seed; }, nulls(13)),
            makeFlatVector<double>(
                size, [&](auto row) { return row * 1e-3 + seed; }, nulls(17)),
            makeFlatVector<StringView>(
                size,
                [&](auto row) {
                  // Inline and long strings, and empty strings.
                  if ((row + seed) % 4 == 0) {
                    return StringView();
                  }
                  stringBuffer_ = fmt::format(
                      "{}-{}-{}",
                      row,
                      seed,
                      std::string((row + seed) % 29, 'x'));
                  return StringView(stringBuffer_);
                },
                nulls(4)),
            makeFlatVector<StringView>(
                size,
                [&](auto row) {
                  stringBuffer_ = fmt::format("{:02x}{}", row % 256, seed);
                  return StringView(stringBuffer_);
                },
                nulls(6),
                VARBINARY()),
            makeFlatVector<Timestamp>(
                size,
                [&](auto row) {
                  return Timestamp::fromMillis(
                      (row - size / 2) * 86'400'123LL + seed);
                },
                nulls(8)),
            makeFlatVector<int32_t>(
                size,
                [&](auto row) { return row - 10'000 + seed; },
                nulls(9),
                DATE()),
            makeFlatVector<int64_t>(
                size,
                [&](auto row) { return (row - size / 3) * 12'345LL + seed; },
                nulls(10),
                DECIMAL(10, 2)),
            makeFlatVector<int128_t>(
                size,
                [&](auto row) {
                  return (static_cast<int128_t>(row - size / 2) << 70) +
                      row * 99'999LL - seed;
                },
                nulls(12),
                DECIMAL(30, 5)),
            makeFlatVector<int64_t>(
                size,
                [&](auto row) { return row * 3'600'000LL - seed; },
                nulls(14),
                INTERVAL_DAY_TIME()),
        });
  }

  /// Checks that the GPU pages of 'plan' hold the rows the CPU path serializes
  /// for the same plan, and returns the GPU run. When both produce one page of
  /// a multiple of 8 rows, the pages must be the same bytes; with other row
  /// counts the CPU path leaves the pad bits of the null bitmaps unspecified.
  Run assertSameRowsAsCpuPath(
      const core::PlanNodePtr& plan,
      const core::PlanNodeId& outputId,
      std::unordered_map<std::string, std::string> config = {}) {
    auto gpu = execute(plan, config);
    EXPECT_EQ(
        operatorTypes(*gpu.task, outputId),
        std::set<std::string>{"CudfPrestoOutput"});

    config[cudf_velox::CudfConfig::kCudfPrestoOutputEnabled] = "false";
    auto cpu = execute(plan, config);
    EXPECT_EQ(
        operatorTypes(*cpu.task, outputId),
        std::set<std::string>{"PartitionedOutput"});

    EXPECT_EQ(gpu.numRows(), cpu.numRows());
    if (gpu.numRows() > 0) {
      assertEqualVectors(concat(cpu.pages()), concat(gpu.pages()));
    }
    if (gpu.pages().size() == 1 && cpu.pages().size() == 1 &&
        gpu.numRows() % 8 == 0) {
      assertSamePageBytes(
          gpu.destinations[0].pageBytes[0], cpu.destinations[0].pageBytes[0]);
    }
    return gpu;
  }

  static void assertSamePageBytes(
      const std::string& gpu,
      const std::string& cpu) {
    ASSERT_EQ(gpu.size(), cpu.size());
    std::string differences;
    int numDifferences = 0;
    for (size_t i = 0; i < gpu.size(); ++i) {
      if (gpu[i] != cpu[i] && numDifferences++ < 8) {
        differences += fmt::format(
            " [{}] {:#04x} vs {:#04x}",
            i,
            static_cast<uint8_t>(gpu[i]),
            static_cast<uint8_t>(cpu[i]));
      }
    }
    EXPECT_EQ(numDifferences, 0)
        << "GPU page bytes differ from the CPU page bytes at" << differences;
  }

  static constexpr uint64_t kMaxBytesPerFetch = 1 << 20;

  const std::shared_ptr<DefaultOutputBufferManager> bufferManager_{
      DefaultOutputBufferManager::getInstanceRef()};
  std::string stringBuffer_;
  std::chrono::milliseconds drainDelay_{0};
  bool savedCpuFallback_{true};
  cudf::type_id savedTimestampUnit_{cudf::type_id::TIMESTAMP_NANOSECONDS};
};

TEST_F(PrestoOutputTest, flatTypes) {
  for (bool withNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("withNulls = {}", withNulls));
    std::vector<RowVectorPtr> inputs = {
        makeFlatRows(1'000, 1, withNulls),
        makeFlatRows(776, 2, withNulls),
        makeFlatRows(1'496, 3, withNulls)};
    core::PlanNodeId outputId;
    PlanBuilder builder;
    builder.values(inputs);
    auto plan = passThrough(builder)
                    .partitionedOutput({}, 1)
                    .capturePlanNodeId(outputId)
                    .planNode();

    // One GPU batch and one page.
    auto run = assertSameRowsAsCpuPath(plan, outputId);
    ASSERT_EQ(run.pages().size(), 1);
    assertEqualVectors(concat(inputs), run.pages()[0]);

    // Three GPU batches, gathered into one table before serialization.
    run = assertSameRowsAsCpuPath(plan, outputId, kPerBatchGpuInput);
    ASSERT_EQ(run.pages().size(), 1);
    assertEqualVectors(concat(inputs), run.pages()[0]);
  }
}

TEST_F(PrestoOutputTest, multiplePages) {
  std::vector<RowVectorPtr> inputs = {
      makeFlatRows(20'000, 4, true), makeFlatRows(20'000, 5, false)};
  core::PlanNodeId outputId;
  PlanBuilder builder;
  builder.values(inputs);
  auto plan = passThrough(builder)
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  // Small pages make several waves of pages and several host copies per wave.
  constexpr int64_t kPageBytes = 64 << 10;
  auto run = assertSameRowsAsCpuPath(
      plan,
      outputId,
      {{cudf_velox::CudfConfig::kCudfPrestoOutputPageBytes,
        std::to_string(kPageBytes)}});
  const auto& pages = run.destinations[0];
  ASSERT_GT(pages.pages.size(), 40);
  assertEqualVectors(concat(inputs), concat(pages.pages));
  for (size_t p = 0; p < pages.pages.size(); ++p) {
    SCOPED_TRACE(fmt::format("page {}", p));
    // A page exceeds the configured size by less than one row plus the block
    // headers, and every page but the last is nearly full.
    const auto size = pages.pageBytes[p].size();
    EXPECT_LE(size, kPageBytes + 2'048);
    if (p + 1 < pages.pages.size()) {
      EXPECT_GT(size, kPageBytes / 2);
    }
    const auto header = pages.header(p);
    EXPECT_EQ(header.numRows, pages.pages[p]->size());
    EXPECT_EQ(header.pageCodecMarker, 0);
    EXPECT_EQ(
        header.uncompressedSize,
        size - serializer::presto::detail::kHeaderSize);
    EXPECT_EQ(header.compressedSize, header.uncompressedSize);
    EXPECT_EQ(header.checksum, 0);
  }
  const auto planStats = toPlanStats(run.task->taskStats());
  const auto& stats = planStats.at(outputId).operatorStats;
  EXPECT_EQ(stats.at("CudfPrestoOutput")->outputVectors, pages.pages.size());
  EXPECT_EQ(stats.at("CudfPrestoOutput")->outputRows, pages.numRows());
}

TEST_F(PrestoOutputTest, checksummedPages) {
  // With the Presto listener installed, as Prestissimo installs it, every
  // page carries the CRC32 that the deserializer verifies.
  bufferManager_->setListenerFactory([]() {
    return std::make_unique<serializer::presto::PrestoOutputStreamListener>();
  });
  auto input = makeFlatRows(3'000, 22, true);
  core::PlanNodeId outputId;
  PlanBuilder builder;
  builder.values({input});
  auto plan = passThrough(builder)
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();

  // One page, equal to the checksummed CPU page.
  auto run = assertSameRowsAsCpuPath(plan, outputId);
  ASSERT_EQ(run.pages().size(), 1);
  auto header = run.destinations[0].header(0);
  EXPECT_EQ(
      header.pageCodecMarker, serializer::presto::detail::kCheckSumBitMask);
  EXPECT_NE(header.checksum, 0);

  // The checksum covers the page body: a damaged page is rejected.
  auto damaged = run.destinations[0].pageBytes[0];
  damaged[damaged.size() / 2] ^= 0x5a;
  serializer::presto::PrestoVectorSerde::PrestoOptions options;
  VELOX_ASSERT_THROW(
      deserialize(
          folly::IOBuf::copyBuffer(damaged.data(), damaged.size()),
          asRowType(plan->outputType()),
          options),
      "corrupted");

  // Several pages, each with its own checksum.
  run = execute(
      plan,
      {{cudf_velox::CudfConfig::kCudfPrestoOutputPageBytes,
        std::to_string(64 << 10)}});
  ASSERT_GT(run.pages().size(), 1);
  for (size_t p = 0; p < run.pages().size(); ++p) {
    header = run.destinations[0].header(p);
    EXPECT_EQ(
        header.pageCodecMarker, serializer::presto::detail::kCheckSumBitMask)
        << "page " << p;
  }
  assertEqualVectors(input, concat(run.pages()));
}

TEST_F(PrestoOutputTest, projectedOutputLayout) {
  // The output layout reorders and drops input columns.
  auto input = makeFlatRows(1'000, 23, true);
  core::PlanNodeId outputId;
  PlanBuilder builder;
  builder.values({input});
  auto plan = passThrough(builder)
                  .partitionedOutput({}, 1, {"v", "l", "b"})
                  .capturePlanNodeId(outputId)
                  .planNode();
  auto run = assertSameRowsAsCpuPath(plan, outputId);
  auto expected = makeRowVector(
      {"v", "l", "b"},
      {input->childAt(7), input->childAt(4), input->childAt(0)});
  assertEqualVectors(expected, concat(run.pages()));
}

TEST_F(PrestoOutputTest, singleRowPages) {
  // A page holds at least one row: with a one-byte page every row is its own
  // page, the rows of empty strings included, and every page is its own wave.
  auto input = makeFlatRows(257, 24, true);
  core::PlanNodeId outputId;
  PlanBuilder builder;
  builder.values({input});
  auto plan = passThrough(builder)
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  auto run = assertSameRowsAsCpuPath(
      plan,
      outputId,
      {{cudf_velox::CudfConfig::kCudfPrestoOutputPageBytes, "1"}});
  ASSERT_EQ(run.pages().size(), 257);
  for (const auto& page : run.pages()) {
    EXPECT_EQ(page->size(), 1);
  }
  assertEqualVectors(input, concat(run.pages()));
}

TEST_F(PrestoOutputTest, timestampUnits) {
  // Values with sub-millisecond parts and negative values: the serde carries
  // milliseconds, rounded down as Timestamp::toMillis() rounds.
  auto input = makeRowVector({makeFlatVector<Timestamp>(
      2'000,
      [](auto row) {
        return Timestamp(
            (row - 1'000) * 86'400LL + row % 7, (row * 7'919) % 1'000'000'000);
      },
      nullEvery(5))});
  core::PlanNodeId outputId;
  PlanBuilder builder;
  builder.values({input});
  auto plan = passThrough(builder)
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  for (auto unit :
       {cudf::type_id::TIMESTAMP_SECONDS,
        cudf::type_id::TIMESTAMP_MILLISECONDS,
        cudf::type_id::TIMESTAMP_MICROSECONDS,
        cudf::type_id::TIMESTAMP_NANOSECONDS}) {
    SCOPED_TRACE(fmt::format("unit = {}", static_cast<int>(unit)));
    cudf_velox::CudfConfig::getInstance().timestampUnit = unit;
    assertSameRowsAsCpuPath(plan, outputId);
  }
}

TEST_F(PrestoOutputTest, timestampOverflow) {
  // The earliest Timestamp has no millisecond form. With seconds on the GPU,
  // the task fails as it fails on the CPU path, instead of wrapping the value.
  cudf_velox::CudfConfig::getInstance().timestampUnit =
      cudf::type_id::TIMESTAMP_SECONDS;
  auto input = makeRowVector({makeFlatVector<Timestamp>(
      {Timestamp(Timestamp::kMinSeconds, 0), Timestamp(0, 0)})});
  PlanBuilder builder;
  builder.values({input});
  auto plan = passThrough(builder).partitionedOutput({}, 1).planNode();
  for (bool gpu : {true, false}) {
    SCOPED_TRACE(fmt::format("gpu = {}", gpu));
    auto task = makeTask(
        plan,
        {{cudf_velox::CudfConfig::kCudfPrestoOutputEnabled,
          gpu ? "true" : "false"}});
    task->start(1);
    ASSERT_TRUE(waitForTaskFailure(task.get(), 10'000'000));
    EXPECT_NE(task->errorMessage().find("milliseconds"), std::string::npos)
        << task->errorMessage();
  }
}

TEST_F(PrestoOutputTest, slicedInput) {
  // A limit with an offset keeps a row range in the middle of the input.
  std::vector<RowVectorPtr> inputs = {
      makeFlatRows(1'003, 6, true), makeFlatRows(997, 7, true)};
  core::PlanNodeId outputId;
  auto plan = PlanBuilder()
                  .values(inputs)
                  .limit(13, 1'496, false)
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  auto run = assertSameRowsAsCpuPath(plan, outputId, kPerBatchGpuInput);
  EXPECT_EQ(run.numRows(), 1'496);
  assertEqualVectors(concat(inputs)->slice(13, 1'496), concat(run.pages()));
}

TEST_F(PrestoOutputTest, eagerFlushUnderPartialLimit) {
  // A small partial limit makes the output flush every batch, so each GPU
  // batch becomes its own page although all fit in one.
  auto input = makeFlatRows(100, 8, false);
  core::PlanNodeId outputId;
  auto plan = PlanBuilder()
                  .values({input}, false, 5)
                  .limit(0, 400, true)
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  auto run = assertSameRowsAsCpuPath(plan, outputId, kPerBatchGpuInput);
  EXPECT_EQ(run.numRows(), 400);
  EXPECT_EQ(run.pages().size(), 4);
}

TEST_F(PrestoOutputTest, zeroColumns) {
  // Pages of row counts only. The row cap splits them and bounds how much
  // input is gathered before a page is serialized.
  auto input =
      makeRowVector({makeFlatVector<int64_t>(300'000, folly::identity)});
  core::PlanNodeId outputId;
  auto plan = PlanBuilder()
                  .values({input}, false, 4)
                  .project({})
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  auto run = assertSameRowsAsCpuPath(plan, outputId);
  EXPECT_EQ(run.numRows(), 4 * 300'000);
  ASSERT_EQ(run.pages().size(), 2);
  EXPECT_EQ(run.pages()[0]->size(), cudf_velox::CudfPrestoOutput::kMaxPageRows);
  for (const auto& page : run.pages()) {
    EXPECT_EQ(page->childrenSize(), 0);
  }
}

TEST_F(PrestoOutputTest, emptyOutput) {
  auto input = makeFlatRows(1'000, 9, false);
  core::PlanNodeId outputId;
  auto plan = PlanBuilder()
                  .values({input})
                  .filter("l < -1000000")
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  auto run = execute(plan);
  EXPECT_TRUE(run.pages().empty());
  EXPECT_EQ(
      operatorTypes(*run.task, outputId),
      std::set<std::string>{"CudfPrestoOutput"});
}

TEST_F(PrestoOutputTest, broadcastAndArbitrary) {
  std::vector<RowVectorPtr> inputs = {
      makeFlatRows(504, 10, true), makeFlatRows(600, 11, false)};
  {
    core::PlanNodeId outputId;
    PlanBuilder builder;
    builder.values(inputs);
    auto plan = passThrough(builder)
                    .partitionedOutputBroadcast()
                    .capturePlanNodeId(outputId)
                    .planNode();
    // Both destinations receive every page.
    auto run = execute(plan, {}, 2);
    EXPECT_EQ(
        operatorTypes(*run.task, outputId),
        std::set<std::string>{"CudfPrestoOutput"});
    ASSERT_EQ(run.destinations.size(), 2);
    assertEqualVectors(concat(inputs), concat(run.destinations[0].pages));
    assertEqualVectors(concat(inputs), concat(run.destinations[1].pages));
  }
  {
    core::PlanNodeId outputId;
    PlanBuilder builder;
    builder.values(inputs);
    auto plan = passThrough(builder)
                    .partitionedOutputArbitrary()
                    .capturePlanNodeId(outputId)
                    .planNode();
    auto run = assertSameRowsAsCpuPath(plan, outputId);
    assertEqualVectors(concat(inputs), concat(run.pages()));
  }
}

TEST_F(PrestoOutputTest, multipleDrivers) {
  // Two drivers serialize their own batches into the same destination, which
  // completes once both are done.
  std::vector<RowVectorPtr> inputs = {
      makeFlatRows(500, 25, true), makeFlatRows(300, 26, false)};
  core::PlanNodeId outputId;
  PlanBuilder builder;
  builder.values(inputs, /*parallelizable=*/true, /*repeatTimes=*/2);
  auto plan = passThrough(builder)
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  auto run = execute(plan, {}, 1, /*numDrivers=*/2);
  EXPECT_EQ(
      operatorTypes(*run.task, outputId),
      std::set<std::string>{"CudfPrestoOutput"});
  std::vector<RowVectorPtr> expected;
  for (int i = 0; i < 2 * 2; ++i) {
    expected.insert(expected.end(), inputs.begin(), inputs.end());
  }
  assertEqualResults(expected, run.pages());
}

TEST_F(PrestoOutputTest, consumerBackpressure) {
  // The output buffer holds 256 KiB and the consumer starts late, so the
  // operator blocks on the consumer and resumes as pages are fetched.
  drainDelay_ = std::chrono::milliseconds(500);
  std::vector<RowVectorPtr> inputs = {
      makeFlatRows(20'000, 12, true), makeFlatRows(20'000, 13, false)};
  core::PlanNodeId outputId;
  auto plan = PlanBuilder()
                  .values(inputs)
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  auto run = execute(
      plan,
      {{cudf_velox::CudfConfig::kCudfPrestoOutputPageBytes,
        std::to_string(64 << 10)},
       {core::QueryConfig::kMaxOutputBufferSize, std::to_string(256 << 10)}});
  assertEqualVectors(concat(inputs), concat(run.pages()));
  const auto planStats = toPlanStats(run.task->taskStats());
  const auto& stats = planStats.at(outputId).operatorStats;
  EXPECT_GT(stats.at("CudfPrestoOutput")->blockedWallNanos, 0);
}

TEST_F(PrestoOutputTest, abortWhileBlocked) {
  // The task is aborted while pages wait in the full output buffer and in the
  // operator; all pinned memory comes back.
  CountingPinnedResource pinned(cudf::get_pinned_memory_resource());
  const auto previous = cudf::set_pinned_memory_resource(pinned);
  auto restore =
      folly::makeGuard([&] { cudf::set_pinned_memory_resource(previous); });

  std::vector<RowVectorPtr> inputs = {
      makeFlatRows(20'000, 27, true), makeFlatRows(20'000, 28, false)};
  auto plan = PlanBuilder()
                  .values(inputs, false, 3)
                  .partitionedOutput({}, 1)
                  .planNode();
  auto task = makeTask(
      plan,
      {{cudf_velox::CudfConfig::kCudfPrestoOutputPageBytes,
        std::to_string(64 << 10)},
       {core::QueryConfig::kMaxOutputBufferSize, std::to_string(256 << 10)}});
  const auto taskId = task->taskId();
  task->start(1);
  // Over the buffer limit, the operator blocks on the consumer.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (bufferManager_->getUtilization(taskId).value_or(0) <= 1.0) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_GT(pinned.live(), 0);
  task->requestAbort();
  ASSERT_TRUE(waitForTaskAborted(task.get(), 10'000'000));
  task.reset();
  waitForAllTasksToBeDeleted();
  EXPECT_EQ(pinned.live(), 0);
}

TEST_F(PrestoOutputTest, runtimeStats) {
  auto input = makeFlatRows(1'000, 14, true);
  core::PlanNodeId outputId;
  auto plan = PlanBuilder()
                  .values({input})
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  auto run = execute(plan);
  const auto planStats = toPlanStats(run.task->taskStats());
  const auto& stats =
      planStats.at(outputId).operatorStats.at("CudfPrestoOutput")->customStats;
  EXPECT_EQ(
      stats.at(std::string(Operator::kShuffleSerdeKind)).sum,
      static_cast<int64_t>(VectorSerde::Kind::kPresto));
  EXPECT_EQ(
      stats.at(std::string(Operator::kShuffleCompressionKind)).sum,
      static_cast<int64_t>(common::CompressionKind::CompressionKind_NONE));
  EXPECT_GT(stats.at("pageLayoutNanos").sum, 0);
  EXPECT_GT(stats.at("deviceToHostNanos").sum, 0);
}

// Outputs the GPU serializer does not serve keep the CPU PartitionedOutput
// behind a CudfToVelox, also with CPU fallback disabled: they are no failed
// GPU replacement.
TEST_F(PrestoOutputTest, cpuOutputs) {
  auto input = makeFlatRows(1'000, 15, true);
  struct TestCase {
    std::string name;
    std::function<PlanBuilder&(PlanBuilder&)> output;
    std::unordered_map<std::string, std::string> config;
    int numDestinations{1};
  };
  std::vector<TestCase> testCases = {
      {"partitioned",
       [](PlanBuilder& builder) -> PlanBuilder& {
         return builder.partitionedOutput({"i"}, 2);
       },
       {},
       2},
      {"compactRow",
       [](PlanBuilder& builder) -> PlanBuilder& {
         return builder.partitionedOutput({}, 1, {}, "CompactRow");
       },
       {}},
      {"compressed",
       [](PlanBuilder& builder) -> PlanBuilder& {
         return builder.partitionedOutput({}, 1);
       },
       {{core::QueryConfig::kShuffleCompressionKind, "lz4"}}},
      {"disabled",
       [](PlanBuilder& builder) -> PlanBuilder& {
         return builder.partitionedOutput({}, 1);
       },
       {{cudf_velox::CudfConfig::kCudfPrestoOutputEnabled, "false"}}},
  };
  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    core::PlanNodeId orderById;
    core::PlanNodeId outputId;
    PlanBuilder builder;
    builder.values({input}).orderBy({"l"}, false).capturePlanNodeId(orderById);
    auto plan = testCase.output(builder).capturePlanNodeId(outputId).planNode();
    auto run = execute(plan, testCase.config, testCase.numDestinations);
    EXPECT_EQ(
        operatorTypes(*run.task, outputId),
        std::set<std::string>{"PartitionedOutput"});
    // The GPU batches are converted before the CPU output.
    EXPECT_TRUE(operatorTypes(*run.task, orderById).count("CudfToVelox"));
    if (testCase.name == "compactRow") {
      // CompactRow pages are not deserialized.
      EXPECT_FALSE(run.destinations[0].pageBytes.empty());
      continue;
    }
    size_t numRows = 0;
    for (const auto& destination : run.destinations) {
      numRows += destination.numRows();
    }
    EXPECT_EQ(numRows, input->size());
  }

  // A column without a flat Presto encoding.
  auto nested = makeRowVector(
      {makeFlatVector<int64_t>(100, folly::identity),
       makeArrayVector<int64_t>(
           100,
           [](auto row) { return row % 3; },
           [](auto row) { return row; })});
  core::PlanNodeId orderById;
  core::PlanNodeId outputId;
  auto plan = PlanBuilder()
                  .values({nested})
                  .orderBy({"c0"}, false)
                  .capturePlanNodeId(orderById)
                  .partitionedOutput({}, 1)
                  .capturePlanNodeId(outputId)
                  .planNode();
  auto run = execute(plan);
  EXPECT_EQ(
      operatorTypes(*run.task, outputId),
      std::set<std::string>{"PartitionedOutput"});
  EXPECT_TRUE(operatorTypes(*run.task, orderById).count("CudfToVelox"));
  EXPECT_EQ(run.numRows(), 100);
}

} // namespace
