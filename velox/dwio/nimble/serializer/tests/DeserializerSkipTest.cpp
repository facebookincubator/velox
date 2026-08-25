/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <folly/Random.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <string_view>
#include <thread>

#include "folly/container/F14Set.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/serializer/StreamDataWriter.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/dwio/nimble/writer/Writer.h"

#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook;
using namespace facebook::nimble;
using facebook::nimble::test::makeTestTabletOptions;

namespace facebook::nimble {

// Exercises BatchedStreamDecoder::skip via the per-batch-rowRanges
// overload of Deserializer::deserialize. Each case serializes one or more
// input vectors, then verifies that
//   deserialize(data, rowRanges, out)
// equals concatenation of per-batch slices
//   deserialize(data[i]).slice(rowRanges[i].startRow, rowRanges[i].numRows()).
class DeserializerSkipTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("skip_test");
    pool_ = rootPool_->addLeafChild("leaf");
    vm_ = std::make_unique<velox::test::VectorMaker>(pool_.get());
  }

  // Serialize the given input vectors as kSerialization batches. Returns the
  // serialized byte strings plus the nimble schema needed to construct a
  // Deserializer.
  std::pair<std::vector<std::string>, std::shared_ptr<const Type>> serialize(
      const velox::TypePtr& type,
      const std::vector<velox::VectorPtr>& inputs,
      const folly::F14FastMap<std::string, std::set<std::string>>&
          flatMapColumns = {}) {
    SerializerOptions options{
        .version = SerializationVersion::kSerialization,
        .flatMapColumns = flatMapColumns,
    };
    Serializer serializer{options, type, pool_.get()};
    std::vector<std::string> serialized;
    serialized.reserve(inputs.size());
    for (const auto& input : inputs) {
      serialized.emplace_back(
          serializer.serialize(input, OrderedRanges::of(0, input->size())));
    }
    auto schema =
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
    return {std::move(serialized), std::move(schema)};
  }

  std::string makeTabletBatch(std::string_view serialized, RowRange rowRange) {
    const DeserializerOptions parserOptions{.hasHeader = true};
    serde::StreamDataParser parser{pool_.get(), parserOptions};
    const auto rowCount = parser.initialize(serialized);
    auto header = serde::createTabletChunkHeader({
        .rowCount = rowCount,
        .requiresNullBarrier = parser.requiresNullBarrier(),
        .streamEncodingUsesVarintRowCount =
            parser.streamEncodingUsesVarintRowCount(),
        .streamHasChunkHeader = false,
        .rowRange = rowRange,
    });
    std::string output(
        reinterpret_cast<const char*>(header.data()), header.length());
    std::vector<uint32_t> streamIds;
    std::vector<uint32_t> streamSizeIndices;
    std::vector<uint32_t> streamSizes;
    parser.iterateStreams([&](uint32_t streamId, std::string_view streamData) {
      if (streamData.empty()) {
        return;
      }
      streamIds.emplace_back(streamId);
      streamSizeIndices.emplace_back(static_cast<uint32_t>(streamSizes.size()));
      streamSizes.emplace_back(static_cast<uint32_t>(streamData.size()));
      output.append(streamData);
    });
    serde::detail::writeTrailer(
        streamIds,
        streamSizeIndices,
        streamSizes,
        EncodingType::Trivial,
        EncodingType::Trivial,
        EncodingType::Trivial,
        output);
    return output;
  }

  // ROW(a INTEGER, s ROW(b INTEGER)). Rows listed in `nestedNullRows` get a
  // null `s`, so s's Row null stream is written with real nulls and the
  // batch's null-barrier flag is set. A nullable *scalar* column would not
  // do it: the flag only tracks structural (Row/FlatMap) null streams.
  static velox::TypePtr barrierType() {
    return velox::ROW(
        {{"a", velox::INTEGER()},
         {"s", velox::ROW({{"b", velox::INTEGER()}})}});
  }

  velox::VectorPtr makeBarrierBatch(
      int32_t base,
      velox::vector_size_t rows,
      const std::vector<velox::vector_size_t>& nestedNullRows) {
    auto type = barrierType();
    auto a = velox::BaseVector::create(velox::INTEGER(), rows, pool_.get());
    auto b = velox::BaseVector::create(velox::INTEGER(), rows, pool_.get());
    for (velox::vector_size_t i = 0; i < rows; ++i) {
      a->asFlatVector<int32_t>()->set(i, base + i);
      b->asFlatVector<int32_t>()->set(i, (base + i) * 10);
    }
    auto nested = std::make_shared<velox::RowVector>(
        pool_.get(),
        type->childAt(1),
        nullptr,
        rows,
        std::vector<velox::VectorPtr>{b});
    for (auto row : nestedNullRows) {
      nested->setNull(row, true);
    }
    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        rows,
        std::vector<velox::VectorPtr>{a, nested});
  }

  // ROW(a INTEGER, s ROW(b INTEGER), m MAP(VARCHAR, DOUBLE)) with `m` written
  // as a FlatMap. Nulls on `s` supply the barrier flag, independently of the
  // FlatMap, so `m` can still have an all-present key. That combination is
  // what reaches the barrier-only in-map path: an all-present key has its
  // in-map stream omitted, and on a barrier batch `appendStreamSegments`
  // records it with the no-arg `addPresentInMapBatch()` sentinel.
  static velox::TypePtr flatMapBarrierType() {
    return velox::ROW(
        {{"a", velox::INTEGER()},
         {"s", velox::ROW({{"b", velox::INTEGER()}})},
         {"m", velox::MAP(velox::VARCHAR(), velox::DOUBLE())}});
  }

  velox::VectorPtr makeFlatMapBarrierBatch(
      const std::vector<std::vector<std::string>>& keysByRow,
      const std::vector<velox::vector_size_t>& nestedNullRows) {
    auto type = flatMapBarrierType();
    const auto rows = static_cast<velox::vector_size_t>(keysByRow.size());
    velox::vector_size_t totalEntries = 0;
    for (const auto& keys : keysByRow) {
      totalEntries += static_cast<velox::vector_size_t>(keys.size());
    }

    auto a = velox::BaseVector::create(velox::INTEGER(), rows, pool_.get());
    auto b = velox::BaseVector::create(velox::INTEGER(), rows, pool_.get());
    for (velox::vector_size_t i = 0; i < rows; ++i) {
      a->asFlatVector<int32_t>()->set(i, i);
      b->asFlatVector<int32_t>()->set(i, i * 10);
    }
    auto nested = std::make_shared<velox::RowVector>(
        pool_.get(),
        type->childAt(1),
        nullptr,
        rows,
        std::vector<velox::VectorPtr>{b});
    for (auto row : nestedNullRows) {
      nested->setNull(row, true);
    }

    auto mapKeys =
        velox::BaseVector::create(velox::VARCHAR(), totalEntries, pool_.get());
    auto mapValues =
        velox::BaseVector::create(velox::DOUBLE(), totalEntries, pool_.get());
    velox::vector_size_t idx = 0;
    for (velox::vector_size_t row = 0; row < rows; ++row) {
      for (const auto& key : keysByRow[row]) {
        mapKeys->asFlatVector<velox::StringView>()->set(
            idx, velox::StringView(key));
        mapValues->asFlatVector<double>()->set(idx, row * 10.0 + idx);
        ++idx;
      }
    }
    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        type->childAt(2),
        nullptr,
        rows,
        velox::allocateOffsets(rows, pool_.get()),
        velox::allocateSizes(rows, pool_.get()),
        mapKeys,
        mapValues);
    auto* rawOffsets =
        mapVector->mutableOffsets(rows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(rows)->asMutable<velox::vector_size_t>();
    velox::vector_size_t offset = 0;
    for (velox::vector_size_t i = 0; i < rows; ++i) {
      rawOffsets[i] = offset;
      rawSizes[i] = static_cast<velox::vector_size_t>(keysByRow[i].size());
      offset += rawSizes[i];
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        rows,
        std::vector<velox::VectorPtr>{a, nested, mapVector});
  }

  // Round-trip check: `deserialize(data, rowRanges)` for rowRanges derived
  // from a run-level [skipRows, skipRows+decodeRows) window must equal
  // `deserialize(data)` sliced by the same window.
  //
  // Per-batch rowRanges are computed by intersecting the run-level window
  // with each batch's [batchStart, batchStart+batchRowCount).
  void checkSkip(
      const std::shared_ptr<const Type>& schema,
      const std::vector<std::string>& serialized,
      uint32_t skipRows,
      uint32_t decodeRows) {
    SCOPED_TRACE(
        fmt::format(
            "skipRows={} decodeRows={} batches={}",
            skipRows,
            decodeRows,
            serialized.size()));

    std::vector<std::string_view> views;
    views.reserve(serialized.size());
    for (const auto& s : serialized) {
      views.push_back(s);
    }

    DeserializerOptions dsOpts{.hasHeader = true};
    Deserializer fullDs{schema, pool_.get(), dsOpts};
    velox::VectorPtr full;
    fullDs.deserialize(views, full);
    ASSERT_GE(full->size(), skipRows + decodeRows);

    // Compute per-batch RowRanges from (skipRows, decodeRows) by discovering
    // each batch's rowCount via a single-batch deserialize.
    std::vector<RowRange> rowRanges;
    rowRanges.reserve(serialized.size());
    const uint64_t windowEnd = static_cast<uint64_t>(skipRows) + decodeRows;
    uint64_t batchStart = 0;
    for (const auto& s : serialized) {
      Deserializer singleDs{schema, pool_.get(), dsOpts};
      velox::VectorPtr singleOut;
      std::vector<std::string_view> singleView{s};
      singleDs.deserialize(singleView, singleOut);
      const uint64_t rowCount = singleOut->size();
      const uint64_t batchEnd = batchStart + rowCount;

      const uint64_t keepStart = std::max<uint64_t>(skipRows, batchStart);
      const uint64_t keepEnd = std::min<uint64_t>(windowEnd, batchEnd);
      if (keepStart >= keepEnd) {
        rowRanges.push_back(RowRange{0, 0});
      } else {
        rowRanges.push_back(
            RowRange{
                static_cast<uint32_t>(keepStart - batchStart),
                static_cast<uint32_t>(keepEnd - batchStart)});
      }
      batchStart = batchEnd;
    }

    Deserializer partialDs{schema, pool_.get(), dsOpts};
    velox::VectorPtr partial;
    partialDs.deserialize(views, rowRanges, partial);

    if (decodeRows == 0) {
      ASSERT_TRUE(partial == nullptr || partial->size() == 0);
      return;
    }
    ASSERT_EQ(partial->size(), decodeRows);
    for (velox::vector_size_t i = 0; i < decodeRows; ++i) {
      ASSERT_TRUE(partial->equalValueAt(full.get(), i, skipRows + i))
          << "Mismatch at partial row " << i << " (full row " << (skipRows + i)
          << ")\n  expected: "
          << full->toString(static_cast<velox::vector_size_t>(skipRows + i))
          << "\n  actual:   "
          << partial->toString(static_cast<velox::vector_size_t>(i));
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<velox::test::VectorMaker> vm_;
};

TEST_F(DeserializerSkipTest, reportsOutputRowsPerInputBatch) {
  const std::vector<velox::VectorPtr> inputs{
      makeBarrierBatch(/*base=*/100, /*rows=*/3, /*nestedNullRows=*/{}),
      makeBarrierBatch(/*base=*/200, /*rows=*/4, /*nestedNullRows=*/{1}),
  };
  auto [serialized, schema] = serialize(barrierType(), inputs);
  struct TestCase {
    std::string name;
    std::array<RowRange, 2> rowRanges;
    std::vector<uint32_t> expectedOutputRows;
  };
  const std::vector<TestCase> testCases{
      {
          .name = "allFull",
          .rowRanges = {RowRange{0, 3}, RowRange{0, 4}},
          .expectedOutputRows = {3, 4},
      },
      {
          .name = "allPartial",
          .rowRanges = {RowRange{1, 2}, RowRange{1, 3}},
          .expectedOutputRows = {1, 2},
      },
      {
          .name = "fullThenPartial",
          .rowRanges = {RowRange{0, 3}, RowRange{1, 3}},
          .expectedOutputRows = {3, 2},
      },
      {
          .name = "partialThenFull",
          .rowRanges = {RowRange{1, 2}, RowRange{0, 4}},
          .expectedOutputRows = {1, 4},
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    std::vector<std::string> tabletBatches;
    tabletBatches.reserve(serialized.size());
    for (size_t i = 0; i < serialized.size(); ++i) {
      tabletBatches.emplace_back(
          makeTabletBatch(serialized[i], testCase.rowRanges[i]));
    }
    std::vector<std::string_view> views;
    views.reserve(tabletBatches.size());
    for (const auto& batch : tabletBatches) {
      views.emplace_back(batch);
    }

    Deserializer deserializer{
        schema, pool_.get(), DeserializerOptions{.hasHeader = true}};
    velox::VectorPtr output;
    std::vector<uint32_t> outputRows;
    deserializer.deserialize(views, output, outputRows);

    EXPECT_EQ(outputRows, testCase.expectedOutputRows);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(
        output->size(),
        std::accumulate(
            testCase.expectedOutputRows.begin(),
            testCase.expectedOutputRows.end(),
            uint32_t{0}));
  }
}

// --- Dense scalar column, no nulls -----------------------------------------

TEST_F(DeserializerSkipTest, denseIntNoNullsSingleBatch) {
  auto type = velox::ROW({{"a", velox::INTEGER()}});
  auto batch = vm_->rowVector(
      {"a"}, {vm_->flatVector<int32_t>(20, [](velox::vector_size_t i) {
        return static_cast<int32_t>(i);
      })});
  auto [serialized, schema] = serialize(type, {batch});

  // Full read.
  checkSkip(schema, serialized, 0, 20);
  // Skip prefix, keep middle.
  checkSkip(schema, serialized, 5, 10);
  // Skip nothing, keep prefix.
  checkSkip(schema, serialized, 0, 7);
  // Skip and keep tail.
  checkSkip(schema, serialized, 15, 5);
  // Zero decode is a no-op (still exercises appendStreamSegments + skip path).
  checkSkip(schema, serialized, 8, 0);
}

// --- Dense scalar column, with nulls ---------------------------------------

TEST_F(DeserializerSkipTest, denseIntWithNullsSingleBatch) {
  auto type = velox::ROW({{"a", velox::INTEGER()}});
  auto batch = vm_->rowVector(
      {"a"},
      {vm_->flatVectorNullable<int32_t>(
          {0, std::nullopt, 2, 3, std::nullopt, 5, 6, std::nullopt, 8, 9})});
  auto [serialized, schema] = serialize(type, {batch});

  checkSkip(schema, serialized, 0, 10);
  checkSkip(schema, serialized, 3, 5);
  checkSkip(schema, serialized, 1, 8);
}

// --- Dense string column, no nulls -----------------------------------------

TEST_F(DeserializerSkipTest, denseStringNoNullsSingleBatch) {
  auto type = velox::ROW({{"s", velox::VARCHAR()}});
  auto batch = vm_->rowVector(
      {"s"},
      {vm_->flatVector<velox::StringView>(
          {"v0",
           "v1",
           "v2",
           "v3",
           "v4",
           "v5",
           "v6",
           "v7",
           "v8",
           "v9",
           "v10",
           "v11",
           "v12",
           "v13",
           "v14"})});
  auto [serialized, schema] = serialize(type, {batch});

  checkSkip(schema, serialized, 4, 6);
  checkSkip(schema, serialized, 10, 5);
}

// --- Dense string column, with nulls ---------------------------------------

TEST_F(DeserializerSkipTest, denseStringWithNullsSingleBatch) {
  auto type = velox::ROW({{"s", velox::VARCHAR()}});
  auto batch = vm_->rowVector(
      {"s"},
      {vm_->flatVectorNullable<velox::StringView>(
          {velox::StringView{"alpha"},
           std::nullopt,
           velox::StringView{"beta"},
           velox::StringView{"gamma"},
           std::nullopt,
           velox::StringView{"delta"},
           std::nullopt,
           velox::StringView{"epsilon"}})});
  auto [serialized, schema] = serialize(type, {batch});

  checkSkip(schema, serialized, 2, 4);
  checkSkip(schema, serialized, 3, 3);
}

// --- Concurrent skip: regression guard for per-decoder skipStringBuffers_ ---
//
// Each `BatchedStreamDecoder` owns a private `skipStringBuffers_` vector
// that backs the uncompressed blob a `TrivialEncoding<string_view>` allocates
// in its constructor when a skip first materializes the segment's stream data.
// If that vector were promoted to a `static` (or `static thread_local`) so
// multiple decoders shared it, concurrent skips would race on
// `emplace_back`/`clear`, and one decoder's end-of-run `reset()` (which runs
// `BatchedStreamDecoder::clear()` and drops the vector) would free the
// backing buffer another decoder's encoding still points into via its raw
// `blob_`/`pos_`. The result is either a data race the sanitizers catch, a
// heap-use-after-free during materialize, or silent string content corruption
// that fails deep equality against the reference decode.
TEST_F(DeserializerSkipTest, concurrentSkipStringNoSharedBuffers) {
  auto type = velox::ROW({{"s", velox::VARCHAR()}});
  constexpr int32_t kRows = 200;
  // >12 bytes forces out-of-line StringView so the output vector actually
  // references the encoding's backing buffer rather than inlining bytes.
  std::vector<std::string> storage;
  storage.reserve(kRows);
  std::vector<velox::StringView> viewsData;
  viewsData.reserve(kRows);
  for (int32_t i = 0; i < kRows; ++i) {
    storage.push_back(fmt::format("row_{:08d}_payload_ABCDEFGHIJKLMNOP", i));
    viewsData.emplace_back(storage.back());
  }
  auto batch =
      vm_->rowVector({"s"}, {vm_->flatVector<velox::StringView>(viewsData)});
  auto serializedPair = serialize(type, {batch});
  auto& serialized = serializedPair.first;
  auto schema = serializedPair.second;
  const std::vector<std::string_view> views{serialized[0]};

  DeserializerOptions dsOpts{.hasHeader = true};
  Deserializer refDs{schema, pool_.get(), dsOpts};
  velox::VectorPtr fullRef;
  refDs.deserialize(views, fullRef);
  ASSERT_EQ(fullRef->size(), kRows);

  constexpr uint32_t kSkip = 50;
  const std::vector<RowRange> rowRanges{RowRange{kSkip, kRows}};
  const auto expectedCount = static_cast<velox::vector_size_t>(kRows - kSkip);

  constexpr int kThreads = 8;
  constexpr int kIterations = 100;

  std::vector<std::shared_ptr<velox::memory::MemoryPool>> threadPools;
  threadPools.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threadPools.push_back(
        rootPool_->addLeafChild(fmt::format("skip_concurrent_{}", t)));
  }

  std::atomic<int> mismatches{0};
  std::atomic<int> exceptions{0};

  auto worker = [&](int threadIndex) {
    auto* threadPool = threadPools[threadIndex].get();
    for (int iter = 0; iter < kIterations; ++iter) {
      Deserializer ds{schema, threadPool, dsOpts};
      velox::VectorPtr out;
      try {
        ds.deserialize(views, rowRanges, out);
      } catch (const std::exception&) {
        ++exceptions;
        return;
      }
      if (out == nullptr || out->size() != expectedCount) {
        ++mismatches;
        return;
      }
      for (velox::vector_size_t i = 0; i < expectedCount; ++i) {
        if (!out->equalValueAt(fullRef.get(), i, kSkip + i)) {
          ++mismatches;
          return;
        }
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(mismatches.load(), 0);
  EXPECT_EQ(exceptions.load(), 0);
}

// --- Multi-batch: skip across batch boundary -------------------------------

TEST_F(DeserializerSkipTest, denseIntMultipleBatchesSkipAcrossBoundary) {
  auto type = velox::ROW({{"a", velox::INTEGER()}});
  auto b1 = vm_->rowVector(
      {"a"}, {vm_->flatVector<int32_t>(10, [](velox::vector_size_t i) {
        return static_cast<int32_t>(i);
      })});
  auto b2 = vm_->rowVector(
      {"a"}, {vm_->flatVector<int32_t>(15, [](velox::vector_size_t i) {
        return static_cast<int32_t>(100 + i);
      })});
  auto b3 = vm_->rowVector(
      {"a"}, {vm_->flatVector<int32_t>(8, [](velox::vector_size_t i) {
        return static_cast<int32_t>(1000 + i);
      })});
  auto [serialized, schema] = serialize(type, {b1, b2, b3});

  // Skip within first batch, decode spanning first + second.
  checkSkip(schema, serialized, 5, 10);
  // Skip past first batch entirely, decode in second batch.
  checkSkip(schema, serialized, 10, 8);
  // Skip past first two batches, decode in third.
  checkSkip(schema, serialized, 25, 5);
  // Skip past first, decode across second and third.
  checkSkip(schema, serialized, 12, 15);
  // Full read.
  checkSkip(schema, serialized, 0, 33);
}

// --- FlatMap column (in-map + scattered value), no null values -------------

TEST_F(DeserializerSkipTest, flatMapNoNullValuesSingleBatch) {
  // Explicit per-row key subsets exercise in-map presence gaps (some rows
  // are missing certain keys) and scattered value reads.
  using MapEntries = std::vector<std::pair<int32_t, std::optional<int32_t>>>;
  std::vector<MapEntries> maps{
      {{0, 100}, {1, 101}},
      {{1, 111}},
      {{0, 200}, {2, 202}},
      {{2, 222}},
      {{0, 300}, {1, 301}, {2, 302}},
      {{1, 411}},
      {{0, 500}, {1, 501}},
      {{2, 622}},
      {{0, 700}, {1, 701}, {2, 702}},
      {{1, 811}, {2, 812}},
      {{0, 900}},
      {{1, 1011}, {2, 1012}},
  };
  auto type =
      velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::INTEGER())}});
  auto batch = vm_->rowVector({"m"}, {vm_->mapVector<int32_t, int32_t>(maps)});

  folly::F14FastMap<std::string, std::set<std::string>> flatMaps{{"m", {}}};
  auto [serialized, schema] = serialize(type, {batch}, flatMaps);

  checkSkip(schema, serialized, 0, 12);
  checkSkip(schema, serialized, 3, 6);
  checkSkip(schema, serialized, 5, 5);
  checkSkip(schema, serialized, 7, 5);
}

// --- FlatMap column with NULL values ---------------------------------------

TEST_F(DeserializerSkipTest, flatMapWithNullValuesSingleBatch) {
  // Some map values are null. The value stream for those keys carries null
  // bits alongside the scatter bitmap (from the in-map presence stream).
  using MapEntries = std::vector<std::pair<int32_t, std::optional<int32_t>>>;
  std::vector<MapEntries> maps{
      {{0, std::nullopt}, {1, 101}},
      {{1, 111}},
      {{0, 200}, {2, std::nullopt}},
      {{2, 222}},
      {{0, std::nullopt}, {1, 301}, {2, 302}},
      {{1, 411}},
      {{0, 500}, {1, std::nullopt}},
      {{2, 622}},
      {{0, 700}, {1, 701}, {2, std::nullopt}},
      {{1, std::nullopt}, {2, 812}},
  };
  auto type =
      velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::INTEGER())}});
  auto batch = vm_->rowVector({"m"}, {vm_->mapVector<int32_t, int32_t>(maps)});

  folly::F14FastMap<std::string, std::set<std::string>> flatMaps{{"m", {}}};
  auto [serialized, schema] = serialize(type, {batch}, flatMaps);

  checkSkip(schema, serialized, 0, 10);
  checkSkip(schema, serialized, 2, 6);
  checkSkip(schema, serialized, 4, 4);
}

// --- FlatMap column across multiple batches --------------------------------

TEST_F(DeserializerSkipTest, flatMapMultipleBatchesSkipAcrossBoundary) {
  using MapEntries = std::vector<std::pair<int32_t, std::optional<int32_t>>>;

  // b1: keys 0 and 1 only.
  std::vector<MapEntries> b1Maps{
      {{0, 1}, {1, 2}},
      {{1, 3}},
      {{0, 4}, {1, 5}},
      {{0, 6}},
      {{0, 7}, {1, 8}},
      {{1, 9}},
      {{0, 10}, {1, 11}},
      {{1, 12}},
  };
  // b2: keys 5 and 6 (absent in b1) — the FlatMap tree adds new key streams
  // that were entirely missing from b1's serialized output.
  std::vector<MapEntries> b2Maps{
      {{5, 100}, {6, 200}},
      {{5, 101}},
      {{6, 201}},
      {{5, 102}, {6, 202}},
      {{5, 103}, {6, 203}},
      {{5, 104}},
      {{6, 204}},
      {{5, 105}, {6, 205}},
      {{5, 106}, {6, 206}},
      {{5, 107}, {6, 207}},
  };
  auto type =
      velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::INTEGER())}});
  auto b1 = vm_->rowVector({"m"}, {vm_->mapVector<int32_t, int32_t>(b1Maps)});
  auto b2 = vm_->rowVector({"m"}, {vm_->mapVector<int32_t, int32_t>(b2Maps)});

  folly::F14FastMap<std::string, std::set<std::string>> flatMaps{{"m", {}}};
  auto [serialized, schema] = serialize(type, {b1, b2}, flatMaps);

  // Skip within b1, decode spanning boundary.
  checkSkip(schema, serialized, 3, 8);
  // Skip past b1 entirely, decode in b2.
  checkSkip(schema, serialized, 8, 6);
  // Full.
  checkSkip(schema, serialized, 0, 18);
}

// --- In-map skip with empty streamSegments_ (regression guard) -------------

// Predefined FlatMap key "3" is declared in the schema but never appears in
// the data — its in-map decoder therefore accumulates zero stream segments
// for this batch. A skip through this decoder (triggered here by a
// rowRange with startRow > 0) must not throw on the empty-streamSegments_
// branch. Before the fix, `BatchedStreamDecoder::skip` on an in-map
// decoder with empty segments tripped `NIMBLE_CHECK(!isInMapStream() && ...)`;
// after the fix `skipInMap` handles the empty case.
TEST_F(DeserializerSkipTest, skipHandlesInMapWithEmptyStreamSegments) {
  auto type =
      velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::INTEGER())}});
  using MapEntries = std::vector<std::pair<int32_t, std::optional<int32_t>>>;
  std::vector<MapEntries> maps{
      {{1, 100}, {2, 200}},
      {{1, 111}},
      {{2, 222}},
      {{1, 133}, {2, 233}},
      {{2, 244}},
      {{1, 155}, {2, 255}},
      {{1, 166}},
      {{2, 277}},
  };
  auto batch = vm_->rowVector({"m"}, {vm_->mapVector<int32_t, int32_t>(maps)});
  // Include "3" in the predefined key set even though no row uses it —
  // this forces the deserializer to construct a decoder for key "3" whose
  // stream segments stay empty across the batch, which is the shape the
  // skip path needs to tolerate.
  folly::F14FastMap<std::string, std::set<std::string>> flatMaps{
      {"m", {"1", "2", "3"}}};
  auto [serialized, schema] = serialize(type, {batch}, flatMaps);

  DeserializerOptions dsOpts{.hasHeader = true};
  Deserializer d{schema, pool_.get(), dsOpts};
  std::vector<std::string_view> views{serialized[0]};
  velox::VectorPtr out;
  // Non-zero startRow forces skip through every FlatMap key's in-map
  // decoder — including key "3" whose streamSegments_ is empty.
  ASSERT_NO_THROW(
      d.deserialize(views, std::vector<RowRange>{RowRange{3, 7}}, out));
  ASSERT_EQ(out->size(), 4);
}

// --- Row-null skip: top-level row nulls ------------------------------------

TEST_F(DeserializerSkipTest, rowLevelNullsSingleBatch) {
  // Top-level row nulls (e.g. via appendNulls) exercise the empty
  // streamSegments_ branch in skip (Row null stream reconstructed as
  // all-non-null when nothing was written on the wire), plus the null
  // barrier flag when a batch actually has row-level nulls.
  auto type = velox::ROW({{"a", velox::INTEGER()}});
  auto inner = vm_->flatVector<int32_t>(
      10, [](velox::vector_size_t i) { return static_cast<int32_t>(i); });
  auto batch = vm_->rowVector({"a"}, {inner});

  auto [serialized, schema] = serialize(type, {batch});
  checkSkip(schema, serialized, 0, 10);
  checkSkip(schema, serialized, 4, 6);
  checkSkip(schema, serialized, 7, 3);
}

// FlatMap where every row carries every key (writer omits in-map streams
// as all-present -> deserializer synthesizes an all-present segment
// spanning the whole batch). A mid-batch skip then lands inside the
// segment; the next read must correctly clamp the partial-segment write.
// Regression guard for a prior `fillInMapGap` invariant that assumed
// segments never straddled a skip boundary.
TEST_F(DeserializerSkipTest, flatMapAllPresentSegmentStraddleSkip) {
  using MapEntries = std::vector<std::pair<int32_t, std::optional<int32_t>>>;
  std::vector<MapEntries> maps;
  maps.reserve(10);
  for (int32_t i = 0; i < 10; ++i) {
    maps.push_back({{1, 100 + i}, {2, 200 + i}});
  }
  auto type =
      velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::INTEGER())}});
  auto batch = vm_->rowVector({"m"}, {vm_->mapVector<int32_t, int32_t>(maps)});
  folly::F14FastMap<std::string, std::set<std::string>> flatMaps{{"m", {}}};
  auto [serialized, schema] = serialize(type, {batch}, flatMaps);

  // Cover: prefix skip, mid slice, tail slice, single-row slice at
  // boundary. Each lands inside the synthesized all-present segment.
  checkSkip(schema, serialized, 3, 5);
  checkSkip(schema, serialized, 0, 1);
  checkSkip(schema, serialized, 9, 1);
  checkSkip(schema, serialized, 4, 3);
}

// --- Large-volume skip/read interleaving -----------------------------------

// Skip almost the whole batch (999 of 1000 rows), decode a single tail
// row. Stresses `skipEncoded` walking many queued segments cheaply.
TEST_F(DeserializerSkipTest, denseIntLargeSkipTinyRead) {
  auto type = velox::ROW({{"a", velox::INTEGER()}});
  auto batch = vm_->rowVector(
      {"a"}, {vm_->flatVector<int32_t>(1000, [](velox::vector_size_t i) {
        return static_cast<int32_t>(i);
      })});
  auto [serialized, schema] = serialize(type, {batch});

  checkSkip(schema, serialized, 999, 1);
  checkSkip(schema, serialized, 500, 1);
  checkSkip(schema, serialized, 1, 999);
}

// Full skip (skipRows == totalRows). Decode 0 rows after skipping the
// whole run — output should be an empty non-null vector.
TEST_F(DeserializerSkipTest, denseIntFullSkip) {
  auto type = velox::ROW({{"a", velox::INTEGER()}});
  auto batch = vm_->rowVector(
      {"a"}, {vm_->flatVector<int32_t>(20, [](velox::vector_size_t i) {
        return static_cast<int32_t>(i);
      })});
  auto [serialized, schema] = serialize(type, {batch});

  checkSkip(schema, serialized, 20, 0);
}

// Many small batches, alternating skip and read via per-batch rowRanges.
// Uses the public `deserialize(data, rowRanges, out)` overload directly
// so ranges are disjoint (can't be expressed as a single run-level
// window). Verifies planReaderOps coalesces contiguous windows and
// separates them by exactly one skip.
TEST_F(DeserializerSkipTest, manyBatchesDisjointRowRanges) {
  auto type = velox::ROW({{"a", velox::INTEGER()}});
  constexpr uint32_t kBatchSize = 10;
  constexpr size_t kNumBatches = 20;

  std::vector<velox::VectorPtr> inputs;
  inputs.reserve(kNumBatches);
  for (size_t b = 0; b < kNumBatches; ++b) {
    inputs.push_back(vm_->rowVector(
        {"a"},
        {vm_->flatVector<int32_t>(kBatchSize, [b](velox::vector_size_t i) {
          return static_cast<int32_t>(b * 1000 + i);
        })}));
  }

  SerializerOptions options{.version = SerializationVersion::kSerialization};
  Serializer serializer{options, type, pool_.get()};
  std::vector<std::string> serialized;
  serialized.reserve(inputs.size());
  for (const auto& input : inputs) {
    serialized.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
  }
  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());

  // Per-batch rowRanges: even batches keep rows [2, 7); odd batches
  // decode nothing. In the run's coordinate space this yields 10
  // disjoint windows separated by 15-row gaps each.
  std::vector<nimble::RowRange> rowRanges;
  rowRanges.reserve(kNumBatches);
  for (size_t b = 0; b < kNumBatches; ++b) {
    rowRanges.push_back(
        b % 2 == 0 ? nimble::RowRange{2, 7} : nimble::RowRange{0, 0});
  }

  std::vector<std::string_view> views;
  views.reserve(serialized.size());
  for (const auto& s : serialized) {
    views.push_back(s);
  }

  DeserializerOptions dsOpts{.hasHeader = true};
  Deserializer ds{schema, pool_.get(), dsOpts};
  velox::VectorPtr out;
  ds.deserialize(views, rowRanges, out);

  // Expected: 5 rows per even batch × 10 even batches = 50 rows.
  ASSERT_EQ(out->size(), 50);
  auto* col =
      out->as<velox::RowVector>()->childAt(0)->as<velox::FlatVector<int32_t>>();
  for (size_t evenIdx = 0; evenIdx < 10; ++evenIdx) {
    const size_t sourceBatch = evenIdx * 2;
    for (uint32_t r = 0; r < 5; ++r) {
      const auto expected = static_cast<int32_t>(sourceBatch * 1000 + (2 + r));
      EXPECT_EQ(col->valueAt(evenIdx * 5 + r), expected)
          << "evenIdx=" << evenIdx << " r=" << r;
    }
  }
}

// --- Nested-type skipping (Array, Map non-FlatMap) -------------------------

// Array<int>: the lengths stream drives per-row width; skip must advance
// both the lengths cursor and (indirectly) the elements cursor via the
// FieldReader tree.
TEST_F(DeserializerSkipTest, arrayIntSingleBatch) {
  auto type = velox::ROW({{"a", velox::ARRAY(velox::INTEGER())}});
  std::vector<std::vector<int32_t>> arrays;
  arrays.reserve(12);
  for (int32_t i = 0; i < 12; ++i) {
    // Row i has (i % 4) + 1 elements: 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4.
    std::vector<int32_t> row;
    for (int32_t j = 0; j <= (i % 4); ++j) {
      row.push_back(i * 100 + j);
    }
    arrays.push_back(std::move(row));
  }
  auto batch = vm_->rowVector({"a"}, {vm_->arrayVector<int32_t>(arrays)});
  auto [serialized, schema] = serialize(type, {batch});

  checkSkip(schema, serialized, 3, 6);
  checkSkip(schema, serialized, 5, 5);
  checkSkip(schema, serialized, 0, 12);
}

// Map<int,int> (non-FlatMap): distinct from FlatMap in that entries are
// per-row, not per-key. Skip advances offsets/lengths/keys/values
// together via the reader tree.
TEST_F(DeserializerSkipTest, mapIntIntSingleBatch) {
  auto type =
      velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::INTEGER())}});
  using MapEntries = std::vector<std::pair<int32_t, std::optional<int32_t>>>;
  std::vector<MapEntries> maps;
  maps.reserve(10);
  for (int32_t i = 0; i < 10; ++i) {
    // Row i has i+1 entries.
    MapEntries row;
    for (int32_t j = 0; j <= i; ++j) {
      row.emplace_back(j, i * 10 + j);
    }
    maps.push_back(std::move(row));
  }
  auto batch = vm_->rowVector({"m"}, {vm_->mapVector<int32_t, int32_t>(maps)});
  // Note: no flatMapColumns entry — this is a regular Map, not FlatMap.
  auto [serialized, schema] = serialize(type, {batch});

  checkSkip(schema, serialized, 4, 4);
  checkSkip(schema, serialized, 7, 3);
}

// --- Bad user input rejected ----------------------------------------------

// Bad per-batch rowRange: startRow > endRow. Would underflow the uint32
// numRows() computation and confuse `planReaderOps` if not rejected.
// Guards every barrier test below by confirming `makeBarrierBatch` produces
// what it claims: nulls in the nested Row set the header's barrier flag, and
// the same schema without nulls leaves it clear. If this helper ever stopped
// producing barrier batches, those tests would silently run on ordinary
// batches and still pass.
TEST_F(DeserializerSkipTest, barrierFlagSetOnlyWhenNestedRowHasNulls) {
  auto readBarrierFlag = [&](const std::string& blob) {
    DeserializerOptions dsOpts{.hasHeader = true};
    serde::StreamDataParser parser{pool_.get(), dsOpts};
    parser.initialize(blob);
    return parser.requiresNullBarrier();
  };

  auto [withNulls, schema1] =
      serialize(barrierType(), {makeBarrierBatch(0, 10, {3, 7})});
  EXPECT_TRUE(readBarrierFlag(withNulls[0]));

  auto [noNulls, schema2] =
      serialize(barrierType(), {makeBarrierBatch(0, 10, {})});
  EXPECT_FALSE(readBarrierFlag(noNulls[0]));
}

// A null-barrier batch decodes standalone, which is exactly the unit a
// rowRange applies to. Uses the file's equivalence property:
// deserialize(data, ranges) == deserialize(data) sliced by the same window.
TEST_F(DeserializerSkipTest, barrierSingleBatchWithRowRange) {
  auto [serialized, schema] =
      serialize(barrierType(), {makeBarrierBatch(0, 10, {3, 7})});
  checkSkip(schema, serialized, /*skipRows=*/2, /*decodeRows=*/5);
}

// A barrier batch sandwiched between ordinary ones: the run is flushed before
// and after it, so the ranges must still line up across all three.
TEST_F(DeserializerSkipTest, barrierBatchAmongNonBarrierBatches) {
  auto [serialized, schema] = serialize(
      barrierType(),
      {makeBarrierBatch(0, 5, {}),
       makeBarrierBatch(5, 5, {1, 3}),
       makeBarrierBatch(10, 5, {})});
  checkSkip(schema, serialized, /*skipRows=*/3, /*decodeRows=*/9);
}

// Every batch flagged: each becomes its own run.
TEST_F(DeserializerSkipTest, barrierEveryBatch) {
  auto [serialized, schema] = serialize(
      barrierType(),
      {makeBarrierBatch(0, 5, {2}),
       makeBarrierBatch(5, 5, {1}),
       makeBarrierBatch(10, 5, {4})});
  checkSkip(schema, serialized, /*skipRows=*/4, /*decodeRows=*/7);
}

// Range boundaries against a barrier batch: whole batch, empty, and a window
// that ends exactly on the batch boundary.
TEST_F(DeserializerSkipTest, barrierBatchRangeBoundaries) {
  auto [serialized, schema] = serialize(
      barrierType(),
      {makeBarrierBatch(0, 6, {0, 5}), makeBarrierBatch(6, 6, {2})});
  checkSkip(schema, serialized, /*skipRows=*/0, /*decodeRows=*/12);
  checkSkip(schema, serialized, /*skipRows=*/0, /*decodeRows=*/6);
  checkSkip(schema, serialized, /*skipRows=*/6, /*decodeRows=*/6);
  checkSkip(schema, serialized, /*skipRows=*/5, /*decodeRows=*/2);
  checkSkip(schema, serialized, /*skipRows=*/3, /*decodeRows=*/0);
}

// The barrier-only in-map path. Key "a" is in every row, so its in-map stream
// is omitted and the reader reconstructs it; on a barrier batch that goes
// through the `addPresentInMapBatch()` sentinel (endRow = kPresentInMapEndRow),
// which was written assuming the whole batch is read. A rowRange makes the
// reader skip instead, exercising the sentinel's clamping in skipInMap and
// fillInMapGap. Key "b" is on even rows only, so its in-map stream is really
// written and both in-map shapes are covered in one batch.
TEST_F(DeserializerSkipTest, barrierFlatMapInMapWithRowRange) {
  std::vector<std::vector<std::string>> keysByRow;
  for (int i = 0; i < 8; ++i) {
    std::vector<std::string> keys{"a"};
    if (i % 2 == 0) {
      keys.emplace_back("b");
    }
    keysByRow.push_back(std::move(keys));
  }
  auto [serialized, schema] = serialize(
      flatMapBarrierType(),
      {makeFlatMapBarrierBatch(keysByRow, /*nestedNullRows=*/{2, 5})},
      /*flatMapColumns=*/{{"m", {}}});

  checkSkip(schema, serialized, /*skipRows=*/0, /*decodeRows=*/8);
  checkSkip(schema, serialized, /*skipRows=*/1, /*decodeRows=*/5);
  checkSkip(schema, serialized, /*skipRows=*/3, /*decodeRows=*/2);
  checkSkip(schema, serialized, /*skipRows=*/5, /*decodeRows=*/3);
}

// Same shape across several barrier batches, so the sentinel is re-created per
// batch and the reader is reset between runs.
TEST_F(DeserializerSkipTest, barrierFlatMapInMapMultipleBatches) {
  auto makeBatch = [&](int rows) {
    std::vector<std::vector<std::string>> keysByRow;
    for (int i = 0; i < rows; ++i) {
      std::vector<std::string> keys{"a"};
      if (i % 3 == 0) {
        keys.emplace_back("b");
      }
      keysByRow.push_back(std::move(keys));
    }
    return makeFlatMapBarrierBatch(keysByRow, /*nestedNullRows=*/{1});
  };
  auto [serialized, schema] = serialize(
      flatMapBarrierType(),
      {makeBatch(6), makeBatch(6), makeBatch(6)},
      /*flatMapColumns=*/{{"m", {}}});

  checkSkip(schema, serialized, /*skipRows=*/2, /*decodeRows=*/11);
}

TEST_F(DeserializerSkipTest, rejectRowRangeStartRowAfterEndRow) {
  auto type = velox::ROW({{"a", velox::INTEGER()}});
  auto batch = vm_->rowVector(
      {"a"}, {vm_->flatVector<int32_t>(10, [](velox::vector_size_t i) {
        return static_cast<int32_t>(i);
      })});
  auto [serialized, schema] = serialize(type, {batch});

  DeserializerOptions dsOpts{.hasHeader = true};
  Deserializer ds{schema, pool_.get(), dsOpts};
  std::vector<std::string_view> views{serialized[0]};
  velox::VectorPtr out;
  EXPECT_THROW(
      ds.deserialize(views, {nimble::RowRange{5, 3}}, out), NimbleUserError);
}

// Bad per-batch rowRange: endRow past batch rowCount.
TEST_F(DeserializerSkipTest, rejectRowRangeEndRowPastBatchRowCount) {
  auto type = velox::ROW({{"a", velox::INTEGER()}});
  auto batch = vm_->rowVector(
      {"a"}, {vm_->flatVector<int32_t>(10, [](velox::vector_size_t i) {
        return static_cast<int32_t>(i);
      })});
  auto [serialized, schema] = serialize(type, {batch});

  DeserializerOptions dsOpts{.hasHeader = true};
  Deserializer ds{schema, pool_.get(), dsOpts};
  std::vector<std::string_view> views{serialized[0]};
  velox::VectorPtr out;
  EXPECT_THROW(
      ds.deserialize(views, {nimble::RowRange{0, 11}}, out), NimbleUserError);
}

// Bad input: data.size() != rowRanges.size().
TEST_F(DeserializerSkipTest, rejectMismatchedRowRangesSize) {
  auto type = velox::ROW({{"a", velox::INTEGER()}});
  auto batch = vm_->rowVector(
      {"a"}, {vm_->flatVector<int32_t>(10, [](velox::vector_size_t i) {
        return static_cast<int32_t>(i);
      })});
  auto [serialized, schema] = serialize(type, {batch});

  DeserializerOptions dsOpts{.hasHeader = true};
  Deserializer ds{schema, pool_.get(), dsOpts};
  std::vector<std::string_view> views{serialized[0]};
  velox::VectorPtr out;
  std::vector<nimble::RowRange> ranges{
      nimble::RowRange{0, 5}, nimble::RowRange{0, 5}};
  EXPECT_THROW(ds.deserialize(views, ranges, out), NimbleUserError);
}

} // namespace facebook::nimble
