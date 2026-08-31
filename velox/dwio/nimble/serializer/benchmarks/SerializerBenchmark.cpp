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

#include <fmt/core.h>
#include <folly/Benchmark.h>
#include <folly/Random.h>
#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/velox/OrderedRanges.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

using namespace facebook::nimble;
namespace velox = facebook::velox;

namespace {

constexpr int kRandom = 0;
constexpr int kMainlyConstant = 1;
constexpr int kIncreasing = 2;
constexpr int kLowCardinality = 3;
constexpr int kRunLength = 4;
constexpr velox::vector_size_t kNumRows = 10'000;
constexpr int32_t kNumKeys = 100;
constexpr uint32_t kMemoryReportIters = 10;

velox::RowVectorPtr makeWideFlatMapBatch(
    velox::vector_size_t numRows,
    int32_t numKeys,
    int dataPattern,
    velox::memory::MemoryPool* pool) {
  auto type = velox::ROW(
      {{"id", velox::BIGINT()},
       {"features", velox::MAP(velox::INTEGER(), velox::BIGINT())}});

  auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool);
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ids->asFlatVector<int64_t>()->set(i, i);
  }

  const auto totalEntries =
      static_cast<velox::vector_size_t>(numRows) * numKeys;
  auto keys = velox::BaseVector::create(velox::INTEGER(), totalEntries, pool);
  auto values = velox::BaseVector::create(velox::BIGINT(), totalEntries, pool);

  auto* rawKeys = keys->asFlatVector<int32_t>();
  auto* rawValues = values->asFlatVector<int64_t>();

  velox::vector_size_t offset = 0;
  int64_t runningValue = 0;
  int64_t runValue = folly::Random::rand64();
  int32_t runRemaining = 0;

  for (velox::vector_size_t row = 0; row < numRows; ++row) {
    for (int32_t k = 0; k < numKeys; ++k) {
      rawKeys->set(offset, k);
      int64_t v;
      switch (dataPattern) {
        case kRandom:
          v = folly::Random::rand64();
          break;
        case kMainlyConstant:
          v = (folly::Random::rand32() % 100 < 95)
              ? 42
              : static_cast<int64_t>(folly::Random::rand64());
          break;
        case kIncreasing:
          runningValue += folly::Random::rand32() % 10;
          v = runningValue;
          break;
        case kLowCardinality:
          v = folly::Random::rand32() % 64;
          break;
        case kRunLength:
          if (runRemaining == 0) {
            runValue = folly::Random::rand64() % 1000;
            runRemaining = 10 + folly::Random::rand32() % 50;
          }
          v = runValue;
          --runRemaining;
          break;
        default:
          v = 0;
      }
      rawValues->set(offset, v);
      ++offset;
    }
  }

  auto mapVector = std::make_shared<velox::MapVector>(
      pool,
      velox::MAP(velox::INTEGER(), velox::BIGINT()),
      /*nulls=*/nullptr,
      numRows,
      velox::allocateOffsets(numRows, pool),
      velox::allocateSizes(numRows, pool),
      keys,
      values);

  auto* rawOffsets =
      mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
  auto* rawSizes =
      mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * numKeys;
    rawSizes[i] = numKeys;
  }

  return std::make_shared<velox::RowVector>(
      pool,
      type,
      /*nulls=*/nullptr,
      numRows,
      std::vector<velox::VectorPtr>{ids, mapVector});
}

/// Builds an EncodingLayoutTree that forces all flat map value streams to use
/// the specified encoding type.
EncodingLayoutTree buildForcedEncodingLayout(
    int32_t numKeys,
    EncodingType valueEncoding) {
  using SI = EncodingLayoutTree::StreamIdentifiers;
  EncodingLayout valueLayout{
      valueEncoding,
      /*encodingConfig=*/{},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt, std::nullopt}};

  std::vector<EncodingLayoutTree> flatMapChildren;
  flatMapChildren.reserve(numKeys);
  for (int32_t k = 0; k < numKeys; ++k) {
    flatMapChildren.emplace_back(
        Kind::Scalar,
        std::unordered_map<uint8_t, EncodingLayout>{
            {SI::Scalar::ScalarStream, valueLayout}},
        std::to_string(k));
  }

  return EncodingLayoutTree{
      Kind::Row,
      {},
      "",
      {EncodingLayoutTree{Kind::Scalar, {}, ""},
       EncodingLayoutTree{Kind::FlatMap, {}, "", std::move(flatMapChildren)}}};
}

struct BenchmarkState {
  std::string serialized;
  std::shared_ptr<const Type> schema;
};

BenchmarkState prepareBenchmark(
    velox::vector_size_t numRows,
    int32_t numKeys,
    int dataPattern,
    std::optional<EncodingType> forcedEncoding = std::nullopt) {
  auto pool = velox::memory::memoryManager()->addLeafPool("prepare");
  BenchmarkState state;

  auto input = makeWideFlatMapBatch(numRows, numKeys, dataPattern, pool.get());

  auto makeOptions = [&]() -> SerializerOptions {
    if (forcedEncoding.has_value()) {
      return SerializerOptions{
          .version = SerializationVersion::kSerialization,
          .flatMapColumns = {{"features", {}}},
          .encodingLayoutTree =
              buildForcedEncodingLayout(numKeys, forcedEncoding.value()),
          .compressionOptions = CompressionOptions{},
      };
    }
    return SerializerOptions{
        .version = SerializationVersion::kSerialization,
        .flatMapColumns = {{"features", {}}},
    };
  };

  Serializer serializer{makeOptions(), input->type(), pool.get()};
  auto sv = serializer.serialize(input, OrderedRanges::of(0, input->size()));
  state.serialized = std::string(sv.data(), sv.size());
  state.schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  return state;
}

struct DeserializeStats {
  int64_t peakMemoryBytes;
  uint64_t cumulativeBytes;
  uint64_t numAllocs;
};

DeserializeStats runDeserializeWithStats(
    const BenchmarkState& state,
    size_t bufferPoolCapacity,
    uint32_t iters) {
  auto pool = velox::memory::memoryManager()->addLeafPool("deser_stats");
  Deserializer deserializer{
      state.schema,
      pool.get(),
      DeserializerOptions{
          .hasHeader = true,
          .bufferPoolCapacity = bufferPoolCapacity,
      }};
  velox::VectorPtr output;

  while (iters--) {
    deserializer.deserialize(state.serialized, output);
    output.reset();
  }

  auto s = pool->stats();
  return DeserializeStats{
      .peakMemoryBytes = pool->peakBytes(),
      .cumulativeBytes = s.cumulativeBytes,
      .numAllocs = s.numAllocs,
  };
}

void runDeserialize(
    const BenchmarkState& state,
    size_t bufferPoolCapacity,
    uint32_t iters) {
  auto pool = velox::memory::memoryManager()->addLeafPool("deser_bench");
  Deserializer deserializer{
      state.schema,
      pool.get(),
      DeserializerOptions{
          .hasHeader = true,
          .bufferPoolCapacity = bufferPoolCapacity,
      }};
  velox::VectorPtr output;
  while (iters--) {
    deserializer.deserialize(state.serialized, output);
    output.reset();
  }
}

void printMemoryReport(
    const char* label,
    velox::vector_size_t numRows,
    int32_t numKeys,
    int dataPattern,
    std::optional<EncodingType> forcedEncoding = std::nullopt) {
  auto state = prepareBenchmark(numRows, numKeys, dataPattern, forcedEncoding);

  auto withPool = runDeserializeWithStats(
      state,
      /*bufferPoolCapacity=*/
      velox::BufferPool::kDefaultCapacity,
      kMemoryReportIters);
  auto noPool = runDeserializeWithStats(
      state, /*bufferPoolCapacity=*/0, kMemoryReportIters);

  double allocSavings = noPool.numAllocs > 0 ? 100.0 *
          (1.0 - static_cast<double>(withPool.numAllocs) / noPool.numAllocs)
                                             : 0.0;

  fmt::print(
      "  {:>20} | allocs: {:>6} vs {:>6} ({:>+6.1f}%)"
      "  peak: {:>8} vs {:>8}\n",
      label,
      withPool.numAllocs,
      noPool.numAllocs,
      allocSavings,
      fmt::format("{:.0f}KB", withPool.peakMemoryBytes / 1024.0),
      fmt::format("{:.0f}KB", noPool.peakMemoryBytes / 1024.0));
}

} // namespace

// === Deserialization throughput with forced encoding types ===

#define DESER_BENCH(Name, Rows, Keys, Pattern, ...)                           \
  BENCHMARK(Deser_##Name##_Pool, iters) {                                     \
    static auto state = prepareBenchmark(Rows, Keys, Pattern, ##__VA_ARGS__); \
    runDeserialize(                                                           \
        state, /*bufferPoolCapacity=*/                                        \
        velox::BufferPool::kDefaultCapacity,                                  \
        iters);                                                               \
  }                                                                           \
  BENCHMARK_RELATIVE(Deser_##Name##_NoPool, iters) {                          \
    static auto state = prepareBenchmark(Rows, Keys, Pattern, ##__VA_ARGS__); \
    runDeserialize(state, /*bufferPoolCapacity=*/0, iters);                   \
  }

// --- Forced encoding types ---
DESER_BENCH(Trivial, kNumRows, kNumKeys, kRandom, EncodingType::Trivial)
BENCHMARK_DRAW_LINE();
DESER_BENCH(
    Dictionary,
    kNumRows,
    kNumKeys,
    kLowCardinality,
    EncodingType::Dictionary)
BENCHMARK_DRAW_LINE();
DESER_BENCH(
    MainlyConst,
    kNumRows,
    kNumKeys,
    kMainlyConstant,
    EncodingType::MainlyConstant)
BENCHMARK_DRAW_LINE();
DESER_BENCH(
    FixedBitWidth,
    kNumRows,
    kNumKeys,
    kRandom,
    EncodingType::FixedBitWidth)
BENCHMARK_DRAW_LINE();
DESER_BENCH(Delta, kNumRows, kNumKeys, kIncreasing, EncodingType::Delta)
BENCHMARK_DRAW_LINE();
DESER_BENCH(RLE, kNumRows, kNumKeys, kRunLength, EncodingType::RLE)
BENCHMARK_DRAW_LINE();

// --- Auto-selected encoding ---
DESER_BENCH(Auto_Random, kNumRows, kNumKeys, kRandom)
BENCHMARK_DRAW_LINE();
DESER_BENCH(Auto_MainlyConst, kNumRows, kNumKeys, kMainlyConstant)
BENCHMARK_DRAW_LINE();

// --- Width scaling ---
DESER_BENCH(Auto_50Keys, kNumRows, 50, kRandom)
BENCHMARK_DRAW_LINE();
DESER_BENCH(Auto_200Keys, kNumRows, 200, kRandom)

#undef DESER_BENCH

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});

  fmt::print(
      "\n=== Allocation Count: BufferPool ON vs OFF ({} rows x {} keys, {} batches) ===\n\n",
      kNumRows,
      kNumKeys,
      kMemoryReportIters);

  fmt::print("  Forced encoding types:\n");
  printMemoryReport(
      "Trivial", kNumRows, kNumKeys, kRandom, EncodingType::Trivial);
  printMemoryReport(
      "Dictionary",
      kNumRows,
      kNumKeys,
      kLowCardinality,
      EncodingType::Dictionary);
  printMemoryReport(
      "MainlyConstant",
      kNumRows,
      kNumKeys,
      kMainlyConstant,
      EncodingType::MainlyConstant);
  printMemoryReport(
      "FixedBitWidth",
      kNumRows,
      kNumKeys,
      kRandom,
      EncodingType::FixedBitWidth);
  printMemoryReport(
      "Delta", kNumRows, kNumKeys, kIncreasing, EncodingType::Delta);
  printMemoryReport("RLE", kNumRows, kNumKeys, kRunLength, EncodingType::RLE);

  fmt::print("\n  Auto-selected encoding:\n");
  printMemoryReport("Auto (Random)", kNumRows, kNumKeys, kRandom);
  printMemoryReport("Auto (MainlyConst)", kNumRows, kNumKeys, kMainlyConstant);
  printMemoryReport("Auto (Increasing)", kNumRows, kNumKeys, kIncreasing);
  printMemoryReport("Auto (LowCard)", kNumRows, kNumKeys, kLowCardinality);
  printMemoryReport("Auto (RunLength)", kNumRows, kNumKeys, kRunLength);

  fmt::print("\n  Width scaling (auto-selected, random data):\n");
  for (int numKeys : {50, 100, 200}) {
    printMemoryReport(
        fmt::format("{}keys", numKeys).c_str(), kNumRows, numKeys, kRandom);
  }

  fmt::print("\n=== Deserialization Throughput: BufferPool ON vs OFF ===\n\n");
  folly::runBenchmarks();
  return 0;
}
