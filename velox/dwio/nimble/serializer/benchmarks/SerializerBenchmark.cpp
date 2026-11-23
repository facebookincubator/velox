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
#include <map>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/OrderedRanges.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

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

// Datasets that the format comparison serializes.
enum class Dataset {
  // An id column and a map with kNumKeys keys per row, written as a flat map.
  kFlatMap,
  // An id column and kNumKeys BIGINT columns, which reads without flat map
  // reconstruction.
  kColumns,
};

// Returns 'flatMapBatch' with each key of its features map as a BIGINT
// column, so that both datasets hold the same values.
velox::RowVectorPtr makeColumnsBatch(
    const velox::RowVectorPtr& flatMapBatch,
    int32_t numKeys,
    velox::memory::MemoryPool* pool) {
  const auto numRows = flatMapBatch->size();

  // Every row holds every key, so key 'key' of row 'row' is at
  // row * numKeys + key.
  const auto* values = flatMapBatch->childAt("features")
                           ->as<velox::MapVector>()
                           ->mapValues()
                           ->asFlatVector<int64_t>();
  std::vector<std::string> names{"id"};
  std::vector<velox::VectorPtr> columns{flatMapBatch->childAt(0)};
  for (int32_t key = 0; key < numKeys; ++key) {
    auto column = std::make_shared<velox::FlatVector<int64_t>>(
        pool,
        velox::BIGINT(),
        /*nulls=*/nullptr,
        numRows,
        velox::AlignedBuffer::allocate<int64_t>(numRows, pool),
        /*stringBuffers=*/std::vector<velox::BufferPtr>{});
    for (velox::vector_size_t row = 0; row < numRows; ++row) {
      column->set(row, values->valueAt(row * numKeys + key));
    }
    names.push_back(fmt::format("c{}", key));
    columns.push_back(std::move(column));
  }
  return std::make_shared<velox::RowVector>(
      pool,
      velox::ROW(std::move(names), velox::BIGINT()),
      /*nulls=*/nullptr,
      numRows,
      std::move(columns));
}

// Returns the input rows for 'dataset' with 'dataPattern' values. Builds them
// on first use and then returns the same rows, so that every format measures
// the same random values.
const velox::RowVectorPtr& comparisonInput(Dataset dataset, int dataPattern) {
  static const auto pool =
      velox::memory::memoryManager()->addLeafPool("format_input");
  static std::map<std::pair<Dataset, int>, velox::RowVectorPtr> inputs;
  auto& input = inputs[{dataset, dataPattern}];
  if (input != nullptr) {
    return input;
  }
  if (dataset == Dataset::kColumns) {
    input = makeColumnsBatch(
        comparisonInput(Dataset::kFlatMap, dataPattern), kNumKeys, pool.get());
  } else {
    input = makeWideFlatMapBatch(kNumRows, kNumKeys, dataPattern, pool.get());
  }
  return input;
}

// Serialization formats that the format comparison measures.
enum class Format {
  // kSerialization: encoding selection picks each stream's encoding.
  kEncoded,
  // kBlob with its default Lz4 blob codec.
  kBlobLz4,
  // kBlob with a Zstd blob codec.
  kBlobZstd,
  // kBlob with Lz4 per stream and an uncompressed blob.
  kStreamLz4,
  // kBlob with no compression at all.
  kBlobUncompressed,
};

std::string_view formatName(Format format) {
  switch (format) {
    case Format::kEncoded:
      return "kSerialization";
    case Format::kBlobLz4:
      return "kBlob, blob Lz4";
    case Format::kBlobZstd:
      return "kBlob, blob Zstd";
    case Format::kStreamLz4:
      return "kBlob, streams Lz4";
    case Format::kBlobUncompressed:
      return "kBlob, uncompressed";
  }
  return "unknown";
}

SerializerOptions comparisonOptions(Format format, Dataset dataset) {
  SerializerOptions options;
  if (dataset == Dataset::kFlatMap) {
    options.flatMapColumns = {{"features", {}}};
  }
  if (format != Format::kEncoded) {
    options.version = SerializationVersion::kBlob;
  }
  switch (format) {
    case Format::kEncoded:
    case Format::kBlobLz4:
      break;
    case Format::kBlobZstd:
      options.blobCompressionOptions.compressionType = CompressionType::Zstd;
      break;
    case Format::kStreamLz4:
      options.compressionOptions =
          CompressionOptions{.compressionType = CompressionType::Lz4};
      options.blobCompressionOptions.compressionType =
          CompressionType::Uncompressed;
      break;
    case Format::kBlobUncompressed:
      options.blobCompressionOptions.compressionType =
          CompressionType::Uncompressed;
      break;
  }
  return options;
}

void runFormatSerialize(
    Dataset dataset,
    int dataPattern,
    Format format,
    uint32_t iters) {
  folly::BenchmarkSuspender suspender;
  const auto& input = comparisonInput(dataset, dataPattern);
  auto pool = velox::memory::memoryManager()->addLeafPool("format_serialize");
  Serializer serializer{
      comparisonOptions(format, dataset), input->type(), pool.get()};

  // The first call discovers the flat map keys and grows the buffers that
  // later calls reuse.
  serializer.serialize(input, OrderedRanges::of(0, input->size()));
  suspender.dismiss();

  while (iters--) {
    folly::doNotOptimizeAway(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
  }
  suspender.rehire();
}

void runFormatDeserialize(
    Dataset dataset,
    int dataPattern,
    Format format,
    uint32_t iters) {
  folly::BenchmarkSuspender suspender;
  const auto& input = comparisonInput(dataset, dataPattern);
  auto pool = velox::memory::memoryManager()->addLeafPool("format_deserialize");
  Serializer serializer{
      comparisonOptions(format, dataset), input->type(), pool.get()};
  const std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, input->size()))};
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool.get()};
  velox::VectorPtr output;

  // The first call allocates the buffers that later calls reuse.
  deserializer.deserialize(serialized, output);
  output.reset();
  suspender.dismiss();

  while (iters--) {
    deserializer.deserialize(serialized, output);
    output.reset();
  }
  suspender.rehire();
}

void printFormatSizes(const char* label, Dataset dataset, int dataPattern) {
  auto pool = velox::memory::memoryManager()->addLeafPool("format_sizes");
  const auto& input = comparisonInput(dataset, dataPattern);
  for (const auto format :
       {Format::kEncoded,
        Format::kBlobLz4,
        Format::kBlobZstd,
        Format::kStreamLz4,
        Format::kBlobUncompressed}) {
    Serializer serializer{
        comparisonOptions(format, dataset), input->type(), pool.get()};
    const auto serialized =
        serializer.serialize(input, OrderedRanges::of(0, input->size()));
    fmt::print(
        "  {:>18} | {:>30} | {:>8.0f}KB\n",
        label,
        formatName(format),
        static_cast<double>(serialized.size()) / 1024.0);
  }
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

// === Serialize and deserialize CPU: kSerialization vs kBlob ===

#define FORMAT_BENCH(Name, InputDataset, Pattern)                           \
  BENCHMARK(Serialize_##Name##_Encoded, iters) {                            \
    runFormatSerialize(InputDataset, Pattern, Format::kEncoded, iters);     \
  }                                                                         \
  BENCHMARK_RELATIVE(Serialize_##Name##_BlobLz4, iters) {                   \
    runFormatSerialize(InputDataset, Pattern, Format::kBlobLz4, iters);     \
  }                                                                         \
  BENCHMARK_RELATIVE(Serialize_##Name##_BlobZstd, iters) {                  \
    runFormatSerialize(InputDataset, Pattern, Format::kBlobZstd, iters);    \
  }                                                                         \
  BENCHMARK_RELATIVE(Serialize_##Name##_StreamLz4, iters) {                 \
    runFormatSerialize(InputDataset, Pattern, Format::kStreamLz4, iters);   \
  }                                                                         \
  BENCHMARK_RELATIVE(Serialize_##Name##_BlobUncompressed, iters) {          \
    runFormatSerialize(                                                     \
        InputDataset, Pattern, Format::kBlobUncompressed, iters);           \
  }                                                                         \
  BENCHMARK_DRAW_LINE();                                                    \
  BENCHMARK(Deserialize_##Name##_Encoded, iters) {                          \
    runFormatDeserialize(InputDataset, Pattern, Format::kEncoded, iters);   \
  }                                                                         \
  BENCHMARK_RELATIVE(Deserialize_##Name##_BlobLz4, iters) {                 \
    runFormatDeserialize(InputDataset, Pattern, Format::kBlobLz4, iters);   \
  }                                                                         \
  BENCHMARK_RELATIVE(Deserialize_##Name##_BlobZstd, iters) {                \
    runFormatDeserialize(InputDataset, Pattern, Format::kBlobZstd, iters);  \
  }                                                                         \
  BENCHMARK_RELATIVE(Deserialize_##Name##_StreamLz4, iters) {               \
    runFormatDeserialize(InputDataset, Pattern, Format::kStreamLz4, iters); \
  }                                                                         \
  BENCHMARK_RELATIVE(Deserialize_##Name##_BlobUncompressed, iters) {        \
    runFormatDeserialize(                                                   \
        InputDataset, Pattern, Format::kBlobUncompressed, iters);           \
  }                                                                         \
  BENCHMARK_DRAW_LINE();

FORMAT_BENCH(FlatMapRandom, Dataset::kFlatMap, kRandom)
FORMAT_BENCH(FlatMapLowCard, Dataset::kFlatMap, kLowCardinality)
FORMAT_BENCH(FlatMapRunLength, Dataset::kFlatMap, kRunLength)
FORMAT_BENCH(ColumnsRandom, Dataset::kColumns, kRandom)
FORMAT_BENCH(ColumnsLowCard, Dataset::kColumns, kLowCardinality)
FORMAT_BENCH(ColumnsRunLength, Dataset::kColumns, kRunLength)

#undef FORMAT_BENCH

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

  fmt::print(
      "\n=== Serialized size by format ({} rows x {} keys) ===\n\n",
      kNumRows,
      kNumKeys);
  printFormatSizes("FlatMap Random", Dataset::kFlatMap, kRandom);
  printFormatSizes("FlatMap LowCard", Dataset::kFlatMap, kLowCardinality);
  printFormatSizes("FlatMap RunLength", Dataset::kFlatMap, kRunLength);
  printFormatSizes("Columns Random", Dataset::kColumns, kRandom);
  printFormatSizes("Columns LowCard", Dataset::kColumns, kLowCardinality);
  printFormatSizes("Columns RunLength", Dataset::kColumns, kRunLength);

  fmt::print(
      "\n=== Deserialization Throughput: BufferPool ON vs OFF, then "
      "Serde CPU by format ===\n\n");
  folly::runBenchmarks();
  return 0;
}
