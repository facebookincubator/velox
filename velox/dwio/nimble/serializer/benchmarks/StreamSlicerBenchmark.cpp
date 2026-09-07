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
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "fb_velox/common/Profiler.h"
#include "velox/common/compression/Compression.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/Projector.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/serializer/StreamSlicer.h"
#include "velox/dwio/nimble/velox/OrderedRanges.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble::serde {
namespace {

constexpr int32_t kDefaultNumRows{5};
constexpr int32_t kNumFeatures{100};
constexpr int32_t kNumProjectedFeatures{20};
constexpr int32_t kFirstProjectedFeature{40};
constexpr int32_t kDefaultArrayElementsPerFeature{1'000};
constexpr int32_t kDefaultSliceOffset{1};
constexpr int32_t kDefaultSliceLength{2};

DEFINE_string(
    stream_slicer_profile_mode,
    "",
    "Run a ScopedProfiler loop for one path and exit. Supported values: "
    "project_slice, presto_zstd, presto_none.");
DEFINE_int32(
    stream_slicer_profile_seconds,
    60,
    "Duration for --stream_slicer_profile_mode tight loop.");
DEFINE_string(
    stream_slicer_encoding,
    "trivial",
    "Encoding used for Nimble streams. Supported values: auto, trivial, rle, "
    "delta, constant, mainly_constant, varint, fixed_bit_width, "
    "block_bit_packing, pfor, simd_for_bitpack, huffman, alp.");
DEFINE_int32(
    stream_slicer_num_rows,
    kDefaultNumRows,
    "Number of rows in the benchmark input batch.");
DEFINE_int32(
    stream_slicer_array_elements_per_feature,
    kDefaultArrayElementsPerFeature,
    "Number of array elements generated for each feature value.");
DEFINE_int32(
    stream_slicer_slice_offset,
    kDefaultSliceOffset,
    "First row to keep in the sliced output.");
DEFINE_int32(
    stream_slicer_slice_length,
    kDefaultSliceLength,
    "Number of rows to keep in the sliced output.");
DEFINE_bool(
    stream_slicer_size_only,
    false,
    "Print projected output sizes and exit without running timing benchmarks.");
using Subfield = velox::common::Subfield;

struct BenchmarkState {
  std::shared_ptr<velox::memory::MemoryPool> pool;
  std::string serialized;
  std::shared_ptr<const Type> schema;
  std::shared_ptr<const Type> projectedSchema;
  std::vector<Subfield> selectedSubfields;
  std::unique_ptr<Projector> projector;
  std::unique_ptr<Deserializer> deserializer;
  std::string encodingName;
};

velox::vector_size_t numRows() {
  if (FLAGS_stream_slicer_num_rows <= 0) {
    throw std::invalid_argument{"stream_slicer_num_rows must be positive."};
  }
  return static_cast<velox::vector_size_t>(FLAGS_stream_slicer_num_rows);
}

velox::vector_size_t arrayElementsPerFeature() {
  if (FLAGS_stream_slicer_array_elements_per_feature <= 0) {
    throw std::invalid_argument{
        "stream_slicer_array_elements_per_feature must be positive."};
  }
  return static_cast<velox::vector_size_t>(
      FLAGS_stream_slicer_array_elements_per_feature);
}

uint32_t sliceOffset() {
  if (FLAGS_stream_slicer_slice_offset < 0) {
    throw std::invalid_argument{
        "stream_slicer_slice_offset must be non-negative."};
  }
  return static_cast<uint32_t>(FLAGS_stream_slicer_slice_offset);
}

uint32_t sliceLength() {
  if (FLAGS_stream_slicer_slice_length <= 0) {
    throw std::invalid_argument{"stream_slicer_slice_length must be positive."};
  }
  return static_cast<uint32_t>(FLAGS_stream_slicer_slice_length);
}

void validateSliceRange() {
  const auto rows = static_cast<uint32_t>(numRows());
  const auto offset = sliceOffset();
  const auto length = sliceLength();
  if (offset > rows || length > rows - offset) {
    throw std::invalid_argument{fmt::format(
        "Invalid slice range: offset={}, length={}, rows={}.",
        offset,
        length,
        rows)};
  }
}

std::string iobufToString(const folly::IOBuf& buffer) {
  std::string output;
  output.reserve(buffer.computeChainDataLength());
  for (const auto range : buffer) {
    output.append(reinterpret_cast<const char*>(range.data()), range.size());
  }
  return output;
}

velox::FlatVectorPtr<int32_t> makeFeatureKeys(velox::memory::MemoryPool* pool) {
  const auto rows = numRows();
  const auto numMapEntries = rows * kNumFeatures;
  auto keys = velox::BaseVector::create<velox::FlatVector<int32_t>>(
      velox::INTEGER(), numMapEntries, pool);
  for (velox::vector_size_t row = 0; row < rows; ++row) {
    for (int32_t feature = 0; feature < kNumFeatures; ++feature) {
      keys->set(row * kNumFeatures + feature, feature);
    }
  }
  return keys;
}

bool isAlpBenchmark() {
  return FLAGS_stream_slicer_encoding == "alp";
}

std::string_view featureValueTypeName() {
  return isAlpBenchmark() ? "DOUBLE" : "BIGINT";
}

velox::TypePtr makeFeatureValueType() {
  if (isAlpBenchmark()) {
    return velox::DOUBLE();
  }
  return velox::BIGINT();
}

velox::TypePtr makeFeatureType() {
  return velox::MAP(velox::INTEGER(), velox::ARRAY(makeFeatureValueType()));
}

int64_t integerFeatureValue(velox::vector_size_t index) {
  const auto& encoding = FLAGS_stream_slicer_encoding;
  if (encoding == "rle") {
    return (index / 128) % 8;
  }
  if (encoding == "delta") {
    return index;
  }
  if (encoding == "constant") {
    return 7;
  }
  if (encoding == "mainly_constant") {
    return index % 16 == 0 ? index % 256 : 7;
  }
  if (encoding == "huffman") {
    return index % 8 == 0 ? index % 64 : 3;
  }
  if (encoding == "varint" || encoding == "fixed_bit_width" ||
      encoding == "block_bit_packing" || encoding == "pfor" ||
      encoding == "simd_for_bitpack") {
    return index % 256;
  }
  return static_cast<int64_t>(
      static_cast<uint64_t>(index) * 11400714819323198485ull);
}

velox::VectorPtr makeFeatureElements(velox::memory::MemoryPool* pool) {
  const auto numMapEntries = numRows() * kNumFeatures;
  const auto numElements = numMapEntries * arrayElementsPerFeature();

  if (isAlpBenchmark()) {
    auto elements = velox::BaseVector::create<velox::FlatVector<double>>(
        velox::DOUBLE(), numElements, pool);
    for (velox::vector_size_t i = 0; i < numElements; ++i) {
      elements->set(i, 1000.0 + static_cast<double>(i % 10'000) * 0.125);
    }
    return elements;
  }

  auto elements = velox::BaseVector::create<velox::FlatVector<int64_t>>(
      velox::BIGINT(), numElements, pool);
  for (velox::vector_size_t i = 0; i < numElements; ++i) {
    elements->set(i, integerFeatureValue(i));
  }
  return elements;
}

velox::ArrayVectorPtr makeFeatureArrays(velox::memory::MemoryPool* pool) {
  const auto numMapEntries = numRows() * kNumFeatures;
  const auto elements = makeFeatureElements(pool);
  auto offsets = velox::allocateOffsets(numMapEntries, pool);
  auto sizes = velox::allocateSizes(numMapEntries, pool);
  auto* rawOffsets = offsets->asMutable<velox::vector_size_t>();
  auto* rawSizes = sizes->asMutable<velox::vector_size_t>();
  const auto arrayElements = arrayElementsPerFeature();
  for (velox::vector_size_t entry = 0; entry < numMapEntries; ++entry) {
    rawOffsets[entry] = entry * arrayElements;
    rawSizes[entry] = arrayElements;
  }

  return std::make_shared<velox::ArrayVector>(
      pool,
      velox::ARRAY(elements->type()),
      /*nulls=*/nullptr,
      numMapEntries,
      offsets,
      sizes,
      elements);
}

velox::MapVectorPtr makeFeatures(velox::memory::MemoryPool* pool) {
  const auto rows = numRows();
  auto offsets = velox::allocateOffsets(rows, pool);
  auto sizes = velox::allocateSizes(rows, pool);
  auto* rawOffsets = offsets->asMutable<velox::vector_size_t>();
  auto* rawSizes = sizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t row = 0; row < rows; ++row) {
    rawOffsets[row] = row * kNumFeatures;
    rawSizes[row] = kNumFeatures;
  }

  return std::make_shared<velox::MapVector>(
      pool,
      makeFeatureType(),
      /*nulls=*/nullptr,
      rows,
      offsets,
      sizes,
      makeFeatureKeys(pool),
      makeFeatureArrays(pool));
}

std::vector<Subfield> makeSelectedSubfields() {
  std::vector<Subfield> subfields;
  subfields.reserve(kNumProjectedFeatures);
  for (int32_t feature = kFirstProjectedFeature;
       feature < kFirstProjectedFeature + kNumProjectedFeatures;
       ++feature) {
    subfields.emplace_back(fmt::format("features[{}]", feature));
  }
  return subfields;
}

std::optional<EncodingType> parseEncodingType() {
  const auto& name = FLAGS_stream_slicer_encoding;
  if (name == "auto") {
    return std::nullopt;
  }
  if (name == "trivial") {
    return EncodingType::Trivial;
  }
  if (name == "rle") {
    return EncodingType::RLE;
  }
  if (name == "delta") {
    return EncodingType::Delta;
  }
  if (name == "constant") {
    return EncodingType::Constant;
  }
  if (name == "mainly_constant") {
    return EncodingType::MainlyConstant;
  }
  if (name == "varint") {
    return EncodingType::Varint;
  }
  if (name == "fixed_bit_width") {
    return EncodingType::FixedBitWidth;
  }
  if (name == "block_bit_packing") {
    return EncodingType::BlockBitPacking;
  }
  if (name == "pfor") {
    return EncodingType::PFOR;
  }
  if (name == "simd_for_bitpack") {
    return EncodingType::SimdForBitpack;
  }
  if (name == "huffman") {
    return EncodingType::Huffman;
  }
  if (name == "alp") {
    return EncodingType::ALP;
  }
  throw std::invalid_argument{
      fmt::format("Unknown stream_slicer_encoding: {}.", name)};
}

EncodingSelectionPolicyCreator makeEncodingSelectionPolicyCreator(
    std::optional<EncodingType> encodingType) {
  if (!encodingType.has_value()) {
    return defaultEncodingSelectionPolicyCreator();
  }

  auto factory = std::make_shared<ManualEncodingSelectionPolicyFactory>(
      std::vector<std::pair<EncodingType, float>>{{encodingType.value(), 1.0}},
      /*compressionOptions=*/std::nullopt);
  return
      [factory](DataType dataType) { return factory->createPolicy(dataType); };
}

BenchmarkState prepareBenchmark() {
  validateSliceRange();
  static std::atomic<uint64_t> nextPoolId{0};
  auto pool = velox::memory::memoryManager()->addLeafPool(
      fmt::format(
          "prepare_stream_slicer_flatmap_arrays_{}", nextPoolId.fetch_add(1)));
  const auto type = velox::ROW({{"features", makeFeatureType()}});
  auto input = std::make_shared<velox::RowVector>(
      pool.get(),
      type,
      /*nulls=*/nullptr,
      numRows(),
      std::vector<velox::VectorPtr>{makeFeatures(pool.get())});

  const auto encodingType = parseEncodingType();
  Serializer serializer{
      SerializerOptions{
          .version = SerializationVersion::kSerialization,
          .flatMapColumns = {{"features", {}}},
          .encodingSelectionPolicyCreator =
              makeEncodingSelectionPolicyCreator(encodingType),
      },
      type,
      pool.get()};
  const auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  auto selectedSubfields = makeSelectedSubfields();
  std::string serializedString{serialized};

  BenchmarkState state{
      .pool = std::move(pool),
      .serialized = std::move(serializedString),
      .schema = std::move(schema),
      .selectedSubfields = std::move(selectedSubfields),
      .encodingName =
          encodingType.has_value() ? toString(encodingType.value()) : "auto",
  };
  state.projector = std::make_unique<Projector>(
      state.schema,
      state.selectedSubfields,
      state.pool.get(),
      Projector::Options{});
  state.projectedSchema = state.projector->projectedSchema();
  state.deserializer = std::make_unique<Deserializer>(
      state.schema,
      state.selectedSubfields,
      state.pool.get(),
      DeserializerOptions{.hasHeader = true});
  return state;
}

std::string projectAndStreamSlice(
    const BenchmarkState& state,
    StreamSlicer& slicer) {
  const auto projected =
      iobufToString(state.projector->project(state.serialized));
  return iobufToString(slicer.slice(projected, sliceOffset(), sliceLength()));
}

velox::RowVectorPtr deserializeProjectedRows(const BenchmarkState& state) {
  velox::VectorPtr output;
  state.deserializer->deserialize(state.serialized, output);
  return std::dynamic_pointer_cast<velox::RowVector>(output);
}

std::string serializePresto(
    const velox::RowVectorPtr& vector,
    velox::BatchVectorSerializer& serializer,
    velox::vector_size_t offset,
    velox::vector_size_t length) {
  const velox::IndexRange range{
      offset,
      length,
  };
  std::ostringstream output;
  velox::OStreamOutputStream outputStream{&output};
  serializer.serialize(vector, folly::Range(&range, 1), &outputStream);
  return output.str();
}

std::string deserializeAndSerializePresto(
    const BenchmarkState& state,
    velox::BatchVectorSerializer& serializer) {
  return serializePresto(
      deserializeProjectedRows(state),
      serializer,
      static_cast<velox::vector_size_t>(sliceOffset()),
      static_cast<velox::vector_size_t>(sliceLength()));
}

std::unique_ptr<velox::BatchVectorSerializer> makePrestoSerializer(
    velox::memory::MemoryPool* pool,
    velox::common::CompressionKind compressionKind) {
  velox::serializer::presto::PrestoVectorSerde serde;
  velox::serializer::presto::PrestoVectorSerde::PrestoOptions options{
      /*_useLosslessTimestamp=*/false,
      compressionKind,
      /*_minCompressionRatio=*/1.0};
  return serde.createBatchSerializer(pool, &options);
}

void runProjectAndStreamSlice(const BenchmarkState& state, uint32_t iters) {
  auto pool =
      velox::memory::memoryManager()->addLeafPool("project_stream_slice_bench");
  StreamSlicer slicer{
      state.projectedSchema, pool.get(), StreamSlicer::Options{}};
  while (iters-- > 0) {
    auto output = projectAndStreamSlice(state, slicer);
    folly::doNotOptimizeAway(output.size());
  }
}

void runDeserializeAndSerializePresto(
    const BenchmarkState& state,
    uint32_t iters,
    velox::common::CompressionKind compressionKind) {
  auto pool = velox::memory::memoryManager()->addLeafPool(
      "deserialize_presto_zstd_bench");
  auto serializer = makePrestoSerializer(pool.get(), compressionKind);
  while (iters-- > 0) {
    auto output = deserializeAndSerializePresto(state, *serializer);
    folly::doNotOptimizeAway(output.size());
  }
}

void printSizeReport(const BenchmarkState& state) {
  auto pool =
      velox::memory::memoryManager()->addLeafPool("stream_slicer_size_report");
  StreamSlicer slicer{
      state.projectedSchema, pool.get(), StreamSlicer::Options{}};
  auto zstdSerializer =
      makePrestoSerializer(pool.get(), velox::common::CompressionKind_ZSTD);
  auto noneSerializer =
      makePrestoSerializer(pool.get(), velox::common::CompressionKind_NONE);
  const auto nimbleOriginal =
      iobufToString(state.projector->project(state.serialized));
  const auto nimbleOutput = projectAndStreamSlice(state, slicer);
  const auto projectedRows = deserializeProjectedRows(state);
  const auto prestoZstdOriginal = serializePresto(
      projectedRows, *zstdSerializer, /*offset=*/0, projectedRows->size());
  const auto prestoNoneOriginal = serializePresto(
      projectedRows, *noneSerializer, /*offset=*/0, projectedRows->size());
  const auto prestoZstdOutput =
      deserializeAndSerializePresto(state, *zstdSerializer);
  const auto prestoNoneOutput =
      deserializeAndSerializePresto(state, *noneSerializer);
  const auto offset = sliceOffset();
  const auto length = sliceLength();
  const auto nimbleTarget =
      nimbleOriginal.size() * length / static_cast<uint64_t>(numRows());
  fmt::print(
      "\n=== Projected Size: 20/100 features, rows [{}, {}) ===\n"
      "  Nimble encoding:                 {}\n"
      "  Schema:                          ROW(features MAP(INTEGER, ARRAY({})))\n"
      "  Projected feature range:         features[{}]..features[{}]\n"
      "  Array elements per feature:      {}\n"
      "  Projector output:                {} bytes/full\n"
      "  Projector proportional target:   {} bytes/slice ({:.2f}x full)\n"
      "  Projector + StreamSlicer output: {} bytes/slice ({:.2f}x full)\n"
      "  Deserializer + Presto ZSTD:      {} bytes/full, {} bytes/slice ({:.2f}x full)\n"
      "  Deserializer + Presto none:      {} bytes/full, {} bytes/slice ({:.2f}x full)\n\n",
      offset,
      offset + length,
      state.encodingName,
      featureValueTypeName(),
      kFirstProjectedFeature,
      kFirstProjectedFeature + kNumProjectedFeatures - 1,
      arrayElementsPerFeature(),
      nimbleOriginal.size(),
      nimbleTarget,
      static_cast<double>(length) / numRows(),
      nimbleOutput.size(),
      static_cast<double>(nimbleOutput.size()) / nimbleOriginal.size(),
      prestoZstdOriginal.size(),
      prestoZstdOutput.size(),
      static_cast<double>(prestoZstdOutput.size()) / prestoZstdOriginal.size(),
      prestoNoneOriginal.size(),
      prestoNoneOutput.size(),
      static_cast<double>(prestoNoneOutput.size()) / prestoNoneOriginal.size());
}

void logProfilerResult(
    std::string_view mode,
    const facebook::fb_velox::common::ScopedProfiler::Result& result) {
  fmt::print(
      "{} CPU profile result: {}\n"
      "{} heap profile result: {}\n",
      mode,
      result.cpuScubaUrl.empty() ? "none" : result.cpuScubaUrl,
      mode,
      result.heapScubaUrl.empty() ? "none" : result.heapScubaUrl);
}

void runProfileMode(const BenchmarkState& state) {
  const auto mode = FLAGS_stream_slicer_profile_mode;
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds{FLAGS_stream_slicer_profile_seconds};
  auto pool = velox::memory::memoryManager()->addLeafPool(
      fmt::format("stream_slicer_profile_{}", mode));
  StreamSlicer slicer{
      state.projectedSchema, pool.get(), StreamSlicer::Options{}};
  auto zstdSerializer =
      makePrestoSerializer(pool.get(), velox::common::CompressionKind_ZSTD);
  auto noneSerializer =
      makePrestoSerializer(pool.get(), velox::common::CompressionKind_NONE);

  uint64_t iterations{0};
  uint64_t outputBytes{0};
  facebook::fb_velox::common::ScopedProfiler::Result profileResult;
  {
    facebook::fb_velox::common::ScopedProfiler profiler{&profileResult};
    while (std::chrono::steady_clock::now() < deadline) {
      std::string output;
      if (mode == "project_slice") {
        output = projectAndStreamSlice(state, slicer);
      } else if (mode == "presto_zstd") {
        output = deserializeAndSerializePresto(state, *zstdSerializer);
      } else if (mode == "presto_none") {
        output = deserializeAndSerializePresto(state, *noneSerializer);
      } else {
        throw std::invalid_argument{
            fmt::format("Unknown stream_slicer_profile_mode: {}.", mode)};
      }
      outputBytes += output.size();
      ++iterations;
      folly::doNotOptimizeAway(output);
    }
  }

  fmt::print(
      "\n=== Scoped Profile: {} ===\n"
      "  Encoding:      {}\n"
      "  Duration:      {}s\n"
      "  Iterations:    {}\n"
      "  Output bytes:  {}\n"
      "  Bytes/iter:    {:.2f}\n",
      mode,
      state.encodingName,
      FLAGS_stream_slicer_profile_seconds,
      iterations,
      outputBytes,
      iterations == 0 ? 0.0 : static_cast<double>(outputBytes) / iterations);
  logProfilerResult(mode, profileResult);
}

BENCHMARK(ProjectAndStreamSlice_FlatMapArray, iters) {
  static const auto state = prepareBenchmark();
  runProjectAndStreamSlice(state, iters);
}

BENCHMARK_RELATIVE(DeserializePrestoZstd_FlatMapArray, iters) {
  static const auto state = prepareBenchmark();
  runDeserializeAndSerializePresto(
      state, iters, velox::common::CompressionKind_ZSTD);
}

BENCHMARK_RELATIVE(DeserializePrestoNone_FlatMapArray, iters) {
  static const auto state = prepareBenchmark();
  runDeserializeAndSerializePresto(
      state, iters, velox::common::CompressionKind_NONE);
}

} // namespace
} // namespace facebook::nimble::serde

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  const auto state = facebook::nimble::serde::prepareBenchmark();
  facebook::nimble::serde::printSizeReport(state);
  if (facebook::nimble::serde::FLAGS_stream_slicer_size_only) {
    return 0;
  }
  if (!facebook::nimble::serde::FLAGS_stream_slicer_profile_mode.empty()) {
    facebook::nimble::serde::runProfileMode(state);
    return 0;
  }
  folly::runBenchmarks();
  return 0;
}
