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

/// Fuzzer tests for all Nimble encodings across supported data types.
///
/// Configuration via CLI flags:
///   --fuzzer_iterations=N     Number of iterations per test (default: 50)
///   --fuzzer_max_rows=N       Maximum rows per iteration (default: 5000)
///   --fuzzer_seed=N           Fixed seed, 0=random (default: 42)
///   --fuzzer_compression      Enable compression testing (default: true)

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
#include "velox/dwio/nimble/encodings/SubIntSplitConfig.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#endif
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"

DEFINE_uint32(fuzzer_iterations, 50, "Number of fuzzer iterations per test");
DEFINE_uint32(fuzzer_max_rows, 5000, "Maximum rows per fuzzer iteration");
DEFINE_uint32(fuzzer_seed, 42, "Fuzzer seed (0 = random each run)");
DEFINE_bool(fuzzer_compression, true, "Test with compression enabled");

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::test;

// Trivial: all types
using TrivialTypes = ::testing::Types<
    TrivialEncoding<bool>,
    TrivialEncoding<int8_t>,
    TrivialEncoding<uint8_t>,
    TrivialEncoding<int16_t>,
    TrivialEncoding<uint16_t>,
    TrivialEncoding<int32_t>,
    TrivialEncoding<uint32_t>,
    TrivialEncoding<int64_t>,
    TrivialEncoding<uint64_t>,
    TrivialEncoding<float>,
    TrivialEncoding<double>,
    TrivialEncoding<std::string_view>>;

template <typename E>
class TrivialFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(TrivialFuzzerTest, TrivialTypes);

TYPED_TEST(TrivialFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// RLE: all types
using RleTypes = ::testing::Types<
    RLEEncoding<bool>,
    RLEEncoding<int8_t>,
    RLEEncoding<uint8_t>,
    RLEEncoding<int16_t>,
    RLEEncoding<uint16_t>,
    RLEEncoding<int32_t>,
    RLEEncoding<uint32_t>,
    RLEEncoding<int64_t>,
    RLEEncoding<uint64_t>,
    RLEEncoding<float>,
    RLEEncoding<double>,
    RLEEncoding<std::string_view>>;

template <typename E>
class RleFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(RleFuzzerTest, RleTypes);

TYPED_TEST(RleFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// Dictionary: all types except bool
using DictionaryTypes = ::testing::Types<
    DictionaryEncoding<int8_t>,
    DictionaryEncoding<uint8_t>,
    DictionaryEncoding<int16_t>,
    DictionaryEncoding<uint16_t>,
    DictionaryEncoding<int32_t>,
    DictionaryEncoding<uint32_t>,
    DictionaryEncoding<int64_t>,
    DictionaryEncoding<uint64_t>,
    DictionaryEncoding<float>,
    DictionaryEncoding<double>,
    DictionaryEncoding<std::string_view>>;

template <typename E>
class DictionaryFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(DictionaryFuzzerTest, DictionaryTypes);

TYPED_TEST(DictionaryFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// FixedBitWidth: all numeric types
using FixedBitWidthTypes = ::testing::Types<
    FixedBitWidthEncoding<int8_t>,
    FixedBitWidthEncoding<uint8_t>,
    FixedBitWidthEncoding<int16_t>,
    FixedBitWidthEncoding<uint16_t>,
    FixedBitWidthEncoding<int32_t>,
    FixedBitWidthEncoding<uint32_t>,
    FixedBitWidthEncoding<int64_t>,
    FixedBitWidthEncoding<uint64_t>,
    FixedBitWidthEncoding<float>,
    FixedBitWidthEncoding<double>>;

template <typename E>
class FixedBitWidthFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(FixedBitWidthFuzzerTest, FixedBitWidthTypes);

TYPED_TEST(FixedBitWidthFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// Delta: integer types
using DeltaTypes = ::testing::Types<
    DeltaEncoding<int8_t>,
    DeltaEncoding<uint8_t>,
    DeltaEncoding<int16_t>,
    DeltaEncoding<uint16_t>,
    DeltaEncoding<int32_t>,
    DeltaEncoding<uint32_t>,
    DeltaEncoding<int64_t>,
    DeltaEncoding<uint64_t>>;

template <typename E>
class DeltaFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(DeltaFuzzerTest, DeltaTypes);

TYPED_TEST(DeltaFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// DeltaBlock: integer types
using DeltaBlockTypes = ::testing::Types<
    DeltaBlockEncoding<int8_t>,
    DeltaBlockEncoding<uint8_t>,
    DeltaBlockEncoding<int16_t>,
    DeltaBlockEncoding<uint16_t>,
    DeltaBlockEncoding<int32_t>,
    DeltaBlockEncoding<uint32_t>,
    DeltaBlockEncoding<int64_t>,
    DeltaBlockEncoding<uint64_t>>;

template <typename E>
class DeltaBlockFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(DeltaBlockFuzzerTest, DeltaBlockTypes);

TYPED_TEST(DeltaBlockFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// SimdForBitpack: unsigned integer types
using SimdForBitpackTypes = ::testing::Types<
    SimdForBitpackEncoding<uint8_t>,
    SimdForBitpackEncoding<uint16_t>,
    SimdForBitpackEncoding<uint32_t>,
    SimdForBitpackEncoding<uint64_t>>;

template <typename E>
class SimdForBitpackFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(SimdForBitpackFuzzerTest, SimdForBitpackTypes);

TYPED_TEST(SimdForBitpackFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// Constant: all types
using ConstantTypes = ::testing::Types<
    ConstantEncoding<bool>,
    ConstantEncoding<int8_t>,
    ConstantEncoding<uint8_t>,
    ConstantEncoding<int16_t>,
    ConstantEncoding<uint16_t>,
    ConstantEncoding<int32_t>,
    ConstantEncoding<uint32_t>,
    ConstantEncoding<int64_t>,
    ConstantEncoding<uint64_t>,
    ConstantEncoding<float>,
    ConstantEncoding<double>,
    ConstantEncoding<std::string_view>>;

template <typename E>
class ConstantFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(ConstantFuzzerTest, ConstantTypes);

TYPED_TEST(ConstantFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// SparseBool: bool only
TEST(SparseBoolFuzzerTest, correctness) {
  EncodingFuzzer<SparseBoolEncoding> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// Varint: 4+ byte types
using VarintTypes = ::testing::Types<
    VarintEncoding<int32_t>,
    VarintEncoding<uint32_t>,
    VarintEncoding<int64_t>,
    VarintEncoding<uint64_t>,
    VarintEncoding<float>,
    VarintEncoding<double>>;

template <typename E>
class VarintFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(VarintFuzzerTest, VarintTypes);

TYPED_TEST(VarintFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// MainlyConstant: all types except bool
using MainlyConstantTypes = ::testing::Types<
    MainlyConstantEncoding<int8_t>,
    MainlyConstantEncoding<uint8_t>,
    MainlyConstantEncoding<int16_t>,
    MainlyConstantEncoding<uint16_t>,
    MainlyConstantEncoding<int32_t>,
    MainlyConstantEncoding<uint32_t>,
    MainlyConstantEncoding<int64_t>,
    MainlyConstantEncoding<uint64_t>,
    MainlyConstantEncoding<float>,
    MainlyConstantEncoding<double>,
    MainlyConstantEncoding<std::string_view>>;

template <typename E>
class MainlyConstantFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(MainlyConstantFuzzerTest, MainlyConstantTypes);

TYPED_TEST(MainlyConstantFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// BlockBitPacking: all numeric types
using BlockBitPackingTypes = ::testing::Types<
    BlockBitPackingEncoding<int8_t>,
    BlockBitPackingEncoding<uint8_t>,
    BlockBitPackingEncoding<int16_t>,
    BlockBitPackingEncoding<uint16_t>,
    BlockBitPackingEncoding<int32_t>,
    BlockBitPackingEncoding<uint32_t>,
    BlockBitPackingEncoding<int64_t>,
    BlockBitPackingEncoding<uint64_t>,
    BlockBitPackingEncoding<float>,
    BlockBitPackingEncoding<double>>;

template <typename E>
class BlockBitPackingFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(BlockBitPackingFuzzerTest, BlockBitPackingTypes);

TYPED_TEST(BlockBitPackingFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// ALP: float and double only
using ALPTypes = ::testing::Types<ALPEncoding<float>, ALPEncoding<double>>;

template <typename E>
class ALPFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(ALPFuzzerTest, ALPTypes);

TYPED_TEST(ALPFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression);
  fuzzer.run();
}

// Huffman: all integral types.
using HuffmanTypes = ::testing::Types<
    HuffmanEncoding<int8_t>,
    HuffmanEncoding<uint8_t>,
    HuffmanEncoding<int16_t>,
    HuffmanEncoding<uint16_t>,
    HuffmanEncoding<int32_t>,
    HuffmanEncoding<uint32_t>,
    HuffmanEncoding<int64_t>,
    HuffmanEncoding<uint64_t>>;

template <typename E>
class HuffmanFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(HuffmanFuzzerTest, HuffmanTypes);

TYPED_TEST(HuffmanFuzzerTest, correctness) {
  EncodingFuzzer<TypeParam> fuzzer(
      FLAGS_fuzzer_iterations,
      FLAGS_fuzzer_max_rows,
      FLAGS_fuzzer_seed,
      FLAGS_fuzzer_compression,
      {},
      /*minDistinctValues=*/2,
      /*maxDistinctValues=*/HuffmanEncoding<int8_t>::kMaxSymbols);
  fuzzer.run();
}

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
// SubIntSplit: 32- and 64-bit numeric types (no bool, no string_view)
using SubIntSplitTypes = ::testing::Types<
    SubIntSplitEncoding<int32_t>,
    SubIntSplitEncoding<uint32_t>,
    SubIntSplitEncoding<int64_t>,
    SubIntSplitEncoding<uint64_t>,
    SubIntSplitEncoding<float>,
    SubIntSplitEncoding<double>>;

template <typename E>
class SubIntSplitFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(SubIntSplitFuzzerTest, SubIntSplitTypes);

TYPED_TEST(SubIntSplitFuzzerTest, correctness) {
  // SubIntSplit's recompute encode runs an expensive DP split selector on every
  // dataset, so keep the per-type budget small -- the dataset generators
  // already cover all data patterns on each iteration. Compression roughly
  // doubles encode cost (and Trivial-section compression is already covered by
  // the other fuzzers), so it is left off here to keep the run within the test
  // time limit. realNestedSelection exercises the diverse per-section encoders;
  // largeInputRows adds a chunk-crossing large-input pass.
  const uint32_t iterations = std::min<uint32_t>(FLAGS_fuzzer_iterations, 5);
  const uint32_t maxRows = std::min<uint32_t>(FLAGS_fuzzer_max_rows, 300);
  EncodingFuzzer<TypeParam> fuzzer(
      iterations,
      maxRows,
      FLAGS_fuzzer_seed,
      /*testCompression=*/false,
      /*options=*/{},
      /*minDistinctValues=*/1,
      /*maxDistinctValues=*/std::numeric_limits<uint32_t>::max(),
      /*largeInputRows=*/65537u,
      /*realNestedSelection=*/true);
  fuzzer.run();
}

namespace {

template <typename T>
std::span<const T> toSpan(const Vector<T>& values) {
  return {values.data(), values.size()};
}

EncodingSelectionPolicyCreator makeLeafPolicyCreator() {
  return [](DataType type) -> std::unique_ptr<EncodingSelectionPolicyBase> {
    auto readFactors =
        ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
    readFactors.erase(
        std::remove_if(
            readFactors.begin(),
            readFactors.end(),
            [](const auto& factor) {
              return factor.first == EncodingType::SubIntSplit;
            }),
        readFactors.end());
    ManualEncodingSelectionPolicyFactory factory{
        std::move(readFactors), std::nullopt};
    return factory.createPolicy(type);
  };
}

// Builds an EncodingSelection from the policy and calls SubIntSplitEncoding
// directly, mirroring EncodingFactory::encode (which has no SubIntSplit case).
template <typename T>
std::string_view encodeWithPolicy(
    std::unique_ptr<EncodingSelectionPolicy<T>> policy,
    std::span<const T> values,
    Buffer& buffer) {
  using physicalType = typename TypeTraits<T>::physicalType;
  const std::span<const physicalType> physicalValues(
      reinterpret_cast<const physicalType*>(values.data()), values.size());
  auto statistics = Statistics<physicalType>::create(physicalValues);
  auto selectionResult =
      policy->select(physicalValues, statistics, /*options=*/{});
  EncodingSelection<physicalType> selection{
      std::move(selectionResult), std::move(statistics), std::move(policy)};
  return SubIntSplitEncoding<T>::encode(
      selection, physicalValues, buffer, /*options=*/{});
}

template <typename T>
std::unique_ptr<Encoding> decodeSubIntSplit(
    std::string_view encoded,
    velox::memory::MemoryPool& pool) {
  return std::make_unique<SubIntSplitEncoding<T>>(
      pool, encoded, [](uint32_t) { return nullptr; });
}

// Bit-exact comparison (float/double checked on their bit pattern, NaN-safe).
template <typename T>
void expectBitwiseEqual(
    std::span<const T> expected,
    std::span<const T> actual,
    std::string_view context) {
  ASSERT_EQ(expected.size(), actual.size()) << context;
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(
        EncodingPhysicalType<T>::asEncodingPhysicalType(expected[i]),
        EncodingPhysicalType<T>::asEncodingPhysicalType(actual[i]))
        << context << " @ row " << i;
  }
}

// Snowflake field widths (matches makeSnowflakeData): 12/10/41 for 64-bit,
// 6/5/20 for 32-bit.
struct SnowflakeLayout {
  int sequenceBits;
  int workerBits;
  int timestampBits;
};

template <typename T>
constexpr SnowflakeLayout snowflakeLayout() {
  if constexpr (sizeof(typename TypeTraits<T>::physicalType) == 8) {
    return {.sequenceBits = 12, .workerBits = 10, .timestampBits = 41};
  } else {
    return {.sequenceBits = 6, .workerBits = 5, .timestampBits = 20};
  }
}

// A random contiguous partition of [0, kBits) into 1..6 sections -- a valid
// preserve-mode boundary set (covers all bits, no gaps/overlaps).
template <typename T>
std::vector<detail::subintsplit::SegmentPlan> makeRandomSegments(
    std::mt19937& rng) {
  constexpr int kBits =
      static_cast<int>(sizeof(typename TypeTraits<T>::physicalType) * 8);
  std::uniform_int_distribution<int> internalCutCount(
      0, std::min(5, kBits - 1));
  const int cuts = internalCutCount(rng);
  std::set<int> boundaries;
  std::uniform_int_distribution<int> cutPos(1, kBits - 1);
  while (static_cast<int>(boundaries.size()) < cuts) {
    boundaries.insert(cutPos(rng));
  }
  std::vector<detail::subintsplit::SegmentPlan> segments;
  int start = 0;
  for (const int boundary : boundaries) {
    segments.push_back({.bitStart = start, .bitEnd = boundary - 1});
    start = boundary;
  }
  segments.push_back({.bitStart = start, .bitEnd = kBits - 1});
  return segments;
}

template <typename T>
EncodingLayout makePreserveLayout(
    const std::vector<detail::subintsplit::SegmentPlan>& segments) {
  std::vector<std::optional<const EncodingLayout>> children(segments.size());
  return EncodingLayout{
      EncodingType::SubIntSplit,
      EncodingLayout::Config{
          detail::subintsplit::makePreserveSplitConfig(segments)},
      CompressionType::Uncompressed,
      std::move(children)};
}

template <typename T>
std::string_view encodePreserve(
    const std::vector<detail::subintsplit::SegmentPlan>& segments,
    std::span<const T> values,
    Buffer& buffer) {
  const auto layout = makePreserveLayout<T>(segments);
  return encodeWithPolicy<T>(
      std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
          layout, std::nullopt, makeLeafPolicyCreator()),
      values,
      buffer);
}

// One dataset per shared data pattern (empties, for type-gated patterns, are
// dropped -- all SubIntSplit types are >= 4-byte numerics, so none are empty).
template <typename T>
std::vector<Vector<T>> makeSubIntSplitDatasets(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  std::vector<Vector<T>> datasets;
  {
    Vector<T> random(&pool);
    random.reserve(rowCount);
    nimble::testing::addRandomData<T>(rng, rowCount, &random, buffer);
    datasets.push_back(std::move(random));
  }
  datasets.push_back(makeSingleValueData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeLowCardinalityData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeSortedData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeBoundaryMixedData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeDominantValueData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeBitStructuredData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeSnowflakeData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeMixedRegimeData<T>(pool, rng, rowCount, buffer));
  std::erase_if(datasets, [](const Vector<T>& d) { return d.empty(); });
  return datasets;
}

uint32_t subIntSplitSeed() {
  return FLAGS_fuzzer_seed == 0 ? 0x1234u : FLAGS_fuzzer_seed;
}

} // namespace

// Preserve-mode encode with randomly generated valid boundary partitions must
// honor those boundaries and roundtrip bit-exactly.
TYPED_TEST(SubIntSplitFuzzerTest, preserveModeRandomBoundaries) {
  using T = typename TypeParam::cppDataType;
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  std::mt19937 rng(subIntSplitSeed());
  Buffer buffer{*pool};
  const auto datasets =
      makeSubIntSplitDatasets<T>(*pool, rng, /*rowCount=*/300, &buffer);
  for (size_t d = 0; d < datasets.size(); ++d) {
    const auto& values = datasets[d];
    const auto segments = makeRandomSegments<T>(rng);
    Buffer encodeBuffer{*pool};
    const auto encoded =
        encodePreserve<T>(segments, toSpan(values), encodeBuffer);
    auto encoding = decodeSubIntSplit<T>(encoded, *pool);
    Vector<T> decoded(pool.get(), values.size());
    encoding->materialize(static_cast<uint32_t>(values.size()), decoded.data());
    expectBitwiseEqual<T>(
        toSpan(values),
        toSpan(decoded),
        "preserve dataset " + std::to_string(d));
  }
}

// Preserve-mode split aligned exactly to the snowflake field boundaries
// (sequence | worker | timestamp) -- SubIntSplit's intended use case -- must
// roundtrip bit-exactly, including at a chunk-crossing size.
TYPED_TEST(SubIntSplitFuzzerTest, snowflakeFieldAlignedSplit) {
  using T = typename TypeParam::cppDataType;
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  const auto layout = snowflakeLayout<T>();
  constexpr int kBits =
      static_cast<int>(sizeof(typename TypeTraits<T>::physicalType) * 8);
  const std::vector<detail::subintsplit::SegmentPlan> segments = {
      {.bitStart = 0, .bitEnd = layout.sequenceBits - 1},
      {.bitStart = layout.sequenceBits,
       .bitEnd = layout.sequenceBits + layout.workerBits - 1},
      {.bitStart = layout.sequenceBits + layout.workerBits,
       .bitEnd = kBits - 1}};

  std::mt19937 rng(subIntSplitSeed());
  for (const uint32_t count : {300u, 4097u}) {
    Buffer buffer{*pool};
    const auto values = makeSnowflakeData<T>(*pool, rng, count, &buffer);
    Buffer encodeBuffer{*pool};
    const auto encoded =
        encodePreserve<T>(segments, toSpan(values), encodeBuffer);
    auto encoding = decodeSubIntSplit<T>(encoded, *pool);
    Vector<T> decoded(pool.get(), count);
    encoding->materialize(count, decoded.data());
    expectBitwiseEqual<T>(
        toSpan(values),
        toSpan(decoded),
        "snowflake field-aligned count=" + std::to_string(count));
  }
}

// SubIntSplit rejects empty input rather than producing an undecodable stream.
TYPED_TEST(SubIntSplitFuzzerTest, emptyInputRejected) {
  using T = typename TypeParam::cppDataType;
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Buffer buffer{*pool};
  const Vector<T> empty(pool.get());
  EXPECT_THROW(Encoder<TypeParam>::encode(buffer, empty), NimbleUserError);
}
#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
