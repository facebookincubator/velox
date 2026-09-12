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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <random>
#include <type_traits>

#include "folly/Random.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"

DEFINE_uint32(
    view_fuzzer_iterations,
    50,
    "Number of EncodingView fuzzer iterations per test");
DEFINE_uint32(
    view_fuzzer_max_rows,
    5000,
    "Maximum rows per EncodingView fuzzer iteration");
DEFINE_uint32(view_fuzzer_seed, 42, "EncodingView fuzzer seed (0 = random)");
DEFINE_uint32(
    view_fuzzer_duration_sec,
    0,
    "Wall-clock budget for the whole binary, across every typed test. Zero "
    "means the iteration counts are the only bound. Prefer setting this in "
    "automation: an iteration count cannot self-limit, so overshoot shows up "
    "as the job being killed rather than a short run.");

static ::testing::Environment* const kViewFuzzerBudget =
    ::testing::AddGlobalTestEnvironment(
        new facebook::nimble::test::FuzzerBudgetEnvironment(
            [] { return FLAGS_view_fuzzer_duration_sec; }));

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::test;

namespace {

template <typename T>
void expectEqual(const T& expected, const T& actual, uint32_t row) {
  if constexpr (isFloatingPointType<T>()) {
    using physicalType = typename TypeTraits<T>::physicalType;
    EXPECT_EQ(
        std::bit_cast<physicalType>(actual),
        std::bit_cast<physicalType>(expected))
        << "Mismatch at row " << row;
  } else {
    EXPECT_EQ(actual, expected) << "Mismatch at row " << row;
  }
}

template <typename T>
std::vector<Vector<T>> makeViewDatasets(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  std::vector<Vector<T>> datasets;

  Vector<T> random(&pool);
  random.reserve(rowCount);
  nimble::testing::addRandomData<T>(rng, rowCount, &random, buffer);
  datasets.push_back(std::move(random));

  datasets.push_back(makeSingleValueData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeMainlyConstantData<T>(pool, rng, rowCount, buffer));

  if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
    datasets.push_back(makeBoundaryMixedData<T>(pool, rng, rowCount, buffer));
    datasets.push_back(makeMonotonicData<T>(pool, rng, rowCount, buffer));
    datasets.push_back(makeBitStructuredData<T>(pool, rng, rowCount, buffer));
  }

  if constexpr (std::is_floating_point_v<T>) {
    datasets.push_back(
        makeFloatingPointDecimalData<T>(pool, rng, rowCount, buffer));
    datasets.push_back(
        makeFloatingPointSpecialData<T>(pool, rng, rowCount, buffer));
    datasets.push_back(
        makeFloatingPointMixedExceptionData<T>(pool, rng, rowCount, buffer));
  }

  for (uint32_t size : {1u, 2u, 3u}) {
    Vector<T> small(&pool);
    small.reserve(size);
    nimble::testing::addRandomData<T>(rng, size, &small, buffer);
    datasets.push_back(std::move(small));
  }

  return datasets;
}

template <typename T>
std::vector<Vector<T>> makeConstantViewDatasets(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  std::vector<Vector<T>> datasets;
  datasets.push_back(makeSingleValueData<T>(pool, rng, rowCount, buffer));

  for (uint32_t size : {1u, 2u, 3u}) {
    datasets.push_back(makeSingleValueData<T>(pool, rng, size, buffer));
  }
  return datasets;
}

template <typename EncodingClass>
std::vector<Vector<typename EncodingClass::cppDataType>> makeDatasets(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  using T = typename EncodingClass::cppDataType;
  if constexpr (
      Encoder<EncodingClass>::encodingType() == EncodingType::Constant) {
    return makeConstantViewDatasets<T>(pool, rng, rowCount, buffer);
  } else if constexpr (
      Encoder<EncodingClass>::encodingType() == EncodingType::DeltaBlock) {
    std::vector<Vector<T>> datasets;
    datasets.push_back(
        makeDeltaBlockSortedData<T>(pool, rng, rowCount, buffer));
    datasets.push_back(makeSingleValueData<T>(pool, rng, rowCount, buffer));
    for (uint32_t size : {1u, 2u, 3u}) {
      datasets.push_back(makeDeltaBlockSortedData<T>(pool, rng, size, buffer));
    }
    return datasets;
  } else {
    return makeViewDatasets<T>(pool, rng, rowCount, buffer);
  }
}

std::vector<uint32_t> makeProbeRows(std::mt19937& rng, uint32_t rowCount) {
  const auto probeCount = std::min<uint32_t>(rowCount, 256);
  std::vector<uint32_t> rows;
  rows.reserve(probeCount + std::min<uint32_t>(rowCount, 16));
  for (uint32_t i = 0; i < probeCount; ++i) {
    rows.push_back(folly::Random::rand32(rng) % rowCount);
  }
  for (uint32_t i = 0; i < rowCount && i < 16; ++i) {
    rows.push_back(i);
  }
  return rows;
}

template <typename EncodingClass>
void runEncodingViewFuzzer(
    uint32_t iterations,
    uint32_t maxRows,
    uint32_t seed) {
  using T = typename EncodingClass::cppDataType;
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  auto dataBuffer = std::make_unique<Buffer>(*pool);

  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "EncodingView fuzzer seed: " << seed
            << " encoding: " << toString(Encoder<EncodingClass>::encodingType())
            << " dtype: " << toString(TypeTraits<T>::dataType)
            << " iterations: " << iterations << " maxRows: " << maxRows;
  FuzzerSeedReporter::setCurrentSeed(seed);
  std::mt19937 rng(seed);

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    if (FuzzerBudget::expired()) {
      break;
    }
    FuzzerBudget::recordIteration();
    const auto rowCount = 1 + folly::Random::rand32(rng) % maxRows;
    auto datasets =
        makeDatasets<EncodingClass>(*pool, rng, rowCount, dataBuffer.get());

    for (const auto& data : datasets) {
      if (data.empty()) {
        continue;
      }

      for (const auto useVarint : {false, true}) {
        SCOPED_TRACE(
            ::testing::Message() << "iter=" << iter << " useVarint="
                                 << useVarint << " rowCount=" << data.size());
        const Encoding::Options options{.useVarintRowCount = useVarint};
        Buffer encodeBuffer(*pool);
        std::string_view serialized;
        try {
          serialized = Encoder<EncodingClass>::encode(
              encodeBuffer, data, CompressionType::Uncompressed, options);
        } catch (const NimbleUserError& e) {
          if (e.errorCode() == error_code::IncompatibleEncoding) {
            continue;
          }
          throw;
        }

        auto view = createEncodingView(serialized, pool.get(), options);
        ASSERT_NE(view, nullptr);

        std::vector<velox::BufferPtr> stringBuffers;
        auto stringBufferFactory = [&](uint32_t bytes) -> void* {
          auto& stringBuffer = stringBuffers.emplace_back(
              velox::AlignedBuffer::allocate<char>(bytes, pool.get()));
          return stringBuffer->asMutable<void>();
        };
        auto encoding = EncodingFactory().create(
            *pool, serialized, stringBufferFactory, options);
        ASSERT_NE(encoding, nullptr);

        T expected;
        T actual;
        for (const auto row : makeProbeRows(rng, data.size())) {
          SCOPED_TRACE(::testing::Message() << "row=" << row);
          encoding->reset();
          encoding->skip(row);
          encoding->materialize(1, &expected);
          view->readAt(row, &actual);
          expectEqual(expected, actual, row);
        }
      }
    }
  }
}

} // namespace

using TrivialViewTypes = ::testing::Types<
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
class TrivialEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(TrivialEncodingViewFuzzerTest, TrivialViewTypes);

TYPED_TEST(TrivialEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}

using ConstantViewTypes = ::testing::Types<
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
class ConstantEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(ConstantEncodingViewFuzzerTest, ConstantViewTypes);

TYPED_TEST(ConstantEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}

using RleViewTypes = ::testing::Types<
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
class RleEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(RleEncodingViewFuzzerTest, RleViewTypes);

TYPED_TEST(RleEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}

using DictionaryViewTypes = ::testing::Types<
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
class DictionaryEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(DictionaryEncodingViewFuzzerTest, DictionaryViewTypes);

TYPED_TEST(DictionaryEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}

using FixedBitWidthViewTypes = ::testing::Types<
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
class FixedBitWidthEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(FixedBitWidthEncodingViewFuzzerTest, FixedBitWidthViewTypes);

TYPED_TEST(FixedBitWidthEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}

using PforViewTypes = ::testing::Types<
    PFOREncoding<int8_t>,
    PFOREncoding<uint8_t>,
    PFOREncoding<int16_t>,
    PFOREncoding<uint16_t>,
    PFOREncoding<int32_t>,
    PFOREncoding<uint32_t>,
    PFOREncoding<int64_t>,
    PFOREncoding<uint64_t>>;

template <typename E>
class PforEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(PforEncodingViewFuzzerTest, PforViewTypes);

TYPED_TEST(PforEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}

using DeltaBlockViewTypes = ::testing::Types<
    DeltaBlockEncoding<int8_t>,
    DeltaBlockEncoding<uint8_t>,
    DeltaBlockEncoding<int16_t>,
    DeltaBlockEncoding<uint16_t>,
    DeltaBlockEncoding<int32_t>,
    DeltaBlockEncoding<uint32_t>,
    DeltaBlockEncoding<int64_t>,
    DeltaBlockEncoding<uint64_t>>;

template <typename E>
class DeltaBlockEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(DeltaBlockEncodingViewFuzzerTest, DeltaBlockViewTypes);

TYPED_TEST(DeltaBlockEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}

using HuffmanViewTypes = ::testing::Types<
    HuffmanEncoding<int8_t>,
    HuffmanEncoding<uint8_t>,
    HuffmanEncoding<int16_t>,
    HuffmanEncoding<uint16_t>,
    HuffmanEncoding<int32_t>,
    HuffmanEncoding<uint32_t>,
    HuffmanEncoding<int64_t>,
    HuffmanEncoding<uint64_t>>;

template <typename E>
class HuffmanEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(HuffmanEncodingViewFuzzerTest, HuffmanViewTypes);

TYPED_TEST(HuffmanEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}

using SimdForBitpackViewTypes = ::testing::Types<
    SimdForBitpackEncoding<uint8_t>,
    SimdForBitpackEncoding<uint16_t>,
    SimdForBitpackEncoding<uint32_t>,
    SimdForBitpackEncoding<uint64_t>>;

template <typename E>
class SimdForBitpackEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(SimdForBitpackEncodingViewFuzzerTest, SimdForBitpackViewTypes);

TYPED_TEST(SimdForBitpackEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}

using BlockBitPackingViewTypes = ::testing::Types<
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
class BlockBitPackingEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(
    BlockBitPackingEncodingViewFuzzerTest,
    BlockBitPackingViewTypes);

TYPED_TEST(BlockBitPackingEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}

using AlpViewTypes = ::testing::Types<ALPEncoding<float>, ALPEncoding<double>>;

template <typename E>
class AlpEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(AlpEncodingViewFuzzerTest, AlpViewTypes);

TYPED_TEST(AlpEncodingViewFuzzerTest, readAtMatchesMaterialize) {
  runEncodingViewFuzzer<TypeParam>(
      FLAGS_view_fuzzer_iterations,
      FLAGS_view_fuzzer_max_rows,
      FLAGS_view_fuzzer_seed);
}
