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

/// Property fuzzer for the size estimates SubIntSplit's sections are priced
/// with. Under subintsplit::Options::sectionEstimatorRefinements, selection
/// compares FixedBitWidth, MainlyConstant and RLE estimates against each
/// other's bytes, so each must stay close to what the encoding really writes.
/// Every stream from the encoding fuzzer's generators is encoded with each of
/// the three, nested selection on, and estimate / actual is checked against a
/// per-encoding band.
///
/// Configuration via CLI flags:
///   --estimator_fuzzer_iterations=N  Iterations per type (default: 20)
///   --estimator_fuzzer_max_rows=N    Maximum rows per stream (default: 5000)
///   --estimator_fuzzer_seed=N        Fixed seed, 0=random (default: 42)

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "folly/Random.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"

DEFINE_uint32(
    estimator_fuzzer_iterations,
    20,
    "Number of estimator refinement fuzzer iterations per type");
DEFINE_uint32(
    estimator_fuzzer_max_rows,
    5000,
    "Maximum rows per estimator refinement fuzzer stream");
DEFINE_uint32(
    estimator_fuzzer_seed,
    42,
    "Estimator refinement fuzzer seed (0 = random)");

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::test;

namespace {

// Streams shorter than this are dominated by fixed headers, which the
// estimates price only roughly and which no section comparison turns on.
constexpr uint32_t kMinRows{256};

// The band estimate / actual must fall in, per encoding. Each band is what
// 12 seeds of these streams measure with room either side, so it catches
// drift rather than admitting any answer.
struct Band {
  double atLeast;
  double atMost;
};

// FixedBitWidth's refined estimate counts every byte it writes.
constexpr Band kFixedBitWidthBand{1.0, 1.0};

// MainlyConstant prices its mask as a bitmap or a sparse list, so a mask that
// nested selection stores as runs writes far less than it is priced at (up to
// 51x measured). Only the lower bound is tight (0.86 measured), and it is the
// one that matters: an underestimate lets MainlyConstant win a section it then
// inflates.
constexpr Band kMainlyConstantBand{
    0.8,
    std::numeric_limits<double>::infinity()};

// RLE measures 0.72 to 1.43.
constexpr Band kRunLengthBand{0.65, 1.6};

template <typename T>
std::vector<Vector<T>> makeStreams(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  std::vector<Vector<T>> streams;
  Vector<T> random(&pool);
  random.reserve(rowCount);
  nimble::testing::addRandomData<T>(rng, rowCount, &random, buffer);
  streams.push_back(std::move(random));
  streams.push_back(makeSingleValueData<T>(pool, rng, rowCount, buffer));
  streams.push_back(makeBoundaryMixedData<T>(pool, rng, rowCount, buffer));
  streams.push_back(makeMonotonicData<T>(pool, rng, rowCount, buffer));
  streams.push_back(makeSortedData<T>(pool, rng, rowCount, buffer));
  streams.push_back(makeMainlyConstantData<T>(pool, rng, rowCount, buffer));
  streams.push_back(makeBitStructuredData<T>(pool, rng, rowCount, buffer));
  streams.push_back(makeLowCardinalityData<T>(pool, rng, rowCount, buffer));
  streams.push_back(makeDominantValueData<T>(pool, rng, rowCount, buffer));
  streams.push_back(makeMixedRegimeData<T>(pool, rng, rowCount, buffer));
  streams.push_back(makeSnowflakeData<T>(pool, rng, rowCount, buffer));
  return streams;
}

// Encodes `data` with EncodingClass and checks the refined estimate for it.
// Returns estimate / actual, or nothing when the encoding does not apply.
template <typename EncodingClass>
std::optional<double> checkEstimate(
    velox::memory::MemoryPool& pool,
    const Vector<typename EncodingClass::cppDataType>& data,
    const Encoding::Options& options,
    Band band) {
  using T = typename EncodingClass::cppDataType;
  using physicalType = typename TypeTraits<T>::physicalType;
  const std::span<const physicalType> values{
      reinterpret_cast<const physicalType*>(data.data()), data.size()};
  const auto statistics = Statistics<physicalType>::create(values);
  const auto encodingType = Encoder<EncodingClass>::encodingType();
  const auto estimate = detail::EncodingSizeEstimation<T>::estimateSize(
      encodingType, values, statistics, options);
  if (!estimate.has_value()) {
    return std::nullopt;
  }

  Buffer buffer(pool);
  std::string_view encoded;
  try {
    encoded = Encoder<EncodingClass>::encode(
        buffer,
        data,
        CompressionType::Uncompressed,
        options,
        /*realNestedSelection=*/true);
  } catch (const NimbleUserError& e) {
    if (e.errorCode() == error_code::IncompatibleEncoding) {
      return std::nullopt;
    }
    throw;
  }
  const double ratio = static_cast<double>(estimate.value()) /
      static_cast<double>(encoded.size());
  EXPECT_GE(ratio, band.atLeast)
      << toString(encodingType) << " estimate " << estimate.value()
      << " actual " << encoded.size();
  EXPECT_LE(ratio, band.atMost)
      << toString(encodingType) << " estimate " << estimate.value()
      << " actual " << encoded.size();
  return ratio;
}

// Tracks the extremes of estimate / actual seen for one encoding.
struct Envelope {
  double lowest{std::numeric_limits<double>::max()};
  double highest{0};
  uint32_t count{0};

  void add(std::optional<double> ratio) {
    if (ratio.has_value()) {
      lowest = std::min(lowest, ratio.value());
      highest = std::max(highest, ratio.value());
      ++count;
    }
  }
};

template <typename T>
void runEstimatorRefinementFuzzer(
    uint32_t iterations,
    uint32_t maxRows,
    uint32_t seed) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Buffer dataBuffer(*pool);
  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "Estimator refinement fuzzer seed: " << seed
            << " dtype: " << toString(TypeTraits<T>::dataType)
            << " iterations: " << iterations << " maxRows: " << maxRows;
  std::mt19937 rng(seed);

  Encoding::Options options;
  options.subIntSplit.sectionEstimatorRefinements = true;
  // SubIntSplit's sections are written at their exact bit widths.
  options.fixedBitWidthUseExactBits = true;
  // FixedBitWidth's estimate must also count a varint row count exactly.
  auto varintOptions = options;
  varintOptions.useVarintRowCount = true;

  Envelope fixedBitWidth;
  Envelope mainlyConstant;
  Envelope runLength;
  const uint32_t rowSpan = std::max(maxRows, kMinRows) - kMinRows + 1;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const uint32_t rowCount = kMinRows + folly::Random::rand32(rng) % rowSpan;
    for (const auto& data : makeStreams<T>(*pool, rng, rowCount, &dataBuffer)) {
      if (data.empty()) {
        continue;
      }
      SCOPED_TRACE(
          ::testing::Message() << "seed=" << seed << " iter=" << iter
                               << " rowCount=" << data.size());
      fixedBitWidth.add(
          checkEstimate<FixedBitWidthEncoding<T>>(
              *pool, data, options, kFixedBitWidthBand));
      fixedBitWidth.add(
          checkEstimate<FixedBitWidthEncoding<T>>(
              *pool, data, varintOptions, kFixedBitWidthBand));
      mainlyConstant.add(
          checkEstimate<MainlyConstantEncoding<T>>(
              *pool, data, options, kMainlyConstantBand));
      runLength.add(
          checkEstimate<RLEEncoding<T>>(*pool, data, options, kRunLengthBand));
    }
  }
  for (const auto& [name, envelope] :
       {std::pair<const char*, const Envelope&>{"FixedBitWidth", fixedBitWidth},
        {"MainlyConstant", mainlyConstant},
        {"RLE", runLength}}) {
    LOG(INFO) << name << " estimate/actual over " << envelope.count
              << " streams: [" << envelope.lowest << ", " << envelope.highest
              << "]";
  }
}

} // namespace

template <typename T>
class EstimatorRefinementFuzzerTest : public ::testing::Test {};

using IntegerTypes = ::testing::Types<
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t>;
TYPED_TEST_SUITE(EstimatorRefinementFuzzerTest, IntegerTypes);

TYPED_TEST(EstimatorRefinementFuzzerTest, estimatesTrackEncodedSize) {
  runEstimatorRefinementFuzzer<TypeParam>(
      FLAGS_estimator_fuzzer_iterations,
      FLAGS_estimator_fuzzer_max_rows,
      FLAGS_estimator_fuzzer_seed);
}

// Defines main() through folly::Init, as NimbleWriterFuzzerTest does, so the
// flags above are parsed; gtest_main would leave them at their defaults.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
