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

/// Fuzzes SubIntSplit's planner settings: decode weighting at random weights,
/// access patterns and read paths, the hybrid planner, and the size regression
/// decode weighting may trade. Every stream must read back exactly, and none
/// may exceed its whole-value floor: one FixedBitWidth section at the exact
/// bit width, times 1 + maxSizeRegression when decode weighting is on.
///
/// Configuration via CLI flags:
///   --sis_planner_fuzzer_iterations=N  Iterations per type (default: 10)
///   --sis_planner_fuzzer_max_rows=N    Maximum rows per stream (default: 3000)
///   --sis_planner_fuzzer_seed=N        Fixed seed, 0=random (default: 42)

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "folly/Random.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/subintsplit/TuningConfig.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"
#include "velox/dwio/nimble/fuzzer/encoding/SubIntSplitFuzzer.h"

DEFINE_uint32(
    sis_planner_fuzzer_iterations,
    10,
    "Number of SubIntSplit planner fuzzer iterations per type");
DEFINE_uint32(
    sis_planner_fuzzer_max_rows,
    3000,
    "Maximum rows per SubIntSplit planner fuzzer stream");
DEFINE_uint32(
    sis_planner_fuzzer_seed,
    42,
    "SubIntSplit planner fuzzer seed (0 = random)");

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::test;

namespace {

subintsplit::TuningConfig randomPlannerTuning(std::mt19937& rng) {
  subintsplit::TuningConfig tuning;
  constexpr double kWeights[] = {0.0, 0.0, 0.25, 0.5, 1.0, 2.0};
  auto& weighting = tuning.selector.decodeWeighting;
  weighting.weight = kWeights[folly::Random::rand32(rng) % std::size(kWeights)];
  weighting.accessPattern = static_cast<subintsplit::DecodeAccessPattern>(
      folly::Random::rand32(4, rng));
  weighting.readPath =
      static_cast<subintsplit::DecodeReadPath>(folly::Random::rand32(4, rng));
  constexpr double kRegressions[] = {0.0, 0.05, 0.25};
  tuning.maxSizeRegression =
      kRegressions[folly::Random::rand32(rng) % std::size(kRegressions)];
  tuning.hybridPlanner = folly::Random::oneIn(2, rng);
  return tuning;
}

// Bytes of the values as one FixedBitWidth section at its exact bit width,
// which is what the floor holds every plan to.
template <typename T>
uint64_t fixedBitWidthSectionBytes(
    velox::memory::MemoryPool& pool,
    const Vector<T>& data,
    const Encoding::Options& options) {
  using physicalType = typename TypeTraits<T>::physicalType;
  Vector<physicalType> values(&pool);
  values.reserve(data.size());
  for (const auto& value : data) {
    values.push_back(std::bit_cast<physicalType>(value));
  }
  auto sectionOptions = options;
  sectionOptions.fixedBitWidthUseExactBits = true;
  Buffer buffer(pool);
  return Encoder<FixedBitWidthEncoding<physicalType>>::encode(
             buffer, values, CompressionType::Uncompressed, sectionOptions)
      .size();
}

template <typename T>
void runSubIntSplitPlannerFuzzer(
    uint32_t iterations,
    uint32_t maxRows,
    uint32_t seed) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Buffer dataBuffer(*pool);
  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "SubIntSplit planner fuzzer seed: " << seed
            << " dtype: " << toString(TypeTraits<T>::dataType)
            << " iterations: " << iterations << " maxRows: " << maxRows;
  std::mt19937 rng(seed);

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const uint32_t rowCount = 1 + folly::Random::rand32(rng) % maxRows;
    std::vector<Vector<T>> streams;
    Vector<T> random(pool.get());
    random.reserve(rowCount);
    nimble::testing::addRandomData<T>(rng, rowCount, &random, &dataBuffer);
    streams.push_back(std::move(random));
    streams.push_back(
        makeSingleValueData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(makeMonotonicData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeBitStructuredData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeDominantValueData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeMixedRegimeData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(makeSnowflakeData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeAdversarialBitPatternData<T>(*pool, rng, rowCount, &dataBuffer));

    for (const auto& data : streams) {
      const auto tuning = randomPlannerTuning(rng);
      const Encoding::Options options{
          .useVarintRowCount = folly::Random::oneIn(2, rng)};
      const auto& weighting = tuning.selector.decodeWeighting;
      SCOPED_TRACE(
          ::testing::Message()
          << "seed=" << seed << " iter=" << iter << " rowCount=" << data.size()
          << " decodeWeight=" << weighting.weight
          << " accessPattern=" << static_cast<int>(weighting.accessPattern)
          << " readPath=" << static_cast<int>(weighting.readPath)
          << " maxSizeRegression=" << tuning.maxSizeRegression
          << " hybridPlanner=" << tuning.hybridPlanner);

      Buffer buffer(*pool);
      const auto encoded = encodeSubIntSplit(data, buffer, options, tuning);
      verifySubIntSplitReads(rng, *pool, encoded, data, options, tuning);

      const uint64_t specificBytes = encoded.size() -
          EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
      const uint64_t floorBytes = subintsplit::specificHeaderSize(1) +
          fixedBitWidthSectionBytes(*pool, data, options);
      const double allowedRegression =
          weighting.weight != 0.0 ? 1.0 + tuning.maxSizeRegression : 1.0;
      // One byte of slack for the floor's integer division.
      EXPECT_LE(
          static_cast<double>(specificBytes),
          allowedRegression * static_cast<double>(floorBytes + 1))
          << "floor " << floorBytes;
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
    }
  }
}

} // namespace

using SubIntSplitPlannerTypes =
    ::testing::Types<int32_t, uint32_t, int64_t, uint64_t, float, double>;

template <typename T>
class SubIntSplitPlannerFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(SubIntSplitPlannerFuzzerTest, SubIntSplitPlannerTypes);

TYPED_TEST(SubIntSplitPlannerFuzzerTest, plansRoundTripUnderTheFloor) {
  runSubIntSplitPlannerFuzzer<TypeParam>(
      FLAGS_sis_planner_fuzzer_iterations,
      FLAGS_sis_planner_fuzzer_max_rows,
      FLAGS_sis_planner_fuzzer_seed);
}

// Defines main() through folly::Init, as NimbleWriterFuzzerTest does, so the
// flags above are parsed; gtest_main would leave them at their defaults.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
