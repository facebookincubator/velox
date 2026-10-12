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

/// Fuzzes SubIntSplit under randomised subintsplit::TuningConfig settings.
///
/// Each iteration draws the decode-only switches (foldConstantSections,
/// passThrough, visitorBlockBuffer, decodeChunkSize) and a restricted set of
/// encodings the planner may cost sections against. The decode-only switches
/// must leave the encoded bytes unchanged; every stream must read back
/// exactly through materialize, skip and reset under the drawn decode
/// switches, and through the stream's EncodingView.
///
/// Configuration via CLI flags:
///   --sis_tuning_fuzzer_iterations=N  Iterations per type (default: 10)
///   --sis_tuning_fuzzer_max_rows=N    Maximum rows per stream (default: 3000)
///   --sis_tuning_fuzzer_seed=N        Fixed seed, 0=random (default: 42)

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "folly/Random.h"
#include "velox/dwio/nimble/encodings/subintsplit/TuningConfig.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"
#include "velox/dwio/nimble/fuzzer/encoding/SubIntSplitFuzzer.h"

DEFINE_uint32(
    sis_tuning_fuzzer_iterations,
    10,
    "Number of SubIntSplit tuning fuzzer iterations per type");
DEFINE_uint32(
    sis_tuning_fuzzer_max_rows,
    3000,
    "Maximum rows per SubIntSplit tuning fuzzer stream");
DEFINE_uint32(
    sis_tuning_fuzzer_seed,
    42,
    "SubIntSplit tuning fuzzer seed (0 = random)");

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::test;

namespace {

// Draws the settings that only change how a stream is read.
void randomizeDecodeSwitches(
    std::mt19937& rng,
    subintsplit::TuningConfig& tuning) {
  tuning.foldConstantSections = folly::Random::oneIn(2, rng);
  tuning.passThrough = folly::Random::oneIn(2, rng);
  tuning.visitorBlockBuffer = folly::Random::oneIn(2, rng);
  constexpr uint32_t kChunkSizes[] = {1, 7, 64, 256, 4'096};
  tuning.decodeChunkSize =
      kChunkSizes[folly::Random::rand32(rng) % std::size(kChunkSizes)];
}

// A random subset of the encodings a section may be costed against, empty
// (every encoding) a third of the time.
subintsplit::AllowedEncodings randomAllowedEncodings(std::mt19937& rng) {
  subintsplit::AllowedEncodings allowed;
  if (folly::Random::oneIn(3, rng)) {
    return allowed;
  }
  constexpr EncodingType kSectionEncodings[] = {
      EncodingType::Trivial,
      EncodingType::Constant,
      EncodingType::FixedBitWidth,
      EncodingType::MainlyConstant,
      EncodingType::Dictionary,
      EncodingType::RLE,
      EncodingType::Varint,
      EncodingType::FOR,
      EncodingType::SimdForBitpack,
      EncodingType::BlockBitPacking,
      EncodingType::PFOR,
  };
  for (const auto encodingType : kSectionEncodings) {
    if (folly::Random::oneIn(2, rng)) {
      allowed.insert(encodingType);
    }
  }
  return allowed;
}

template <typename T>
void runSubIntSplitTuningFuzzer(
    uint32_t iterations,
    uint32_t maxRows,
    uint32_t seed) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Buffer dataBuffer(*pool);
  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "SubIntSplit tuning fuzzer seed: " << seed
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
        makeMainlyConstantData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeLowCardinalityData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(makeSnowflakeData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeAdversarialBitPatternData<T>(*pool, rng, rowCount, &dataBuffer));

    for (const auto& data : streams) {
      subintsplit::TuningConfig tuning;
      tuning.allowedEncodings = randomAllowedEncodings(rng);
      auto decodeTuning = tuning;
      randomizeDecodeSwitches(rng, decodeTuning);
      const Encoding::Options options{
          .useVarintRowCount = folly::Random::oneIn(2, rng)};
      SCOPED_TRACE(
          ::testing::Message()
          << "seed=" << seed << " iter=" << iter << " rowCount=" << data.size()
          << " allowedEncodings=" << tuning.allowedEncodings.size()
          << " fold=" << decodeTuning.foldConstantSections
          << " passThrough=" << decodeTuning.passThrough
          << " visitorBlockBuffer=" << decodeTuning.visitorBlockBuffer
          << " decodeChunkSize=" << decodeTuning.decodeChunkSize);

      Buffer buffer(*pool);
      const auto encoded = encodeSubIntSplit(data, buffer, options, tuning);
      // The decode switches are read at decode only, so encoding under them
      // must produce the same bytes.
      Buffer decodeBuffer(*pool);
      ASSERT_EQ(
          encodeSubIntSplit(data, decodeBuffer, options, decodeTuning),
          encoded);

      verifySubIntSplitReads(rng, *pool, encoded, data, options, decodeTuning);
      verifySubIntSplitView(rng, *pool, encoded, data, options);
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
    }
  }
}

} // namespace

using SubIntSplitTuningTypes =
    ::testing::Types<int32_t, uint32_t, int64_t, uint64_t, float, double>;

template <typename T>
class SubIntSplitTuningFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(SubIntSplitTuningFuzzerTest, SubIntSplitTuningTypes);

TYPED_TEST(SubIntSplitTuningFuzzerTest, randomTuningRoundTrips) {
  runSubIntSplitTuningFuzzer<TypeParam>(
      FLAGS_sis_tuning_fuzzer_iterations,
      FLAGS_sis_tuning_fuzzer_max_rows,
      FLAGS_sis_tuning_fuzzer_seed);
}

// Defines main() through folly::Init, as NimbleWriterFuzzerTest does, so the
// flags above are parsed; gtest_main would leave them at their defaults.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
