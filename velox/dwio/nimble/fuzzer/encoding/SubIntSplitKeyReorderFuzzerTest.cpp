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

/// Fuzzes SubIntSplit's key-derived section reorder: a key pinned to a random
/// section, the key searched by price (autoTransform), and the transform
/// forced on every eligible section. Every stream must read back exactly
/// through the encoding and through its EncodingView, whether or not it was
/// written reordered.
///
/// Configuration via CLI flags:
///   --sis_reorder_fuzzer_iterations=N  Iterations per type (default: 10)
///   --sis_reorder_fuzzer_max_rows=N    Maximum rows per stream (default: 3000)
///   --sis_reorder_fuzzer_seed=N        Fixed seed, 0=random (default: 42)

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "folly/Random.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"
#include "velox/dwio/nimble/encodings/subintsplit/TuningConfig.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"
#include "velox/dwio/nimble/fuzzer/encoding/SubIntSplitFuzzer.h"

DEFINE_uint32(
    sis_reorder_fuzzer_iterations,
    10,
    "Number of SubIntSplit key reorder fuzzer iterations per type");
DEFINE_uint32(
    sis_reorder_fuzzer_max_rows,
    3000,
    "Maximum rows per SubIntSplit key reorder fuzzer stream");
DEFINE_uint32(
    sis_reorder_fuzzer_seed,
    42,
    "SubIntSplit key reorder fuzzer seed (0 = random)");

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::test;

namespace {

enum class ReorderMode { kPinnedKey, kAuto, kForced };

// Values whose high field is a key with few distinct values and whose low
// field depends on the key, so sorting the low field by the key groups it.
// The key repeats in shuffled order rather than in runs, which is where a
// reorder pays.
template <typename T>
Vector<T> makeKeyedData(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    uint32_t rowCount) {
  using UnsignedT = std::make_unsigned_t<T>;
  constexpr int kBits = sizeof(T) * 8;
  const int lowBits = 4 + folly::Random::rand32(rng) % (kBits / 2 - 4);
  const uint32_t numKeys = 1 + folly::Random::rand32(rng) % 64;
  std::vector<UnsignedT> keys(numKeys);
  std::vector<UnsignedT> lowByKey(numKeys);
  const auto lowMask = static_cast<UnsignedT>((UnsignedT{1} << lowBits) - 1);
  for (uint32_t key = 0; key < numKeys; ++key) {
    keys[key] = static_cast<UnsignedT>(folly::Random::rand64(rng) >> lowBits);
    lowByKey[key] =
        static_cast<UnsignedT>(folly::Random::rand64(rng)) & lowMask;
  }
  const uint32_t noiseBits = folly::Random::rand32(rng) % 4;
  Vector<T> data(&pool);
  data.reserve(rowCount);
  for (uint32_t row = 0; row < rowCount; ++row) {
    const uint32_t key = folly::Random::rand32(rng) % numKeys;
    auto low = lowByKey[key];
    if (noiseBits > 0) {
      low ^= static_cast<UnsignedT>(
          folly::Random::rand32(rng) & ((1u << noiseBits) - 1));
    }
    data.push_back(
        static_cast<T>(
            static_cast<UnsignedT>((keys[key] << lowBits) | (low & lowMask))));
  }
  return data;
}

subintsplit::TuningConfig reorderTuning(ReorderMode mode, std::mt19937& rng) {
  subintsplit::TuningConfig tuning;
  switch (mode) {
    case ReorderMode::kPinnedKey:
      tuning.transform =
          static_cast<uint8_t>(subintsplit::TransformId::KeyDerived);
      tuning.keySection = static_cast<uint8_t>(folly::Random::rand32(3, rng));
      break;
    case ReorderMode::kAuto:
      tuning.autoTransform = true;
      break;
    case ReorderMode::kForced:
      tuning.transform =
          static_cast<uint8_t>(subintsplit::TransformId::KeyDerived);
      tuning.forceApply = true;
      break;
  }
  tuning.rowFrame = folly::Random::oneIn(2, rng);
  return tuning;
}

template <typename T>
void runSubIntSplitKeyReorderFuzzer(
    uint32_t iterations,
    uint32_t maxRows,
    uint32_t seed) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Buffer dataBuffer(*pool);
  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "SubIntSplit key reorder fuzzer seed: " << seed
            << " dtype: " << toString(TypeTraits<T>::dataType)
            << " iterations: " << iterations << " maxRows: " << maxRows;
  std::mt19937 rng(seed);

  uint32_t numReordered{0};
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const uint32_t rowCount = 2 + folly::Random::rand32(rng) % maxRows;
    std::vector<Vector<T>> streams;
    streams.push_back(makeKeyedData<T>(*pool, rng, rowCount));
    streams.push_back(makeKeyedData<T>(*pool, rng, rowCount));
    streams.push_back(makeSnowflakeData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeLowCardinalityData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeMixedRegimeData<T>(*pool, rng, rowCount, &dataBuffer));

    for (const auto& data : streams) {
      for (const auto mode :
           {ReorderMode::kPinnedKey,
            ReorderMode::kAuto,
            ReorderMode::kForced}) {
        const auto tuning = reorderTuning(mode, rng);
        const Encoding::Options options{
            .useVarintRowCount = folly::Random::oneIn(2, rng)};
        SCOPED_TRACE(
            ::testing::Message()
            << "seed=" << seed << " iter=" << iter << " rowCount="
            << data.size() << " reorderMode=" << static_cast<int>(mode)
            << " keySection=" << static_cast<int>(tuning.keySection)
            << " rowFrame=" << tuning.rowFrame);

        Buffer buffer(*pool);
        const auto encoded = encodeSubIntSplit(data, buffer, options, tuning);
        numReordered += EncodingPrefix::encodingType(encoded) ==
            EncodingType::SubIntSplitReordered;

        verifySubIntSplitReads(rng, *pool, encoded, data, options, tuning);
        verifySubIntSplitView(rng, *pool, encoded, data, options);
        if (::testing::Test::HasFatalFailure()) {
          return;
        }
      }
    }
  }
  // Keyed streams reorder under the forced mode at least, so a run with no
  // reordered stream did not test the transform.
  EXPECT_GT(numReordered, 0);
}

} // namespace

using SubIntSplitKeyReorderTypes =
    ::testing::Types<int32_t, uint32_t, int64_t, uint64_t>;

template <typename T>
class SubIntSplitKeyReorderFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(SubIntSplitKeyReorderFuzzerTest, SubIntSplitKeyReorderTypes);

TYPED_TEST(SubIntSplitKeyReorderFuzzerTest, reorderedStreamsRoundTrip) {
  runSubIntSplitKeyReorderFuzzer<TypeParam>(
      FLAGS_sis_reorder_fuzzer_iterations,
      FLAGS_sis_reorder_fuzzer_max_rows,
      FLAGS_sis_reorder_fuzzer_seed);
}

// Defines main() through folly::Init, as NimbleWriterFuzzerTest does, so the
// flags above are parsed; gtest_main would leave them at their defaults.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
