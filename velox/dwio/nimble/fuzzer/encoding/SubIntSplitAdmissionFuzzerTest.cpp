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

/// Fuzzes top-level selection with SubIntSplit offered as a candidate under
/// random admission settings: the admission mode, whether an admitted stream
/// is forced, how many pairs the profile samples, and the compression guard.
/// A stream the bit-flip gate rejects must not be SubIntSplit, an admitted
/// stream under admissionForces must be, and whatever selection picks, down
/// through nested streams, must read back exactly.
///
/// Configuration via CLI flags:
///   --sis_admission_fuzzer_iterations=N  Iterations per type (default: 10)
///   --sis_admission_fuzzer_max_rows=N    Maximum rows per stream (default:
///                                        5000)
///   --sis_admission_fuzzer_seed=N        Fixed seed, 0=random (default: 42)

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "folly/Random.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/subintsplit/TopLevelPolicy.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"

DEFINE_uint32(
    sis_admission_fuzzer_iterations,
    10,
    "Number of SubIntSplit admission fuzzer iterations per type");
DEFINE_uint32(
    sis_admission_fuzzer_max_rows,
    5000,
    "Maximum rows per SubIntSplit admission fuzzer stream");
DEFINE_uint32(
    sis_admission_fuzzer_seed,
    42,
    "SubIntSplit admission fuzzer seed (0 = random)");

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::test;

namespace {

Encoding::Options randomAdmissionOptions(std::mt19937& rng) {
  Encoding::Options options;
  options.useVarintRowCount = folly::Random::oneIn(2, rng);
  auto& subIntSplit = options.subIntSplit;
  subIntSplit.admission = static_cast<subintsplit::SubIntSplitAdmission>(
      folly::Random::rand32(3, rng));
  subIntSplit.admissionForces = folly::Random::oneIn(2, rng);
  constexpr uint32_t kProfilePairs[] = {0, 1, 16, 1'024, 4'096};
  subIntSplit.admissionProfilePairs =
      kProfilePairs[folly::Random::rand32(rng) % std::size(kProfilePairs)];
  subIntSplit.inNestedStreams = folly::Random::oneIn(2, rng);
  subIntSplit.estimateCompressionGuard = folly::Random::oneIn(2, rng);
  return options;
}

std::vector<std::pair<EncodingType, float>> readFactorsWithSubIntSplit() {
  auto readFactors =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  if (std::none_of(
          readFactors.begin(), readFactors.end(), [](const auto& factor) {
            return factor.first == EncodingType::SubIntSplit;
          })) {
    readFactors.emplace_back(EncodingType::SubIntSplit, 1.0f);
  }
  return readFactors;
}

template <typename T>
void runSubIntSplitAdmissionFuzzer(
    uint32_t iterations,
    uint32_t maxRows,
    uint32_t seed) {
  using physicalType = typename TypeTraits<T>::physicalType;
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Buffer dataBuffer(*pool);
  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "SubIntSplit admission fuzzer seed: " << seed
            << " dtype: " << toString(TypeTraits<T>::dataType)
            << " iterations: " << iterations << " maxRows: " << maxRows;
  std::mt19937 rng(seed);

  uint32_t numSubIntSplit{0};
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
        makeLowCardinalityData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeMixedRegimeData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(makeSnowflakeData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeAdversarialBitPatternData<T>(*pool, rng, rowCount, &dataBuffer));

    for (const auto& data : streams) {
      const auto options = randomAdmissionOptions(rng);
      const auto& subIntSplit = options.subIntSplit;
      const bool compress = folly::Random::oneIn(3, rng);
      SCOPED_TRACE(
          ::testing::Message()
          << "seed=" << seed << " iter=" << iter << " rowCount=" << data.size()
          << " admission=" << static_cast<int>(subIntSplit.admission)
          << " forces=" << subIntSplit.admissionForces
          << " pairs=" << subIntSplit.admissionProfilePairs
          << " inNestedStreams=" << subIntSplit.inNestedStreams
          << " compressionGuard=" << subIntSplit.estimateCompressionGuard
          << " compress=" << compress);
      const std::span<const physicalType> values{
          reinterpret_cast<const physicalType*>(data.data()), data.size()};
      const std::span<const T> typedValues{data.data(), data.size()};

      std::optional<CompressionOptions> compressionOptions;
      if (compress) {
        compressionOptions = CompressionOptions{};
      }
      Buffer buffer(*pool);
      const auto encoded = EncodingFactory::encode<T>(
          std::make_unique<ManualEncodingSelectionPolicy<T>>(
              readFactorsWithSubIntSplit(), compressionOptions, std::nullopt),
          typedValues,
          buffer,
          options);
      const auto encodingType = EncodingPrefix::encodingType(encoded);
      numSubIntSplit += encodingType == EncodingType::SubIntSplit;

      if (subIntSplit.admission !=
          subintsplit::SubIntSplitAdmission::kEstimate) {
        const bool admitted = subintsplit::bitFlipAdmits(
            subintsplit::bitFlipAdmissionProfile(
                values,
                subIntSplit.admission,
                subIntSplit.admissionProfilePairs),
            subIntSplit.admission,
            subintsplit::TopLevelPolicyConfig{});
        if (!admitted) {
          EXPECT_NE(encodingType, EncodingType::SubIntSplit);
        } else if (subIntSplit.admissionForces) {
          EXPECT_EQ(encodingType, EncodingType::SubIntSplit);
        }
      }

      std::vector<velox::BufferPtr> stringBuffers;
      auto encoding = EncodingFactory{options}.create(
          *pool, encoded, [&](uint32_t totalLength) {
            auto& stringBuffer = stringBuffers.emplace_back(
                velox::AlignedBuffer::allocate<char>(totalLength, pool.get()));
            return stringBuffer->template asMutable<void>();
          });
      ASSERT_EQ(encoding->rowCount(), data.size());
      Vector<T> actual(pool.get(), data.size());
      encoding->materialize(data.size(), actual.data());
      for (uint32_t row = 0; row < data.size(); ++row) {
        ASSERT_EQ(
            std::bit_cast<physicalType>(actual[row]),
            std::bit_cast<physicalType>(data[row]))
            << "Mismatch at row " << row << " of " << toString(encodingType);
      }
    }
  }
  LOG(INFO) << numSubIntSplit << " streams selected SubIntSplit";
}

} // namespace

// Admission reads integral physical types of 4 or 8 bytes.
using SubIntSplitAdmissionTypes =
    ::testing::Types<int32_t, uint32_t, int64_t, uint64_t>;

template <typename T>
class SubIntSplitAdmissionFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(SubIntSplitAdmissionFuzzerTest, SubIntSplitAdmissionTypes);

TYPED_TEST(SubIntSplitAdmissionFuzzerTest, admissionDecidesCandidacy) {
  runSubIntSplitAdmissionFuzzer<TypeParam>(
      FLAGS_sis_admission_fuzzer_iterations,
      FLAGS_sis_admission_fuzzer_max_rows,
      FLAGS_sis_admission_fuzzer_seed);
}

// Defines main() through folly::Init, as NimbleWriterFuzzerTest does, so the
// flags above are parsed; gtest_main would leave them at their defaults.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
