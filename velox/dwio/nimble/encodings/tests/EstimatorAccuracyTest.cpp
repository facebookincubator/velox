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

// What the size estimators quote against what the encoder writes.
//
// The other estimator tests in this directory pin an estimate to an arithmetic
// expression, which says the formula was transcribed and nothing about whether
// the formula is right. These encode the stream and compare, so a model that
// drifts from the wire format fails here even when its arithmetic is intact.
//
// Every bound is the ratio these streams measure with room either side of it,
// not a round number: the point is to catch drift, and a bound loose enough to
// admit any answer catches nothing. Several of the bounds therefore record a
// bias rather than deny one -- the planner's FOR model quotes below what FOR
// writes and its FrequencyPartition model well above -- and each says which,
// so that closing one is a deliberate change to these numbers and not a
// surprise.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <iostream>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/subintsplit/CostModel.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

// Rows per stream. Above FOR's 128-row frame by enough frames that per-frame
// metadata is a small share of the payload, which is the regime sections are
// encoded in.
constexpr size_t kRows{8'192};

// Sixty-two random bits under two constant ones, which is a UUIDv7's low half:
// the stream FOR's model quoted at 59.21 bits a value and wrote at 64.85.
std::vector<uint64_t> randomHighEntropyValues() {
  std::mt19937_64 generator{0x5eed};
  std::vector<uint64_t> values;
  values.reserve(kRows);
  for (size_t i = 0; i < kRows; ++i) {
    values.push_back(generator() >> 2);
  }
  return values;
}

// A counter with jitter: adjacent values are close, so every frame spans far
// less than the stream does. This is the arrangement FOR exists for, and the
// one a model that measures frames has to keep pricing cheaply.
std::vector<uint64_t> locallyClusteredValues() {
  std::mt19937_64 generator{0x1dea};
  std::vector<uint64_t> values;
  values.reserve(kRows);
  uint64_t current = 1'000'000'000;
  for (size_t i = 0; i < kRows; ++i) {
    current += generator() % 64;
    values.push_back(current);
  }
  return values;
}

// A skewed alphabet: two values take most of the rows, six more take most of
// the rest, and a long tail holds the remainder. This is the shape that puts
// rows in FrequencyPartition's narrow tiers, and so the shape whose tag stream
// the index estimate has to price.
std::vector<uint64_t> skewedFrequencyValues() {
  std::mt19937_64 generator{0xb00c};
  std::vector<uint64_t> values;
  values.reserve(kRows);
  for (size_t i = 0; i < kRows; ++i) {
    const uint64_t draw = generator() % 100;
    if (draw < 60) {
      values.push_back(draw % 2);
    } else if (draw < 90) {
      values.push_back(2 + draw % 6);
    } else {
      values.push_back(100 + generator() % 400);
    }
  }
  return values;
}

class EstimatorAccuracyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  // Section options, since every stream these estimators are judged on is a
  // SubIntSplit section and the options decide what the encoder writes.
  Encoding::Options options() const {
    return subintsplit::sectionEncodingOptions(Encoding::Options{});
  }

  // Bytes encoding `E` writes for `values`, through the real encoder.
  //
  // The outer encoding is pinned and its sub-streams are not: an encoding with
  // children writes what nested selection gives those children, and pinning
  // them too would measure a stream nobody encodes. The outer encoding is
  // encoded directly rather than through a selection policy, since selection
  // has no size estimate for FOR or FrequencyPartition to admit them by.
  template <typename E>
  uint64_t encodedBytes(const std::vector<uint64_t>& values) {
    const nimble::Vector<uint64_t> vector{pool_.get(), values.begin(), values.end()};
    return test::Encoder<E>::encode(
               *buffer_,
               vector,
               CompressionType::Uncompressed,
               options(),
               /*realNestedSelection=*/true)
        .size();
  }

  // What the planner's cost models quote for `values`, in bytes, so that a
  // model and an encode can be compared in the same unit.
  double plannerModelBytes(
      double (*model)(const subintsplit::SectionMetrics&, size_t, int) noexcept,
      const std::vector<uint64_t>& values) {
    subintsplit::MetricCollector collector;
    const auto metrics = collector.compute(values);
    return model(metrics, values.size(), 64) / 8.0;
  }

  // Reports the ratio as well as asserting on it, so that a run of these
  // tests is also the measurement the bounds were set from.
  void expectRatioWithin(
      const char* what,
      double estimate,
      double actual,
      double atLeast,
      double atMost) {
    const double ratio = estimate / actual;
    RecordProperty(what, std::to_string(ratio));
    std::cout << "  " << what << " estimate/actual = " << ratio << " ("
              << estimate << " / " << actual << ")\n";
    EXPECT_GT(ratio, atLeast) << what;
    EXPECT_LT(ratio, atMost) << what;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
};

// The planner's FOR model quotes 0.90x of what FOR writes on a high-entropy
// stream, because it takes a frame's local range from the average step rather
// than from the frames. Under-quoting is the direction that costs bytes: it
// wins FOR sections it then inflates. The bound admits the bias and stops it
// growing.
TEST_F(EstimatorAccuracyTest, forModelOnHighEntropyStream) {
  const auto values = randomHighEntropyValues();
  const double estimate = plannerModelBytes(subintsplit::forCostBits, values);
  const double actual =
      static_cast<double>(encodedBytes<ForEncoding<uint64_t>>(values));
  expectRatioWithin("forModelHighEntropy", estimate, actual, 0.82, 1.05);
}

// The same model on the arrangement FOR exists for, where it quotes 0.74x.
// The upper bound is what matters here: a model that priced every stream at
// its global range would quote this one many times over and FOR would never
// win the sections it is best at.
TEST_F(EstimatorAccuracyTest, forModelOnLocallyClusteredStream) {
  const auto values = locallyClusteredValues();
  const double estimate = plannerModelBytes(subintsplit::forCostBits, values);
  const double actual =
      static_cast<double>(encodedBytes<ForEncoding<uint64_t>>(values));
  expectRatioWithin("forModelClustered", estimate, actual, 0.65, 1.20);
}

// The planner's FrequencyPartition model prices from tier coverage rather
// than the counts, and over-quotes at 2.36x. Over-quoting costs the encoding
// sections it would have stored smaller.
TEST_F(EstimatorAccuracyTest, frequencyPartitionModelOnSkewedStream) {
  const auto values = skewedFrequencyValues();
  const double estimate =
      plannerModelBytes(subintsplit::frequencyPartitionCostBits, values);
  const double actual = static_cast<double>(
      encodedBytes<FrequencyPartitionEncoding<uint64_t>>(values));
  expectRatioWithin(
      "frequencyPartitionModelSkewed", estimate, actual, 1.80, 2.90);
}

} // namespace

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
