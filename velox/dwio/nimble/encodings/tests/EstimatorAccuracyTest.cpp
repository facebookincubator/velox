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
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/CostModel.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"

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

  // Bytes `encodingType` writes for `values`, through the real encoder.
  //
  // The outer encoding is pinned and its sub-streams are not: an encoding with
  // children writes what nested selection gives those children, and pinning
  // them too would measure a stream nobody encodes. Every estimator here
  // prices its children through the estimators selection applies to them, so
  // this is the encode those prices describe.
  uint64_t encodedBytes(
      EncodingType encodingType,
      const std::vector<uint64_t>& values) {
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<uint64_t>>(
        std::vector<std::pair<EncodingType, float>>{{encodingType, 1.0}},
        std::nullopt,
        std::nullopt,
        ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors());
    return EncodingFactory::encode<uint64_t>(
               std::move(policy),
               std::span<const uint64_t>{values.data(), values.size()},
               *buffer_,
               options())
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
      static_cast<double>(encodedBytes(EncodingType::FOR, values));
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
      static_cast<double>(encodedBytes(EncodingType::FOR, values));
  expectRatioWithin("forModelClustered", estimate, actual, 0.65, 1.20);
}

// FOR's own values-based estimator walks the same frames the encoder does and
// prices the per-frame streams through the estimators nested selection applies
// to them, so it is exact up to those, and measures 1.00x.
TEST_F(EstimatorAccuracyTest, forEstimatorOnHighEntropyStream) {
  const auto values = randomHighEntropyValues();
  const double estimate =
      static_cast<double>(ForEncoding<uint64_t>::estimateSize(
          {values.data(), values.size()}, options()));
  const double actual =
      static_cast<double>(encodedBytes(EncodingType::FOR, values));
  expectRatioWithin("forEstimatorHighEntropy", estimate, actual, 0.97, 1.04);
}

// FrequencyPartition's estimate runs above what it writes -- 1.30x here, and
// 1.23x at the median over the nine evaluation columns -- because it prices
// the TierTagArray tag stream at a flat tagBits a row while the encoder puts
// that stream through nested selection. Over-quoting costs the encoding
// sections it would have stored smaller, and costs the whole-value fallback a
// hard-coded slack to undo (SubIntSplitEncoding's wholeValueEstimateSlack).
TEST_F(EstimatorAccuracyTest, frequencyPartitionEstimatorOnSkewedStream) {
  const auto values = skewedFrequencyValues();
  const auto statistics = Statistics<uint64_t>::create(
      std::span<const uint64_t>{values.data(), values.size()});
  const double estimate =
      static_cast<double>(FrequencyPartitionEncoding<uint64_t>::estimateSize(
          values.size(), statistics, options()));
  const double actual = static_cast<double>(
      encodedBytes(EncodingType::FrequencyPartition, values));
  expectRatioWithin(
      "frequencyPartitionEstimatorSkewed", estimate, actual, 1.10, 1.50);
}

// The planner's FrequencyPartition model prices the same wire format from
// coarser inputs -- tier coverage rather than the counts -- and is further out
// again, at 2.36x. Both paths over-quote for the same reason, so closing one
// without the other would leave the planner and section selection disagreeing
// about the same encoding.
TEST_F(EstimatorAccuracyTest, frequencyPartitionModelOnSkewedStream) {
  const auto values = skewedFrequencyValues();
  const double estimate =
      plannerModelBytes(subintsplit::frequencyPartitionCostBits, values);
  const double actual = static_cast<double>(
      encodedBytes(EncodingType::FrequencyPartition, values));
  expectRatioWithin(
      "frequencyPartitionModelSkewed", estimate, actual, 1.80, 2.90);
}

// Delta prices its delta stream as a bit-packed array, and on a stream whose
// deltas are in fact bit-packed it is exact. It is not exact in general:
// nested selection chooses by estimate times read factor, so it can hand the
// delta stream to an encoding that writes more, and the estimator cannot see
// the read factors the policy holds. That gap is a property of the policy, not
// of this formula, which is why this test pins the case the formula covers.
TEST_F(EstimatorAccuracyTest, deltaEstimatorOnLocallyClusteredStream) {
  const auto values = locallyClusteredValues();
  const auto statistics = Statistics<uint64_t>::create(
      std::span<const uint64_t>{values.data(), values.size()});
  const double estimate =
      static_cast<double>(DeltaEncoding<uint64_t>::estimateSize(
          values.size(), statistics, options()));
  const double actual =
      static_cast<double>(encodedBytes(EncodingType::Delta, values));
  expectRatioWithin("deltaEstimatorClustered", estimate, actual, 0.95, 1.10);
}

} // namespace

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
