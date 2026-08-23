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
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "fmt/core.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleCompare.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"

#include <array>
#include <limits>
#include <random>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace facebook;

template <typename DataType, bool UseVarint>
struct TestConfig {
  using data_type = DataType;
  static constexpr bool useVarint = UseVarint;
};

#define TC(T) TestConfig<T, false>, TestConfig<T, true>

template <typename Config>
class ALPEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> toVector(std::initializer_list<T> l) {
    nimble::Vector<T> v{pool_.get()};
    v.insert(v.end(), l.begin(), l.end());
    return v;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

using TestTypes = ::testing::Types<TC(float), TC(double)>;

TYPED_TEST_CASE(ALPEncodingTest, TestTypes);

nimble::EncodingLayout fixedBitWidthLayout() {
  return nimble::EncodingLayout{
      nimble::EncodingType::FixedBitWidth,
      {},
      nimble::CompressionType::Uncompressed};
}

nimble::EncodingLayout trivialLayout() {
  return nimble::EncodingLayout{
      nimble::EncodingType::Trivial, {}, nimble::CompressionType::Uncompressed};
}

nimble::EncodingLayout varintLayout() {
  return nimble::EncodingLayout{
      nimble::EncodingType::Varint, {}, nimble::CompressionType::Uncompressed};
}

nimble::EncodingLayout alpWithFixedBitWidthPayloadLayout() {
  return nimble::EncodingLayout{
      nimble::EncodingType::ALP,
      {},
      nimble::CompressionType::Uncompressed,
      {fixedBitWidthLayout(), varintLayout(), trivialLayout()}};
}

nimble::EncodingLayout alpWithFixedBitWidthExceptionStreamsLayout() {
  return nimble::EncodingLayout{
      nimble::EncodingType::ALP,
      {},
      nimble::CompressionType::Uncompressed,
      {fixedBitWidthLayout(), fixedBitWidthLayout(), fixedBitWidthLayout()}};
}

nimble::EncodingLayout dictionaryWithAlpAlphabetLayout() {
  return nimble::EncodingLayout{
      nimble::EncodingType::Dictionary,
      {},
      nimble::CompressionType::Uncompressed,
      {alpWithFixedBitWidthPayloadLayout(), fixedBitWidthLayout()}};
}

nimble::EncodingSelectionPolicyCreator unusedNestedPolicyCreator() {
  return [](nimble::DataType) {
    return std::unique_ptr<nimble::EncodingSelectionPolicyBase>{};
  };
}

// On a dataset engineered to expose the count-vs-size gap -- 980 clean
// two-decimal values (exactly representable at a low exponent with a tiny
// integer domain) plus 20 four-decimal, large-magnitude outliers (only
// exactly representable at a higher exponent, and whose scaled integers span
// many bits) -- the size-based estimator's winning score must never lose to
// what the count-based winner would score at. That is the property
// `findBestExponentFactorBySize` actually enforces (it iterates every (e, f)
// and picks the smallest estimated bytes).
//
// The looser property "actual encoded bytes of size-based choice < actual
// bytes of count-based choice" only holds once the exception-placeholder
// change lands so that estimator bytes track real bytes on all tie-break
// candidates. Today the estimator can tie two (e, f)s that produce different
// real bytes -- notably for float, where precision loss at the (e=17, f=15)
// tie-break winner inflates the real bit-width beyond what the estimator
// saw. We check actual-bytes reduction as a soft signal (LE on float, LT on
// double), and gate the strict assertion on double.
// TODO(alp): tighten to EXPECT_LT for float once encode() writes a non-zero
// placeholder for exception slots (see scoreCombination header comment).
TYPED_TEST(ALPEncodingTest, sizeBeatsCountOnHighRangeOutliers) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // 980 clean two-decimal values in [0, 100): exactly representable at e=2
  // with a tiny integer domain. 20 four-decimal, large-magnitude values: only
  // exactly representable at e>=4, and their scaled integers span >=30 bits.
  constexpr uint32_t kCleanCount = 980;
  constexpr uint32_t kOutlierCount = 20;
  nimble::Vector<D> values{this->pool_.get()};
  values.reserve(kCleanCount + kOutlierCount);
  for (uint32_t i = 0; i < kCleanCount; ++i) {
    values.push_back(static_cast<D>(i % 10'000) / static_cast<D>(100));
  }
  for (uint32_t i = 0; i < kOutlierCount; ++i) {
    values.push_back(
        static_cast<D>(987'000) + static_cast<D>(i) + static_cast<D>(0.4321));
  }

  const std::span<const D> logical{values.data(), values.size()};

  const auto [countExp, countFactor] =
      nimble::ALPEncoding<D>::findBestExponentFactorByCount(logical);
  const auto [sizeExp, sizeFactor] =
      nimble::ALPEncoding<D>::findBestExponentFactorBySize(logical, options);

  // Core guarantee at the layer this changes: the size-based winner's
  // *estimated* bytes must be <= the count-based winner's estimated bytes.
  // This is what `findBestExponentFactorBySize` optimizes for.
  const auto countScore = nimble::ALPEncoding<D>::scoreCombination(
      logical, countExp, countFactor, options);
  const auto sizeScore = nimble::ALPEncoding<D>::scoreCombination(
      logical, sizeExp, sizeFactor, options);
  EXPECT_LE(sizeScore.estimatedBytes, countScore.estimatedBytes)
      << "count (" << int(countExp) << "," << int(countFactor)
      << ") est=" << countScore.estimatedBytes << "; size (" << int(sizeExp)
      << "," << int(sizeFactor) << ") est=" << sizeScore.estimatedBytes;

  // Sanity-log actual encoded bytes with `realNestedSelection=true` so the
  // nested uint64 stream is picked by the real cost-based factory (matches
  // what production selection would emit). Under the default test policy the
  // nested stream is forced to Trivial and the FOR bit-width advantage the
  // size-based scorer weighs is invisible.
  nimble::Buffer countBuffer{*this->pool_};
  nimble::Buffer sizeBuffer{*this->pool_};
  const auto countEncoded =
      nimble::test::Encoder<nimble::ALPEncoding<D>>::encodeWithExponentFactor(
          countBuffer,
          values,
          countExp,
          countFactor,
          nimble::CompressionType::Uncompressed,
          options,
          /*realNestedSelection=*/true);
  const auto sizeEncoded =
      nimble::test::Encoder<nimble::ALPEncoding<D>>::encodeWithExponentFactor(
          sizeBuffer,
          values,
          sizeExp,
          sizeFactor,
          nimble::CompressionType::Uncompressed,
          options,
          /*realNestedSelection=*/true);

  LOG(INFO) << "ALP selection A/B: count (" << int(countExp) << ","
            << int(countFactor) << ") est=" << countScore.estimatedBytes
            << " real=" << countEncoded.size() << "; size (" << int(sizeExp)
            << "," << int(sizeFactor) << ") est=" << sizeScore.estimatedBytes
            << " real=" << sizeEncoded.size();

  // With the exception placeholder in place (encode() writes the first
  // representable value's ZigZag into exception slots instead of 0), the
  // estimator's bit-width model matches encode time, so the estimate no longer
  // over-counts and selection is trustworthy. For double the size-based choice
  // is strictly smaller. For float, count-based selection already lands on a
  // byte-optimal (e, f) for this data and size-based selection ties along the
  // precision diagonal (e.g. (2,0) vs (10,8) both at 2001 real bytes), so the
  // guarantee here is no-regression rather than strict improvement.
  if constexpr (std::is_same_v<D, double>) {
    EXPECT_LT(sizeEncoded.size(), countEncoded.size());
  } else {
    EXPECT_LE(sizeEncoded.size(), countEncoded.size());
  }

  // Sanity: the size-based encoding must still round-trip losslessly.
  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buf = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buf->template asMutable<void>();
  };
  auto encoding = nimble::test::Encoder<nimble::ALPEncoding<D>>::createEncoding(
      *this->buffer_,
      values,
      stringBufferFactory,
      nimble::CompressionType::Uncompressed,
      options);
  nimble::Vector<D> result{this->pool_.get(), values.size()};
  encoding->materialize(values.size(), result.data());
  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]));
  }
}

// When every value is exactly representable at a small (exponent, factor), the
// size-based strategy must not regress on encoded byte count relative to
// count-based selection. (The two strategies do NOT necessarily return the
// same (e, f) on such data: any (e, f) with the same `e - f` produces the
// same scaled integers and therefore ties in bytes, so size-based selection
// legitimately drifts along the diagonal to the largest (e, f) — DuckDB's
// tie-break rule for preserving out-of-sample precision. What must hold is
// that the encoded output is byte-equal, i.e., no regression on "easy" data.)
TYPED_TEST(ALPEncodingTest, sizeSelectionNoRegressionOnExactlyRepresentable) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Two-decimal values in [0, 100): count-based picks (e=2, f=0), size-based
  // may pick any (e, f) with e - f == 2. All such combinations produce the
  // same integer stream and therefore the same encoded byte count.
  nimble::Vector<D> values{this->pool_.get()};
  values.reserve(512);
  for (uint32_t i = 0; i < 512; ++i) {
    values.push_back(static_cast<D>(i % 10'000) / static_cast<D>(100));
  }
  const std::span<const D> logical{values.data(), values.size()};

  const auto [countExp, countFactor] =
      nimble::ALPEncoding<D>::findBestExponentFactorByCount(logical);
  const auto [sizeExp, sizeFactor] =
      nimble::ALPEncoding<D>::findBestExponentFactorBySize(logical, options);

  nimble::Buffer countBuffer{*this->pool_};
  nimble::Buffer sizeBuffer{*this->pool_};
  const auto countEncoded =
      nimble::test::Encoder<nimble::ALPEncoding<D>>::encodeWithExponentFactor(
          countBuffer,
          values,
          countExp,
          countFactor,
          nimble::CompressionType::Uncompressed,
          options,
          /*realNestedSelection=*/true);
  const auto sizeEncoded =
      nimble::test::Encoder<nimble::ALPEncoding<D>>::encodeWithExponentFactor(
          sizeBuffer,
          values,
          sizeExp,
          sizeFactor,
          nimble::CompressionType::Uncompressed,
          options,
          /*realNestedSelection=*/true);
  EXPECT_LE(sizeEncoded.size(), countEncoded.size());
}

// DuckDB's tie-break rule: on equal estimated bytes, prefer the LARGER
// (exponent, factor) so more decimal precision is preserved for values not
// seen in the sample. `findBestExponentFactorBySize` iterates (e, f) in
// ascending order and uses `<=` on the score, so the last-scored equal
// candidate wins — the largest (e, f). This test picks a constant value that
// is exactly representable at every (e, f) considered and produces the same
// tiny integer domain regardless (both the FOR bit width and the exception
// count are constant), so every candidate ties in bytes.
TYPED_TEST(ALPEncodingTest, sizeSelectionPrefersLargerExponentOnTie) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Value 0 encodes to the same integer 0 at every (e, f), so zigZagMin ==
  // zigZagMax == 0, exceptionCount == 0, and estimatedBytes is identical for
  // every candidate. The tie-break rule picks the largest (e, f).
  nimble::Vector<D> values{this->pool_.get(), 64};
  values.fill(D{0});
  const std::span<const D> logical{values.data(), values.size()};

  const auto [exp, factor] =
      nimble::ALPEncoding<D>::findBestExponentFactorBySize(logical, options);
  // The loop upper bound is the per-type `kMaxExponent` (float=10, double=18,
  // matching DuckDB's caps), and factor is bounded by exponent, so the
  // largest-(e,f) winner is (kMaxExponent, kMaxExponent).
  constexpr int kExpectedMax = std::is_same_v<D, float> ? 10 : 18;
  EXPECT_EQ(exp, kExpectedMax);
  EXPECT_EQ(factor, kExpectedMax);
}

// White-box coverage for `scoreCombination`: a hand-computed sample whose
// expected `estimatedBytes` matches the closed-form
//   min(FixedBitWidthEncoding<uint64>::estimateSize(sampleSize, zzMin, zzMax,
//       options),
//       TrivialEncoding<uint64>::estimateSize(sampleSize))
//   + exceptionCount * (sizeof(uint32_t) + sizeof(physicalType))
// and a separate sample where every value is an exception, asserting
// `kUnusableScore`.
TYPED_TEST(ALPEncodingTest, scoreCombinationBytesMatchClosedForm) {
  using D = typename TypeParam::data_type;
  using physicalType = typename nimble::TypeTraits<D>::physicalType;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // All values exactly representable at (e=2, f=0). Their ZigZag-encoded
  // integers span the small range [0, 200], so bitWidth(200 - 0) = 8. With
  // fixedBitWidthUseExactBits = false (the default), the width is rounded up
  // to a byte — here 8 bits already. Zero exceptions → exceptionBytes == 0.
  constexpr uint32_t kSize = 8;
  const std::array<D, kSize> raw{
      D{0.00}, D{0.01}, D{0.50}, D{0.99}, D{1.00}, D{1.25}, D{1.50}, D{2.00}};
  std::span<const D> sample{raw.data(), raw.size()};

  const auto score = nimble::ALPEncoding<D>::scoreCombination(
      sample, /*e=*/2, /*f=*/0, options);
  EXPECT_EQ(score.exceptionCount, 0u);

  // Hand-compute the expected byte count. ZigZag(x) for non-negative x is 2*x,
  // so zzMin = ZigZag(0) = 0 and zzMax = ZigZag(200) = 400. bitsRequired(400)
  // = 9, rounded up to 16 (two bytes) when fixedBitWidthUseExactBits is off.
  const uint64_t zzMin = 0;
  const uint64_t zzMax = 400;
  const uint64_t expectedFixed =
      nimble::FixedBitWidthEncoding<uint64_t>::estimateSize(
          kSize, zzMin, zzMax, options);
  const uint64_t expectedTrivial =
      nimble::TrivialEncoding<uint64_t>::estimateSize(kSize);
  const uint64_t expectedBytes = std::min(expectedFixed, expectedTrivial);
  EXPECT_EQ(score.estimatedBytes, expectedBytes);

  // Sanity: `exceptionCount * (sizeof(uint32_t) + sizeof(physicalType))`
  // contributes 0 when exceptionCount is 0. Guard against accidental sign or
  // offset drift by asserting the exception term explicitly.
  EXPECT_EQ(
      score.estimatedBytes - expectedBytes,
      0u * (sizeof(uint32_t) + sizeof(physicalType)));
}

TYPED_TEST(ALPEncodingTest, scoreCombinationUnusableWhenAllExceptions) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // At (e=0, f=0), the encoded integer equals llround(v). Non-integer values
  // like 0.5 round to 1, restore to 1.0, which does not equal 0.5 — every
  // element is an exception. representableCount == 0 → kUnusableScore.
  const std::array<D, 4> raw{D{0.5}, D{1.5}, D{2.5}, D{3.5}};
  std::span<const D> sample{raw.data(), raw.size()};

  const auto score = nimble::ALPEncoding<D>::scoreCombination(
      sample, /*e=*/0, /*f=*/0, options);
  EXPECT_EQ(score.exceptionCount, sample.size());
  EXPECT_EQ(score.estimatedBytes, nimble::ALPEncoding<D>::kUnusableScore);
}

// After the estimator unification lands, `estimateSizeFromSample` no longer
// rebuilds its own ZigZag min/max on the encoded stream -- it takes them
// directly from `scoreCombination`'s return. That coupling only stays
// byte-exact if the two paths size their integer stream with the same
// `(min, max)` inputs. This test asserts that byte-exact contract on the
// chosen `(e, f)` for an all-representable sample (zero exceptions), so the
// exception-payload branch drops out and the integer-stream subtotal alone
// drives the estimate:
//
//   estimateSizeFromSample(rowCount=sampleSize) == prefix + metadata
//     + min(FBW::estimateSize(N, zzMin, zzMax), Trivial::estimateSize(N))
//
// with zzMin/zzMax coming from scoreCombination(sample, e, f).zigZagMin/Max
// on the chosen (e, f). If a future refactor drifts the two, this test
// fails on the exact byte before any benchmark regression shows up.
TYPED_TEST(ALPEncodingTest, scoreCombinationMatchesEstimator) {
  using D = typename TypeParam::data_type;
  using PhysicalType = typename nimble::TypeTraits<D>::physicalType;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // All two-decimal values -- exactly representable at (e=2, f=0). The
  // size-based selector will pick some (e, f) that yields zero exceptions;
  // we don't hard-code which pair it picks, only that the sample has that
  // property so the shared code path is exercised end-to-end.
  const std::array<D, 8> raw{
      D{0.00}, D{0.01}, D{0.50}, D{0.99}, D{1.00}, D{1.25}, D{1.50}, D{2.00}};
  std::vector<PhysicalType> sampledValues;
  sampledValues.reserve(raw.size());
  for (const auto value : raw) {
    sampledValues.push_back(nimble::detail::alp::toPhysical<D>(value));
  }
  const std::span<const PhysicalType> physicalSpan{
      sampledValues.data(), sampledValues.size()};

  // Setting rowCount == sampleSize eliminates the estimator's rowCount
  // scale-up so byte totals directly compare.
  const uint64_t rowCount = sampledValues.size();
  const auto estimate = nimble::ALPEncoding<D>::estimateSizeFromSample(
      rowCount, physicalSpan, options);
  ASSERT_TRUE(estimate.has_value());

  // Rebuild the estimator's integer-stream expectation from scoreCombination's
  // ZigZag range on the (e, f) the selector chose.
  std::vector<D> logicalValues;
  logicalValues.reserve(sampledValues.size());
  for (const auto v : sampledValues) {
    logicalValues.push_back(nimble::detail::alp::toLogical<D>(v));
  }
  const std::span<const D> logicalSpan{
      logicalValues.data(), logicalValues.size()};
  const auto [chosenE, chosenF] =
      nimble::ALPEncoding<D>::findBestExponentFactorBySize(
          logicalSpan, options);
  const auto score = nimble::ALPEncoding<D>::scoreCombination(
      logicalSpan, chosenE, chosenF, options);
  ASSERT_NE(score.estimatedBytes, nimble::ALPEncoding<D>::kUnusableScore);
  ASSERT_EQ(score.exceptionCount, 0u)
      << "Test setup expects zero exceptions on chosen (e, f).";

  const uint64_t integerStreamBytes = std::min(
      nimble::FixedBitWidthEncoding<uint64_t>::estimateSize(
          rowCount, score.zigZagMin, score.zigZagMax, options),
      nimble::TrivialEncoding<uint64_t>::estimateSize(rowCount));

  // exceptionCount == 0 → no exception-count varint, no per-stream varints,
  // no exception-payload bytes. Just prefix + header + int-stream size varint
  // + integer stream.
  constexpr uint64_t kHeaderSize = 3;
  const uint64_t metadataSize =
      kHeaderSize + nimble::varint::varintSize(integerStreamBytes);
  const uint64_t expected =
      nimble::EncodingPrefix::serializedSize(
          static_cast<uint32_t>(rowCount), options.useVarintRowCount) +
      metadataSize + integerStreamBytes;
  EXPECT_EQ(*estimate, expected);
}

// batchTransform (xsimd) must produce lane-by-lane byte-identical
// output to the scalar path (scalarTransformOne). This is the correctness
// contract that lets scoreCombination and encodeWithExponentFactor route
// through the vectorized helper without changing encoded bytes.
//
// Covers:
//   * pathological finite inputs (0, +/-0, denormals, huge, tiny)
//   * non-finite inputs (NaN, +/-Inf)
//   * out-of-int64-range scaled values
//   * halfway cases where round-half-away-from-zero rules matter
//   * randomized fuzz over a moderately-sized sample
// for every (exponent, factor) pair in the search grid.
TYPED_TEST(ALPEncodingTest, batchTransformMatchesScalar) {
  using D = typename TypeParam::data_type;
  using PhysicalType = typename nimble::TypeTraits<D>::physicalType;
  using Alp = nimble::ALPEncoding<D>;

  // Pathological + edge-case inputs. Chosen so at least one lane in each
  // batch tickles: NaN, +/-Inf, +/-0, subnormal, huge, tiny, and negatives
  // of half-values. Padded to a multiple of kBatchSize so we cover the
  // full-batch path (the scalar tail is separately covered by the fuzz
  // block below).
  std::vector<D> edge{
      D{0.0},
      -D{0.0},
      D{0.5},
      -D{0.5},
      D{1.25},
      -D{1.25},
      D{2.5},
      -D{2.5},
      D{1e-6},
      -D{1e-6},
      std::numeric_limits<D>::min(),
      std::numeric_limits<D>::denorm_min(),
      D{1e6},
      -D{1e6},
      D{1.234567},
      -D{7.654321},
      std::numeric_limits<D>::infinity(),
      -std::numeric_limits<D>::infinity(),
      std::numeric_limits<D>::quiet_NaN(),
      -std::numeric_limits<D>::quiet_NaN(),
      D{9.2233720368547758e18}, // ~int64::max as double
      -D{9.2233720368547758e18},
      D{1e30}, // overflows int64 after any positive exponent
      -D{1e30},
  };
  // Pad to a multiple of kBatchSize by repeating a benign representable value.
  while (edge.size() % Alp::kBatchSize != 0) {
    edge.push_back(D{1.0});
  }

  // Randomized fuzz block: 4096 samples across [-1e6, 1e6], mostly two-decimal
  // to keep exception counts realistic. Same seed for reproducibility.
  std::mt19937_64 rng(0xA1FDA1FD01D3B0FDULL);
  std::uniform_int_distribution<int64_t> centDist(-100'000'000, 100'000'000);
  std::vector<D> fuzz;
  fuzz.reserve(4096);
  for (int i = 0; i < 4096; ++i) {
    fuzz.push_back(static_cast<D>(centDist(rng)) / static_cast<D>(100));
  }

  auto physicalOf = [](D v) { return nimble::detail::alp::toPhysical<D>(v); };

  auto checkSpan = [&](const std::vector<D>& logicals, int e, int f) {
    std::vector<PhysicalType> physicals;
    physicals.reserve(logicals.size());
    for (const auto v : logicals) {
      physicals.push_back(physicalOf(v));
    }
    const double expMul = Alp::kPow10Double[e];
    const double facMul = Alp::kPow10Double[f];

    const std::size_t n = logicals.size();
    const std::size_t batches = n / Alp::kBatchSize;
    for (std::size_t b = 0; b < batches; ++b) {
      const std::size_t base = b * Alp::kBatchSize;
      std::array<uint64_t, 64> batchZigZag{}; // upper-bounded by any real
                                              // kBatchSize the build produces
      std::array<bool, 64> batchOk{};
      Alp::batchTransform(
          logicals.data() + base,
          physicals.data() + base,
          expMul,
          facMul,
          batchZigZag.data(),
          batchOk.data());
      for (std::size_t k = 0; k < Alp::kBatchSize; ++k) {
        uint64_t scalarZigZag = 0;
        const bool scalarOk = Alp::scalarTransformOne(
            logicals[base + k],
            physicals[base + k],
            expMul,
            facMul,
            scalarZigZag);
        EXPECT_EQ(batchOk[k], scalarOk)
            << "mask mismatch at lane " << (base + k) << " (e=" << e
            << ", f=" << f << ", value=" << +logicals[base + k] << ")";
        if (scalarOk) {
          EXPECT_EQ(batchZigZag[k], scalarZigZag)
              << "zigzag mismatch at lane " << (base + k) << " (e=" << e
              << ", f=" << f << ", value=" << +logicals[base + k] << ")";
        }
      }
    }
  };

  // Enumerate the full (e, f) grid used by findBestExponentFactorBySize.
  // Only combinations with f <= e are considered by production selection.
  constexpr int kMaxE = std::is_same_v<D, float> ? 10 : 18;
  for (int e = 0; e <= kMaxE; ++e) {
    for (int f = 0; f <= e; ++f) {
      checkSpan(edge, e, f);
      checkSpan(fuzz, e, f);
    }
  }
}

TEST(ALPSizeEstimationTest, invalidSampleRejected) {
  const std::vector<uint32_t> sample = {0, 1};

  NIMBLE_ASSERT_THROW(
      nimble::ALPEncoding<float>::estimateSizeFromSample(
          /*rowCount=*/0, std::span<const uint32_t>{sample.data(), 1}),
      "ALP estimation requires non-empty input.");
  NIMBLE_ASSERT_THROW(
      nimble::ALPEncoding<float>::estimateSizeFromSample(
          /*rowCount=*/1, std::span<const uint32_t>{}),
      "ALP estimation requires a non-empty sample.");
  NIMBLE_ASSERT_THROW(
      nimble::ALPEncoding<float>::estimateSizeFromSample(
          /*rowCount=*/1, sample),
      "ALP sample size cannot exceed the input row count.");
}

template <typename D>
void expectEstimateUsesPackedExceptionValues() {
  using PhysicalType = typename nimble::TypeTraits<D>::physicalType;
  constexpr uint32_t kRowCount{256};
  const nimble::Encoding::Options options{.fixedBitWidthUseExactBits = true};

  const PhysicalType exceptionValue =
      nimble::detail::alp::toPhysical<D>(std::numeric_limits<D>::infinity());
  std::vector<PhysicalType> values(kRowCount, exceptionValue);

  const auto estimate = nimble::ALPEncoding<D>::estimateSizeFromSample(
      kRowCount,
      std::span<const PhysicalType>{values.data(), values.size()},
      options);
  ASSERT_TRUE(estimate.has_value());

  std::vector<uint64_t> encodedValues(kRowCount, 0);
  const auto encodedStats = nimble::Statistics<uint64_t>::create(
      std::span<const uint64_t>{encodedValues.data(), encodedValues.size()});
  const auto encodedValuesSize = std::min(
      nimble::FixedBitWidthEncoding<uint64_t>::estimateSize(
          kRowCount, encodedStats, options),
      nimble::TrivialEncoding<uint64_t>::estimateSize(kRowCount));

  std::vector<uint32_t> exceptionPositions;
  exceptionPositions.reserve(kRowCount);
  for (uint32_t i = 0; i < kRowCount; ++i) {
    exceptionPositions.push_back(i);
  }
  const auto positionStats = nimble::Statistics<uint32_t>::create(
      std::span<const uint32_t>{
          exceptionPositions.data(), exceptionPositions.size()});
  const auto exceptionPositionsSize = std::min(
      nimble::TrivialEncoding<uint32_t>::estimateSize(kRowCount),
      nimble::FixedBitWidthEncoding<uint32_t>::estimateSize(
          kRowCount, positionStats, options));

  const auto trivialExceptionValuesSize =
      nimble::TrivialEncoding<PhysicalType>::estimateSize(kRowCount);
  const uint64_t trivialExceptionValuesBound =
      nimble::EncodingPrefix::serializedSize(
          kRowCount, options.useVarintRowCount) +
      3 + nimble::varint::varintSize(kRowCount) +
      nimble::varint::varintSize(encodedValuesSize) +
      nimble::varint::varintSize(exceptionPositionsSize) +
      nimble::varint::varintSize(trivialExceptionValuesSize) +
      encodedValuesSize + exceptionPositionsSize + trivialExceptionValuesSize;

  EXPECT_LT(estimate.value(), trivialExceptionValuesBound);
}

TEST(ALPSizeEstimationTest, packsExceptionValuesWhenEstimating) {
  expectEstimateUsesPackedExceptionValues<float>();
  expectEstimateUsesPackedExceptionValues<double>();
}

TEST(ALPEncodingHeaderTest, compactControlWordRoundTrip) {
  std::array<char, 3> serialized{};
  char* writePosition = serialized.data();
  nimble::detail::alp::writeHeader(
      nimble::detail::alp::Header{
          .exponent = 23, .factor = 17, .hasExceptions = true},
      writePosition);

  const std::array<char, 3> expected = {
      static_cast<char>(0x37),
      static_cast<char>(0x02),
      static_cast<char>(0x01)};
  EXPECT_EQ(serialized, expected);
  EXPECT_EQ(writePosition, serialized.data() + serialized.size());

  const char* readPosition = serialized.data();
  const auto header = nimble::detail::alp::readHeader(readPosition);
  EXPECT_EQ(header.exponent, 23);
  EXPECT_EQ(header.factor, 17);
  EXPECT_TRUE(header.hasExceptions);
  EXPECT_EQ(readPosition, serialized.data() + serialized.size());
}

template <typename D>
nimble::Vector<D> makeRandomDecimalValues(
    velox::memory::MemoryPool* pool,
    uint32_t rowCount,
    uint32_t uniqueCount,
    uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int32_t> valueDistribution{-5000, 5000};
  std::uniform_int_distribution<uint32_t> indexDistribution{0, uniqueCount - 1};

  std::vector<D> dictionary;
  dictionary.reserve(uniqueCount);
  for (uint32_t i = 0; i < uniqueCount; ++i) {
    dictionary.push_back(
        static_cast<D>(valueDistribution(rng)) / static_cast<D>(100));
  }

  nimble::Vector<D> values{pool};
  values.reserve(rowCount);
  for (uint32_t i = 0; i < rowCount; ++i) {
    values.push_back(dictionary[indexDistribution(rng)]);
  }
  return values;
}

void expectEncodingLayout(
    std::string_view serialized,
    const std::vector<
        std::tuple<nimble::EncodingType, nimble::DataType, std::string>>&
        expected) {
  std::vector<std::tuple<nimble::EncodingType, nimble::DataType, std::string>>
      actual;
  nimble::tools::traverseEncodings(
      serialized,
      [&](auto encodingType,
          auto dataType,
          auto /* level */,
          auto /* index */,
          auto nestedEncodingName,
          std::unordered_map<
              nimble::tools::EncodingPropertyType,
              nimble::tools::EncodingProperty> /* properties */) {
        actual.emplace_back(encodingType, dataType, nestedEncodingName);
        return true;
      });

  EXPECT_EQ(actual, expected);
}

std::unique_ptr<nimble::Encoding> createEncoding(
    velox::memory::MemoryPool* pool,
    std::string_view serialized,
    const nimble::Encoding::Options& options,
    std::vector<velox::BufferPtr>& stringBuffers) {
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& stringBuffer = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, pool));
    return stringBuffer->template asMutable<void>();
  };
  return nimble::EncodingFactory().create(
      *pool, serialized, stringBufferFactory, options);
}

template <typename D>
std::string_view encodeWithLayout(
    nimble::Buffer& buffer,
    const nimble::Vector<D>& values,
    nimble::EncodingLayout layout,
    const nimble::Encoding::Options& options) {
  auto policy = std::make_unique<nimble::ReplayedEncodingSelectionPolicy<D>>(
      std::move(layout), std::nullopt, unusedNestedPolicyCreator());
  return nimble::EncodingFactory::encode<D>(
      std::move(policy),
      std::span<const D>{values.data(), values.size()},
      buffer,
      options);
}

template <typename D>
void expectInterleavedMaterializeAndSkip(
    nimble::Encoding& encoding,
    const nimble::Vector<D>& values,
    velox::memory::MemoryPool* pool,
    uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> skipDistribution{0, 13};
  std::uniform_int_distribution<uint32_t> readDistribution{1, 17};

  nimble::Vector<D> output{pool};
  uint32_t cursor{0};
  while (cursor < values.size()) {
    const auto remainingRows = static_cast<uint32_t>(values.size() - cursor);
    const auto rowsToSkip = std::min(skipDistribution(rng), remainingRows);
    encoding.skip(rowsToSkip);
    cursor += rowsToSkip;

    if (cursor == values.size()) {
      break;
    }

    const auto rowsToRead = std::min(
        readDistribution(rng), static_cast<uint32_t>(values.size() - cursor));
    output.resize(rowsToRead);
    encoding.materialize(rowsToRead, output.data());

    for (uint32_t i = 0; i < rowsToRead; ++i) {
      SCOPED_TRACE(fmt::format("cursor={} i={}", cursor, i));
      EXPECT_TRUE(
          nimble::NimbleCompare<D>::equals(output[i], values[cursor + i]));
    }
    cursor += rowsToRead;
  }

  std::uniform_int_distribution<uint32_t> targetDistribution{
      0, static_cast<uint32_t>(values.size() - 1)};
  for (uint32_t attempt = 0; attempt < 32; ++attempt) {
    const auto target = targetDistribution(rng);
    const auto rowsToRead = std::min(
        readDistribution(rng), static_cast<uint32_t>(values.size() - target));
    encoding.reset();
    encoding.skip(target);
    output.resize(rowsToRead);
    encoding.materialize(rowsToRead, output.data());

    for (uint32_t i = 0; i < rowsToRead; ++i) {
      SCOPED_TRACE(fmt::format("target={} i={}", target, i));
      EXPECT_TRUE(
          nimble::NimbleCompare<D>::equals(output[i], values[target + i]));
    }
  }
}

TYPED_TEST(ALPEncodingTest, roundTrip) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->template toVector<D>({1, 2, 3, 4, 5});

  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buf = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buf->template asMutable<void>();
  };

  auto encoding = nimble::test::Encoder<nimble::ALPEncoding<D>>::createEncoding(
      *this->buffer_,
      values,
      stringBufferFactory,
      nimble::CompressionType::Uncompressed,
      options);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::ALP);
  EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<D>::dataType);
  EXPECT_EQ(encoding->rowCount(), values.size());

  nimble::Vector<D> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());

  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]));
  }
}

TYPED_TEST(ALPEncodingTest, headerMetadataUsesVarints) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint,
      .fixedBitWidthUseExactBits = true};

  nimble::Vector<D> values{this->pool_.get(), 200};
  values.fill(D{1.25});
  values[150] = std::numeric_limits<D>::infinity();
  const auto serialized = encodeWithLayout<D>(
      *this->buffer_, values, alpWithFixedBitWidthPayloadLayout(), options);

  const char* pos = serialized.data() +
      nimble::EncodingPrefix::prefixSize(serialized, options.useVarintRowCount);
  const auto header = nimble::detail::alp::readHeader(pos);
  EXPECT_TRUE(header.hasExceptions);

  const auto* exceptionCountStart = pos;
  EXPECT_EQ(nimble::varint::readVarint32(&pos), 1);
  EXPECT_EQ(pos - exceptionCountStart, 1);

  const auto* encodedValuesSizeStart = pos;
  const auto encodedValuesSize = nimble::varint::readVarint32(&pos);
  EXPECT_EQ(
      pos - encodedValuesSizeStart,
      nimble::varint::varintSize(encodedValuesSize));
  pos += encodedValuesSize;

  const auto exceptionPositionsSize = nimble::varint::readVarint32(&pos);
  const auto exceptionPositions = std::string_view{pos, exceptionPositionsSize};
  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(exceptionPositions),
      nimble::EncodingType::Varint);
  pos += exceptionPositionsSize;

  const auto exceptionValuesSize = nimble::varint::readVarint32(&pos);
  const auto exceptionValues = std::string_view{pos, exceptionValuesSize};
  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(exceptionValues),
      nimble::EncodingType::Trivial);

  std::vector<velox::BufferPtr> stringBuffers;
  auto encoding =
      createEncoding(this->pool_.get(), serialized, options, stringBuffers);
  nimble::Vector<D> result{this->pool_.get(), values.size()};
  encoding->materialize(values.size(), result.data());
  for (uint32_t i = 0; i < values.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]));
  }
}

TYPED_TEST(ALPEncodingTest, traverseEncodingsVisitsExceptionStreams) {
  using D = typename TypeParam::data_type;
  using PhysicalType = typename nimble::TypeTraits<D>::physicalType;
  const nimble::Encoding::Options options{
      .useVarintRowCount = false, .fixedBitWidthUseExactBits = true};

  nimble::Vector<D> values{this->pool_.get(), 200};
  values.fill(D{1.25});
  values[150] = std::numeric_limits<D>::infinity();
  const auto serialized = encodeWithLayout<D>(
      *this->buffer_, values, alpWithFixedBitWidthPayloadLayout(), options);

  expectEncodingLayout(
      serialized,
      {
          {nimble::EncodingType::ALP, nimble::TypeTraits<D>::dataType, ""},
          {nimble::EncodingType::FixedBitWidth,
           nimble::DataType::Uint64,
           "EncodedValues"},
          {nimble::EncodingType::Varint,
           nimble::DataType::Uint32,
           "ExceptionPositions"},
          {nimble::EncodingType::Trivial,
           nimble::TypeTraits<PhysicalType>::dataType,
           "ExceptionValues"},
      });
}

TYPED_TEST(ALPEncodingTest, slice) {
  using DataType = typename TypeParam::data_type;
  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint,
      .fixedBitWidthUseExactBits = true};

  nimble::Vector<DataType> values{this->pool_.get()};
  for (uint32_t i = 0; i < 128; ++i) {
    values.push_back(
        static_cast<DataType>((i % 37) - 18) / static_cast<DataType>(10));
  }

  const auto serialized = encodeWithLayout<DataType>(
      *this->buffer_, values, alpWithFixedBitWidthPayloadLayout(), options);

  for (const auto range :
       {Range{/*offset=*/0, /*length=*/11},
        Range{/*offset=*/7, /*length=*/19},
        Range{/*offset=*/64, /*length=*/32},
        Range{/*offset=*/120, /*length=*/8}}) {
    SCOPED_TRACE(
        testing::Message() << "offset=" << range.offset
                           << " length=" << range.length);
    nimble::Buffer sliceBuffer{*this->pool_};
    const auto sliced = nimble::EncodingFactory::slice(
        serialized, range.offset, range.length, sliceBuffer, options);

    EXPECT_EQ(
        nimble::EncodingPrefix::encodingType(sliced),
        nimble::EncodingType::ALP);
    EXPECT_EQ(
        nimble::EncodingPrefix::readRowCount(sliced, options.useVarintRowCount),
        range.length);

    std::vector<velox::BufferPtr> stringBuffers;
    auto encoding =
        createEncoding(this->pool_.get(), sliced, options, stringBuffers);
    nimble::Vector<DataType> result{this->pool_.get(), range.length};
    encoding->materialize(range.length, result.data());
    for (uint32_t i = 0; i < range.length; ++i) {
      SCOPED_TRACE(fmt::format("i={}", i));
      EXPECT_TRUE(
          nimble::NimbleCompare<DataType>::equals(
              result[i], values[range.offset + i]));
    }
  }
}

TYPED_TEST(ALPEncodingTest, rejectsZeroLengthSlice) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint,
      .fixedBitWidthUseExactBits = true};

  const auto values = this->template toVector<DataType>({1, 2, 3});
  const auto serialized = encodeWithLayout<DataType>(
      *this->buffer_, values, alpWithFixedBitWidthPayloadLayout(), options);

  nimble::Buffer sliceBuffer{*this->pool_};
  NIMBLE_ASSERT_THROW(
      nimble::EncodingFactory::slice(
          serialized,
          /*offset=*/1,
          /*length=*/0,
          sliceBuffer,
          options),
      "Cannot slice zero rows.");
}

TYPED_TEST(ALPEncodingTest, sliceRebasesExceptions) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint,
      .fixedBitWidthUseExactBits = true};

  nimble::Vector<DataType> values{this->pool_.get(), 128};
  values.fill(DataType{1.25});
  values[3] = std::numeric_limits<DataType>::infinity();
  values[17] = -std::numeric_limits<DataType>::infinity();
  values[31] = std::numeric_limits<DataType>::quiet_NaN();
  values[97] = std::numeric_limits<DataType>::infinity();

  const auto serialized = encodeWithLayout<DataType>(
      *this->buffer_, values, alpWithFixedBitWidthPayloadLayout(), options);

  constexpr uint32_t kOffset{16};
  constexpr uint32_t kLength{32};
  nimble::Buffer sliceBuffer{*this->pool_};
  const auto sliced = nimble::EncodingFactory::slice(
      serialized, kOffset, kLength, sliceBuffer, options);

  const char* pos = sliced.data() +
      nimble::EncodingPrefix::prefixSize(sliced, options.useVarintRowCount);
  const auto header = nimble::detail::alp::readHeader(pos);
  EXPECT_TRUE(header.hasExceptions);
  EXPECT_EQ(nimble::varint::readVarint32(&pos), 2);
  const auto encodedValuesSize = nimble::varint::readVarint32(&pos);
  pos += encodedValuesSize;

  const auto exceptionPositionsSize = nimble::varint::readVarint32(&pos);
  std::vector<velox::BufferPtr> positionStringBuffers;
  auto exceptionPositionsEncoding = createEncoding(
      this->pool_.get(),
      {pos, exceptionPositionsSize},
      options,
      positionStringBuffers);
  nimble::Vector<uint32_t> exceptionPositions{this->pool_.get(), 2};
  exceptionPositionsEncoding->materialize(2, exceptionPositions.data());
  // Source exceptions at rows 17 and 31 become slice-local rows 1 and 15.
  const std::vector<uint32_t> expectedPositions{1, 15};
  EXPECT_EQ(
      std::vector<uint32_t>(
          exceptionPositions.begin(), exceptionPositions.end()),
      expectedPositions);

  std::vector<velox::BufferPtr> stringBuffers;
  auto encoding =
      createEncoding(this->pool_.get(), sliced, options, stringBuffers);
  nimble::Vector<DataType> result{this->pool_.get(), kLength};
  encoding->materialize(kLength, result.data());
  for (uint32_t i = 0; i < kLength; ++i) {
    SCOPED_TRACE(fmt::format("i={}", i));
    EXPECT_TRUE(
        nimble::NimbleCompare<DataType>::equals(
            result[i], values[kOffset + i]));
  }
}

TYPED_TEST(ALPEncodingTest, slicePreservesExceptionStreamLayouts) {
  using DataType = typename TypeParam::data_type;
  using PhysicalType = typename nimble::TypeTraits<DataType>::physicalType;
  const nimble::Encoding::Options options{
      .useVarintRowCount = false, .fixedBitWidthUseExactBits = true};

  nimble::Vector<DataType> values{this->pool_.get(), 128};
  values.fill(DataType{1.25});
  values[17] = -std::numeric_limits<DataType>::infinity();
  values[31] = std::numeric_limits<DataType>::quiet_NaN();

  const auto serialized = encodeWithLayout<DataType>(
      *this->buffer_,
      values,
      alpWithFixedBitWidthExceptionStreamsLayout(),
      options);

  nimble::Buffer sliceBuffer{*this->pool_};
  const auto sliced = nimble::EncodingFactory::slice(
      serialized, /*offset=*/16, /*length=*/32, sliceBuffer, options);

  expectEncodingLayout(
      sliced,
      {
          {nimble::EncodingType::ALP,
           nimble::TypeTraits<DataType>::dataType,
           ""},
          {nimble::EncodingType::FixedBitWidth,
           nimble::DataType::Uint64,
           "EncodedValues"},
          {nimble::EncodingType::FixedBitWidth,
           nimble::DataType::Uint32,
           "ExceptionPositions"},
          {nimble::EncodingType::FixedBitWidth,
           nimble::TypeTraits<PhysicalType>::dataType,
           "ExceptionValues"},
      });

  std::vector<velox::BufferPtr> stringBuffers;
  auto encoding =
      createEncoding(this->pool_.get(), sliced, options, stringBuffers);
  nimble::Vector<DataType> result{this->pool_.get(), 32};
  encoding->materialize(32, result.data());
  for (uint32_t i = 0; i < result.size(); ++i) {
    SCOPED_TRACE(fmt::format("i={}", i));
    EXPECT_TRUE(
        nimble::NimbleCompare<DataType>::equals(result[i], values[16 + i]));
  }
}

TYPED_TEST(ALPEncodingTest, roundTripSignedDecimals) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values =
      this->template toVector<D>({-12.5, -1.25, -0.5, 0, 0.5, 1.25, 12.5});

  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buf = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buf->template asMutable<void>();
  };

  auto encoding = nimble::test::Encoder<nimble::ALPEncoding<D>>::createEncoding(
      *this->buffer_,
      values,
      stringBufferFactory,
      nimble::CompressionType::Uncompressed,
      options);

  nimble::Vector<D> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());

  for (uint32_t i = 0; i < values.size(); ++i) {
    SCOPED_TRACE(fmt::format("i={}", i));
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]));
  }
}

TYPED_TEST(ALPEncodingTest, manualSelectionUsesAlpEstimate) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = false, .fixedBitWidthUseExactBits = true};

  nimble::Vector<D> values{this->pool_.get()};
  values.reserve(2048);
  for (auto i = 0; i < 2048; ++i) {
    values.push_back(static_cast<D>((i % 129) - 64) / static_cast<D>(10));
  }

  auto policy = std::make_unique<nimble::ManualEncodingSelectionPolicy<D>>(
      std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::ALP, 1.0},
          {nimble::EncodingType::Trivial, 1.0},
          {nimble::EncodingType::FixedBitWidth, 1.0},
      },
      std::nullopt,
      std::nullopt);

  const auto serialized = nimble::EncodingFactory::encode<D>(
      std::move(policy),
      std::span<const D>{values.data(), values.size()},
      *this->buffer_,
      options);

  std::vector<std::tuple<nimble::EncodingType, nimble::DataType, std::string>>
      actual;
  nimble::tools::traverseEncodings(
      serialized,
      [&](auto encodingType,
          auto dataType,
          auto /* level */,
          auto /* index */,
          auto nestedEncodingName,
          std::unordered_map<
              nimble::tools::EncodingPropertyType,
              nimble::tools::EncodingProperty> /* properties */) {
        actual.emplace_back(encodingType, dataType, nestedEncodingName);
        return true;
      });

  const std::vector<
      std::tuple<nimble::EncodingType, nimble::DataType, std::string>>
      expected{
          {nimble::EncodingType::ALP, nimble::TypeTraits<D>::dataType, ""},
          {nimble::EncodingType::FixedBitWidth,
           nimble::DataType::Uint64,
           "EncodedValues"},
      };
  EXPECT_EQ(actual, expected);

  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buf = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buf->template asMutable<void>();
  };
  auto encoding = nimble::EncodingFactory().create(
      *this->pool_, serialized, stringBufferFactory, options);

  nimble::Vector<D> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());

  for (uint32_t i = 0; i < values.size(); ++i) {
    SCOPED_TRACE(fmt::format("i={}", i));
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]));
  }
}

TYPED_TEST(ALPEncodingTest, dictionaryAlphabetUsesNestedAlpWhenEnabled) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = false,
      .fixedBitWidthUseExactBits = true,
      .allowNestedAlpSelection = true};

  nimble::Vector<D> values{this->pool_.get()};
  for (auto i = 0; i < 256; ++i) {
    values.push_back(static_cast<D>((i % 17) - 8) / static_cast<D>(4));
  }

  auto policy = std::make_unique<nimble::ManualEncodingSelectionPolicy<D>>(
      std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::Dictionary, 1.0},
      },
      std::nullopt,
      std::nullopt);

  const auto serialized = nimble::EncodingFactory::encode<D>(
      std::move(policy),
      std::span<const D>{values.data(), values.size()},
      *this->buffer_,
      options);

  std::vector<std::tuple<nimble::EncodingType, nimble::DataType, std::string>>
      actual;
  nimble::tools::traverseEncodings(
      serialized,
      [&](auto encodingType,
          auto dataType,
          auto /* level */,
          auto /* index */,
          auto nestedEncodingName,
          std::unordered_map<
              nimble::tools::EncodingPropertyType,
              nimble::tools::EncodingProperty> /* properties */) {
        actual.emplace_back(encodingType, dataType, nestedEncodingName);
        return true;
      });

  const std::vector<
      std::tuple<nimble::EncodingType, nimble::DataType, std::string>>
      expected{
          {nimble::EncodingType::Dictionary,
           nimble::TypeTraits<D>::dataType,
           ""},
          {nimble::EncodingType::ALP,
           nimble::TypeTraits<D>::dataType,
           "Alphabet"},
          {nimble::EncodingType::Trivial,
           nimble::DataType::Uint64,
           "EncodedValues"},
          {nimble::EncodingType::Trivial, nimble::DataType::Uint32, "Indices"},
      };
  EXPECT_EQ(actual, expected);

  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buf = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buf->template asMutable<void>();
  };
  auto encoding = nimble::EncodingFactory().create(
      *this->pool_, serialized, stringBufferFactory, options);

  nimble::Vector<D> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());

  for (uint32_t i = 0; i < values.size(); ++i) {
    SCOPED_TRACE(fmt::format("i={}", i));
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]));
  }
}

TYPED_TEST(ALPEncodingTest, rleRunValuesUseNestedAlpWhenEnabled) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = false,
      .fixedBitWidthUseExactBits = true,
      .allowNestedAlpSelection = true};

  nimble::Vector<D> values{this->pool_.get()};
  for (auto i = 0; i < 128; ++i) {
    const auto value = static_cast<D>((i % 17) - 8) / static_cast<D>(4);
    values.push_back(value);
    values.push_back(value);
    values.push_back(value);
  }

  auto policy = std::make_unique<nimble::ManualEncodingSelectionPolicy<D>>(
      std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::RLE, 1.0},
      },
      std::nullopt,
      std::nullopt);

  const auto serialized = nimble::EncodingFactory::encode<D>(
      std::move(policy),
      std::span<const D>{values.data(), values.size()},
      *this->buffer_,
      options);

  std::vector<std::tuple<nimble::EncodingType, nimble::DataType, std::string>>
      actual;
  nimble::tools::traverseEncodings(
      serialized,
      [&](auto encodingType,
          auto dataType,
          auto /* level */,
          auto /* index */,
          auto nestedEncodingName,
          std::unordered_map<
              nimble::tools::EncodingPropertyType,
              nimble::tools::EncodingProperty> /* properties */) {
        actual.emplace_back(encodingType, dataType, nestedEncodingName);
        return true;
      });

  const std::vector<
      std::tuple<nimble::EncodingType, nimble::DataType, std::string>>
      expected{
          {nimble::EncodingType::RLE, nimble::TypeTraits<D>::dataType, ""},
          {nimble::EncodingType::Trivial, nimble::DataType::Uint32, "Lengths"},
          {nimble::EncodingType::ALP,
           nimble::TypeTraits<D>::dataType,
           "Values"},
          {nimble::EncodingType::Trivial,
           nimble::DataType::Uint64,
           "EncodedValues"},
      };
  EXPECT_EQ(actual, expected);

  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buf = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buf->template asMutable<void>();
  };
  auto encoding = nimble::EncodingFactory().create(
      *this->pool_, serialized, stringBufferFactory, options);

  nimble::Vector<D> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());

  for (uint32_t i = 0; i < values.size(); ++i) {
    SCOPED_TRACE(fmt::format("i={}", i));
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]));
  }
}

TYPED_TEST(ALPEncodingTest, mainlyConstantOtherValuesUseNestedAlpWhenEnabled) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = false,
      .fixedBitWidthUseExactBits = true,
      .allowNestedAlpSelection = true};

  nimble::Vector<D> values{this->pool_.get()};
  for (auto i = 0; i < 256; ++i) {
    values.push_back(D{0});
    if (i % 8 == 0) {
      values.back() = static_cast<D>((i % 17) - 8) / static_cast<D>(4);
    }
  }

  auto policy = std::make_unique<nimble::ManualEncodingSelectionPolicy<D>>(
      std::vector<std::pair<nimble::EncodingType, float>>{
          {nimble::EncodingType::MainlyConstant, 1.0},
      },
      std::nullopt,
      std::nullopt);

  const auto serialized = nimble::EncodingFactory::encode<D>(
      std::move(policy),
      std::span<const D>{values.data(), values.size()},
      *this->buffer_,
      options);

  std::vector<std::tuple<nimble::EncodingType, nimble::DataType, std::string>>
      actual;
  nimble::tools::traverseEncodings(
      serialized,
      [&](auto encodingType,
          auto dataType,
          auto /* level */,
          auto /* index */,
          auto nestedEncodingName,
          std::unordered_map<
              nimble::tools::EncodingPropertyType,
              nimble::tools::EncodingProperty> /* properties */) {
        actual.emplace_back(encodingType, dataType, nestedEncodingName);
        return true;
      });

  const std::vector<
      std::tuple<nimble::EncodingType, nimble::DataType, std::string>>
      expected{
          {nimble::EncodingType::MainlyConstant,
           nimble::TypeTraits<D>::dataType,
           ""},
          {nimble::EncodingType::Trivial, nimble::DataType::Bool, "IsCommon"},
          {nimble::EncodingType::ALP,
           nimble::TypeTraits<D>::dataType,
           "OtherValues"},
          {nimble::EncodingType::Trivial,
           nimble::DataType::Uint64,
           "EncodedValues"},
      };
  EXPECT_EQ(actual, expected);

  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buf = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buf->template asMutable<void>();
  };
  auto encoding = nimble::EncodingFactory().create(
      *this->pool_, serialized, stringBufferFactory, options);

  nimble::Vector<D> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());

  for (uint32_t i = 0; i < values.size(); ++i) {
    SCOPED_TRACE(fmt::format("i={}", i));
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]));
  }
}

TYPED_TEST(ALPEncodingTest, randomizedFixedLayoutMaterializeAndSkip) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = false, .fixedBitWidthUseExactBits = true};

  struct Scenario {
    std::string name;
    nimble::EncodingLayout layout;
    uint32_t rowCount;
    uint32_t uniqueCount;
    std::vector<std::tuple<nimble::EncodingType, nimble::DataType, std::string>>
        expectedLayout;
  };

  const std::vector<Scenario> scenarios{
      {
          "alp",
          alpWithFixedBitWidthPayloadLayout(),
          4096,
          257,
          {
              {nimble::EncodingType::ALP, nimble::TypeTraits<D>::dataType, ""},
              {nimble::EncodingType::FixedBitWidth,
               nimble::DataType::Uint64,
               "EncodedValues"},
          },
      },
      {
          "dictionary",
          dictionaryWithAlpAlphabetLayout(),
          4096,
          97,
          {
              {nimble::EncodingType::Dictionary,
               nimble::TypeTraits<D>::dataType,
               ""},
              {nimble::EncodingType::ALP,
               nimble::TypeTraits<D>::dataType,
               "Alphabet"},
              {nimble::EncodingType::FixedBitWidth,
               nimble::DataType::Uint64,
               "EncodedValues"},
              {nimble::EncodingType::FixedBitWidth,
               nimble::DataType::Uint32,
               "Indices"},
          },
      },
  };

  for (uint32_t scenarioIndex = 0; scenarioIndex < scenarios.size();
       ++scenarioIndex) {
    const auto& scenario = scenarios[scenarioIndex];
    SCOPED_TRACE(
        fmt::format(
            "scenario={} type={} useVarint={}",
            scenario.name,
            nimble::TypeTraits<D>::dataType,
            options.useVarintRowCount));

    nimble::Buffer buffer{*this->pool_};
    auto values = makeRandomDecimalValues<D>(
        this->pool_.get(),
        scenario.rowCount,
        scenario.uniqueCount,
        0xC0FFEE + scenarioIndex);
    const auto serialized =
        encodeWithLayout<D>(buffer, values, scenario.layout, options);
    expectEncodingLayout(serialized, scenario.expectedLayout);

    std::vector<velox::BufferPtr> stringBuffers;
    auto encoding =
        createEncoding(this->pool_.get(), serialized, options, stringBuffers);
    expectInterleavedMaterializeAndSkip(
        *encoding, values, this->pool_.get(), 0xA11CE + scenarioIndex);
  }
}

TYPED_TEST(ALPEncodingTest, skipAndMaterialize) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->template toVector<D>({10, 20, 30, 40, 50, 60});

  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buf = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buf->template asMutable<void>();
  };

  auto encoding = nimble::test::Encoder<nimble::ALPEncoding<D>>::createEncoding(
      *this->buffer_,
      values,
      stringBufferFactory,
      nimble::CompressionType::Uncompressed,
      options);

  encoding->skip(2);

  nimble::Vector<D> result(this->pool_.get(), 3);
  encoding->materialize(3, result.data());

  EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[0], D{30}));
  EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[1], D{40}));
  EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[2], D{50}));
}

TYPED_TEST(ALPEncodingTest, resetAndRematerialize) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->template toVector<D>({7, 8, 9});

  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buf = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buf->template asMutable<void>();
  };

  auto encoding = nimble::test::Encoder<nimble::ALPEncoding<D>>::createEncoding(
      *this->buffer_,
      values,
      stringBufferFactory,
      nimble::CompressionType::Uncompressed,
      options);

  nimble::Vector<D> first(this->pool_.get(), values.size());
  encoding->materialize(values.size(), first.data());

  encoding->reset();

  nimble::Vector<D> second(this->pool_.get(), values.size());
  encoding->materialize(values.size(), second.data());

  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(first[i], second[i]));
  }
}

TYPED_TEST(ALPEncodingTest, emptyDataRejected) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<D> empty{this->pool_.get()};

  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buf = stringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buf->template asMutable<void>();
  };

  NIMBLE_ASSERT_USER_THROW(
      nimble::test::Encoder<nimble::ALPEncoding<D>>::createEncoding(
          *this->buffer_,
          empty,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options),
      "ALP encoding cannot encode empty data.");
}
