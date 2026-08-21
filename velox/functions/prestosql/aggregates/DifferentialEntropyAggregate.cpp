/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#include "velox/functions/prestosql/aggregates/DifferentialEntropyAggregate.h"

#include "velox/common/base/IOUtils.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"

#include <folly/Random.h>
#include <folly/lang/Bits.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

using namespace facebook::velox::exec;

namespace facebook::velox::aggregate::prestosql {

namespace {

// Largest reservoir/bucket count we're willing to allocate per group; mirrors
// the cap Presto's Java implementation applies to reservoir sizes.
constexpr int64_t kMaxSizeLimit = 1'000'000;

double xlogx(double x) {
  return x <= 0.0 ? 0.0 : x * std::log(x);
}

// See FixedHistogramMleStateStrategy / EntropyCalculations.
// calculateEntropyFromHistogramAggregates in Presto's Java implementation.
double entropyFromHistogramAggregates(
    double width,
    double sumWeight,
    double sumWeightLogWeight) {
  VELOX_CHECK_GT(sumWeight, 0.0);
  return std::max(
      (std::log(width * sumWeight) - sumWeightLogWeight / sumWeight) /
          std::log(2.0),
      0.0);
}

// Vasicek (1976) order-statistics entropy estimator, as refined by Alizadeh
// Noughabi & Arghami (2010), "A New Estimator of Entropy". Used for both the
// unweighted and weighted reservoir-sampling variants: once the (possibly
// weight-biased) sample has been drawn, the estimator itself only looks at
// the sampled values.
double calculateFromSamplesUsingVasicek(std::vector<double> samples) {
  if (samples.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(samples.begin(), samples.end());
  const int64_t n = samples.size();
  const int64_t m =
      std::max<int64_t>(std::llround(std::sqrt(static_cast<double>(n))), 2);
  double entropy = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double sIPlusM = (i + m < n) ? samples[i + m] : samples[n - 1];
    const double sIMinusM = (i - m > 0) ? samples[i - m] : samples[0];
    const double aI = (i + m < n && i - m > 0) ? 2 : 1;
    entropy +=
        std::log(static_cast<double>(n) / (aI * m) * (sIPlusM - sIMinusM));
  }
  return entropy / n / std::log(2);
}

void validateHistogramParams(int64_t bucketCount, double min, double max) {
  VELOX_USER_CHECK_GT(
      bucketCount,
      0,
      "In differential_entropy UDF, bucket count must be positive: {}",
      bucketCount);
  VELOX_USER_CHECK_LT(
      min,
      max,
      "In differential_entropy UDF, min must be less than max: min={}, max={}",
      min,
      max);
}

int64_t histogramBucketIndex(
    double value,
    double min,
    double width,
    int64_t bucketCount) {
  const auto idx = static_cast<int64_t>(std::floor((value - min) / width));
  return std::clamp<int64_t>(idx, 0, bucketCount - 1);
}

// ---------------------------------------------------------------------------
// Unweighted reservoir sample (Algorithm R) + Vasicek estimator.
// Mirrors Presto's UnweightedDoubleReservoirSample.
// ---------------------------------------------------------------------------
class UnweightedReservoir {
 public:
  UnweightedReservoir(int64_t maxSamples, HashStringAllocator* allocator)
      : maxSamples_(maxSamples), samples_{StlAllocator<double>(allocator)} {
    VELOX_USER_CHECK_GT(
        maxSamples,
        0,
        "In differential_entropy UDF, max samples must be positive: {}",
        maxSamples);
    VELOX_USER_CHECK_LE(
        maxSamples,
        kMaxSizeLimit,
        "In differential_entropy UDF, max samples must be capped: {} vs {}",
        maxSamples,
        kMaxSizeLimit);
    samples_.reserve(maxSamples);
  }

  UnweightedReservoir(
      common::InputByteStream& in,
      HashStringAllocator* allocator)
      : samples_{StlAllocator<double>(allocator)} {
    seenCount_ = in.read<int64_t>();
    maxSamples_ = in.read<int64_t>();
    const auto storedCount = std::min(seenCount_, maxSamples_);
    samples_.resize(storedCount);
    for (int64_t i = 0; i < storedCount; ++i) {
      samples_[i] = in.read<double>();
    }
  }

  int64_t maxSamples() const {
    return maxSamples_;
  }

  void add(double value) {
    ++seenCount_;
    if (seenCount_ <= maxSamples_) {
      samples_.push_back(value);
      return;
    }
    const auto index = folly::Random::rand64(seenCount_);
    if (index < static_cast<uint64_t>(maxSamples_)) {
      samples_[index] = value;
    }
  }

  // Combines `other` into `this`, following the same three cases as
  // Presto's UnweightedDoubleReservoirSample.mergeWith.
  void mergeWith(UnweightedReservoir& other) {
    VELOX_USER_CHECK_EQ(
        maxSamples_,
        other.maxSamples_,
        "In differential_entropy UDF, inconsistent size: {} vs {}",
        maxSamples_,
        other.maxSamples_);

    if (other.seenCount_ < other.maxSamples_) {
      for (double value : other.samples_) {
        add(value);
      }
      return;
    }

    if (seenCount_ < maxSamples_) {
      auto targetSamples = other.samples_;
      int64_t targetSeenCount = other.seenCount_;
      for (double value : samples_) {
        ++targetSeenCount;
        const auto index = folly::Random::rand64(targetSeenCount);
        if (index < static_cast<uint64_t>(maxSamples_)) {
          targetSamples[index] = value;
        }
      }
      samples_ = std::move(targetSamples);
      seenCount_ = targetSeenCount;
      return;
    }

    // Both reservoirs are full: shuffle each and randomly interleave,
    // weighted by how many elements each reservoir has actually seen.
    auto shuffledThis = samples_;
    auto shuffledOther = other.samples_;
    folly::ThreadLocalPRNG rng;
    std::shuffle(shuffledThis.begin(), shuffledThis.end(), rng);
    std::shuffle(shuffledOther.begin(), shuffledOther.end(), rng);

    std::vector<double, StlAllocator<double>> merged{
        StlAllocator<double>(samples_.get_allocator())};
    merged.resize(maxSamples_);
    int64_t nextThis = 0;
    int64_t nextOther = 0;
    for (int64_t i = 0; i < maxSamples_; ++i) {
      if (folly::Random::rand64(seenCount_ + other.seenCount_) <
          static_cast<uint64_t>(seenCount_)) {
        merged[i] = shuffledThis[nextThis++];
      } else {
        merged[i] = shuffledOther[nextOther++];
      }
    }
    seenCount_ += other.seenCount_;
    samples_ = std::move(merged);
  }

  double calculateEntropy() const {
    return calculateFromSamplesUsingVasicek(
        std::vector<double>(samples_.begin(), samples_.end()));
  }

  size_t serializedSize() const {
    return 2 * sizeof(int64_t) + samples_.size() * sizeof(double);
  }

  void serialize(common::OutputByteStream& out) const {
    out.appendOne(seenCount_);
    out.appendOne(maxSamples_);
    for (double value : samples_) {
      out.appendOne(value);
    }
  }

 private:
  int64_t maxSamples_{0};
  int64_t seenCount_{0};
  std::vector<double, StlAllocator<double>> samples_;
};

// ---------------------------------------------------------------------------
// Weighted reservoir sample (Efraimidis-Spirakis A-ExpJ: a min-heap keyed by
// U^(1/weight)) + Vasicek estimator. Mirrors Presto's
// WeightedDoubleReservoirSample, but uses standard 0-indexed min-heap
// parent/child arithmetic rather than replicating Java's off-by-one
// leftChild(0)==0 quirk (which leaves the heap root unable to bubble down and
// can make the eviction check compare against a stale, non-minimum key).
// ---------------------------------------------------------------------------
class WeightedReservoir {
 public:
  WeightedReservoir(int64_t maxSamples, HashStringAllocator* allocator)
      : maxSamples_(maxSamples),
        samples_{StlAllocator<double>(allocator)},
        weights_{StlAllocator<double>(allocator)} {
    VELOX_USER_CHECK_GT(
        maxSamples,
        0,
        "In differential_entropy UDF, max samples must be positive: {}",
        maxSamples);
    VELOX_USER_CHECK_LE(
        maxSamples,
        kMaxSizeLimit,
        "In differential_entropy UDF, max samples must be capped: {} vs {}",
        maxSamples,
        kMaxSizeLimit);
    samples_.reserve(maxSamples);
    weights_.reserve(maxSamples);
  }

  WeightedReservoir(common::InputByteStream& in, HashStringAllocator* allocator)
      : samples_{StlAllocator<double>(allocator)},
        weights_{StlAllocator<double>(allocator)} {
    const auto count = in.read<int64_t>();
    maxSamples_ = in.read<int64_t>();
    samples_.resize(count);
    weights_.resize(count);
    for (int64_t i = 0; i < count; ++i) {
      samples_[i] = in.read<double>();
    }
    for (int64_t i = 0; i < count; ++i) {
      weights_[i] = in.read<double>();
    }
    totalPopulationWeight_ = in.read<double>();
  }

  int64_t maxSamples() const {
    return maxSamples_;
  }

  void add(double value, double weight) {
    VELOX_USER_CHECK_GE(
        weight,
        0.0,
        "In differential_entropy UDF, weight must be non-negative: {}",
        weight);
    totalPopulationWeight_ += weight;
    // Efraimidis-Spirakis A-ExpJ key: U^(1/w), U ~ Uniform(0,1). A zero
    // weight yields a key of 0, the minimum possible, so it is never
    // preferred over any positive-weight item already in the reservoir.
    const double adjustedWeight = weight > 0.0
        ? std::pow(folly::Random::randDouble01(), 1.0 / weight)
        : 0.0;
    addWithAdjustedWeight(value, adjustedWeight);
  }

  void mergeWith(WeightedReservoir& other) {
    VELOX_USER_CHECK_EQ(
        maxSamples_,
        other.maxSamples_,
        "In differential_entropy UDF, inconsistent size: {} vs {}",
        maxSamples_,
        other.maxSamples_);
    totalPopulationWeight_ += other.totalPopulationWeight_;
    for (size_t i = 0; i < other.samples_.size(); ++i) {
      addWithAdjustedWeight(other.samples_[i], other.weights_[i]);
    }
  }

  double calculateEntropy() const {
    return calculateFromSamplesUsingVasicek(
        std::vector<double>(samples_.begin(), samples_.end()));
  }

  size_t serializedSize() const {
    return 2 * sizeof(int64_t) + 2 * samples_.size() * sizeof(double) +
        sizeof(double);
  }

  void serialize(common::OutputByteStream& out) const {
    out.appendOne<int64_t>(samples_.size());
    out.appendOne(maxSamples_);
    for (double value : samples_) {
      out.appendOne(value);
    }
    for (double weight : weights_) {
      out.appendOne(weight);
    }
    out.appendOne(totalPopulationWeight_);
  }

 private:
  void addWithAdjustedWeight(double value, double adjustedWeight) {
    if (static_cast<int64_t>(samples_.size()) < maxSamples_) {
      samples_.push_back(value);
      weights_.push_back(adjustedWeight);
      bubbleUp(samples_.size() - 1);
      return;
    }
    if (adjustedWeight <= weights_[0]) {
      return;
    }
    samples_[0] = value;
    weights_[0] = adjustedWeight;
    bubbleDown();
  }

  void swapAt(size_t i, size_t j) {
    std::swap(samples_[i], samples_[j]);
    std::swap(weights_[i], weights_[j]);
  }

  void bubbleUp(size_t index) {
    while (index > 0) {
      const size_t parent = (index - 1) / 2;
      if (weights_[index] >= weights_[parent]) {
        break;
      }
      swapAt(index, parent);
      index = parent;
    }
  }

  void bubbleDown() {
    size_t index = 0;
    const size_t n = samples_.size();
    for (;;) {
      const size_t left = 2 * index + 1;
      const size_t right = 2 * index + 2;
      size_t smallest = index;
      if (left < n && weights_[left] < weights_[smallest]) {
        smallest = left;
      }
      if (right < n && weights_[right] < weights_[smallest]) {
        smallest = right;
      }
      if (smallest == index) {
        break;
      }
      swapAt(index, smallest);
      index = smallest;
    }
  }

  int64_t maxSamples_{0};
  double totalPopulationWeight_{0};
  std::vector<double, StlAllocator<double>> samples_;
  std::vector<double, StlAllocator<double>> weights_;
};

// ---------------------------------------------------------------------------
// Fixed-histogram maximum-likelihood estimator. Mirrors
// FixedHistogramMleStateStrategy.
// ---------------------------------------------------------------------------
class FixedHistogramMle {
 public:
  FixedHistogramMle(
      int64_t bucketCount,
      double min,
      double max,
      HashStringAllocator* allocator)
      : bucketCount_(bucketCount),
        min_(min),
        max_(max),
        width_((max - min) / bucketCount),
        bucketWeights_{StlAllocator<double>(allocator)} {
    validateHistogramParams(bucketCount, min, max);
    bucketWeights_.assign(bucketCount, 0.0);
  }

  FixedHistogramMle(common::InputByteStream& in, HashStringAllocator* allocator)
      : bucketWeights_{StlAllocator<double>(allocator)} {
    bucketCount_ = in.read<int64_t>();
    min_ = in.read<double>();
    max_ = in.read<double>();
    width_ = (max_ - min_) / bucketCount_;
    bucketWeights_.resize(bucketCount_);
    for (int64_t i = 0; i < bucketCount_; ++i) {
      bucketWeights_[i] = in.read<double>();
    }
  }

  int64_t bucketCount() const {
    return bucketCount_;
  }
  double min() const {
    return min_;
  }
  double max() const {
    return max_;
  }

  void add(double value, double weight) {
    bucketWeights_[histogramBucketIndex(value, min_, width_, bucketCount_)] +=
        weight;
  }

  void mergeWith(FixedHistogramMle& other) {
    checkConsistentParams(other.bucketCount_, other.min_, other.max_);
    for (int64_t i = 0; i < bucketCount_; ++i) {
      bucketWeights_[i] += other.bucketWeights_[i];
    }
  }

  double calculateEntropy() const {
    double sum = 0;
    for (double w : bucketWeights_) {
      sum += w;
    }
    if (sum == 0.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double rawEntropy = 0;
    for (double w : bucketWeights_) {
      rawEntropy -= xlogx(w / sum);
    }
    // Unlike the jacknife estimator, MLE is not floored at 0: a
    // sufficiently narrow histogram can legitimately estimate a negative
    // differential entropy (differential entropy, unlike Shannon entropy,
    // is not bounded below).
    return (rawEntropy + std::log(width_)) / std::log(2);
  }

  size_t serializedSize() const {
    return sizeof(int64_t) + 2 * sizeof(double) +
        bucketWeights_.size() * sizeof(double);
  }

  void serialize(common::OutputByteStream& out) const {
    out.appendOne(bucketCount_);
    out.appendOne(min_);
    out.appendOne(max_);
    for (double w : bucketWeights_) {
      out.appendOne(w);
    }
  }

  void checkConsistentParams(int64_t bucketCount, double min, double max)
      const {
    VELOX_USER_CHECK_EQ(
        bucketCount_,
        bucketCount,
        "In differential_entropy UDF, inconsistent bucket count: {} vs {}",
        bucketCount_,
        bucketCount);
    VELOX_USER_CHECK_EQ(
        min_,
        min,
        "In differential_entropy UDF, inconsistent min: {} vs {}",
        min_,
        min);
    VELOX_USER_CHECK_EQ(
        max_,
        max,
        "In differential_entropy UDF, inconsistent max: {} vs {}",
        max_,
        max);
  }

 private:
  int64_t bucketCount_{0};
  double min_{0};
  double max_{0};
  double width_{0};
  std::vector<double, StlAllocator<double>> bucketWeights_;
};

// ---------------------------------------------------------------------------
// Fixed-histogram jacknife-corrected estimator. Mirrors
// FixedHistogramJacknifeStateStrategy. Unlike the MLE variant, this needs a
// "breakdown" of each spatial bucket by exact per-row weight value (not just
// a bucket weight total), because the leave-one-out correction removes one
// *observation* at a time: observations that share both a bucket and an
// exact weight value produce the same holdout entropy, so grouping them
// lets the correction run in O(bucketCount * distinct weights per bucket)
// rather than O(row count).
//
// Known scaffolding limitation: the per-bucket (weight, count) breakdown
// uses a plain std::vector with the default allocator, so it is not counted
// against the query's memory pool the way the outer bucket array is. This
// is fine for the common case of few distinct weights per bucket (e.g. the
// default weight=1.0), but should be revisited (e.g. a pool-backed hash map)
// before relying on this for adversarial high-cardinality-weight inputs.
// ---------------------------------------------------------------------------
class FixedHistogramJacknife {
 public:
  FixedHistogramJacknife(
      int64_t bucketCount,
      double min,
      double max,
      HashStringAllocator* allocator)
      : bucketCount_(bucketCount),
        min_(min),
        max_(max),
        width_((max - min) / bucketCount),
        buckets_{StlAllocator<BucketBreakdown>(allocator)} {
    validateHistogramParams(bucketCount, min, max);
    buckets_.resize(bucketCount);
  }

  FixedHistogramJacknife(
      common::InputByteStream& in,
      HashStringAllocator* allocator)
      : buckets_{StlAllocator<BucketBreakdown>(allocator)} {
    bucketCount_ = in.read<int64_t>();
    min_ = in.read<double>();
    max_ = in.read<double>();
    width_ = (max_ - min_) / bucketCount_;
    buckets_.resize(bucketCount_);
    for (int64_t i = 0; i < bucketCount_; ++i) {
      const auto entryCount = in.read<int64_t>();
      buckets_[i].reserve(entryCount);
      for (int64_t j = 0; j < entryCount; ++j) {
        const auto weight = in.read<double>();
        const auto count = in.read<int64_t>();
        buckets_[i].emplace_back(weight, count);
      }
    }
  }

  int64_t bucketCount() const {
    return bucketCount_;
  }
  double min() const {
    return min_;
  }
  double max() const {
    return max_;
  }

  void add(double value, double weight) {
    addToBucket(
        buckets_[histogramBucketIndex(value, min_, width_, bucketCount_)],
        weight,
        1);
  }

  void mergeWith(FixedHistogramJacknife& other) {
    checkConsistentParams(other.bucketCount_, other.min_, other.max_);
    for (int64_t i = 0; i < bucketCount_; ++i) {
      for (const auto& [weight, count] : other.buckets_[i]) {
        addToBucket(buckets_[i], weight, count);
      }
    }
  }

  double calculateEntropy() const {
    std::vector<double> bucketWeight(bucketCount_, 0.0);
    int64_t n = 0;
    double sumWeight = 0;
    for (int64_t i = 0; i < bucketCount_; ++i) {
      for (const auto& [weight, count] : buckets_[i]) {
        bucketWeight[i] += weight * static_cast<double>(count);
        n += count;
      }
      sumWeight += bucketWeight[i];
    }
    // n<=1 makes the leave-one-out holdout degenerate (removing the only
    // observation leaves an empty histogram); Presto's Java implementation
    // would throw here, we return NaN instead, consistent with the
    // reservoir estimators' "not enough data" convention.
    if (sumWeight == 0.0 || n <= 1) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    double sumWeightLogWeight = 0;
    for (double w : bucketWeight) {
      sumWeightLogWeight += xlogx(w);
    }

    double entropy = static_cast<double>(n) *
        entropyFromHistogramAggregates(width_, sumWeight, sumWeightLogWeight);
    for (int64_t i = 0; i < bucketCount_; ++i) {
      if (bucketWeight[i] <= 0.0) {
        continue;
      }
      for (const auto& [entryWeight, entryCount] : buckets_[i]) {
        entropy -= holdOutEntropy(
            n,
            width_,
            sumWeight,
            sumWeightLogWeight,
            bucketWeight[i],
            entryWeight,
            entryCount);
      }
    }
    return entropy;
  }

  size_t serializedSize() const {
    size_t size = sizeof(int64_t) + 2 * sizeof(double);
    for (const auto& bucket : buckets_) {
      size +=
          sizeof(int64_t) + bucket.size() * (sizeof(double) + sizeof(int64_t));
    }
    return size;
  }

  void serialize(common::OutputByteStream& out) const {
    out.appendOne(bucketCount_);
    out.appendOne(min_);
    out.appendOne(max_);
    for (const auto& bucket : buckets_) {
      out.appendOne<int64_t>(bucket.size());
      for (const auto& [weight, count] : bucket) {
        out.appendOne(weight);
        out.appendOne(count);
      }
    }
  }

  void checkConsistentParams(int64_t bucketCount, double min, double max)
      const {
    VELOX_USER_CHECK_EQ(
        bucketCount_,
        bucketCount,
        "In differential_entropy UDF, inconsistent bucket count: {} vs {}",
        bucketCount_,
        bucketCount);
    VELOX_USER_CHECK_EQ(
        min_,
        min,
        "In differential_entropy UDF, inconsistent min: {} vs {}",
        min_,
        min);
    VELOX_USER_CHECK_EQ(
        max_,
        max,
        "In differential_entropy UDF, inconsistent max: {} vs {}",
        max_,
        max);
  }

 private:
  using BucketBreakdown = std::vector<std::pair<double, int64_t>>;

  static void
  addToBucket(BucketBreakdown& bucket, double weight, int64_t count) {
    for (auto& [existingWeight, existingCount] : bucket) {
      if (existingWeight == weight) {
        existingCount += count;
        return;
      }
    }
    bucket.emplace_back(weight, count);
  }

  // See FixedHistogramJacknifeStateStrategy.getHoldOutEntropy: the
  // contribution to the jacknife correction sum from removing `entryCount`
  // copies of a single (bucket, weight-value) breakdown entry, one at a
  // time, and averaging.
  static double holdOutEntropy(
      int64_t n,
      double width,
      double sumWeight,
      double sumWeightLogWeight,
      double bucketWeight,
      double entryWeight,
      int64_t entryCount) {
    const double holdoutBucketWeight =
        std::max(bucketWeight - entryWeight, 0.0);
    const double holdoutSumWeight =
        sumWeight - bucketWeight + holdoutBucketWeight;
    const double holdoutSumWeightLogWeight =
        sumWeightLogWeight - xlogx(bucketWeight) + xlogx(holdoutBucketWeight);
    if (holdoutSumWeight <= 0.0) {
      return 0.0;
    }
    return static_cast<double>(entryCount) * static_cast<double>(n - 1) *
        entropyFromHistogramAggregates(
               width, holdoutSumWeight, holdoutSumWeightLogWeight) /
        static_cast<double>(n);
  }

  int64_t bucketCount_{0};
  double min_{0};
  double max_{0};
  double width_{0};
  std::vector<BucketBreakdown, StlAllocator<BucketBreakdown>> buckets_;
};

enum class HistogramMethod : int8_t {
  kUnset = 0,
  kMle = 1,
  kJacknife = 2,
};

HistogramMethod parseHistogramMethod(const StringView& method) {
  std::string lower(method.data(), method.size());
  std::transform(
      lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
  if (lower == "fixed_histogram_mle") {
    return HistogramMethod::kMle;
  }
  if (lower == "fixed_histogram_jacknife") {
    return HistogramMethod::kJacknife;
  }
  VELOX_USER_FAIL(
      "In differential_entropy UDF, invalid method: {}",
      std::string(method.data(), method.size()));
}

// ---------------------------------------------------------------------------
// Accumulators (see velox/exec/SimpleAggregateAdapter.h for the contract).
// ---------------------------------------------------------------------------

struct UnweightedReservoirAccumulator {
  static constexpr bool is_fixed_size_ = false;
  static constexpr bool use_external_memory_ = true;
  static constexpr bool is_aligned_ = true;

  std::optional<UnweightedReservoir> reservoir_;

  UnweightedReservoirAccumulator(HashStringAllocator*, void*) {}

  bool addInput(
      HashStringAllocator* allocator,
      exec::arg_type<int64_t> size,
      exec::arg_type<double> sample) {
    if (!reservoir_.has_value()) {
      reservoir_.emplace(size, allocator);
    }
    VELOX_USER_CHECK_EQ(
        reservoir_->maxSamples(),
        size,
        "In differential_entropy UDF, inconsistent size: {} vs {}",
        reservoir_->maxSamples(),
        size);
    reservoir_->add(sample);
    return true;
  }

  void combine(
      HashStringAllocator* allocator,
      exec::arg_type<Varbinary> other) {
    common::InputByteStream stream(other.data());
    UnweightedReservoir otherReservoir(stream, allocator);
    if (!reservoir_.has_value()) {
      reservoir_.emplace(otherReservoir.maxSamples(), allocator);
    }
    reservoir_->mergeWith(otherReservoir);
  }

  bool writeIntermediateResult(exec::out_type<Varbinary>& out) {
    if (!reservoir_.has_value()) {
      return false;
    }
    const auto size = reservoir_->serializedSize();
    out.reserve(size);
    common::OutputByteStream stream(out.data());
    reservoir_->serialize(stream);
    out.resize(size);
    return true;
  }

  bool writeFinalResult(exec::out_type<double>& out) {
    if (!reservoir_.has_value()) {
      return false;
    }
    out = reservoir_->calculateEntropy();
    return true;
  }
};

struct WeightedReservoirAccumulator {
  static constexpr bool is_fixed_size_ = false;
  static constexpr bool use_external_memory_ = true;
  static constexpr bool is_aligned_ = true;

  std::optional<WeightedReservoir> reservoir_;

  WeightedReservoirAccumulator(HashStringAllocator*, void*) {}

  bool addInput(
      HashStringAllocator* allocator,
      exec::arg_type<int64_t> size,
      exec::arg_type<double> sample,
      exec::arg_type<double> weight) {
    if (!reservoir_.has_value()) {
      reservoir_.emplace(size, allocator);
    }
    VELOX_USER_CHECK_EQ(
        reservoir_->maxSamples(),
        size,
        "In differential_entropy UDF, inconsistent size: {} vs {}",
        reservoir_->maxSamples(),
        size);
    reservoir_->add(sample, weight);
    return true;
  }

  void combine(
      HashStringAllocator* allocator,
      exec::arg_type<Varbinary> other) {
    common::InputByteStream stream(other.data());
    WeightedReservoir otherReservoir(stream, allocator);
    if (!reservoir_.has_value()) {
      reservoir_.emplace(otherReservoir.maxSamples(), allocator);
    }
    reservoir_->mergeWith(otherReservoir);
  }

  bool writeIntermediateResult(exec::out_type<Varbinary>& out) {
    if (!reservoir_.has_value()) {
      return false;
    }
    const auto size = reservoir_->serializedSize();
    out.reserve(size);
    common::OutputByteStream stream(out.data());
    reservoir_->serialize(stream);
    out.resize(size);
    return true;
  }

  bool writeFinalResult(exec::out_type<double>& out) {
    if (!reservoir_.has_value()) {
      return false;
    }
    out = reservoir_->calculateEntropy();
    return true;
  }
};

struct FixedHistogramAccumulator {
  static constexpr bool is_fixed_size_ = false;
  static constexpr bool use_external_memory_ = true;
  static constexpr bool is_aligned_ = true;

  HistogramMethod method_{HistogramMethod::kUnset};
  std::optional<FixedHistogramMle> mle_;
  std::optional<FixedHistogramJacknife> jacknife_;

  FixedHistogramAccumulator(HashStringAllocator*, void*) {}

  bool addInput(
      HashStringAllocator* allocator,
      exec::arg_type<int64_t> bucketCount,
      exec::arg_type<double> sample,
      exec::arg_type<double> weight,
      exec::arg_type<Varchar> method,
      exec::arg_type<double> min,
      exec::arg_type<double> max) {
    const auto parsedMethod = parseHistogramMethod(method);
    if (method_ == HistogramMethod::kUnset) {
      method_ = parsedMethod;
      if (method_ == HistogramMethod::kMle) {
        mle_.emplace(bucketCount, min, max, allocator);
      } else {
        jacknife_.emplace(bucketCount, min, max, allocator);
      }
    } else {
      VELOX_USER_CHECK(
          method_ == parsedMethod,
          "In differential_entropy UDF, method must be consistent within a "
          "single aggregation");
      if (method_ == HistogramMethod::kMle) {
        mle_->checkConsistentParams(bucketCount, min, max);
      } else {
        jacknife_->checkConsistentParams(bucketCount, min, max);
      }
    }
    VELOX_USER_CHECK_GE(
        weight,
        0.0,
        "In differential_entropy UDF, weight must be non-negative: {}",
        weight);
    VELOX_USER_CHECK_GE(
        sample,
        min,
        "In differential_entropy UDF, sample must be at least min: sample={}, min={}",
        sample,
        min);
    VELOX_USER_CHECK_LE(
        sample,
        max,
        "In differential_entropy UDF, sample must be at most max: sample={}, max={}",
        sample,
        max);
    if (method_ == HistogramMethod::kMle) {
      mle_->add(sample, weight);
    } else {
      jacknife_->add(sample, weight);
    }
    return true;
  }

  void combine(
      HashStringAllocator* allocator,
      exec::arg_type<Varbinary> other) {
    common::InputByteStream stream(other.data());
    const auto otherMethod =
        static_cast<HistogramMethod>(stream.read<int8_t>());
    if (method_ == HistogramMethod::kUnset) {
      method_ = otherMethod;
    }
    VELOX_USER_CHECK(
        method_ == otherMethod,
        "In differential_entropy UDF, method must be consistent within a "
        "single aggregation");

    if (method_ == HistogramMethod::kMle) {
      FixedHistogramMle otherMle(stream, allocator);
      if (!mle_.has_value()) {
        mle_.emplace(
            otherMle.bucketCount(), otherMle.min(), otherMle.max(), allocator);
      }
      mle_->mergeWith(otherMle);
    } else {
      FixedHistogramJacknife otherJacknife(stream, allocator);
      if (!jacknife_.has_value()) {
        jacknife_.emplace(
            otherJacknife.bucketCount(),
            otherJacknife.min(),
            otherJacknife.max(),
            allocator);
      }
      jacknife_->mergeWith(otherJacknife);
    }
  }

  bool writeIntermediateResult(exec::out_type<Varbinary>& out) {
    if (method_ == HistogramMethod::kUnset) {
      return false;
    }
    const auto bodySize = method_ == HistogramMethod::kMle
        ? mle_->serializedSize()
        : jacknife_->serializedSize();
    const auto totalSize = sizeof(int8_t) + bodySize;
    out.reserve(totalSize);
    common::OutputByteStream stream(out.data());
    stream.appendOne(static_cast<int8_t>(method_));
    if (method_ == HistogramMethod::kMle) {
      mle_->serialize(stream);
    } else {
      jacknife_->serialize(stream);
    }
    out.resize(totalSize);
    return true;
  }

  bool writeFinalResult(exec::out_type<double>& out) {
    if (method_ == HistogramMethod::kUnset) {
      return false;
    }
    out = method_ == HistogramMethod::kMle ? mle_->calculateEntropy()
                                           : jacknife_->calculateEntropy();
    return true;
  }
};

template <int kNumArgs>
class DifferentialEntropyAggregate {};

template <>
class DifferentialEntropyAggregate<2> {
 public:
  using InputType = Row<int64_t, double>;
  using IntermediateType = Varbinary;
  using OutputType = double;
  using AccumulatorType = UnweightedReservoirAccumulator;
};

template <>
class DifferentialEntropyAggregate<3> {
 public:
  using InputType = Row<int64_t, double, double>;
  using IntermediateType = Varbinary;
  using OutputType = double;
  using AccumulatorType = WeightedReservoirAccumulator;
};

template <>
class DifferentialEntropyAggregate<6> {
 public:
  using InputType = Row<int64_t, double, double, Varchar, double, double>;
  using IntermediateType = Varbinary;
  using OutputType = double;
  using AccumulatorType = FixedHistogramAccumulator;
};

} // namespace

void registerDifferentialEntropyAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("varbinary")
          .argumentType("bigint")
          .argumentType("double")
          .build());
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("varbinary")
          .argumentType("bigint")
          .argumentType("double")
          .argumentType("double")
          .build());
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("varbinary")
          .argumentType("bigint")
          .argumentType("double")
          .argumentType("double")
          .argumentType("varchar")
          .argumentType("double")
          .argumentType("double")
          .build());

  exec::registerAggregateFunction(
      names,
      std::move(signatures),
      [names](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        const std::string& name = names.front();
        switch (argTypes.size()) {
          case 2:
            return std::make_unique<
                exec::SimpleAggregateAdapter<DifferentialEntropyAggregate<2>>>(
                step, argTypes, resultType);
          case 3:
            return std::make_unique<
                exec::SimpleAggregateAdapter<DifferentialEntropyAggregate<3>>>(
                step, argTypes, resultType);
          case 6:
            return std::make_unique<
                exec::SimpleAggregateAdapter<DifferentialEntropyAggregate<6>>>(
                step, argTypes, resultType);
          default:
            VELOX_USER_FAIL(
                "{} takes 2, 3, or 6 arguments, got {}", name, argTypes.size());
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
