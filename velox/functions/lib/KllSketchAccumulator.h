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

#pragma once

#include <optional>
#include <vector>

#include "velox/common/base/RandomUtil.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/lib/KllSketch.h"

namespace facebook::velox::aggregate {

/// Type traits: selects NaN-aware comparator for floating-point types,
/// and std::less for other types.
template <typename T, typename Allocator>
struct KllSketchTypeTraits {
  using KllSketchType = functions::kll::KllSketch<T, Allocator, std::less<T>>;
};

template <>
struct KllSketchTypeTraits<float, StlAllocator<float>> {
  using KllSketchType = functions::kll::KllSketch<
      float,
      StlAllocator<float>,
      util::floating_point::NaNAwareLessThan<float>>;
};

template <>
struct KllSketchTypeTraits<double, StlAllocator<double>> {
  using KllSketchType = functions::kll::KllSketch<
      double,
      StlAllocator<double>,
      util::floating_point::NaNAwareLessThan<double>>;
};

template <>
struct KllSketchTypeTraits<float, std::allocator<float>> {
  using KllSketchType = functions::kll::KllSketch<
      float,
      std::allocator<float>,
      util::floating_point::NaNAwareLessThan<float>>;
};

template <>
struct KllSketchTypeTraits<double, std::allocator<double>> {
  using KllSketchType = functions::kll::KllSketch<
      double,
      std::allocator<double>,
      util::floating_point::NaNAwareLessThan<double>>;
};

template <typename T, typename Allocator = StlAllocator<T>>
using KllSketch = typename KllSketchTypeTraits<T, Allocator>::KllSketchType;

template <typename T>
using KllView = functions::kll::detail::View<T>;

inline unsigned getRandomSeed(std::optional<uint32_t> fixedRandomSeed) {
  return fixedRandomSeed.has_value() ? *fixedRandomSeed : random::getSeed();
}

/// Aggregate accumulator for KllSketch, supporting weighted insertions
/// with optimization for large weight values.
template <typename T>
struct KllSketchAccumulator {
  explicit KllSketchAccumulator(
      HashStringAllocator* allocator,
      std::optional<uint32_t> fixedRandomSeed)
      : sketch_(
            functions::kll::kDefaultK,
            StlAllocator<T>(allocator),
            getRandomSeed(fixedRandomSeed)),
        largeCountValues_(StlAllocator<std::pair<T, int64_t>>(allocator)) {}

  void setPrestoAccuracy(double accuracy) {
    sketch_.setK(functions::kll::kFromEpsilon(accuracy));
  }

  void setSparkAccuracy(int32_t sparkAccuracy) {
    double epsilon = 1.0 / static_cast<double>(sparkAccuracy);
    sketch_.setK(functions::kll::kFromEpsilon(epsilon));
  }

  void append(T value) {
    sketch_.insert(value);
  }

  void append(
      T value,
      int64_t count,
      HashStringAllocator* allocator,
      std::optional<uint32_t> fixedRandomSeed) {
    constexpr size_t kMaxBufferSize = 4096;
    constexpr int64_t kMinCountToBuffer = 512;
    if (count < kMinCountToBuffer) {
      for (int i = 0; i < count; ++i) {
        sketch_.insert(value);
      }
    } else {
      largeCountValues_.emplace_back(value, count);
      if (largeCountValues_.size() >= kMaxBufferSize) {
        flush(allocator, fixedRandomSeed);
      }
    }
  }

  void append(const KllView<T>& view) {
    sketch_.mergeViews(folly::Range(&view, 1));
  }

  void append(const std::vector<KllView<T>>& views) {
    sketch_.mergeViews(views);
  }

  // Creates a copy of the KllSketch, merges the largeCountValues_ into it,
  // compacts it, and returns it.
  // Makes a copy so that uses std::allocator so that this is safe to call
  // during spilling which may run in parallel.  HashStringAllocator is not
  // thread safe, so merging into/compacting the original KllSketch which
  // depends on it can lead to concurrency bugs.
  KllSketch<T, std::allocator<T>> compact(
      std::optional<uint32_t> fixedRandomSeed) const {
    KllSketch<T, std::allocator<T>> newSketch =
        KllSketch<T, std::allocator<T>>::fromView(
            sketch_.toView(),
            std::allocator<T>(),
            getRandomSeed(fixedRandomSeed));

    mergeLargeCountValuesIntoSketch(
        std::allocator<T>(), newSketch, fixedRandomSeed);

    newSketch.compact();

    return newSketch;
  }

  const KllSketch<T>& getSketch() const {
    return sketch_;
  }

  // This must be called before the KllSketch can be used for estimateQuantile()
  // or estimateQuantiles().
  void flush(
      HashStringAllocator* allocator,
      std::optional<uint32_t> fixedRandomSeed) {
    mergeLargeCountValuesIntoSketch(
        StlAllocator<T>(allocator), sketch_, fixedRandomSeed);
    largeCountValues_.clear();

    sketch_.finish();
  }

 private:
  template <typename Allocator, typename Compare>
  void mergeLargeCountValuesIntoSketch(
      const Allocator& allocator,
      functions::kll::KllSketch<T, Allocator, Compare>& sketch,
      std::optional<uint32_t> fixedRandomSeed) const {
    if (!largeCountValues_.empty()) {
      std::vector<functions::kll::KllSketch<T, Allocator, Compare>> sketches;
      sketches.reserve(largeCountValues_.size());
      for (auto [x, n] : largeCountValues_) {
        sketches.push_back(
            functions::kll::KllSketch<T, Allocator, Compare>::fromRepeatedValue(
                x, n, sketch_.k(), allocator, getRandomSeed(fixedRandomSeed)));
      }
      sketch.merge(folly::Range(sketches.begin(), sketches.end()));
    }
  }

  KllSketch<T> sketch_;
  std::vector<std::pair<T, int64_t>, StlAllocator<std::pair<T, int64_t>>>
      largeCountValues_;
};

} // namespace facebook::velox::aggregate
