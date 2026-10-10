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

#include "velox/functions/sparksql/aggregates/SparkNumericHistogram.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#include <folly/lang/Bits.h>

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

// Implements the public Apache Spark state machine at revision
// b84dc909a8856388faddc154c6a1d3aba271474e:
// https://github.com/apache/spark/blob/b84dc909a8856388faddc154c6a1d3aba271474e/sql/catalyst/src/main/java/org/apache/spark/sql/util/NumericHistogram.java
// The corresponding NumericHistogramSerializer is in
// sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/HistogramNumeric.scala.
// Raw insertion intentionally differs from the total order used during merge.
constexpr uint64_t kRandomMultiplier = 0x5deece66dULL;
constexpr uint64_t kRandomMask = (1ULL << 48) - 1;
constexpr uint64_t kHeaderSize = 8;
constexpr uint64_t kBinSize = 16;

void validateNumBins(int32_t numBins, int32_t currentNumBins) {
  VELOX_USER_CHECK_GE(
      numBins, 2, "histogram_numeric requires numBins at least 2");
  VELOX_USER_CHECK(
      currentNumBins == 0 || currentNumBins == numBins,
      "histogram_numeric numBins mismatch: {} vs {}",
      currentNumBins,
      numBins);
}

template <typename T>
T readBigEndian(const char*& input) {
  T value;
  std::memcpy(&value, input, sizeof(T));
  input += sizeof(T);
  return folly::Endian::big(value);
}

template <typename T>
void writeBigEndian(T value, char*& output) {
  value = folly::Endian::big(value);
  std::memcpy(output, &value, sizeof(T));
  output += sizeof(T);
}

SparkNumericHistogram::Bin readBin(const char*& input) {
  return {
      std::bit_cast<double>(readBigEndian<uint64_t>(input)),
      std::bit_cast<double>(readBigEndian<uint64_t>(input)),
  };
}

// Matches Java Double.compare without canonicalizing any stored NaN payload.
// All NaNs compare equal and follow every number; -0 precedes +0.
bool javaDoubleLess(double left, double right) {
  if (std::isnan(left)) {
    return false;
  }
  if (std::isnan(right)) {
    return true;
  }
  if (left == right) {
    return std::signbit(left) && !std::signbit(right);
  }
  return left < right;
}

// Breaks comparator ties by original concatenation order. std::sort on these
// records implements a stable merge without std::stable_sort's untracked heap
// scratch allocation.
struct OrderedBin {
  SparkNumericHistogram::Bin bin;
  size_t ordinal;
};

} // namespace

SparkNumericHistogram::SparkNumericHistogram(HashStringAllocator* allocator)
    : bins_{AlignedStlAllocator<Bin, alignof(Bin)>(allocator)},
      randomState_{(31'183ULL ^ kRandomMultiplier) & kRandomMask} {}

void SparkNumericHistogram::initialize(int32_t numBins) {
  validateNumBins(numBins, numBins_);
  numBins_ = numBins;
}

void SparkNumericHistogram::add(double value) {
  VELOX_CHECK_GE(numBins_, 2, "histogram_numeric is not initialized");
  size_t bin = 0;
  for (size_t left = 0, right = bins_.size(); left < right;) {
    bin = (left + right) / 2;
    if (bins_[bin].x > value) {
      right = bin;
    } else if (bins_[bin].x < value) {
      left = ++bin;
    } else {
      // Unordered NaN comparisons also stop here, but do not satisfy the
      // primitive equality check below.
      break;
    }
  }
  if (bin < bins_.size() && bins_[bin].x == value) {
    ++bins_[bin].y;
    return;
  }

  serializedSize(std::min<uint64_t>(bins_.size() + 1, numBins_));
  // Vector insertion has the strong exception guarantee for the paired,
  // trivially copyable Bin. No RNG transition precedes successful insertion.
  bins_.insert(bins_.begin() + bin, Bin{value, 1});
  trim();
}

void SparkNumericHistogram::mergeSerialized(std::string_view payload) {
  VELOX_USER_CHECK_GE(
      payload.size(),
      kHeaderSize,
      "histogram_numeric payload has a short header");
  const char* input = payload.data();
  const auto numBins = readBigEndian<int32_t>(input);
  const auto numUsedBins = readBigEndian<int32_t>(input);
  validateNumBins(numBins, 0);
  VELOX_USER_CHECK_GE(
      numUsedBins, 0, "histogram_numeric usedBins must be nonnegative");
  VELOX_USER_CHECK_LE(
      numUsedBins, numBins, "histogram_numeric usedBins exceeds numBins");
  const auto expectedSize = serializedSize(numUsedBins);
  VELOX_USER_CHECK_EQ(
      payload.size(),
      expectedSize,
      "histogram_numeric payload length mismatch");

  const uint64_t numMergedBins =
      bins_.size() + static_cast<uint64_t>(numUsedBins);
  const auto mergedNumBins = bins_.empty() ? numBins : numBins_;
  serializedSize(std::min<uint64_t>(numMergedBins, mergedNumBins));
  BinVector merged(bins_.get_allocator());
  VELOX_USER_CHECK_LE(
      numMergedBins, merged.max_size(), "histogram_numeric state is too large");
  if (bins_.empty()) {
    merged.reserve(numUsedBins);
    for (int32_t i = 0; i < numUsedBins; ++i) {
      merged.push_back(readBin(input));
    }
  } else {
    using OrderedAllocator =
        AlignedStlAllocator<OrderedBin, alignof(OrderedBin)>;
    std::vector<OrderedBin, OrderedAllocator> ordered{
        OrderedAllocator(bins_.get_allocator().allocator())};
    VELOX_USER_CHECK_LE(
        numMergedBins,
        ordered.max_size(),
        "histogram_numeric state is too large");
    ordered.reserve(numMergedBins);
    for (const auto& bin : bins_) {
      ordered.push_back({bin, ordered.size()});
    }
    for (int32_t i = 0; i < numUsedBins; ++i) {
      ordered.push_back({readBin(input), ordered.size()});
    }
    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const auto& left, const auto& right) {
          if (javaDoubleLess(left.bin.x, right.bin.x)) {
            return true;
          }
          if (javaDoubleLess(right.bin.x, left.bin.x)) {
            return false;
          }
          return left.ordinal < right.ordinal;
        });
    merged.reserve(numMergedBins);
    for (const auto& entry : ordered) {
      merged.push_back(entry.bin);
    }
  }

  // Commit only after all validation and allocations succeed. Trimming uses no
  // further allocation and keeps the destination's random stream.
  bins_.swap(merged);
  numBins_ = mergedNumBins;
  trim();
}

uint32_t SparkNumericHistogram::nextRandomBits(uint32_t bits) {
  randomState_ = (randomState_ * kRandomMultiplier + 0xb) & kRandomMask;
  return static_cast<uint32_t>(randomState_ >> (48 - bits));
}

double SparkNumericHistogram::nextRandomDouble() {
  // Follow java.util.Random's public specification, not the XORShift generator
  // used by other Spark functions. Keep transitions sequenced explicitly.
  const uint64_t high = nextRandomBits(26);
  const uint64_t low = nextRandomBits(27);
  return static_cast<double>((high << 27) + low) / (1ULL << 53);
}

void SparkNumericHistogram::trim() {
  while (bins_.size() > numBins_) {
    double smallestGap = bins_[1].x - bins_[0].x;
    size_t closest = 0;
    uint64_t numTies = 1;
    for (size_t i = 1; i + 1 < bins_.size(); ++i) {
      const double gap = bins_[i + 1].x - bins_[i].x;
      if (gap < smallestGap) {
        smallestGap = gap;
        closest = i;
        numTies = 1;
      } else if (gap == smallestGap) {
        ++numTies;
        if (nextRandomDouble() <= 1.0 / numTies) {
          closest = i;
        }
      }
    }

    auto& left = bins_[closest];
    const auto& right = bins_[closest + 1];
    const double weight = left.y + right.y;
    // Each binary64 rounding step is observable. Compile this translation unit
    // with FP contraction disabled; do not reassociate the weighted mean.
    left.x = left.x * (left.y / weight);
    const double rightContribution = (right.x / weight) * right.y;
    left.x = left.x + rightContribution;
    left.y = weight;
    bins_.erase(bins_.begin() + closest + 1);
  }
}

int32_t SparkNumericHistogram::serializedSize(uint64_t numUsedBins) {
  // Bound the count before multiplication or narrowing. numBins itself can be
  // INT32_MAX: only actual stored/output bins consume memory and wire space.
  VELOX_USER_CHECK_LE(
      numUsedBins,
      (std::numeric_limits<int32_t>::max() - kHeaderSize) / kBinSize,
      "histogram_numeric serialized state is too large");
  return static_cast<int32_t>(kHeaderSize + kBinSize * numUsedBins);
}

int32_t SparkNumericHistogram::serializedSize() const {
  VELOX_CHECK_GE(numBins_, 2, "histogram_numeric is not initialized");
  return serializedSize(bins_.size());
}

void SparkNumericHistogram::serialize(char* output) const {
  serializedSize();
  writeBigEndian(numBins_, output);
  writeBigEndian(static_cast<int32_t>(bins_.size()), output);
  for (const auto& bin : bins_) {
    writeBigEndian(std::bit_cast<uint64_t>(bin.x), output);
    writeBigEndian(std::bit_cast<uint64_t>(bin.y), output);
  }
}

} // namespace facebook::velox::functions::aggregate::sparksql
