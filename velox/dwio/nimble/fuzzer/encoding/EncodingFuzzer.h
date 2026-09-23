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
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <random>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "folly/Random.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

namespace facebook::nimble::test {

// ============================================================================
// Data generators
// ============================================================================

template <typename T, typename RNG>
Vector<T> makeMonotonicData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    [[maybe_unused]] Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
    T current = T{0};
    for (uint32_t i = 0; i < rowCount; ++i) {
      data.push_back(current);
      current += static_cast<T>(1 + folly::Random::rand32(rng) % 10);
    }
  }
  return data;
}

template <typename T, typename RNG>
Vector<T> makeSingleValueData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  Vector<T> seed(&pool);
  seed.reserve(1);
  nimble::testing::addRandomData<T>(rng, 1, &seed, buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    data.push_back(seed[0]);
  }
  return data;
}

template <typename T, typename RNG>
Vector<T> makeDeltaBlockSortedData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    [[maybe_unused]] Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    if constexpr (std::is_signed_v<T>) {
      int64_t current{0};
      constexpr int64_t kMaxValue{
          static_cast<int64_t>(std::numeric_limits<T>::max())};
      for (uint32_t i = 0; i < rowCount; ++i) {
        data.push_back(static_cast<T>(current));
        const auto step = static_cast<int64_t>(folly::Random::rand32(rng) % 4);
        current = std::min<int64_t>(current + step, kMaxValue);
      }
    } else {
      uint64_t current{0};
      constexpr uint64_t kMaxValue{
          static_cast<uint64_t>(std::numeric_limits<T>::max())};
      for (uint32_t i = 0; i < rowCount; ++i) {
        data.push_back(static_cast<T>(current));
        const auto step = static_cast<uint64_t>(folly::Random::rand32(rng) % 4);
        current = kMaxValue - current < step ? kMaxValue : current + step;
      }
    }
  }
  return data;
}

template <typename T, typename RNG>
Vector<T> makeBoundaryMixedData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  nimble::testing::addRandomData<T>(rng, rowCount, &data, buffer);
  if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
    for (uint32_t i = 0; i < rowCount && i < 20; i += 3) {
      if (i < data.size()) {
        data[i] = std::numeric_limits<T>::min();
      }
      if (i + 1 < data.size()) {
        data[i + 1] = std::numeric_limits<T>::max();
      }
      if (i + 2 < data.size()) {
        data[i + 2] = T{0};
      }
    }
  }
  return data;
}

template <typename T, typename RNG>
Vector<T> makeMainlyConstantData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  // Generate a constant value and a few other values.
  Vector<T> seed(&pool);
  seed.reserve(2);
  nimble::testing::addRandomData<T>(rng, 2, &seed, buffer);
  // ~90% constant, ~10% other values.
  for (uint32_t i = 0; i < rowCount; ++i) {
    if (folly::Random::rand32(rng) % 10 == 0) {
      data.push_back(seed[1]);
    } else {
      data.push_back(seed[0]);
    }
  }
  return data;
}

// Values with a fixed upper-half prefix and varying lower 16 bits.
// This is the primary use case for SubIntSplitEncoding: e.g. small counters
// stored in a 64-bit field, or timestamps where the high bits are constant.
template <typename T, typename RNG>
Vector<T> makeBitStructuredData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    [[maybe_unused]] Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  if constexpr (
      std::is_arithmetic_v<T> && !std::is_same_v<T, bool> && sizeof(T) >= 4) {
    using UintType = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    UintType prefix{};
    if constexpr (sizeof(T) == 4) {
      prefix = static_cast<UintType>(0x12340000u);
    } else {
      prefix = static_cast<UintType>(0x1234567800000000ULL);
    }
    for (uint32_t i = 0; i < rowCount; ++i) {
      auto bits =
          static_cast<UintType>(prefix + (folly::Random::rand32(rng) & 0xFFFF));
      data.push_back(std::bit_cast<T>(bits));
    }
  }
  return data;
}

template <typename T, typename RNG>
Vector<T> makeFloatingPointDecimalData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    [[maybe_unused]] Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  if constexpr (std::is_floating_point_v<T>) {
    constexpr double kDivisors[] = {1.0, 10.0, 100.0, 1000.0, 10000.0};
    const auto divisor =
        kDivisors[folly::Random::rand32(rng) % std::size(kDivisors)];
    for (uint32_t i = 0; i < rowCount; ++i) {
      const auto base =
          static_cast<int32_t>((i * 37 + folly::Random::rand32(rng)) % 2001) -
          1000;
      data.push_back(static_cast<T>(static_cast<double>(base) / divisor));
    }
  }
  return data;
}

template <typename T, typename RNG>
Vector<T> makeFloatingPointSpecialData(
    velox::memory::MemoryPool& pool,
    [[maybe_unused]] RNG& rng,
    uint32_t rowCount,
    [[maybe_unused]] Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  if constexpr (std::is_floating_point_v<T>) {
    const T values[] = {
        T{0},
        -T{0},
        std::numeric_limits<T>::infinity(),
        -std::numeric_limits<T>::infinity(),
        std::numeric_limits<T>::quiet_NaN(),
        std::numeric_limits<T>::denorm_min(),
        std::numeric_limits<T>::min(),
        -std::numeric_limits<T>::min()};
    for (uint32_t i = 0; i < rowCount; ++i) {
      data.push_back(values[i % std::size(values)]);
    }
  }
  return data;
}

template <typename T, typename RNG>
Vector<T> makeFloatingPointMixedExceptionData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    [[maybe_unused]] Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  if constexpr (std::is_floating_point_v<T>) {
    for (uint32_t i = 0; i < rowCount; ++i) {
      const auto scaled =
          static_cast<T>(static_cast<int32_t>(i % 257) - 128) / T{100};
      if ((i + folly::Random::rand32(rng)) % 5 == 0) {
        data.push_back(std::nextafter(scaled, std::numeric_limits<T>::max()));
      } else {
        data.push_back(scaled);
      }
    }
  }
  return data;
}

// A sorted (non-decreasing by bit pattern) run built by accumulating small
// random steps, including 0 so values repeat and can wrap. Complements
// makeMonotonicData (strictly increasing) and, for float/double, sorts on the
// physical bit pattern so the roundtrip stays bit-exact.
template <typename T, typename RNG>
Vector<T> makeSortedData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    [[maybe_unused]] Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
    using UintType = std::conditional_t<
        sizeof(T) == 1,
        uint8_t,
        std::conditional_t<
            sizeof(T) == 2,
            uint16_t,
            std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;
    UintType accumulator = 0;
    for (uint32_t i = 0; i < rowCount; ++i) {
      accumulator = static_cast<UintType>(
          accumulator + (folly::Random::rand32(rng) % 256));
      data.push_back(std::bit_cast<T>(accumulator));
    }
  }
  return data;
}

// A handful of distinct values repeated throughout, exercising low-cardinality
// encoders (Dictionary/RLE). Works for every type.
template <typename T, typename RNG>
Vector<T> makeLowCardinalityData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  constexpr uint32_t kPoolSize = 4;
  Vector<T> valuePool(&pool);
  valuePool.reserve(kPoolSize);
  nimble::testing::addRandomData<T>(rng, kPoolSize, &valuePool, buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    data.push_back(valuePool[folly::Random::rand32(rng) % kPoolSize]);
  }
  return data;
}

// One dominant value (~90%) with a high-cardinality random exception tail
// (~10%). Unlike makeMainlyConstantData (which uses a single other value), the
// exceptions here are all distinct, stressing MainlyConstant's exception path.
template <typename T, typename RNG>
Vector<T> makeDominantValueData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  Vector<T> dominant(&pool);
  dominant.reserve(1);
  nimble::testing::addRandomData<T>(rng, 1, &dominant, buffer);
  Vector<T> exceptions(&pool);
  exceptions.reserve(rowCount);
  nimble::testing::addRandomData<T>(rng, rowCount, &exceptions, buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    data.push_back(
        folly::Random::rand32(rng) % 10 == 0 ? exceptions[i] : dominant[0]);
  }
  return data;
}

// Realistic composite (snowflake) IDs, LSB-first [sequence | worker |
// timestamp] with the top bit left 0: a near-constant high timestamp prefix, a
// few worker ids in the middle bits, and a cyclic low sequence counter. This is
// SubIntSplitEncoding's flagship target -- distinct structure in disjoint bit
// ranges -- but also stresses other encoders on composite integer data. Only
// meaningful for 32-/64-bit numeric types.
template <typename T, typename RNG>
Vector<T> makeSnowflakeData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    [[maybe_unused]] Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  if constexpr (
      std::is_arithmetic_v<T> && !std::is_same_v<T, bool> && sizeof(T) >= 4) {
    using UintType = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    // 32-bit uses a scaled analog (6/5/20) of the classic 64-bit snowflake
    // (12/10/41) so both widths get a valid composite id.
    constexpr int kSequenceBits = sizeof(T) == 4 ? 6 : 12;
    constexpr int kWorkerBits = sizeof(T) == 4 ? 5 : 10;
    constexpr int kTimestampBits = sizeof(T) == 4 ? 20 : 41;
    const UintType sequenceMask =
        static_cast<UintType>((UintType{1} << kSequenceBits) - 1);
    const UintType workerMask =
        static_cast<UintType>((UintType{1} << kWorkerBits) - 1);
    const UintType timestampMask =
        static_cast<UintType>((UintType{1} << kTimestampBits) - 1);
    const int workerShift = kSequenceBits;
    const int timestampShift = kSequenceBits + kWorkerBits;

    UintType workers[4];
    for (auto& worker : workers) {
      worker = static_cast<UintType>(folly::Random::rand64(rng)) & workerMask;
    }
    UintType timestamp =
        static_cast<UintType>(folly::Random::rand64(rng)) & timestampMask;
    UintType sequence = 0;
    for (uint32_t i = 0; i < rowCount; ++i) {
      const UintType worker = workers[folly::Random::rand32(rng) % 4];
      const UintType bits = static_cast<UintType>(
          (static_cast<UintType>(timestamp & timestampMask) << timestampShift) |
          (static_cast<UintType>(worker) << workerShift) |
          static_cast<UintType>(sequence & sequenceMask));
      data.push_back(std::bit_cast<T>(bits));

      if (sequence == sequenceMask) {
        sequence = 0;
        timestamp = static_cast<UintType>((timestamp + 1) & timestampMask);
      } else {
        ++sequence;
      }
      // Occasionally jump the timestamp forward (an idle gap) to vary the
      // high-bit run lengths.
      if (folly::Random::rand32(rng) % 16 == 0) {
        timestamp = static_cast<UintType>(
            (timestamp + (folly::Random::rand32(rng) % 16) + 1) &
            timestampMask);
        sequence = 0;
      }
    }
  }
  return data;
}

// A stream stitched from contiguous runs of several base patterns, so its
// statistics shift across the row range (regime shifts). Stresses samplers and
// selectors on non-homogeneous data. Uses only the all-type generators so it
// stays valid for every T.
template <typename T, typename RNG>
Vector<T> makeMixedRegimeData(
    velox::memory::MemoryPool& pool,
    RNG& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  Vector<T> data(&pool);
  data.reserve(rowCount);
  const uint32_t segmentCount =
      std::min<uint32_t>(2 + folly::Random::rand32(rng) % 4, rowCount);
  uint32_t remaining = rowCount;
  for (uint32_t segment = 0; segment < segmentCount; ++segment) {
    const uint32_t segmentsLeft = segmentCount - segment;
    const uint32_t segmentLength = (segmentsLeft == 1)
        ? remaining
        : std::max<uint32_t>(1, remaining / segmentsLeft);
    Vector<T> run(&pool);
    switch (folly::Random::rand32(rng) % 5) {
      case 0:
        run = makeSingleValueData<T>(pool, rng, segmentLength, buffer);
        break;
      case 1:
        run = makeLowCardinalityData<T>(pool, rng, segmentLength, buffer);
        break;
      case 2:
        run = makeMainlyConstantData<T>(pool, rng, segmentLength, buffer);
        break;
      case 3:
        run = makeDominantValueData<T>(pool, rng, segmentLength, buffer);
        break;
      default:
        run.reserve(segmentLength);
        nimble::testing::addRandomData<T>(rng, segmentLength, &run, buffer);
        break;
    }
    for (const auto& value : run) {
      data.push_back(value);
    }
    remaining -= segmentLength;
  }
  return data;
}

// ============================================================================
// Fuzzer operations
// ============================================================================

enum class FuzzerOp {
  MaterializeAll,
  MaterializeChunked,
  MaterializeOneByOne,
  SkipAndMaterialize,
  ResetAndMaterialize,
  MaterializeVariableBlocks,
};

// ============================================================================
// Core fuzzer
// ============================================================================

/// Generic encoding fuzzer parameterized by encoding class. Performs randomized
/// encode-decode-verify cycles with diverse data and access patterns.
template <typename EncodingClass>
class EncodingFuzzer {
  using T = typename EncodingClass::cppDataType;

 public:
  EncodingFuzzer(
      uint32_t iterations,
      uint32_t maxRows,
      uint32_t seed,
      bool testCompression,
      const Encoding::Options& options = {},
      uint32_t minDistinctValues = 1,
      uint32_t maxDistinctValues = std::numeric_limits<uint32_t>::max(),
      uint32_t largeInputRows = 0,
      bool realNestedSelection = false)
      : iterations_{iterations},
        maxRows_{maxRows},
        seed_{seed},
        testCompression_{testCompression},
        options_{options},
        minDistinctValues_{minDistinctValues},
        maxDistinctValues_{maxDistinctValues},
        largeInputRows_{largeInputRows},
        realNestedSelection_{realNestedSelection} {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  void run() {
    uint32_t seed = seed_;
    if (seed == 0) {
      seed = folly::Random::rand32();
    }
    LOG(INFO) << "EncodingFuzzer seed: " << seed << " encoding: "
              << toString(Encoder<EncodingClass>::encodingType())
              << " dtype: " << toString(TypeTraits<T>::dataType)
              << " iterations: " << iterations_ << " maxRows: " << maxRows_;
    std::mt19937 rng(seed);

    for (uint32_t iter = 0; iter < iterations_; ++iter) {
      buffer_ = std::make_unique<Buffer>(*pool_);
      auto datasets = generateDatasets(rng);

      for (auto& data : datasets) {
        if (data.empty() || !hasSupportedCardinality(data)) {
          continue;
        }
        auto compressionModes = testCompression_
            ? std::vector<bool>{false, true}
            : std::vector<bool>{false};

        for (bool compress : compressionModes) {
          runSingleIteration(rng, data, compress);
        }
      }
    }

    if (largeInputRows_ > 0) {
      runLargeInput(rng);
    }
  }

 private:
  // Generic large-input coverage: encode a few big datasets (crossing the
  // 4096-value decode-chunk boundary many times) and verify chunked, variable-
  // block, and skip traversals. Enabled via the largeInputRows constructor arg.
  void runLargeInput(std::mt19937& rng) {
    std::vector<Vector<T>> datasets;
    {
      Vector<T> data(pool_.get());
      data.reserve(largeInputRows_);
      nimble::testing::addRandomData<T>(
          rng, largeInputRows_, &data, buffer_.get());
      datasets.push_back(std::move(data));
    }
    datasets.push_back(
        makeMixedRegimeData<T>(*pool_, rng, largeInputRows_, buffer_.get()));
    if constexpr (
        std::is_arithmetic_v<T> && !std::is_same_v<T, bool> && sizeof(T) >= 4) {
      datasets.push_back(
          makeSnowflakeData<T>(*pool_, rng, largeInputRows_, buffer_.get()));
    }

    for (auto& data : datasets) {
      if (data.empty()) {
        continue;
      }
      Buffer encodeBuffer(*pool_);
      std::vector<velox::BufferPtr> stringBuffers;
      auto stringBufferFactory = [&](uint32_t totalLength) {
        auto& buf = stringBuffers.emplace_back(
            velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
        return buf->template asMutable<void>();
      };
      std::string_view encoded;
      try {
        encoded = Encoder<EncodingClass>::encode(
            encodeBuffer,
            data,
            CompressionType::Uncompressed,
            options_,
            realNestedSelection_);
      } catch (const NimbleUserError& e) {
        if (e.errorCode() == error_code::IncompatibleEncoding) {
          continue;
        }
        throw;
      }

      auto encoding =
          std::make_unique<EncodingClass>(*pool_, encoded, stringBufferFactory);
      EXPECT_EQ(encoding->rowCount(), data.size());

      encoding->reset();
      verifyMaterializeVariableBlocks(*encoding, data);
      encoding->reset();
      verifyMaterializeChunked(rng, *encoding, data);
      encoding->reset();
      verifySkipAndMaterialize(rng, *encoding, data);
    }
  }

  bool hasSupportedCardinality(const Vector<T>& data) const {
    if (minDistinctValues_ == 1 &&
        maxDistinctValues_ == std::numeric_limits<uint32_t>::max()) {
      return true;
    }
    std::unordered_set<T> distinctValues;
    for (const auto& value : data) {
      distinctValues.insert(value);
      if (distinctValues.size() > maxDistinctValues_) {
        return false;
      }
    }
    return distinctValues.size() >= minDistinctValues_;
  }

  std::vector<Vector<T>> generateDatasets(std::mt19937& rng) {
    std::vector<Vector<T>> datasets;
    uint32_t rowCount = 1 + folly::Random::rand32(rng) % maxRows_;

    if constexpr (
        Encoder<EncodingClass>::encodingType() == EncodingType::DeltaBlock) {
      datasets.push_back(
          makeDeltaBlockSortedData<T>(*pool_, rng, rowCount, buffer_.get()));
      datasets.push_back(
          makeSingleValueData<T>(*pool_, rng, rowCount, buffer_.get()));
      for (uint32_t sz : {1u, 2u, 3u}) {
        datasets.push_back(
            makeDeltaBlockSortedData<T>(*pool_, rng, sz, buffer_.get()));
      }
      return datasets;
    }

    // Random data.
    {
      Vector<T> data(pool_.get());
      data.reserve(rowCount);
      nimble::testing::addRandomData<T>(rng, rowCount, &data, buffer_.get());
      datasets.push_back(std::move(data));
    }

    // Single value (constant).
    datasets.push_back(
        makeSingleValueData<T>(*pool_, rng, rowCount, buffer_.get()));

    // Boundary/edge case values mixed in.
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
      datasets.push_back(
          makeBoundaryMixedData<T>(*pool_, rng, rowCount, buffer_.get()));
    }

    // Monotonic data (good for varint).
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
      datasets.push_back(
          makeMonotonicData<T>(*pool_, rng, rowCount, buffer_.get()));
    }

    // Sorted-by-bit-pattern data (non-decreasing, with repeats and wrap).
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
      datasets.push_back(
          makeSortedData<T>(*pool_, rng, rowCount, buffer_.get()));
    }

    // Mainly constant data (good for MainlyConstant encoding).
    datasets.push_back(
        makeMainlyConstantData<T>(*pool_, rng, rowCount, buffer_.get()));

    // Bit-structured data (fixed upper-half prefix, varying low 16 bits).
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
      datasets.push_back(
          makeBitStructuredData<T>(*pool_, rng, rowCount, buffer_.get()));
    }

    // Low-cardinality data (good for Dictionary/RLE).
    datasets.push_back(
        makeLowCardinalityData<T>(*pool_, rng, rowCount, buffer_.get()));

    // Dominant value with a high-cardinality random exception tail.
    datasets.push_back(
        makeDominantValueData<T>(*pool_, rng, rowCount, buffer_.get()));

    // Regime-shifting mixed stream (non-homogeneous statistics).
    datasets.push_back(
        makeMixedRegimeData<T>(*pool_, rng, rowCount, buffer_.get()));

    // Composite (snowflake) IDs: distinct structure in disjoint bit ranges.
    if constexpr (
        std::is_arithmetic_v<T> && !std::is_same_v<T, bool> && sizeof(T) >= 4) {
      datasets.push_back(
          makeSnowflakeData<T>(*pool_, rng, rowCount, buffer_.get()));
    }

    if constexpr (std::is_floating_point_v<T>) {
      datasets.push_back(
          makeFloatingPointDecimalData<T>(
              *pool_, rng, rowCount, buffer_.get()));
      datasets.push_back(
          makeFloatingPointSpecialData<T>(
              *pool_, rng, rowCount, buffer_.get()));
      datasets.push_back(
          makeFloatingPointMixedExceptionData<T>(
              *pool_, rng, rowCount, buffer_.get()));
    }

    // Small data (1-3 rows) for edge cases.
    for (uint32_t sz : {1u, 2u, 3u}) {
      Vector<T> small(pool_.get());
      small.reserve(sz);
      nimble::testing::addRandomData<T>(rng, sz, &small, buffer_.get());
      datasets.push_back(std::move(small));
    }

    return datasets;
  }

  void
  runSingleIteration(std::mt19937& rng, const Vector<T>& data, bool compress) {
    Buffer encodeBuffer(*pool_);
    std::vector<velox::BufferPtr> stringBuffers;
    auto stringBufferFactory = [&](uint32_t totalLength) {
      auto& buf = stringBuffers.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
      return buf->template asMutable<void>();
    };

    auto compressionType =
        compress ? CompressionType::Zstd : CompressionType::Uncompressed;

    std::string_view encoded;
    try {
      encoded = Encoder<EncodingClass>::encode(
          encodeBuffer, data, compressionType, options_, realNestedSelection_);
    } catch (const NimbleUserError& e) {
      if (e.errorCode() == error_code::IncompatibleEncoding) {
        return;
      }
      throw;
    }

    auto encoding =
        std::make_unique<EncodingClass>(*pool_, encoded, stringBufferFactory);

    EXPECT_EQ(encoding->dataType(), TypeTraits<T>::dataType);
    EXPECT_EQ(encoding->rowCount(), data.size());

    // Pick random operations to perform.
    std::vector<FuzzerOp> ops = {
        FuzzerOp::MaterializeAll,
        FuzzerOp::MaterializeChunked,
        FuzzerOp::MaterializeOneByOne,
        FuzzerOp::SkipAndMaterialize,
        FuzzerOp::ResetAndMaterialize,
        FuzzerOp::MaterializeVariableBlocks,
    };

    std::shuffle(ops.begin(), ops.end(), rng);
    uint32_t numOps =
        1 + folly::Random::rand32(rng) % static_cast<uint32_t>(ops.size());

    for (uint32_t opIdx = 0; opIdx < numOps; ++opIdx) {
      encoding->reset();

      switch (ops[opIdx]) {
        case FuzzerOp::MaterializeAll:
          verifyMaterializeAll(*encoding, data);
          break;
        case FuzzerOp::MaterializeChunked:
          verifyMaterializeChunked(rng, *encoding, data);
          break;
        case FuzzerOp::MaterializeOneByOne:
          verifyMaterializeOneByOne(*encoding, data);
          break;
        case FuzzerOp::SkipAndMaterialize:
          verifySkipAndMaterialize(rng, *encoding, data);
          break;
        case FuzzerOp::ResetAndMaterialize:
          verifyResetAndMaterialize(*encoding, data);
          break;
        case FuzzerOp::MaterializeVariableBlocks:
          verifyMaterializeVariableBlocks(*encoding, data);
          break;
      }
    }
  }

  void verifyMaterializeAll(Encoding& encoding, const Vector<T>& expected) {
    Vector<T> actual(pool_.get(), expected.size());
    encoding.materialize(expected.size(), actual.data());

    for (uint32_t i = 0; i < expected.size(); ++i) {
      verifyEqual(expected[i], actual[i], i);
    }
  }

  void verifyMaterializeChunked(
      std::mt19937& rng,
      Encoding& encoding,
      const Vector<T>& expected) {
    uint32_t splitPoint = 1 + folly::Random::rand32(rng) % (expected.size());
    if (splitPoint >= expected.size()) {
      splitPoint = expected.size() / 2;
    }
    if (splitPoint == 0) {
      splitPoint = 1;
    }

    Vector<T> actual(pool_.get(), expected.size());

    encoding.materialize(splitPoint, actual.data());
    for (uint32_t i = 0; i < splitPoint; ++i) {
      verifyEqual(expected[i], actual[i], i);
    }

    encoding.materialize(
        static_cast<uint32_t>(expected.size()) - splitPoint, actual.data());
    for (uint32_t i = 0; i < expected.size() - splitPoint; ++i) {
      verifyEqual(expected[splitPoint + i], actual[i], splitPoint + i);
    }
  }

  void verifyMaterializeOneByOne(
      Encoding& encoding,
      const Vector<T>& expected) {
    Vector<T> single(pool_.get(), 1);
    for (uint32_t i = 0; i < expected.size(); ++i) {
      encoding.materialize(1, single.data());
      verifyEqual(expected[i], single[0], i);
    }
  }

  void verifySkipAndMaterialize(
      std::mt19937& rng,
      Encoding& encoding,
      const Vector<T>& expected) {
    uint32_t offset = folly::Random::rand32(rng) % expected.size();
    uint32_t length =
        1 + folly::Random::rand32(rng) % (expected.size() - offset);

    encoding.skip(offset);
    Vector<T> actual(pool_.get(), length);
    encoding.materialize(length, actual.data());

    for (uint32_t i = 0; i < length; ++i) {
      verifyEqual(expected[offset + i], actual[i], offset + i);
    }
  }

  void verifyResetAndMaterialize(
      Encoding& encoding,
      const Vector<T>& expected) {
    Vector<T> first(pool_.get(), expected.size());
    encoding.materialize(expected.size(), first.data());
    for (uint32_t i = 0; i < expected.size(); ++i) {
      verifyEqual(expected[i], first[i], i);
    }

    encoding.reset();
    Vector<T> second(pool_.get(), expected.size());
    encoding.materialize(static_cast<uint32_t>(expected.size()), second.data());
    for (uint32_t i = 0; i < expected.size(); ++i) {
      verifyEqual(expected[i], second[i], i);
    }
  }

  void verifyMaterializeVariableBlocks(
      Encoding& encoding,
      const Vector<T>& expected) {
    Vector<T> actual(pool_.get(), expected.size());
    uint32_t start = 0;
    uint32_t len = 1;
    while (start + len <= expected.size()) {
      encoding.materialize(len, actual.data());
      for (uint32_t j = 0; j < len; ++j) {
        verifyEqual(expected[start + j], actual[j], start + j);
      }
      start += len;
      ++len;
    }
    uint32_t remaining = expected.size() - start;
    if (remaining > 0) {
      encoding.materialize(remaining, actual.data());
      for (uint32_t j = 0; j < remaining; ++j) {
        verifyEqual(expected[start + j], actual[j], start + j);
      }
    }
  }

  void verifyEqual(const T& expected, const T& actual, uint32_t index) {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(expected)) {
        ASSERT_TRUE(std::isnan(actual))
            << "Expected NaN at index " << index << " but got " << actual;
        using UintType =
            std::conditional_t<std::is_same_v<T, float>, uint32_t, uint64_t>;
        auto expectedBits = *reinterpret_cast<const UintType*>(&expected);
        auto actualBits = *reinterpret_cast<const UintType*>(&actual);
        ASSERT_EQ(expectedBits, actualBits)
            << "NaN bit pattern mismatch at index " << index;
      } else {
        ASSERT_EQ(expected, actual) << "Mismatch at index " << index;
      }
    } else {
      ASSERT_EQ(expected, actual) << "Mismatch at index " << index;
    }
  }

  uint32_t iterations_;
  uint32_t maxRows_;
  uint32_t seed_;
  bool testCompression_;
  Encoding::Options options_;
  uint32_t minDistinctValues_;
  uint32_t maxDistinctValues_;
  uint32_t largeInputRows_;
  bool realNestedSelection_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
};

} // namespace facebook::nimble::test
