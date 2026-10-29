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
#include "velox/dwio/nimble/encodings/PartitionedEliasFanoEncoding.h"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <random>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

using namespace facebook;

namespace {

// Returns the number of maximal ascending runs in 'values'.
template <typename T>
uint32_t countRuns(const std::vector<T>& values) {
  uint32_t runs{values.empty() ? 0u : 1u};
  for (size_t i = 1; i < values.size(); ++i) {
    runs += values[i] < values[i - 1] ? 1 : 0;
  }
  return runs;
}

// Returns 'numRuns' ascending runs of random lengths and starting values.
template <typename T>
std::vector<T> makeRuns(uint32_t numRuns, uint64_t seed) {
  std::mt19937_64 random{seed};
  std::uniform_int_distribution<uint32_t> lengthDistribution{64, 200};
  std::uniform_int_distribution<int64_t> startDistribution{
      std::is_signed_v<T> ? -1'000 : 0, 1'000};
  std::uniform_int_distribution<int64_t> deltaDistribution{0, 50};
  std::vector<T> values;
  for (uint32_t run = 0; run < numRuns; ++run) {
    auto value = static_cast<T>(startDistribution(random));
    for (auto length = lengthDistribution(random); length > 0; --length) {
      values.push_back(value);
      value = static_cast<T>(value + deltaDistribution(random));
    }
  }
  return values;
}

// Returns ascending runs of 1 to 300 rows. Runs start at the type's extremes,
// near zero (so signed runs cross it) or anywhere, and grow by random steps
// without wrapping.
template <typename T>
std::vector<T> makeRandomRuns(std::mt19937_64& random) {
  constexpr auto kMin = std::numeric_limits<T>::min();
  constexpr auto kMax = std::numeric_limits<T>::max();
  std::uniform_int_distribution<uint32_t> numRunsDistribution{1, 20};
  std::uniform_int_distribution<uint32_t> lengthDistribution{1, 300};
  std::vector<T> values;
  const auto numRuns = numRunsDistribution(random);
  for (uint32_t run = 0; run < numRuns; ++run) {
    T value{};
    switch (random() % 4) {
      case 0:
        value = kMin;
        break;
      case 1:
        value = static_cast<T>(kMax - static_cast<T>(random() % 100));
        break;
      case 2:
        value = std::is_signed_v<T>
            ? static_cast<T>(-static_cast<int64_t>(random() % 50))
            : T{0};
        break;
      default:
        value = static_cast<T>(random());
    }
    const auto step = static_cast<T>(1 + random() % 16);
    for (auto length = lengthDistribution(random); length > 0; --length) {
      values.push_back(value);
      if (random() % 3 != 0) {
        value = value > kMax - step ? kMax : static_cast<T>(value + step);
      }
    }
  }
  return values;
}

} // namespace

class PartitionedEliasFanoEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> toVector(const std::vector<T>& input) {
    nimble::Vector<T> values{pool_.get()};
    values.reserve(input.size());
    for (const auto value : input) {
      values.push_back(value);
    }
    return values;
  }

  template <typename T>
  std::unique_ptr<nimble::Encoding> createEncoding(
      const std::vector<T>& input) {
    return nimble::test::Encoder<nimble::PartitionedEliasFanoEncoding<T>>::
        createEncoding(*buffer_, toVector(input), nullptr);
  }

  template <typename T>
  void roundTrip(const std::vector<T>& input) {
    auto encoding = createEncoding(input);
    ASSERT_EQ(
        encoding->encodingType(), nimble::EncodingType::PartitionedEliasFano);
    const auto& partitioned =
        dynamic_cast<const nimble::PartitionedEliasFanoEncoding<T>&>(*encoding);
    EXPECT_EQ(partitioned.numPartitions(), countRuns(input));

    std::vector<T> output(input.size());
    encoding->materialize(static_cast<uint32_t>(input.size()), output.data());
    EXPECT_EQ(output, input);

    std::vector<T> randomOutput(input.size());
    for (uint32_t row = 0; row < input.size(); ++row) {
      randomOutput[row] = partitioned.valueAt(row);
    }
    EXPECT_EQ(randomOutput, input);
  }

  // Encodes random runs of T and checks every read path against the input,
  // including lowerBound() against std::lower_bound on random ranges within
  // one partition.
  template <typename T>
  void checkRandomRuns(uint64_t seed) {
    using Encoding = nimble::PartitionedEliasFanoEncoding<T>;
    using PhysicalType = typename nimble::TypeTraits<T>::physicalType;
    std::mt19937_64 random{seed};
    for (uint32_t iteration = 0; iteration < 20; ++iteration) {
      SCOPED_TRACE(fmt::format("seed={} iteration={}", seed, iteration));
      const auto input = makeRandomRuns<T>(random);
      const auto numRows = static_cast<uint32_t>(input.size());
      const auto encoded =
          nimble::test::Encoder<Encoding>::encode(*buffer_, toVector(input));
      auto encoding = nimble::EncodingFactory{}.create(
          *pool_, encoded, /*stringBufferFactory=*/nullptr);
      const auto& partitioned = dynamic_cast<const Encoding&>(*encoding);
      ASSERT_EQ(partitioned.numPartitions(), countRuns(input));

      std::vector<T> output(numRows);
      encoding->materialize(numRows, output.data());
      ASSERT_EQ(output, input);

      const auto skipped = static_cast<uint32_t>(random() % numRows);
      encoding->reset();
      encoding->skip(skipped);
      std::vector<T> rest(numRows - skipped);
      encoding->materialize(numRows - skipped, rest.data());
      ASSERT_TRUE(
          std::equal(rest.begin(), rest.end(), input.begin() + skipped));

      auto view = nimble::createEncodingView(encoded, pool_.get());
      const auto& typedView =
          dynamic_cast<const nimble::TypedEncodingView<T>&>(*view);
      for (uint32_t read = 0; read < 50; ++read) {
        const auto row = static_cast<uint32_t>(random() % numRows);
        ASSERT_EQ(static_cast<T>(partitioned.valueAt(row)), input[row]);
        ASSERT_EQ(typedView.readAt(row), input[row]);
        const auto length =
            static_cast<uint32_t>(random() % (numRows - row + 1));
        std::vector<T> range(length);
        partitioned.materializeAt(
            row, length, reinterpret_cast<PhysicalType*>(range.data()));
        ASSERT_TRUE(
            std::equal(range.begin(), range.end(), input.begin() + row));
        std::vector<T> viewRange(length);
        typedView.read(
            row, length, reinterpret_cast<PhysicalType*>(viewRange.data()));
        ASSERT_EQ(viewRange, range);
      }

      // Partition p covers rows [starts[p], starts[p + 1]).
      std::vector<uint32_t> starts{0};
      for (uint32_t row = 1; row < numRows; ++row) {
        if (input[row] < input[row - 1]) {
          starts.push_back(row);
        }
      }
      starts.push_back(numRows);
      for (uint32_t probe = 0; probe < 50; ++probe) {
        const auto partition = random() % (starts.size() - 1);
        const auto size = starts[partition + 1] - starts[partition];
        const auto begin =
            starts[partition] + static_cast<uint32_t>(random() % size);
        const auto end = begin +
            static_cast<uint32_t>(
                             random() % (starts[partition + 1] - begin + 1));
        T value{};
        switch (random() % 4) {
          case 0:
            value = std::numeric_limits<T>::min();
            break;
          case 1:
            value = std::numeric_limits<T>::max();
            break;
          case 2:
            value = input[starts[partition] + random() % size];
            break;
          default:
            value = static_cast<T>(random());
        }
        const auto expected = static_cast<uint32_t>(
            std::lower_bound(
                input.begin() + begin, input.begin() + end, value) -
            input.begin());
        PhysicalType found{};
        ASSERT_EQ(
            partitioned.lowerBound(
                begin, end, static_cast<PhysicalType>(value), found),
            expected);
        if (expected < end) {
          ASSERT_EQ(static_cast<T>(found), input[expected]);
        }
        T viewFound{};
        ASSERT_EQ(typedView.lowerBound(begin, end, value, viewFound), expected);
        if (expected < end) {
          ASSERT_EQ(viewFound, input[expected]);
        }
      }
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(PartitionedEliasFanoEncodingTest, roundTripRandomPartitions) {
  roundTrip(makeRuns<uint32_t>(/*numRuns=*/200, /*seed=*/1));
  roundTrip(makeRuns<uint64_t>(/*numRuns=*/200, /*seed=*/2));
  roundTrip(makeRuns<int32_t>(/*numRuns=*/200, /*seed=*/3));
  roundTrip(makeRuns<int64_t>(/*numRuns=*/200, /*seed=*/4));
  roundTrip(makeRuns<int16_t>(/*numRuns=*/50, /*seed=*/5));
}

TEST_F(PartitionedEliasFanoEncodingTest, roundTripDocumentedExample) {
  const std::vector<uint32_t> input{10, 20, 5, 9};
  roundTrip(input);
  const auto encoding = createEncoding(input);
  EXPECT_EQ(
      dynamic_cast<const nimble::PartitionedEliasFanoEncoding<uint32_t>&>(
          *encoding)
          .numGroups(),
      1);
}

TEST_F(PartitionedEliasFanoEncodingTest, roundTripSinglePartition) {
  roundTrip<uint64_t>({3, 5, 5, 9, 1'000, 1'000'000});
  roundTrip<int32_t>({-9, -9, 0, 7});
}

TEST_F(PartitionedEliasFanoEncodingTest, roundTripAllDescending) {
  roundTrip<uint32_t>({100, 90, 50, 49, 1, 0});
  roundTrip<int64_t>({5, 0, -5, -1'000});
}

TEST_F(PartitionedEliasFanoEncodingTest, roundTripOneRow) {
  roundTrip<uint64_t>({42});
  roundTrip<int8_t>({-3});
}

TEST_F(PartitionedEliasFanoEncodingTest, roundTripDuplicates) {
  roundTrip<uint32_t>({7, 7, 7, 7});
  roundTrip<uint32_t>({1, 4, 4, 2, 2, 2, 4, 1, 1});
}

TEST_F(PartitionedEliasFanoEncodingTest, roundTripLimits) {
  roundTrip<uint64_t>({
      std::numeric_limits<uint64_t>::max() - 1,
      std::numeric_limits<uint64_t>::max(),
      0,
      1,
  });
  roundTrip<int64_t>({
      std::numeric_limits<int64_t>::max() - 1,
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::min() + 1,
      -1,
      0,
  });
  roundTrip<int8_t>({
      std::numeric_limits<int8_t>::max(),
      std::numeric_limits<int8_t>::min(),
      std::numeric_limits<int8_t>::max(),
  });
}

TEST_F(PartitionedEliasFanoEncodingTest, splitsGroupsOnShiftedOverflow) {
  const std::vector<uint64_t> input{
      0,
      std::numeric_limits<uint64_t>::max(),
      0,
      std::numeric_limits<uint64_t>::max(),
  };
  roundTrip(input);
  const auto encoding = createEncoding(input);
  EXPECT_EQ(
      dynamic_cast<const nimble::PartitionedEliasFanoEncoding<uint64_t>&>(
          *encoding)
          .numGroups(),
      2);
}

TEST_F(PartitionedEliasFanoEncodingTest, groupsPartitionsBySize) {
  // Ten dense runs share one list. A sparse run after a dense one gets its own
  // list, which keeps the dense run's narrow lower bits.
  std::vector<uint32_t> dense;
  dense.reserve(1'000);
  for (uint32_t run = 0; run < 10; ++run) {
    for (uint32_t value = 0; value < 100; ++value) {
      dense.push_back(value);
    }
  }
  std::vector<uint64_t> mixed;
  mixed.reserve(1'010);
  for (uint64_t value = 0; value < 1'000; ++value) {
    mixed.push_back(value);
  }
  for (uint64_t value = 0; value < 10; ++value) {
    mixed.push_back(value * 1'000'000'000'000);
  }
  roundTrip(dense);
  roundTrip(mixed);

  const auto denseEncoding = createEncoding(dense);
  EXPECT_EQ(
      dynamic_cast<const nimble::PartitionedEliasFanoEncoding<uint32_t>&>(
          *denseEncoding)
          .numGroups(),
      1);
  const auto mixedEncoding = createEncoding(mixed);
  EXPECT_EQ(
      dynamic_cast<const nimble::PartitionedEliasFanoEncoding<uint64_t>&>(
          *mixedEncoding)
          .numGroups(),
      2);
}

TEST_F(
    PartitionedEliasFanoEncodingTest,
    estimateRejectsFewerThanTwoPartitions) {
  const auto values = toVector(std::vector<uint64_t>{1, 5, 9});
  EXPECT_FALSE(
      nimble::PartitionedEliasFanoEncoding<uint64_t>::estimateSize(
          std::span<const uint64_t>{})
          .has_value());
  EXPECT_FALSE(
      nimble::PartitionedEliasFanoEncoding<uint64_t>::estimateSize(
          std::span<const uint64_t>{values.data(), values.size()})
          .has_value());
}

TEST_F(PartitionedEliasFanoEncodingTest, estimateRejectsShortPartitions) {
  // Two 63-row runs average below the 64 rows a partition needs; two 64-row
  // runs do not.
  const auto estimate = [&](uint32_t runLength) {
    std::vector<uint64_t> input;
    for (uint32_t run = 0; run < 2; ++run) {
      for (uint64_t value = 0; value < runLength; ++value) {
        input.push_back(value);
      }
    }
    const auto values = toVector(input);
    return nimble::PartitionedEliasFanoEncoding<uint64_t>::estimateSize(
        std::span<const uint64_t>{values.data(), values.size()});
  };
  EXPECT_FALSE(estimate(63).has_value());
  EXPECT_TRUE(estimate(64).has_value());
}

TEST_F(PartitionedEliasFanoEncodingTest, resetSkipAndMaterialize) {
  const std::vector<uint32_t> input{10, 11, 12, 0, 1, 2, 5, 6, 7};
  auto encoding = createEncoding(input);

  encoding->skip(2);
  std::vector<uint32_t> output(4);
  encoding->materialize(static_cast<uint32_t>(output.size()), output.data());
  EXPECT_EQ(output, (std::vector<uint32_t>{12, 0, 1, 2}));

  encoding->reset();
  output.resize(input.size());
  encoding->materialize(static_cast<uint32_t>(output.size()), output.data());
  EXPECT_EQ(output, input);
}

TEST_F(PartitionedEliasFanoEncodingTest, lowerBoundWithinPartition) {
  // Partitions: [0, 4) = {10, 20, 20, 40}, [4, 7) = {5, 6, 30},
  // [7, 9) = {-3, 8}.
  const std::vector<int32_t> input{10, 20, 20, 40, 5, 6, 30, -3, 8};
  const auto encoding = createEncoding(input);
  const auto& partitioned =
      dynamic_cast<const nimble::PartitionedEliasFanoEncoding<int32_t>&>(
          *encoding);

  struct Probe {
    uint32_t begin;
    uint32_t end;
    int32_t value;
    uint32_t expected;
  };
  const auto lowerBound = [&](uint32_t begin, uint32_t end, int32_t value) {
    // The physical type of int32_t values is uint32_t.
    uint32_t found{0};
    return partitioned.lowerBound(
        begin, end, static_cast<uint32_t>(value), found);
  };
  for (const auto& probe : std::vector<Probe>{
           {0, 4, -100, 0},
           {0, 4, 10, 0},
           {0, 4, 11, 1},
           {0, 4, 20, 1},
           {0, 4, 21, 3},
           {0, 4, 41, 4},
           {1, 3, 20, 1},
           {1, 3, 21, 3},
           {4, 7, 5, 4},
           {4, 7, 7, 6},
           {4, 7, 31, 7},
           {5, 6, 6, 5},
           {5, 6, 7, 6},
           {7, 9, -4, 7},
           {7, 9, 0, 8},
           {7, 9, 9, 9},
           {3, 3, 0, 3},
       }) {
    SCOPED_TRACE(
        ::testing::Message() << "begin=" << probe.begin << ", end=" << probe.end
                             << ", value=" << probe.value);
    EXPECT_EQ(lowerBound(probe.begin, probe.end, probe.value), probe.expected);
  }
}

TEST_F(PartitionedEliasFanoEncodingTest, lowerBoundReturnsFoundValue) {
  // Partitions: [0, 4) = {10, 20, 20, 40}, [4, 7) = {5, 6, 30}.
  const std::vector<int32_t> input{10, 20, 20, 40, 5, 6, 30};
  const auto encoding = createEncoding(input);
  const auto& partitioned =
      dynamic_cast<const nimble::PartitionedEliasFanoEncoding<int32_t>&>(
          *encoding);

  for (const auto& [begin, end, value] :
       std::vector<std::tuple<uint32_t, uint32_t, int32_t>>{
           {0, 4, 0},
           {0, 4, 15},
           {1, 4, 0},
           {1, 3, 20},
           {4, 7, 6},
           {5, 7, 1},
           {4, 7, 7},
       }) {
    SCOPED_TRACE(
        ::testing::Message()
        << "begin=" << begin << ", end=" << end << ", value=" << value);
    // The physical type of int32_t values is uint32_t.
    uint32_t found{0};
    const auto row =
        partitioned.lowerBound(begin, end, static_cast<uint32_t>(value), found);
    ASSERT_LT(row, end);
    EXPECT_EQ(static_cast<int32_t>(found), input[row]);
  }
}

TEST_F(PartitionedEliasFanoEncodingTest, lowerBoundRejectsInvalidRanges) {
  // Partitions: [0, 4) = {10, 20, 20, 40}, [4, 7) = {5, 6, 30}.
  const std::vector<uint32_t> input{10, 20, 20, 40, 5, 6, 30};
  const auto encoding = createEncoding(input);
  const auto& partitioned =
      dynamic_cast<const nimble::PartitionedEliasFanoEncoding<uint32_t>&>(
          *encoding);
  uint32_t found{0};
  NIMBLE_ASSERT_THROW(
      partitioned.lowerBound(3, 2, 0, found), "Invalid lowerBound range");
  NIMBLE_ASSERT_THROW(
      partitioned.lowerBound(4, 8, 0, found), "Invalid lowerBound range");
  NIMBLE_ASSERT_THROW(
      partitioned.lowerBound(2, 5, 0, found), "range crosses a partition");
  EXPECT_EQ(partitioned.lowerBound(7, 7, 0, found), 7);
}

TEST_F(PartitionedEliasFanoEncodingTest, findsPartitionsOfUnevenLengths) {
  // One long run followed by many short ones puts many partitions in one
  // lookup bucket.
  std::vector<uint32_t> input;
  input.reserve(10'200);
  for (uint32_t value = 0; value < 10'000; ++value) {
    input.push_back(value);
  }
  for (uint32_t run = 0; run < 100; ++run) {
    input.push_back(5);
    input.push_back(6);
  }
  roundTrip(input);
  const auto encoding = createEncoding(input);
  const auto& partitioned =
      dynamic_cast<const nimble::PartitionedEliasFanoEncoding<uint32_t>&>(
          *encoding);
  for (uint32_t row = 10'000; row < input.size(); row += 2) {
    uint32_t found{0};
    EXPECT_EQ(partitioned.lowerBound(row, row + 2, 6, found), row + 1);
    EXPECT_EQ(found, 6);
  }
}

TEST_F(PartitionedEliasFanoEncodingTest, factoryRoundTripWithNestedSelection) {
  const auto input = makeRuns<uint64_t>(/*numRuns=*/64, /*seed=*/7);
  const auto values = toVector(input);
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint64_t>>(
          std::vector<std::pair<nimble::EncodingType, float>>{
              {nimble::EncodingType::PartitionedEliasFano, 1.0},
              {nimble::EncodingType::FixedBitWidth, 10.0},
              {nimble::EncodingType::Trivial, 10.0},
          },
          std::nullopt,
          std::nullopt);
  const auto encoded = nimble::EncodingFactory::encode<uint64_t>(
      std::move(policy), values, *buffer_);
  auto encoding = nimble::EncodingFactory{}.create(
      *pool_, encoded, /*stringBufferFactory=*/nullptr);
  std::vector<uint64_t> output(input.size());
  encoding->materialize(static_cast<uint32_t>(input.size()), output.data());

  EXPECT_EQ(
      encoding->encodingType(), nimble::EncodingType::PartitionedEliasFano);
  EXPECT_EQ(output, input);
  EXPECT_NE(
      encoding->debugString(0).find(
          fmt::format("partitions={}", countRuns(input))),
      std::string::npos);
}

TEST_F(PartitionedEliasFanoEncodingTest, estimateTracksEncodedSize) {
  const auto input = makeRuns<uint64_t>(/*numRuns=*/500, /*seed=*/8);
  const auto values = toVector(input);
  const auto estimate =
      nimble::PartitionedEliasFanoEncoding<uint64_t>::estimateSize(
          std::span<const uint64_t>{values.data(), values.size()});
  ASSERT_TRUE(estimate.has_value());

  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint64_t>>(
          std::vector<std::pair<nimble::EncodingType, float>>{
              {nimble::EncodingType::PartitionedEliasFano, 1.0},
              {nimble::EncodingType::FixedBitWidth, 10.0},
          },
          std::nullopt,
          std::nullopt);
  const auto encoded = nimble::EncodingFactory::encode<uint64_t>(
      std::move(policy), values, *buffer_);
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(encoded),
      nimble::EncodingType::PartitionedEliasFano);
  EXPECT_NEAR(
      static_cast<double>(*estimate),
      static_cast<double>(encoded.size()),
      0.05 * encoded.size());
}

TEST_F(PartitionedEliasFanoEncodingTest, rejectsTruncatedStreams) {
  const auto encoded =
      nimble::test::Encoder<nimble::PartitionedEliasFanoEncoding<uint32_t>>::
          encode(*buffer_, toVector<uint32_t>({10, 20, 5, 9, 1, 2}));
  const auto prefixSize =
      nimble::EncodingPrefix::prefixSize(encoded, /*useVarint=*/false);
  for (auto size = prefixSize; size < encoded.size(); ++size) {
    SCOPED_TRACE(size);
    // Copies the bytes so a read past them is a sanitizer error.
    const std::string truncated{encoded.substr(0, size)};
    NIMBLE_ASSERT_THROW(
        nimble::EncodingFactory{}.create(
            *pool_, truncated, /*stringBufferFactory=*/nullptr),
        "Truncated");
  }
}

TEST_F(PartitionedEliasFanoEncodingTest, rejectsOverflowingVarint) {
  const auto encoded =
      nimble::test::Encoder<nimble::PartitionedEliasFanoEncoding<uint32_t>>::
          encode(*buffer_, toVector<uint32_t>({10, 20, 5, 9}));
  // The fifth byte of a uint32 varint may hold only four bits.
  const std::string overflowing =
      std::string{encoded.substr(0, nimble::EncodingPrefix::kFixedPrefixSize)} +
      std::string{"\xff\xff\xff\xff\x1f", 5};
  NIMBLE_ASSERT_THROW(
      nimble::EncodingFactory{}.create(
          *pool_, overflowing, /*stringBufferFactory=*/nullptr),
      "Invalid uint32 varint");
}

TEST_F(PartitionedEliasFanoEncodingTest, rejectsStreamsOfWrongType) {
  // Assembles the stream of {10, 20, 5, 9}: two partitions in one group.
  const auto assemble = [](const std::vector<std::string_view>& streams) {
    std::string data(64, '\0');
    for (const auto stream : streams) {
      data.resize(data.size() + stream.size() + 5);
    }
    char* pos = data.data();
    nimble::EncodingPrefix::serialize(
        nimble::EncodingType::PartitionedEliasFano,
        nimble::DataType::Uint32,
        /*rowCount=*/4,
        /*useVarint=*/false,
        pos);
    nimble::varint::writeVarint(uint32_t{2}, &pos);
    nimble::varint::writeVarint(uint32_t{1}, &pos);
    for (const auto stream : streams) {
      nimble::encoding::writeVarintString(stream, pos);
    }
    data.resize(pos - data.data());
    return data;
  };
  const auto trivialUint32 = [&](std::initializer_list<uint32_t> values) {
    return nimble::test::Encoder<nimble::TrivialEncoding<uint32_t>>::encode(
        *buffer_, toVector<uint32_t>(values));
  };
  const auto trivialUint64 = [&](std::initializer_list<uint64_t> values) {
    return nimble::test::Encoder<nimble::TrivialEncoding<uint64_t>>::encode(
        *buffer_, toVector<uint64_t>(values));
  };
  const auto starts = trivialUint32({0, 2});
  const auto bases = trivialUint32({10, 5});
  const auto groupStarts = trivialUint32({0});
  const auto group = nimble::EliasFanoEncoding<uint64_t>::encodeNonDecreasing(
      std::vector<uint64_t>{0, 10, 10, 14}, *buffer_, {});
  const auto open = [&](const std::string& data) {
    return nimble::EncodingFactory{}.create(
        *pool_, data, /*stringBufferFactory=*/nullptr);
  };

  {
    const auto data = assemble({starts, bases, groupStarts, group});
    auto encoding = open(data);
    std::vector<uint32_t> output(4);
    encoding->materialize(4, output.data());
    EXPECT_EQ(output, (std::vector<uint32_t>{10, 20, 5, 9}));
  }
  {
    // A uint64 start of 1 << 32 would read as 0 through a uint32 destination.
    const auto data = assemble(
        {trivialUint64({0, uint64_t{1} << 32}), bases, groupStarts, group});
    NIMBLE_ASSERT_FILE_THROW(open(data), "directory has the wrong data type");
  }
  {
    const auto data =
        assemble({starts, trivialUint64({10, 5}), groupStarts, group});
    NIMBLE_ASSERT_FILE_THROW(open(data), "directory has the wrong data type");
  }
  {
    const auto data = assemble({starts, bases, trivialUint64({0}), group});
    NIMBLE_ASSERT_FILE_THROW(open(data), "directory has the wrong data type");
  }
  {
    const auto data =
        assemble({starts, bases, groupStarts, trivialUint64({0, 10, 10, 14})});
    NIMBLE_ASSERT_FILE_THROW(open(data), "group is not an EliasFano stream");
  }
  {
    const auto narrowGroup =
        nimble::EliasFanoEncoding<uint32_t>::encodeNonDecreasing(
            std::vector<uint32_t>{0, 10, 10, 14}, *buffer_, {});
    const auto data = assemble({starts, bases, groupStarts, narrowGroup});
    NIMBLE_ASSERT_FILE_THROW(open(data), "group has the wrong data type");
  }
}

TEST_F(PartitionedEliasFanoEncodingTest, randomRunsAllTypes) {
  checkRandomRuns<int8_t>(/*seed=*/11);
  checkRandomRuns<uint8_t>(/*seed=*/12);
  checkRandomRuns<int16_t>(/*seed=*/13);
  checkRandomRuns<uint16_t>(/*seed=*/14);
  checkRandomRuns<int32_t>(/*seed=*/15);
  checkRandomRuns<uint32_t>(/*seed=*/16);
  checkRandomRuns<int64_t>(/*seed=*/17);
  checkRandomRuns<uint64_t>(/*seed=*/18);
}

TEST_F(PartitionedEliasFanoEncodingTest, estimateTracksSignedBasesAcrossZero) {
  // Bases alternate between -1 and 1. Their order-preserving forms are close,
  // but the stored physical values span the whole uint64 range.
  std::vector<int64_t> input;
  input.reserve(300 * 64);
  for (int64_t run = 0; run < 300; ++run) {
    const int64_t base = run % 2 == 0 ? -1 : 1;
    for (int64_t row = 0; row < 64; ++row) {
      input.push_back(base + row);
    }
  }
  const auto values = toVector(input);
  const auto estimate =
      nimble::PartitionedEliasFanoEncoding<int64_t>::estimateSize(
          std::span<const uint64_t>{
              reinterpret_cast<const uint64_t*>(values.data()), values.size()});
  ASSERT_TRUE(estimate.has_value());

  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<int64_t>>(
          std::vector<std::pair<nimble::EncodingType, float>>{
              {nimble::EncodingType::PartitionedEliasFano, 1.0},
              {nimble::EncodingType::FixedBitWidth, 10.0},
          },
          std::nullopt,
          std::nullopt);
  const auto encoded = nimble::EncodingFactory::encode<int64_t>(
      std::move(policy), values, *buffer_);
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(encoded),
      nimble::EncodingType::PartitionedEliasFano);
  EXPECT_NEAR(
      static_cast<double>(*estimate),
      static_cast<double>(encoded.size()),
      0.05 * encoded.size());
}

TEST(PartitionGroupPlannerTest, mergesDenseAndRejectsOversizedGroups) {
  const nimble::Encoding::Options options;
  nimble::detail::partitioned_elias_fano::PartitionGroupPlanner planner{
      options};
  EXPECT_TRUE(planner.add(/*numRows=*/100, /*range=*/99).newGroup);
  const auto merged = planner.add(/*numRows=*/100, /*range=*/99);
  EXPECT_FALSE(merged.newGroup);
  EXPECT_EQ(merged.baseShift, 99);
  // A sparse partition after dense ones keeps its own list.
  EXPECT_TRUE(
      planner.add(/*numRows=*/10, /*range=*/uint64_t{1} << 60).newGroup);
  EXPECT_TRUE(planner.totalBytes().has_value());
  // Four billion rows spread over 2^62 need more than 4 GB.
  planner.add(/*numRows=*/4'000'000'000, /*range=*/uint64_t{1} << 62);
  EXPECT_FALSE(planner.totalBytes().has_value());
}
