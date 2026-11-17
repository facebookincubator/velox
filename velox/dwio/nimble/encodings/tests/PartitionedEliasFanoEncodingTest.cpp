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
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
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
    sliceBuffer_ = std::make_unique<nimble::Buffer>(*pool_);
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

  // Returns the first row of each partition of 'input'.
  template <typename T>
  static std::vector<uint32_t> partitionStarts(const std::vector<T>& input) {
    std::vector<uint32_t> starts{0};
    for (uint32_t row = 1; row < input.size(); ++row) {
      if (input[row] < input[row - 1]) {
        starts.push_back(row);
      }
    }
    return starts;
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
      auto starts = partitionStarts(input);
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

  // Encodes 'input' with the standard selection, preferring
  // PartitionedEliasFano. The selection also picks the directory encodings,
  // which can be Constant for equal partition bases.
  template <typename T>
  std::string_view encodeWithNestedSelection(const std::vector<T>& input) {
    auto policy = std::make_unique<nimble::ManualEncodingSelectionPolicy<T>>(
        std::vector<std::pair<nimble::EncodingType, float>>{
            {nimble::EncodingType::PartitionedEliasFano, 1.0},
            {nimble::EncodingType::Trivial, 100.0},
        },
        /*compressionOptions=*/std::nullopt,
        /*identifier=*/std::nullopt,
        nimble::ManualEncodingSelectionPolicyFactory::
            defaultEncodingReadFactors());
    const auto values = toVector(input);
    return nimble::EncodingFactory::encode<T>(
        std::move(policy), values, *buffer_);
  }

  // Slices rows [offset, offset + length) of 'encoded', a PartitionedEliasFano
  // stream of 'input', through the slice factory and checks the slice against
  // 'input': its encoding, row count and data type, a full read and a read
  // after a skip. A slice that starts at a partition's first row stays a
  // PartitionedEliasFano stream, which must keep only the partitions it
  // overlaps and read every row by random access. Any other slice comes back
  // wrapped in a SliceEncoding. The source bytes are cleared
  // before the slice is read, so the slice cannot refer to them. Returns the
  // sliced stream.
  template <typename T>
  std::string_view checkSlice(
      std::string_view encoded,
      const std::vector<T>& input,
      uint32_t offset,
      uint32_t length,
      const nimble::Encoding::Options& options = {}) {
    SCOPED_TRACE(fmt::format("offset={} length={}", offset, length));
    std::string source{encoded};
    const auto sliced = nimble::EncodingFactory::slice(
        source, offset, length, *sliceBuffer_, options);
    std::fill(source.begin(), source.end(), '\0');
    const bool startsPartition =
        offset == 0 || input[offset] < input[offset - 1];
    EXPECT_EQ(
        nimble::EncodingPrefix::encodingType(sliced),
        startsPartition ? nimble::EncodingType::PartitionedEliasFano
                        : nimble::EncodingType::Slice);
    auto encoding = nimble::EncodingFactory{options}.create(
        *pool_, sliced, /*stringBufferFactory=*/nullptr);
    EXPECT_EQ(encoding->rowCount(), length);
    EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<T>::dataType);

    const std::vector<T> expected(
        input.begin() + offset, input.begin() + offset + length);
    std::vector<T> output(length);
    encoding->materialize(length, output.data());
    EXPECT_EQ(output, expected);

    const uint32_t skipped = length / 2;
    encoding->reset();
    encoding->skip(skipped);
    std::vector<T> rest(length - skipped);
    encoding->materialize(length - skipped, rest.data());
    EXPECT_TRUE(
        std::equal(rest.begin(), rest.end(), expected.begin() + skipped));

    if (startsPartition) {
      const auto& partitioned =
          dynamic_cast<const nimble::PartitionedEliasFanoEncoding<T>&>(
              *encoding);
      EXPECT_EQ(partitioned.numPartitions(), countRuns(expected));
      for (uint32_t row = 0; row < length; ++row) {
        EXPECT_EQ(static_cast<T>(partitioned.valueAt(row)), expected[row]);
      }
    }
    return sliced;
  }

  // Slices every range of 'input', encoded with Trivial directories under
  // each row count form.
  template <typename T>
  void checkEverySlice(const std::vector<T>& input) {
    const auto numRows = static_cast<uint32_t>(input.size());
    for (const bool useVarintRowCount : {false, true}) {
      SCOPED_TRACE(fmt::format("useVarintRowCount={}", useVarintRowCount));
      const nimble::Encoding::Options options{
          .useVarintRowCount = useVarintRowCount,
      };
      const auto encoded = nimble::test::
          Encoder<nimble::PartitionedEliasFanoEncoding<T>>::encode(
              *buffer_,
              toVector(input),
              nimble::CompressionType::Uncompressed,
              options);
      for (uint32_t offset = 0; offset < numRows; ++offset) {
        for (uint32_t length = 1; length <= numRows - offset; ++length) {
          checkSlice(encoded, input, offset, length, options);
        }
      }
    }
  }

  // Slices random ranges of random runs of T, encoded both with Trivial
  // directories and with the standard selection. Also slices every range that
  // starts at a partition's first row, and the whole stream.
  template <typename T>
  void checkRandomSlices(uint64_t seed) {
    std::mt19937_64 random{seed};
    for (uint32_t iteration = 0; iteration < 10; ++iteration) {
      SCOPED_TRACE(fmt::format("seed={} iteration={}", seed, iteration));
      const auto input = makeRandomRuns<T>(random);
      const auto numRows = static_cast<uint32_t>(input.size());
      std::vector<std::string_view> encodings{
          nimble::test::Encoder<nimble::PartitionedEliasFanoEncoding<T>>::
              encode(*buffer_, toVector(input)),
      };
      // The selection declines PartitionedEliasFano for one partition or for
      // short partitions.
      const auto selected = encodeWithNestedSelection(input);
      if (nimble::EncodingPrefix::encodingType(selected) ==
          nimble::EncodingType::PartitionedEliasFano) {
        encodings.push_back(selected);
      }
      for (const auto encoded : encodings) {
        for (uint32_t probe = 0; probe < 50; ++probe) {
          const auto offset = static_cast<uint32_t>(random() % numRows);
          const auto length =
              1 + static_cast<uint32_t>(random() % (numRows - offset));
          checkSlice(encoded, input, offset, length);
        }
        for (const auto start : partitionStarts(input)) {
          checkSlice(encoded, input, start, numRows - start);
        }
        // Calls the encoding's slice directly, since the slice factory copies
        // a whole stream without slicing it.
        const auto whole = nimble::PartitionedEliasFanoEncoding<T>::slice(
            encoded, /*offset=*/0, numRows, *sliceBuffer_, {});
        auto encoding = nimble::EncodingFactory{}.create(
            *pool_, whole, /*stringBufferFactory=*/nullptr);
        std::vector<T> output(numRows);
        encoding->materialize(numRows, output.data());
        EXPECT_EQ(output, input);
      }
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  // Holds sliced streams apart from the encoded ones they come from.
  std::unique_ptr<nimble::Buffer> sliceBuffer_;
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

TEST_F(PartitionedEliasFanoEncodingTest, slicesRandomRangesAllTypes) {
  checkRandomSlices<int8_t>(/*seed=*/21);
  checkRandomSlices<uint8_t>(/*seed=*/22);
  checkRandomSlices<int16_t>(/*seed=*/23);
  checkRandomSlices<uint16_t>(/*seed=*/24);
  checkRandomSlices<int32_t>(/*seed=*/25);
  checkRandomSlices<uint32_t>(/*seed=*/26);
  checkRandomSlices<int64_t>(/*seed=*/27);
  checkRandomSlices<uint64_t>(/*seed=*/28);
}

TEST_F(PartitionedEliasFanoEncodingTest, slicesEveryRange) {
  constexpr auto kMin = std::numeric_limits<int64_t>::min();
  constexpr auto kMax = std::numeric_limits<int64_t>::max();
  // Partitions of one row, of duplicates, across zero and at the limits. The
  // sparse ones do not share a group with the dense ones.
  const std::vector<int64_t> input{
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      5,
      -3,
      -2,
      -1,
      0,
      1,
      -7,
      -7,
      -7,
      kMin,
      -1'000'000'000'000,
      kMax,
      -5,
      -4,
      -3,
      -2,
      -10,
      1'000'000'000'000,
      2'000'000'000'000,
  };
  const auto encoding = createEncoding(input);
  const auto& partitioned =
      dynamic_cast<const nimble::PartitionedEliasFanoEncoding<int64_t>&>(
          *encoding);
  ASSERT_EQ(partitioned.numPartitions(), 7);
  ASSERT_GE(partitioned.numGroups(), 2);
  checkEverySlice(input);

  // A narrow type, with a partition of one row between two others.
  checkEverySlice(
      std::vector<uint8_t>{
          0,
          255,
          10,
          11,
          12,
          13,
          14,
          15,
          3,
          3,
          3,
          2,
          1,
          101,
          255,
          0,
          4,
          8,
      });
}

TEST_F(PartitionedEliasFanoEncodingTest, slicesAtPartitionAndGroupBoundaries) {
  // Ten dense runs share one group and a sparse run gets its own, as in
  // groupsPartitionsBySize. The partitions are long enough for the standard
  // selection to pick PartitionedEliasFano.
  std::vector<uint64_t> input;
  for (uint32_t run = 0; run < 10; ++run) {
    for (uint64_t value = 0; value < 100; ++value) {
      input.push_back(value + run);
    }
  }
  for (uint64_t value = 0; value < 100; ++value) {
    input.push_back(value * 10'000'000'000);
  }
  const auto numRows = static_cast<uint32_t>(input.size());
  using Encoding = nimble::PartitionedEliasFanoEncoding<uint64_t>;
  const auto numGroups = [&](std::string_view encoded) {
    const auto encoding = nimble::EncodingFactory{}.create(
        *pool_, encoded, /*stringBufferFactory=*/nullptr);
    return dynamic_cast<const Encoding&>(*encoding).numGroups();
  };
  const auto encoded =
      nimble::test::Encoder<Encoding>::encode(*buffer_, toVector(input));
  const auto selected = encodeWithNestedSelection(input);
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(selected),
      nimble::EncodingType::PartitionedEliasFano);
  ASSERT_EQ(numGroups(encoded), 2);
  ASSERT_EQ(numGroups(selected), 2);

  // Rows next to each partition boundary, and the last row.
  std::vector<uint32_t> rows;
  for (const auto start : partitionStarts(input)) {
    for (const auto row : {start - 1, start, start + 1}) {
      if (row < numRows) {
        rows.push_back(row);
      }
    }
  }
  rows.push_back(numRows - 1);
  for (const auto source : {encoded, selected}) {
    for (const auto offset : rows) {
      for (const auto lastRow : rows) {
        if (lastRow >= offset) {
          checkSlice(source, input, offset, lastRow - offset + 1);
        }
      }
    }
  }

  // Keeps three partitions of the dense group.
  const auto denseSlice = checkSlice(encoded, input, /*offset=*/200, 250);
  const auto encoding = nimble::EncodingFactory{}.create(
      *pool_, denseSlice, /*stringBufferFactory=*/nullptr);
  EXPECT_EQ(dynamic_cast<const Encoding&>(*encoding).numPartitions(), 3);
  EXPECT_EQ(numGroups(denseSlice), 1);

  // Spans the end of the dense group and the start of the sparse one.
  EXPECT_EQ(numGroups(checkSlice(encoded, input, /*offset=*/900, 105)), 2);
}

TEST_F(PartitionedEliasFanoEncodingTest, slicesAcrossWholeInteriorGroup) {
  std::vector<uint64_t> input;
  const auto appendDenseGroup = [&] {
    for (uint32_t run = 0; run < 10; ++run) {
      for (uint64_t value = 0; value < 100; ++value) {
        input.push_back(value + run);
      }
    }
  };
  appendDenseGroup();
  for (uint64_t value = 0; value < 100; ++value) {
    input.push_back(value * 10'000'000'000);
  }
  appendDenseGroup();

  using Encoding = nimble::PartitionedEliasFanoEncoding<uint64_t>;
  const auto encoded =
      nimble::test::Encoder<Encoding>::encode(*buffer_, toVector(input));
  auto encoding = nimble::EncodingFactory{}.create(
      *pool_, encoded, /*stringBufferFactory=*/nullptr);
  ASSERT_EQ(dynamic_cast<const Encoding&>(*encoding).numGroups(), 3);

  // Slices the first and last groups while copying the middle group whole.
  const auto sliced = checkSlice(encoded, input, /*offset=*/200, 950);
  encoding = nimble::EncodingFactory{}.create(
      *pool_, sliced, /*stringBufferFactory=*/nullptr);
  EXPECT_EQ(dynamic_cast<const Encoding&>(*encoding).numGroups(), 3);
}

TEST_F(PartitionedEliasFanoEncodingTest, slicesCompressedDirectories) {
  // Many short partitions give directories long enough to compress.
  std::vector<uint32_t> input;
  for (uint32_t run = 0; run < 300; ++run) {
    for (uint32_t value = 0; value < 3; ++value) {
      input.push_back(run % 7 + value);
    }
  }
  const auto numRows = static_cast<uint32_t>(input.size());
  const auto encoded =
      nimble::test::Encoder<nimble::PartitionedEliasFanoEncoding<uint32_t>>::
          encode(*buffer_, toVector(input), nimble::CompressionType::Zstd);
  const auto layout = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});
  for (uint8_t child = 0; child < 2; ++child) {
    ASSERT_TRUE(layout.child(child).has_value());
    ASSERT_EQ(
        layout.child(child)->compressionType(), nimble::CompressionType::Zstd);
  }

  std::mt19937_64 random{13};
  for (uint32_t offset = 0; offset < numRows; ++offset) {
    checkSlice(encoded, input, offset, numRows - offset);
    checkSlice(
        encoded,
        input,
        offset,
        1 + static_cast<uint32_t>(random() % (numRows - offset)));
  }
}

TEST_F(PartitionedEliasFanoEncodingTest, slicesConstantBasesInsidePartition) {
  // Every partition starts at 0, so the standard selection stores the bases
  // as Constant. Materializing and re-encoding a slice that starts inside a
  // partition would need a different base for it, which a Constant layout
  // cannot hold.
  std::vector<uint32_t> input;
  for (uint32_t run = 0; run < 8; ++run) {
    for (uint32_t value = 0; value < 100; ++value) {
      input.push_back(value * 3);
    }
  }
  const auto encoded = encodeWithNestedSelection(input);
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(encoded),
      nimble::EncodingType::PartitionedEliasFano);
  const auto layout = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});
  ASSERT_TRUE(layout.child(1).has_value());
  ASSERT_EQ(layout.child(1)->encodingType(), nimble::EncodingType::Constant);

  checkSlice(encoded, input, /*offset=*/150, 400);
  checkSlice(encoded, input, /*offset=*/799, 1);
}

TEST_F(PartitionedEliasFanoEncodingTest, sliceRejectsInvalidRange) {
  using Encoding = nimble::PartitionedEliasFanoEncoding<uint32_t>;
  const auto encoded = nimble::test::Encoder<Encoding>::encode(
      *buffer_, toVector(std::vector<uint32_t>{3, 5, 1, 2}));
  NIMBLE_ASSERT_THROW(
      Encoding::slice(encoded, /*offset=*/0, /*length=*/0, *sliceBuffer_),
      "Cannot slice zero rows.");
  NIMBLE_ASSERT_THROW(
      Encoding::slice(encoded, /*offset=*/5, /*length=*/1, *sliceBuffer_),
      "Slice starts past the last row.");
  NIMBLE_ASSERT_THROW(
      Encoding::slice(encoded, /*offset=*/3, /*length=*/2, *sliceBuffer_),
      "Slice ends past the last row.");
}

TEST_F(PartitionedEliasFanoEncodingTest, sliceOfRebasedSliceIsNotSupported) {
  using Encoding = nimble::PartitionedEliasFanoEncoding<uint32_t>;
  // Partitions [3, 5], [1, 2] and [0, 4]. Slicing from the second partition
  // shifts the offset directories with a SliceEncoding value delta.
  const auto encoded = nimble::test::Encoder<Encoding>::encode(
      *buffer_, toVector(std::vector<uint32_t>{3, 5, 1, 2, 0, 4}));
  const auto sliced =
      Encoding::slice(encoded, /*offset=*/2, /*length=*/4, *sliceBuffer_);
  NIMBLE_ASSERT_THROW(
      Encoding::slice(sliced, /*offset=*/2, /*length=*/2, *sliceBuffer_),
      "Encoding Slice is not supported.");
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

TEST_F(PartitionedEliasFanoEncodingTest, rejectsMalformedStreams) {
  // Assembles a stream of 4 rows in two partitions from 'streams', the
  // directories and then the 'numGroups' group streams.
  const auto assemble = [](const std::vector<std::string_view>& streams,
                           uint32_t numGroups = 1) {
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
    nimble::varint::writeVarint(numGroups, &pos);
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
  const auto eliasFano = [&](std::vector<uint64_t> values) {
    return nimble::EliasFanoEncoding<uint64_t>::encodeNonDecreasing(
        values, *buffer_, {});
  };
  // {10, 20, 5, 9} in one group.
  const auto group = eliasFano({0, 10, 10, 14});
  const auto open = [&](const std::string& data) {
    return nimble::EncodingFactory{}.create(
        *pool_, data, /*stringBufferFactory=*/nullptr);
  };

  {
    const auto data = assemble({starts, bases, group});
    auto encoding = open(data);
    std::vector<uint32_t> output(4);
    encoding->materialize(4, output.data());
    EXPECT_EQ(output, (std::vector<uint32_t>{10, 20, 5, 9}));
  }
  {
    // A uint64 start of 1 << 32 would read as 0 through a uint32 destination.
    const auto data =
        assemble({trivialUint64({0, uint64_t{1} << 32}), bases, group});
    NIMBLE_ASSERT_FILE_THROW(open(data), "directory has the wrong data type");
  }
  {
    const auto data = assemble({starts, trivialUint64({10, 5}), group});
    NIMBLE_ASSERT_FILE_THROW(open(data), "directory has the wrong data type");
  }
  {
    const auto data = assemble({starts, bases, trivialUint64({0, 10, 10, 14})});
    NIMBLE_ASSERT_FILE_THROW(open(data), "group is not an EliasFano stream");
  }
  {
    const auto narrowGroup =
        nimble::EliasFanoEncoding<uint32_t>::encodeNonDecreasing(
            std::vector<uint32_t>{0, 10, 10, 14}, *buffer_, {});
    const auto data = assemble({starts, bases, narrowGroup});
    NIMBLE_ASSERT_FILE_THROW(open(data), "group has the wrong data type");
  }
  {
    // One group per partition reads the same rows.
    const auto data =
        assemble({starts, bases, eliasFano({0, 10}), eliasFano({0, 4})}, 2);
    auto encoding = open(data);
    std::vector<uint32_t> output(4);
    encoding->materialize(4, output.data());
    EXPECT_EQ(output, (std::vector<uint32_t>{10, 20, 5, 9}));
  }
  {
    const auto data = assemble({starts, bases, eliasFano({0, 10, 10, 14, 20})});
    NIMBLE_ASSERT_FILE_THROW(open(data), "group extends past the last row");
  }
  {
    const auto data = assemble({starts, bases, eliasFano({0, 10, 10})});
    NIMBLE_ASSERT_FILE_THROW(
        open(data), "group must end at a partition boundary");
  }
  {
    const auto data = assemble({starts, bases, eliasFano({0, 10})});
    NIMBLE_ASSERT_FILE_THROW(open(data), "groups must cover every row");
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
