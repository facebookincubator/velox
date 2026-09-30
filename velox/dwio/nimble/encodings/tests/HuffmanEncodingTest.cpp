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
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <utility>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

namespace facebook::nimble::test {
namespace {

class HuffmanEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  template <typename T>
  std::unique_ptr<Encoding> encode(const Vector<T>& values) {
    return Encoder<HuffmanEncoding<T>>::createEncoding(
        *buffer_, values, nullptr);
  }

  template <typename T>
  std::string_view encodeData(const Vector<T>& values) {
    return Encoder<HuffmanEncoding<T>>::encode(*buffer_, values);
  }

  template <typename T>
  void verifyFactoryRoundTrip() {
    Vector<T> values{pool_.get()};
    constexpr std::array<T, 5> alphabet{
        std::numeric_limits<T>::lowest(),
        T{0},
        T{1},
        std::numeric_limits<T>::max(),
        static_cast<T>(std::numeric_limits<T>::max() / 2)};
    for (uint32_t row = 0; row < 1025; ++row) {
      values.push_back(alphabet[row % alphabet.size()]);
    }

    const auto encoded = encodeData(values);
    for (const auto useLegacy : {false, true}) {
      SCOPED_TRACE(fmt::format("legacy={}", useLegacy));
      std::unique_ptr<Encoding> encoding = useLegacy
          ? legacy::EncodingFactory().create(
                *pool_, encoded, nullptr, Encoding::Options{})
          : EncodingFactory().create(
                *pool_, encoded, nullptr, Encoding::Options{});
      EXPECT_EQ(encoding->encodingType(), EncodingType::Huffman);
      EXPECT_EQ(encoding->dataType(), TypeTraits<T>::dataType);
      EXPECT_EQ(encoding->rowCount(), values.size());

      Vector<T> result{pool_.get(), values.size()};
      encoding->materialize(
          static_cast<uint32_t>(values.size()), result.data());
      EXPECT_TRUE(std::equal(result.begin(), result.end(), values.begin()));
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
};

class TestReader {
 public:
  velox::BufferPtr& nullsInReadRange() {
    return nullsInReadRange_;
  }

  const uint64_t* rawNullsInReadRange() const {
    return nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  }

  bool returnReaderNulls() const {
    return false;
  }

 private:
  velox::BufferPtr nullsInReadRange_;
};

template <typename T>
class IntegralReadWithVisitor {
 public:
  using DataType = T;
  using Extract = std::nullptr_t;

  explicit IntegralReadWithVisitor(std::vector<vector_size_t> rows)
      : rows_{std::move(rows)} {}

  TestReader& reader() {
    return reader_;
  }

  vector_size_t numRows() const {
    return rows_.size();
  }

  vector_size_t rowAt(vector_size_t index) const {
    return rows_[index];
  }

  vector_size_t currentRow() const {
    return rowAt(rowIndex_);
  }

  void process(T value, bool& atEnd) {
    values_.push_back(value);
    addRowIndex(1);
    atEnd = this->atEnd();
  }

  void processNull(bool& atEnd) {
    addRowIndex(1);
    atEnd = this->atEnd();
  }

  bool allowNulls() const {
    return false;
  }

  void addRowIndex(vector_size_t count) {
    rowIndex_ += count;
  }

  void addNumValues(vector_size_t /*count*/) {}

  bool atEnd() const {
    return rowIndex_ >= rows_.size();
  }

  const std::vector<T>& values() const {
    return values_;
  }

 private:
  TestReader reader_;
  std::vector<vector_size_t> rows_;
  vector_size_t rowIndex_{0};
  std::vector<T> values_;
};

TEST_F(HuffmanEncodingTest, factoryRoundTripAllIntegralTypes) {
  verifyFactoryRoundTrip<int8_t>();
  verifyFactoryRoundTrip<uint8_t>();
  verifyFactoryRoundTrip<int16_t>();
  verifyFactoryRoundTrip<uint16_t>();
  verifyFactoryRoundTrip<int32_t>();
  verifyFactoryRoundTrip<uint32_t>();
  verifyFactoryRoundTrip<int64_t>();
  verifyFactoryRoundTrip<uint64_t>();
}

TEST_F(HuffmanEncodingTest, signedRoundTrip) {
  Vector<int32_t> values{pool_.get()};
  for (uint32_t i = 0; i < 700; ++i) {
    values.push_back(i % 10 == 0 ? -7 : static_cast<int32_t>(i % 4));
  }

  auto encoding = encode(values);
  Vector<int32_t> result{pool_.get(), values.size()};
  encoding->materialize(static_cast<uint32_t>(values.size()), result.data());

  EXPECT_TRUE(std::equal(result.begin(), result.end(), values.begin()));
}

TEST_F(HuffmanEncodingTest, skewedDistributionLengthLimitedRoundTrip) {
  // Fibonacci frequencies force a maximally skewed Huffman tree whose deepest
  // code is ~n-1 bits, far beyond kMaxCodeBits. Encoding must length-limit the
  // tree instead of throwing, and the code must still round-trip exactly. A
  // successful decode also proves the stored table log is <= kMaxCodeBits,
  // since the decoder rejects anything deeper.
  Vector<int32_t> values{pool_.get()};
  uint64_t previous = 0;
  uint64_t current = 1;
  for (int32_t symbol = 0; symbol < 28; ++symbol) {
    for (uint64_t i = 0; i < current; ++i) {
      values.push_back(symbol);
    }
    const uint64_t next = previous + current;
    previous = current;
    current = next;
  }

  // encode() throwing NIMBLE_INCOMPATIBLE_ENCODING here fails the test, which
  // is the regression being guarded.
  auto encoding = encode(values);

  Vector<int32_t> result{pool_.get(), values.size()};
  encoding->materialize(static_cast<uint32_t>(values.size()), result.data());
  EXPECT_TRUE(std::equal(result.begin(), result.end(), values.begin()));
}

TEST_F(HuffmanEncodingTest, skipAndPartialReadsCrossCheckpoints) {
  Vector<uint64_t> values{pool_.get()};
  for (uint32_t i = 0; i < 900; ++i) {
    values.push_back(i % 17);
  }

  auto encoding = encode(values);
  encoding->skip(250);
  Vector<uint64_t> first{pool_.get(), 40};
  encoding->materialize(static_cast<uint32_t>(first.size()), first.data());
  EXPECT_TRUE(std::equal(first.begin(), first.end(), values.begin() + 250));

  encoding->reset();
  encoding->skip(510);
  Vector<uint64_t> second{pool_.get(), 300};
  encoding->materialize(static_cast<uint32_t>(second.size()), second.data());
  EXPECT_TRUE(std::equal(second.begin(), second.end(), values.begin() + 510));
}

TEST_F(HuffmanEncodingTest, skipRejectsPastEnd) {
  Vector<uint32_t> values{pool_.get()};
  values.push_back(1);
  values.push_back(2);

  auto encoding = encode(values);
  encoding->skip(2);
  NIMBLE_ASSERT_THROW(encoding->skip(1), "(1 vs. 0)");
}

TEST_F(HuffmanEncodingTest, resetAndMaterializeOneAtATime) {
  Vector<int32_t> values{pool_.get()};
  for (uint32_t i = 0; i < 513; ++i) {
    values.push_back(static_cast<int32_t>(i % 7) - 3);
  }

  auto encoding = encode(values);
  for (uint32_t i = 0; i < values.size(); ++i) {
    int32_t result{0};
    encoding->materialize(1, &result);
    EXPECT_EQ(result, values[i]) << "row=" << i;
  }

  encoding->reset();
  Vector<int32_t> result{pool_.get(), values.size()};
  encoding->materialize(static_cast<uint32_t>(values.size()), result.data());
  EXPECT_TRUE(std::equal(result.begin(), result.end(), values.begin()));
}

TEST_F(HuffmanEncodingTest, readWithVisitorDenseAndSparseAcrossCheckpoints) {
  Vector<int64_t> values{pool_.get()};
  for (uint32_t i = 0; i < 1025; ++i) {
    values.push_back(i % 13 == 0 ? -9 : static_cast<int64_t>(i % 5));
  }
  const auto encoded = encodeData(values);

  struct TestCase {
    std::string name;
    std::vector<vector_size_t> rows;
  };
  for (const auto& testCase : std::vector<TestCase>{
           {"dense around checkpoints", {254, 255, 256, 257, 510, 511, 512}},
           {"sparse across checkpoints", {0, 17, 255, 513, 768, 1024}},
       }) {
    SCOPED_TRACE(testCase.name);
    HuffmanEncoding<int64_t> encoding{*pool_, encoded};
    IntegralReadWithVisitor<int64_t> visitor{testCase.rows};
    ReadWithVisitorParams params;
    params.numScanned = 0;

    encoding.readWithVisitor(visitor, params);

    ASSERT_EQ(visitor.values().size(), testCase.rows.size());
    for (size_t i = 0; i < testCase.rows.size(); ++i) {
      EXPECT_EQ(visitor.values()[i], values[testCase.rows[i]]);
    }
  }
}

TEST_F(HuffmanEncodingTest, estimateRejectsFewerThanTwoRows) {
  const std::vector<uint32_t> empty;
  const auto emptyStatistics = Statistics<uint32_t>::create(empty);
  EXPECT_EQ(
      HuffmanEncoding<uint32_t>::estimateSize(empty, emptyStatistics),
      std::nullopt);

  const std::vector<uint32_t> single{7};
  const auto singleStatistics = Statistics<uint32_t>::create(single);
  EXPECT_EQ(
      HuffmanEncoding<uint32_t>::estimateSize(single, singleStatistics),
      std::nullopt);
}

TEST_F(HuffmanEncodingTest, estimateRejectsUnsupportedCardinality) {
  const std::vector<uint32_t> singleValue(100, 7);
  EXPECT_EQ(
      HuffmanEncoding<uint32_t>::estimateSize(
          singleValue, Statistics<uint32_t>::create(singleValue)),
      std::nullopt);

  std::vector<uint32_t> tooManySymbols(
      HuffmanEncoding<uint32_t>::kMaxSymbols + 1);
  std::iota(tooManySymbols.begin(), tooManySymbols.end(), 0);
  EXPECT_EQ(
      HuffmanEncoding<uint32_t>::estimateSize(
          tooManySymbols, Statistics<uint32_t>::create(tooManySymbols)),
      std::nullopt);
}

TEST_F(HuffmanEncodingTest, estimateAcceptsCodeTreeAtLimit) {
  // One Fibonacci weight short of estimateAcceptsCodeTreePastLimit below.
  // Fibonacci weights are what drive a Huffman tree to its deepest, so these
  // two tests sit either side of the 12-bit boundary: this one codes its
  // rarest symbol in exactly 12 bits and is priced exactly, the next needs 13
  // and is priced at its Shannon bound, since encode() length-limits it.
  constexpr std::array<uint32_t, 13> kFrequencies = {
      1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
  Vector<uint32_t> values{pool_.get()};
  for (uint32_t symbol = 0; symbol < kFrequencies.size(); ++symbol) {
    for (uint32_t count = 0; count < kFrequencies[symbol]; ++count) {
      values.push_back(symbol);
    }
  }

  const std::span<const uint32_t> input{values.data(), values.size()};
  EXPECT_NO_THROW(encode(values));
  EXPECT_TRUE(
      HuffmanEncoding<uint32_t>::estimateSize(
          input, Statistics<uint32_t>::create(input))
          .has_value());
}

TEST_F(HuffmanEncodingTest, estimateAcceptsBalancedMaximumAlphabet) {
  // kMaxSymbols equally frequent symbols build a perfectly balanced tree of
  // depth log2(4096) = 12, the deepest tree the table can still hold.
  std::vector<uint32_t> values(HuffmanEncoding<uint32_t>::kMaxSymbols);
  std::iota(values.begin(), values.end(), 0);
  EXPECT_TRUE(
      HuffmanEncoding<uint32_t>::estimateSize(
          values, Statistics<uint32_t>::create(values))
          .has_value());
}

// Inverts the former estimateRejectsCodeTreePastLimit, which asserted that
// encode() throws on a tree deeper than kMaxCodeBits and that estimateSize
// declines the same input so selection never offers it. encode() now
// length-limits instead of throwing, so there is nothing left to decline and
// the depth gate is gone from both sides. Same Fibonacci input, both
// assertions flipped.
TEST_F(HuffmanEncodingTest, estimateAcceptsCodeTreePastLimit) {
  constexpr std::array<uint32_t, 14> kFrequencies = {
      1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377};
  Vector<uint32_t> values{pool_.get()};
  for (uint32_t symbol = 0; symbol < kFrequencies.size(); ++symbol) {
    for (uint32_t count = 0; count < kFrequencies[symbol]; ++count) {
      values.push_back(symbol);
    }
  }

  const std::span<const uint32_t> input{values.data(), values.size()};
  // Priced by default, since encode() length-limits it; declined when the
  // option asks for the selection made before length limiting.
  EXPECT_NE(
      HuffmanEncoding<uint32_t>::estimateSize(
          input, Statistics<uint32_t>::create(input)),
      std::nullopt);
  Encoding::Options declineDeepTrees;
  declineDeepTrees.huffmanPriceLengthLimited = false;
  EXPECT_EQ(
      HuffmanEncoding<uint32_t>::estimateSize(
          input, Statistics<uint32_t>::create(input), declineDeepTrees),
      std::nullopt);

  auto encoding = encode(values);
  Vector<uint32_t> result{pool_.get(), values.size()};
  encoding->materialize(static_cast<uint32_t>(values.size()), result.data());
  EXPECT_TRUE(std::equal(result.begin(), result.end(), values.begin()));
}

TEST_F(HuffmanEncodingTest, estimateSizeChargesExactHuffmanBits) {
  // Three equally frequent symbols code as 1 + 2 + 2 bits, so the bitstream is
  // 3 * 1 + 3 * 2 + 3 * 2 = 15 bits. Shannon lengths charge every symbol
  // ceil(log2(9 / 3)) = 2 bits and reach 18, which is what the estimate used to
  // return: no prefix code is that long, because the two-bit codes leave a
  // one-bit code unused.
  std::vector<uint32_t> values;
  values.insert(values.end(), 3, 0);
  values.insert(values.end(), 3, 1);
  values.insert(values.end(), 3, 2);

  const auto estimate = HuffmanEncoding<uint32_t>::estimateSize(
      values, Statistics<uint32_t>::create(values));
  const uint64_t bitstreamBytes = (15 + 7) / 8 + 4;
  const uint64_t expected =
      EncodingPrefix::serializedSize(9, /*useVarint=*/false) +
      varint::varintSize(3) + 1 + 3 * sizeof(uint32_t) + 3 +
      varint::varintSize(1) + 2 + varint::varintSize(bitstreamBytes) +
      bitstreamBytes;
  EXPECT_EQ(estimate, expected);
}

TEST_F(HuffmanEncodingTest, estimateSizeIncludesCodesAndCheckpoints) {
  std::vector<uint32_t> values;
  values.insert(values.end(), 128, 0);
  values.insert(values.end(), 64, 1);
  values.insert(values.end(), 64, 2);

  const auto estimate = HuffmanEncoding<uint32_t>::estimateSize(
      values, Statistics<uint32_t>::create(values));
  const uint64_t bitstreamBytes = 48 + 4;
  const uint64_t expected =
      EncodingPrefix::serializedSize(256, /*useVarint=*/false) +
      varint::varintSize(3) + 1 + 3 * sizeof(uint32_t) + 3 +
      varint::varintSize(1) + 2 + varint::varintSize(bitstreamBytes) +
      bitstreamBytes;
  EXPECT_EQ(estimate, expected);

  values.push_back(0);
  const auto estimateAcrossCheckpoint = HuffmanEncoding<uint32_t>::estimateSize(
      values, Statistics<uint32_t>::create(values));
  // {129, 64, 64} codes as 1 + 2 + 2 bits, so 129 + 256 = 385 bits. Shannon
  // lengths charged 513 for the same stream; the counts above are dyadic and
  // the two agree there, which is why only this half of the test moved.
  const uint64_t bitstreamBytesAcrossCheckpoint = 49 + 4;
  const uint64_t expectedAcrossCheckpoint =
      EncodingPrefix::serializedSize(257, /*useVarint=*/false) +
      varint::varintSize(3) + 1 + 3 * sizeof(uint32_t) + 3 +
      varint::varintSize(2) + 4 +
      varint::varintSize(bitstreamBytesAcrossCheckpoint) +
      bitstreamBytesAcrossCheckpoint;
  EXPECT_EQ(estimateAcrossCheckpoint, expectedAcrossCheckpoint);
}

struct HuffmanTreeShape {
  // Length of the longest code, which is what kMaxCodeBits limits.
  uint32_t maxCodeLength{0};
  // sum(frequency * code length), the length of the bitstream in bits.
  uint64_t encodedBits{0};
};

// Shape of the Huffman tree over `frequencies`, built the long way: an explicit
// tree through a priority queue, then a traversal that reads each leaf's depth
// off the path taken to reach it. Deliberately a different algorithm from the
// estimator's own merge, so the test below compares two implementations rather
// than one against a copy of itself.
HuffmanTreeShape huffmanTreeShape(const std::vector<uint64_t>& frequencies) {
  struct Entry {
    uint64_t frequency;
    int32_t node;

    bool operator>(const Entry& other) const {
      return frequency != other.frequency ? frequency > other.frequency
                                          : node > other.node;
    }
  };
  struct Node {
    uint64_t frequency;
    int32_t left;
    int32_t right;
  };

  std::vector<Node> nodes;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
  for (uint32_t symbol = 0; symbol < frequencies.size(); ++symbol) {
    nodes.push_back({frequencies[symbol], -1, -1});
    queue.push({frequencies[symbol], static_cast<int32_t>(symbol)});
  }
  while (queue.size() > 1) {
    const auto left = queue.top();
    queue.pop();
    const auto right = queue.top();
    queue.pop();
    const auto node = static_cast<int32_t>(nodes.size());
    nodes.push_back({left.frequency + right.frequency, left.node, right.node});
    queue.push({left.frequency + right.frequency, node});
  }

  HuffmanTreeShape shape;
  std::vector<std::pair<int32_t, uint32_t>> pending{{queue.top().node, 0}};
  while (!pending.empty()) {
    const auto [node, depth] = pending.back();
    pending.pop_back();
    if (nodes[node].left < 0) {
      shape.maxCodeLength = std::max(shape.maxCodeLength, depth);
      shape.encodedBits += nodes[node].frequency * depth;
      continue;
    }
    pending.push_back({nodes[node].left, depth + 1});
    pending.push_back({nodes[node].right, depth + 1});
  }
  return shape;
}

TEST_F(HuffmanEncodingTest, estimateMatchesExplicitHuffmanTree) {
  std::mt19937 rng{20260907};
  size_t trialsAtLimit = 0;
  size_t trialsPastLimit = 0;

  for (int trial = 0; trial < 400; ++trial) {
    // Half the trials grow along a Fibonacci-like chain, from random seeds and
    // with one weight nudged so the chain is not always exact, because that is
    // what drives a tree to the 12-bit boundary the estimator has to decide.
    // The rest stay flat and repetitive to exercise equal-frequency tie-breaks.
    const bool skewed = trial % 2 == 0;
    const uint32_t symbolCount = skewed
        ? std::uniform_int_distribution<uint32_t>{2, 18}(rng)
        : std::uniform_int_distribution<uint32_t>{2, 40}(rng);

    std::vector<uint64_t> frequencies(symbolCount);
    if (skewed) {
      std::uniform_int_distribution<uint64_t> seed{1, 3};
      frequencies[0] = seed(rng);
      frequencies[1] = seed(rng);
      for (uint32_t symbol = 2; symbol < symbolCount; ++symbol) {
        frequencies[symbol] = frequencies[symbol - 1] + frequencies[symbol - 2];
      }
      const uint32_t nudged =
          std::uniform_int_distribution<uint32_t>{0, symbolCount - 1}(rng);
      frequencies[nudged] += std::uniform_int_distribution<uint64_t>{0, 2}(rng);
    } else {
      for (uint32_t symbol = 0; symbol < symbolCount; ++symbol) {
        frequencies[symbol] =
            std::uniform_int_distribution<uint64_t>{1, 4}(rng);
      }
    }

    std::vector<uint32_t> values;
    for (uint32_t symbol = 0; symbol < symbolCount; ++symbol) {
      values.insert(values.end(), frequencies[symbol], symbol);
    }
    // Shuffled so that the estimator numbers its symbols in first-appearance
    // order rather than in frequency order. The answer must depend on the
    // multiset of frequencies alone, never on which value carries which count.
    std::shuffle(values.begin(), values.end(), rng);

    const std::span<const uint32_t> input{values.data(), values.size()};
    const auto estimate = HuffmanEncoding<uint32_t>::estimateSize(
        input, Statistics<uint32_t>::create(input));
    const auto shape = huffmanTreeShape(frequencies);
    const bool fits =
        shape.maxCodeLength <= HuffmanEncoding<uint32_t>::kMaxCodeBits;

    // encode() length-limits a tree that does not fit, so by default every
    // shape is priced; only one that fits is priced exactly.
    EXPECT_TRUE(estimate.has_value())
        << "trial " << trial << " with " << symbolCount
        << " symbols reaching depth " << shape.maxCodeLength;

    if (fits) {
      // The estimator sums the weight of every node it merges instead of
      // measuring leaf depths, so this pins that identity against depths read
      // off a real tree.
      const uint64_t bitstreamBytes = (shape.encodedBits + 7) / 8 + 4;
      const uint64_t checkpoints = velox::bits::divRoundUp(
          values.size(), HuffmanEncoding<uint32_t>::kCheckpointStride);
      const uint64_t expected =
          EncodingPrefix::serializedSize(values.size(), /*useVarint=*/false) +
          varint::varintSize(symbolCount) + 1 + symbolCount * sizeof(uint32_t) +
          symbolCount + varint::varintSize(checkpoints) + checkpoints * 2 +
          varint::varintSize(bitstreamBytes) + bitstreamBytes;
      EXPECT_EQ(estimate, expected) << "trial " << trial;
    }

    if (shape.maxCodeLength == HuffmanEncoding<uint32_t>::kMaxCodeBits) {
      ++trialsAtLimit;
    } else if (
        shape.maxCodeLength == HuffmanEncoding<uint32_t>::kMaxCodeBits + 1) {
      ++trialsPastLimit;
    }
  }

  // An off-by-one in the depth accounting only shows up on the boundary, so
  // fail loudly if the generated trials never reached it.
  EXPECT_GT(trialsAtLimit, 0u);
  EXPECT_GT(trialsPastLimit, 0u);
}

} // namespace
} // namespace facebook::nimble::test
