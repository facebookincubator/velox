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

#include <gtest/gtest.h>

#include "velox/row/CompactRow.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::test;

namespace facebook::velox::row {
namespace {

class CompactRowTest : public ::testing::Test, public VectorTestBase {
 protected:
  void testRoundTrip(const RowVectorPtr& data) {
    SCOPED_TRACE(data->toString());

    auto rowType = asRowType(data->type());
    auto numRows = data->size();

    CompactRow row(data);

    size_t totalSize = 0;
    if (auto fixedRowSize = CompactRow::fixedRowSize(rowType)) {
      totalSize = fixedRowSize.value() * numRows;
    } else {
      for (auto i = 0; i < numRows; ++i) {
        totalSize += row.rowSize(i);
      }
    }

    std::vector<std::string_view> serialized;

    BufferPtr buffer = AlignedBuffer::allocate<char>(totalSize, pool(), 0);
    auto* rawBuffer = buffer->asMutable<char>();
    size_t offset = 0;
    for (auto i = 0; i < numRows; ++i) {
      auto size = row.serialize(i, rawBuffer + offset);
      serialized.push_back(std::string_view(rawBuffer + offset, size));
      offset += size;

      VELOX_CHECK_EQ(size, row.rowSize(i), "Row {}", i);
    }

    VELOX_CHECK_EQ(offset, totalSize);

    auto copy = CompactRow::deserialize(serialized, rowType, pool());

    //    LOG(ERROR) << copy->toString(0, 10);

    assertEqualVectors(data, copy);
  }
};

TEST_F(CompactRowTest, fixedRowSize) {
  ASSERT_EQ(1 + 1, CompactRow::fixedRowSize(ROW({BOOLEAN()})));
  ASSERT_EQ(1 + 8, CompactRow::fixedRowSize(ROW({BIGINT()})));
  ASSERT_EQ(1 + 4, CompactRow::fixedRowSize(ROW({INTEGER()})));
  ASSERT_EQ(1 + 2, CompactRow::fixedRowSize(ROW({SMALLINT()})));
  ASSERT_EQ(1 + 8, CompactRow::fixedRowSize(ROW({DOUBLE()})));
  ASSERT_EQ(std::nullopt, CompactRow::fixedRowSize(ROW({VARCHAR()})));
  ASSERT_EQ(std::nullopt, CompactRow::fixedRowSize(ROW({ARRAY(BIGINT())})));
  ASSERT_EQ(
      1 + 1 + 8 + 4 + 2 + 8,
      CompactRow::fixedRowSize(
          ROW({BOOLEAN(), BIGINT(), INTEGER(), SMALLINT(), DOUBLE()})));

  ASSERT_EQ(std::nullopt, CompactRow::fixedRowSize(ROW({BIGINT(), VARCHAR()})));
  ASSERT_EQ(
      std::nullopt,
      CompactRow::fixedRowSize(ROW({BIGINT(), ROW({VARCHAR()})})));
}

TEST_F(CompactRowTest, rowSizeString) {
  auto data = makeRowVector({
      makeFlatVector<std::string>({"a", "abc", "Longer string", "d", ""}),
  });

  CompactRow row(data);

  // 1 byte for null flags. 4 bytes for string size. N bytes for the string
  // itself.
  ASSERT_EQ(1 + 4 + 1, row.rowSize(0));
  ASSERT_EQ(1 + 4 + 3, row.rowSize(1));
  ASSERT_EQ(1 + 4 + 13, row.rowSize(2));
  ASSERT_EQ(1 + 4 + 1, row.rowSize(3));
  ASSERT_EQ(1 + 4 + 0, row.rowSize(4));
}

TEST_F(CompactRowTest, rowSizeArrayOfBigint) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
          {4, 5},
          {},
          {6},
      }),
  });

  {
    CompactRow row(data);

    // 1 byte for null flags. 4 bytes for array size. 1 byte for null flags
    // for elements. N bytes for array elements.
    ASSERT_EQ(1 + 4 + 1 + 8 * 3, row.rowSize(0));
    ASSERT_EQ(1 + 4 + 1 + 8 * 2, row.rowSize(1));
    ASSERT_EQ(1 + 4, row.rowSize(2));
    ASSERT_EQ(1 + 4 + 1 + 8, row.rowSize(3));
  }

  data = makeRowVector({
      makeNullableArrayVector<int64_t>({
          {{1, 2, std::nullopt, 3}},
          {{4, 5}},
          {{}},
          std::nullopt,
          {{6}},
      }),
  });

  {
    CompactRow row(data);

    // 1 byte for null flags. 4 bytes for array size. 1 byte for null flags
    // for elements. N bytes for array elements.
    ASSERT_EQ(1 + 4 + 1 + 8 * 4, row.rowSize(0));
    ASSERT_EQ(1 + 4 + 1 + 8 * 2, row.rowSize(1));
    ASSERT_EQ(1 + 4, row.rowSize(2));
    ASSERT_EQ(1, row.rowSize(3));
    ASSERT_EQ(1 + 4 + 1 + 8, row.rowSize(4));
  }
}

TEST_F(CompactRowTest, rowSizeMixed) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, 2, 3, std::nullopt}),
      makeNullableFlatVector<std::string>({"a", "abc", "", std::nullopt}),
  });

  CompactRow row(data);

  // 1 byte for null flags. 8 bytes for bigint field. 4 bytes for string size.
  // N bytes for the string itself.
  ASSERT_EQ(1 + 8 + (4 + 1), row.rowSize(0));
  ASSERT_EQ(1 + 8 + (4 + 3), row.rowSize(1));
  ASSERT_EQ(1 + 8 + (4 + 0), row.rowSize(2));
  ASSERT_EQ(1 + 8, row.rowSize(3));
}

TEST_F(CompactRowTest, rowSizeArrayOfStrings) {
  auto data = makeRowVector({
      makeArrayVector<std::string>({
          {"a", "Abc"},
          {},
          {"a", "Longer string", "abc"},
      }),
  });

  {
    CompactRow row(data);

    // 1 byte for null flags. 4 bytes for array size. 1 byte for nulls flags
    // for elements. N bytes for elements. Each string element is 4 bytes for
    // size + string length.
    ASSERT_EQ(1 + 4 + 1 + (4 + 1) + (4 + 3), row.rowSize(0));
    ASSERT_EQ(1 + 4, row.rowSize(1));
    ASSERT_EQ(1 + 4 + 1 + (4 + 1) + (4 + 13) + (4 + 3), row.rowSize(2));
  }

  data = makeRowVector({
      makeNullableArrayVector<std::string>({
          {{"a", "Abc", std::nullopt}},
          {{}},
          std::nullopt,
          {{"a", std::nullopt, "Longer string", "abc"}},
      }),
  });

  {
    CompactRow row(data);

    // Null strings do use take space.
    ASSERT_EQ(1 + 4 + 1 + (4 + 1) + (4 + 3) + 0, row.rowSize(0));
    ASSERT_EQ(1 + 4, row.rowSize(1));
    ASSERT_EQ(1, row.rowSize(2));
    ASSERT_EQ(1 + 4 + 1 + (4 + 1) + 0 + (4 + 13) + (4 + 3), row.rowSize(3));
  }
}

TEST_F(CompactRowTest, boolean) {
  auto data = makeRowVector({
      makeFlatVector<bool>(
          {true, false, true, true, false, false, true, false}),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeNullableFlatVector<bool>({
          true,
          false,
          std::nullopt,
          true,
          std::nullopt,
          false,
          true,
          false,
      }),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, bigint) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {1, std::nullopt, 3, std::nullopt, 5, std::nullopt}),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, timestamp) {
  auto data = makeRowVector({
      makeFlatVector<Timestamp>({
          Timestamp::fromMicros(0),
          Timestamp::fromMicros(1),
          Timestamp::fromMicros(2),
      }),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeNullableFlatVector<Timestamp>({
          Timestamp::fromMicros(0),
          std::nullopt,
          Timestamp::fromMicros(123'456),
      }),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, string) {
  auto data = makeRowVector({
      makeFlatVector<std::string>({"a", "Abc", "", "Longer test string"}),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, mix) {
  auto data = makeRowVector({
      makeFlatVector<std::string>({"a", "Abc", "", "Longer test string"}),
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, arrayOfBigint) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
          {4, 5},
          {6},
          {},
      }),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeNullableArrayVector<int64_t>({
          {{1, 2, std::nullopt, 3}},
          {{4, 5, std::nullopt}},
          {{std::nullopt, 6}},
          {{std::nullopt}},
          std::nullopt,
          {{}},
      }),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, arrayOfString) {
  auto data = makeRowVector({
      makeArrayVector<std::string>({
          {"a", "abc", "Longer test string"},
          {"b", "Abc 12345 ...test", "foo"},
          {},
      }),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeNullableArrayVector<std::string>({
          {{"a", std::nullopt, "abc", "Longer test string"}},
          {{std::nullopt,
            "b",
            std::nullopt,
            "Abc 12345 ...test",
            std::nullopt,
            "foo"}},
          {{}},
          {{std::nullopt}},
          std::nullopt,
      }),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, map) {
  auto data = makeRowVector({
      makeMapVector<int16_t, int64_t>(
          {{{1, 10}, {2, 20}, {3, 30}}, {{1, 11}, {2, 22}}, {{4, 444}}, {}}),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeMapVector<std::string, std::string>({
          {{"a", "100"},
           {"b", "200"},
           {"Long string for testing", "Another long string"}},
          {{"abc", "300"}, {"d", "400"}},
          {},
      }),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, row) {
  auto data = makeRowVector({
      makeRowVector({
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<double>({1.05, 2.05, 3.05, 4.05, 5.05}),
      }),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeRowVector({
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>(
              {"a", "Abc", "Long test string", "", "d"}),
          makeFlatVector<double>({1.05, 2.05, 3.05, 4.05, 5.05}),
      }),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeRowVector(
          {
              makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
              makeNullableFlatVector<int64_t>({-1, 2, -3, std::nullopt, -5}),
              makeFlatVector<double>({1.05, 2.05, 3.05, 4.05, 5.05}),
          },
          nullEvery(2)),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, fuzz) {
  auto rowType = ROW(
      {BIGINT(),
       ARRAY(BIGINT()),
       DOUBLE(),
       MAP(VARCHAR(), VARCHAR()),
       VARCHAR()});

  VectorFuzzer::Options opts;
  opts.vectorSize = 1'000;
  opts.nullRatio = 0.1;
  opts.containerHasNulls = false;
  opts.dictionaryHasNulls = false;
  opts.stringVariableLength = true;
  opts.stringLength = 20;
  opts.containerVariableLength = true;
  opts.complexElementsMaxSize = 1'000;

  // Spark uses microseconds to store timestamp
  opts.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMicroSeconds,
  opts.containerLength = 10;

  VectorFuzzer fuzzer(opts, pool_.get());

  const auto iterations = 200;
  for (size_t i = 0; i < iterations; ++i) {
    auto seed = folly::Random::rand32();

    LOG(INFO) << i << ": seed: " << seed;
    SCOPED_TRACE(fmt::format("seed: {}", seed));

    fuzzer.reSeed(seed);
    auto data = fuzzer.fuzzInputRow(rowType);

    testRoundTrip(data);
  }
}

} // namespace
} // namespace facebook::velox::row
