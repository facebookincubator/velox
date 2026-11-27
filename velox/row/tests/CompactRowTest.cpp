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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/OptionalEmpty.h"
#include "velox/row/CompactRow.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::test;

namespace facebook::velox::row {
namespace {

class CompactRowTest : public ::testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  /// TODO Replace with VectorFuzzer::fuzzInputRow once
  /// https://github.com/facebookincubator/velox/issues/6195 is fixed.
  RowVectorPtr fuzzInputRow(VectorFuzzer& fuzzer, const RowTypePtr& rowType) {
    const auto size = fuzzer.getOptions().vectorSize;
    std::vector<VectorPtr> children;
    for (auto i = 0; i < rowType->size(); ++i) {
      children.push_back(fuzzer.fuzz(rowType->childAt(i), size));
    }

    return std::make_shared<RowVector>(
        pool_.get(), rowType, nullptr, size, std::move(children));
  }

  // Serializes 'data', checks that each row's bytes match the same row with
  // its flat maps as regular maps, and that deserializing gives the regular
  // maps.
  void testFlatMapRoundTrip(const RowVectorPtr& data) {
    std::vector<VectorPtr> mapChildren;
    for (const auto& child : data->children()) {
      mapChildren.push_back(withMapBase(child));
    }
    testFlatMapRoundTrip(data, makeRowVector(mapChildren));
  }

  // Same as above, with 'mapData' being 'data' with its flat maps as regular
  // maps.
  void testFlatMapRoundTrip(
      const RowVectorPtr& data,
      const RowVectorPtr& mapData) {
    CompactRow row(data);
    CompactRow mapRow(mapData);
    std::vector<std::string> buffers;
    for (auto i = 0; i < data->size(); ++i) {
      ASSERT_EQ(row.rowSize(i), mapRow.rowSize(i)) << "Row " << i;
      std::string buffer(row.rowSize(i), '\0');
      ASSERT_EQ(row.serialize(i, buffer.data()), buffer.size());
      std::string mapBuffer(mapRow.rowSize(i), '\0');
      mapRow.serialize(i, mapBuffer.data());
      ASSERT_EQ(buffer, mapBuffer) << "Row " << i;
      buffers.push_back(std::move(buffer));
    }
    std::vector<std::string_view> serialized(buffers.begin(), buffers.end());

    assertEqualVectors(
        mapData,
        CompactRow::deserialize(serialized, asRowType(data->type()), pool()));
  }

  BufferPtr makeInMap(const std::vector<bool>& present) {
    auto inMap = AlignedBuffer::allocate<bool>(present.size(), pool(), false);
    auto* rawInMap = inMap->asMutable<uint64_t>();
    for (auto i = 0; i < present.size(); ++i) {
      bits::setBit(rawInMap, i, present[i]);
    }
    return inMap;
  }

  void testRoundTrip(const RowVectorPtr& data) {
    SCOPED_TRACE(data->toString());

    auto rowType = asRowType(data->type());
    auto numRows = data->size();
    std::vector<size_t> rowSize(numRows);
    std::vector<size_t> offsets(numRows);

    CompactRow row(data);

    size_t totalSize = 0;
    if (auto fixedRowSize = CompactRow::fixedRowSize(rowType)) {
      totalSize = fixedRowSize.value() * numRows;
      for (auto i = 0; i < numRows; ++i) {
        rowSize[i] = fixedRowSize.value();
        offsets[i] = fixedRowSize.value() * i;
      }
    } else {
      for (auto i = 0; i < numRows; ++i) {
        rowSize[i] = row.rowSize(i);
        offsets[i] = totalSize;
        totalSize += rowSize[i];
      }
    }

    std::vector<vector_size_t> rows(numRows);
    std::iota(rows.begin(), rows.end(), 0);
    std::vector<vector_size_t> serializedRowSizes(numRows);
    std::vector<vector_size_t*> serializedRowSizesPtr(numRows);
    for (auto i = 0; i < numRows; ++i) {
      serializedRowSizesPtr[i] = &serializedRowSizes[i];
    }
    row.serializedRowSizes(
        folly::Range(rows.data(), numRows), serializedRowSizesPtr.data());
    for (auto i = 0; i < numRows; ++i) {
      // The serialized row includes the size of the row.
      ASSERT_EQ(serializedRowSizes[i], row.rowSize(i) + sizeof(uint32_t));
    }

    BufferPtr buffer = AlignedBuffer::allocate<char>(totalSize, pool(), 0);
    auto* rawBuffer = buffer->asMutable<char>();
    {
      // Test serialize row-by-row.
      size_t offset = 0;
      std::vector<std::string_view> serialized;
      for (auto i = 0; i < numRows; ++i) {
        auto size = row.serialize(i, rawBuffer + offset);
        serialized.push_back(std::string_view(rawBuffer + offset, size));
        offset += size;

        VELOX_CHECK_EQ(
            size, row.rowSize(i), "Row {}: {}", i, data->toString(i));
      }

      VELOX_CHECK_EQ(offset, totalSize);

      auto copy = CompactRow::deserialize(serialized, rowType, pool());
      assertEqualVectors(data, copy);
    }
    {
      // Test serialize by range.
      memset(rawBuffer, 0, totalSize);

      std::vector<std::string_view> serialized;
      vector_size_t offset = 0;
      vector_size_t rangeSize = 1;
      // Serialize with different range size.
      while (offset < numRows) {
        auto size = std::min<vector_size_t>(rangeSize, numRows - offset);
        row.serialize(offset, size, offsets.data() + offset, rawBuffer);
        offset += size;
        rangeSize = checkedMultiply<vector_size_t>(rangeSize, 2);
      }

      for (auto i = 0; i < numRows; ++i) {
        serialized.push_back(
            std::string_view(rawBuffer + offsets[i], rowSize[i]));
      }
      auto copy = CompactRow::deserialize(serialized, rowType, pool());
      assertEqualVectors(data, copy);
    }
  }
};

TEST_F(CompactRowTest, fixedRowSize) {
  ASSERT_EQ(1 + 1, CompactRow::fixedRowSize(ROW({BOOLEAN()})));
  ASSERT_EQ(1 + 8, CompactRow::fixedRowSize(ROW({BIGINT()})));
  ASSERT_EQ(1 + 4, CompactRow::fixedRowSize(ROW({INTEGER()})));
  ASSERT_EQ(1 + 2, CompactRow::fixedRowSize(ROW({SMALLINT()})));
  ASSERT_EQ(1 + 8, CompactRow::fixedRowSize(ROW({DOUBLE()})));
  ASSERT_EQ(1 + 8, CompactRow::fixedRowSize(ROW({TIMESTAMP_UTC()})));
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

  ASSERT_EQ(1, CompactRow::fixedRowSize(ROW({UNKNOWN()})));
}

TEST_F(CompactRowTest, deserializeWithNulls) {
  const auto data = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40}),
      makeFlatVector<std::string>({"a", "bb", "", "long"}),
      makeArrayVector<int64_t>({{1, 2}, {3}, {}, {4, 5}}),
      makeRowVector({
          makeFlatVector<int32_t>({1, 2, 3, 4}),
          makeFlatVector<std::string>({"x", "y", "z", "w"}),
      }),
  });
  const auto rowType = asRowType(data->type());
  const CompactRow compactRow{data};
  std::vector<std::string> serialized;
  for (vector_size_t row = 0; row < data->size(); ++row) {
    serialized.emplace_back(compactRow.rowSize(row), '\0');
    compactRow.serialize(row, serialized.back().data());
  }
  for (const bool allNull : {false, true}) {
    SCOPED_TRACE(allNull);
    auto nulls = allocateNulls(data->size(), pool());
    auto* rawNulls = nulls->asMutable<uint64_t>();
    std::vector<std::string_view> views{serialized.begin(), serialized.end()};
    for (vector_size_t row = 0; row < data->size(); ++row) {
      const bool isNull = allNull || row % 2 == 1;
      bits::setNull(rawNulls, row, isNull);
      data->setNull(row, isNull);
      if (isNull) {
        views[row] = {};
      }
    }

    const auto decoded = CompactRow::deserialize(views, nulls, rowType, pool());

    assertEqualVectors(data, decoded);
    for (vector_size_t row = 0; row < data->size(); ++row) {
      if (data->isNullAt(row)) {
        for (const auto& child : decoded->children()) {
          EXPECT_TRUE(child->isNullAt(row));
        }
      }
    }
  }
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

    // 1 byte for null flags. 4 bytes for array
    // size. 1 byte for null flags for elements. N bytes for array elements.
    ASSERT_EQ(1 + 4 + 1 + 8 * 3, row.rowSize(0));
    ASSERT_EQ(1 + 4 + 1 + 8 * 2, row.rowSize(1));
    ASSERT_EQ(1 + 4, row.rowSize(2));
    ASSERT_EQ(1 + 4 + 1 + 8, row.rowSize(3));
  }

  data = makeRowVector({
      makeNullableArrayVector<int64_t>({
          {{1, 2, std::nullopt, 3}},
          {{4, 5}},
          common::testutil::optionalEmpty,
          std::nullopt,
          {{6}},
      }),
  });

  {
    CompactRow row(data);

    // 1 byte for null flags. 4 bytes for array
    // size. 1 byte for null flags for elements. N bytes for array elements.
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

    // 1 byte for null flags. 4 bytes for array
    // size. 1 byte for nulls flags for elements. 4 bytes for serialized size. 4
    // bytes per offset of an element. N bytes for elements. Each string element
    // is 4 bytes for size + string length.
    ASSERT_EQ(1 + 4 + 1 + (4 + 1) + (4 + 3), row.rowSize(0));
    ASSERT_EQ(1 + 4, row.rowSize(1));
    ASSERT_EQ(1 + 4 + 1 + (4 + 1) + (4 + 13) + (4 + 3), row.rowSize(2));
  }

  data = makeRowVector({
      makeNullableArrayVector<std::string>({
          {{"a", "Abc", std::nullopt}},
          common::testutil::optionalEmpty,
          std::nullopt,
          {{"a", std::nullopt, "Longer string", "abc"}},
      }),
  });

  {
    CompactRow row(data);

    // Null strings do not take space.
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

TEST_F(CompactRowTest, hugeint) {
  auto data = makeRowVector({
      makeFlatVector<int128_t>({1, 2, 3, 4, 5}),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeNullableFlatVector<int128_t>(
          {std::nullopt, 1, 2, std::nullopt, std::nullopt, 3, 4, 5}),
  });

  testRoundTrip(data);
}

Timestamp ts(int64_t micros) {
  return Timestamp::fromMicros(micros);
}

TEST_F(CompactRowTest, timestamp) {
  auto data = makeRowVector({
      makeFlatVector<Timestamp>({
          ts(0),
          ts(1),
          ts(2),
      }),
  });

  testRoundTrip(data);

  // Serialize null Timestamp values with null flags set over a large
  // non-serializable value (e.g. a value that triggers an exception in
  // Timestamp::toMicros()).
  data = makeRowVector({
      makeFlatVector<Timestamp>({
          ts(0),
          Timestamp::max(),
          ts(123'456),
          Timestamp::min(),
      }),
  });

  data->childAt(0)->setNull(1, true);
  data->childAt(0)->setNull(3, true);

  testRoundTrip(data);
}

TEST_F(CompactRowTest, timestampUtc) {
  auto data = makeRowVector({
      makeFlatVector<Timestamp>(
          {
              ts(0),
              ts(1),
              ts(2),
          },
          TIMESTAMP_UTC()),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeFlatVector<Timestamp>(
          {
              ts(0),
              Timestamp::max(),
              ts(123'456),
              Timestamp::min(),
          },
          TIMESTAMP_UTC()),
  });

  data->childAt(0)->setNull(1, true);
  data->childAt(0)->setNull(3, true);

  testRoundTrip(data);
}

TEST_F(CompactRowTest, string) {
  auto data = makeRowVector({
      makeFlatVector<std::string>({"a", "Abc", "", "Longer test string"}),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, unknown) {
  auto data = makeRowVector({
      makeAllNullFlatVector<UnknownValue>(10),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeArrayVector({0, 3, 5, 9}, makeAllNullFlatVector<UnknownValue>(10)),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, mix) {
  auto data = makeRowVector({
      makeFlatVector<std::string>({"a", "Abc", "", "Longer test string"}),
      makeAllNullFlatVector<UnknownValue>(4),
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, unknownArrayBeforeValue) {
  auto arrays =
      makeArrayVector({0, 1, 3, 3}, makeAllNullFlatVector<UnknownValue>(13));
  arrays->setNull(1, true);
  auto data = makeRowVector({
      arrays,
      makeFlatVector<int64_t>({11, 22, 33, 44}),
  });

  CompactRow row(data);
  // Non-null arrays use just their element count, regardless of length.
  constexpr auto kRowWithArraySize = 1 + sizeof(int64_t) + sizeof(int32_t);
  EXPECT_EQ(kRowWithArraySize, row.rowSize(0));
  EXPECT_EQ(1 + sizeof(int64_t), row.rowSize(1));
  EXPECT_EQ(kRowWithArraySize, row.rowSize(2));
  EXPECT_EQ(kRowWithArraySize, row.rowSize(3));

  testRoundTrip(data);
}

TEST_F(CompactRowTest, unknownArrayBeforeLongString) {
  auto data = makeRowVector({
      makeArrayVector({0}, makeAllNullFlatVector<UnknownValue>(1)),
      makeFlatVector<int64_t>({-1}),
      makeFlatVector<std::string>({std::string(1 << 23, 'x')}),
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
          common::testutil::optionalEmpty,
      }),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, arrayOfTimestamp) {
  auto data = makeRowVector({
      makeArrayVector<Timestamp>({
          {ts(1), ts(2), ts(3)},
          {ts(4), ts(5)},
          {ts(6)},
          {},
      }),
  });

  testRoundTrip(data);

  data = makeRowVector({
      makeNullableArrayVector<Timestamp>({
          {{ts(1), ts(2), std::nullopt, ts(3)}},
          {{ts(4), ts(5), std::nullopt}},
          {{std::nullopt, ts(6)}},
          {{std::nullopt}},
          std::nullopt,
          common::testutil::optionalEmpty,
      }),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, arrayOfString) {
  auto data = makeRowVector({
      makeArrayVector<std::string>(std::vector<std::vector<std::string>>{
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
          common::testutil::optionalEmpty,
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

TEST_F(CompactRowTest, flatMap) {
  {
    SCOPED_TRACE("integer keys");
    auto data = makeRowVector({
        makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
        makeNullableFlatMapVector<int32_t, int64_t>({
            {{{1, 10}, {2, std::nullopt}}},
            std::nullopt,
            common::testutil::optionalEmpty,
            {{{3, 30}}},
            {{{1, 11}, {4, 44}, {5, std::nullopt}}},
        }),
    });
    testFlatMapRoundTrip(data);
  }
  {
    SCOPED_TRACE("string keys");
    auto data = makeRowVector({
        makeFlatMapVector<std::string, std::string>({
            {{"a", "short"}, {"a key longer than twelve bytes", "value"}},
            {},
            {{"b", std::nullopt}, {"a", "a value longer than twelve bytes"}},
        }),
    });
    testFlatMapRoundTrip(data);
  }
  {
    SCOPED_TRACE("complex values");
    auto keys = makeFlatVector<int64_t>({7, 8});
    auto idLists = makeArrayVectorFromJson<int64_t>({
        "[1, 2, 3]",
        "null",
        "[]",
        "[4]",
    });
    auto idScoreLists = makeMapVectorFromJson<int64_t, float>({
        "{1: 0.5, 2: 1.5}",
        "{}",
        "null",
        "{3: 2.5}",
    });
    auto data = makeRowVector({
        std::make_shared<FlatMapVector>(
            pool(),
            MAP(BIGINT(), ARRAY(BIGINT())),
            nullptr,
            4,
            keys,
            std::vector<VectorPtr>{idLists, idLists},
            std::vector<BufferPtr>{
                makeInMap({true, false, true, true}),
                makeInMap({false, true, true, false})}),
        std::make_shared<FlatMapVector>(
            pool(),
            MAP(BIGINT(), MAP(BIGINT(), REAL())),
            nullptr,
            4,
            keys,
            std::vector<VectorPtr>{idScoreLists, idScoreLists},
            std::vector<BufferPtr>{
                nullptr, makeInMap({false, false, true, true})}),
    });
    testFlatMapRoundTrip(data);
  }
  {
    SCOPED_TRACE("unknown values");
    auto inMap = makeInMap({true, false, true});
    auto data = makeRowVector({
        std::make_shared<FlatMapVector>(
            pool(),
            MAP(INTEGER(), UNKNOWN()),
            nullptr,
            3,
            makeFlatVector<int32_t>({1, 2}),
            std::vector<VectorPtr>{
                makeAllNullFlatVector<UnknownValue>(3),
                makeAllNullFlatVector<UnknownValue>(3),
            },
            std::vector<BufferPtr>{nullptr, inMap}),
        makeFlatVector<int64_t>({1, 2, 3}),
    });
    testFlatMapRoundTrip(data);
  }
}

TEST_F(CompactRowTest, flatMapManyKeys) {
  constexpr int32_t kNumKeys{20'000};
  constexpr vector_size_t kNumRows{100};
  std::vector<std::vector<std::pair<int32_t, std::optional<float>>>> maps(
      kNumRows);
  for (vector_size_t row = 0; row < kNumRows; ++row) {
    for (int32_t key = row; key < kNumKeys; key += 50) {
      maps[row].emplace_back(key, key * 0.5f);
    }
  }
  testFlatMapRoundTrip(
      makeRowVector({makeFlatMapVector<int32_t, float>(maps)}));
}

TEST_F(CompactRowTest, flatMapValueEncodings) {
  auto keys = makeFlatVector<int32_t>({1, 2, 3});
  auto inMaps = std::vector<BufferPtr>{
      makeInMap({true, true, false, true}),
      nullptr,
      makeInMap({false, false, false, false}),
  };
  {
    SCOPED_TRACE("fixed-width values");
    auto data = makeRowVector({std::make_shared<FlatMapVector>(
        pool(),
        MAP(INTEGER(), BIGINT()),
        nullptr,
        4,
        keys,
        std::vector<VectorPtr>{
            makeNullableFlatVector<int64_t>({10, std::nullopt, 0, 40}),
            wrapInDictionary(
                makeIndicesInReverse(4),
                makeFlatVector<int64_t>({21, 22, 23, 24})),
            nullptr,
        },
        inMaps)});
    testFlatMapRoundTrip(data);
  }
  {
    SCOPED_TRACE("array values");
    auto data = makeRowVector({std::make_shared<FlatMapVector>(
        pool(),
        MAP(INTEGER(), ARRAY(BIGINT())),
        nullptr,
        4,
        keys,
        std::vector<VectorPtr>{
            makeNullableArrayVector<int64_t>({
                {{1, std::nullopt, 3}},
                std::nullopt,
                common::testutil::optionalEmpty,
                {{std::nullopt}},
            }),
            makeConstantArray<int64_t>(4, {5, 6}),
            nullptr,
        },
        inMaps)});
    testFlatMapRoundTrip(data);
  }
}

TEST_F(CompactRowTest, flatMapEncodings) {
  auto flatMap = makeFlatMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20}",
      "{3: 30}",
      "null",
      "{2: 21, 4: 40}",
  });
  {
    SCOPED_TRACE("dictionary");
    testFlatMapRoundTrip(
        makeRowVector({wrapInDictionary(makeIndicesInReverse(4), flatMap)}));
  }
  {
    SCOPED_TRACE("slice");
    testFlatMapRoundTrip(makeRowVector({flatMap->slice(1, 3)}));
  }
  {
    SCOPED_TRACE("constant");
    testFlatMapRoundTrip(
        makeRowVector({BaseVector::wrapInConstant(3, 3, flatMap)}));
  }
  {
    SCOPED_TRACE("null constant");
    testFlatMapRoundTrip(
        makeRowVector({std::make_shared<ConstantVector<ComplexType>>(
            pool(), 3, 2, flatMap)}));
  }
  {
    SCOPED_TRACE("lazy");
    testFlatMapRoundTrip(
        makeRowVector({std::make_shared<LazyVector>(
            pool(),
            flatMap->type(),
            flatMap->size(),
            std::make_unique<SimpleVectorLoader>(
                [&](RowSet /*rows*/) { return flatMap; }))}),
        makeRowVector({flatMap->toMapVector()}));
  }
}

TEST_F(CompactRowTest, flatMapNonRawValues) {
  {
    SCOPED_TRACE("boolean values");
    testFlatMapRoundTrip(
        makeRowVector({makeNullableFlatMapVector<int32_t, bool>({
            {{{1, true}, {2, std::nullopt}, {3, false}}},
            std::nullopt,
            common::testutil::optionalEmpty,
            {{{2, false}}},
            {{{1, std::nullopt}, {3, true}}},
        })}));
  }
  {
    SCOPED_TRACE("timestamp values");
    testFlatMapRoundTrip(
        makeRowVector({makeNullableFlatMapVector<int32_t, Timestamp>({
            {{{1, Timestamp(1, 2'000)}, {2, std::nullopt}}},
            {{{3, Timestamp(-1'000, 999'999'000)}}},
            std::nullopt,
            common::testutil::optionalEmpty,
            {{{1, std::nullopt}, {2, Timestamp(0, 0)}, {3, Timestamp(5, 0)}}},
        })}));
  }
}

TEST_F(CompactRowTest, flatMapRawValues) {
  {
    SCOPED_TRACE("array values");
    auto values = makeNullableArrayVector<int64_t>({
        {{1, 2}},
        common::testutil::optionalEmpty,
        {{std::nullopt, 3}},
        std::nullopt,
        common::testutil::optionalEmpty,
        {{4, std::nullopt, std::nullopt}},
        std::nullopt,
        {{5}},
        {{std::nullopt}},
        common::testutil::optionalEmpty,
    });
    auto keys = makeFlatVector<int32_t>({1, 2, 3, 1, 3, 1, 2, 3, 2, 3});
    testFlatMapRoundTrip(makeRowVector({FlatMapVector::fromMapVector(
        *makeMapVector({0, 3, 5, 5, 5, 8}, keys, values, {3}))}));
  }
  {
    SCOPED_TRACE("fixed-width values");
    testFlatMapRoundTrip(
        makeRowVector({makeNullableFlatMapVector<int32_t, int64_t>({
            {{{1, 10}, {2, std::nullopt}, {3, 30}}},
            {{{3, std::nullopt}}},
            std::nullopt,
            common::testutil::optionalEmpty,
            {{{1, std::nullopt}, {2, 22}}},
            {{{1, 11}, {2, std::nullopt}, {3, std::nullopt}}},
        })}));
  }
}

TEST_F(CompactRowTest, flatMapNoKeys) {
  {
    SCOPED_TRACE("integer keys");
    testFlatMapRoundTrip(makeRowVector({std::make_shared<FlatMapVector>(
        pool(),
        MAP(INTEGER(), BIGINT()),
        makeNulls({false, true, false}),
        3,
        makeFlatVector<int32_t>(std::vector<int32_t>{}),
        std::vector<VectorPtr>{},
        std::vector<BufferPtr>{})}));
  }
  {
    SCOPED_TRACE("unknown keys");
    testFlatMapRoundTrip(makeRowVector({std::make_shared<FlatMapVector>(
        pool(),
        MAP(UNKNOWN(), UNKNOWN()),
        makeNulls({false, true, false}),
        3,
        BaseVector::create(UNKNOWN(), 0, pool()),
        std::vector<VectorPtr>{},
        std::vector<BufferPtr>{})}));
  }
}

TEST_F(CompactRowTest, flatMapAllNull) {
  testFlatMapRoundTrip(makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      std::make_shared<FlatMapVector>(
          pool(),
          MAP(INTEGER(), BIGINT()),
          makeNulls({true, true, true}),
          3,
          makeFlatVector<int32_t>({1, 2}),
          std::vector<VectorPtr>{
              makeFlatVector<int64_t>({10, 11, 12}),
              makeFlatVector<int64_t>({20, 21, 22}),
          },
          std::vector<BufferPtr>{nullptr, nullptr}),
  }));
}

TEST_F(CompactRowTest, nestedFlatMap) {
  auto flatMap = makeNullableFlatMapVector<int32_t, int64_t>({
      {{{1, 10}}},
      {{{2, 20}, {3, 30}}},
      std::nullopt,
      common::testutil::optionalEmpty,
  });
  auto map = makeNullableMapVector<int32_t, int64_t>({
      {{{1, 10}}},
      {{{2, 20}, {3, 30}}},
      std::nullopt,
      common::testutil::optionalEmpty,
  });
  {
    SCOPED_TRACE("row field");
    testFlatMapRoundTrip(
        makeRowVector({makeRowVector({flatMap})}),
        makeRowVector({makeRowVector({map})}));
  }
  {
    SCOPED_TRACE("array elements");
    testFlatMapRoundTrip(
        makeRowVector({makeArrayVector({0, 1}, flatMap)}),
        makeRowVector({makeArrayVector({0, 1}, map)}));
  }
  {
    SCOPED_TRACE("map values");
    auto keys = makeFlatVector<int32_t>({1, 2, 3, 4});
    testFlatMapRoundTrip(
        makeRowVector({makeMapVector({0, 2}, keys, flatMap)}),
        makeRowVector({makeMapVector({0, 2}, keys, map)}));
  }
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
              makeFlatVector<std::string>(
                  {"a", "Abc", "Long test string", "", "d"}),
          },
          nullEvery(2)),
  });

  testRoundTrip(data);
}

TEST_F(CompactRowTest, fuzz) {
  auto rowType = ROW({
      ROW({BIGINT(), VARCHAR(), DOUBLE()}),
      MAP(VARCHAR(), ROW({ARRAY(BIGINT()), ARRAY(VARCHAR()), REAL()})),
      ARRAY(ROW({BIGINT(), DOUBLE()})),
      ARRAY(MAP(BIGINT(), DOUBLE())),
      BIGINT(),
      ARRAY(MAP(BIGINT(), VARCHAR())),
      ARRAY(MAP(VARCHAR(), REAL())),
      MAP(BIGINT(), ARRAY(BIGINT())),
      BIGINT(),
      ARRAY(BIGINT()),
      DOUBLE(),
      MAP(VARCHAR(), VARCHAR()),
      VARCHAR(),
      ARRAY(ARRAY(BIGINT())),
      BIGINT(),
      ARRAY(ARRAY(VARCHAR())),
  });

  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.containerLength = 5;
  opts.nullRatio = 0.1;
  opts.dictionaryHasNulls = false;
  opts.stringVariableLength = true;
  opts.stringLength = 20;
  opts.containerVariableLength = true;
  opts.complexElementsMaxSize = 1'000;

  // Spark uses microseconds to store timestamp
  opts.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMicroSeconds;

  VectorFuzzer fuzzer(opts, pool_.get());

  const auto iterations = 200;
  for (size_t i = 0; i < iterations; ++i) {
    auto seed = folly::Random::rand32();

    LOG(INFO) << i << ": seed: " << seed;
    SCOPED_TRACE(fmt::format("seed: {}", seed));

    fuzzer.reSeed(seed);
    auto data = fuzzInputRow(fuzzer, rowType);

    testRoundTrip(data);

    if (Test::HasFailure()) {
      break;
    }
  }
}

} // namespace
} // namespace facebook::velox::row
