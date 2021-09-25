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
#include "velox/exec/RowContainer.h"
#include <gtest/gtest.h>
#include <array>
#include <random>
#include "velox/dwio/dwrf/test/utils/BatchMaker.h"
#include "velox/dwio/type/fbhive/HiveTypeParser.h"
#include "velox/exec/ContainerRowSerde.h"
#include "velox/exec/VectorHasher.h"
#include "velox/vector/tests/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;
using namespace facebook::dwio;

class RowContainerTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_ = memory::getDefaultScopedMemoryPool();
    mappedMemory_ = memory::MappedMemory::getInstance();
  }

  RowVectorPtr makeDataset(
      const std::string& columns,
      const size_t size,
      std::function<void(RowVectorPtr)> customizeData) {
    std::string schema = fmt::format("struct<{}>", columns);
    facebook::dwio::type::fbhive::HiveTypeParser parser;
    auto rowType =
        std::dynamic_pointer_cast<const RowType>(parser.parse(schema));
    auto batch = std::static_pointer_cast<RowVector>(
        BatchMaker::createBatch(rowType, size, *pool_));
    if (customizeData) {
      customizeData(batch);
    }
    return batch;
  }

  std::unique_ptr<RowContainer> makeRowContainer(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<TypePtr>& dependentTypes) {
    static const std::vector<std::unique_ptr<Aggregate>> kEmptyAggregates;
    return std::make_unique<RowContainer>(
        keyTypes,
        false,
        kEmptyAggregates,
        dependentTypes,
        true,
        true,
        true,
        true,
        mappedMemory_,
        ContainerRowSerde::instance());
  }

  void testExtractColumnForOddRows(
      RowContainer& container,
      const std::vector<char*>& rows,
      int column,
      const VectorPtr& expected) {
    auto size = rows.size();

    // Extract only odd rows.
    std::vector<char*> oddRows(size, nullptr);
    for (auto i = 1; i < size; i += 2) {
      oddRows[i] = rows[i];
    }

    auto result = BaseVector::create(expected->type(), size, pool_.get());
    container.extractColumn(oddRows.data(), size, column, result);
    EXPECT_EQ(size, result->size());
    for (size_t row = 0; row < size; ++row) {
      if (row % 2 == 0) {
        EXPECT_TRUE(result->isNullAt(row)) << "at " << row;
      } else {
        EXPECT_TRUE(expected->equalValueAt(result.get(), row, row))
            << "at " << row << ": expected " << expected->toString(row)
            << ", got " << result->toString();
      }
    }
  }

  void testExtractColumnForAllRows(
      RowContainer& container,
      const std::vector<char*>& rows,
      int column,
      const VectorPtr& expected) {
    auto size = rows.size();

    auto result = BaseVector::create(expected->type(), size, pool_.get());
    container.extractColumn(rows.data(), size, column, result);
    EXPECT_EQ(size, result->size());
    for (size_t row = 0; row < size; ++row) {
      EXPECT_TRUE(expected->equalValueAt(result.get(), row, row))
          << "at " << row << ": expected " << expected->toString(row)
          << ", got " << result->toString();
    }
  }

  // Stores the input vector in Row Container, extracts it and compares.
  void roundTrip(const VectorPtr& input) {
    // Create row container.
    std::vector<TypePtr> types{input->type()};
    auto data = makeRowContainer(types, std::vector<TypePtr>{});

    // Store the vector in the rowContainer.
    RowContainer rowContainer(types, mappedMemory_);
    auto size = input->size();
    SelectivityVector allRows(size);
    std::vector<char*> rows(size);
    DecodedVector decoded(*input, allRows);
    for (size_t row = 0; row < size; ++row) {
      rows[row] = rowContainer.newRow();
      rowContainer.store(decoded, row, rows[row], 0);
    }

    testExtractColumnForAllRows(rowContainer, rows, 0, input);

    testExtractColumnForOddRows(rowContainer, rows, 0, input);
  }

  std::unique_ptr<memory::MemoryPool> pool_;
  memory::MappedMemory* mappedMemory_;
};

namespace {
// Ensures that 'column is non-null by copying a non-null value over nulls.
void makeNonNull(RowVectorPtr batch, int32_t column) {
  auto values = batch->childAt(column);
  std::vector<vector_size_t> nonNulls;
  for (auto i = 0; i < values->size(); ++i) {
    if (!values->isNullAt(i)) {
      nonNulls.push_back(i);
    }
  }
  VELOX_CHECK(!nonNulls.empty(), "Whole column must not be null");
  vector_size_t nonNull = 0;
  for (auto i = 0; i < values->size(); ++i) {
    if (values->isNullAt(i)) {
      values->copy(values.get(), i, nonNulls[nonNull++ % nonNulls.size()], 1);
    }
  }
}

// Makes a string with deterministic distinct content.
void makeLargeString(int32_t approxSize, std::string& string) {
  string.reserve(approxSize);
  while (string.size() < string.capacity() - 20) {
    auto size = string.size();
    auto number =
        fmt::format("{}", (size + 1 + approxSize) * 0xc6a4a7935bd1e995L);
    string.resize(size + number.size());
    memcpy(&string[size], &number[0], number.size());
  }
}
} // namespace

namespace {
static int32_t sign(int32_t n) {
  return n < 0 ? -1 : n == 0 ? 0 : 1;
}
} // namespace

TEST_F(RowContainerTest, storeExtractArrayOfVarchar) {
  // Make a string vector with two rows each having 2 elements.
  // Here it is important, that one of the 1st row's elements has more than 12
  // characters (14 and 13 in this case), since 12 is the number of chars being
  // inlined in the String View.
  facebook::velox::test::VectorMaker vectorMaker{pool_.get()};
  std::vector<std::string> strings{
      "aaaaaaaaaaaaaa", "bbbbbbbbbbbbb", "cccccccc", "ddddd"};
  std::vector<std::vector<StringView>> elements(2);
  elements[0].emplace_back(strings[0]);
  elements[0].emplace_back(strings[1]);
  elements[1].emplace_back(strings[2]);
  elements[1].emplace_back(strings[3]);
  auto input = vectorMaker.arrayVector<StringView>(elements);
  roundTrip(input);
}

TEST_F(RowContainerTest, types) {
  constexpr int32_t kNumRows = 100;
  auto batch = makeDataset(
      "bool_val:boolean,"
      "tiny_val:tinyint,"
      "small_val:bigint,"
      "int_val:int,"
      "long_val:bigint,"
      "float_val:float,"
      "double_val:double,"
      "string_val:string,"
      "array_val:array<array<string>>,"
      "struct_val:struct<s_int:int, s_array:array<float>>,"
      "map_val:map<string, map<bigint, struct<s2_int:int, s2_string:string>>>"
      "bool_val:boolean,"
      "tiny_val:tinyint,"
      "small_val:bigint,"
      "int_val:int,"
      "long_val:bigint,"
      "float_val:float,"
      "double_val:double,"
      "string_val:string,"
      "array_val:array<array<string>>,"
      "struct_val:struct<s_int:int, s_array:array<float>>,"
      "map_val:map<string, map<bigint, struct<s2_int:int, s2_string:string>>>",
      kNumRows,
      [](RowVectorPtr rows) {
        auto strings =
            rows->childAt(
                    rows->type()->as<TypeKind::ROW>().getChildIdx("string_val"))
                ->as<FlatVector<StringView>>();
        for (auto i = 0; i < strings->size(); i += 11) {
          std::string chars;
          makeLargeString(i * 10000, chars);
          strings->set(i, StringView(chars));
        }
      });
  const auto& types = batch->type()->as<TypeKind::ROW>().children();
  std::vector<TypePtr> keys;
  // We have the same types twice, once as non-null keys, next as
  // nullable non-keys.
  keys.insert(keys.begin(), types.begin(), types.begin() + types.size() / 2);
  std::vector<TypePtr> dependents;
  dependents.insert(
      dependents.begin(), types.begin() + types.size() / 2, types.end());
  auto data = makeRowContainer(keys, dependents);

  EXPECT_GT(data->nextOffset(), 0);
  EXPECT_GT(data->probedFlagOffset(), 0);
  for (int i = 0; i < kNumRows; ++i) {
    auto row = data->newRow();
    if (i < kNumRows / 2) {
      RowContainer::normalizedKey(row) = i;
    } else {
      data->disableNormalizedKeys();
    }
  }
  EXPECT_EQ(kNumRows, data->numRows());
  std::vector<char*> rows(kNumRows);
  RowContainerIterator iter;
  EXPECT_EQ(kNumRows, data->listRows(&iter, kNumRows, rows.data()));
  EXPECT_EQ(data->listRows(&iter, kNumRows, rows.data()), 0);

  SelectivityVector allRows(kNumRows);
  for (auto column = 0; column < batch->childrenSize(); ++column) {
    if (column < keys.size()) {
      makeNonNull(batch, column);
      EXPECT_EQ(data->columnAt(column).nullMask(), 0);
    } else {
      EXPECT_NE(data->columnAt(column).nullMask(), 0);
    }
    DecodedVector decoded(*batch->childAt(column), allRows);
    for (auto index = 0; index < kNumRows; ++index) {
      data->store(decoded, index, rows[index], column);
    }
  }
  data->checkConsistency();
  auto copy = std::static_pointer_cast<RowVector>(
      BaseVector::create(batch->type(), batch->size(), pool_.get()));
  for (auto column = 0; column < batch->childrenSize(); ++column) {
    testExtractColumnForAllRows(*data, rows, column, batch->childAt(column));
    testExtractColumnForOddRows(*data, rows, column, batch->childAt(column));

    auto extracted = copy->childAt(column);
    extracted->resize(kNumRows);
    data->extractColumn(rows.data(), rows.size(), column, extracted);
    std::vector<uint64_t> hashes(kNumRows);
    auto source = batch->childAt(column);
    auto columnType = batch->type()->as<TypeKind::ROW>().childAt(column);
    VectorHasher hasher(columnType, column);
    hasher.hash(*source, allRows, false, &hashes);
    DecodedVector decoded(*extracted, allRows);
    std::vector<uint64_t> rowHashes(kNumRows);
    data->hash(
        column,
        folly::Range<char**>(rows.data(), rows.size()),
        false,
        rowHashes.data());
    for (auto i = 0; i < kNumRows; ++i) {
      EXPECT_EQ(hashes[i], rowHashes[i]);
      EXPECT_TRUE(source->equalValueAt(extracted.get(), i, i));
      EXPECT_EQ(source->compare(extracted.get(), i, i), 0);
      EXPECT_EQ(source->hashValueAt(i), hashes[i]);
      // Test non-null and nullable variants of equals.
      if (column < keys.size()) {
        EXPECT_TRUE(
            data->equals<false>(rows[i], data->columnAt(column), decoded, i));
      } else {
        EXPECT_TRUE(
            data->equals<true>(rows[i], data->columnAt(column), decoded, i));
      }
      if (i > 0) {
        // We compare the values on consecutive rows
        auto result = sign(source->compare(source.get(), i, i - 1));
        EXPECT_EQ(
            result,
            sign(data->compare(
                rows[i], data->columnAt(column), decoded, i - 1)));
        EXPECT_EQ(result, sign(data->compare(rows[i], rows[i - 1], column)));
      }
    }
  }
}

TEST_F(RowContainerTest, erase) {
  constexpr int32_t kNumRows = 100;
  auto data = makeRowContainer({SMALLINT()}, {SMALLINT()});

  // The layout is expected to be smallint - 6 bytes of padding - 1 byte of bits
  // - smallint - next pointer. The bits are a null flag for the second
  // smallint, a probed flag and a free flag.
  EXPECT_EQ(data->nextOffset(), 11);
  // 2nd bit in first byte of flags.
  EXPECT_EQ(data->probedFlagOffset(), 8 * 8 + 1);
  std::unordered_set<char*> rowSet;
  std::vector<char*> rows;
  for (int i = 0; i < kNumRows; ++i) {
    rows.push_back(data->newRow());
    rowSet.insert(rows.back());
  }
  EXPECT_EQ(kNumRows, data->numRows());
  std::vector<char*> rowsFromContainer(kNumRows);
  RowContainerIterator iter;
  EXPECT_EQ(
      data->listRows(&iter, kNumRows, rowsFromContainer.data()), kNumRows);
  EXPECT_EQ(0, data->listRows(&iter, kNumRows, rows.data()));
  EXPECT_EQ(rows, rowsFromContainer);

  std::vector<char*> erased;
  for (auto i = 0; i < rows.size(); i += 2) {
    erased.push_back(rows[i]);
  }
  data->eraseRows(folly::Range<char**>(erased.data(), erased.size()));
  data->checkConsistency();

  std::vector<char*> remaining(data->numRows());
  iter.reset();
  EXPECT_EQ(
      remaining.size(),
      data->listRows(&iter, data->numRows(), remaining.data()));
  // We check that the next new rows reuse the erased rows.
  for (auto i = 0; i < erased.size(); ++i) {
    auto reused = data->newRow();
    EXPECT_NE(rowSet.end(), rowSet.find(reused));
  }
  // The next row will be a new one.
  auto newRow = data->newRow();
  EXPECT_EQ(rowSet.end(), rowSet.find(newRow));
  data->checkConsistency();
}
