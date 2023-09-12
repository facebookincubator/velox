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
#include "velox/exec/ContainerRowSerde.h"
#include <gtest/gtest.h>
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::exec {

namespace {

class ContainerRowSerdeTest : public testing::Test,
                              public velox::test::VectorTestBase {
 protected:
  HashStringAllocator::Position serialize(const VectorPtr& data) {
    ByteStream out(&allocator_);
    auto position = allocator_.newWrite(out);
    for (auto i = 0; i < data->size(); ++i) {
      ContainerRowSerde::serialize(*data, i, out);
    }
    allocator_.finishWrite(out, 0);
    return position;
  }

  std::vector<HashStringAllocator::Position> serializeWithPositions(
      const VectorPtr& data) {
    std::vector<HashStringAllocator::Position> positions;
    auto size = data->size();
    positions.reserve(size);

    for (auto i = 0; i < size; ++i) {
      ByteStream out(&allocator_);
      auto position = allocator_.newWrite(out);
      ContainerRowSerde::serialize(*data, i, out);
      allocator_.finishWrite(out, 0);

      auto cur = std::make_shared<ByteStream>();
      HashStringAllocator::prepareRead(position.header, *cur);
      positions.emplace_back(position);
    }

    return positions;
  }

  VectorPtr deserialize(
      HashStringAllocator::Position position,
      const TypePtr& type,
      vector_size_t numRows) {
    auto data = BaseVector::create(type, numRows, pool());
    // Set all rows in data to NULL to verify that deserialize can clear nulls
    // correctly.
    for (auto i = 0; i < numRows; ++i) {
      data->setNull(i, true);
    }

    ByteStream in;
    HashStringAllocator::prepareRead(position.header, in);
    for (auto i = 0; i < numRows; ++i) {
      ContainerRowSerde::deserialize(in, i, data.get());
    }
    return data;
  }

  void testRoundTrip(const VectorPtr& data) {
    auto position = serialize(data);
    auto copy = deserialize(position, data->type(), data->size());
    test::assertEqualVectors(data, copy);

    allocator_.clear();
  }

  void testCompareWithNulls(
      const DecodedVector& decodedVector,
      const std::vector<HashStringAllocator::Position>& positions,
      const std::vector<std::optional<int32_t>>& expected,
      bool equalsOnly,
      CompareFlags::NullHandlingMode mode) {
    CompareFlags compareFlags{
        true, // nullsFirst
        true, // ascending
        equalsOnly,
        mode};

    for (auto i = 0; i < expected.size(); ++i) {
      ByteStream buf;
      HashStringAllocator::prepareRead(positions.at(i).header, buf);
      ASSERT_EQ(
          expected.at(i),
          ContainerRowSerde::compareWithNulls(
              buf, decodedVector, i, compareFlags));
    }
  }

  HashStringAllocator allocator_{pool()};
};

TEST_F(ContainerRowSerdeTest, bigint) {
  auto data = makeFlatVector<int64_t>({1, 2, 3, 4, 5});

  testRoundTrip(data);
}

TEST_F(ContainerRowSerdeTest, string) {
  auto data =
      makeFlatVector<std::string>({"a", "Abc", "Long test sentence.", "", "d"});

  testRoundTrip(data);
}

TEST_F(ContainerRowSerdeTest, arrayOfBigint) {
  auto data = makeArrayVector<int64_t>({
      {1, 2, 3},
      {4, 5},
      {6},
      {},
  });

  testRoundTrip(data);

  data = makeNullableArrayVector<int64_t>({
      {{{1, std::nullopt, 2, 3}}},
      {{{std::nullopt, 4, 5}}},
      {{{6, std::nullopt}}},
      {{std::vector<std::optional<int64_t>>({})}},
  });

  testRoundTrip(data);
}

TEST_F(ContainerRowSerdeTest, arrayOfString) {
  auto data = makeArrayVector<std::string>({
      {"a", "b", "Longer string ...."},
      {"c", "Abc", "Mountains and rivers"},
      {},
      {"Oceans and skies"},
  });

  testRoundTrip(data);

  data = makeNullableArrayVector<std::string>({
      {{{std::nullopt,
         std::nullopt,
         "a",
         std::nullopt,
         "b",
         "Longer string ...."}}},
      {{{"c", "Abc", std::nullopt, "Mountains and rivers"}}},
      {{std::vector<std::optional<std::string>>({})}},
      {{{"Oceans and skies"}}},
  });

  testRoundTrip(data);
}

TEST_F(ContainerRowSerdeTest, nested) {
  auto data = makeRowVector(
      {makeNullableFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<std::string>({"a", "", "Long test sentence ......"}),
       makeNullableArrayVector<std::string>({{"a", "b", "c"}, {}, {"d"}})});

  testRoundTrip(data);

  auto nestedArray = makeNullableNestedArrayVector<std::string>(
      {{{{{"1", "2"}}, {{"3", "4"}}}}, {{}}, {{std::nullopt, {}}}});

  testRoundTrip(nestedArray);

  std::vector<std::pair<std::string, std::optional<int64_t>>> map{
      {"a", {1}}, {"b", {2}}, {"c", {3}}, {"d", {4}}};
  nestedArray = makeArrayOfMapVector<std::string, int64_t>(
      {{map, std::nullopt}, {std::nullopt}});

  testRoundTrip(nestedArray);
}

TEST_F(ContainerRowSerdeTest, compareNullsInArrayVector) {
  auto data = makeArrayVector<int64_t>({
      {1, 2},
      {1, 5},
      {1, 3, 5},
      {1, 2, 3, 4},
  });

  std::vector<HashStringAllocator::Position> positions =
      serializeWithPositions(data);
  auto arrayVector = makeNullableArrayVector<int64_t>({
      {1, 2},
      {1, 3},
      {1, 5},
      {std::nullopt, 1},
  });
  SelectivityVector rows(arrayVector->size());
  DecodedVector decodedVector;
  decodedVector.decode(*arrayVector, rows);

  testCompareWithNulls(
      decodedVector,
      positions,
      {{0}, {1}, {-1}, std::nullopt},
      false,
      CompareFlags::NullHandlingMode::StopAtNull);
  testCompareWithNulls(
      decodedVector,
      positions,
      {{0}, {1}, {1}, {1}},
      true,
      CompareFlags::NullHandlingMode::StopAtNull);
  testCompareWithNulls(
      decodedVector,
      positions,
      {{0}, {1}, {-1}, {1}},
      false,
      CompareFlags::NullHandlingMode::NoStop);

  allocator_.clear();
}

TEST_F(ContainerRowSerdeTest, compareNullsInMapVector) {
  auto data = makeNullableMapVector<int64_t, int64_t>({
      {{{1, 10}, {4, 30}, {2, 3}}},
      {{{2, 20}}},
      {{{3, 50}}},
      {{{4, std::nullopt}}},
  });
  std::vector<HashStringAllocator::Position> positions =
      serializeWithPositions(data);
  auto mapVector = makeNullableMapVector<int64_t, int64_t>({
      {{{1, 10}, {3, 20}}},
      {{{2, 20}}},
      {{{3, 40}}},
      {{{4, std::nullopt}}},
  });

  SelectivityVector rows(mapVector->size());
  DecodedVector decodedVector;
  decodedVector.decode(*mapVector, rows);

  testCompareWithNulls(
      decodedVector,
      positions,
      {{-1}, {0}, {1}, std::nullopt},
      false,
      CompareFlags::NullHandlingMode::StopAtNull);
  testCompareWithNulls(
      decodedVector,
      positions,
      {{1}, {0}, {1}, std::nullopt},
      true,
      CompareFlags::NullHandlingMode::StopAtNull);
  testCompareWithNulls(
      decodedVector,
      positions,
      {{1}, {0}, {1}, {0}},
      true,
      CompareFlags::NullHandlingMode::NoStop);

  allocator_.clear();
}

TEST_F(ContainerRowSerdeTest, compareNullsInRowVector) {
  auto data = makeRowVector({makeFlatVector<int32_t>({
      1,
      2,
      3,
      4,
  })});
  std::vector<HashStringAllocator::Position> positions =
      serializeWithPositions(data);

  auto someNulls = makeNullableFlatVector<int32_t>({
      {1},
      {3},
      {2},
      std::nullopt,
  });
  auto rowVector = makeRowVector({someNulls});
  SelectivityVector rows(rowVector->size());
  DecodedVector decodedVector;
  decodedVector.decode(*rowVector, rows);

  testCompareWithNulls(
      decodedVector,
      positions,
      {{0}, {-1}, {1}, std::nullopt},
      false,
      CompareFlags::NullHandlingMode::StopAtNull);
  testCompareWithNulls(
      decodedVector,
      positions,
      {{0}, {-1}, {1}, {1}},
      false,
      CompareFlags::NullHandlingMode::NoStop);

  allocator_.clear();
}

} // namespace
} // namespace facebook::velox::exec
