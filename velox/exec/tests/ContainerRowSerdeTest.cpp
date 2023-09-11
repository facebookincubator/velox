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

  std::vector<std::shared_ptr<ByteStream>> serialize(
      const VectorPtr& data,
      std::vector<HashStringAllocator::Position>& positions) {
    std::vector<std::shared_ptr<ByteStream>> byteStreams;
    auto size = data->size();
    byteStreams.reserve(size);
    positions.reserve(size);

    for (auto i = 0; i < size; ++i) {
      ByteStream out(&allocator_);
      auto position = allocator_.newWrite(out);
      ContainerRowSerde::serialize(*data, i, out);
      allocator_.finishWrite(out, 0);

      auto cur = std::make_shared<ByteStream>();
      HashStringAllocator::prepareRead(position.header, *cur);
      byteStreams.emplace_back(cur);
      positions.emplace_back(position);
    }

    return byteStreams;
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
      {},
      {1, 2, 3, 4},
      {1, 3, 5},
  });

  std::vector<HashStringAllocator::Position> positions;
  auto byteStreams = serialize(data, positions);
  auto arrayVector = makeNullableArrayVector<int64_t>({
      {{1, 2}},
      std::nullopt,
      {{std::nullopt, 1}},
      {{1, 5}},
  });
  SelectivityVector rows(arrayVector->size());
  DecodedVector decodedVector;
  decodedVector.decode(*arrayVector, rows);

  static const CompareFlags kCompareFlags{
      true, // nullsFirst
      true, // ascending
      false, // equalsOnly
      CompareFlags::NullHandlingMode::StopAtNull};

  ASSERT_EQ(
      0,
      ContainerRowSerde::compareWithNulls(
          *byteStreams.at(0), decodedVector, 0, kCompareFlags)
          .value());
  ASSERT_EQ(
      0,
      ContainerRowSerde::compareWithNulls(
          *byteStreams.at(1), decodedVector, 1, kCompareFlags)
          .value());
  ASSERT_FALSE(ContainerRowSerde::compareWithNulls(
                   *byteStreams.at(2), decodedVector, 2, kCompareFlags)
                   .has_value());
  ASSERT_EQ(
      -1,
      ContainerRowSerde::compareWithNulls(
          *byteStreams.at(3), decodedVector, 3, kCompareFlags)
          .value());

  allocator_.clear();
}

TEST_F(ContainerRowSerdeTest, compareNullsInMapVector) {
  auto data = makeNullableMapVector<int64_t, int64_t>({
      {{{1, 10}, {4, 30}}},
      {{{2, 20}}},
      {{{3, 30}}},
      {{{4, std::nullopt}}},
  });
  std::vector<HashStringAllocator::Position> positions;
  auto byteStreams = serialize(data, positions);
  auto mapVector = makeNullableMapVector<int64_t, int64_t>({
      {{{1, 10}, {3, 20}}},
      {{{2, 20}}},
      {{{3, 40}}},
      {{{4, std::nullopt}}},
  });

  SelectivityVector rows(mapVector->size());
  DecodedVector decodedVector;
  decodedVector.decode(*mapVector, rows);

  static const CompareFlags kCompareFlags{
      true, // nullsFirst
      true, // ascending
      false, // equalsOnly
      CompareFlags::NullHandlingMode::StopAtNull};

  ASSERT_EQ(
      1,
      ContainerRowSerde::compareWithNulls(
          *byteStreams.at(0), decodedVector, 0, kCompareFlags)
          .value());
  ASSERT_EQ(
      0,
      ContainerRowSerde::compareWithNulls(
          *byteStreams.at(1), decodedVector, 1, kCompareFlags)
          .value());
  ASSERT_EQ(
      -1,
      ContainerRowSerde::compareWithNulls(
          *byteStreams.at(2), decodedVector, 2, kCompareFlags)
          .value());
  ASSERT_FALSE(ContainerRowSerde::compareWithNulls(
                   *byteStreams.at(3), decodedVector, 3, kCompareFlags)
                   .has_value());
  allocator_.clear();
}

TEST_F(ContainerRowSerdeTest, compareNullsInRowVector) {
  auto data = makeRowVector({makeFlatVector<int32_t>({
      1,
      2,
      3,
      4,
  })});

  std::vector<HashStringAllocator::Position> positions;
  auto byteStreams = serialize(data, positions);
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

  static const CompareFlags kCompareFlags{
      true, // nullsFirst
      true, // ascending
      false, // equalsOnly
      CompareFlags::NullHandlingMode::StopAtNull};

  ASSERT_EQ(
      0,
      ContainerRowSerde::compareWithNulls(
          *byteStreams.at(0), decodedVector, 0, kCompareFlags)
          .value());
  ASSERT_EQ(
      -1,
      ContainerRowSerde::compareWithNulls(
          *byteStreams.at(1), decodedVector, 1, kCompareFlags)
          .value());
  ASSERT_EQ(
      1,
      ContainerRowSerde::compareWithNulls(
          *byteStreams.at(2), decodedVector, 2, kCompareFlags)
          .value());
  ASSERT_FALSE(ContainerRowSerde::compareWithNulls(
                   *byteStreams.at(3), decodedVector, 3, kCompareFlags)
                   .has_value());

  allocator_.clear();
}

} // namespace
} // namespace facebook::velox::exec
