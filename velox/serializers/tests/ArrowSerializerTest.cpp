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
#include "velox/serializers/ArrowSerializer.h"

#include <gtest/gtest.h>

#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::serializer::arrow {
namespace {

class ArrowSerializerTest : public testing::Test,
                            public velox::test::VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    ArrowVectorSerde::registerNamedVectorSerde();
    serde_ = std::make_unique<ArrowVectorSerde>();
  }

  void TearDown() override {
    deregisterNamedVectorSerde("ArrowIpc");
    deregisterVectorSerde();
  }

  // Serialize and deserialize a RowVector via the batch path.
  RowVectorPtr roundTrip(const RowVectorPtr& input) {
    auto iobuf = rowVectorToIOBufBatch(input, *pool_, serde_.get());
    return IOBufToRowVector(
        iobuf, asRowType(input->type()), *pool_, serde_.get());
  }

  std::unique_ptr<ArrowVectorSerde> serde_;
};

TEST_F(ArrowSerializerTest, integers) {
  auto input = makeRowVector({
      makeFlatVector<int8_t>({1, -2, 3}),
      makeFlatVector<int16_t>({100, -200, 300}),
      makeFlatVector<int32_t>({1'000, -2'000, 3'000}),
      makeFlatVector<int64_t>({10'000, -20'000, 30'000}),
  });
  auto result = roundTrip(input);
  test::assertEqualVectors(input, result);
}

TEST_F(ArrowSerializerTest, floats) {
  auto input = makeRowVector({
      makeFlatVector<float>({1.5f, -2.5f, 3.5f}),
      makeFlatVector<double>({1.1, -2.2, 3.3}),
  });
  auto result = roundTrip(input);
  test::assertEqualVectors(input, result);
}

TEST_F(ArrowSerializerTest, booleans) {
  auto input = makeRowVector({
      makeFlatVector<bool>({true, false, true, false, true}),
  });
  auto result = roundTrip(input);
  test::assertEqualVectors(input, result);
}

TEST_F(ArrowSerializerTest, strings) {
  auto input = makeRowVector({
      makeFlatVector<StringView>({"short", "", "a longer string value here"}),
  });
  auto result = roundTrip(input);
  test::assertEqualVectors(input, result);
}

TEST_F(ArrowSerializerTest, nulls) {
  auto input = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 3}),
      makeNullableFlatVector<StringView>(
          {std::nullopt, "hello"_sv, std::nullopt}),
  });
  auto result = roundTrip(input);
  test::assertEqualVectors(input, result);
}

TEST_F(ArrowSerializerTest, allNulls) {
  auto input = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
  });
  auto result = roundTrip(input);
  test::assertEqualVectors(input, result);
}

TEST_F(ArrowSerializerTest, emptyBatch) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({}),
  });
  auto result = roundTrip(input);
  ASSERT_EQ(result->size(), 0);
}

TEST_F(ArrowSerializerTest, arrayType) {
  auto input = makeRowVector({
      makeArrayVector<int32_t>({{1, 2, 3}, {4, 5}, {}}),
  });
  auto result = roundTrip(input);
  test::assertEqualVectors(input, result);
}

TEST_F(ArrowSerializerTest, mapType) {
  auto input = makeRowVector({
      makeMapVector<int32_t, StringView>(
          {{{1, "a"_sv}, {2, "b"_sv}}, {{3, "c"_sv}}}),
  });
  auto result = roundTrip(input);
  test::assertEqualVectors(input, result);
}

TEST_F(ArrowSerializerTest, rowType) {
  auto input = makeRowVector({
      makeRowVector({
          makeFlatVector<int32_t>({1, 2}),
          makeFlatVector<StringView>({"x"_sv, "y"_sv}),
      }),
  });
  auto result = roundTrip(input);
  test::assertEqualVectors(input, result);
}

TEST_F(ArrowSerializerTest, iterativeSerializer) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });

  auto iobuf = rowVectorToIOBuf(input, *pool_, serde_.get());
  auto result =
      IOBufToRowVector(iobuf, asRowType(input->type()), *pool_, serde_.get());
  test::assertEqualVectors(input, result);
}

TEST_F(ArrowSerializerTest, namedSerdeRegistration) {
  ASSERT_TRUE(isRegisteredNamedVectorSerde("ArrowIpc"));
  auto* serde = getNamedVectorSerde("ArrowIpc");
  ASSERT_NE(serde, nullptr);
  ASSERT_EQ(serde->kind(), "ArrowIpc");
}

} // namespace
} // namespace facebook::velox::serializer::arrow
