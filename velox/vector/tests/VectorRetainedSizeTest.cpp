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
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::test {
namespace {

class VectorRetainedSizeTest : public testing::Test,
                               public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  uint64_t getTotalStringBufferSize(const VectorPtr& vector) {
    auto* flatVector = vector->asFlatVector<StringView>();
    VELOX_CHECK_NOT_NULL(flatVector, nullptr);
    return flatVector->stringBufferSize();
  }
};

} // namespace

TEST_F(VectorRetainedSizeTest, flatNoStrings) {
  auto vector = makeFlatVector<int32_t>(1'000, folly::identity);
  uint64_t totalStringBufferSize = 0;
  auto retainedSize = vector->retainedSize(totalStringBufferSize);

  EXPECT_EQ(totalStringBufferSize, 0);
  EXPECT_EQ(retainedSize, vector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, flatInlinedStrings) {
  auto vector = makeFlatVector<std::string>(
      1'000, [](auto row) { return std::string(row % 3, '.'); });
  uint64_t totalStringBufferSize = 0;
  auto retainedSize = vector->retainedSize(totalStringBufferSize);

  EXPECT_EQ(totalStringBufferSize, 0);
  EXPECT_EQ(retainedSize, vector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, flatMultipleStringBuffers) {
  auto vector = makeFlatVector<StringView>({});
  vector->resize(3);

  std::vector<BufferPtr> buffers;
  for (auto i = 1; i <= 3; ++i) {
    vector->getBufferWithSpace(i * 100);
    // Update buffers to hold a shared pointer to every existing string buffer,
    // so that the next getBufferWithSpace call will create a new buffer.
    buffers = vector->stringBuffers();
  }
  EXPECT_EQ(vector->stringBuffers().size(), 3);

  for (auto& buffer : buffers) {
    std::string str(100, 'a');
    memcpy(buffer->asMutable<char>(), str.data(), str.size());
    vector->setNoCopy(0, StringView(buffer->as<char>(), str.size()));
  }

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = vector->retainedSize(totalStringBufferSize);
  uint64_t expectedBufferSize = getTotalStringBufferSize(vector);

  EXPECT_EQ(totalStringBufferSize, expectedBufferSize);
  EXPECT_EQ(retainedSize, vector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, constantWithStringBuffer) {
  auto vector = makeConstant(std::string(100, 'a'), 50);

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = vector->retainedSize(totalStringBufferSize);

  EXPECT_EQ(
      totalStringBufferSize,
      vector->as<ConstantVector<StringView>>()->getStringBuffer()->capacity());
  EXPECT_EQ(retainedSize, vector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, constantWithInlinedString) {
  auto vector = makeConstant("short", 100);

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = vector->retainedSize(totalStringBufferSize);

  EXPECT_EQ(totalStringBufferSize, 0);
  EXPECT_EQ(retainedSize, vector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, constantWithValueVector) {
  auto valueVector = makeFlatVector<std::string>(
      10, [&](auto /*row*/) { return std::string(100, 'a'); });
  auto vector =
      std::make_shared<ConstantVector<StringView>>(pool(), 100, 5, valueVector);

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = vector->retainedSize(totalStringBufferSize);

  auto stringBuffer =
      vector->as<ConstantVector<StringView>>()->getStringBuffer();
  EXPECT_TRUE(stringBuffer != nullptr);
  EXPECT_EQ(totalStringBufferSize, stringBuffer->capacity());
  EXPECT_EQ(retainedSize, vector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, dictionaryWithStrings) {
  auto baseVector = makeFlatVector<std::string>(
      1'000, [&](auto /*row*/) { return std::string(100, 'a'); });
  auto indices = makeIndices(100, folly::identity);
  auto dictVector = wrapInDictionary(indices, 100, baseVector);

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = dictVector->retainedSize(totalStringBufferSize);

  uint64_t expectedBufferSize = getTotalStringBufferSize(baseVector);

  EXPECT_EQ(totalStringBufferSize, expectedBufferSize);
  EXPECT_EQ(retainedSize, dictVector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, dictionaryNoStrings) {
  auto baseVector = makeFlatVector<int32_t>(1'000, folly::identity);
  auto indices = makeIndices(100, folly::identity);
  auto dictVector = wrapInDictionary(indices, 100, baseVector);

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = dictVector->retainedSize(totalStringBufferSize);

  EXPECT_EQ(totalStringBufferSize, 0);
  EXPECT_EQ(retainedSize, dictVector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, rowVectorMultipleStringChildren) {
  auto longStringAt = [&](auto /*row*/) { return std::string(100, 'a'); };

  auto rowVector = makeRowVector({
      makeFlatVector<std::string>(1'000, longStringAt),
      makeFlatVector<int32_t>(1'000, folly::identity),
      makeFlatVector<std::string>(1'000, longStringAt),
  });

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = rowVector->retainedSize(totalStringBufferSize);
  uint64_t expectedBufferSize =
      getTotalStringBufferSize(rowVector->childAt(0)) +
      getTotalStringBufferSize(rowVector->childAt(2));

  EXPECT_EQ(totalStringBufferSize, expectedBufferSize);
  EXPECT_EQ(retainedSize, rowVector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, arrayWithStringElements) {
  auto arrayVector = makeArrayVector<std::string>(
      1'000,
      [](auto /*row*/) { return 3; },
      [&](auto /*row*/) { return std::string(100, 'a'); });

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = arrayVector->retainedSize(totalStringBufferSize);
  uint64_t expectedBufferSize =
      getTotalStringBufferSize(arrayVector->elements());

  EXPECT_EQ(totalStringBufferSize, expectedBufferSize);
  EXPECT_EQ(retainedSize, arrayVector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, arrayNoStrings) {
  auto arrayVector = makeArrayVector<int32_t>(
      1'000,
      [](auto /*row*/) { return 3; },
      [](auto row, auto index) { return row + index; });

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = arrayVector->retainedSize(totalStringBufferSize);

  EXPECT_EQ(totalStringBufferSize, 0);
  EXPECT_EQ(retainedSize, arrayVector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, mapWithStringKeysAndValues) {
  auto longStringAt = [&](auto /*row*/) { return std::string(100, 'a'); };

  auto mapVector = makeMapVector<std::string, std::string>(
      1'000, [](auto /*row*/) { return 2; }, longStringAt, longStringAt);

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = mapVector->retainedSize(totalStringBufferSize);
  uint64_t expectedBufferSize = getTotalStringBufferSize(mapVector->mapKeys()) +
      getTotalStringBufferSize(mapVector->mapValues());

  EXPECT_EQ(totalStringBufferSize, expectedBufferSize);
  EXPECT_EQ(retainedSize, mapVector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, nestedComplexWithStrings) {
  auto innerArray = makeArrayVector<std::string>(
      1'000,
      [](auto /*row*/) { return 2; },
      [&](auto /*row*/) { return std::string(100, 'a'); });

  auto outerArray = makeArrayVector({0, 10, 20}, innerArray);

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = outerArray->retainedSize(totalStringBufferSize);
  uint64_t expectedBufferSize = getTotalStringBufferSize(
      outerArray->elements()->as<ArrayVector>()->elements());

  EXPECT_EQ(totalStringBufferSize, expectedBufferSize);
  EXPECT_EQ(retainedSize, outerArray->retainedSize());
}

TEST_F(VectorRetainedSizeTest, lazyNotLoaded) {
  auto lazyVector = std::make_shared<LazyVector>(
      pool(),
      VARCHAR(),
      100,
      std::make_unique<test::SimpleVectorLoader>([&](auto /*rows*/) {
        return makeFlatVector<std::string>(
            100, [&](auto /*row*/) { return std::string(100, 'a'); });
      }));

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = lazyVector->retainedSize(totalStringBufferSize);

  EXPECT_EQ(totalStringBufferSize, 0);
  EXPECT_EQ(retainedSize, lazyVector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, lazyLoaded) {
  auto lazyVector = std::make_shared<LazyVector>(
      pool(),
      VARCHAR(),
      100,
      std::make_unique<test::SimpleVectorLoader>([&](auto /*rows*/) {
        return makeFlatVector<std::string>(
            100, [&](auto /*row*/) { return std::string(100, 'a'); });
      }));

  SelectivityVector rows(100);
  LazyVector::ensureLoadedRows(lazyVector, rows);

  uint64_t totalStringBufferSize = 0;
  auto retainedSize = lazyVector->retainedSize(totalStringBufferSize);
  uint64_t expectedBufferSize =
      getTotalStringBufferSize(lazyVector->loadedVectorShared());

  EXPECT_EQ(totalStringBufferSize, expectedBufferSize);
  EXPECT_EQ(retainedSize, lazyVector->retainedSize());
}

TEST_F(VectorRetainedSizeTest, sharedStringBuffers) {
  auto baseVector = makeFlatVector<std::string>(
      1'000, [&](auto /*row*/) { return std::string(100, 'a'); });

  auto indices = makeIndices(100, folly::identity);
  std::vector<VectorPtr> dictionaries;
  dictionaries.emplace_back(wrapInDictionary(indices, 100, baseVector));
  dictionaries.emplace_back(wrapInDictionary(indices, 100, baseVector));

  auto expectedBufferSize = getTotalStringBufferSize(baseVector);
  for (auto& dictVector : dictionaries) {
    uint64_t totalStringBufferSize = 0;
    auto retainedSize = dictVector->retainedSize(totalStringBufferSize);

    EXPECT_EQ(totalStringBufferSize, expectedBufferSize);
    EXPECT_EQ(retainedSize, dictVector->retainedSize());
  }
}

} // namespace facebook::velox::test
