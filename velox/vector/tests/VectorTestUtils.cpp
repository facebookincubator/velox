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

#include "velox/vector/tests/VectorTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::test {

void checkBaseVectorFlagsSet(const BaseVector& vector) {
  EXPECT_TRUE(vector.getNullCount().has_value());
  EXPECT_TRUE(vector.getDistinctValueCount().has_value());
  EXPECT_TRUE(vector.representedBytes().has_value());
  EXPECT_TRUE(vector.storageBytes().has_value());
}

void checkBaseVectorFlagsCleared(const BaseVector& vector) {
  EXPECT_FALSE(vector.getNullCount().has_value());
  EXPECT_FALSE(vector.getDistinctValueCount().has_value());
  EXPECT_FALSE(vector.representedBytes().has_value());
  EXPECT_FALSE(vector.storageBytes().has_value());
}

void checkVectorFlagsSet(const MapVector& vector) {
  EXPECT_TRUE(vector.hasSortedKeys());
}

void checkVectorFlagsCleared(const MapVector& vector) {
  EXPECT_FALSE(vector.hasSortedKeys());
}

FlatVectorPtr<StringView> makeFlatVectorWithFlags(
    vector_size_t kSize,
    const BufferPtr& nulls,
    memory::MemoryPool* pool) {
  auto values = AlignedBuffer::allocate<StringView>(kSize, pool, "a"_sv);
  auto vector = std::make_shared<FlatVector<StringView>>(
      pool,
      VARCHAR(),
      nulls,
      kSize,
      std::move(values),
      std::vector<BufferPtr>(),
      /*stats*/ SimpleVectorStats<StringView>{"a"_sv, "a"_sv},
      /*distinctValueCount*/ 1,
      /*nullCount*/ 1,
      /*isSorted*/ true,
      /*representedBytes*/ 0,
      /*storageByteCount*/ 0);
  vector->computeAndSetIsAscii(SelectivityVector(kSize - 1));
  checkVectorFlagsSet(*vector);
  return vector;
}

ConstantVectorPtr<StringView> makeConstantVectorWithFlags(
    vector_size_t kSize,
    memory::MemoryPool* pool) {
  auto vector = std::make_shared<ConstantVector<StringView>>(
      pool,
      kSize,
      false,
      VARCHAR(),
      "a"_sv,
      /*stats*/ SimpleVectorStats<StringView>{"a"_sv, "a"_sv},
      /*representedBytes*/ 0,
      /*storageByteCount*/ 0);
  vector->computeAndSetIsAscii(SelectivityVector(kSize));
  checkVectorFlagsSet(*vector);
  return vector;
}

DictionaryVectorPtr<StringView> makeDictionaryVectorWithFlags(
    vector_size_t kSize,
    const BufferPtr& nulls,
    memory::MemoryPool* pool) {
  auto base = BaseVector::createConstant(VARCHAR(), "a", kSize, pool);
  auto vector = std::make_shared<DictionaryVector<StringView>>(
      pool,
      nulls,
      kSize,
      base,
      test::makeIndices(
          kSize, [](auto row) { return row; }, pool),
      /*stats*/ SimpleVectorStats<StringView>{"a"_sv, "a"_sv},
      /*distinctValueCount*/ 1,
      /*nullCount*/ 1,
      /*isSorted*/ true,
      /*representedBytes*/ 0,
      /*storageByteCount*/ 0);
  vector->computeAndSetIsAscii(SelectivityVector(kSize - 1));
  checkVectorFlagsSet(*vector);
  return vector;
}

MapVectorPtr makeMapVectorWithFlags(
    vector_size_t kSize,
    const BufferPtr& nulls,
    memory::MemoryPool* pool) {
  auto keys = BaseVector::createConstant(VARCHAR(), "a", kSize, pool);
  auto values = BaseVector::createConstant(VARCHAR(), "b", kSize, pool);

  auto offsets = allocateOffsets(kSize, pool);
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto sizes = allocateSizes(kSize, pool);
  auto* rawSizes = sizes->asMutable<vector_size_t>();
  for (auto i = 0; i < kSize; ++i) {
    rawOffsets[i] = i;
    rawSizes[i] = 1;
  }

  auto vector = std::make_shared<MapVector>(
      pool,
      MAP(VARCHAR(), VARCHAR()),
      nulls,
      kSize,
      offsets,
      sizes,
      keys,
      values,
      /*nullCount*/ 1,
      /*sortedKeys*/ true);
  checkVectorFlagsSet(*vector);
  return vector;
}

} // namespace facebook::velox::test
