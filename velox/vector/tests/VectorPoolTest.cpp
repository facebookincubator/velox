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

#include "velox/vector/VectorPool.h"
#include "velox/vector/tests/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

class VectorPoolTest : public testing::Test, public VectorTestBase {
 protected:
  // Makes 'size' strings of ''stringSize' characters.  If 'hasNulls' is true,
  // sets 1/5 of the strings to null. If 'overwriteNulls' is true, there are
  // null flags but no null values.
  FlatVectorPtr<StringView> makeStrings(
      vector_size_t size,
      int32_t stringSize,
      bool hasNulls,
      bool overwriteNulls = false) {
    std::string content;
    content.resize(stringSize);
    auto vector = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), size, pool_.get());
    for (auto i = 0; i < size; ++i) {
      bool isNull = hasNulls && (i % 5) == 1;
      if (isNull) {
        vector->setNull(i, true);
      }
      if (!isNull || overwriteNulls) {
        std::fill(content.begin(), content.end(), static_cast<char>(i));
        vector->set(i, StringView(content));
      }
    }
    return vector;
  }

  VectorPool vectorPool_;
};

TEST_F(VectorPoolTest, strings) {
  std::vector<VectorPtr> vectors;
  std::vector<VectorPtr> secondReferences;
  std::vector<BufferPtr> buffers;
  // Element 0: recyclable
  vectors.push_back(makeStrings(10000, 100, false));
  // Element 1 - non-unique
  vectors.push_back(makeStrings(10000, 100, false));
  secondReferences.push_back(vectors.back());
  // Element 2 - recyclable with nulls
  vectors.push_back(makeStrings(10000, 100, true));
  // Element 3 - recyclable with nulls array but no nulls
  vectors.push_back(makeStrings(10000, 100, true, true));

  // Element 4 - Not recyclable because buffers not uique
  vectors.push_back(makeStrings(200000, 10, true));
  buffers.push_back(vectors.back()->as<FlatVector<StringView>>()->values());
  // Element 5: Recyclable but no recycle of string buffers
  vectors.push_back(makeStrings(10, 2000000, false));
  std::vector<BaseVector*> rawPointers;
  for (auto& vector : vectors) {
    rawPointers.push_back(vector.get());
  }
  vectorPool_.release(vectors);
  EXPECT_TRUE(!vectors[0]);
  EXPECT_FALSE(!vectors[1]);
  EXPECT_TRUE(!vectors[2]);
  EXPECT_TRUE(!vectors[3]);
  EXPECT_FALSE(!vectors[4]);
  EXPECT_TRUE(!vectors[5]);

  vectors[5] = vectorPool_.get(VARCHAR(), 100, *pool_);
  EXPECT_EQ(vectors[5].get(), rawPointers[5]);
  // Strings zeroed out.
  EXPECT_EQ(0, vectors[5]->as<FlatVector<StringView>>()->valueAt(1).size());
  // No buffers, the past ones were too large.
  EXPECT_TRUE(
      vectors[5]->as<FlatVector<StringView>>()->stringBuffers().empty());

  vectors[3] = vectorPool_.get(VARCHAR(), 100, *pool_);
  EXPECT_EQ(vectors[3].get(), rawPointers[3]);
  // No nulls array.
  EXPECT_TRUE(!vectors[3]->rawNulls());

  vectors[2] = vectorPool_.get(VARCHAR(), 100, *pool_);
  EXPECT_FALSE(!vectors[2]->rawNulls());
  EXPECT_EQ(
      0, BaseVector::countNulls(vectors[2]->nulls(), 0, vectors[2]->size()));

  vectors[0] = vectorPool_.get(VARCHAR(), 100, *pool_);
  EXPECT_EQ(vectors[0].get(), rawPointers[0]);
  // No nulls buffer.
  EXPECT_TRUE(!vectors[0]->rawNulls());
}
