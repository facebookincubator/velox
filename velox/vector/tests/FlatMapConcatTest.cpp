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
#include "velox/vector/FlatMapConcat.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox {
namespace {

// Owns the DecodedVectors for as long as the pointers handed to the concat are
// in use.
struct Decoded {
  std::vector<std::unique_ptr<DecodedVector>> owners;
  std::vector<DecodedVector*> pointers;
};

class FlatMapConcatTest : public testing::Test,
                          public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  Decoded decode(const std::vector<VectorPtr>& inputs) {
    Decoded decoded;
    decoded.owners.reserve(inputs.size());
    decoded.pointers.reserve(inputs.size());
    for (const auto& input : inputs) {
      decoded.owners.push_back(std::make_unique<DecodedVector>(*input));
      decoded.pointers.push_back(decoded.owners.back().get());
    }
    return decoded;
  }

  FlatMapVectorPtr concat(
      const std::vector<VectorPtr>& inputs,
      const SelectivityVector& rows,
      const MapConcatConfig& config = {}) {
    auto decoded = decode(inputs);
    return flatMapConcat(
        pool(), inputs[0]->type(), decoded.pointers, rows, config);
  }
};

TEST_F(FlatMapConcatTest, flatMapEncoded) {
  // One row each, one entry each, disjoint keys: {1->10} + {2->20}.
  auto map1 = makeFlatMapVector<int64_t, int64_t>({
      {{1, 10}},
  });
  auto map2 = makeFlatMapVector<int64_t, int64_t>({
      {{2, 20}},
  });

  SelectivityVector rows(1);
  auto result = concat({map1, map2}, rows);

  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}},
  });
  ASSERT_TRUE(expected->equalValueAt(result.get(), 0, 0))
      << "expected " << expected->toString(0) << ", got "
      << result->toString(0);
}

TEST_F(FlatMapConcatTest, flatMapEncodedPartialRowsAndNulls) {
  auto map1 = makeNullableFlatMapVector<int64_t, int64_t>({
      {{{1, 10}}},
      std::nullopt,
      {{{1, 30}}},
      {{{1, 40}}},
  });
  auto map2 = makeNullableFlatMapVector<int64_t, int64_t>({
      {{{2, 20}}},
      {{{2, 21}}},
      {{{2, 31}}},
      {{{2, 41}}},
  });

  // Select 0, 1, 3 -- row 2 unselected, row 1 null in map1.
  SelectivityVector rows(4, false);
  rows.setValid(0, true);
  rows.setValid(1, true);
  rows.setValid(3, true);
  rows.updateBounds();

  auto result = concat({map1, map2}, rows);
  auto* flatMap = result->as<FlatMapVector>();
  ASSERT_NE(flatMap, nullptr);
  ASSERT_EQ(flatMap->size(), 4);

  EXPECT_FALSE(flatMap->isNullAt(0));
  EXPECT_EQ(flatMap->sizeAt(0), 2);
  EXPECT_TRUE(flatMap->isNullAt(1)); // null propagated from map1
  EXPECT_EQ(flatMap->sizeAt(2), 0); // unselected -> size 0
  EXPECT_EQ(flatMap->sizeAt(3), 2);

  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}},
      {},
      {},
      {{1, 40}, {2, 41}},
  });
  for (int i : {0, 3}) {
    EXPECT_TRUE(expected->equalValueAt(result.get(), i, i))
        << "at " << i << ": expected " << expected->toString(i) << ", got "
        << result->toString(i);
  }
}

TEST_F(FlatMapConcatTest, flatMapEncodedDuplicateKeys) {
  // Key 1 is in both inputs.  Merging shared keys is not implemented yet.
  auto map1 = makeFlatMapVector<int64_t, int64_t>({
      {{1, 10}},
  });
  auto map2 = makeFlatMapVector<int64_t, int64_t>({
      {{1, 100}},
  });

  SelectivityVector rows(1);
  VELOX_ASSERT_THROW(
      concat({map1, map2}, rows), "duplicate keys across inputs");
}

TEST_F(FlatMapConcatTest, allInputsAreFlatMapTrueForFlatMaps) {
  auto first = makeFlatMapVector<int64_t, int64_t>({
      {{1, 10}},
  });
  auto second = makeFlatMapVector<int64_t, int64_t>({
      {{2, 20}},
  });

  auto decoded = decode({first, second});
  EXPECT_TRUE(allInputsAreFlatMap(decoded.pointers));
}

TEST_F(FlatMapConcatTest, allInputsAreFlatMapFalseForMapVectors) {
  auto first = makeMapVector<int64_t, int64_t>({
      {{1, 10}},
  });
  auto second = makeMapVector<int64_t, int64_t>({
      {{2, 20}},
  });

  auto decoded = decode({first, second});
  EXPECT_FALSE(allInputsAreFlatMap(decoded.pointers));
}

TEST_F(FlatMapConcatTest, mixedEncodingsRejected) {
  auto flatMap = makeFlatMapVector<int64_t, int64_t>({
      {{1, 10}},
  });
  auto map = makeMapVector<int64_t, int64_t>({
      {{2, 20}},
  });

  auto decoded = decode({flatMap, map});
  VELOX_ASSERT_THROW(
      allInputsAreFlatMap(decoded.pointers),
      "mix of MapVector and FlatMapVector");
}

} // namespace
} // namespace facebook::velox
