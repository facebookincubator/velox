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

#include "velox/exec/HashBitRange.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

class HashRangeBitTest : public test::VectorTestBase, public testing::Test {};

TEST_F(HashRangeBitTest, hashBitRange) {
  HashBitRange bitRange(29, 31);
  ASSERT_EQ(29, bitRange.begin());
  ASSERT_EQ(4, bitRange.numPartitions());
  ASSERT_EQ(2, bitRange.numBits());
  ASSERT_EQ(31, bitRange.end());
  ASSERT_EQ(bitRange, bitRange);

  HashBitRange defaultRange;
  ASSERT_EQ(0, defaultRange.begin());
  ASSERT_EQ(1, defaultRange.numPartitions());
  ASSERT_EQ(0, defaultRange.numBits());
  ASSERT_EQ(0, defaultRange.end());
  ASSERT_EQ(defaultRange, defaultRange);
  ASSERT_NE(defaultRange, bitRange);

  // Error test cases.
  HashBitRange validRange(63, 64);
  ASSERT_ANY_THROW(HashBitRange(63, 65));
  ASSERT_ANY_THROW(HashBitRange(65, 65));
}
