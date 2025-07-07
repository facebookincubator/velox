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

#include "velox/functions/lib/Hash.h"

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"

namespace {
TEST(Murmur3Hash32Test, bigint) {
  auto hash = [](uint64_t input, uint32_t seed) {
    return facebook::velox::functions::Murmur3Hash32::hashInt64(input, seed);
  };
  EXPECT_EQ(hash(10, 0), -289985220);
  EXPECT_EQ(hash(0, 0), 1669671676);
  EXPECT_EQ(hash(-5, 0), 1222806974);
  EXPECT_EQ(hash(-42, 0), -846261623);
  EXPECT_EQ(hash(42, 0), 1871679806);
  EXPECT_EQ(hash(INT64_MAX, 0), -2106506049);
  EXPECT_EQ(hash(INT64_MIN, 0), 1366273829);

  EXPECT_EQ(hash(0xcafecafedeadbeef, 42), -256235155);
  EXPECT_EQ(hash(0xdeadbeefcafecafe, 42), 673261790);
  EXPECT_EQ(hash(INT64_MAX, 42), -1604625029);
  EXPECT_EQ(hash(INT64_MIN, 42), -853646085);
  EXPECT_EQ(hash(1, 42), -1712319331);
  EXPECT_EQ(hash(0, 42), -1670924195);
  EXPECT_EQ(hash(-1, 42), -939490007);
}
} // namespace
