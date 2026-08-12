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

#include "velox/dwio/parquet/common/PageIndex.h"

using namespace facebook::velox::parquet;

TEST(PageIndexRegionTest, validatesSignedBoundsBeforeConversion) {
  auto negative = validatePageIndexRegion(-1, 10, 100);
  EXPECT_FALSE(negative);
  EXPECT_EQ(negative.reason, PageIndexFallbackReason::kInvalidLocation);

  auto zeroLength = validatePageIndexRegion(10, 0, 100);
  EXPECT_FALSE(zeroLength);

  auto beyondFile = validatePageIndexRegion(95, 10, 100);
  EXPECT_FALSE(beyondFile);
}

TEST(PageIndexRegionTest, acceptsCheckedInFileRegion) {
  auto valid = validatePageIndexRegion(10, 20, 100);
  ASSERT_TRUE(valid);
  EXPECT_EQ(valid.value->offset, 10);
  EXPECT_EQ(valid.value->length, 20);
}
