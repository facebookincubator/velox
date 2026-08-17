/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/common/Checksum.h"
#include <gtest/gtest.h>

using namespace facebook::nimble;

TEST(ChecksumTests, CreateXxh3_64) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  ASSERT_NE(checksum, nullptr);
  EXPECT_EQ(checksum->getType(), ChecksumType::XXH3_64);
}

TEST(ChecksumTests, EmptyDataChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto result = checksum->getChecksum();
  EXPECT_NE(result, 0);
}

TEST(ChecksumTests, SingleUpdateChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  checksum->update("hello world");
  auto result = checksum->getChecksum();
  EXPECT_NE(result, 0);
}

TEST(ChecksumTests, MultipleUpdatesChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  checksum->update("hello");
  checksum->update(" ");
  checksum->update("world");
  auto result = checksum->getChecksum();
  EXPECT_NE(result, 0);
}

TEST(ChecksumTests, SameDataProducesSameChecksum) {
  auto checksum1 = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto checksum2 = ChecksumFactory::create(ChecksumType::XXH3_64);

  checksum1->update("test data");
  checksum2->update("test data");

  EXPECT_EQ(checksum1->getChecksum(), checksum2->getChecksum());
}

TEST(ChecksumTests, DifferentDataProducesDifferentChecksum) {
  auto checksum1 = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto checksum2 = ChecksumFactory::create(ChecksumType::XXH3_64);

  checksum1->update("data1");
  checksum2->update("data2");

  EXPECT_NE(checksum1->getChecksum(), checksum2->getChecksum());
}

TEST(ChecksumTests, GetChecksumWithoutReset) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  checksum->update("test");

  auto result1 = checksum->getChecksum(false);
  auto result2 = checksum->getChecksum(false);

  EXPECT_EQ(result1, result2);
}

TEST(ChecksumTests, GetChecksumWithReset) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  checksum->update("test");

  auto resultBeforeReset = checksum->getChecksum(true);
  auto resultAfterReset = checksum->getChecksum();

  EXPECT_NE(resultBeforeReset, resultAfterReset);
}

TEST(ChecksumTests, ResetAllowsReuse) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);

  checksum->update("first data");
  auto firstChecksum = checksum->getChecksum(true);

  checksum->update("first data");
  auto secondChecksum = checksum->getChecksum();

  EXPECT_EQ(firstChecksum, secondChecksum);
}

TEST(ChecksumTests, IncrementalUpdateMatchesSingleUpdate) {
  auto incrementalChecksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto singleChecksum = ChecksumFactory::create(ChecksumType::XXH3_64);

  incrementalChecksum->update("abc");
  incrementalChecksum->update("def");
  incrementalChecksum->update("ghi");

  singleChecksum->update("abcdefghi");

  EXPECT_EQ(incrementalChecksum->getChecksum(), singleChecksum->getChecksum());
}

TEST(ChecksumTests, BinaryDataChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  std::string binaryData = {'\x00', '\x01', '\x02', '\xff', '\xfe'};
  checksum->update(binaryData);
  auto result = checksum->getChecksum();
  EXPECT_NE(result, 0);
}

TEST(ChecksumTests, LargeDataChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  std::string largeData(1024 * 1024, 'x');
  checksum->update(largeData);
  auto result = checksum->getChecksum();
  EXPECT_NE(result, 0);
}
