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

TEST(ChecksumTests, createXxh3_64) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  ASSERT_NE(checksum, nullptr);
  EXPECT_EQ(checksum->getType(), ChecksumType::XXH3_64);
}

TEST(ChecksumTests, emptyDataChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto result = checksum->getChecksum();
  EXPECT_NE(result, 0);
}

TEST(ChecksumTests, singleUpdateChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  checksum->update("hello world");
  auto result = checksum->getChecksum();
  EXPECT_NE(result, 0);
}

TEST(ChecksumTests, multipleUpdatesChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  checksum->update("hello");
  checksum->update(" ");
  checksum->update("world");
  auto result = checksum->getChecksum();
  EXPECT_NE(result, 0);
}

TEST(ChecksumTests, sameDataProducesSameChecksum) {
  auto checksum1 = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto checksum2 = ChecksumFactory::create(ChecksumType::XXH3_64);

  checksum1->update("test data");
  checksum2->update("test data");

  EXPECT_EQ(checksum1->getChecksum(), checksum2->getChecksum());
}

TEST(ChecksumTests, differentDataProducesDifferentChecksum) {
  auto checksum1 = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto checksum2 = ChecksumFactory::create(ChecksumType::XXH3_64);

  checksum1->update("data1");
  checksum2->update("data2");

  EXPECT_NE(checksum1->getChecksum(), checksum2->getChecksum());
}

TEST(ChecksumTests, getChecksumWithoutReset) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  checksum->update("test");

  auto result1 = checksum->getChecksum(false);
  auto result2 = checksum->getChecksum(false);

  EXPECT_EQ(result1, result2);
}

TEST(ChecksumTests, getChecksumWithReset) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  checksum->update("test");

  auto resultBeforeReset = checksum->getChecksum(true);
  auto resultAfterReset = checksum->getChecksum();

  EXPECT_NE(resultBeforeReset, resultAfterReset);
}

TEST(ChecksumTests, resetAllowsReuse) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);

  checksum->update("first data");
  auto firstChecksum = checksum->getChecksum(true);

  checksum->update("first data");
  auto secondChecksum = checksum->getChecksum();

  EXPECT_EQ(firstChecksum, secondChecksum);
}

TEST(ChecksumTests, incrementalUpdateMatchesSingleUpdate) {
  auto incrementalChecksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto singleChecksum = ChecksumFactory::create(ChecksumType::XXH3_64);

  incrementalChecksum->update("abc");
  incrementalChecksum->update("def");
  incrementalChecksum->update("ghi");

  singleChecksum->update("abcdefghi");

  EXPECT_EQ(incrementalChecksum->getChecksum(), singleChecksum->getChecksum());
}

TEST(ChecksumTests, explicitResetMatchesResetOnRead) {
  auto viaReset = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto viaGetChecksum = ChecksumFactory::create(ChecksumType::XXH3_64);

  viaReset->update("discarded");
  viaReset->reset();
  viaReset->update("kept");

  viaGetChecksum->update("discarded");
  viaGetChecksum->getChecksum(/*reset=*/true);
  viaGetChecksum->update("kept");

  EXPECT_EQ(viaReset->getChecksum(), viaGetChecksum->getChecksum());
}

TEST(ChecksumTests, getChecksum32MatchesNarrowedGetChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  checksum->update("some stream bytes");

  const auto wide = checksum->getChecksum(/*reset=*/false);
  const auto narrow = checksum->getChecksum32(/*reset=*/false);

  EXPECT_EQ(narrow, static_cast<uint32_t>(wide ^ (wide >> 32)));
}

TEST(ChecksumTests, getChecksum32HonorsReset) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);

  checksum->update("payload");
  const auto first = checksum->getChecksum32(/*reset=*/true);
  checksum->update("payload");
  const auto second = checksum->getChecksum32(/*reset=*/true);

  EXPECT_EQ(first, second);
}

// The writer accumulates a stream across its chunk buffers while the reader
// hashes the reassembled stream in one call. The two must agree or every read
// of a healthy file reports corruption.
TEST(ChecksumTests, computeChecksum32MatchesIncrementalUpdates) {
  auto oneShot = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto incremental = ChecksumFactory::create(ChecksumType::XXH3_64);

  incremental->update("chunk0");
  incremental->update("chunk1");
  incremental->update("chunk2");

  EXPECT_EQ(
      oneShot->computeChecksum32("chunk0chunk1chunk2"),
      incremental->getChecksum32(/*reset=*/true));
}

TEST(ChecksumTests, computeChecksumIgnoresAccumulatedState) {
  auto polluted = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto clean = ChecksumFactory::create(ChecksumType::XXH3_64);

  polluted->update("leftover from a previous stream");

  EXPECT_EQ(
      polluted->computeChecksum32("payload"),
      clean->computeChecksum32("payload"));
  EXPECT_EQ(
      polluted->computeChecksum("payload"), clean->computeChecksum("payload"));
}

TEST(ChecksumTests, computeChecksumResetsForSubsequentUpdates) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  auto reference = ChecksumFactory::create(ChecksumType::XXH3_64);

  checksum->computeChecksum32("one shot");
  checksum->update("streamed");

  EXPECT_EQ(checksum->getChecksum(), reference->computeChecksum("streamed"));
}

TEST(ChecksumTests, computeChecksum32MatchesNarrowedComputeChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);

  const auto wide = checksum->computeChecksum("stream bytes");
  const auto narrow = checksum->computeChecksum32("stream bytes");

  EXPECT_EQ(narrow, static_cast<uint32_t>(wide ^ (wide >> 32)));
}

// Per-stream checksums are persisted in stripe-group metadata, so the value
// this produces is on-disk format. Changing the algorithm or the 64->32
// narrowing invalidates the checksums in every file already written; this
// pins both so such a change cannot land unnoticed.
TEST(ChecksumTests, xxh3NarrowingIsStableOnDisk) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);

  const uint64_t wide = checksum->computeChecksum("nimble stream checksum");
  EXPECT_EQ(wide, 0x5ac64c904475cee8ULL);
  EXPECT_EQ(checksum->computeChecksum32("nimble stream checksum"), 0x1eb38278U);
}

TEST(ChecksumTests, binaryDataChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  std::string binaryData = {'\x00', '\x01', '\x02', '\xff', '\xfe'};
  checksum->update(binaryData);
  auto result = checksum->getChecksum();
  EXPECT_NE(result, 0);
}

TEST(ChecksumTests, largeDataChecksum) {
  auto checksum = ChecksumFactory::create(ChecksumType::XXH3_64);
  std::string largeData(1024 * 1024, 'x');
  checksum->update(largeData);
  auto result = checksum->getChecksum();
  EXPECT_NE(result, 0);
}
