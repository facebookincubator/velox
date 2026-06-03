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
#include "velox/connectors/hive/paimon/PaimonDeletionFile.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

using namespace facebook::velox::connector::hive::paimon;
using namespace facebook::velox;

TEST(PaimonDeletionFileTest, serializeRoundTrip) {
  PaimonDeletionFile df("s3://bucket/deletion.bin", 64, 256, 10);

  auto serialized = df.serialize();
  auto deserialized = PaimonDeletionFile::create(serialized);

  EXPECT_EQ(deserialized.path, "s3://bucket/deletion.bin");
  EXPECT_EQ(deserialized.offset, 64);
  EXPECT_EQ(deserialized.length, 256);
  EXPECT_EQ(deserialized.cardinality, 10);
}

TEST(PaimonDeletionFileTest, serializeRoundTripZeroOffset) {
  PaimonDeletionFile df("deletion-standalone.bin", 0, 512, 42);

  auto serialized = df.serialize();
  auto deserialized = PaimonDeletionFile::create(serialized);

  EXPECT_EQ(deserialized.path, "deletion-standalone.bin");
  EXPECT_EQ(deserialized.offset, 0);
  EXPECT_EQ(deserialized.length, 512);
  EXPECT_EQ(deserialized.cardinality, 42);
}

TEST(PaimonDeletionFileTest, serializeRoundTripLargeValues) {
  PaimonDeletionFile df(
      "s3://bucket/container.bin", 1ULL << 32, 1ULL << 20, 1000000);

  auto serialized = df.serialize();
  auto deserialized = PaimonDeletionFile::create(serialized);

  EXPECT_EQ(deserialized.path, "s3://bucket/container.bin");
  EXPECT_EQ(deserialized.offset, 1ULL << 32);
  EXPECT_EQ(deserialized.length, 1ULL << 20);
  EXPECT_EQ(deserialized.cardinality, 1000000);
}

TEST(PaimonDeletionFileTest, toString) {
  PaimonDeletionFile df("del.bin", 0, 128, 5);

  auto str = df.toString();
  EXPECT_THAT(str, testing::HasSubstr("del.bin"));
  EXPECT_THAT(str, testing::HasSubstr("offset=0"));
  EXPECT_THAT(str, testing::HasSubstr("length=128"));
  EXPECT_THAT(str, testing::HasSubstr("cardinality=5"));
}

TEST(PaimonDeletionFileTest, toStringContainerFile) {
  PaimonDeletionFile df("s3://bucket/container.bin", 1024, 256, 15);

  auto str = df.toString();
  EXPECT_THAT(str, testing::HasSubstr("s3://bucket/container.bin"));
  EXPECT_THAT(str, testing::HasSubstr("offset=1024"));
  EXPECT_THAT(str, testing::HasSubstr("length=256"));
  EXPECT_THAT(str, testing::HasSubstr("cardinality=15"));
}

TEST(PaimonDeletionFileTest, zeroLengthThrows) {
  VELOX_ASSERT_THROW(
      PaimonDeletionFile("del.bin", 0, 0, 5),
      "PaimonDeletionFile length must be > 0");
}

TEST(PaimonDeletionFileTest, zeroCardinalityThrows) {
  VELOX_ASSERT_THROW(
      PaimonDeletionFile("del.bin", 0, 128, 0),
      "PaimonDeletionFile cardinality must be > 0");
}

TEST(PaimonDeletionFileTest, zeroLengthAndCardinalityThrows) {
  VELOX_ASSERT_THROW(
      PaimonDeletionFile("del.bin", 0, 0, 0),
      "PaimonDeletionFile length must be > 0");
}
