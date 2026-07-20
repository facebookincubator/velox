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
#include "velox/connectors/hive/paimon/PaimonDataFileMeta.h"

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

using namespace facebook::velox::connector::hive::paimon;
using namespace facebook::velox;

TEST(PaimonDataFileTest, serializeRoundTrip) {
  PaimonDataFile file;
  file.path = "data.parquet";
  file.size = 4096;
  file.rowCount = 500;
  file.level = 1;
  file.minSequenceNumber = 5;
  file.maxSequenceNumber = 15;
  file.deleteRowCount = 3;
  file.creationTimeMs = 1700000000;
  file.type = PaimonDataFile::Type::kChangelog;
  file.source = PaimonDataFile::Source::kCompact;
  file.deletionFile = PaimonDeletionFile{"del.bin", 0, 64, 2};

  auto serialized = file.serialize();
  auto deserialized = PaimonDataFile::create(serialized);

  EXPECT_EQ(deserialized.path, "data.parquet");
  EXPECT_EQ(deserialized.size, 4096);
  EXPECT_EQ(deserialized.rowCount, 500);
  EXPECT_EQ(deserialized.level, 1);
  EXPECT_EQ(deserialized.minSequenceNumber, 5);
  EXPECT_EQ(deserialized.maxSequenceNumber, 15);
  EXPECT_EQ(deserialized.deleteRowCount, 3);
  EXPECT_EQ(deserialized.creationTimeMs, 1700000000);
  EXPECT_EQ(deserialized.type, PaimonDataFile::Type::kChangelog);
  EXPECT_EQ(deserialized.source, PaimonDataFile::Source::kCompact);
  ASSERT_TRUE(deserialized.deletionFile.has_value());
  EXPECT_EQ(deserialized.deletionFile->path, "del.bin");
  EXPECT_EQ(deserialized.deletionFile->offset, 0);
  EXPECT_EQ(deserialized.deletionFile->length, 64);
  EXPECT_EQ(deserialized.deletionFile->cardinality, 2);
}

TEST(PaimonDataFileTest, serializeNoDeletionFile) {
  PaimonDataFile file;
  file.path = "data.orc";
  file.size = 1024;

  auto serialized = file.serialize();
  auto deserialized = PaimonDataFile::create(serialized);

  EXPECT_EQ(deserialized.path, "data.orc");
  EXPECT_EQ(deserialized.size, 1024);
  EXPECT_FALSE(deserialized.deletionFile.has_value());
}

TEST(PaimonDataFileTest, serializeDefaultValues) {
  PaimonDataFile file;
  file.path = "data.orc";

  auto serialized = file.serialize();
  auto deserialized = PaimonDataFile::create(serialized);

  EXPECT_EQ(deserialized.path, "data.orc");
  EXPECT_EQ(deserialized.size, 0);
  EXPECT_EQ(deserialized.rowCount, 0);
  EXPECT_EQ(deserialized.level, 0);
  EXPECT_EQ(deserialized.minSequenceNumber, 0);
  EXPECT_EQ(deserialized.maxSequenceNumber, 0);
  EXPECT_EQ(deserialized.deleteRowCount, 0);
  EXPECT_EQ(deserialized.creationTimeMs, 0);
  EXPECT_EQ(deserialized.type, PaimonDataFile::Type::kData);
  EXPECT_EQ(deserialized.source, PaimonDataFile::Source::kAppend);
  EXPECT_FALSE(deserialized.deletionFile.has_value());
}

TEST(PaimonDataFileTest, toString) {
  PaimonDataFile file;
  file.path = "data.orc";
  file.size = 1024;
  file.rowCount = 100;
  file.level = 0;

  auto str = file.toString();
  EXPECT_THAT(str, testing::HasSubstr("data.orc"));
  EXPECT_THAT(str, testing::HasSubstr("size=1024"));
  EXPECT_THAT(str, testing::HasSubstr("rows=100"));
  EXPECT_THAT(str, testing::HasSubstr("level=0"));
  EXPECT_THAT(str, testing::HasSubstr("type=DATA"));
  EXPECT_THAT(str, testing::HasSubstr("source=APPEND"));
  EXPECT_THAT(str, testing::HasSubstr("deletionFile=none"));
}

TEST(PaimonDataFileTest, toStringChangelog) {
  PaimonDataFile file;
  file.path = "changelog.orc";
  file.size = 2048;
  file.rowCount = 200;
  file.level = 1;
  file.type = PaimonDataFile::Type::kChangelog;
  file.source = PaimonDataFile::Source::kCompact;
  file.deletionFile = PaimonDeletionFile{"del.bin", 0, 64, 2};

  auto str = file.toString();
  EXPECT_THAT(str, testing::HasSubstr("type=CHANGELOG"));
  EXPECT_THAT(str, testing::HasSubstr("source=COMPACT"));
  EXPECT_THAT(str, testing::HasSubstr("del.bin"));
  EXPECT_THAT(str, testing::HasSubstr("cardinality=2"));
}

TEST(PaimonDataFileTest, defaultValues) {
  PaimonDataFile file;
  EXPECT_TRUE(file.path.empty());
  EXPECT_EQ(file.size, 0);
  EXPECT_EQ(file.rowCount, 0);
  EXPECT_EQ(file.level, 0);
  EXPECT_EQ(file.minSequenceNumber, 0);
  EXPECT_EQ(file.maxSequenceNumber, 0);
  EXPECT_EQ(file.deleteRowCount, 0);
  EXPECT_EQ(file.creationTimeMs, 0);
  EXPECT_EQ(file.type, PaimonDataFile::Type::kData);
  EXPECT_EQ(file.source, PaimonDataFile::Source::kAppend);
  EXPECT_FALSE(file.deletionFile.has_value());
}

TEST(PaimonDataFileTest, typeStringAndParse) {
  EXPECT_EQ(PaimonDataFile::typeString(PaimonDataFile::Type::kData), "DATA");
  EXPECT_EQ(
      PaimonDataFile::typeString(PaimonDataFile::Type::kChangelog),
      "CHANGELOG");

  EXPECT_EQ(
      PaimonDataFile::typeFromString("DATA"), PaimonDataFile::Type::kData);
  EXPECT_EQ(
      PaimonDataFile::typeFromString("CHANGELOG"),
      PaimonDataFile::Type::kChangelog);

  VELOX_ASSERT_THROW(
      PaimonDataFile::typeFromString("UNKNOWN"),
      "Unknown PaimonDataFile::Type: UNKNOWN");
}

TEST(PaimonDataFileTest, sourceStringAndParse) {
  EXPECT_EQ(
      PaimonDataFile::sourceString(PaimonDataFile::Source::kAppend), "APPEND");
  EXPECT_EQ(
      PaimonDataFile::sourceString(PaimonDataFile::Source::kCompact),
      "COMPACT");

  EXPECT_EQ(
      PaimonDataFile::sourceFromString("APPEND"),
      PaimonDataFile::Source::kAppend);
  EXPECT_EQ(
      PaimonDataFile::sourceFromString("COMPACT"),
      PaimonDataFile::Source::kCompact);

  VELOX_ASSERT_THROW(
      PaimonDataFile::sourceFromString("UNKNOWN"),
      "Unknown PaimonDataFile::Source: UNKNOWN");
}

TEST(PaimonDataFileTest, typeStreamAndFormat) {
  {
    std::ostringstream os;
    os << PaimonDataFile::Type::kData;
    EXPECT_EQ(os.str(), "DATA");
  }
  {
    std::ostringstream os;
    os << PaimonDataFile::Type::kChangelog;
    EXPECT_EQ(os.str(), "CHANGELOG");
  }

  EXPECT_EQ(fmt::format("{}", PaimonDataFile::Type::kData), "DATA");
  EXPECT_EQ(fmt::format("{}", PaimonDataFile::Type::kChangelog), "CHANGELOG");
}

TEST(PaimonDataFileTest, sourceStreamAndFormat) {
  {
    std::ostringstream os;
    os << PaimonDataFile::Source::kAppend;
    EXPECT_EQ(os.str(), "APPEND");
  }
  {
    std::ostringstream os;
    os << PaimonDataFile::Source::kCompact;
    EXPECT_EQ(os.str(), "COMPACT");
  }

  EXPECT_EQ(fmt::format("{}", PaimonDataFile::Source::kAppend), "APPEND");
  EXPECT_EQ(fmt::format("{}", PaimonDataFile::Source::kCompact), "COMPACT");
}
