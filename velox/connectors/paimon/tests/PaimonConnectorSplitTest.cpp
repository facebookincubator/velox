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
#include "velox/connectors/paimon/PaimonConnectorSplit.h"

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>

#include "velox/common/base/tests/GTestUtils.h"

using namespace facebook::velox::connector::paimon;
using namespace facebook::velox;
using namespace facebook::velox::dwio::common;

namespace {
using PartitionKeys =
    std::unordered_map<std::string, std::optional<std::string>>;
} // namespace

class PaimonConnectorSplitTest : public testing::Test {
 protected:
  const std::string kConnectorId{"test-connector"};
};

TEST_F(PaimonConnectorSplitTest, basic) {
  const PaimonDataFile file0(
      /*schemaId=*/0,
      "s3://bucket/table/dt=2024-01-01/bucket-0/data-001.orc",
      /*size=*/1024,
      /*rowCount=*/100,
      /*level=*/0);

  const PaimonDataFile file1(
      /*schemaId=*/1,
      "s3://bucket/table/dt=2024-01-01/bucket-0/data-002.orc",
      /*size=*/2048,
      /*rowCount=*/200,
      /*level=*/1);

  std::vector<PaimonDataFile> files{file0, file1};
  PartitionKeys partitionKeys{{"dt", "2024-01-01"}};

  auto split = std::make_shared<PaimonConnectorSplit>(
      kConnectorId,
      /*snapshotId=*/1,
      PaimonTableType::kPrimaryKey,
      FileFormat::DWRF,
      files,
      partitionKeys,
      0);

  EXPECT_EQ(split->snapshotId(), 1);
  EXPECT_EQ(split->tableType(), PaimonTableType::kPrimaryKey);
  EXPECT_EQ(split->fileFormat(), FileFormat::DWRF);
  EXPECT_TRUE(split->rawConvertible());
  EXPECT_EQ(split->tableBucketNumber(), 0);
  ASSERT_EQ(split->dataFiles().size(), 2);
  EXPECT_EQ(split->dataFiles()[0].path, files[0].path);
  EXPECT_EQ(split->dataFiles()[0].size, 1024);
  EXPECT_EQ(split->dataFiles()[1].path, files[1].path);
  EXPECT_EQ(split->dataFiles()[1].size, 2048);
  ASSERT_EQ(split->partitionKeys().count("dt"), 1);
  EXPECT_EQ(split->partitionKeys().at("dt"), "2024-01-01");
}

TEST_F(PaimonConnectorSplitTest, builderPreservesFileSchemaIdentity) {
  const PaimonDataFile file(
      /*schemaId=*/0,
      "data.nimble",
      /*size=*/1024,
      /*rowCount=*/9,
      /*level=*/0,
      /*minSequenceNumber=*/0,
      /*maxSequenceNumber=*/0,
      /*deleteRowCount=*/0,
      /*creationTimeMs=*/0,
      PaimonDataFile::Type::kData,
      PaimonDataFile::Source::kCompact);
  auto builder = [&] {
    return PaimonConnectorSplitBuilder(
        kConnectorId, 7, PaimonTableType::kAppendOnly, FileFormat::NIMBLE);
  };
  const auto split =
      builder().addFile(file).addFile(17, "current.nimble", 2048).build();
  const auto copy = PaimonConnectorSplit::create(split->serialize());
  ASSERT_EQ(copy->dataFiles().size(), 2);
  EXPECT_EQ(copy->dataFiles()[0].schemaId, 0);
  EXPECT_EQ(copy->dataFiles()[0].size, 1024);
  EXPECT_EQ(copy->dataFiles()[0].rowCount, 9);
  EXPECT_EQ(copy->dataFiles()[0].source, PaimonDataFile::Source::kCompact);
  EXPECT_EQ(copy->dataFiles()[1].schemaId, 17);
  EXPECT_EQ(copy->dataFiles()[1].size, 2048);
  for (int i = 0; i < 2; ++i) {
    auto wire = split->serialize();
    wire["dataFiles"][i].erase("schemaId");
    VELOX_ASSERT_THROW(
        PaimonConnectorSplit::create(wire), "Expected Paimon file schema ID");
  }
}

TEST_F(PaimonConnectorSplitTest, nimbleFormat) {
  const PaimonDataFile file0(
      /*schemaId=*/0,
      "s3://bucket/table/data-001.nimble",
      /*size=*/4096,
      /*rowCount=*/100);
  std::vector<PaimonDataFile> files{file0};

  auto split = std::make_shared<PaimonConnectorSplit>(
      kConnectorId,
      /*snapshotId=*/1,
      PaimonTableType::kAppendOnly,
      FileFormat::NIMBLE,
      files,
      PartitionKeys{},
      std::nullopt);

  ASSERT_EQ(split->dataFiles().size(), 1);
  EXPECT_EQ(split->tableType(), PaimonTableType::kAppendOnly);
  EXPECT_EQ(split->fileFormat(), FileFormat::NIMBLE);
}

TEST_F(PaimonConnectorSplitTest, parquetFormat) {
  const PaimonDataFile file0(
      /*schemaId=*/0,
      "s3://bucket/table/data-001.parquet",
      /*size=*/4096,
      /*rowCount=*/100);
  std::vector<PaimonDataFile> files{file0};

  auto split = std::make_shared<PaimonConnectorSplit>(
      kConnectorId,
      /*snapshotId=*/1,
      PaimonTableType::kAppendOnly,
      FileFormat::PARQUET,
      files,
      PartitionKeys{},
      std::nullopt);

  ASSERT_EQ(split->dataFiles().size(), 1);
  EXPECT_EQ(split->fileFormat(), FileFormat::PARQUET);
}

TEST_F(PaimonConnectorSplitTest, emptyFilesThrows) {
  std::vector<PaimonDataFile> noFiles;
  VELOX_ASSERT_THROW(
      std::make_shared<PaimonConnectorSplit>(
          kConnectorId,
          /*snapshotId=*/1,
          PaimonTableType::kAppendOnly,
          FileFormat::NIMBLE,
          noFiles,
          PartitionKeys{},
          std::nullopt),
      "PaimonConnectorSplit requires non-empty dataFiles");
}

TEST_F(PaimonConnectorSplitTest, rawConvertibleWithDeleteRowCountThrows) {
  const PaimonDataFile file(
      /*schemaId=*/0,
      "data-001.orc",
      /*size=*/1024,
      /*rowCount=*/100,
      /*level=*/1,
      /*minSequenceNumber=*/0,
      /*maxSequenceNumber=*/0,
      /*deleteRowCount=*/5);

  std::vector<PaimonDataFile> files{file};
  VELOX_ASSERT_THROW(
      std::make_shared<PaimonConnectorSplit>(
          kConnectorId,
          /*snapshotId=*/1,
          PaimonTableType::kPrimaryKey,
          FileFormat::DWRF,
          files,
          PartitionKeys{},
          std::nullopt,
          /*rawConvertible=*/true),
      "rawConvertible split cannot have files with deleteRowCount > 0");
}

TEST_F(PaimonConnectorSplitTest, notRawConvertibleWithDeleteRowCount) {
  const PaimonDataFile file(
      /*schemaId=*/0,
      "data-001.orc",
      /*size=*/1024,
      /*rowCount=*/100,
      /*level=*/1,
      /*minSequenceNumber=*/0,
      /*maxSequenceNumber=*/0,
      /*deleteRowCount=*/5);

  std::vector<PaimonDataFile> files{file};
  auto split = std::make_shared<PaimonConnectorSplit>(
      kConnectorId,
      /*snapshotId=*/1,
      PaimonTableType::kPrimaryKey,
      FileFormat::DWRF,
      files,
      PartitionKeys{},
      std::nullopt,
      /*rawConvertible=*/false);

  EXPECT_FALSE(split->rawConvertible());
  EXPECT_EQ(split->dataFiles()[0].deleteRowCount, 5);
}

TEST_F(PaimonConnectorSplitTest, toStringOutput) {
  const PaimonDataFile f0(
      /*schemaId=*/0, "data-001.orc", /*size=*/1024, /*rowCount=*/100);

  const PaimonDataFile f1(
      /*schemaId=*/0,
      "data-002.orc",
      /*size=*/2048,
      /*rowCount=*/200,
      /*level=*/1);

  const PaimonDataFile f2(
      /*schemaId=*/0,
      "data-003.orc",
      /*size=*/512,
      /*rowCount=*/50,
      /*level=*/2);

  std::vector<PaimonDataFile> files{f0, f1, f2};

  auto split = std::make_shared<PaimonConnectorSplit>(
      kConnectorId,
      /*snapshotId=*/42,
      PaimonTableType::kPrimaryKey,
      FileFormat::DWRF,
      files,
      PartitionKeys{},
      std::nullopt);

  EXPECT_THAT(split->toString(), testing::HasSubstr("snapshot 42"));
  EXPECT_THAT(split->toString(), testing::HasSubstr("PRIMARY_KEY"));
  EXPECT_THAT(split->toString(), testing::HasSubstr(kConnectorId));
  EXPECT_THAT(split->toString(), testing::HasSubstr("data-001.orc"));
  EXPECT_THAT(split->toString(), testing::HasSubstr("data-003.orc"));
  EXPECT_THAT(split->toString(), testing::HasSubstr("size=1024"));
  EXPECT_THAT(split->toString(), testing::HasSubstr("rows=100"));
  EXPECT_THAT(split->toString(), testing::HasSubstr("level=1"));
  EXPECT_THAT(split->toString(), testing::HasSubstr("deletionFile=none"));
}

TEST_F(PaimonConnectorSplitTest, nullPartitionValue) {
  const PaimonDataFile file0(
      /*schemaId=*/0, "data-001.orc", /*size=*/1024, /*rowCount=*/100);
  std::vector<PaimonDataFile> files{file0};
  PartitionKeys partitionKeys{{"country", std::nullopt}};

  auto split = std::make_shared<PaimonConnectorSplit>(
      kConnectorId,
      /*snapshotId=*/1,
      PaimonTableType::kAppendOnly,
      FileFormat::DWRF,
      files,
      partitionKeys,
      std::nullopt);

  ASSERT_EQ(split->partitionKeys().count("country"), 1);
  EXPECT_EQ(split->partitionKeys().at("country"), std::nullopt);
}

TEST_F(PaimonConnectorSplitTest, builder) {
  auto split = PaimonConnectorSplitBuilder(
                   kConnectorId,
                   /*snapshotId=*/5,
                   PaimonTableType::kPrimaryKey,
                   FileFormat::PARQUET)
                   .addFile(0, "s3://bucket/data-001.parquet", 1024, 0)
                   .addFile(1, "s3://bucket/data-002.parquet", 2048, 1)
                   .partitionKey("dt", "2024-01-01")
                   .partitionKey("country", std::nullopt)
                   .tableBucketNumber(3)
                   .build();

  EXPECT_EQ(split->snapshotId(), 5);
  EXPECT_EQ(split->fileFormat(), FileFormat::PARQUET);
  EXPECT_TRUE(split->rawConvertible());

  ASSERT_EQ(split->dataFiles().size(), 2);
  EXPECT_EQ(split->dataFiles()[0].path, "s3://bucket/data-001.parquet");
  EXPECT_EQ(split->dataFiles()[0].size, 1024);
  EXPECT_EQ(split->dataFiles()[0].level, 0);
  EXPECT_EQ(split->dataFiles()[1].path, "s3://bucket/data-002.parquet");
  EXPECT_EQ(split->dataFiles()[1].size, 2048);
  EXPECT_EQ(split->dataFiles()[1].level, 1);
  EXPECT_EQ(split->tableBucketNumber(), 3);
  EXPECT_EQ(split->partitionKeys().at("dt"), "2024-01-01");
  EXPECT_EQ(split->partitionKeys().at("country"), std::nullopt);
}

TEST_F(PaimonConnectorSplitTest, builderDefaults) {
  auto split = PaimonConnectorSplitBuilder(
                   kConnectorId,
                   /*snapshotId=*/1,
                   PaimonTableType::kAppendOnly,
                   FileFormat::NIMBLE)
                   .addFile(0, "data.nimble", 512)
                   .build();

  ASSERT_EQ(split->dataFiles().size(), 1);
  const auto& file = split->dataFiles()[0];
  EXPECT_EQ(file.schemaId, 0);
  EXPECT_EQ(file.path, "data.nimble");
  EXPECT_EQ(file.size, 512);
  EXPECT_EQ(file.rowCount, 0);
  EXPECT_EQ(file.level, 0);
  EXPECT_EQ(file.minSequenceNumber, 0);
  EXPECT_EQ(file.maxSequenceNumber, 0);
  EXPECT_EQ(file.deleteRowCount, 0);
  EXPECT_EQ(file.creationTimeMs, 0);
  EXPECT_EQ(file.type, PaimonDataFile::Type::kData);
  EXPECT_EQ(file.source, PaimonDataFile::Source::kAppend);
  EXPECT_FALSE(file.deletionFile.has_value());
  EXPECT_EQ(split->fileFormat(), FileFormat::NIMBLE);
  EXPECT_EQ(split->tableBucketNumber(), std::nullopt);
  EXPECT_TRUE(split->partitionKeys().empty());
}

TEST_F(PaimonConnectorSplitTest, serializeRoundTrip) {
  const PaimonDataFile meta(
      /*schemaId=*/0,
      "s3://bucket/data-001.parquet",
      /*size=*/4096,
      /*rowCount=*/1000,
      /*level=*/2,
      /*minSequenceNumber=*/10,
      /*maxSequenceNumber=*/20,
      /*deleteRowCount=*/5,
      /*creationTimeMs=*/1700000000,
      PaimonDataFile::Type::kData,
      PaimonDataFile::Source::kCompact,
      PaimonDeletionFile{"s3://bucket/deletion-001.bin", 0, 128, 5});

  std::vector<PaimonDataFile> files{meta};
  PartitionKeys partitionKeys{{"dt", "2024-01-01"}, {"country", std::nullopt}};

  auto original = std::make_shared<PaimonConnectorSplit>(
      kConnectorId,
      /*snapshotId=*/42,
      PaimonTableType::kPrimaryKey,
      FileFormat::PARQUET,
      files,
      partitionKeys,
      /*tableBucketNumber=*/7,
      /*rawConvertible=*/false);

  auto serialized = original->serialize();
  auto deserialized = PaimonConnectorSplit::create(serialized);

  EXPECT_EQ(deserialized->connectorId, kConnectorId);
  EXPECT_EQ(deserialized->snapshotId(), 42);
  EXPECT_EQ(deserialized->tableType(), PaimonTableType::kPrimaryKey);
  EXPECT_EQ(deserialized->fileFormat(), FileFormat::PARQUET);
  EXPECT_FALSE(deserialized->rawConvertible());
  EXPECT_EQ(deserialized->tableBucketNumber(), 7);

  ASSERT_EQ(deserialized->dataFiles().size(), 1);
  const auto& file = deserialized->dataFiles()[0];
  EXPECT_EQ(file.path, "s3://bucket/data-001.parquet");
  EXPECT_EQ(file.size, 4096);
  EXPECT_EQ(file.rowCount, 1000);
  EXPECT_EQ(file.level, 2);
  EXPECT_EQ(file.minSequenceNumber, 10);
  EXPECT_EQ(file.maxSequenceNumber, 20);
  EXPECT_EQ(file.deleteRowCount, 5);
  EXPECT_EQ(file.creationTimeMs, 1700000000);
  EXPECT_EQ(file.source, PaimonDataFile::Source::kCompact);

  ASSERT_TRUE(file.deletionFile.has_value());
  EXPECT_EQ(file.deletionFile->path, "s3://bucket/deletion-001.bin");
  EXPECT_EQ(file.deletionFile->offset, 0);
  EXPECT_EQ(file.deletionFile->length, 128);
  EXPECT_EQ(file.deletionFile->cardinality, 5);

  ASSERT_EQ(deserialized->partitionKeys().count("dt"), 1);
  EXPECT_EQ(deserialized->partitionKeys().at("dt"), "2024-01-01");
  ASSERT_EQ(deserialized->partitionKeys().count("country"), 1);
  EXPECT_EQ(deserialized->partitionKeys().at("country"), std::nullopt);

  ASSERT_EQ(deserialized->dataFiles().size(), 1);
  EXPECT_EQ(deserialized->dataFiles()[0].path, "s3://bucket/data-001.parquet");
  EXPECT_EQ(deserialized->dataFiles()[0].size, 4096);
}

TEST_F(PaimonConnectorSplitTest, paimonTableTypeStringAndParse) {
  EXPECT_EQ(paimonTableTypeString(PaimonTableType::kAppendOnly), "APPEND_ONLY");
  EXPECT_EQ(paimonTableTypeString(PaimonTableType::kPrimaryKey), "PRIMARY_KEY");

  EXPECT_EQ(
      paimonTableTypeFromString("APPEND_ONLY"), PaimonTableType::kAppendOnly);
  EXPECT_EQ(
      paimonTableTypeFromString("PRIMARY_KEY"), PaimonTableType::kPrimaryKey);

  VELOX_ASSERT_THROW(
      paimonTableTypeFromString("UNKNOWN"), "Unknown PaimonTableType: UNKNOWN");
}

TEST_F(PaimonConnectorSplitTest, tableTypeStreamAndFormat) {
  {
    std::ostringstream os;
    os << PaimonTableType::kAppendOnly;
    EXPECT_EQ(os.str(), "APPEND_ONLY");
  }
  {
    std::ostringstream os;
    os << PaimonTableType::kPrimaryKey;
    EXPECT_EQ(os.str(), "PRIMARY_KEY");
  }

  EXPECT_EQ(fmt::format("{}", PaimonTableType::kAppendOnly), "APPEND_ONLY");
  EXPECT_EQ(fmt::format("{}", PaimonTableType::kPrimaryKey), "PRIMARY_KEY");
}
