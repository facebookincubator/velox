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

#include "velox/connectors/hive/hudi/HudiLogFileReader.h"

#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive::hudi;

namespace {

// Reads the entire contents of the named example log file into a string.
std::string readExample(const std::string& relativePath) {
  const auto path = test::getDataFilePath(
      "velox/connectors/hive/hudi/tests", "examples/" + relativePath);
  std::ifstream file(path, std::ios::binary);
  VELOX_CHECK(file.is_open(), "Failed to open example log file: {}", path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

// The fixtures below are copied verbatim from the Apache hudi-rs test corpus so
// that the C++ parser is validated against bytes produced by Hudi itself.
constexpr auto kAvroDataLog =
    "valid_log_avro_data/"
    ".ff32ab89-5ad0-4968-83b4-89a34c95d32f-0_20250316025816068.log.1_0-54-122";
constexpr auto kDeleteLog =
    "valid_log_delete/"
    ".6d3d1d6e-2298-4080-a0c1-494877d6f40a-0_20250618054711154.log.1_0-26-85";
constexpr auto kRollbackLog =
    "valid_log_rollback/"
    ".0712b9f9-d2d5-4cae-bcf4-8fd7146af503-0_20250126040823628.log.2_1-0-1";

} // namespace

TEST(HudiLogFileReaderTest, rollbackCommandBlock) {
  const auto data = readExample(kRollbackLog);
  HudiLogFileReader reader(data);
  const auto blocks = reader.readAllBlocks();

  ASSERT_EQ(blocks.size(), 1);
  const auto& block = blocks[0];
  EXPECT_EQ(block.formatVersion, LogFormatVersion::kV1);
  EXPECT_EQ(block.blockType, HudiLogBlockType::kCommand);
  EXPECT_TRUE(block.isRollbackBlock());
  EXPECT_FALSE(block.isDataBlock());
  EXPECT_FALSE(block.isDeleteBlock());
  // Command blocks carry InstantTime, CommandBlockType and TargetInstantTime.
  EXPECT_EQ(block.header.size(), 3);
  EXPECT_EQ(block.instantTime(), "20250126040936578");
  ASSERT_TRUE(block.targetInstantTime().has_value());
  EXPECT_EQ(block.targetInstantTime().value(), "20250126040826878");
  ASSERT_TRUE(block.commandBlockType().has_value());
  EXPECT_EQ(block.commandBlockType().value(), HudiCommandBlockType::kRollback);
  EXPECT_TRUE(block.content.empty());
}

TEST(HudiLogFileReaderTest, deleteBlockFraming) {
  const auto data = readExample(kDeleteLog);
  HudiLogFileReader reader(data);
  const auto blocks = reader.readAllBlocks();

  ASSERT_EQ(blocks.size(), 1);
  const auto& block = blocks[0];
  EXPECT_EQ(block.formatVersion, LogFormatVersion::kV1);
  EXPECT_EQ(block.blockType, HudiLogBlockType::kDelete);
  EXPECT_TRUE(block.isDeleteBlock());
  EXPECT_FALSE(block.isDataBlock());
  EXPECT_FALSE(block.isRollbackBlock());
  // Delete blocks carry InstantTime and Schema.
  EXPECT_EQ(block.header.size(), 2);
  EXPECT_EQ(block.instantTime(), "20250618054714114");
  EXPECT_FALSE(block.targetInstantTime().has_value());
  EXPECT_FALSE(block.content.empty());
}

TEST(HudiLogFileReaderTest, avroDataBlockFraming) {
  const auto data = readExample(kAvroDataLog);
  HudiLogFileReader reader(data);
  const auto blocks = reader.readAllBlocks();

  ASSERT_EQ(blocks.size(), 1);
  const auto& block = blocks[0];
  EXPECT_EQ(block.formatVersion, LogFormatVersion::kV1);
  EXPECT_EQ(block.blockType, HudiLogBlockType::kAvroData);
  EXPECT_TRUE(block.isDataBlock());
  EXPECT_FALSE(block.isDeleteBlock());
  EXPECT_FALSE(block.isRollbackBlock());
  // Data blocks carry InstantTime and Schema.
  EXPECT_EQ(block.header.size(), 2);
  EXPECT_EQ(block.instantTime(), "20250316025828811");
  ASSERT_TRUE(block.schemaJson().has_value());
  EXPECT_FALSE(block.schemaJson().value().empty());
  EXPECT_FALSE(block.content.empty());
}

TEST(HudiLogFileReaderTest, emptyFile) {
  HudiLogFileReader reader(std::string_view{});
  EXPECT_TRUE(reader.readAllBlocks().empty());
}

TEST(HudiLogFileReaderTest, truncatedFile) {
  const auto data = readExample(kDeleteLog);
  ASSERT_GT(data.size(), 25);
  const auto truncated = data.substr(0, 25);

  HudiLogFileReader reader(truncated);
  VELOX_ASSERT_RUNTIME_THROW(
      reader.readAllBlocks(), "Truncated Hudi log file while reading block");
}

TEST(HudiLogFileReaderTest, corruptedMagicMarker) {
  // Enough bytes for a magic marker, but not the expected one: this is not a
  // clean end of file, so it must be reported as corruption rather than
  // silently ending the scan.
  const std::string data{"GARBAGE"};

  HudiLogFileReader reader(data);
  VELOX_ASSERT_USER_THROW(reader.readAllBlocks(), "Hudi log file is corrupted");
}

TEST(HudiLogFileReaderTest, invalidFormatVersion) {
  // Magic marker, an 8-byte block length (unused, since the format version
  // check fires before it) and a 4-byte format version of 99.
  std::string data{kHudiLogMagic};
  data.append(8, '\0');
  data.append({'\0', '\0', '\0', static_cast<char>(99)});

  HudiLogFileReader reader(data);
  VELOX_ASSERT_USER_THROW(
      reader.readAllBlocks(), "Unsupported Hudi log format version: 99");
}
