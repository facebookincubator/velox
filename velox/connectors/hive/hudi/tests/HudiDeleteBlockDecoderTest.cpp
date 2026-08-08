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

#include "velox/connectors/hive/hudi/HudiDeleteBlockDecoder.h"

#include <fstream>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/hudi/HudiLogFileReader.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive::hudi;

namespace {

std::string readExample(const std::string& relativePath) {
  const auto path = test::getDataFilePath(
      "velox/connectors/hive/hudi/tests", "examples/" + relativePath);
  std::ifstream file(path, std::ios::binary);
  VELOX_CHECK(file.is_open(), "Failed to open example log file: {}", path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

constexpr auto kDeleteLog =
    "valid_log_delete/"
    ".6d3d1d6e-2298-4080-a0c1-494877d6f40a-0_20250618054711154.log.1_0-26-85";

} // namespace

TEST(HudiDeleteBlockDecoderTest, decodesThreeDeletes) {
  const auto data = readExample(kDeleteLog);
  HudiLogFileReader reader(data);
  const auto blocks = reader.readAllBlocks();
  ASSERT_EQ(blocks.size(), 1);
  ASSERT_TRUE(blocks[0].isDeleteBlock());

  const auto records = decodeHudiDeleteBlock(blocks[0]);
  ASSERT_EQ(records.size(), 3);

  // The table's primary key is `uuid` (string) and its precombine field is `ts`
  // (bigint); the fixture deletes rider-A, rider-C and rider-D from
  // san_francisco. These deletes carry no precombine value, so the ordering
  // union selects the long branch with value 0 (verified against hudi-rs).
  std::vector<std::string> recordKeys;
  for (const auto& record : records) {
    EXPECT_EQ(record.partitionPath, "city=san_francisco");
    ASSERT_TRUE(record.orderingValue.has_value());
    EXPECT_EQ(record.orderingValue.value(), 0);
    recordKeys.push_back(record.recordKey);
  }

  EXPECT_THAT(
      recordKeys,
      testing::UnorderedElementsAre(
          "334e26e9-8355-45cc-97c6-c31daf0df330",
          "e96c4396-3fad-413a-a942-4cb36106d721",
          "9909a8b1-2d15-4d3d-8ec9-efc48c536a00"));
}

TEST(HudiDeleteBlockDecoderTest, rejectsNonDeleteBlock) {
  HudiLogBlock block;
  block.blockType = HudiLogBlockType::kAvroData;
  VELOX_ASSERT_RUNTIME_THROW(
      decodeHudiDeleteBlock(block), "Expected a Hudi delete log block");
}

TEST(HudiDeleteBlockDecoderTest, rejectsUnsupportedVersion) {
  HudiLogBlock block;
  block.blockType = HudiLogBlockType::kDelete;
  // Version 2 predates the Avro-encoded delete block format decoded here.
  block.content =
      std::string({'\0', '\0', '\0', '\x02', '\0', '\0', '\0', '\0'});
  VELOX_ASSERT_RUNTIME_THROW(
      decodeHudiDeleteBlock(block), "Unsupported Hudi delete block version");
}

TEST(HudiDeleteBlockDecoderTest, rejectsTruncatedAvroContent) {
  HudiLogBlock block;
  block.blockType = HudiLogBlockType::kDelete;
  // Version 3 header followed by a single Avro varint byte whose
  // continuation bit is set, with no further byte to complete it.
  block.content = std::string(
      {'\0',
       '\0',
       '\0',
       '\x03',
       '\0',
       '\0',
       '\0',
       '\0',
       static_cast<char>(0x80)});
  VELOX_ASSERT_RUNTIME_THROW(
      decodeHudiDeleteBlock(block), "Truncated Avro datum");
}

TEST(HudiDeleteBlockDecoderTest, rejectsInvalidUnionBranch) {
  HudiLogBlock block;
  block.blockType = HudiLogBlockType::kDelete;
  // Version 3 header followed by an array block of one item whose recordKey
  // union selects branch 5, which is not a valid [null, string] branch.
  block.content = std::string(
      {'\0', '\0', '\0', '\x03', '\0', '\0', '\0', '\0', '\x02', '\x0a'});
  VELOX_ASSERT_RUNTIME_THROW(
      decodeHudiDeleteBlock(block),
      "Invalid Avro union branch for nullable string: 5");
}
