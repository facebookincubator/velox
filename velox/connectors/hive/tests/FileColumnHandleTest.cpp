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

#include "velox/connectors/common/FileColumnHandle.h"
#include <fmt/format.h>
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/TableHandle.h"

namespace facebook::velox::connector {
namespace {

TEST(FileColumnHandleTest, columnTypeName) {
  EXPECT_EQ(
      FileColumnHandle::columnTypeName(
          FileColumnHandle::ColumnType::kPartitionKey),
      "PartitionKey");
  EXPECT_EQ(
      FileColumnHandle::columnTypeName(FileColumnHandle::ColumnType::kRegular),
      "Regular");
  EXPECT_EQ(
      FileColumnHandle::columnTypeName(
          FileColumnHandle::ColumnType::kSynthesized),
      "Synthesized");
  EXPECT_EQ(
      FileColumnHandle::columnTypeName(FileColumnHandle::ColumnType::kRowIndex),
      "RowIndex");
  EXPECT_EQ(
      FileColumnHandle::columnTypeName(FileColumnHandle::ColumnType::kRowId),
      "RowId");
  EXPECT_EQ(
      FileColumnHandle::columnTypeName(
          FileColumnHandle::ColumnType::kReadRange),
      "ReadRange");
}

TEST(FileColumnHandleTest, columnTypeFromName) {
  EXPECT_EQ(
      FileColumnHandle::columnTypeFromName("PartitionKey"),
      FileColumnHandle::ColumnType::kPartitionKey);
  EXPECT_EQ(
      FileColumnHandle::columnTypeFromName("Regular"),
      FileColumnHandle::ColumnType::kRegular);
  EXPECT_EQ(
      FileColumnHandle::columnTypeFromName("Synthesized"),
      FileColumnHandle::ColumnType::kSynthesized);
  EXPECT_EQ(
      FileColumnHandle::columnTypeFromName("RowIndex"),
      FileColumnHandle::ColumnType::kRowIndex);
  EXPECT_EQ(
      FileColumnHandle::columnTypeFromName("RowId"),
      FileColumnHandle::ColumnType::kRowId);
  EXPECT_EQ(
      FileColumnHandle::columnTypeFromName("ReadRange"),
      FileColumnHandle::ColumnType::kReadRange);
}

TEST(FileColumnHandleTest, columnTypeFromNameInvalid) {
  VELOX_ASSERT_THROW(
      FileColumnHandle::columnTypeFromName("Unknown"),
      "Unknown column type name: Unknown");
}

TEST(FileColumnHandleTest, columnTypeRoundTrip) {
  const std::vector<FileColumnHandle::ColumnType> allTypes = {
      FileColumnHandle::ColumnType::kPartitionKey,
      FileColumnHandle::ColumnType::kRegular,
      FileColumnHandle::ColumnType::kSynthesized,
      FileColumnHandle::ColumnType::kRowIndex,
      FileColumnHandle::ColumnType::kRowId,
      FileColumnHandle::ColumnType::kReadRange,
  };
  for (auto type : allTypes) {
    EXPECT_EQ(
        FileColumnHandle::columnTypeFromName(
            FileColumnHandle::columnTypeName(type)),
        type);
  }
}

TEST(FileColumnHandleTest, fmtFormatter) {
  EXPECT_EQ(
      fmt::format("{}", FileColumnHandle::ColumnType::kRegular), "Regular");
  EXPECT_EQ(
      fmt::format("{}", FileColumnHandle::ColumnType::kPartitionKey),
      "PartitionKey");
}

TEST(FileColumnHandleTest, checkConsistentWith) {
  using ColumnType = FileColumnHandle::ColumnType;
  const hive::HiveColumnHandle handle(
      "col1", ColumnType::kRegular, BIGINT(), BIGINT());

  handle.checkConsistentWith(
      hive::HiveColumnHandle("col1", ColumnType::kRegular, BIGINT(), BIGINT()));

  VELOX_ASSERT_THROW(
      handle.checkConsistentWith(
          hive::HiveColumnHandle(
              "col1", ColumnType::kPartitionKey, BIGINT(), BIGINT())),
      "Inconsistent column handle type: col1, expected Regular, got PartitionKey");

  VELOX_ASSERT_THROW(
      handle.checkConsistentWith(
          hive::HiveColumnHandle(
              "col1", ColumnType::kRegular, VARCHAR(), VARCHAR())),
      "Inconsistent column handle data type: col1, expected BIGINT, got VARCHAR");
}

} // namespace
} // namespace facebook::velox::connector
