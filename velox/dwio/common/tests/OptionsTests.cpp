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

// #include "velox/common/compression/Compression.h"
#include "velox/dwio/common/Options.h"

using namespace ::testing;
using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;

TEST(OptionsTests, defaultRowNumberColumnInfoTest) {
  // appendRowNumberColumn flag should be false by default
  RowReaderOptions rowReaderOptions;
  ASSERT_EQ(std::nullopt, rowReaderOptions.rowNumberColumnInfo());
}

TEST(OptionsTests, setRowNumberColumnInfoTest) {
  RowReaderOptions rowReaderOptions;
  RowNumberColumnInfo rowNumberColumnInfo;
  rowNumberColumnInfo.insertPosition = 0;
  rowNumberColumnInfo.name = "test";
  rowReaderOptions.setRowNumberColumnInfo(rowNumberColumnInfo);
  auto rowNumberColumn = rowReaderOptions.rowNumberColumnInfo().value();
  ASSERT_EQ(rowNumberColumnInfo.insertPosition, rowNumberColumn.insertPosition);
  ASSERT_EQ(rowNumberColumnInfo.name, rowNumberColumn.name);
}

TEST(OptionsTests, testRowNumberColumnInfoInCopy) {
  RowReaderOptions rowReaderOptions;
  RowReaderOptions rowReaderOptionsCopy{rowReaderOptions};
  ASSERT_EQ(std::nullopt, rowReaderOptionsCopy.rowNumberColumnInfo());

  RowNumberColumnInfo rowNumberColumnInfo;
  rowNumberColumnInfo.insertPosition = 0;
  rowNumberColumnInfo.name = "test";
  rowReaderOptions.setRowNumberColumnInfo(rowNumberColumnInfo);
  RowReaderOptions rowReaderOptionsSecondCopy{rowReaderOptions};
  auto rowNumberColumn =
      rowReaderOptionsSecondCopy.rowNumberColumnInfo().value();
  ASSERT_EQ(rowNumberColumnInfo.insertPosition, rowNumberColumn.insertPosition);
  ASSERT_EQ(rowNumberColumnInfo.name, rowNumberColumn.name);
}

TEST(OptionsTests, WriterOptionsSerdeTestRoundTripWithAllFields) {
  Type::registerSerDe();
  WriterOptions::registerSerDe();

  auto opts = std::make_shared<WriterOptions>();

  // Schema: row<a:bigint>
  TypePtr schema = ROW({{"a", BIGINT()}});
  opts->schema = schema;
  opts->compressionKind = CompressionKind::CompressionKind_ZSTD;
  opts->serdeParameters = {{"k1", "v1"}, {"k2", "v2"}};
  opts->sessionTimezoneName = "America/Los_Angeles";
  opts->adjustTimestampToTimezone = true;

  // Note: We intentionally do NOT set memoryPool, nonReclaimableSection, or the
  // factory callables because those are not serialized by design.

  // Serialize
  folly::dynamic serialized = opts->serialize();

  // Basic shape checks on serialized output
  ASSERT_TRUE(serialized.isObject());
  // Always present:
  ASSERT_TRUE(serialized.count("adjustTimestampToTimezone") == 1);
  // Populated in this test:
  EXPECT_TRUE(serialized.count("schema") == 1);
  EXPECT_TRUE(serialized.count("compressionKind") == 1);
  EXPECT_TRUE(serialized.count("serdeParameters") == 1);
  EXPECT_TRUE(serialized.count("sessionTimezoneName") == 1);
  // Not serialized:
  EXPECT_EQ(serialized.count("memoryPool"), 0);
  EXPECT_EQ(serialized.count("spillConfig"), 0);
  EXPECT_EQ(serialized.count("nonReclaimableSection"), 0);
  EXPECT_EQ(serialized.count("memoryReclaimerFactory"), 0);
  EXPECT_EQ(serialized.count("flushPolicyFactory"), 0);

  // Deserialize
  auto roundTripped = ISerializable::deserialize<WriterOptions>(serialized);
  ASSERT_NE(roundTripped, nullptr);

  // Validate schema equality (compare types by string or deep equals)
  ASSERT_TRUE(roundTripped->schema != nullptr);
  EXPECT_EQ(roundTripped->schema->toString(), schema->toString());

  // Validate compression kind
  ASSERT_TRUE(roundTripped->compressionKind.has_value());
  EXPECT_EQ(
      *roundTripped->compressionKind, CompressionKind::CompressionKind_ZSTD);

  // Validate serde parameters
  EXPECT_EQ(roundTripped->serdeParameters.size(), 2);
  EXPECT_EQ(roundTripped->serdeParameters.at("k1"), "v1");
  EXPECT_EQ(roundTripped->serdeParameters.at("k2"), "v2");

  // Validate timezone + adjust flag
  EXPECT_EQ(roundTripped->sessionTimezoneName, "America/Los_Angeles");
  EXPECT_TRUE(roundTripped->adjustTimestampToTimezone);

  // Validate that non-serialized fields remain default/null
  EXPECT_EQ(roundTripped->memoryPool, nullptr);
  EXPECT_EQ(roundTripped->spillConfig, nullptr);
  EXPECT_EQ(roundTripped->nonReclaimableSection, nullptr);

  // Factories should be default-initialized (callables present but return
  // nullptr / no-op)
  ASSERT_TRUE(static_cast<bool>(roundTripped->memoryReclaimerFactory));
  EXPECT_EQ(roundTripped->memoryReclaimerFactory(), nullptr);
  // flushPolicyFactory is default-constructed empty unless you set it;
  // implementation may leave it empty by default — check it doesn't crash when
  // inspected. We only assert it is empty here (adjust if your constructor sets
  // one).
  EXPECT_FALSE(static_cast<bool>(roundTripped->flushPolicyFactory));
}

TEST(WriterOptionsSerdeTest, DefaultsAndUnknownFieldsAreSafe) {
  Type::registerSerDe();
  WriterOptions::registerSerDe();

  // Minimal object: only adjustTimestampToTimezone is always emitted by
  // serialize(). Here we simulate missing keys and an extra unknown key to
  // ensure robustness.
  folly::dynamic minimal = folly::dynamic::object;
  minimal["adjustTimestampToTimezone"] = false;
  minimal["unknownExtraKey"] = "ignored";

  auto opts = WriterOptions::deserialize(minimal);
  ASSERT_NE(opts, nullptr);

  // Absent -> defaults
  EXPECT_TRUE(opts->schema == nullptr);
  EXPECT_FALSE(opts->compressionKind.has_value());
  EXPECT_TRUE(opts->serdeParameters.empty());
  EXPECT_TRUE(opts->sessionTimezoneName.empty());
  EXPECT_FALSE(opts->adjustTimestampToTimezone); // false from our dynamic

  // Non-serialized pointers/factories remain default/null
  EXPECT_EQ(opts->memoryPool, nullptr);
  EXPECT_EQ(opts->spillConfig, nullptr);
  EXPECT_EQ(opts->nonReclaimableSection, nullptr);
  // factory callables default as in your implementation
  ASSERT_TRUE(static_cast<bool>(opts->memoryReclaimerFactory));
  EXPECT_EQ(opts->memoryReclaimerFactory(), nullptr);
  EXPECT_FALSE(static_cast<bool>(opts->flushPolicyFactory));
}
