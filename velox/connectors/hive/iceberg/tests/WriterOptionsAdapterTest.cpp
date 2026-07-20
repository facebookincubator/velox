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

#include "velox/connectors/hive/iceberg/WriterOptionsAdapter.h"

#include <gtest/gtest.h>
#ifdef VELOX_ENABLE_NIMBLE
#include "dwio/nimble/velox/writer/fb/NimbleWriter.h"
#endif
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/dwio/parquet/common/ParquetConfig.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

// Verifies IcebergColumnHandle carries the Iceberg V3 metadata supplied at
// construction and exposes it via icebergMetadata(), and that the default
// (omitted) metadata is empty for legacy callers.
TEST(WriterOptionsAdapterTest, icebergColumnHandleCarriesMetadata) {
  IcebergFieldMetadata metadata;
  metadata.required = true;
  metadata.binaryType = "UUID";
  metadata.length = 16;

  IcebergColumnHandle handle(
      "u",
      HiveColumnHandle::ColumnType::kRegular,
      VARBINARY(),
      parquet::ParquetFieldId{/*fieldId*/ 5, /*children*/ {}},
      /*requiredSubfields*/ {},
      /*initialDefaultValue*/ std::nullopt,
      metadata);

  const auto& recovered = handle.icebergMetadata();
  EXPECT_TRUE(recovered.required);
  EXPECT_EQ(recovered.binaryType, "UUID");
  EXPECT_EQ(recovered.length, 16);

  // Legacy callers that omit metadata get an empty instance.
  IcebergColumnHandle legacyHandle(
      "a",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      parquet::ParquetFieldId{/*fieldId*/ 1, /*children*/ {}});
  EXPECT_TRUE(legacyHandle.icebergMetadata().empty());
}

// Verifies the dispatch table in createWriterOptionsAdapter():
// PARQUET, ORC, DWRF, NIMBLE return non-null adapters; everything else
// returns null. This is the single source of truth for which file formats
// the Iceberg DataSink supports on the write path. ORC routes through the
// same DwrfWriterOptionsAdapter because Meta's DWRF is an ORC
// implementation.
TEST(WriterOptionsAdapterTest, createWriterOptionsAdapterDispatch) {
  EXPECT_NE(
      createWriterOptionsAdapter(dwio::common::FileFormat::PARQUET), nullptr);
  EXPECT_NE(createWriterOptionsAdapter(dwio::common::FileFormat::ORC), nullptr);
  EXPECT_NE(
      createWriterOptionsAdapter(dwio::common::FileFormat::DWRF), nullptr);
#ifdef VELOX_ENABLE_NIMBLE
  EXPECT_NE(
      createWriterOptionsAdapter(dwio::common::FileFormat::NIMBLE), nullptr);
#endif

  // TEXT, JSON, ALPHA, etc. are intentionally unsupported on the
  // write path until each gets its own end-to-end coverage.
  EXPECT_EQ(
      createWriterOptionsAdapter(dwio::common::FileFormat::TEXT), nullptr);
  EXPECT_EQ(
      createWriterOptionsAdapter(dwio::common::FileFormat::JSON), nullptr);
}

// Verifies isSupportedFileFormat() agrees with createWriterOptionsAdapter().
TEST(WriterOptionsAdapterTest, isSupportedFileFormatMatchesDispatch) {
  EXPECT_TRUE(isSupportedFileFormat(dwio::common::FileFormat::PARQUET));
  EXPECT_TRUE(isSupportedFileFormat(dwio::common::FileFormat::ORC));
  EXPECT_TRUE(isSupportedFileFormat(dwio::common::FileFormat::DWRF));
#ifdef VELOX_ENABLE_NIMBLE
  EXPECT_TRUE(isSupportedFileFormat(dwio::common::FileFormat::NIMBLE));
#endif

  EXPECT_FALSE(isSupportedFileFormat(dwio::common::FileFormat::TEXT));
  EXPECT_FALSE(isSupportedFileFormat(dwio::common::FileFormat::JSON));
}

// Verifies the manifest format string written into Iceberg commit messages
// matches the cross-engine convention. Iceberg's manifest vocabulary has no
// DWRF/NIMBLE enum, so both report "ORC" (matching the Java planner's
// FileFormat.{DWRF,NIMBLE}.toIceberg()) so downstream Iceberg consumers can
// interpret the message. Parquet reports "PARQUET" because Iceberg has a
// native enum for it. The on-disk format is identified at read time via
// the file extension and on-disk magic bytes, so writing "ORC" for NIMBLE
// is safe.
TEST(WriterOptionsAdapterTest, toManifestFormatString) {
  EXPECT_EQ(
      toManifestFormatString(dwio::common::FileFormat::PARQUET), "PARQUET");
  EXPECT_EQ(toManifestFormatString(dwio::common::FileFormat::ORC), "ORC");
  EXPECT_EQ(toManifestFormatString(dwio::common::FileFormat::DWRF), "ORC");
#ifdef VELOX_ENABLE_NIMBLE
  EXPECT_EQ(toManifestFormatString(dwio::common::FileFormat::NIMBLE), "ORC");
#endif
}

// Verifies the Parquet adapter's pre-config hook installs the Iceberg-spec
// timestamp serdeParameters. These values stay in common writer options until
// the Parquet writer constructor reads them.
TEST(WriterOptionsAdapterTest, parquetPreConfigsSetsTimestampSerdeParameters) {
  auto adapter = createWriterOptionsAdapter(dwio::common::FileFormat::PARQUET);
  ASSERT_NE(adapter, nullptr);

  dwio::common::WriterOptions options;
  adapter->applyPreConfigs(options);

  EXPECT_EQ(
      options.serdeParameters[std::string(
          parquet::ParquetConfig::kWriterSerdeTimestampUnit)],
      "6");
  EXPECT_EQ(
      options.serdeParameters[std::string(
          parquet::ParquetConfig::kWriterSerdeTimestampTimezone)],
      "");
}

// Verifies the DWRF adapter's post-config hook overrides timestamp settings
// regardless of what config processing left in place. The Iceberg spec
// requires timestamps NOT be adjusted to UTC;
// IcebergDataSink::createWriterOptions must use this adapter contract.
TEST(WriterOptionsAdapterTest, dwrfPostConfigsOverridesTimestampFields) {
  auto adapter = createWriterOptionsAdapter(dwio::common::FileFormat::DWRF);
  ASSERT_NE(adapter, nullptr);

  dwrf::WriterOptions options;
  options.adjustTimestampToTimezone = true;
  options.sessionTimezone = tz::locateZone("America/Los_Angeles");

  adapter->applyPostConfigs(options);

  EXPECT_FALSE(options.adjustTimestampToTimezone);
  EXPECT_EQ(options.sessionTimezone, nullptr);
}

// Verifies ORC routes through the same DwrfWriterOptionsAdapter as DWRF:
// the post-config hook overrides the same dwrf::WriterOptions timestamp
// fields. This proves that the ORC dispatch is wired to the DWRF adapter
// (the cross-engine convention — Meta's DWRF is an ORC implementation).
TEST(WriterOptionsAdapterTest, orcRoutesToDwrfAdapter) {
  auto adapter = createWriterOptionsAdapter(dwio::common::FileFormat::ORC);
  ASSERT_NE(adapter, nullptr);

  EXPECT_EQ(adapter->manifestFormatString(), "ORC");

  dwrf::WriterOptions options;
  options.adjustTimestampToTimezone = true;
  options.sessionTimezone = tz::locateZone("America/Los_Angeles");

  adapter->applyPostConfigs(options);

  EXPECT_FALSE(options.adjustTimestampToTimezone);
  EXPECT_EQ(options.sessionTimezone, nullptr);
}

// Verifies toManifestFormatString() throws for unsupported formats rather
// than silently returning an incorrect string.
TEST(WriterOptionsAdapterTest, toManifestFormatStringThrowsForUnsupported) {
  VELOX_ASSERT_THROW(
      toManifestFormatString(dwio::common::FileFormat::TEXT),
      "Unsupported file format for Iceberg manifest");
  VELOX_ASSERT_THROW(
      toManifestFormatString(dwio::common::FileFormat::JSON),
      "Unsupported file format for Iceberg manifest");
}

#ifdef VELOX_ENABLE_NIMBLE
// ---------------------------------------------------------------------------
// NIMBLE Iceberg V3 attribute stamping.
//
// Verifies that IcebergDataSink correctly stamps Iceberg field IDs and V3
// type attributes onto Nimble writer options via the WriterOptionsAdapter.
// The adapter converts the IcebergFieldId tree into dotted-path attributes
// (e.g., "a.b" -> {"iceberg.id":"2", "iceberg.required":"true"}) that survive
// through NimbleWriterFactory into the Nimble file schema. These tests ensure
// the writer-side contract is preserved: empty trees are no-ops for backward
// compatibility, and populated trees produce correct per-column attributes.
// ---------------------------------------------------------------------------

// Verifies an empty Iceberg field-id tree is a no-op: attributesByColumn
// stays empty after applyPostConfigs. This is the no-op upgrade path for
// every existing NIMBLE writer call site -- the on-disk file is
// byte-identical to pre-stack output.
TEST(WriterOptionsAdapterTest, nimbleApplyPostConfigsEmptyTreeIsNoOp) {
  auto adapter = createWriterOptionsAdapter(
      dwio::common::FileFormat::NIMBLE, IcebergFieldId{});
  ASSERT_NE(adapter, nullptr);

  velox::nimble::NimbleWriterOptions options;
  options.schema = ROW({"a"}, {INTEGER()});
  adapter->applyPostConfigs(options);

  EXPECT_TRUE(options.attributesByColumn.empty());
}

// Verifies the NIMBLE adapter stamps `iceberg.id` onto top-level columns
// in the dotted-path keyed map. The walk goes through the schema in
// lockstep with the field-id tree, so each column gets the right id.
TEST(WriterOptionsAdapterTest, nimbleApplyPostConfigsStampsTopLevelIcebergIds) {
  // Build a synthetic top-level wrapper with one child per column,
  // matching how IcebergDataSink constructs the tree.
  IcebergFieldId icebergFieldIds;
  icebergFieldIds.children.emplace_back(
      IcebergFieldId{/*fieldId*/ 7, /*children*/ {}});
  icebergFieldIds.children.emplace_back(
      IcebergFieldId{/*fieldId*/ 12, /*children*/ {}});

  auto adapter = createWriterOptionsAdapter(
      dwio::common::FileFormat::NIMBLE, std::move(icebergFieldIds));
  ASSERT_NE(adapter, nullptr);

  velox::nimble::NimbleWriterOptions options;
  options.schema = ROW({"id", "name"}, {BIGINT(), VARCHAR()});

  adapter->applyPostConfigs(options);

  ASSERT_EQ(options.attributesByColumn.size(), 2u);
  ASSERT_EQ(options.attributesByColumn.count("id"), 1u);
  ASSERT_EQ(options.attributesByColumn.count("name"), 1u);

  const std::vector<std::pair<std::string, std::string>> kIdAttrs = {
      {"iceberg.id", "7"},
  };
  const std::vector<std::pair<std::string, std::string>> kNameAttrs = {
      {"iceberg.id", "12"},
  };
  EXPECT_EQ(options.attributesByColumn.at("id"), kIdAttrs);
  EXPECT_EQ(options.attributesByColumn.at("name"), kNameAttrs);
}

// Verifies the NIMBLE adapter recurses into nested struct (RowType)
// children, producing dotted-path keys like "user.name". Diff B's
// writer-side resolveDottedPath supports this case directly. Array/Map
// nested field-ids are deferred to a follow-up.
TEST(WriterOptionsAdapterTest, nimbleApplyPostConfigsStampsNestedStructIds) {
  IcebergFieldId icebergFieldIds;
  // Top-level child 0 = column "user" (id 1) with two nested struct
  // children: "name" (id 2) and "age" (id 3).
  IcebergFieldId userField;
  userField.fieldId = 1;
  userField.children.emplace_back(
      IcebergFieldId{/*fieldId*/ 2, /*children*/ {}});
  userField.children.emplace_back(
      IcebergFieldId{/*fieldId*/ 3, /*children*/ {}});
  icebergFieldIds.children.emplace_back(std::move(userField));

  auto adapter = createWriterOptionsAdapter(
      dwio::common::FileFormat::NIMBLE, std::move(icebergFieldIds));
  ASSERT_NE(adapter, nullptr);

  velox::nimble::NimbleWriterOptions options;
  options.schema =
      ROW({"user"}, {ROW({"name", "age"}, {VARCHAR(), INTEGER()})});

  adapter->applyPostConfigs(options);

  // Parent struct + both leaves all annotated with their field-ids.
  ASSERT_EQ(options.attributesByColumn.size(), 3u);
  const std::vector<std::pair<std::string, std::string>> kUserAttrs = {
      {"iceberg.id", "1"},
  };
  const std::vector<std::pair<std::string, std::string>> kNameAttrs = {
      {"iceberg.id", "2"},
  };
  const std::vector<std::pair<std::string, std::string>> kAgeAttrs = {
      {"iceberg.id", "3"},
  };
  EXPECT_EQ(options.attributesByColumn.at("user"), kUserAttrs);
  EXPECT_EQ(options.attributesByColumn.at("user.name"), kNameAttrs);
  EXPECT_EQ(options.attributesByColumn.at("user.age"), kAgeAttrs);
}

// Verifies the NIMBLE adapter stamps the Iceberg V3 type attributes carried
// in the parallel IcebergFieldMetadata tree. The UUID case sets
// `iceberg.binary-type=UUID` and `iceberg.length=16` together alongside
// `iceberg.required`, in addition to the always-present `iceberg.id`.
TEST(WriterOptionsAdapterTest, nimbleApplyPostConfigsStampsV3Attributes) {
  IcebergFieldId icebergFieldIds;
  icebergFieldIds.children.emplace_back(
      IcebergFieldId{/*fieldId*/ 5, /*children*/ {}});

  IcebergFieldMetadata metadata;
  IcebergFieldMetadata uuidColumn;
  uuidColumn.required = true;
  uuidColumn.binaryType = "UUID";
  uuidColumn.length = 16;
  metadata.children.push_back(std::move(uuidColumn));

  auto adapter = createWriterOptionsAdapter(
      dwio::common::FileFormat::NIMBLE,
      std::move(icebergFieldIds),
      std::move(metadata));
  ASSERT_NE(adapter, nullptr);

  velox::nimble::NimbleWriterOptions options;
  options.schema = ROW({"u"}, {VARBINARY()});
  adapter->applyPostConfigs(options);

  const std::vector<std::pair<std::string, std::string>> kExpected = {
      {"iceberg.id", "5"},
      {"iceberg.required", "true"},
      {"iceberg.binary-type", "UUID"},
      {"iceberg.length", "16"},
  };
  EXPECT_EQ(options.attributesByColumn.at("u"), kExpected);
}

// Verifies V3 attributes are stamped only on the columns that supply them;
// a column with empty metadata keeps the `iceberg.id`-only shape.
TEST(
    WriterOptionsAdapterTest,
    nimbleApplyPostConfigsStampsRequiredOnlyWhenSet) {
  IcebergFieldId icebergFieldIds;
  icebergFieldIds.children.emplace_back(
      IcebergFieldId{/*fieldId*/ 1, /*children*/ {}});
  icebergFieldIds.children.emplace_back(
      IcebergFieldId{/*fieldId*/ 2, /*children*/ {}});

  IcebergFieldMetadata metadata;
  IcebergFieldMetadata requiredColumn;
  requiredColumn.required = true;
  metadata.children.push_back(std::move(requiredColumn));
  // Second column intentionally has empty metadata.
  metadata.children.emplace_back();

  auto adapter = createWriterOptionsAdapter(
      dwio::common::FileFormat::NIMBLE,
      std::move(icebergFieldIds),
      std::move(metadata));
  ASSERT_NE(adapter, nullptr);

  velox::nimble::NimbleWriterOptions options;
  options.schema = ROW({"a", "b"}, {BIGINT(), BIGINT()});
  adapter->applyPostConfigs(options);

  const std::vector<std::pair<std::string, std::string>> kRequiredAttrs = {
      {"iceberg.id", "1"},
      {"iceberg.required", "true"},
  };
  const std::vector<std::pair<std::string, std::string>> kPlainAttrs = {
      {"iceberg.id", "2"},
  };
  EXPECT_EQ(options.attributesByColumn.at("a"), kRequiredAttrs);
  EXPECT_EQ(options.attributesByColumn.at("b"), kPlainAttrs);
}

// Verifies the V3 attribute tree is walked in lockstep with nested struct
// children, so each leaf gets its own attributes at the right dotted path.
TEST(WriterOptionsAdapterTest, nimbleApplyPostConfigsStampsNestedStructV3) {
  IcebergFieldId icebergFieldIds;
  IcebergFieldId userField;
  userField.fieldId = 1;
  userField.children.emplace_back(
      IcebergFieldId{/*fieldId*/ 2, /*children*/ {}});
  userField.children.emplace_back(
      IcebergFieldId{/*fieldId*/ 3, /*children*/ {}});
  icebergFieldIds.children.emplace_back(std::move(userField));

  IcebergFieldMetadata metadata;
  IcebergFieldMetadata userMetadata;
  IcebergFieldMetadata idMetadata;
  idMetadata.required = true;
  idMetadata.longType = "LONG";
  IcebergFieldMetadata tsMetadata;
  tsMetadata.timestampUnit = "MICROS";
  userMetadata.children.push_back(std::move(idMetadata));
  userMetadata.children.push_back(std::move(tsMetadata));
  metadata.children.push_back(std::move(userMetadata));

  auto adapter = createWriterOptionsAdapter(
      dwio::common::FileFormat::NIMBLE,
      std::move(icebergFieldIds),
      std::move(metadata));
  ASSERT_NE(adapter, nullptr);

  velox::nimble::NimbleWriterOptions options;
  options.schema = ROW({"user"}, {ROW({"id", "ts"}, {BIGINT(), TIMESTAMP()})});
  adapter->applyPostConfigs(options);

  const std::vector<std::pair<std::string, std::string>> kIdAttrs = {
      {"iceberg.id", "2"},
      {"iceberg.required", "true"},
      {"iceberg.long-type", "LONG"},
  };
  const std::vector<std::pair<std::string, std::string>> kTsAttrs = {
      {"iceberg.id", "3"},
      {"iceberg.timestamp-unit", "MICROS"},
  };
  EXPECT_EQ(options.attributesByColumn.at("user.id"), kIdAttrs);
  EXPECT_EQ(options.attributesByColumn.at("user.ts"), kTsAttrs);
}

// Defends against passing a non-NIMBLE WriterOptions to
// NimbleWriterOptionsAdapter -- e.g. via a wrong dispatch in the factory.
// The adapter should silently no-op rather than mutate an unrelated
// options object.
TEST(WriterOptionsAdapterTest, nimbleApplyPostConfigsIgnoresNonNimbleOptions) {
  IcebergFieldId icebergFieldIds;
  icebergFieldIds.children.emplace_back(
      IcebergFieldId{/*fieldId*/ 1, /*children*/ {}});

  auto adapter = createWriterOptionsAdapter(
      dwio::common::FileFormat::NIMBLE, std::move(icebergFieldIds));
  ASSERT_NE(adapter, nullptr);

  // Pass a plain WriterOptions (NOT a NimbleWriterOptions). The
  // dynamic_cast should fail and the adapter should leave the object
  // untouched.
  dwio::common::WriterOptions options;
  options.schema = ROW({"a"}, {INTEGER()});
  EXPECT_NO_THROW(adapter->applyPostConfigs(options));
}
#endif // VELOX_ENABLE_NIMBLE

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
