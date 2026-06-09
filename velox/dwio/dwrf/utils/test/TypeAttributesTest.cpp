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

#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"

namespace facebook::velox::dwrf::test {

// ---------------------------------------------------------------------------
// Iceberg interop: per-Type `attributes` field on the DWRF proto.
//
// DWRF files are reported as `FileFormat.ORC` in Iceberg manifests, so they
// must encode column ids and type-disambiguation metadata using the same
// string-keyed attribute convention the Iceberg spec mandates for ORC
// (Appendix A: Format-specific Requirements -> ORC). These tests exercise the
// raw proto wire format -- the writer/reader plumbing lands as a follow-up.
// ---------------------------------------------------------------------------

namespace {

// Build a Type with the given key/value attributes set on it.
proto::Type buildTypeWithAttributes(
    const std::vector<std::pair<std::string, std::string>>& attrs) {
  proto::Type type;
  type.set_kind(proto::Type_Kind_INT);
  for (const auto& [key, value] : attrs) {
    auto* pair = type.add_attributes();
    pair->set_key(key);
    pair->set_value(value);
  }
  return type;
}

} // namespace

TEST(TypeAttributesTest, AttributesAbsentByDefault) {
  // A Type built without attributes -- the existing wire format for every
  // DWRF file written today -- must round-trip and surface as an empty
  // attributes list. Protects the no-op upgrade path for existing files.
  proto::Type type;
  type.set_kind(proto::Type_Kind_INT);

  std::string serialized;
  ASSERT_TRUE(type.SerializeToString(&serialized));

  proto::Type parsed;
  ASSERT_TRUE(parsed.ParseFromString(serialized));
  EXPECT_EQ(parsed.kind(), proto::Type_Kind_INT);
  EXPECT_EQ(parsed.attributes_size(), 0);
}

TEST(TypeAttributesTest, AttributesRoundTripIcebergKeys) {
  // All Iceberg ORC-spec attribute keys must survive the proto round-trip
  // with their string values, exactly mirroring the Apache ORC attribute
  // convention.
  const std::vector<std::pair<std::string, std::string>> kIcebergAttrs = {
      {"iceberg.id", "12"},
      {"iceberg.required", "true"},
      {"iceberg.long-type", "LONG"},
      {"iceberg.timestamp-unit", "NANOS"},
      {"iceberg.binary-type", "UUID"},
      {"iceberg.length", "16"},
      {"iceberg.struct-type", "Variant"},
  };

  auto type = buildTypeWithAttributes(kIcebergAttrs);
  std::string serialized;
  ASSERT_TRUE(type.SerializeToString(&serialized));

  proto::Type parsed;
  ASSERT_TRUE(parsed.ParseFromString(serialized));
  ASSERT_EQ(parsed.attributes_size(), static_cast<int>(kIcebergAttrs.size()));

  for (size_t i = 0; i < kIcebergAttrs.size(); ++i) {
    EXPECT_EQ(parsed.attributes(i).key(), kIcebergAttrs[i].first);
    EXPECT_EQ(parsed.attributes(i).value(), kIcebergAttrs[i].second);
  }
}

TEST(TypeAttributesTest, LegacyBufferIsForwardCompatible) {
  // A buffer produced before this proto change has no attributes field set.
  // The new proto schema must parse it cleanly with an empty attributes
  // list, leaving the other fields intact. Forward-compat invariant for
  // every DWRF file on disk today.
  proto::Type legacy;
  legacy.set_kind(proto::Type_Kind_STRUCT);
  legacy.add_subtypes(1);
  legacy.add_fieldnames("id");

  std::string serialized;
  ASSERT_TRUE(legacy.SerializeToString(&serialized));

  proto::Type parsed;
  ASSERT_TRUE(parsed.ParseFromString(serialized));
  EXPECT_EQ(parsed.kind(), proto::Type_Kind_STRUCT);
  ASSERT_EQ(parsed.subtypes_size(), 1);
  EXPECT_EQ(parsed.subtypes(0), 1u);
  ASSERT_EQ(parsed.fieldnames_size(), 1);
  EXPECT_EQ(parsed.fieldnames(0), "id");
  EXPECT_EQ(parsed.attributes_size(), 0);
}

} // namespace facebook::velox::dwrf::test
