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

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/SchemaGenerated.h"

namespace facebook::nimble::test {

namespace fbs = ::facebook::nimble::serialization;

// ---------------------------------------------------------------------------
// Iceberg V3 interop: per-SchemaNode `attributes` field on the NIMBLE schema
// flatbuffer.
//
// NIMBLE files are recorded as `FileFormat.ORC` in Iceberg manifests, so they
// must encode column ids and type-disambiguation metadata using the same
// string-keyed attribute convention the Iceberg spec mandates for ORC
// (Appendix A: Format-specific Requirements -> ORC). These tests exercise the
// raw flatbuffer wire format -- the C++ SchemaNode plumbing lands as a
// follow-up.
// ---------------------------------------------------------------------------

namespace {

// Build a flatbuffer Schema with a single SchemaNode and attributes.
flatbuffers::DetachedBuffer buildSchemaWithAttributes(
    const std::vector<std::pair<std::string, std::string>>& attrs) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<fbs::StringPair>> attrOffsets;
  attrOffsets.reserve(attrs.size());
  for (const auto& [k, v] : attrs) {
    attrOffsets.push_back(
        fbs::CreateStringPair(
            builder, builder.CreateString(k), builder.CreateString(v)));
  }

  auto name = builder.CreateString("id");
  auto attributes = builder.CreateVector(attrOffsets);

  fbs::SchemaNodeBuilder nodeBuilder(builder);
  nodeBuilder.add_kind(fbs::Kind_Int64);
  nodeBuilder.add_children(0);
  nodeBuilder.add_name(name);
  nodeBuilder.add_offset(0);
  nodeBuilder.add_attributes(attributes);
  auto node = nodeBuilder.Finish();

  auto nodes = builder.CreateVector(
      std::vector<flatbuffers::Offset<fbs::SchemaNode>>{node});
  auto schema = fbs::CreateSchema(builder, nodes);
  builder.Finish(schema);
  return builder.Release();
}

flatbuffers::DetachedBuffer buildSchemaWithoutAttributes() {
  flatbuffers::FlatBufferBuilder builder;
  auto name = builder.CreateString("id");
  // Note: deliberately NOT calling add_attributes(...). Simulates a writer on
  // the pre-attributes flatbuffer schema.
  fbs::SchemaNodeBuilder nodeBuilder(builder);
  nodeBuilder.add_kind(fbs::Kind_Int64);
  nodeBuilder.add_children(0);
  nodeBuilder.add_name(name);
  nodeBuilder.add_offset(0);
  auto node = nodeBuilder.Finish();
  auto nodes = builder.CreateVector(
      std::vector<flatbuffers::Offset<fbs::SchemaNode>>{node});
  auto schema = fbs::CreateSchema(builder, nodes);
  builder.Finish(schema);
  return builder.Release();
}

} // namespace

TEST(NimbleSchemaAttributesTest, attributesAbsentByDefault) {
  // A SchemaNode built without any attributes (the existing wire format for
  // every NIMBLE file written today) must round-trip and surface as either a
  // nullptr or empty vector through the new flatbuffer schema. This protects
  // the no-op upgrade path for existing files.
  auto buf = buildSchemaWithoutAttributes();
  auto schema = fbs::GetSchema(buf.data());
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->nodes(), nullptr);
  ASSERT_EQ(schema->nodes()->size(), 1u);

  const auto* node = schema->nodes()->Get(0);
  EXPECT_EQ(node->name()->str(), "id");
  EXPECT_EQ(node->kind(), fbs::Kind_Int64);
  EXPECT_EQ(node->offset(), 0u);
  EXPECT_EQ(node->children(), 0u);
  // The flatbuffer accessor for an unset trailing field returns nullptr.
  EXPECT_EQ(node->attributes(), nullptr);
}

TEST(NimbleSchemaAttributesTest, attributesRoundTripIcebergV3Keys) {
  // Verify every Iceberg V3 attribute key the spec recognizes survives the
  // flatbuffer round-trip with string values, exactly mirroring the Apache
  // ORC attribute convention.
  const std::vector<std::pair<std::string, std::string>> kIcebergAttrs = {
      {"iceberg.id", "12"},
      {"iceberg.required", "true"},
      {"iceberg.long-type", "LONG"},
      {"iceberg.timestamp-unit", "NANOS"},
      {"iceberg.binary-type", "UUID"},
      {"iceberg.length", "16"},
      {"iceberg.struct-type", "Variant"},
  };

  auto buf = buildSchemaWithAttributes(kIcebergAttrs);
  auto schema = fbs::GetSchema(buf.data());
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->nodes(), nullptr);
  ASSERT_EQ(schema->nodes()->size(), 1u);

  const auto* node = schema->nodes()->Get(0);
  ASSERT_NE(node->attributes(), nullptr);

  std::vector<std::pair<std::string, std::string>> roundTripped;
  roundTripped.reserve(node->attributes()->size());
  for (const auto* pair : *node->attributes()) {
    ASSERT_NE(pair->key(), nullptr);
    ASSERT_NE(pair->value(), nullptr);
    roundTripped.emplace_back(pair->key()->str(), pair->value()->str());
  }
  EXPECT_EQ(roundTripped, kIcebergAttrs);
}

TEST(NimbleSchemaAttributesTest, legacyBufferIsForwardCompatible) {
  // A buffer produced before this fbs change has no `attributes` vtable slot.
  // The new schema's flatbuffer Verifier must accept that buffer (no
  // required fields broken), and the missing attributes accessor must return
  // nullptr. This is the forward-compat invariant for every NIMBLE file on
  // disk today.
  auto buf = buildSchemaWithoutAttributes();
  flatbuffers::Verifier verifier(buf.data(), buf.size());
  ASSERT_TRUE(fbs::VerifySchemaBuffer(verifier));

  auto schema = fbs::GetSchema(buf.data());
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->nodes(), nullptr);
  ASSERT_EQ(schema->nodes()->size(), 1u);
  EXPECT_EQ(schema->nodes()->Get(0)->attributes(), nullptr);
}

TEST(NimbleSchemaAttributesTest, emptyAttributeListIsDistinctFromAbsent) {
  // An attribute list explicitly created but empty surfaces as a non-null
  // vector of size 0. Useful so writers can signal "I emit attributes
  // capability, this node just happens to have none" vs absent. The accessor
  // distinction matters for downstream readers that branch on presence.
  auto buf = buildSchemaWithAttributes({});
  auto schema = fbs::GetSchema(buf.data());
  ASSERT_NE(schema, nullptr);
  const auto* node = schema->nodes()->Get(0);
  ASSERT_NE(node->attributes(), nullptr);
  EXPECT_EQ(node->attributes()->size(), 0u);
}

} // namespace facebook::nimble::test
