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

#include "velox/dwio/dwrf/common/Common.h"

using namespace ::testing;

namespace facebook::velox::dwrf {

class CommonTest : public ::testing::Test {
 protected:
  proto::Stream createDwrfStream(
      proto::Stream::Kind kind,
      uint32_t node,
      uint32_t sequence,
      uint32_t column) {
    proto::Stream stream;
    stream.set_kind(kind);
    stream.set_node(node);
    stream.set_sequence(sequence);
    stream.set_column(column);
    return stream;
  }

  proto::orc::Stream createOrcStream(
      proto::orc::Stream::Kind kind,
      uint32_t column) {
    proto::orc::Stream stream;
    stream.set_kind(kind);
    stream.set_column(column);
    return stream;
  }
};

TEST_F(
    CommonTest,
    DwrfStreamIdentifierGetFormat_WithStreamTypes_GetCorrectFormat) {
  EXPECT_EQ(
      DwrfStreamIdentifier(createDwrfStream({}, {}, {}, {})).format(),
      DwrfFormat::kDwrf);

  EXPECT_EQ(
      DwrfStreamIdentifier(createOrcStream({}, {})).format(), DwrfFormat::kOrc);
}

TEST_F(CommonTest, DwrfStreamIdentifier_WithDwrfStream_GetCorrectInfo) {
  auto streamId = DwrfStreamIdentifier(
      createDwrfStream(proto::Stream::Kind::Stream_Kind_DATA, 1, 2, 3));

  EXPECT_EQ(streamId.format(), DwrfFormat::kDwrf);
  EXPECT_EQ(streamId.kind(), StreamKind::StreamKind_DATA);
  EXPECT_EQ(streamId.encodingKey().node(), 1);
  EXPECT_EQ(streamId.encodingKey().sequence(), 2);
  EXPECT_EQ(streamId.column(), 3);
}

TEST_F(CommonTest, DwrfStreamIdentifier_WithOrcStream_GetCorrectInfo) {
  auto streamId = DwrfStreamIdentifier(
      createOrcStream(proto::orc::Stream::Kind::Stream_Kind_DATA, 1));

  // ORC doesn't use node directly, column is used as node
  EXPECT_EQ(streamId.format(), DwrfFormat::kOrc);
  EXPECT_EQ(streamId.kind(), StreamKind::StreamKindOrc_DATA);
  EXPECT_EQ(streamId.encodingKey().node(), 1);
  EXPECT_EQ(streamId.encodingKey().sequence(), 0);
  EXPECT_EQ(streamId.column(), dwio::common::MAX_UINT32);
}

TEST_F(
    CommonTest,
    DwrfStreamIdentifierGetKind_WithAllStreamKinds_GetCorrectConversion) {
  // DWRF
  for (auto [dwrfStreamKind, veloxStreamKind] :
       std::vector<std::tuple<proto::Stream_Kind, StreamKind>>{
           // clang-format off
           {proto::Stream_Kind_PRESENT, StreamKind::StreamKind_PRESENT},
           {proto::Stream_Kind_DATA, StreamKind::StreamKind_DATA},
           {proto::Stream_Kind_LENGTH, StreamKind::StreamKind_LENGTH},
           {proto::Stream_Kind_DICTIONARY_DATA, StreamKind::StreamKind_DICTIONARY_DATA},
           {proto::Stream_Kind_DICTIONARY_COUNT, StreamKind::StreamKind_DICTIONARY_COUNT},
           {proto::Stream_Kind_NANO_DATA, StreamKind::StreamKind_NANO_DATA},
           {proto::Stream_Kind_ROW_INDEX, StreamKind::StreamKind_ROW_INDEX},
           {proto::Stream_Kind_IN_DICTIONARY, StreamKind::StreamKind_IN_DICTIONARY}, {proto::Stream_Kind_STRIDE_DICTIONARY, StreamKind::StreamKind_STRIDE_DICTIONARY},
           {proto::Stream_Kind_STRIDE_DICTIONARY_LENGTH, StreamKind::StreamKind_STRIDE_DICTIONARY_LENGTH},
           {proto::Stream_Kind_BLOOM_FILTER_UTF8, StreamKind::StreamKind_BLOOM_FILTER_UTF8},
           {proto::Stream_Kind_IN_MAP, StreamKind::StreamKind_IN_MAP},
           // clang-format on
       }) {
    EXPECT_EQ(
        DwrfStreamIdentifier(createDwrfStream(dwrfStreamKind, {}, {}, {}))
            .kind(),
        veloxStreamKind);
  }

  for (auto [orcStreamKind, veloxStreamKind] :
       std::vector<std::tuple<proto::orc::Stream_Kind, StreamKind>>{
           // clang-format off
           {proto::orc::Stream_Kind_PRESENT, StreamKind::StreamKindOrc_PRESENT},
           {proto::orc::Stream_Kind_DATA, StreamKind::StreamKindOrc_DATA},
           {proto::orc::Stream_Kind_LENGTH, StreamKind::StreamKindOrc_LENGTH},
           {proto::orc::Stream_Kind_DICTIONARY_DATA, StreamKind::StreamKindOrc_DICTIONARY_DATA},
           {proto::orc::Stream_Kind_DICTIONARY_COUNT, StreamKind::StreamKindOrc_DICTIONARY_COUNT},
           {proto::orc::Stream_Kind_SECONDARY, StreamKind::StreamKindOrc_SECONDARY},
           {proto::orc::Stream_Kind_ROW_INDEX, StreamKind::StreamKindOrc_ROW_INDEX},
           {proto::orc::Stream_Kind_BLOOM_FILTER, StreamKind::StreamKindOrc_BLOOM_FILTER},
           {proto::orc::Stream_Kind_BLOOM_FILTER_UTF8, StreamKind::StreamKindOrc_BLOOM_FILTER_UTF8},
           {proto::orc::Stream_Kind_ENCRYPTED_INDEX, StreamKind::StreamKindOrc_ENCRYPTED_INDEX},
           {proto::orc::Stream_Kind_ENCRYPTED_DATA, StreamKind::StreamKindOrc_ENCRYPTED_DATA},
           {proto::orc::Stream_Kind_STRIPE_STATISTICS, StreamKind::StreamKindOrc_STRIPE_STATISTICS},
           {proto::orc::Stream_Kind_FILE_STATISTICS, StreamKind::StreamKindOrc_FILE_STATISTICS},
           // clang-format on
       }) {
    EXPECT_EQ(
        DwrfStreamIdentifier(createOrcStream(orcStreamKind, {})).kind(),
        veloxStreamKind);
  }
}

TEST_F(
    CommonTest,
    EncodingKeyGetKindFor_WithAllStreamKinds_GetCorrectConversion) {
  // DWRF
  for (auto [dwrfStreamKind, veloxStreamKind] :
       std::vector<std::tuple<proto::Stream_Kind, StreamKind>>{
           // clang-format off
           {proto::Stream_Kind_PRESENT, StreamKind::StreamKind_PRESENT},
           {proto::Stream_Kind_DATA, StreamKind::StreamKind_DATA},
           {proto::Stream_Kind_LENGTH, StreamKind::StreamKind_LENGTH},
           {proto::Stream_Kind_DICTIONARY_DATA, StreamKind::StreamKind_DICTIONARY_DATA},
           {proto::Stream_Kind_DICTIONARY_COUNT, StreamKind::StreamKind_DICTIONARY_COUNT},
           {proto::Stream_Kind_NANO_DATA, StreamKind::StreamKind_NANO_DATA},
           {proto::Stream_Kind_ROW_INDEX, StreamKind::StreamKind_ROW_INDEX},
           {proto::Stream_Kind_IN_DICTIONARY, StreamKind::StreamKind_IN_DICTIONARY}, {proto::Stream_Kind_STRIDE_DICTIONARY, StreamKind::StreamKind_STRIDE_DICTIONARY},
           {proto::Stream_Kind_STRIDE_DICTIONARY_LENGTH, StreamKind::StreamKind_STRIDE_DICTIONARY_LENGTH},
           {proto::Stream_Kind_BLOOM_FILTER_UTF8, StreamKind::StreamKind_BLOOM_FILTER_UTF8},
           {proto::Stream_Kind_IN_MAP, StreamKind::StreamKind_IN_MAP},
           // clang-format on
       }) {
    EncodingKey encodingKey;
    auto stream = encodingKey.forKind(dwrfStreamKind);

    EXPECT_EQ(stream.kind(), veloxStreamKind);
  }

  for (auto [orcStreamKind, veloxStreamKind] :
       std::vector<std::tuple<proto::orc::Stream_Kind, StreamKind>>{
           // clang-format off
           {proto::orc::Stream_Kind_PRESENT, StreamKind::StreamKindOrc_PRESENT},
           {proto::orc::Stream_Kind_DATA, StreamKind::StreamKindOrc_DATA},
           {proto::orc::Stream_Kind_LENGTH, StreamKind::StreamKindOrc_LENGTH},
           {proto::orc::Stream_Kind_DICTIONARY_DATA, StreamKind::StreamKindOrc_DICTIONARY_DATA},
           {proto::orc::Stream_Kind_DICTIONARY_COUNT, StreamKind::StreamKindOrc_DICTIONARY_COUNT},
           {proto::orc::Stream_Kind_SECONDARY, StreamKind::StreamKindOrc_SECONDARY},
           {proto::orc::Stream_Kind_ROW_INDEX, StreamKind::StreamKindOrc_ROW_INDEX},
           {proto::orc::Stream_Kind_BLOOM_FILTER, StreamKind::StreamKindOrc_BLOOM_FILTER},
           {proto::orc::Stream_Kind_BLOOM_FILTER_UTF8, StreamKind::StreamKindOrc_BLOOM_FILTER_UTF8},
           {proto::orc::Stream_Kind_ENCRYPTED_INDEX, StreamKind::StreamKindOrc_ENCRYPTED_INDEX},
           {proto::orc::Stream_Kind_ENCRYPTED_DATA, StreamKind::StreamKindOrc_ENCRYPTED_DATA},
           {proto::orc::Stream_Kind_STRIPE_STATISTICS, StreamKind::StreamKindOrc_STRIPE_STATISTICS},
           {proto::orc::Stream_Kind_FILE_STATISTICS, StreamKind::StreamKindOrc_FILE_STATISTICS},
           // clang-format on
       }) {
    EncodingKey encodingKey;
    auto stream = encodingKey.forKind(orcStreamKind);

    EXPECT_EQ(stream.kind(), veloxStreamKind);
  }
}
// ---------------------------------------------------------------------------
// Iceberg V3 interop: per-Type `attributes` field on the DWRF `Type` proto.
//
// The Iceberg spec (Appendix A: Format-specific Requirements -> ORC) requires
// ORC-family Iceberg files to encode their column ids and type-disambiguation
// metadata as string-keyed attributes on each schema node. DWRF and NIMBLE
// files are reported as `FileFormat.ORC` in Iceberg manifests, so they must
// carry the same attribute bag to be Iceberg-spec compliant. The tests below
// pin the wire-format invariants this depends on.
// ---------------------------------------------------------------------------

TEST_F(CommonTest, TypeAttributesFieldNumberMatchesApacheOrc) {
  // Apache ORC's Type.attributes lives at proto field number 7. Keeping our
  // slot number aligned is what lets a spec-compliant external reader -- one
  // that only knows Apache ORC -- consume our DWRF files (manifest-tagged as
  // ORC) and find `iceberg.id` where it expects to. If this assertion ever
  // changes, the property-bag contract with the Iceberg spec is broken.
  EXPECT_EQ(proto::Type::kAttributesFieldNumber, 7);
  EXPECT_EQ(proto::StringPair::kKeyFieldNumber, 1);
  EXPECT_EQ(proto::StringPair::kValueFieldNumber, 2);
}

TEST_F(CommonTest, TypeAttributesEmptyByDefaultForBackwardCompat) {
  // Existing DWRF files do not set `attributes`. Verify that a freshly
  // constructed Type behaves as if the field is absent (empty list, no
  // attributes considered present). This protects the no-op upgrade path for
  // every DWRF file written before this proto bump.
  proto::Type type;
  type.set_kind(proto::Type_Kind_LONG);
  EXPECT_EQ(type.attributes_size(), 0);

  std::string bytes;
  ASSERT_TRUE(type.SerializeToString(&bytes));

  proto::Type parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes));
  EXPECT_EQ(parsed.attributes_size(), 0);
  EXPECT_EQ(parsed.kind(), proto::Type_Kind_LONG);
}

TEST_F(CommonTest, TypeAttributesRoundTripIcebergV3Keys) {
  // All Iceberg V3 attribute keys round-trip through the proto with string
  // values, matching the encoding documented in Iceberg's ORC mapping and
  // produced by Presto-Iceberg's TypeConverter (see ORC_ICEBERG_ID_KEY /
  // ORC_ICEBERG_REQUIRED_KEY in presto-iceberg's TypeConverter.java).
  proto::Type type;
  type.set_kind(proto::Type_Kind_LONG);

  const std::vector<std::pair<std::string, std::string>> kIcebergAttrs = {
      {"iceberg.id", "12"},
      {"iceberg.required", "true"},
      {"iceberg.long-type", "LONG"},
      {"iceberg.timestamp-unit", "NANOS"},
      {"iceberg.binary-type", "UUID"},
      {"iceberg.length", "16"},
      {"iceberg.struct-type", "Variant"},
  };
  for (const auto& [k, v] : kIcebergAttrs) {
    auto* attr = type.add_attributes();
    attr->set_key(k);
    attr->set_value(v);
  }

  std::string bytes;
  ASSERT_TRUE(type.SerializeToString(&bytes));

  proto::Type parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes));
  ASSERT_EQ(parsed.attributes_size(), static_cast<int>(kIcebergAttrs.size()));
  for (size_t i = 0; i < kIcebergAttrs.size(); ++i) {
    EXPECT_EQ(parsed.attributes(i).key(), kIcebergAttrs[i].first);
    EXPECT_EQ(parsed.attributes(i).value(), kIcebergAttrs[i].second);
  }
}

TEST_F(CommonTest, TypeAttributesUnknownFieldFromOldReaderIsPreserved) {
  // proto2 forward compatibility: bytes carrying `attributes` produced by a
  // new writer must round-trip cleanly through a Type containing only
  // recognized fields. The unknown-field set keeps the bytes intact so a
  // pass-through reader does not silently strip Iceberg metadata.
  proto::Type writerType;
  writerType.set_kind(proto::Type_Kind_LONG);
  auto* attr = writerType.add_attributes();
  attr->set_key("iceberg.id");
  attr->set_value("42");

  std::string bytes;
  ASSERT_TRUE(writerType.SerializeToString(&bytes));

  // Parse, then re-serialize. The repeated `attributes` field must be
  // emitted again byte-for-byte. (We compare normalized re-serialization
  // rather than the raw byte sequence because protobuf does not guarantee
  // canonical encoding ordering across versions.)
  proto::Type roundTripped;
  ASSERT_TRUE(roundTripped.ParseFromString(bytes));
  ASSERT_EQ(roundTripped.attributes_size(), 1);
  EXPECT_EQ(roundTripped.attributes(0).key(), "iceberg.id");
  EXPECT_EQ(roundTripped.attributes(0).value(), "42");

  std::string reSerialized;
  ASSERT_TRUE(roundTripped.SerializeToString(&reSerialized));
  proto::Type secondParse;
  ASSERT_TRUE(secondParse.ParseFromString(reSerialized));
  ASSERT_EQ(secondParse.attributes_size(), 1);
  EXPECT_EQ(secondParse.attributes(0).key(), "iceberg.id");
  EXPECT_EQ(secondParse.attributes(0).value(), "42");
}

} // namespace facebook::velox::dwrf
