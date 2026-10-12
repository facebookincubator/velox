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
#include "velox/dwio/nimble/tablet/FileProperties.h"

#include <gtest/gtest.h>
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/tablet/FilePropertiesGenerated.h"

#include "flatbuffers/flatbuffers.h"

#include <limits>
#include <string>
#include <vector>

namespace facebook::nimble {
namespace {

TEST(FilePropertiesTest, basic) {
  FileProperties features{/*compactRowCountEncoding=*/false,
                          /*clusterIndexKeyColumnStorageOmitted=*/false,
                          /*clusterIndexKeyColumnsWithOmittedStorage=*/{}};

  const auto decoded = FileProperties::deserialize(features.serialize());
  EXPECT_FALSE(decoded.compactRowCountEncoding());
  EXPECT_FALSE(decoded.clusterIndexKeyColumnStorageOmitted());
  EXPECT_TRUE(decoded.clusterIndexKeyColumnsWithOmittedStorage().empty());
}

TEST(FilePropertiesTest, roundTripKeyColumnsWithOmittedStorage) {
  FileProperties features{
      /*compactRowCountEncoding=*/false,
      /*clusterIndexKeyColumnStorageOmitted=*/true,
      /*clusterIndexKeyColumnsWithOmittedStorage=*/{"key0", "key1"}};

  const auto decoded = FileProperties::deserialize(features.serialize());
  EXPECT_FALSE(decoded.compactRowCountEncoding());
  EXPECT_TRUE(decoded.clusterIndexKeyColumnStorageOmitted());
  EXPECT_EQ(
      decoded.clusterIndexKeyColumnsWithOmittedStorage(),
      (std::vector<std::string>{"key0", "key1"}));
}

TEST(FilePropertiesTest, roundTripCompactRowCountEncoding) {
  FileProperties enabled{/*compactRowCountEncoding=*/true,
                         /*clusterIndexKeyColumnStorageOmitted=*/false,
                         /*clusterIndexKeyColumnsWithOmittedStorage=*/{}};

  auto decoded = FileProperties::deserialize(enabled.serialize());
  EXPECT_TRUE(decoded.compactRowCountEncoding());

  FileProperties disabled{/*compactRowCountEncoding=*/false,
                          /*clusterIndexKeyColumnStorageOmitted=*/false,
                          /*clusterIndexKeyColumnsWithOmittedStorage=*/{}};

  decoded = FileProperties::deserialize(disabled.serialize());
  EXPECT_FALSE(decoded.compactRowCountEncoding());
}

TEST(FilePropertiesTest, rejectsConstructedOmittedStorageWithoutColumns) {
  NIMBLE_ASSERT_THROW(
      FileProperties(
          /*compactRowCountEncoding=*/false,
          /*clusterIndexKeyColumnStorageOmitted=*/true,
          /*clusterIndexKeyColumnsWithOmittedStorage=*/{}),
      "clusterIndexKeyColumnStorageOmitted must match clusterIndexKeyColumnsWithOmittedStorage presence");

  NIMBLE_ASSERT_THROW(
      FileProperties(
          /*compactRowCountEncoding=*/false,
          /*clusterIndexKeyColumnStorageOmitted=*/false,
          /*clusterIndexKeyColumnsWithOmittedStorage=*/{"key0"}),
      "clusterIndexKeyColumnStorageOmitted must match clusterIndexKeyColumnsWithOmittedStorage presence");
}

TEST(FilePropertiesTest, rejectsOmittedStorageWithoutColumns) {
  auto serialized =
      [](bool clusterIndexKeyColumnStorageOmitted,
         std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage) {
        flatbuffers::FlatBufferBuilder builder;
        auto columns =
            builder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
                clusterIndexKeyColumnsWithOmittedStorage.size(),
                [&builder,
                 &clusterIndexKeyColumnsWithOmittedStorage](size_t i) {
                  return builder.CreateString(
                      clusterIndexKeyColumnsWithOmittedStorage[i]);
                });
        builder.Finish(
            serialization::CreateFileProperties(
                builder,
                clusterIndexKeyColumnStorageOmitted,
                columns,
                /*compact_encoding=*/0));

        return std::string{
            reinterpret_cast<const char*>(builder.GetBufferPointer()),
            builder.GetSize()};
      };

  struct Case {
    bool clusterIndexKeyColumnStorageOmitted;
    std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage;
  };
  const std::vector<Case> cases{{true, {}}, {false, {"key0"}}};
  for (const auto& testCase : cases) {
    SCOPED_TRACE(
        testing::Message() << "clusterIndexKeyColumnStorageOmitted="
                           << testCase.clusterIndexKeyColumnStorageOmitted);
    NIMBLE_ASSERT_THROW(
        FileProperties::deserialize(serialized(
            testCase.clusterIndexKeyColumnStorageOmitted,
            testCase.clusterIndexKeyColumnsWithOmittedStorage)),
        "cluster_index_key_column_storage_omitted must match cluster_index_key_columns_with_omitted_storage presence");
  }
}

TEST(FilePropertiesTest, roundTripStreamChecksums) {
  FileProperties absent{/*compactRowCountEncoding=*/false,
                        /*clusterIndexKeyColumnStorageOmitted=*/false,
                        /*clusterIndexKeyColumnsWithOmittedStorage=*/{}};
  EXPECT_FALSE(
      FileProperties::deserialize(absent.serialize()).hasStreamChecksums());

  FileProperties present{/*compactRowCountEncoding=*/false,
                         /*clusterIndexKeyColumnStorageOmitted=*/false,
                         /*clusterIndexKeyColumnsWithOmittedStorage=*/{},
                         /*hasStreamChecksums=*/true};
  EXPECT_TRUE(
      FileProperties::deserialize(present.serialize()).hasStreamChecksums());
}

// A file whose properties section exists only to record stream checksums must
// still round trip; the writer skips the section entirely when nothing is set.
TEST(FilePropertiesTest, streamChecksumsCoexistWithOtherProperties) {
  FileProperties properties{
      /*compactRowCountEncoding=*/true,
      /*clusterIndexKeyColumnStorageOmitted=*/true,
      /*clusterIndexKeyColumnsWithOmittedStorage=*/{"key0"},
      /*hasStreamChecksums=*/true};

  const auto decoded = FileProperties::deserialize(properties.serialize());
  EXPECT_TRUE(decoded.compactRowCountEncoding());
  EXPECT_TRUE(decoded.clusterIndexKeyColumnStorageOmitted());
  EXPECT_TRUE(decoded.hasStreamChecksums());
}

// Serializes a properties section by hand, so a test can write layouts that
// FileProperties itself refuses to build.
std::string serializeStreamTrailer(
    const std::vector<serialization::StreamTrailerField>& fields,
    bool hasStreamChecksums = false) {
  flatbuffers::FlatBufferBuilder builder;
  const auto streamTrailer = builder.CreateVectorOfStructs(fields);
  builder.Finish(
      serialization::CreateFileProperties(
          builder,
          /*cluster_index_key_column_storage_omitted=*/false,
          /*cluster_index_key_columns_with_omitted_storage=*/0,
          /*compact_encoding=*/0,
          hasStreamChecksums,
          streamTrailer));
  return std::string{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

serialization::StreamTrailerField onDiskField(uint8_t kind, uint8_t size) {
  return serialization::StreamTrailerField{kind, size};
}

TEST(FilePropertiesTest, roundTripStreamTrailer) {
  FileProperties properties{/*compactRowCountEncoding=*/false,
                            /*clusterIndexKeyColumnStorageOmitted=*/false,
                            /*clusterIndexKeyColumnsWithOmittedStorage=*/{},
                            /*hasStreamChecksums=*/false,
                            StreamTrailerLayout::defaultLayout()};
  const auto serialized = properties.serialize();

  const auto decoded = FileProperties::deserialize(serialized);
  EXPECT_FALSE(decoded.hasStreamChecksums());
  EXPECT_EQ(
      decoded.streamTrailerLayout().fields(),
      (std::vector<StreamTrailerField>{
          {StreamTrailerFieldKind::kChecksum32, 4}}));
  EXPECT_EQ(decoded.streamTrailerLayout().size(), 4);
  EXPECT_EQ(decoded.streamTrailerLayout().checksumOffset(), 0);

  // Readers that predate stream trailers see no checksums at all.
  const auto* root =
      flatbuffers::GetRoot<serialization::FileProperties>(serialized.data());
  EXPECT_FALSE(root->has_stream_checksums());
}

// Files without a trailer, including every file written before trailers
// existed, read as an empty layout, and so does an empty field list.
TEST(FilePropertiesTest, absentStreamTrailerIsEmpty) {
  FileProperties properties{/*compactRowCountEncoding=*/true,
                            /*clusterIndexKeyColumnStorageOmitted=*/false,
                            /*clusterIndexKeyColumnsWithOmittedStorage=*/{}};
  for (const auto& serialized :
       {properties.serialize(), serializeStreamTrailer({})}) {
    const auto decoded = FileProperties::deserialize(serialized);
    EXPECT_TRUE(decoded.streamTrailerLayout().empty());
    EXPECT_EQ(decoded.streamTrailerLayout().size(), 0);
    EXPECT_EQ(decoded.streamTrailerLayout().checksumOffset(), std::nullopt);
  }
}

// A reader skips kinds it does not know by their declared size, so it still
// finds the checksum wherever a newer writer placed it.
TEST(FilePropertiesTest, streamTrailerSkipsUnknownKinds) {
  const auto decoded = FileProperties::deserialize(serializeStreamTrailer(
      {onDiskField(/*kind=*/7, /*size=*/3),
       onDiskField(/*kind=*/0, /*size=*/4),
       onDiskField(/*kind=*/9, /*size=*/2)}));
  EXPECT_EQ(
      decoded.streamTrailerLayout().fields(),
      (std::vector<StreamTrailerField>{
          {static_cast<StreamTrailerFieldKind>(7), 3},
          {StreamTrailerFieldKind::kChecksum32, 4},
          {static_cast<StreamTrailerFieldKind>(9), 2}}));
  EXPECT_EQ(decoded.streamTrailerLayout().size(), 9);
  EXPECT_EQ(decoded.streamTrailerLayout().checksumOffset(), 3);
}

// A trailer without a checksum leaves the file without one: readers skip the
// trailer and read the streams unverified.
TEST(FilePropertiesTest, streamTrailerWithoutChecksum) {
  const auto decoded = FileProperties::deserialize(
      serializeStreamTrailer({onDiskField(/*kind=*/5, /*size=*/4)}));
  EXPECT_EQ(decoded.streamTrailerLayout().size(), 4);
  EXPECT_EQ(decoded.streamTrailerLayout().checksumOffset(), std::nullopt);
  EXPECT_FALSE(decoded.hasStreamChecksums());
}

TEST(FilePropertiesTest, rejectsMalformedStreamTrailer) {
  struct TestCase {
    std::vector<serialization::StreamTrailerField> fields;
    std::string error;
  };

  // One field more than there are kinds.
  std::vector<serialization::StreamTrailerField> tooManyFields;
  for (uint32_t i = 0; i <= std::numeric_limits<uint8_t>::max() + 1; ++i) {
    tooManyFields.push_back(onDiskField(static_cast<uint8_t>(i), /*size=*/1));
  }

  // The format forbids empty fields and repeated kinds whether or not the
  // reader knows the kind.
  const std::vector<TestCase> testCases{
      {tooManyFields, "Stream trailer has too many fields"},
      {{onDiskField(/*kind=*/0, /*size=*/3)},
       "Stream trailer checksum field has the wrong size"},
      {{onDiskField(/*kind=*/0, /*size=*/0)},
       "Stream trailer field is empty, kind: 0"},
      {{onDiskField(/*kind=*/5, /*size=*/0)},
       "Stream trailer field is empty, kind: 5"},
      {{onDiskField(/*kind=*/0, /*size=*/4),
        onDiskField(/*kind=*/0, /*size=*/4)},
       "Stream trailer field kind repeats: 0"},
      {{onDiskField(/*kind=*/5, /*size=*/1),
        onDiskField(/*kind=*/0, /*size=*/4),
        onDiskField(/*kind=*/5, /*size=*/2)},
       "Stream trailer field kind repeats: 5"},
  };
  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.error);
    NIMBLE_ASSERT_FILE_THROW(
        FileProperties::deserialize(serializeStreamTrailer(testCase.fields)),
        testCase.error);
  }
}

// No writer records checksums in both places, so a file claiming both is
// malformed rather than verified half one way and half the other.
TEST(FilePropertiesTest, rejectsChecksumsInStripeGroupsAndTrailers) {
  NIMBLE_ASSERT_FILE_THROW(
      FileProperties::deserialize(serializeStreamTrailer(
          {onDiskField(/*kind=*/0, /*size=*/4)},
          /*hasStreamChecksums=*/true)),
      "File properties record per-stream checksums both in stripe groups and in stream trailers");
  NIMBLE_ASSERT_THROW(
      (FileProperties{/*compactRowCountEncoding=*/false,
                      /*clusterIndexKeyColumnStorageOmitted=*/false,
                      /*clusterIndexKeyColumnsWithOmittedStorage=*/{},
                      /*hasStreamChecksums=*/true,
                      StreamTrailerLayout::defaultLayout()}),
      "Stream checksums cannot be recorded both in stripe groups and in stream trailers");

  // A trailer that carries no checksum does not conflict with the arrays.
  const auto decoded = FileProperties::deserialize(serializeStreamTrailer(
      {onDiskField(/*kind=*/5, /*size=*/4)}, /*hasStreamChecksums=*/true));
  EXPECT_TRUE(decoded.hasStreamChecksums());
  EXPECT_EQ(decoded.streamTrailerLayout().checksumOffset(), std::nullopt);
}

} // namespace
} // namespace facebook::nimble
