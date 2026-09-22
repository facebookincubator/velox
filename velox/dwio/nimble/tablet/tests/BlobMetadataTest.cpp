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
#include "velox/dwio/nimble/tablet/BlobMetadata.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"

#include "velox/common/file/File.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/tablet/BlobMetadataGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"

namespace facebook::nimble {
namespace {

TEST(BlobMetadataTest, roundTripManifest) {
  BlobMetadata metadata{{
      BlobGroup{
          .blobStoreId = 7,
          .stripeId = 4,
          .segments =
              {
                  BlobSegment{
                      .firstRow = 100,
                      .rowCount = 25,
                      .fileOffset = 4096,
                      .compressedSize = 512,
                      .uncompressedSize = 1024,
                      .compressionType = CompressionType::Zstd,
                      .checksumType = ChecksumType::XXH3_64,
                      .checksum = 12345,
                  },
                  BlobSegment{
                      .firstRow = 125,
                      .rowCount = 10,
                      .fileOffset = 8192,
                      .compressedSize = 256,
                      .uncompressedSize = 256,
                      .checksum = 54321,
                  },
              },
          .entryCount = 2,
          .entries = MetadataSection{/*offset=*/2048,
                                     /*size=*/128,
                                     CompressionType::Uncompressed,
                                     /*uncompressedSize=*/128},
      },
  }};

  const auto decoded = BlobMetadata::deserialize(metadata.serialize());
  EXPECT_EQ(decoded.groups(), metadata.groups());
}

TEST(BlobMetadataTest, groupsAreKeyedByStoreAndStripe) {
  const auto group = [](uint32_t blobStoreId, uint32_t stripeId) {
    return BlobGroup{
        .blobStoreId = blobStoreId,
        .stripeId = stripeId,
        .segments = {BlobSegment{.fileOffset = 1024 * (stripeId + 1)}},
        .entryCount = 1,
        .entries = MetadataSection{/*offset=*/stripeId * 64,
                                   /*size=*/32,
                                   CompressionType::Uncompressed,
                                   /*uncompressedSize=*/32},
    };
  };

  const BlobMetadata metadata{{
      group(/*blobStoreId=*/0, /*stripeId=*/0),
      group(/*blobStoreId=*/1, /*stripeId=*/0),
      group(/*blobStoreId=*/0, /*stripeId=*/1),
      group(/*blobStoreId=*/1, /*stripeId=*/1),
  }};

  const auto decoded = BlobMetadata::deserialize(metadata.serialize());
  ASSERT_EQ(decoded.groups().size(), 4);
  EXPECT_EQ(decoded.groups(), metadata.groups());

  // The same store appears once per stripe, and the pair distinguishes them.
  const auto& first = decoded.groups()[0];
  const auto& third = decoded.groups()[2];
  EXPECT_EQ(first.blobStoreId, third.blobStoreId);
  EXPECT_NE(first.stripeId, third.stripeId);
  EXPECT_NE(first.segments[0].fileOffset, third.segments[0].fileOffset);
  EXPECT_NE(first.entries.offset(), third.entries.offset());
}

TEST(BlobMetadataTest, roundTripEmptyManifest) {
  const BlobMetadata metadata{{}};
  const auto decoded = BlobMetadata::deserialize(metadata.serialize());
  EXPECT_TRUE(decoded.groups().empty());
}

TEST(BlobMetadataTest, roundTripGroupWithoutSegments) {
  const BlobMetadata metadata{
      {BlobGroup{.blobStoreId = 3, .stripeId = 1, .entries = {}}}};
  const auto decoded = BlobMetadata::deserialize(metadata.serialize());
  ASSERT_EQ(decoded.groups().size(), 1);
  EXPECT_TRUE(decoded.groups()[0].segments.empty());
  EXPECT_EQ(decoded.groups()[0].entryCount, 0);
}

TEST(BlobMetadataTest, rejectsEmptyOrCorruptManifest) {
  NIMBLE_ASSERT_FILE_THROW(BlobMetadata::deserialize({}), "must not be empty");
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserialize("not a flatbuffer"), "Invalid BlobMetadata");

  // Truncating a valid manifest must be rejected rather than read past the end.
  const BlobMetadata metadata{{BlobGroup{
      .blobStoreId = 1,
      .stripeId = 2,
      .segments = {BlobSegment{.fileOffset = 1024}},
      .entries = {},
  }}};
  const auto serialized = metadata.serialize();
  ASSERT_GT(serialized.size(), 8);
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserialize(
          std::string_view{serialized}.substr(0, serialized.size() / 2)),
      "Invalid BlobMetadata");
}

TEST(BlobMetadataTest, rejectsUnsupportedVersion) {
  // A well formed manifest that only a future reader could interpret.
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(
      serialization::CreateBlobMetadata(
          builder, BlobMetadata::kVersion + 1, /*groups=*/0));
  const std::string_view serialized{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserialize(serialized), "Unsupported blob metadata");
}

TEST(BlobMetadataTest, rejectsUnorderedGroups) {
  const auto group = [](uint32_t blobStoreId, uint32_t stripeId) {
    return BlobGroup{
        .blobStoreId = blobStoreId, .stripeId = stripeId, .entries = {}};
  };

  // Stripes must come in order.
  const BlobMetadata stripesOutOfOrder{{group(0, 1), group(0, 0)}};
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserialize(stripesOutOfOrder.serialize()),
      "stripe 0 store 0 follows stripe 1 store 0");

  // Within a stripe, stores must come in order.
  const BlobMetadata storesOutOfOrder{{group(1, 0), group(0, 0)}};
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserialize(storesOutOfOrder.serialize()),
      "stripe 0 store 0 follows stripe 0 store 1");

  // A repeated group is not strictly increasing either.
  const BlobMetadata duplicated{{group(2, 5), group(2, 5)}};
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserialize(duplicated.serialize()),
      "stripe 5 store 2 follows stripe 5 store 2");
}

TEST(BlobMetadataTest, rejectsEntryCountWithoutEntrySection) {
  const BlobMetadata metadata{{BlobGroup{
      .blobStoreId = 1,
      .segments = {BlobSegment{.fileOffset = 1024}},
      .entryCount = 3,
      .entries = {},
  }}};
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserialize(metadata.serialize()),
      "entry section is empty");
}

TEST(BlobMetadataTest, rejectsEntriesWithoutSegments) {
  const BlobMetadata metadata{{BlobGroup{
      .blobStoreId = 1,
      .segments = {},
      .entryCount = 3,
      .entries = MetadataSection{/*offset=*/64,
                                 /*size=*/32,
                                 CompressionType::Uncompressed,
                                 /*uncompressedSize=*/32},
  }}};
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserialize(metadata.serialize()),
      "has entries but no segments");
}

TEST(BlobMetadataTest, rejectsCorruptOrUnorderedEntries) {
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserializeEntries({}), "must not be empty");
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserializeEntries("not a flatbuffer"),
      "Invalid BlobEntries");

  const std::vector<BlobEntry> outOfOrder{
      BlobEntry{.blobId = 5},
      BlobEntry{.blobId = 2},
  };
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserializeEntries(
          BlobMetadata::serializeEntries(outOfOrder)),
      "not ordered by blob id");

  const std::vector<BlobEntry> duplicated{
      BlobEntry{.blobId = 1},
      BlobEntry{.blobId = 1},
  };
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserializeEntries(
          BlobMetadata::serializeEntries(duplicated)),
      "not ordered by blob id");
}

TEST(BlobMetadataTest, roundTripEntries) {
  const std::vector<BlobEntry> entries{
      BlobEntry{
          .blobId = 0,
          .checksum = 111,
          .segmentId = 0,
          .offsetInSegment = 0,
          .size = 12,
          .uncompressedSize = 12,
      },
      BlobEntry{
          .blobId = 1,
          .checksum = 222,
          .segmentId = 1,
          .offsetInSegment = 0,
          .size = 8,
          .uncompressedSize = 40,
          .compressionType = CompressionType::Zstd,
      },
  };

  EXPECT_EQ(
      BlobMetadata::deserializeEntries(BlobMetadata::serializeEntries(entries)),
      entries);
}

TEST(BlobMetadataTest, rejectsUncompressedEntryWithMismatchedSizes) {
  const std::vector<BlobEntry> entries{
      BlobEntry{.blobId = 0, .size = 8, .uncompressedSize = 40},
  };
  NIMBLE_ASSERT_FILE_THROW(
      BlobMetadata::deserializeEntries(BlobMetadata::serializeEntries(entries)),
      "Uncompressed blob 0 has mismatched sizes: 8 and 40.");
}

TEST(BlobMetadataTest, writesBlobMetadataAsOptionalSection) {
  velox::memory::MemoryManager::testingSetInstance({});
  auto pool = velox::memory::memoryManager()->addLeafPool();
  std::string file;
  velox::InMemoryWriteFile writeFile{&file};

  auto tabletWriter = TabletWriter::create(&writeFile, *pool, {});
  const auto entries = BlobMetadata::serializeEntries({
      BlobEntry{
          .blobId = 0,
          .checksum = 17,
          .segmentId = 0,
          .offsetInSegment = 0,
          .size = 5,
          .uncompressedSize = 5,
      },
  });
  const auto entriesSection = tabletWriter->createMetadataSection(entries);

  const BlobMetadata metadata{{
      BlobGroup{
          .blobStoreId = 1,
          .segments =
              {
                  BlobSegment{
                      .firstRow = 0,
                      .rowCount = 1,
                      .fileOffset = 1024,
                      .compressedSize = 5,
                      .uncompressedSize = 5,
                      .checksum = 17,
                  },
              },
          .entryCount = 1,
          .entries =
              MetadataSection{
                  entriesSection.offset(),
                  entriesSection.size(),
                  entriesSection.compressionType(),
                  entriesSection.uncompressedSize()},
      },
  }};
  tabletWriter->writeOptionalSection(
      std::string(kBlobMetadataSection), metadata.serialize());
  tabletWriter->close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  TabletReader::Options options;
  options.ioOptions = velox::io::ReaderOptions{pool.get()};
  options.ioOptions->setMetadataIoStats(
      std::make_shared<velox::io::IoStatistics>());
  options.preloadOptionalSections.emplace_back(kBlobMetadataSection);

  auto tablet = TabletReader::create(readFile, pool.get(), options);
  ASSERT_TRUE(tablet->hasOptionalSection(std::string(kBlobMetadataSection)));
  auto section = tablet->loadOptionalSection(
      std::string(kBlobMetadataSection), /*keepCache=*/true);
  ASSERT_TRUE(section.has_value());
  const auto decoded = BlobMetadata::deserialize(section->content());
  EXPECT_EQ(decoded.groups(), metadata.groups());
}

} // namespace
} // namespace facebook::nimble
