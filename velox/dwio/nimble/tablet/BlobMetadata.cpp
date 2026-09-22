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

#include <optional>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/BlobMetadataGenerated.h"

namespace facebook::nimble {
namespace {

std::string_view asView(const flatbuffers::FlatBufferBuilder& builder) {
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

flatbuffers::Offset<serialization::MetadataSection> createMetadataSection(
    flatbuffers::FlatBufferBuilder& builder,
    const MetadataSection& section) {
  return serialization::CreateMetadataSection(
      builder,
      section.offset(),
      section.size(),
      static_cast<serialization::CompressionType>(section.compressionType()),
      section.uncompressedSize().value_or(section.size()));
}

MetadataSection toMetadataSection(
    const serialization::MetadataSection* section) {
  if (section == nullptr) {
    return {};
  }
  return MetadataSection{
      section->offset(),
      section->size(),
      static_cast<CompressionType>(section->compression_type()),
      section->uncompressed_size()};
}

ChecksumType toChecksumType(uint8_t checksumType) {
  NIMBLE_CHECK_FILE_EQ(
      checksumType,
      static_cast<uint8_t>(ChecksumType::XXH3_64),
      "Unsupported blob checksum type: {}",
      checksumType);
  return ChecksumType::XXH3_64;
}

flatbuffers::Offset<serialization::BlobSegment> createSegment(
    flatbuffers::FlatBufferBuilder& builder,
    const BlobSegment& segment) {
  return serialization::CreateBlobSegment(
      builder,
      segment.firstRow,
      segment.rowCount,
      segment.fileOffset,
      segment.compressedSize,
      segment.uncompressedSize,
      static_cast<serialization::CompressionType>(segment.compressionType),
      static_cast<uint8_t>(segment.checksumType),
      segment.checksum);
}

BlobSegment toSegment(const serialization::BlobSegment* segment) {
  return BlobSegment{
      .firstRow = segment->first_row(),
      .rowCount = segment->row_count(),
      .fileOffset = segment->file_offset(),
      .compressedSize = segment->compressed_size(),
      .uncompressedSize = segment->uncompressed_size(),
      .compressionType =
          static_cast<CompressionType>(segment->compression_type()),
      .checksumType = toChecksumType(segment->checksum_type()),
      .checksum = segment->checksum(),
  };
}

} // namespace

bool BlobSegment::operator==(const BlobSegment& other) const {
  return firstRow == other.firstRow && rowCount == other.rowCount &&
      fileOffset == other.fileOffset &&
      compressedSize == other.compressedSize &&
      uncompressedSize == other.uncompressedSize &&
      compressionType == other.compressionType &&
      checksumType == other.checksumType && checksum == other.checksum;
}

bool BlobEntry::operator==(const BlobEntry& other) const {
  return blobId == other.blobId && checksum == other.checksum &&
      segmentId == other.segmentId &&
      offsetInSegment == other.offsetInSegment && size == other.size &&
      uncompressedSize == other.uncompressedSize &&
      compressionType == other.compressionType;
}

bool BlobGroup::operator==(const BlobGroup& other) const {
  return blobStoreId == other.blobStoreId && stripeId == other.stripeId &&
      segments == other.segments && entryCount == other.entryCount &&
      entries.offset() == other.entries.offset() &&
      entries.size() == other.entries.size() &&
      entries.compressionType() == other.entries.compressionType() &&
      entries.uncompressedSize() == other.entries.uncompressedSize();
}

BlobMetadata::BlobMetadata(std::vector<BlobGroup> groups)
    : groups_{std::move(groups)} {}

std::string BlobMetadata::serialize() const {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<serialization::BlobGroup>> groupOffsets;
  groupOffsets.reserve(groups_.size());
  for (const auto& group : groups_) {
    std::vector<flatbuffers::Offset<serialization::BlobSegment>> segmentOffsets;
    segmentOffsets.reserve(group.segments.size());
    for (const auto& segment : group.segments) {
      segmentOffsets.push_back(createSegment(builder, segment));
    }
    groupOffsets.push_back(
        serialization::CreateBlobGroup(
            builder,
            group.blobStoreId,
            group.stripeId,
            builder.CreateVector(segmentOffsets),
            group.entryCount,
            createMetadataSection(builder, group.entries)));
  }

  builder.Finish(
      serialization::CreateBlobMetadata(
          builder, kVersion, builder.CreateVector(groupOffsets)));
  return std::string{asView(builder)};
}

BlobMetadata BlobMetadata::deserialize(std::string_view data) {
  NIMBLE_CHECK_FILE(!data.empty(), "Blob metadata must not be empty.");
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(data.data()), data.size());
  NIMBLE_CHECK_FILE(
      serialization::VerifyBlobMetadataBuffer(verifier),
      "Invalid BlobMetadata FlatBuffer.");

  const auto* metadata = flatbuffers::GetRoot<serialization::BlobMetadata>(
      reinterpret_cast<const uint8_t*>(data.data()));
  NIMBLE_CHECK_FILE_NOT_NULL(metadata, "Blob metadata is null.");
  NIMBLE_CHECK_FILE_EQ(
      metadata->version(), kVersion, "Unsupported blob metadata version.");

  std::vector<BlobGroup> groups;
  std::optional<uint64_t> previousKey;
  if (const auto* serializedGroups = metadata->groups()) {
    groups.reserve(serializedGroups->size());
    for (const auto* group : *serializedGroups) {
      NIMBLE_CHECK_FILE_NOT_NULL(group, "Blob group is null.");

      // Groups are ordered by (stripeId, blobStoreId), which is the order a
      // writer produces them in: stripes are written in sequence, and each
      // stripe emits its stores. Requiring it lets a reader binary search for
      // a stripe's groups, and rules out duplicates on the way.
      const auto key = (static_cast<uint64_t>(group->stripe_id()) << 32) |
          group->blob_store_id();
      NIMBLE_CHECK_FILE(
          !previousKey.has_value() || key > *previousKey,
          "Blob groups are not ordered by stripe then store: "
          "stripe {} store {} follows stripe {} store {}.",
          group->stripe_id(),
          group->blob_store_id(),
          static_cast<uint32_t>(*previousKey >> 32),
          static_cast<uint32_t>(*previousKey));
      previousKey = key;

      std::vector<BlobSegment> segments;
      if (const auto* serializedSegments = group->segments()) {
        segments.reserve(serializedSegments->size());
        for (const auto* segment : *serializedSegments) {
          NIMBLE_CHECK_FILE_NOT_NULL(segment, "Blob segment is null.");
          segments.push_back(toSegment(segment));
        }
      }

      auto entries = toMetadataSection(group->entries());
      // Entries live outside the manifest and are parsed separately, so a group
      // can only be checked for self-consistency: claiming entries obliges it
      // to point at a section holding them and at segments to hold their bytes.
      // Whether the entries themselves agree with the group is up to the caller
      // that loads them.
      if (group->entry_count() > 0) {
        NIMBLE_CHECK_FILE_GT(
            entries.size(),
            0,
            "Blob group for store {} stripe {} claims entries but its entry "
            "section is empty.",
            group->blob_store_id(),
            group->stripe_id());
        NIMBLE_CHECK_FILE_GT(
            segments.size(),
            0,
            "Blob group for store {} stripe {} has entries but no segments.",
            group->blob_store_id(),
            group->stripe_id());
      }

      groups.push_back(
          BlobGroup{
              .blobStoreId = group->blob_store_id(),
              .stripeId = group->stripe_id(),
              .segments = std::move(segments),
              .entryCount = group->entry_count(),
              .entries = entries,
          });
    }
  }

  return BlobMetadata{std::move(groups)};
}

std::string BlobMetadata::serializeEntries(
    const std::vector<BlobEntry>& entries) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<serialization::BlobEntry>> entryOffsets;
  entryOffsets.reserve(entries.size());
  for (const auto& entry : entries) {
    entryOffsets.push_back(
        serialization::CreateBlobEntry(
            builder,
            entry.blobId,
            entry.checksum,
            entry.segmentId,
            entry.offsetInSegment,
            entry.size,
            entry.uncompressedSize,
            static_cast<serialization::CompressionType>(
                entry.compressionType)));
  }
  builder.Finish(
      serialization::CreateBlobEntries(
          builder, builder.CreateVector(entryOffsets)));
  return std::string{asView(builder)};
}

std::vector<BlobEntry> BlobMetadata::deserializeEntries(std::string_view data) {
  NIMBLE_CHECK_FILE(!data.empty(), "Blob entries must not be empty.");
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(data.data()), data.size());
  // BlobEntries is not the schema's root_type, so verify the buffer against it
  // explicitly instead of through a generated VerifyBlobEntriesBuffer.
  NIMBLE_CHECK_FILE(
      verifier.VerifyBuffer<serialization::BlobEntries>(nullptr),
      "Invalid BlobEntries FlatBuffer.");

  const auto* root = flatbuffers::GetRoot<serialization::BlobEntries>(
      reinterpret_cast<const uint8_t*>(data.data()));
  NIMBLE_CHECK_FILE_NOT_NULL(root, "Blob entries is null.");

  std::vector<BlobEntry> entries;
  if (const auto* serializedEntries = root->entries()) {
    entries.reserve(serializedEntries->size());
    for (const auto* entry : *serializedEntries) {
      NIMBLE_CHECK_FILE_NOT_NULL(entry, "Blob entry is null.");
      // Blob ids are assigned densely in append order, so an entry list is
      // strictly increasing. Readers locate an entry by relying on that.
      NIMBLE_CHECK_FILE(
          entries.empty() || entry->blob_id() > entries.back().blobId,
          "Blob entries are not ordered by blob id.");
      // An uncompressed blob is copied out of its segment at 'size' bytes into
      // a buffer sized from 'uncompressed_size', so the two disagreeing would
      // overrun it.
      NIMBLE_CHECK_FILE(
          static_cast<CompressionType>(entry->compression_type()) !=
                  CompressionType::Uncompressed ||
              entry->size() == entry->uncompressed_size(),
          "Uncompressed blob {} has mismatched sizes: {} and {}.",
          entry->blob_id(),
          entry->size(),
          entry->uncompressed_size());
      entries.push_back(
          BlobEntry{
              .blobId = entry->blob_id(),
              .checksum = entry->checksum(),
              .segmentId = entry->segment_id(),
              .offsetInSegment = entry->offset_in_segment(),
              .size = entry->size(),
              .uncompressedSize = entry->uncompressed_size(),
              .compressionType =
                  static_cast<CompressionType>(entry->compression_type()),
          });
    }
  }
  return entries;
}

} // namespace facebook::nimble
