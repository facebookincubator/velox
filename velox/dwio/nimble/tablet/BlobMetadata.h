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
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble {

/// Describes one embedded run of blob payload bytes. A segment belongs to the
/// BlobGroup holding it, and its index in that group's segment list is the id
/// BlobEntry::segmentId refers to.
struct BlobSegment {
  /// Range of rows whose payloads this segment holds, within the group's
  /// stripe. Rows and blob ids are different spaces: a null row consumes a row
  /// but no blob id, so rowCount is not the number of entries.
  uint64_t firstRow{0};
  uint32_t rowCount{0};
  /// Start of the payload bytes in the tablet, and their extent. An entry's
  /// offsetInSegment is relative to fileOffset.
  uint64_t fileOffset{0};
  uint32_t compressedSize{0};
  uint32_t uncompressedSize{0};
  /// Applies to the segment as a whole, so a reader restores it before slicing
  /// any payload out of it. Independent of BlobEntry::compressionType, which
  /// sits inside this one: a segment may carry both, either, or neither.
  CompressionType compressionType{CompressionType::Uncompressed};
  /// Algorithm for 'checksum' below, and for the per-payload checksum a reader
  /// verifies from BlobEntry.
  ChecksumType checksumType{ChecksumType::XXH3_64};
  uint64_t checksum{0};

  bool operator==(const BlobSegment& other) const;
};

/// Maps one blob id to its bytes inside one of the enclosing group's segments.
struct BlobEntry {
  /// Unique within the enclosing group, dense and assigned in append order.
  uint64_t blobId{0};
  /// Covers the bytes as they sit in the segment, so a reader can verify them
  /// before handing a compressed blob to the decompressor.
  uint64_t checksum{0};
  /// Index into the enclosing group's segment list, not a file-wide id.
  uint32_t segmentId{0};
  /// Location and extent within the segment's uncompressed bytes, not within
  /// the file.
  uint32_t offsetInSegment{0};
  uint32_t size{0};
  /// What those bytes expand to. Equals 'size' unless this blob was compressed
  /// on its own.
  uint32_t uncompressedSize{0};
  /// Applies to this blob alone, leaving the rest of the segment readable
  /// without expanding it. Sits inside the segment's own compression, so a
  /// reader restores the segment before expanding this.
  CompressionType compressionType{CompressionType::Uncompressed};

  bool operator==(const BlobEntry& other) const;
};

/// Everything needed to read one blob store within one stripe: the segments
/// holding its payload bytes, and the location of the entry list that addresses
/// them. A group is identified by (blobStoreId, stripeId), and both the
/// segments and the entry list cover only that stripe.
///
/// Segments are held inline because a group has only a handful. The entry list
/// is not: its size is proportional to the number of blobs, so 'entries' is a
/// file offset and the entries are read separately with deserializeEntries.
struct BlobGroup {
  uint32_t blobStoreId{0};
  uint32_t stripeId{0};
  std::vector<BlobSegment> segments;
  /// Number of entries the section below holds. Recorded here so a reader can
  /// size its allocation, and detect a truncated entry list, before parsing it.
  uint32_t entryCount{0};
  MetadataSection entries;

  bool operator==(const BlobGroup& other) const;
};

/// Blob metadata manifest stored in the "blob.metadata" optional section.
class BlobMetadata {
 public:
  static constexpr uint32_t kVersion{1};

  explicit BlobMetadata(std::vector<BlobGroup> groups);

  /// Groups ordered by (stripeId, blobStoreId), so a stripe's groups are
  /// adjacent and can be located by binary search.
  const std::vector<BlobGroup>& groups() const {
    return groups_;
  }

  /// Serializes the root blob metadata manifest.
  std::string serialize() const;

  /// Deserializes the root blob metadata manifest.
  static BlobMetadata deserialize(std::string_view data);

  /// Serializes the entry list referenced by BlobGroup::entries.
  static std::string serializeEntries(const std::vector<BlobEntry>& entries);

  /// Deserializes the entry list referenced by BlobGroup::entries. Verifies
  /// that the entries are ordered by blob id, but cannot check them against
  /// their group, which it never sees: callers must confirm the count matches
  /// BlobGroup::entryCount and that every segmentId indexes that group's
  /// segment list.
  static std::vector<BlobEntry> deserializeEntries(std::string_view data);

 private:
  std::vector<BlobGroup> groups_;
};

} // namespace facebook::nimble
