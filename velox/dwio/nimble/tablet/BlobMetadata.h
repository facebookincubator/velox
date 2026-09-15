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
  uint64_t firstRow{0};
  uint32_t rowCount{0};
  uint64_t fileOffset{0};
  uint32_t compressedSize{0};
  uint32_t uncompressedSize{0};
  CompressionType compressionType{CompressionType::Uncompressed};
  ChecksumType checksumType{ChecksumType::XXH3_64};
  uint64_t checksum{0};

  bool operator==(const BlobSegment& other) const;
};

/// Maps one blob id to its bytes inside one of the enclosing group's segments.
struct BlobEntry {
  uint64_t blobId{0};
  uint64_t checksum{0};
  uint32_t segmentId{0};
  uint32_t offsetInSegment{0};
  uint32_t compressedSize{0};
  uint32_t uncompressedSize{0};

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
  uint32_t entryCount{0};
  MetadataSection entries;

  bool operator==(const BlobGroup& other) const;
};

/// Blob metadata manifest stored in the "blob.metadata" optional section.
class BlobMetadata {
 public:
  static constexpr uint32_t kVersion{1};

  explicit BlobMetadata(std::vector<BlobGroup> groups);

  const std::vector<BlobGroup>& groups() const {
    return groups_;
  }

  /// Serializes the root blob metadata manifest.
  std::string serialize() const;

  /// Deserializes the root blob metadata manifest.
  static BlobMetadata deserialize(std::string_view data);

  /// Serializes the entry list referenced by BlobGroup::entries.
  static std::string serializeEntries(const std::vector<BlobEntry>& entries);

  /// Deserializes the entry list referenced by BlobGroup::entries.
  static std::vector<BlobEntry> deserializeEntries(std::string_view data);

 private:
  std::vector<BlobGroup> groups_;
};

} // namespace facebook::nimble
