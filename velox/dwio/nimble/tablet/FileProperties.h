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

#include "velox/dwio/nimble/common/Types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::nimble {

/// What one field of the stream trailer holds. Mirrors the on-disk
/// StreamTrailerFieldKind, so values are append-only. A file may carry kinds
/// this enum does not name; readers skip them by size.
enum class StreamTrailerFieldKind : uint8_t {
  /// The stream's checksum: Checksum::computeChecksum32() of its bytes, stored
  /// little-endian.
  kChecksum32 = 0,
};

/// One field of the stream trailer.
struct StreamTrailerField {
  StreamTrailerFieldKind kind;
  uint8_t size;

  bool operator==(const StreamTrailerField&) const = default;
};

/// Layout of the trailer that follows every non-empty stream in the stripe
/// data, declared once per file. The trailer sits after the stream's bytes,
/// outside its recorded size, so readers that do not look for it read the
/// stream exactly as if it were absent.
class StreamTrailerLayout {
 public:
  /// Bytes of a kChecksum32 field.
  static constexpr uint8_t kChecksum32Size{4};

  /// No trailer.
  StreamTrailerLayout() = default;

  /// The trailer writers emit when stream checksums are enabled: the 32-bit
  /// checksum alone.
  static StreamTrailerLayout defaultLayout();

  const std::vector<StreamTrailerField>& fields() const {
    return fields_;
  }

  bool empty() const {
    return fields_.empty();
  }

  /// Bytes the trailer adds after each non-empty stream.
  uint32_t size() const {
    return size_;
  }

  /// Offset of the kChecksum32 field within the trailer, or nullopt when the
  /// trailer carries no checksum.
  std::optional<uint32_t> checksumOffset() const {
    return checksumOffset_;
  }

 private:
  // Only FileProperties::deserialize() builds layouts from arbitrary fields,
  // the ones a file declares; writers emit defaultLayout().
  friend class FileProperties;

  // `fields` in on-disk order. Throws if a field is empty, a kind repeats, or
  // a kind this reader knows has the wrong size; the file format forbids all
  // three. Kinds it does not know only add their size.
  explicit StreamTrailerLayout(std::vector<StreamTrailerField> fields);

  std::vector<StreamTrailerField> fields_;
  uint32_t size_{0};
  std::optional<uint32_t> checksumOffset_;
};

/// File-level properties that readers need before loading other optional
/// metadata.
class FileProperties {
 public:
  /// A file records per-stream checksums in at most one place: stripe-group
  /// arrays (`hasStreamChecksums`, legacy files only) or a stream trailer that
  /// carries kChecksum32.
  FileProperties(
      bool compactRowCountEncoding,
      bool clusterIndexKeyColumnStorageOmitted,
      std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage,
      bool hasStreamChecksums = false,
      StreamTrailerLayout streamTrailerLayout = {});

  /// Returns whether encoded stream row counts use compact varint encoding.
  bool compactRowCountEncoding() const {
    return compactRowCountEncoding_;
  }

  /// Returns whether cluster index key columns were omitted from data storage.
  bool clusterIndexKeyColumnStorageOmitted() const {
    return clusterIndexKeyColumnStorageOmitted_;
  }

  /// Returns cluster index key columns whose normal data streams are absent.
  const std::vector<std::string>& clusterIndexKeyColumnsWithOmittedStorage()
      const {
    return clusterIndexKeyColumnsWithOmittedStorage_;
  }

  /// Returns whether stripe groups carry per-stream checksum arrays. Only
  /// legacy files do; current writers put each stream's checksum in its
  /// trailer instead (see streamTrailerLayout()). The algorithm is the file's
  /// ChecksumType, from the postscript.
  bool hasStreamChecksums() const {
    return hasStreamChecksums_;
  }

  /// Returns the layout of the trailer that follows every non-empty stream;
  /// empty when streams have none.
  const StreamTrailerLayout& streamTrailerLayout() const {
    return streamTrailerLayout_;
  }

  /// Serializes file properties into the `columnar.properties` optional
  /// section.
  std::string serialize() const;

  /// Deserializes the `columnar.properties` optional section.
  static FileProperties deserialize(std::string_view data);

 private:
  bool compactRowCountEncoding_{false};
  bool clusterIndexKeyColumnStorageOmitted_{false};
  std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage_;
  bool hasStreamChecksums_{false};
  StreamTrailerLayout streamTrailerLayout_;
};

} // namespace facebook::nimble
