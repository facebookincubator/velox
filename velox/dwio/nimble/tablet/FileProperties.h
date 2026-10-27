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

#include <array>
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
  /// The stream's checksum: Checksum::computeChecksum32() of its bytes, with
  /// the file's ChecksumType from the postscript, stored little-endian.
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

/// A feature a reader must support to read a file correctly. The value is the
/// feature's bit in the file's `required_features`. Give each new feature the
/// next value after the last one, written out explicitly. Files on disk depend
/// on these values, so never renumber, reuse or remove one. TabletReader checks
/// these bits after it parses the postscript, the footer, the stripes section
/// and possibly the first stripe group, so a bit cannot gate changes to those.
enum class RequiredFeature : uint32_t {};

/// File-level properties that readers need before loading other optional
/// metadata.
class FileProperties {
 public:
  /// Features this reader supports, in bit order. A new feature is appended
  /// here together with its reader support. Entries are never removed, so every
  /// later reader supports them too.
  static constexpr std::array<RequiredFeature, 0> kSupportedRequiredFeatures{};

  /// `requiredFeatures` must be in increasing bit order, each once.
  FileProperties(
      bool compactRowCountEncoding,
      bool clusterIndexKeyColumnStorageOmitted,
      std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage,
      StreamTrailerLayout streamTrailerLayout = {},
      bool hasTopLevelNulls = false,
      std::vector<RequiredFeature> requiredFeatures = {});

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

  /// Returns the layout of the trailer that follows every non-empty stream;
  /// empty when streams have none.
  const StreamTrailerLayout& streamTrailerLayout() const {
    return streamTrailerLayout_;
  }

  /// Returns whether any top-level row in the file is null.
  bool hasTopLevelNulls() const {
    return hasTopLevelNulls_;
  }

  /// Returns the features a reader must support to read the file, each once,
  /// in increasing bit order.
  const std::vector<RequiredFeature>& requiredFeatures() const {
    return requiredFeatures_;
  }

  /// Serializes file properties into the `columnar.properties` optional
  /// section.
  std::string serialize() const;

  /// Deserializes the `columnar.properties` optional section. Throws a
  /// NotSupported user error if the file requires a feature that is not in
  /// kSupportedRequiredFeatures. That check runs before any other field is
  /// validated, so such a file never fails as corrupt first.
  static FileProperties deserialize(std::string_view data);

 private:
  bool compactRowCountEncoding_{false};
  bool clusterIndexKeyColumnStorageOmitted_{false};
  std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage_;
  StreamTrailerLayout streamTrailerLayout_;
  bool hasTopLevelNulls_{false};
  std::vector<RequiredFeature> requiredFeatures_;
};

} // namespace facebook::nimble
