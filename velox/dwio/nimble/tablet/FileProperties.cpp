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

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/FilePropertiesGenerated.h"

#include "flatbuffers/flatbuffers.h"

#include <limits>
#include <utility>

namespace facebook::nimble {

namespace {
// Written when the file carries no stream checksums, so the type byte never
// reads as a real ChecksumType for a reader that skips has_stream_checksums.
constexpr uint8_t kInvalidStreamChecksumType =
    std::numeric_limits<uint8_t>::max();
} // namespace

FileProperties::FileProperties(
    bool compactRowCountEncoding,
    bool clusterIndexKeyColumnStorageOmitted,
    std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage,
    std::optional<uint8_t> streamChecksumType)
    : compactRowCountEncoding_{compactRowCountEncoding},
      clusterIndexKeyColumnStorageOmitted_{clusterIndexKeyColumnStorageOmitted},
      clusterIndexKeyColumnsWithOmittedStorage_{
          std::move(clusterIndexKeyColumnsWithOmittedStorage)},
      streamChecksumType_{streamChecksumType} {
  NIMBLE_CHECK_EQ(
      clusterIndexKeyColumnStorageOmitted_,
      !clusterIndexKeyColumnsWithOmittedStorage_.empty(),
      "clusterIndexKeyColumnStorageOmitted must match clusterIndexKeyColumnsWithOmittedStorage presence");
}

std::string FileProperties::serialize() const {
  flatbuffers::FlatBufferBuilder builder;

  flatbuffers::Offset<serialization::CompactEncoding> compactEncodingOffset;
  if (compactRowCountEncoding_) {
    compactEncodingOffset =
        serialization::CreateCompactEncoding(builder, compactRowCountEncoding_);
  }

  auto clusterIndexKeyColumnsWithOmittedStorage =
      builder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
          clusterIndexKeyColumnsWithOmittedStorage_.size(),
          [this, &builder](size_t i) {
            return builder.CreateString(
                clusterIndexKeyColumnsWithOmittedStorage_[i]);
          });

  builder.Finish(
      serialization::CreateFileProperties(
          builder,
          clusterIndexKeyColumnStorageOmitted_,
          clusterIndexKeyColumnsWithOmittedStorage,
          compactEncodingOffset,
          streamChecksumType_.has_value(),
          streamChecksumType_.value_or(kInvalidStreamChecksumType)));

  return std::string{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

FileProperties FileProperties::deserialize(std::string_view data) {
  const auto* serialized = flatbuffers::GetRoot<serialization::FileProperties>(
      reinterpret_cast<const uint8_t*>(data.data()));

  bool compactRowCountEncoding{false};
  if (const auto* serializedCompactEncoding = serialized->compact_encoding()) {
    compactRowCountEncoding = serializedCompactEncoding->row_count();
  }

  std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage;
  if (const auto* columns =
          serialized->cluster_index_key_columns_with_omitted_storage()) {
    clusterIndexKeyColumnsWithOmittedStorage.reserve(columns->size());
    for (const auto* column : *columns) {
      clusterIndexKeyColumnsWithOmittedStorage.push_back(column->str());
    }
  }
  NIMBLE_CHECK_EQ(
      serialized->cluster_index_key_column_storage_omitted(),
      !clusterIndexKeyColumnsWithOmittedStorage.empty(),
      "cluster_index_key_column_storage_omitted must match cluster_index_key_columns_with_omitted_storage presence");

  // Deliberately not validated here. initProperties() runs for every reader,
  // including full scans that never look at stream checksums, so rejecting an
  // unrecognized type at this point would make the first file written with a
  // future checksum type unreadable by every already-deployed binary. Readers
  // that actually verify resolve the type through ChecksumFactory, which
  // rejects what it cannot construct.
  std::optional<uint8_t> streamChecksumType;
  if (serialized->has_stream_checksums()) {
    streamChecksumType = serialized->stream_checksum_type();
  }

  return FileProperties{
      compactRowCountEncoding,
      serialized->cluster_index_key_column_storage_omitted(),
      std::move(clusterIndexKeyColumnsWithOmittedStorage),
      streamChecksumType};
}

} // namespace facebook::nimble
