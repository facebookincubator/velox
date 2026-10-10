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

#include <bitset>
#include <limits>
#include <utility>

namespace facebook::nimble {

// serialize() and deserialize() convert between this enum and the on-disk one
// by value.
static_assert(
    static_cast<uint8_t>(StreamTrailerFieldKind::kChecksum32) ==
        static_cast<uint8_t>(serialization::StreamTrailerFieldKind_Checksum32),
    "StreamTrailerFieldKind must mirror the on-disk enum.");

StreamTrailerLayout::StreamTrailerLayout(std::vector<StreamTrailerField> fields)
    : fields_{std::move(fields)} {
  std::bitset<std::numeric_limits<uint8_t>::max() + 1> seenKinds;
  for (const auto& field : fields_) {
    const int kind = static_cast<uint8_t>(field.kind);
    NIMBLE_CHECK_FILE_GT(
        field.size, 0, "Stream trailer field is empty, kind: {}", kind);
    NIMBLE_CHECK_FILE(
        !seenKinds.test(kind), "Stream trailer field kind repeats: {}", kind);
    seenKinds.set(kind);
    if (field.kind == StreamTrailerFieldKind::kChecksum32) {
      NIMBLE_CHECK_FILE_EQ(
          field.size,
          kChecksum32Size,
          "Stream trailer checksum field has the wrong size");
      checksumOffset_ = size_;
    }
    size_ += field.size;
  }
}

StreamTrailerLayout StreamTrailerLayout::defaultLayout() {
  return StreamTrailerLayout{
      {{StreamTrailerFieldKind::kChecksum32, kChecksum32Size}}};
}

FileProperties::FileProperties(
    bool compactRowCountEncoding,
    bool clusterIndexKeyColumnStorageOmitted,
    std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage,
    bool hasStreamChecksums,
    StreamTrailerLayout streamTrailerLayout)
    : compactRowCountEncoding_{compactRowCountEncoding},
      clusterIndexKeyColumnStorageOmitted_{clusterIndexKeyColumnStorageOmitted},
      clusterIndexKeyColumnsWithOmittedStorage_{
          std::move(clusterIndexKeyColumnsWithOmittedStorage)},
      hasStreamChecksums_{hasStreamChecksums},
      streamTrailerLayout_{std::move(streamTrailerLayout)} {
  NIMBLE_CHECK_EQ(
      clusterIndexKeyColumnStorageOmitted_,
      !clusterIndexKeyColumnsWithOmittedStorage_.empty(),
      "clusterIndexKeyColumnStorageOmitted must match clusterIndexKeyColumnsWithOmittedStorage presence");
  NIMBLE_CHECK(
      !(hasStreamChecksums_ &&
        streamTrailerLayout_.checksumOffset().has_value()),
      "Stream checksums cannot be recorded both in stripe groups and in stream trailers");
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

  flatbuffers::Offset<
      flatbuffers::Vector<const serialization::StreamTrailerField*>>
      streamTrailer;
  if (!streamTrailerLayout_.empty()) {
    std::vector<serialization::StreamTrailerField> fields;
    fields.reserve(streamTrailerLayout_.fields().size());
    for (const auto& field : streamTrailerLayout_.fields()) {
      fields.emplace_back(static_cast<uint8_t>(field.kind), field.size);
    }
    streamTrailer = builder.CreateVectorOfStructs(fields);
  }

  builder.Finish(
      serialization::CreateFileProperties(
          builder,
          clusterIndexKeyColumnStorageOmitted_,
          clusterIndexKeyColumnsWithOmittedStorage,
          compactEncodingOffset,
          hasStreamChecksums_,
          streamTrailer));

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

  StreamTrailerLayout streamTrailerLayout;
  if (const auto* fields = serialized->stream_trailer()) {
    // Kinds may not repeat, so a valid layout has at most one field per kind.
    NIMBLE_CHECK_FILE_LE(
        fields->size(),
        std::numeric_limits<uint8_t>::max() + 1,
        "Stream trailer has too many fields");
    std::vector<StreamTrailerField> parsed;
    parsed.reserve(fields->size());
    for (const auto* field : *fields) {
      parsed.push_back(
          {static_cast<StreamTrailerFieldKind>(field->kind()), field->size()});
    }
    streamTrailerLayout = StreamTrailerLayout{std::move(parsed)};
  }
  NIMBLE_CHECK_FILE(
      !(serialized->has_stream_checksums() &&
        streamTrailerLayout.checksumOffset().has_value()),
      "File properties record per-stream checksums both in stripe groups and in stream trailers");

  return FileProperties{
      compactRowCountEncoding,
      serialized->cluster_index_key_column_storage_omitted(),
      std::move(clusterIndexKeyColumnsWithOmittedStorage),
      serialized->has_stream_checksums(),
      std::move(streamTrailerLayout)};
}

} // namespace facebook::nimble
