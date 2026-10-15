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

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/FilePropertiesGenerated.h"

#include "flatbuffers/flatbuffers.h"

#include <algorithm>
#include <bitset>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <utility>

namespace facebook::nimble {

namespace {

// Encodes `features` as a `required_features` bitmap. The bitmap ends at the
// byte that holds the highest feature, and no features give an empty bitmap.
std::vector<uint8_t> toRequiredFeatureBitmap(
    std::span<const RequiredFeature> features) {
  if (features.empty()) {
    return {};
  }
  std::vector<uint8_t> bitmap(
      velox::bits::nbytes(
          static_cast<uint64_t>(std::ranges::max(features)) + 1));
  for (const auto feature : features) {
    velox::bits::setBit(bitmap.data(), static_cast<uint32_t>(feature));
  }
  return bitmap;
}

// Copies `bitmap` into whole 64-bit words, which velox::bits functions read.
// Bytes past the end of `bitmap` are zero.
std::vector<uint64_t> toWords(std::span<const uint8_t> bitmap) {
  std::vector<uint64_t> words(velox::bits::nwords(bitmap.size() * 8));
  if (!bitmap.empty()) {
    std::memcpy(words.data(), bitmap.data(), bitmap.size());
  }
  return words;
}

// Returns kSupportedRequiredFeatures as the words of a `required_features`
// bitmap, built once.
const std::vector<uint64_t>& supportedWords() {
  static const std::vector<uint64_t> words{
      toWords(toRequiredFeatureBitmap(kSupportedRequiredFeatures))};
  return words;
}

// Decodes the `required_features` bitmap, or returns no features when it is
// absent. Throws if the bitmap sets a bit this reader does not support.
std::vector<RequiredFeature> readRequiredFeatures(
    const flatbuffers::Vector<uint8_t>* requiredFeaturesBitmap) {
  if (requiredFeaturesBitmap == nullptr) {
    return {};
  }
  // velox::bits addresses bits with int32_t.
  NIMBLE_CHECK_FILE_LE(
      requiredFeaturesBitmap->size(),
      static_cast<uint32_t>(std::numeric_limits<int32_t>::max() / 8),
      "Required features bitmap is too long");
  const auto numBits = static_cast<int32_t>(requiredFeaturesBitmap->size() * 8);
  const auto requiredFeaturesWords =
      toWords({requiredFeaturesBitmap->data(), requiredFeaturesBitmap->size()});

  // Keeps only the bits of features this reader does not support. Bits past the
  // end of the supported bitmap stay set.
  auto unsupportedFeaturesWords = requiredFeaturesWords;
  const auto& supportedFeaturesWords = supportedWords();
  velox::bits::andWithNegatedBits(
      unsupportedFeaturesWords.data(),
      supportedFeaturesWords.data(),
      0,
      std::min(
          numBits, static_cast<int32_t>(supportedFeaturesWords.size() * 64)));
  if (const auto numUnsupportedFeatures =
          velox::bits::countBits(unsupportedFeaturesWords.data(), 0, numBits);
      numUnsupportedFeatures > 0) {
    NIMBLE_UNSUPPORTED(
        "File requires features this reader does not support. Count: {}, lowest bit: {}",
        numUnsupportedFeatures,
        velox::bits::findFirstBit(unsupportedFeaturesWords.data(), 0, numBits));
  }

  std::vector<RequiredFeature> requiredFeatures;
  velox::bits::forEachSetBit(
      requiredFeaturesWords.data(),
      0,
      numBits,
      [&requiredFeatures](int32_t bit) {
        requiredFeatures.push_back(static_cast<RequiredFeature>(bit));
      });
  return requiredFeatures;
}

} // namespace

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
    StreamTrailerLayout streamTrailerLayout,
    bool hasTopLevelNulls,
    std::vector<RequiredFeature> requiredFeatures)
    : compactRowCountEncoding_{compactRowCountEncoding},
      clusterIndexKeyColumnStorageOmitted_{clusterIndexKeyColumnStorageOmitted},
      clusterIndexKeyColumnsWithOmittedStorage_{
          std::move(clusterIndexKeyColumnsWithOmittedStorage)},
      hasStreamChecksums_{hasStreamChecksums},
      streamTrailerLayout_{std::move(streamTrailerLayout)},
      hasTopLevelNulls_{hasTopLevelNulls},
      requiredFeatures_{std::move(requiredFeatures)} {
  NIMBLE_CHECK_EQ(
      clusterIndexKeyColumnStorageOmitted_,
      !clusterIndexKeyColumnsWithOmittedStorage_.empty(),
      "clusterIndexKeyColumnStorageOmitted must match clusterIndexKeyColumnsWithOmittedStorage presence");
  NIMBLE_CHECK(
      !(hasStreamChecksums_ &&
        streamTrailerLayout_.checksumOffset().has_value()),
      "Stream checksums cannot be recorded both in stripe groups and in stream trailers");
  NIMBLE_CHECK(
      std::ranges::adjacent_find(
          requiredFeatures_, std::ranges::greater_equal{}) ==
          requiredFeatures_.end(),
      "Required features must be in increasing bit order, each once");
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

  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> requiredFeatures;
  if (!requiredFeatures_.empty()) {
    requiredFeatures =
        builder.CreateVector(toRequiredFeatureBitmap(requiredFeatures_));
  }

  builder.Finish(
      serialization::CreateFileProperties(
          builder,
          clusterIndexKeyColumnStorageOmitted_,
          clusterIndexKeyColumnsWithOmittedStorage,
          compactEncodingOffset,
          hasStreamChecksums_,
          streamTrailer,
          hasTopLevelNulls_,
          requiredFeatures));

  return std::string{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

FileProperties FileProperties::deserialize(std::string_view data) {
  const auto* serialized = flatbuffers::GetRoot<serialization::FileProperties>(
      reinterpret_cast<const uint8_t*>(data.data()));

  // A feature this reader lacks may change what the other fields mean or which
  // values they may hold, so refuse such a file before checking them.
  auto requiredFeatures = readRequiredFeatures(serialized->required_features());

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
      std::move(streamTrailerLayout),
      serialized->has_top_level_nulls(),
      std::move(requiredFeatures)};
}

} // namespace facebook::nimble
