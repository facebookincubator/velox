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

#include "velox/dwio/nimble/encodings/SharedDictionaryCatalog.h"

#include <utility>

#include "flatbuffers/flatbuffers.h"
#include "folly/container/F14Set.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryGenerated.h"

namespace facebook::nimble {
namespace {

using SerializedDictionaryReferences = flatbuffers::Vector<
    flatbuffers::Offset<serialization::SharedDictionaryReference>>;
using CatalogIndex = folly::F14FastMap<uint32_t, size_t>;

auto serializeReferences(
    flatbuffers::FlatBufferBuilder& builder,
    std::span<const SharedDictionaryReference> references,
    folly::F14FastSet<uint32_t>& streamIds) {
  return builder.CreateVector<
      flatbuffers::Offset<serialization::SharedDictionaryReference>>(
      references.size(), [&](size_t i) {
        const auto& reference = references[i];
        NIMBLE_CHECK(
            streamIds.insert(reference.valueStreamId).second,
            "Duplicate shared dictionary value stream {}.",
            reference.valueStreamId);
        NIMBLE_CHECK_NE(
            reference.dictionaryId,
            kInvalidSharedDictionaryId,
            "Shared dictionary reference requires a valid dictionary id.");
        return serialization::CreateSharedDictionaryReference(
            builder,
            reference.valueStreamId,
            reference.dictionaryId,
            static_cast<uint8_t>(reference.dataType));
      });
}

void validateFileDictionaries(
    std::span<const SharedDictionaryReference> references,
    std::span<const FileDictionary> fileDictionaries) {
  CatalogIndex fileDictionaryIndices;
  fileDictionaryIndices.reserve(fileDictionaries.size());
  for (size_t i = 0; i < fileDictionaries.size(); ++i) {
    const auto& fileDictionary = fileDictionaries[i];
    NIMBLE_CHECK_GT(
        fileDictionary.length,
        0,
        "File dictionary must contain a non-empty alphabet.");
    NIMBLE_CHECK(
        fileDictionaryIndices.emplace(fileDictionary.dictionaryId, i).second,
        "Duplicate file shared dictionary ID {}.",
        fileDictionary.dictionaryId);
  }

  for (const auto& reference : references) {
    const auto fileDictionaryEntry =
        fileDictionaryIndices.find(reference.dictionaryId);
    NIMBLE_CHECK(
        fileDictionaryEntry != fileDictionaryIndices.end(),
        "File shared dictionary {} does not exist.",
        reference.dictionaryId);
    NIMBLE_CHECK_EQ(
        fileDictionaries[fileDictionaryEntry->second].dataType,
        reference.dataType,
        "File shared dictionary {} has an inconsistent type.",
        reference.dictionaryId);
  }
}

void parseReferences(
    const SerializedDictionaryReferences* serializedReferences,
    std::vector<SharedDictionaryReference>& references,
    CatalogIndex& referenceIndices,
    folly::F14FastSet<uint32_t>& streamIds) {
  if (serializedReferences == nullptr) {
    return;
  }
  references.reserve(serializedReferences->size());
  referenceIndices.reserve(serializedReferences->size());
  for (const auto* serializedReference : *serializedReferences) {
    NIMBLE_CHECK_FILE_NOT_NULL(
        serializedReference,
        "Shared dictionary catalog contains an invalid reference.");
    SharedDictionaryReference reference{
        .valueStreamId = serializedReference->value_stream_id(),
        .dictionaryId = serializedReference->dictionary_id(),
        .dataType = static_cast<DataType>(serializedReference->data_type())};
    NIMBLE_CHECK_FILE_NE(
        reference.dictionaryId,
        kInvalidSharedDictionaryId,
        "Shared dictionary value stream {} has an invalid dictionary ID.",
        reference.valueStreamId);
    NIMBLE_CHECK_FILE(
        streamIds.insert(reference.valueStreamId).second,
        "Duplicate shared dictionary value stream {}.",
        reference.valueStreamId);
    referenceIndices.emplace(reference.valueStreamId, references.size());
    references.push_back(reference);
  }
}

template <typename Entry>
const Entry* findEntry(
    const std::vector<Entry>& entries,
    const CatalogIndex& indices,
    uint32_t id) {
  const auto entry = indices.find(id);
  return entry == indices.end() ? nullptr : &entries[entry->second];
}

} // namespace

std::string SharedDictionaryCatalog::serialize(
    std::span<const SharedDictionaryReference> stripeDictionaryReferences,
    std::span<const SharedDictionaryReference> fileDictionaryReferences,
    std::span<const SharedDictionaryReference> externalDictionaryReferences,
    std::span<const FileDictionary> fileDictionaries) {
  NIMBLE_CHECK(
      !stripeDictionaryReferences.empty() ||
          !fileDictionaryReferences.empty() ||
          !externalDictionaryReferences.empty() || !fileDictionaries.empty(),
      "Shared dictionary catalog must contain at least one entry.");
  flatbuffers::FlatBufferBuilder builder;
  folly::F14FastSet<uint32_t> streamIds;
  const auto stripeDictionaryReferencesOffset =
      serializeReferences(builder, stripeDictionaryReferences, streamIds);
  const auto fileDictionaryReferencesOffset =
      serializeReferences(builder, fileDictionaryReferences, streamIds);
  const auto externalDictionaryReferencesOffset =
      serializeReferences(builder, externalDictionaryReferences, streamIds);
  validateFileDictionaries(fileDictionaryReferences, fileDictionaries);

  const auto fileDictionariesOffset =
      builder.CreateVector<flatbuffers::Offset<serialization::FileDictionary>>(
          fileDictionaries.size(), [&](size_t i) {
            const auto& fileDictionary = fileDictionaries[i];
            return serialization::CreateFileDictionary(
                builder,
                fileDictionary.dictionaryId,
                static_cast<uint8_t>(fileDictionary.dataType),
                fileDictionary.offset,
                fileDictionary.length);
          });
  const auto root = serialization::CreateSharedDictionaryCatalog(
      builder,
      stripeDictionaryReferencesOffset,
      fileDictionaryReferencesOffset,
      externalDictionaryReferencesOffset,
      fileDictionariesOffset);
  serialization::FinishSharedDictionaryCatalogBuffer(builder, root);
  return std::string{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

SharedDictionaryCatalog SharedDictionaryCatalog::deserialize(
    std::string_view catalog) {
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(catalog.data()), catalog.size()};
  NIMBLE_CHECK_FILE(
      serialization::VerifySharedDictionaryCatalogBuffer(verifier),
      "Invalid shared dictionary catalog.");
  const auto* serializedCatalog =
      serialization::GetSharedDictionaryCatalog(catalog.data());

  SharedDictionaryCatalog parsed;
  folly::F14FastSet<uint32_t> streamIds;
  parseReferences(
      serializedCatalog->stripe_dictionary_references(),
      parsed.stripeDictionaryReferences_,
      parsed.stripeDictionaryReferenceIndices_,
      streamIds);
  parseReferences(
      serializedCatalog->file_dictionary_references(),
      parsed.fileDictionaryReferences_,
      parsed.fileDictionaryReferenceIndices_,
      streamIds);
  parseReferences(
      serializedCatalog->external_dictionary_references(),
      parsed.externalDictionaryReferences_,
      parsed.externalDictionaryReferenceIndices_,
      streamIds);

  if (const auto* serializedFileDictionaries =
          serializedCatalog->file_dictionaries()) {
    parsed.fileDictionaries_.reserve(serializedFileDictionaries->size());
    parsed.fileDictionaryIndices_.reserve(serializedFileDictionaries->size());
    for (const auto* serializedFileDictionary : *serializedFileDictionaries) {
      NIMBLE_CHECK_FILE_NOT_NULL(
          serializedFileDictionary,
          "Shared dictionary catalog contains an invalid file dictionary.");
      NIMBLE_CHECK_FILE_GT(
          serializedFileDictionary->length(),
          0,
          "File dictionary {} has no alphabet.",
          serializedFileDictionary->dictionary_id());
      FileDictionary fileDictionary{
          .dictionaryId = serializedFileDictionary->dictionary_id(),
          .dataType =
              static_cast<DataType>(serializedFileDictionary->data_type()),
          .offset = serializedFileDictionary->offset(),
          .length = serializedFileDictionary->length()};
      NIMBLE_CHECK_FILE(
          parsed.fileDictionaryIndices_
              .emplace(
                  fileDictionary.dictionaryId, parsed.fileDictionaries_.size())
              .second,
          "Duplicate file shared dictionary ID {}.",
          fileDictionary.dictionaryId);
      parsed.fileDictionaries_.push_back(std::move(fileDictionary));
    }
  }
  for (const auto& reference : parsed.fileDictionaryReferences_) {
    const auto* fileDictionary =
        parsed.findFileDictionary(reference.dictionaryId);
    NIMBLE_CHECK_FILE(
        fileDictionary != nullptr,
        "File shared dictionary {} does not exist.",
        reference.dictionaryId);
    NIMBLE_CHECK_FILE_EQ(
        fileDictionary->dataType,
        reference.dataType,
        "File shared dictionary {} has an inconsistent type.",
        reference.dictionaryId);
  }
  return parsed;
}

const SharedDictionaryReference*
SharedDictionaryCatalog::findStripeDictionaryReference(
    uint32_t valueStreamId) const {
  return findEntry(
      stripeDictionaryReferences_,
      stripeDictionaryReferenceIndices_,
      valueStreamId);
}

const SharedDictionaryReference*
SharedDictionaryCatalog::findFileDictionaryReference(
    uint32_t valueStreamId) const {
  return findEntry(
      fileDictionaryReferences_,
      fileDictionaryReferenceIndices_,
      valueStreamId);
}

const SharedDictionaryReference*
SharedDictionaryCatalog::findExternalDictionaryReference(
    uint32_t valueStreamId) const {
  return findEntry(
      externalDictionaryReferences_,
      externalDictionaryReferenceIndices_,
      valueStreamId);
}

const FileDictionary* SharedDictionaryCatalog::findFileDictionary(
    uint32_t dictionaryId) const {
  return findEntry(fileDictionaries_, fileDictionaryIndices_, dictionaryId);
}

} // namespace facebook::nimble
