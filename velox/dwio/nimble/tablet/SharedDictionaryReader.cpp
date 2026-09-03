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

#include "velox/dwio/nimble/tablet/SharedDictionaryReader.h"

#include <utility>

#include "velox/common/Casts.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"

namespace facebook::nimble {

std::unique_ptr<const SharedDictionaryReaderFactory>
SharedDictionaryReaderFactory::create(
    std::string_view catalog,
    std::shared_ptr<const ExternalDictionaryResolver> externalResolver,
    const TabletReader* tabletReader,
    velox::memory::MemoryPool* pool) {
  return std::unique_ptr<SharedDictionaryReaderFactory>(
      new SharedDictionaryReaderFactory{
          catalog,
          std::move(externalResolver),
          /*tabletReader=*/tabletReader,
          /*pool=*/pool});
}

SharedDictionaryReaderFactory::SharedDictionaryReaderFactory(
    std::string_view catalog,
    std::shared_ptr<const ExternalDictionaryResolver> externalResolver,
    const TabletReader* tabletReader,
    velox::memory::MemoryPool* pool)
    : tabletReader_{velox::checkedNotNull(tabletReader)},
      pool_{velox::checkedNotNull(pool)},
      externalResolver_{std::move(externalResolver)},
      catalog_{SharedDictionaryCatalog::deserialize(catalog)},
      cache_{
          [this](AlphabetCacheKey cacheKey) { return loadAlphabet(cacheKey); },
          /*pinEntries=*/true} {}

std::optional<uint32_t> SharedDictionaryReaderFactory::dictionaryStreamId(
    uint32_t valueStreamId) const {
  const auto* reference = catalog_.findStripeDictionaryReference(valueStreamId);
  return reference == nullptr
      ? std::nullopt
      : std::optional<uint32_t>{reference->dictionaryId};
}

folly::F14FastMap<uint32_t, uint32_t>
SharedDictionaryReaderFactory::dictionaryStreamIds(
    std::span<const uint32_t> valueStreamIds) const {
  if (catalog_.stripeDictionaryReferences().empty()) {
    return {};
  }

  folly::F14FastMap<uint32_t, uint32_t> dictionaryStreamIds;
  dictionaryStreamIds.reserve(valueStreamIds.size());
  for (const auto valueStreamId : valueStreamIds) {
    const auto* reference =
        catalog_.findStripeDictionaryReference(valueStreamId);
    if (reference != nullptr) {
      dictionaryStreamIds.emplace(valueStreamId, reference->dictionaryId);
    }
  }
  return dictionaryStreamIds;
}

std::shared_ptr<const SharedDictionaryAlphabet>
SharedDictionaryReaderFactory::resolveAlphabet(uint32_t valueStreamId) const {
  const auto* fileReference =
      catalog_.findFileDictionaryReference(valueStreamId);
  if (fileReference != nullptr) {
    return cache_.getOrCreate(alphabetCacheKey(
        SharedDictionaryScope::File,
        fileReference->dictionaryId,
        fileReference->dataType));
  }

  const auto* externalReference =
      catalog_.findExternalDictionaryReference(valueStreamId);
  return externalReference == nullptr ? nullptr
                                      : cache_.getOrCreate(alphabetCacheKey(
                                            SharedDictionaryScope::External,
                                            externalReference->dictionaryId,
                                            externalReference->dataType));
}

SharedDictionaryReaderFactory::AlphabetCacheKey
SharedDictionaryReaderFactory::alphabetCacheKey(
    SharedDictionaryScope scope,
    uint32_t dictionaryId,
    DataType dataType) {
  return (static_cast<uint64_t>(static_cast<uint8_t>(scope)) << 40) |
      (static_cast<uint64_t>(static_cast<uint8_t>(dataType)) << 32) |
      dictionaryId;
}

SharedDictionaryReaderFactory::AlphabetPtr
SharedDictionaryReaderFactory::loadAlphabet(AlphabetCacheKey cacheKey) const {
  const auto scope =
      toSharedDictionaryScope(static_cast<uint8_t>(cacheKey >> 40));
  const auto dataType =
      static_cast<DataType>(static_cast<uint8_t>(cacheKey >> 32));
  const auto dictionaryId = static_cast<uint32_t>(cacheKey);

  AlphabetPtr alphabet;
  switch (scope) {
    case SharedDictionaryScope::Stripe:
      NIMBLE_UNREACHABLE("Stripe dictionaries are decoded by the reader.");
    case SharedDictionaryScope::File: {
      const auto* fileDictionary = catalog_.findFileDictionary(dictionaryId);
      NIMBLE_CHECK_FILE_NOT_NULL(
          fileDictionary,
          "File shared dictionary {} does not exist.",
          dictionaryId);
      NIMBLE_CHECK_FILE_LE(
          fileDictionary->offset,
          tabletReader_->fileSize_,
          "File shared dictionary {} points outside the tablet.",
          dictionaryId);
      NIMBLE_CHECK_FILE_LE(
          fileDictionary->length,
          tabletReader_->fileSize_ - fileDictionary->offset,
          "File shared dictionary {} points outside the tablet.",
          dictionaryId);
      std::shared_ptr<const MetadataBuffer> encodedAlphabetOwner{
          tabletReader_->readMetadata(
              MetadataSection{
                  fileDictionary->offset,
                  fileDictionary->length,
                  CompressionType::Uncompressed,
                  fileDictionary->length})};
      const auto encodedAlphabet = encodedAlphabetOwner->content();
      alphabet = SharedDictionaryAlphabet::create(
          encodedAlphabet, std::move(encodedAlphabetOwner), pool_);
      NIMBLE_CHECK_FILE_EQ(
          alphabet->dataType(),
          fileDictionary->dataType,
          "File shared dictionary {} has an inconsistent type.",
          dictionaryId);
      break;
    }
    case SharedDictionaryScope::External:
      NIMBLE_USER_CHECK_NOT_NULL(
          externalResolver_,
          "External shared dictionary {} requires an ExternalDictionaryResolver.",
          dictionaryId);
      alphabet = externalResolver_->resolve(dictionaryId, dataType);
      break;
  }

  NIMBLE_CHECK_NOT_NULL(alphabet);
  return alphabet;
}

} // namespace facebook::nimble
