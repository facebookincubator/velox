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

#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include "folly/container/F14Map.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryCatalog.h"
#include "velox/dwio/nimble/tablet/MetadataCache.h"

namespace facebook::nimble {

class TabletReader;

/// Resolves External shared dictionaries referenced by a tablet.
class ExternalDictionaryResolver {
 public:
  virtual ~ExternalDictionaryResolver() = default;

  virtual std::shared_ptr<const SharedDictionaryAlphabet> resolve(
      uint32_t dictionaryId,
      DataType dataType) const = 0;
};

/// Resolves file-wide value-stream bindings and caches decoded alphabets.
class SharedDictionaryReaderFactory {
 public:
  /// Creates and initializes a reader factory from serialized catalog data.
  static std::unique_ptr<const SharedDictionaryReaderFactory> create(
      std::string_view catalog,
      std::shared_ptr<const ExternalDictionaryResolver> externalResolver,
      const TabletReader* tabletReader,
      velox::memory::MemoryPool* pool);

  /// Returns whether the tablet contains file or external dictionaries.
  bool hasFileOrExternalDictionaries() const {
    return !catalog_.fileDictionaryReferences().empty() ||
        !catalog_.externalDictionaryReferences().empty();
  }

  /// Returns whether the tablet contains stripe dictionary bindings.
  bool hasStripeDictionaries() const {
    return !catalog_.stripeDictionaryReferences().empty();
  }

  /// Returns the stripe dictionary stream for a value stream, if any.
  std::optional<uint32_t> dictionaryStreamId(uint32_t valueStreamId) const;

  /// Returns value stream to stripe dictionary stream bindings. Returns empty
  /// when none of the supplied streams uses a stripe dictionary.
  folly::F14FastMap<uint32_t, uint32_t> dictionaryStreamIds(
      std::span<const uint32_t> valueStreamIds) const;

  /// Returns the file or external alphabet for a value stream, if any.
  std::shared_ptr<const SharedDictionaryAlphabet> resolveAlphabet(
      uint32_t valueStreamId) const;

 private:
  using AlphabetPtr = std::shared_ptr<const SharedDictionaryAlphabet>;
  using AlphabetCacheKey = uint64_t;

  // Parses the catalog and configures lazy alphabet loading.
  SharedDictionaryReaderFactory(
      std::string_view catalog,
      std::shared_ptr<const ExternalDictionaryResolver> externalResolver,
      const TabletReader* tabletReader,
      velox::memory::MemoryPool* pool);

  // Packs a scope, type, and dictionary ID into a cache key.
  static AlphabetCacheKey alphabetCacheKey(
      SharedDictionaryScope scope,
      uint32_t dictionaryId,
      DataType dataType);

  // Loads and decodes a file or external alphabet on first use.
  AlphabetPtr loadAlphabet(AlphabetCacheKey cacheKey) const;

  // Tablet that owns file-scope alphabet ranges.
  const TabletReader* const tabletReader_;
  // Pool used to decode alphabet entries.
  velox::memory::MemoryPool* const pool_;
  // Optional provider for dictionaries not stored in the tablet.
  const std::shared_ptr<const ExternalDictionaryResolver> externalResolver_;
  // Immutable scope bindings and file dictionary locations.
  const SharedDictionaryCatalog catalog_;

  // Decoded file and external alphabets retained for the reader lifetime.
  mutable MetadataCache<AlphabetCacheKey, const SharedDictionaryAlphabet>
      cache_;
};

} // namespace facebook::nimble
