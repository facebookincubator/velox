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

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "folly/container/F14Map.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryTypes.h"

namespace facebook::nimble {

/// Location of a file dictionary parsed from a shared dictionary catalog.
struct FileDictionary {
  /// Dictionary identifier within the file scope.
  uint32_t dictionaryId{};
  /// Nimble type of the alphabet values.
  DataType dataType{DataType::Undefined};
  /// Absolute file offset of the encoded alphabet.
  uint64_t offset{};
  /// Byte length of the encoded alphabet.
  uint32_t length{};
};

/// References a dictionary from a value stream within one catalog scope.
struct SharedDictionaryReference {
  /// Logical stream whose values use this dictionary.
  uint32_t valueStreamId{};
  /// Dictionary identifier interpreted within the reference scope.
  uint32_t dictionaryId{kInvalidSharedDictionaryId};
  /// Nimble type of the dictionary values.
  DataType dataType{DataType::Undefined};
};

/// Immutable parsed contents of the shared dictionary section.
class SharedDictionaryCatalog {
 public:
  /// Serializes value-stream references and file dictionary locations.
  static std::string serialize(
      std::span<const SharedDictionaryReference> stripeDictionaryReferences,
      std::span<const SharedDictionaryReference> fileDictionaryReferences,
      std::span<const SharedDictionaryReference> externalDictionaryReferences,
      std::span<const FileDictionary> fileDictionaries);

  /// Deserializes and validates shared dictionary metadata.
  static SharedDictionaryCatalog deserialize(std::string_view catalog);

  const std::vector<SharedDictionaryReference>& stripeDictionaryReferences()
      const {
    return stripeDictionaryReferences_;
  }

  const std::vector<SharedDictionaryReference>& fileDictionaryReferences()
      const {
    return fileDictionaryReferences_;
  }

  const std::vector<SharedDictionaryReference>& externalDictionaryReferences()
      const {
    return externalDictionaryReferences_;
  }

  const std::vector<FileDictionary>& fileDictionaries() const {
    return fileDictionaries_;
  }

  /// Finds a stripe dictionary reference by value stream ID.
  const SharedDictionaryReference* findStripeDictionaryReference(
      uint32_t valueStreamId) const;

  /// Finds a file dictionary reference by value stream ID.
  const SharedDictionaryReference* findFileDictionaryReference(
      uint32_t valueStreamId) const;

  /// Finds an external dictionary reference by value stream ID.
  const SharedDictionaryReference* findExternalDictionaryReference(
      uint32_t valueStreamId) const;

  /// Finds a file dictionary by dictionary ID.
  const FileDictionary* findFileDictionary(uint32_t dictionaryId) const;

 private:
  // References to stripe-local dictionary streams.
  std::vector<SharedDictionaryReference> stripeDictionaryReferences_;
  // Value-stream bindings. Multiple streams may share one file dictionary.
  std::vector<SharedDictionaryReference> fileDictionaryReferences_;
  // References resolved by an external provider.
  std::vector<SharedDictionaryReference> externalDictionaryReferences_;
  // File-scope alphabet locations.
  std::vector<FileDictionary> fileDictionaries_;

  // Entry positions indexed by their lookup IDs.
  folly::F14FastMap<uint32_t, size_t> stripeDictionaryReferenceIndices_;
  folly::F14FastMap<uint32_t, size_t> fileDictionaryReferenceIndices_;
  folly::F14FastMap<uint32_t, size_t> externalDictionaryReferenceIndices_;
  folly::F14FastMap<uint32_t, size_t> fileDictionaryIndices_;
};

} // namespace facebook::nimble
