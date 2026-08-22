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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"

namespace facebook::nimble {

/// Encoded external dictionary alphabet and its standalone artifact metadata.
struct ExternalDictionary {
  /// Nimble type of the dictionary values.
  DataType dataType{DataType::Undefined};

  /// Root encoding type used by the serialized alphabet.
  EncodingType alphabetEncodingType{};

  /// Whether the artifact builder sorted unique values before encoding.
  bool sortValues{true};

  /// Number of unique values encoded in encodedAlphabet.
  uint64_t valueCount{};

  /// Encoded alphabet bytes.
  std::string encodedAlphabet;
};

/// Builds, serializes, and loads standalone external dictionary artifacts.
class ExternalDictionaryBuilder {
 public:
  /// Options for building an encoded external dictionary alphabet.
  struct Options {
    /// Whether to sort unique values before encoding the alphabet.
    bool sortValues{true};

    /// Optional forced alphabet encoding. When unset, readFactors or default
    /// encoding selection picks the alphabet encoding.
    std::optional<EncodingType> alphabetEncoding;

    /// Optional parsed manual read factors used by encoding selection.
    std::vector<std::pair<EncodingType, float>> readFactors;
  };

  explicit ExternalDictionaryBuilder(velox::memory::MemoryPool* pool);

  /// Builds an encoded alphabet from typed integer values. Duplicates are
  /// removed using sorted order or first-seen order based on options.
  ExternalDictionary build(const Vector<int8_t>& source, const Options& options)
      const;
  ExternalDictionary build(
      const Vector<uint8_t>& source,
      const Options& options) const;
  ExternalDictionary build(
      const Vector<int16_t>& source,
      const Options& options) const;
  ExternalDictionary build(
      const Vector<uint16_t>& source,
      const Options& options) const;
  ExternalDictionary build(
      const Vector<int32_t>& source,
      const Options& options) const;
  ExternalDictionary build(
      const Vector<uint32_t>& source,
      const Options& options) const;
  ExternalDictionary build(
      const Vector<int64_t>& source,
      const Options& options) const;
  ExternalDictionary build(
      const Vector<uint64_t>& source,
      const Options& options) const;

  /// Serializes a standalone external dictionary artifact.
  std::string serialize(const ExternalDictionary& alphabet) const;

  /// Deserializes an artifact from serialized bytes.
  ExternalDictionary deserialize(std::string_view data) const;

  /// Reads an artifact file and returns its encoded alphabet.
  ExternalDictionary deserializeFromFile(std::string_view path) const;

 private:
  ExternalDictionary deserializeImpl(std::string_view data) const;

  // Pool used for temporary alphabet buffers and loaded alphabet storage.
  velox::memory::MemoryPool* const pool_;
};

} // namespace facebook::nimble
