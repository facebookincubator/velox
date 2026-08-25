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
#include <memory>
#include <string>
#include <vector>

#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/tablet/SharedDictionaryReader.h"

namespace facebook::nimble {

class SharedDictionaryConfigBuilder;

/// Shared dictionary writer configuration for one value stream.
struct SharedDictionaryConfig {
  /// Where the dictionary alphabet is stored or resolved.
  SharedDictionaryScope scope{SharedDictionaryScope::Stripe};
  /// Dictionary id within File or External scope. Stripe scope assigns this
  /// from the generated auxiliary alphabet stream.
  uint32_t dictionaryId{};
  /// Resolves a provided alphabet instead of building one from written values.
  bool useExternalAlphabet{false};
  /// Candidate encodings for alphabets stored in this file.
  std::vector<EncodingType> alphabetEncodings{};
};

/// Shared dictionary settings for one regular column value stream.
struct ColumnDictionary {
  /// Velox subfield from the writer input root to the configured value stream.
  /// When the path resolves to an array or map, the dictionary applies to the
  /// element or value stream recursively.
  std::string fieldPath;

  /// Dictionary encoding settings for the resolved value stream.
  SharedDictionaryConfig dictionary;
};

/// Shared dictionary settings for one flat-map key.
struct FlatmapKeyDictionary {
  /// Flat-map key to configure.
  int64_t key{};

  /// Velox subfield below the keyed flat-map value. Empty selects the keyed
  /// value itself; if that value is an array or map, the dictionary applies to
  /// its element or value stream recursively.
  std::string valueSubfield;

  /// Dictionary encoding settings for the key value stream.
  SharedDictionaryConfig dictionary;
};

/// Shared dictionary settings for one top-level flat-map column.
struct FlatmapColumnDictionary {
  /// Row-field path from the writer input root to the configured flat-map
  /// column. This currently names a top-level writer input column.
  std::string fieldPath;

  /// Per-key shared dictionary settings.
  std::vector<FlatmapKeyDictionary> keys;
};

/// Shared dictionary writer configuration for regular columns and flat-map
/// value streams in a file.
struct SharedDictionaryEncodingConfig {
  /// Regular columns eligible for shared dictionary encoding. The resolved
  /// value must be an integer scalar, an array with an integer element stream,
  /// or a map with an integer value stream. For File scope, callers are
  /// responsible for assigning a unique dictionaryId per configured value
  /// stream.
  std::vector<ColumnDictionary> columns;

  /// Top-level flat-map columns eligible for shared dictionary encoding.
  /// valueSubfield paths resolve below the keyed value. The configured value
  /// stream must be an integer scalar, an array with an integer element stream,
  /// or a map with an integer value stream. For File scope, callers are
  /// responsible for assigning a unique dictionaryId per configured key value
  /// stream.
  std::vector<FlatmapColumnDictionary> flatMaps;

  /// Supplies external alphabets for External shared dictionary configurations
  /// and File configurations that set useExternalAlphabet.
  std::shared_ptr<const ExternalDictionaryResolver> externalResolver;

  /// Returns true when no value streams request shared dictionary encoding.
  bool empty() const {
    return columns.empty() && flatMaps.empty();
  }

  /// Creates a builder, optionally seeded from an existing config.
  static SharedDictionaryConfigBuilder builder(
      SharedDictionaryEncodingConfig&& config = {});
};

/// Builder for SharedDictionaryEncodingConfig.
class SharedDictionaryConfigBuilder {
 public:
  /// Creates a builder seeded from config.
  explicit SharedDictionaryConfigBuilder(
      SharedDictionaryEncodingConfig&& config = {});

  /// Sets the resolver used by external dictionaries and provided file
  /// alphabets.
  SharedDictionaryConfigBuilder& setExternalResolver(
      std::shared_ptr<const ExternalDictionaryResolver> externalResolver);

  /// Adds dictionary settings for one regular column value stream.
  SharedDictionaryConfigBuilder& addColumnDictionary(
      std::string fieldPath,
      SharedDictionaryConfig dictionary);

  /// Adds dictionary settings for one top-level flat-map key value stream.
  SharedDictionaryConfigBuilder& addFlatmapValueDictionary(
      std::string fieldPath,
      int64_t key,
      SharedDictionaryConfig dictionary,
      std::string valueSubfield = "");

  /// Returns the completed config.
  SharedDictionaryEncodingConfig build();

 private:
  SharedDictionaryEncodingConfig config_;
};

} // namespace facebook::nimble
