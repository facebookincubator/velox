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

#include "folly/container/F14Map.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"

namespace facebook::nimble {

class SharedDictionaryConfigBuilder;

/// Shared dictionary writer configuration for one value stream.
struct SharedDictionaryConfig {
  /// Where the dictionary alphabet is stored or resolved.
  SharedDictionaryScope scope{SharedDictionaryScope::Stripe};

  /// Identifies the dictionary inside the written file. The writer assigns
  /// this for Stripe and File scope -- from the generated auxiliary alphabet
  /// stream and from a per-file counter respectively -- so callers must leave
  /// it unset for those scopes. External scope stores no alphabet in the file,
  /// so the caller sets this and it serves as the resolver key on both the
  /// write and the read side.
  uint32_t dictionaryId{};

  /// Resolves a provided alphabet instead of building one from written values.
  bool useExternalAlphabet{false};

  /// Names this dictionary in the caller's own store, for File scope with
  /// useExternalAlphabet. Needed only there: External scope is caller-named
  /// end to end, so its |dictionaryId| already is the resolver key, while a
  /// File dictionary gets a writer-assigned id and so must carry the caller's
  /// key separately. Caller-owned, so unlike |dictionaryId| it may repeat
  /// across value streams that share one provided dictionary.
  uint32_t resolverKey{};

  /// Candidate encodings for alphabets stored in this file.
  std::vector<EncodingType> alphabetEncodings{};
};

/// Shared dictionary settings for one value stream whose dictionary is named
/// by a caller-owned resolver key.
struct ProvidedDictionary {
  /// Velox subfield from the writer input root to the configured value stream.
  /// When the path resolves to an array or map, the dictionary applies to the
  /// element or value stream recursively.
  std::string fieldPath;

  /// Dictionary encoding settings for the resolved value stream. The scope
  /// must be External, or File with useExternalAlphabet.
  SharedDictionaryConfig dictionary;
};

/// Shared dictionary settings for one flat-map key served by a provided
/// dictionary.
struct ProvidedFlatmapKeyDictionary {
  /// Flat-map key to configure.
  int64_t key{};

  /// Velox subfield below the keyed flat-map value. Empty selects the keyed
  /// value itself; if that value is an array or map, the dictionary applies to
  /// its element or value stream recursively.
  std::string valueSubfield;

  /// Dictionary encoding settings for the key value stream. The scope must be
  /// External, or File with useExternalAlphabet.
  SharedDictionaryConfig dictionary;
};

/// Provided-dictionary settings for one top-level flat-map column.
struct ProvidedFlatmapColumnDictionary {
  /// Row-field path from the writer input root to the configured flat-map
  /// column. This currently names a top-level writer input column.
  std::string fieldPath;

  /// Per-key shared dictionary settings.
  std::vector<ProvidedFlatmapKeyDictionary> keys;
};

/// Shared dictionary writer configuration for a file.
///
/// The configuration has two halves, split by who owns the dictionary's
/// identity rather than by scope.
///
/// Stripe and File dictionaries are named inside the written file and the
/// writer assigns their ids, so one entry can cover however many value streams
/// a column expands into. Those are enabled by naming top-level columns in
/// |columns|, with |scope| and |alphabetEncodings| applying to all of them.
///
/// External dictionaries, and File dictionaries reading a provided alphabet,
/// are named by a caller-owned resolver key. A key means something only in the
/// caller's store and cannot be synthesized, so one has to be supplied per
/// value stream; those configurations stay addressed by path in |provided| and
/// |providedFlatMaps|.
struct SharedDictionaryEncodingConfig {
  /// Candidate encodings for alphabets stored in this file, for |columns|.
  std::vector<EncodingType> alphabetEncodings;

  /// Top-level writer input columns to enable, each mapped to where its
  /// alphabets are stored. Only Stripe and File appear here; a provided
  /// dictionary carries its own scope. Columns are free to differ, so one file
  /// can hold both stripe-scoped and file-scoped dictionaries.
  ///
  /// Every eligible value stream the column expands into gets its own
  /// dictionary: the value stream of a scalar column, the element or value
  /// stream of an array or map column, and one per key for a flat-map column.
  /// Eligible means an integer or string scalar; a named column with no
  /// eligible value stream is an error.
  ///
  /// Struct members cannot be named. Reaching one needs a path, and a path can
  /// address only a single value stream, which is what this half of the
  /// configuration exists to avoid.
  folly::F14FastMap<std::string, SharedDictionaryScope> columns;

  /// Value streams served by a caller-provided dictionary.
  std::vector<ProvidedDictionary> provided;

  /// Top-level flat-map columns with per-key provided dictionaries.
  std::vector<ProvidedFlatmapColumnDictionary> providedFlatMaps;

  /// Supplies external alphabets for External shared dictionary configurations
  /// and File configurations that set useExternalAlphabet.
  std::shared_ptr<const ExternalDictionaryResolver> externalResolver;

  /// Returns true when no value streams request shared dictionary encoding.
  bool empty() const {
    return columns.empty() && provided.empty() && providedFlatMaps.empty();
  }

  /// Returns true when any configured dictionary is stripe scoped.
  bool hasStripeDictionary() const;

  /// Creates an empty builder.
  static SharedDictionaryConfigBuilder builder();

  /// Creates a builder seeded from an existing config. Separate from the
  /// overload above rather than a defaulted argument, because a `= {}` here
  /// would need this class's own default member initializers before they are
  /// parsed.
  static SharedDictionaryConfigBuilder builder(
      SharedDictionaryEncodingConfig&& config);
};

/// Builder for SharedDictionaryEncodingConfig.
class SharedDictionaryConfigBuilder {
 public:
  /// Creates an empty builder.
  SharedDictionaryConfigBuilder();

  /// Creates a builder seeded from config.
  explicit SharedDictionaryConfigBuilder(
      SharedDictionaryEncodingConfig&& config);

  /// Sets the resolver used by external dictionaries and provided file
  /// alphabets.
  SharedDictionaryConfigBuilder& setExternalResolver(
      std::shared_ptr<const ExternalDictionaryResolver> externalResolver);

  /// Sets the alphabet encodings shared by every enabled column.
  SharedDictionaryConfigBuilder& setAlphabetEncodings(
      std::vector<EncodingType> alphabetEncodings);

  /// Enables shared dictionary encoding for one top-level column.
  SharedDictionaryConfigBuilder& addColumn(
      std::string column,
      SharedDictionaryScope scope = SharedDictionaryScope::Stripe);

  /// Adds a provided dictionary for one value stream.
  SharedDictionaryConfigBuilder& addProvidedDictionary(
      std::string fieldPath,
      SharedDictionaryConfig dictionary);

  /// Adds a provided dictionary for one top-level flat-map key value stream.
  SharedDictionaryConfigBuilder& addProvidedFlatmapValueDictionary(
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
