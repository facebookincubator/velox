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

#include "velox/dwio/nimble/common/SharedDictionaryConfig.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/type/Subfield.h"

namespace facebook::nimble {
namespace {

velox::common::Subfield parseFieldPath(const std::string& fieldPath) {
  NIMBLE_USER_CHECK(
      !fieldPath.empty(), "Shared dictionary path must not be empty.");
  velox::common::Subfield subfield;
  try {
    subfield = velox::common::Subfield{fieldPath};
  } catch (const velox::VeloxException&) {
    NIMBLE_USER_FAIL(
        "Shared dictionary path '{}' must start with a field name.", fieldPath);
  }
  NIMBLE_USER_CHECK(
      subfield.valid(),
      "Shared dictionary path '{}' must start with a field name.",
      fieldPath);
  return subfield;
}

void validateValueStreamPath(
    const std::string& fieldPath,
    std::string_view configType) {
  const auto subfield = parseFieldPath(fieldPath);
  for (const auto& pathElement : subfield.path()) {
    NIMBLE_USER_CHECK(
        pathElement->is(velox::common::SubfieldKind::kNestedField) ||
            pathElement->is(velox::common::SubfieldKind::kAllSubscripts),
        "Shared dictionary {} path '{}' only supports nested row fields and "
        "all-subscript array/map elements.",
        configType,
        fieldPath);
  }
}

velox::common::Subfield validateFlatMapColumnPath(
    const std::string& fieldPath) {
  auto subfield = parseFieldPath(fieldPath);
  NIMBLE_USER_CHECK_EQ(
      subfield.path().size(),
      1,
      "Shared dictionary flat-map column path '{}' must be a top-level writer "
      "input column.",
      fieldPath);
  return subfield;
}

void validateValueSubfield(const std::string& valueSubfield) {
  if (valueSubfield.empty()) {
    return;
  }
  const auto subfield = parseFieldPath(valueSubfield);
  for (const auto& pathElement : subfield.path()) {
    NIMBLE_USER_CHECK(
        pathElement->is(velox::common::SubfieldKind::kNestedField) ||
            pathElement->is(velox::common::SubfieldKind::kAllSubscripts),
        "Shared dictionary flat-map value subfield '{}' only supports nested "
        "row fields and all-subscript array/map elements.",
        valueSubfield);
  }
}

// Provided dictionaries exist to carry a caller-owned resolver key, so reject
// the configurations that have no key to carry.
void validateProvidedDictionary(
    const SharedDictionaryConfig& dictionary,
    const std::string& fieldPath) {
  NIMBLE_USER_CHECK(
      dictionary.scope == SharedDictionaryScope::External ||
          (dictionary.scope == SharedDictionaryScope::File &&
           dictionary.useExternalAlphabet),
      "Provided shared dictionary '{}' must use external scope, or file scope "
      "with useExternalAlphabet. Writer-assigned dictionaries are enabled by "
      "column instead.",
      fieldPath);
}

} // namespace

SharedDictionaryConfigBuilder::SharedDictionaryConfigBuilder() = default;

SharedDictionaryConfigBuilder::SharedDictionaryConfigBuilder(
    SharedDictionaryEncodingConfig&& config)
    : config_{std::move(config)} {}

SharedDictionaryConfigBuilder SharedDictionaryEncodingConfig::builder() {
  return SharedDictionaryConfigBuilder{};
}

SharedDictionaryConfigBuilder SharedDictionaryEncodingConfig::builder(
    SharedDictionaryEncodingConfig&& config) {
  return SharedDictionaryConfigBuilder{std::move(config)};
}

bool SharedDictionaryEncodingConfig::hasStripeDictionary() const {
  for (const auto& [_, columnScope] : columns) {
    if (columnScope == SharedDictionaryScope::Stripe) {
      return true;
    }
  }
  // The builder rejects stripe-scoped provided dictionaries, but the fields are
  // public and assignable, so scan rather than trust that invariant here: the
  // writer sizes stripe state off this answer and under-reporting is silent.
  const auto isStripe = [](const SharedDictionaryConfig& dictionary) {
    return dictionary.scope == SharedDictionaryScope::Stripe;
  };
  return std::any_of(
             provided.begin(),
             provided.end(),
             [&](const auto& entry) { return isStripe(entry.dictionary); }) ||
      std::any_of(
             providedFlatMaps.begin(),
             providedFlatMaps.end(),
             [&](const auto& column) {
               return std::any_of(
                   column.keys.begin(),
                   column.keys.end(),
                   [&](const auto& key) { return isStripe(key.dictionary); });
             });
}

SharedDictionaryConfigBuilder&
SharedDictionaryConfigBuilder::setExternalResolver(
    std::shared_ptr<const ExternalDictionaryResolver> externalResolver) {
  config_.externalResolver = std::move(externalResolver);
  return *this;
}

SharedDictionaryConfigBuilder&
SharedDictionaryConfigBuilder::setAlphabetEncodings(
    std::vector<EncodingType> alphabetEncodings) {
  config_.alphabetEncodings = std::move(alphabetEncodings);
  return *this;
}

SharedDictionaryConfigBuilder& SharedDictionaryConfigBuilder::addColumn(
    std::string column,
    SharedDictionaryScope scope) {
  NIMBLE_USER_CHECK_NE(
      scope,
      SharedDictionaryScope::External,
      "Enabled column '{}' cannot use external scope; an external dictionary "
      "is named by a caller-owned resolver key and must be configured per "
      "value stream.",
      column);
  NIMBLE_USER_CHECK(
      !column.empty(), "Shared dictionary column name must not be empty.");
  NIMBLE_USER_CHECK(
      column.find_first_of(".[") == std::string::npos,
      "Shared dictionary column '{}' must be a top-level writer input column "
      "name, not a path. Struct members are only reachable through a provided "
      "dictionary.",
      column);
  const auto inserted = config_.columns.emplace(column, scope).second;
  NIMBLE_USER_CHECK(
      inserted, "Duplicate shared dictionary column '{}'.", column);
  return *this;
}

SharedDictionaryConfigBuilder&
SharedDictionaryConfigBuilder::addProvidedDictionary(
    std::string fieldPath,
    SharedDictionaryConfig dictionary) {
  validateValueStreamPath(fieldPath, "column");
  validateProvidedDictionary(dictionary, fieldPath);

  const auto duplicate = std::any_of(
      config_.provided.begin(),
      config_.provided.end(),
      [&](const auto& candidate) { return candidate.fieldPath == fieldPath; });
  NIMBLE_USER_CHECK(
      !duplicate,
      "Duplicate provided shared dictionary configuration for path '{}'.",
      fieldPath);
  config_.provided.push_back(
      ProvidedDictionary{
          .fieldPath = std::move(fieldPath),
          .dictionary = std::move(dictionary)});
  return *this;
}

SharedDictionaryConfigBuilder&
SharedDictionaryConfigBuilder::addProvidedFlatmapValueDictionary(
    std::string fieldPath,
    int64_t key,
    SharedDictionaryConfig dictionary,
    std::string valueSubfield) {
  const auto subfield = validateFlatMapColumnPath(fieldPath);
  validateValueSubfield(valueSubfield);
  validateProvidedDictionary(dictionary, fieldPath);

  auto column = std::find_if(
      config_.providedFlatMaps.begin(),
      config_.providedFlatMaps.end(),
      [&](const auto& candidate) { return candidate.fieldPath == fieldPath; });
  if (column == config_.providedFlatMaps.end()) {
    config_.providedFlatMaps.push_back(
        ProvidedFlatmapColumnDictionary{.fieldPath = fieldPath, .keys = {}});
    column = std::prev(config_.providedFlatMaps.end());
  }

  const auto duplicate = std::any_of(
      column->keys.begin(), column->keys.end(), [&](const auto& item) {
        return item.key == key && item.valueSubfield == valueSubfield;
      });
  NIMBLE_USER_CHECK(
      !duplicate,
      "Duplicate provided shared dictionary flat-map value configuration for "
      "path '{}', key {}, and value subfield '{}'.",
      fieldPath,
      key,
      valueSubfield);
  column->keys.push_back(
      ProvidedFlatmapKeyDictionary{
          .key = key,
          .valueSubfield = std::move(valueSubfield),
          .dictionary = std::move(dictionary)});
  return *this;
}

SharedDictionaryEncodingConfig SharedDictionaryConfigBuilder::build() {
  return std::move(config_);
}

} // namespace facebook::nimble
