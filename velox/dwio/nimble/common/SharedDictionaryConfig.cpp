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

} // namespace

SharedDictionaryConfigBuilder::SharedDictionaryConfigBuilder(
    SharedDictionaryEncodingConfig&& config)
    : config_{std::move(config)} {}

SharedDictionaryConfigBuilder SharedDictionaryEncodingConfig::builder(
    SharedDictionaryEncodingConfig&& config) {
  return SharedDictionaryConfigBuilder{std::move(config)};
}

SharedDictionaryConfigBuilder&
SharedDictionaryConfigBuilder::setExternalResolver(
    std::shared_ptr<const ExternalDictionaryResolver> externalResolver) {
  config_.externalResolver = std::move(externalResolver);
  return *this;
}

SharedDictionaryConfigBuilder&
SharedDictionaryConfigBuilder::addColumnDictionary(
    std::string fieldPath,
    SharedDictionaryConfig dictionary) {
  validateValueStreamPath(fieldPath, "column");

  const auto duplicate = std::any_of(
      config_.columns.begin(),
      config_.columns.end(),
      [&](const auto& candidate) { return candidate.fieldPath == fieldPath; });
  NIMBLE_USER_CHECK(
      !duplicate,
      "Duplicate shared dictionary column configuration for path '{}'.",
      fieldPath);
  config_.columns.push_back(
      ColumnDictionary{
          .fieldPath = std::move(fieldPath),
          .dictionary = std::move(dictionary)});
  return *this;
}

SharedDictionaryConfigBuilder&
SharedDictionaryConfigBuilder::addFlatmapValueDictionary(
    std::string fieldPath,
    int64_t key,
    SharedDictionaryConfig dictionary,
    std::string valueSubfield) {
  const auto subfield = validateFlatMapColumnPath(fieldPath);
  validateValueSubfield(valueSubfield);

  auto column = std::find_if(
      config_.flatMaps.begin(),
      config_.flatMaps.end(),
      [&](const auto& candidate) { return candidate.fieldPath == fieldPath; });
  if (column == config_.flatMaps.end()) {
    config_.flatMaps.push_back(
        FlatmapColumnDictionary{.fieldPath = fieldPath, .keys = {}});
    column = std::prev(config_.flatMaps.end());
  }

  const auto duplicate = std::any_of(
      column->keys.begin(), column->keys.end(), [&](const auto& item) {
        return item.key == key && item.valueSubfield == valueSubfield;
      });
  NIMBLE_USER_CHECK(
      !duplicate,
      "Duplicate shared dictionary flat-map value configuration for path '{}', "
      "key {}, and value subfield '{}'.",
      fieldPath,
      key,
      valueSubfield);
  column->keys.push_back(
      FlatmapKeyDictionary{
          .key = key,
          .valueSubfield = std::move(valueSubfield),
          .dictionary = std::move(dictionary)});
  return *this;
}

SharedDictionaryEncodingConfig SharedDictionaryConfigBuilder::build() {
  return std::move(config_);
}

} // namespace facebook::nimble
