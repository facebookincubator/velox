/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#include "velox/connectors/hive/paimon/PaimonSubfield.h"

namespace facebook::velox::connector::hive::paimon {

// Encode path elements explicitly to distinguish literal dots and brackets
// in names from path separators.
folly::dynamic serializeSubfield(const common::Subfield& subfield) {
  folly::dynamic path = folly::dynamic::array;
  for (const auto& element : subfield.path()) {
    folly::dynamic obj = folly::dynamic::object;
    if (const auto* field = element->as<common::Subfield::NestedField>()) {
      obj["kind"] = "field";
      obj["value"] = field->name();
    } else if (
        const auto* subscript =
            element->as<common::Subfield::LongSubscript>()) {
      obj["kind"] = "long";
      obj["value"] = subscript->index();
    } else if (
        const auto* subscript =
            element->as<common::Subfield::StringSubscript>()) {
      obj["kind"] = "string";
      obj["value"] = subscript->index();
    } else {
      VELOX_USER_CHECK(element->is(common::SubfieldKind::kAllSubscripts));
      obj["kind"] = "all";
    }
    path.push_back(std::move(obj));
  }
  return path;
}

common::Subfield deserializeSubfield(const folly::dynamic& obj) {
  VELOX_USER_CHECK(obj.isArray(), "Expected Paimon subfield path");
  std::vector<std::unique_ptr<common::Subfield::PathElement>> path;
  for (const auto& element : obj) {
    const auto kind = element.at("kind").asString();
    if (kind == "field") {
      path.push_back(
          std::make_unique<common::Subfield::NestedField>(
              element.at("value").asString()));
    } else if (kind == "long") {
      path.push_back(
          std::make_unique<common::Subfield::LongSubscript>(
              element.at("value").asInt()));
    } else if (kind == "string") {
      path.push_back(
          std::make_unique<common::Subfield::StringSubscript>(
              element.at("value").asString()));
    } else {
      VELOX_USER_CHECK_EQ(kind, "all", "Unknown Paimon path kind");
      path.push_back(std::make_unique<common::Subfield::AllSubscripts>());
    }
  }
  return common::Subfield(std::move(path));
}

} // namespace facebook::velox::connector::hive::paimon
