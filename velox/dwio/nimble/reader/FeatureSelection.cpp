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
#include "velox/dwio/nimble/reader/FeatureSelection.h"

#include <fmt/format.h>
#include <folly/json/json.h>

#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/common/exception/Exception.h"

namespace facebook::nimble {

namespace {

void populateFeatureSelectorForFlatMapAsStruct(
    const velox::dwio::common::ColumnSelector& selector,
    const std::unordered_map<uint32_t, std::vector<std::string>>& asStructMap,
    nimble::BatchReadParams& params) {
  // asStructMap is using TypeWithId identifiers as the map key.
  // Nimble understands column names. Therefore, we perform id to name
  // conversion here.
  const auto typeWithId =
      velox::dwio::common::TypeWithId::create(selector.getSchema());
  const auto& names = typeWithId->type()->as<velox::TypeKind::ROW>().names();
  std::unordered_map<uint32_t, std::string_view> lookup;
  for (auto i = 0; i < names.size(); ++i) {
    lookup[typeWithId->childAt(i)->id()] = names[i];
  }
  for (const auto& pair : asStructMap) {
    auto it = lookup.find(pair.first);
    DWIO_ENSURE(
        it != lookup.end(), "Unable to resolve column name from node id.");
    params.readFlatMapFieldAsStruct.emplace(it->second);
    auto& featureSelector = params.flatMapFeatureSelector[it->second];
    featureSelector.mode = nimble::SelectionMode::Include;
    featureSelector.features.reserve(pair.second.size());
    for (const auto& feature : pair.second) {
      featureSelector.features.push_back(feature);
    }
  }
}

void populateOtherFeatureSelector(
    const velox::dwio::common::ColumnSelector& selector,
    nimble::BatchReadParams& params) {
  if (selector.getProjection().empty()) {
    return;
  }
  constexpr char rejectPrefix = '!';
  for (const auto& filterNode : selector.getProjection()) {
    if (filterNode.expression.empty() ||
        params.readFlatMapFieldAsStruct.contains(filterNode.name)) {
      continue;
    }
    auto expressionJson = folly::parseJson(filterNode.expression);
    if (expressionJson.empty()) {
      continue;
    }
    auto& featureSelector = params.flatMapFeatureSelector[filterNode.name];
    featureSelector.features.reserve(expressionJson.size());
    auto exp = expressionJson[0].asString();
    DWIO_ENSURE(
        !exp.empty(),
        fmt::format("First feature in {} is empty", filterNode.name));
    bool isInclude = exp.front() != rejectPrefix;
    if (isInclude) {
      featureSelector.mode = nimble::SelectionMode::Include;
    } else {
      featureSelector.mode = nimble::SelectionMode::Exclude;
    }
    for (auto itor = expressionJson.begin(); itor != expressionJson.end();
         ++itor) {
      auto feature = itor->asString();
      DWIO_ENSURE(
          !feature.empty() && (feature.front() != rejectPrefix) == isInclude,
          fmt::format(
              "Empty or mixed included/excluded feature '{}' found "
              "for flatmap: {}.",
              feature,
              filterNode.name));
      featureSelector.features.emplace_back(
          isInclude ? feature : feature.substr(1));
    }
  }
}

} // namespace

void populateFeatureSelector(
    const velox::dwio::common::ColumnSelector& selector,
    const std::unordered_map<uint32_t, std::vector<std::string>>& asStructMap,
    nimble::BatchReadParams& params) {
  if (!asStructMap.empty()) {
    populateFeatureSelectorForFlatMapAsStruct(selector, asStructMap, params);
  }
  populateOtherFeatureSelector(selector, params);
}

} // namespace facebook::nimble
