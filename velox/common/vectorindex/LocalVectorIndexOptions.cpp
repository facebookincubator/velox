/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/common/vectorindex/LocalVectorIndexOptions.h"

#include <folly/Conv.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::vectorindex {

namespace {

bool isIvf(VectorIndexType type) {
  return type == VectorIndexType::kIvfFlat || type == VectorIndexType::kIvfSq;
}

void validateSetting(
    const std::optional<int32_t>& value,
    std::string_view key,
    bool applies,
    VectorIndexType type) {
  if (!value.has_value()) {
    return;
  }
  VELOX_USER_CHECK(
      applies,
      "Vector index option does not apply to the index type. Option: {}. Index type: {}",
      key,
      VectorIndexTypeName::toName(type));
  VELOX_USER_CHECK_GT(
      *value, 0, "Vector index option must be positive: {}", key);
}

// Parses 'options' without checking whether they apply to an index type.
// With 'searchOnly', accepts only search settings.
LocalVectorIndexOptions parseOptions(
    const std::vector<std::pair<std::string_view, std::string_view>>& options,
    bool searchOnly) {
  using Options = LocalVectorIndexOptions;
  Options result;
  for (const auto& [key, value] : options) {
    std::optional<int32_t>* setting;
    if (key == Options::kNlist) {
      setting = &result.nlist;
    } else if (key == Options::kNprobe) {
      setting = &result.nprobe;
    } else if (key == Options::kM) {
      setting = &result.m;
    } else if (key == Options::kEfConstruction) {
      setting = &result.efConstruction;
    } else if (key == Options::kEfSearch) {
      setting = &result.efSearch;
    } else {
      VELOX_USER_FAIL("Unknown vector index option: {}", key);
    }
    VELOX_USER_CHECK(
        !searchOnly || setting == &result.nprobe || setting == &result.efSearch,
        "Vector index option is not a search setting: {}",
        key);
    const auto parsed = folly::tryTo<int32_t>(value);
    VELOX_USER_CHECK(
        parsed.hasValue(),
        "Vector index option must be an integer: {}={}",
        key,
        value);
    *setting = parsed.value();
  }
  return result;
}

} // namespace

LocalVectorIndexOptions LocalVectorIndexOptions::parse(
    VectorIndexType type,
    const std::vector<std::pair<std::string_view, std::string_view>>& options) {
  auto result = parseOptions(options, /*searchOnly=*/false);
  result.validate(type);
  return result;
}

LocalVectorIndexOptions LocalVectorIndexOptions::parseSearch(
    const std::vector<std::pair<std::string_view, std::string_view>>& options) {
  auto result = parseOptions(options, /*searchOnly=*/true);
  for (const auto& [setting, key] :
       {std::pair{result.nprobe, kNprobe},
        std::pair{result.efSearch, kEfSearch}}) {
    VELOX_USER_CHECK(
        !setting.has_value() || *setting > 0,
        "Vector index option must be positive: {}",
        key);
  }
  return result;
}

void LocalVectorIndexOptions::validate(VectorIndexType type) const {
  validateSetting(nlist, kNlist, isIvf(type), type);
  validateSetting(nprobe, kNprobe, isIvf(type), type);

  const bool graph =
      type == VectorIndexType::kHnswSq || type == VectorIndexType::kCagra;
  validateSetting(m, kM, graph, type);
  if (m.has_value()) {
    VELOX_USER_CHECK_GE(*m, kMinM, "Vector index option is too small: {}", kM);
    VELOX_USER_CHECK_LE(*m, kMaxM, "Vector index option is too large: {}", kM);
  }

  // efConstruction is the HNSW build's candidate count. A CAGRA graph is built
  // by NN-descent, so that setting does not apply to it.
  validateSetting(
      efConstruction, kEfConstruction, type == VectorIndexType::kHnswSq, type);
  validateSetting(efSearch, kEfSearch, graph, type);
}

} // namespace facebook::velox::common::vectorindex
