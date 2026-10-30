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

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/common/vectorindex/VectorIndexType.h"

namespace facebook::velox::common::vectorindex {

/// Settings of a local index. A vector index's options give them as string
/// keys and values. Unset settings take defaults that depend on the number of
/// embeddings (see LocalVectorIndex::build). Search settings are recorded in
/// the local index, where a search reads them as defaults.
struct LocalVectorIndexOptions {
  /// Keys of the options.
  static constexpr std::string_view kNlist{"ivf.nlist"};
  static constexpr std::string_view kNprobe{"ivf.nprobe"};
  static constexpr std::string_view kM{"hnsw.M"};
  static constexpr std::string_view kEfConstruction{"hnsw.efConstruction"};
  static constexpr std::string_view kEfSearch{"hnsw.efSearch"};

  /// The range of 'hnsw.M'. FAISS needs at least 2 neighbors per embedding to
  /// lay out the graph's levels.
  static constexpr int32_t kMinM{2};
  static constexpr int32_t kMaxM{256};

  /// The number of lists of an IVF index.
  std::optional<int32_t> nlist{};
  /// The number of lists an IVF search scans.
  std::optional<int32_t> nprobe{};
  /// The number of neighbors of each embedding in an HNSW graph.
  std::optional<int32_t> m{};
  /// The number of candidates an HNSW build explores.
  std::optional<int32_t> efConstruction{};
  /// The number of candidates an HNSW search explores.
  std::optional<int32_t> efSearch{};

  bool operator==(const LocalVectorIndexOptions&) const = default;

  /// Parses options, a list of key-value pairs, for an index of 'type'.
  /// Throws a user error for an unknown key or a value that is not an
  /// integer, and the errors of validate().
  static LocalVectorIndexOptions parse(
      VectorIndexType type,
      const std::vector<std::pair<std::string_view, std::string_view>>&
          options);

  /// Parses search settings, a list of key-value pairs that may set only
  /// 'ivf.nprobe' and 'hnsw.efSearch'. Throws a user error for any other key
  /// or a value that is not a positive integer. Whether a setting applies to
  /// an index depends on the index's type (see validate()).
  static LocalVectorIndexOptions parseSearch(
      const std::vector<std::pair<std::string_view, std::string_view>>&
          options);

  /// Throws a user error for a setting that does not apply to 'type', a value
  /// that is not positive or an 'hnsw.M' outside [kMinM, kMaxM].
  void validate(VectorIndexType type) const;
};

} // namespace facebook::velox::common::vectorindex
