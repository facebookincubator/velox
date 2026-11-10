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
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::velox::common::vectorindex {

/// Settings of the training of a vector index's codec. A vector index's codec
/// options give them as string keys and values.
struct CodecOptions {
  /// Keys of the options.
  static constexpr std::string_view kMaxFlatClusters{"train.maxFlatClusters"};

  /// The most clusters trained in one level. More train in two levels (see
  /// FaissLocalVectorIndex::trainCentroids).
  int64_t maxFlatClusters{1'000};

  bool operator==(const CodecOptions&) const = default;

  /// Parses options, a list of key-value pairs. Throws a user error for an
  /// unknown key or a value that is not a positive integer.
  static CodecOptions parse(
      const std::vector<std::pair<std::string_view, std::string_view>>&
          options);
};

} // namespace facebook::velox::common::vectorindex
