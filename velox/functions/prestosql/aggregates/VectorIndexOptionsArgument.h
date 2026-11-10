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

#include <string_view>
#include <utility>
#include <vector>

#include "velox/vector/BaseVector.h"

namespace facebook::velox::aggregate::prestosql {

/// The options argument of the vector index aggregates: a constant
/// map(varchar, varchar) of settings by key.
class VectorIndexOptionsArgument {
 public:
  /// Returns the key-value pairs of 'options', which refer to its strings.
  /// Throws a user error if 'options' is not a constant, is null or has a null
  /// value.
  static std::vector<std::pair<std::string_view, std::string_view>> read(
      const VectorPtr& options);
};

} // namespace facebook::velox::aggregate::prestosql
