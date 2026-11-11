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

#include "velox/functions/prestosql/aggregates/VectorIndexOptionsArgument.h"

#include "velox/common/base/Exceptions.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::aggregate::prestosql {

// static
std::vector<std::pair<std::string_view, std::string_view>>
VectorIndexOptionsArgument::read(const VectorPtr& options) {
  VELOX_USER_CHECK_NOT_NULL(options, "Options must be a constant");
  VELOX_USER_CHECK(!options->isNullAt(0), "Options must not be null");
  DecodedVector decodedOptions(*options);
  const auto* map = decodedOptions.base()->asChecked<MapVector>();
  const auto row = decodedOptions.index(0);
  std::vector<std::pair<std::string_view, std::string_view>> pairs;
  if (map->sizeAt(row) > 0) {
    const auto* keys = map->mapKeys()->asChecked<SimpleVector<StringView>>();
    const auto* values =
        map->mapValues()->asChecked<SimpleVector<StringView>>();
    for (auto i = map->offsetAt(row); i < map->offsetAt(row) + map->sizeAt(row);
         ++i) {
      VELOX_USER_CHECK(!values->isNullAt(i), "Option values must not be null");
      pairs.emplace_back(keys->valueAt(i), values->valueAt(i));
    }
  }
  return pairs;
}

} // namespace facebook::velox::aggregate::prestosql
