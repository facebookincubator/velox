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

#include "velox/dwio/common/IcebergStatistics.h"

namespace facebook::velox::dwio::common {

folly::dynamic IcebergDataFileStatistics::toJson() const {
  folly::dynamic json = folly::dynamic::object;
  json["recordCount"] = numRecords;

  auto mapToJson = [](const auto& map) {
    folly::dynamic result = folly::dynamic::object;
    for (const auto& pair : map) {
      result[folly::to<std::string>(pair.first)] = pair.second;
    }
    return result;
  };

  json["columnSizes"] = mapToJson(columnsSizes);
  json["valueCounts"] = mapToJson(valueCounts);
  json["nullValueCounts"] = mapToJson(nullValueCounts);
  json["nanValueCounts"] = mapToJson(nanValueCounts);
  json["lowerBounds"] = mapToJson(lowerBounds);
  json["upperBounds"] = mapToJson(upperBounds);

  return json;
}

} // namespace facebook::velox::dwio::common
