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

#include "velox/connectors/hive/iceberg/IcebergDataFileStatistics.h"

namespace facebook::velox::connector::hive::iceberg {

folly::dynamic IcebergDataFileStatistics::toJson() const {
  folly::dynamic json = folly::dynamic::object;
  json["recordCount"] = numRecords;

  folly::dynamic columnSizes = folly::dynamic::object;
  folly::dynamic valueCounts = folly::dynamic::object;
  folly::dynamic nullValueCounts = folly::dynamic::object;
  folly::dynamic nanValueCounts = folly::dynamic::object;
  folly::dynamic lowerBounds = folly::dynamic::object;
  folly::dynamic upperBounds = folly::dynamic::object;

  for (const auto& [fieldId, stats] : columnStats) {
    auto fieldIdStr = folly::to<std::string>(fieldId);
    columnSizes[fieldIdStr] = stats.columnSize;
    valueCounts[fieldIdStr] = stats.valueCount;
    nullValueCounts[fieldIdStr] = stats.nullValueCount;
    if (stats.nanValueCount.has_value()) {
      nanValueCounts[fieldIdStr] = stats.nanValueCount.value();
    }
    if (stats.lowerBound.has_value()) {
      lowerBounds[fieldIdStr] = stats.lowerBound.value();
    }
    if (stats.upperBound.has_value()) {
      upperBounds[fieldIdStr] = stats.upperBound.value();
    }
  }

  json["columnSizes"] = std::move(columnSizes);
  json["valueCounts"] = std::move(valueCounts);
  json["nullValueCounts"] = std::move(nullValueCounts);
  json["nanValueCounts"] = std::move(nanValueCounts);
  json["lowerBounds"] = std::move(lowerBounds);
  json["upperBounds"] = std::move(upperBounds);

  return json;
}

} // namespace facebook::velox::connector::hive::iceberg
