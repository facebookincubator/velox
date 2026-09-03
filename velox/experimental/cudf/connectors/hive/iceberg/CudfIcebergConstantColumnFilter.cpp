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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergConstantColumnFilter.h"

#include "velox/connectors/hive/PartitionValue.h"

#include <folly/Conv.h>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_hive = ::facebook::velox::connector::hive;

bool isDaysSinceEpoch(
    const TypePtr& type,
    const std::optional<std::string>& value) {
  return type->isDate() and value.has_value() and
      folly::tryTo<int32_t>(value.value()).hasValue();
}

ConstantFilterFold foldFilterOnConstant(
    const common::Filter& filter,
    const TypePtr& type,
    const std::optional<std::string>& value,
    bool readTimestampAsLocalTime) {
  if (not value.has_value()) {
    return filter.testNull() ? ConstantFilterFold::kAlwaysTrue
                             : ConstantFilterFold::kAlwaysFalse;
  }

  const auto typedValue = velox_hive::PartitionValue::fromString(
      value.value(),
      *type,
      readTimestampAsLocalTime
          ? velox_hive::PartitionValue::TimestampMode::kLocalTime
          : velox_hive::PartitionValue::TimestampMode::kUtc,
      isDaysSinceEpoch(type, value)
          ? velox_hive::PartitionValue::DateMode::kDaysSinceEpoch
          : velox_hive::PartitionValue::DateMode::kIsoString);

  return common::applyFilter(filter, typedValue)
      ? ConstantFilterFold::kAlwaysTrue
      : ConstantFilterFold::kAlwaysFalse;
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
