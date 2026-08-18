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

#include "velox/connectors/hive/HivePartitionName.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/PartitionValue.h"
#include "velox/dwio/catalog/fbhive/FileUtils.h"

namespace facebook::velox::connector::hive {

using namespace facebook::velox::dwio::catalog::fbhive;

namespace {

// The writer names every partition in the default timezone, with ISO dates.
std::string toPartitionString(const Variant& value, const TypePtr& type) {
  return PartitionValue::toString(
      value,
      *type,
      PartitionValue::TimestampMode::kLocalTime,
      PartitionValue::DateMode::kIsoString);
}

} // namespace

std::string HivePartitionName::toName(int32_t value, const TypePtr& type) {
  return toPartitionString(Variant(value), type);
}

std::string HivePartitionName::toName(int64_t value, const TypePtr& type) {
  return toPartitionString(Variant(value), type);
}

std::string HivePartitionName::toName(int128_t value, const TypePtr& type) {
  return toPartitionString(Variant(value), type);
}

std::string HivePartitionName::toName(Timestamp value, const TypePtr& type) {
  return toPartitionString(Variant(value), type);
}

std::string HivePartitionName::partitionName(
    uint32_t partitionId,
    const RowVectorPtr& partitionValues,
    bool partitionKeyAsLowerCase) {
  auto toPartitionName =
      [](auto value, const TypePtr& type, int /*columnIndex*/) {
        return HivePartitionName::toName(value, type);
      };
  return FileUtils::makePartName(
      partitionKeyValues(
          partitionId,
          partitionValues,
          /*nullValueString=*/"",
          toPartitionName),
      partitionKeyAsLowerCase);
}

} // namespace facebook::velox::connector::hive
