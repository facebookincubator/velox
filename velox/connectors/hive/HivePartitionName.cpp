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
#include "velox/dwio/catalog/fbhive/FileUtils.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::connector::hive {

using namespace facebook::velox::dwio::catalog::fbhive;

namespace {

template <typename T>
std::string formatDecimal(T value, const TypePtr& type) {
  const auto& [p, s] = getDecimalPrecisionScale(*type);
  const auto& maxSize = DecimalUtil::maxStringViewSize(p, s);
  std::string buffer(maxSize, '\0');
  const auto& actualSize =
      DecimalUtil::castToString(value, s, maxSize, buffer.data());
  buffer.resize(actualSize);
  return buffer;
}

} // namespace

std::string HivePartitionName::toName(int32_t value, const TypePtr& type) {
  if (type->isDate()) {
    return DateType::toIso8601(value);
  }
  return fmt::to_string(value);
}

std::string HivePartitionName::toName(int64_t value, const TypePtr& type) {
  if (type->isShortDecimal()) {
    return formatDecimal(value, type);
  }
  return fmt::to_string(value);
}

std::string HivePartitionName::toName(int128_t value, const TypePtr& type) {
  if (type->isLongDecimal()) {
    return formatDecimal(value, type);
  }
  return fmt::to_string(value);
}

std::string HivePartitionName::toName(Timestamp value, const TypePtr& type) {
  value.toTimezone(Timestamp::defaultTimezone());
  TimestampToStringOptions options;
  options.dateTimeSeparator = ' ';
  // Set the precision to milliseconds, and enable the skipTrailingZeros match
  // the timestamp precision and truncation behavior of Presto.
  options.precision = TimestampPrecision::kMilliseconds;
  options.skipTrailingZeros = true;

  auto result = value.toString(options);

  // Presto's java.sql.Timestamp.toString() always keeps at least one decimal
  // place even when all fractional seconds are zero.
  // If skipTrailingZeros removed all fractional digits, add back ".0" to match
  // Presto's behavior.
  if (auto dotPos = result.find_last_of('.'); dotPos == std::string::npos) {
    // No decimal point found, add ".0"
    result += ".0";
  }

  return result;
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
