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

#include "velox/connectors/hive/HivePartitionUtil.h"
#include "velox/common/encode/Base64.h"
#include "velox/type/DecimalUtil.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::connector::hive {

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

template <TypeKind Kind>
std::pair<std::string, std::string> makePartitionKeyValueString(
    const HivePartitionUtilPtr& formatter,
    const BaseVector* partitionVector,
    vector_size_t row,
    const std::string& name,
    const TypePtr& type) {
  using T = typename TypeTraits<Kind>::NativeType;
  if (partitionVector->as<SimpleVector<T>>()->isNullAt(row)) {
    return std::make_pair(name, "");
  }

  return std::make_pair(
      name,
      formatter->toPartitionString(
          partitionVector->as<SimpleVector<T>>()->valueAt(row), type));
}

} // namespace

std::string HivePartitionUtil::toPartitionString(
    int32_t value,
    const TypePtr& type) const {
  if (type->isDate()) {
    return formatDate(value);
  }
  return folly::to<std::string>(value);
}

std::string HivePartitionUtil::toPartitionString(
    int64_t value,
    const TypePtr& type) const {
  if (type->isShortDecimal()) {
    return formatDecimal(value, type);
  }
  return folly::to<std::string>(value);
}

std::string HivePartitionUtil::toPartitionString(
    int128_t value,
    const TypePtr& type) const {
  if (type->isLongDecimal()) {
    return formatDecimal(value, type);
  }
  return folly::to<std::string>(value);
}

std::string HivePartitionUtil::toPartitionString(
    Timestamp value,
    const TypePtr& type) const {
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

#define PARTITION_TYPE_DISPATCH(TEMPLATE_FUNC, typeKind, ...)                  \
  [&]() {                                                                      \
    switch (typeKind) {                                                        \
      case TypeKind::BOOLEAN:                                                  \
      case TypeKind::TINYINT:                                                  \
      case TypeKind::SMALLINT:                                                 \
      case TypeKind::INTEGER:                                                  \
      case TypeKind::BIGINT:                                                   \
      case TypeKind::VARCHAR:                                                  \
      case TypeKind::VARBINARY:                                                \
      case TypeKind::TIMESTAMP:                                                \
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(                             \
            TEMPLATE_FUNC, typeKind, __VA_ARGS__);                             \
      default:                                                                 \
        VELOX_UNSUPPORTED(                                                     \
            "Unsupported partition type: {}", TypeKindName::toName(typeKind)); \
    }                                                                          \
  }()

std::vector<std::pair<std::string, std::string>>
HivePartitionUtil::extractPartitionKeyValues(
    const RowVectorPtr& partitionsVector,
    vector_size_t row) {
  const auto& formatter = std::make_shared<HivePartitionUtil>();
  std::vector<std::pair<std::string, std::string>> partitionKeyValues;
  for (auto i = 0; i < partitionsVector->childrenSize(); i++) {
    partitionKeyValues.push_back(PARTITION_TYPE_DISPATCH(
        makePartitionKeyValueString,
        partitionsVector->childAt(i)->typeKind(),
        formatter,
        partitionsVector->childAt(i)->loadedVector(),
        row,
        asRowType(partitionsVector->type())->nameOf(i),
        partitionsVector->childAt(i)->type()));
  }
  return partitionKeyValues;
}

} // namespace facebook::velox::connector::hive
