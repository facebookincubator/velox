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

#include "velox/connectors/hive/ConstantFromString.h"

#include "velox/connectors/hive/PartitionValue.h"

namespace facebook::velox::connector::hive {

VectorPtr newConstantFromString(
    const TypePtr& type,
    const std::optional<std::string>& value,
    velox::memory::MemoryPool* pool,
    bool isLocalTimestamp,
    bool isDaysSinceEpoch,
    const tz::TimeZone* timezone) {
  if (!value.has_value()) {
    return BaseVector::createNullConstant(type, 1, pool);
  }

  return BaseVector::createConstant(
      type,
      PartitionValue::fromString(
          *value,
          *type,
          isLocalTimestamp ? PartitionValue::TimestampMode::kLocalTime
                           : PartitionValue::TimestampMode::kUtc,
          isDaysSinceEpoch ? PartitionValue::DateMode::kDaysSinceEpoch
                           : PartitionValue::DateMode::kIsoString,
          timezone),
      1,
      pool);
}

} // namespace facebook::velox::connector::hive
