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
#include "velox/connectors/hive/iceberg/FilterUtil.h"

namespace facebook::velox::connector::hive::iceberg {

using namespace facebook::velox::exec;
using namespace facebook::velox::core;

std::unique_ptr<common::Filter> createNegatedBytesValuesFilter(
    const VectorPtr& valuesVector,
    vector_size_t offset,
    vector_size_t size) {
  auto [values, hasNull] =
      facebook::velox::common::deDuplicateValues<std::string, StringView>(
          valuesVector, offset, size);

  VELOX_USER_CHECK(
      !values.empty() || hasNull,
      "NOT IN filter must contain at least one non-null value");

  return std::make_unique<common::NegatedBytesValues>(values, !hasNull);
}

template <typename T>
std::unique_ptr<common::Filter> createNegatedBigintValuesFilter(
    const VectorPtr& valuesVector,
    vector_size_t offset,
    vector_size_t size) {
  auto valuesPair =
      common::deDuplicateValues<int64_t, T>(valuesVector, offset, size);

  const auto& values = valuesPair.first;
  bool hasNull = valuesPair.second;

  return common::createNegatedBigintValues(values, !hasNull);
}

std::unique_ptr<common::Filter> createNotInFilter(
    const VectorPtr& elements,
    vector_size_t offset,
    vector_size_t size,
    TypeKind type) {
  std::unique_ptr<common::Filter> filter;
  switch (type) {
    case TypeKind::HUGEINT:
      // TODO: createNegatedHugeintValuesFilter is not implemented yet.
      VELOX_NYI("createNegatedHugeintValuesFilter is not implemented yet");
    case TypeKind::BIGINT:
      filter = createNegatedBigintValuesFilter<int64_t>(elements, offset, size);
      break;
    case TypeKind::INTEGER:
      filter = createNegatedBigintValuesFilter<int32_t>(elements, offset, size);
      break;
    case TypeKind::SMALLINT:
      filter = createNegatedBigintValuesFilter<int16_t>(elements, offset, size);
      break;
    case TypeKind::TINYINT:
      filter = createNegatedBigintValuesFilter<int8_t>(elements, offset, size);
      break;
    case TypeKind::BOOLEAN:
      // Hack: using BIGINT filter for bool, which is essentially "int1_t".
      filter = createNegatedBigintValuesFilter<bool>(elements, offset, size);
      break;
    case TypeKind::TIMESTAMP:
      filter =
          createNegatedBigintValuesFilter<Timestamp>(elements, offset, size);
      break;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      filter = createNegatedBytesValuesFilter(elements, offset, size);
      break;
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::UNKNOWN:
      [[fallthrough]];
    case TypeKind::ARRAY:
      [[fallthrough]];
    case TypeKind::MAP:
      [[fallthrough]];
    case TypeKind::ROW:
      [[fallthrough]];
    default:
      VELOX_USER_FAIL(
          "Iceberg equality delete column cannot be of type ", type);
  }
  return filter;
}
} // namespace facebook::velox::connector::hive::iceberg
