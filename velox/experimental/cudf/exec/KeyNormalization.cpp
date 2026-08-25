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

#include "velox/experimental/cudf/exec/KeyNormalization.h"

#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace facebook::velox::cudf_velox {
namespace {

cudf::data_type int64Type() {
  return cudf::data_type{cudf::type_id::INT64};
}

// Clears the zone key of one packed column. Mirrors tswtzZoneKey in
// expression/TimestampWithTimeZoneColumn.cpp, which ANDs with kTimezoneMask to
// extract the zone; this ANDs with its complement to drop it.
//
// binary_operation propagates the input's null mask, so a null row stays null
// rather than masking to a value -- which matters because a null key must
// remain distinct from a real one under cudf::null_equality::UNEQUAL.
std::unique_ptr<cudf::column> clearZoneKey(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const cudf::numeric_scalar<int64_t> mask(
      ~static_cast<int64_t>(kTimezoneMask), true, stream);
  return cudf::binary_operation(
      packed,
      mask,
      cudf::binary_operator::BITWISE_AND,
      int64Type(),
      stream,
      mr);
}

} // namespace

NormalizedKeys normalizeKeyColumns(
    cudf::table_view keys,
    const std::vector<bool>& isTswtz,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(
      static_cast<size_t>(keys.num_columns()),
      isTswtz.size(),
      "One normalization flag is required per key column");

  NormalizedKeys result;
  // Nothing to do is the common case, so return the caller's view untouched
  // rather than rebuilding an identical one.
  if (std::none_of(isTswtz.begin(), isTswtz.end(), [](bool b) { return b; })) {
    result.view = keys;
    return result;
  }

  std::vector<cudf::column_view> columns;
  columns.reserve(keys.num_columns());
  // Reserve so the owned columns never reallocate: `columns` holds views into
  // them, and a reallocation would leave those views dangling.
  result.owned.reserve(keys.num_columns());

  for (auto i = 0; i < keys.num_columns(); ++i) {
    if (!isTswtz[i]) {
      columns.push_back(keys.column(i));
      continue;
    }
    VELOX_CHECK_EQ(
        static_cast<int>(keys.column(i).type().id()),
        static_cast<int>(cudf::type_id::INT64),
        "A TIMESTAMP WITH TIME ZONE key column must be physically INT64");
    result.owned.push_back(clearZoneKey(keys.column(i), stream, mr));
    columns.push_back(result.owned.back()->view());
  }

  result.view = cudf::table_view{columns};
  return result;
}

NormalizedKeys normalizeKeyColumns(
    cudf::table_view keys,
    const RowTypePtr& rowType,
    const std::vector<column_index_t>& keyChannels,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(
      static_cast<size_t>(keys.num_columns()),
      keyChannels.size(),
      "One key channel is required per key column");

  std::vector<bool> isTswtz;
  isTswtz.reserve(keyChannels.size());
  for (const auto channel : keyChannels) {
    isTswtz.push_back(isTimestampWithTimeZoneType(rowType->childAt(channel)));
  }
  return normalizeKeyColumns(keys, isTswtz, stream, mr);
}

bool anyKeyNeedsNormalization(
    const RowTypePtr& rowType,
    const std::vector<column_index_t>& keyChannels) {
  for (const auto channel : keyChannels) {
    if (isTimestampWithTimeZoneType(rowType->childAt(channel))) {
      return true;
    }
  }
  return false;
}

} // namespace facebook::velox::cudf_velox
