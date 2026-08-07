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

#include "velox/connectors/hive/FileSplitReader.h"
#include "velox/vector/SimpleVector.h"

#include <folly/Conv.h>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

template <TypeKind kind>
bool testFilterOnValue(
    const common::Filter& filter,
    const VectorPtr& constant) {
  using T = typename TypeTraits<kind>::NativeType;
  return common::applyFilter(
      filter, constant->as<SimpleVector<T>>()->valueAt(0));
}

} // namespace

VectorPtr makeInjectedConstant(
    const TypePtr& type,
    const std::optional<std::string>& value,
    memory::MemoryPool* pool,
    bool readAsLocalTime) {
  // DATE values arrive in two format-disjoint encodings: Iceberg-native
  // days-since-epoch integers (e.g. "20244") or Hive-migrated date strings
  // (e.g. "2025-06-05"). A bare integer is unambiguously days-since-epoch
  // (date strings contain '-' separators that fail an integer parse).
  const bool isDaysSinceEpoch = type->isDate() and value.has_value() and
      folly::tryTo<int32_t>(value.value()).hasValue();
  return velox::connector::hive::newConstantFromString(
      type, value, pool, readAsLocalTime, isDaysSinceEpoch);
}

ConstantFilterFold foldFilterOnConstant(
    const common::Filter& filter,
    const TypePtr& type,
    const std::optional<std::string>& value,
    memory::MemoryPool* pool,
    bool readAsLocalTime) {
  VectorPtr constant;
  try {
    constant = makeInjectedConstant(type, value, pool, readAsLocalTime);
  } catch (const VeloxException&) {
    return ConstantFilterFold::kUnknown;
  }

  if (constant->isNullAt(0)) {
    return filter.testNull() ? ConstantFilterFold::kAlwaysTrue
                             : ConstantFilterFold::kAlwaysFalse;
  }

  try {
    const bool accepted = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
        testFilterOnValue, type->kind(), filter, constant);
    return accepted ? ConstantFilterFold::kAlwaysTrue
                    : ConstantFilterFold::kAlwaysFalse;
  } catch (const VeloxException&) {
    // The type is not one 'applyFilter' can test against.
    return ConstantFilterFold::kUnknown;
  }
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
