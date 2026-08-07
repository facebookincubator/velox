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

#pragma once

#include "velox/common/memory/Memory.h"
#include "velox/type/Filter.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

#include <optional>
#include <string>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// Result of folding a subfield filter against the constant value an injected
/// column is materialized with.
enum class ConstantFilterFold {
  /// The filter accepts the constant, so the predicate holds for every row of
  /// the split.
  kAlwaysTrue,
  /// The filter rejects the constant, so the predicate holds for no row of the
  /// split.
  kAlwaysFalse,
  /// The constant could not be built or its type is not one filters can be
  /// tested against. Callers must keep evaluating the filter over the
  /// materialized column.
  kUnknown,
};

/// Builds the size-1 constant vector an injected column is materialized with.
///
/// @param type Velox type of the column.
/// @param value String form of the value, `nullopt` for a typed NULL.
/// @param readAsLocalTime Whether a TIMESTAMP value is a local time that must
/// be converted to GMT.
VectorPtr makeInjectedConstant(
    const TypePtr& type,
    const std::optional<std::string>& value,
    memory::MemoryPool* pool,
    bool readAsLocalTime);

/// Folds 'filter' against the constant an injected column is materialized with.
///
/// Never throws: a value that cannot be converted yields `kUnknown` and is
/// reported later, when the column is actually materialized.
ConstantFilterFold foldFilterOnConstant(
    const common::Filter& filter,
    const TypePtr& type,
    const std::optional<std::string>& value,
    memory::MemoryPool* pool,
    bool readAsLocalTime);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
