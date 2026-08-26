/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <folly/container/F14Set.h>
#include "velox/dwio/common/Range.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/velox/RawSizeContext.h"
#include "velox/vector/BaseVector.h"

#include <optional>

namespace facebook::nimble {

constexpr uint64_t kNullSize = 1;

/// Returns the logical byte size for a fixed-width Velox type.
/// For TIMESTAMP returns 12 bytes (8 + 4) to match DWRF/Nimble convention.
/// Returns std::nullopt for variable-length or complex types.
std::optional<size_t> getTypeSize(const velox::Type& type);

// Get raw size from vector with schema and flatmap node IDs.
// flatMapNodeIds contains the node IDs that are configured as flatmaps.
// A passthrough flatmap is detected when:
// 1. The node's id is in flatMapNodeIds, AND
// 2. The vector at that level is a ROW vector, AND
// 3. The schema type is MAP (not ROW from readSchema conversion)
// ignoreTopLevelNulls: when true, top-level row nulls are ignored (treated as
// non-null) to match FieldWriter behavior when that option is enabled.
// type can be nullptr when schema is not available.
uint64_t getRawSizeFromVector(
    const velox::VectorPtr& vector,
    const velox::common::Ranges& ranges,
    RawSizeContext& context,
    const velox::dwio::common::TypeWithId* type,
    const folly::F14FastSet<uint32_t>& flatMapNodeIds,
    bool ignoreTopLevelNulls = false);

uint64_t getRawSizeFromRowVector(
    const velox::VectorPtr& vector,
    const velox::common::Ranges& ranges,
    RawSizeContext& context,
    const velox::dwio::common::TypeWithId* type,
    const folly::F14FastSet<uint32_t>& flatMapNodeIds,
    bool topLevel = false,
    bool ignoreTopLevelNulls = false);

// Overloads without schema - used when schema is not available.
// These cannot detect passthrough flatmaps.
uint64_t getRawSizeFromVector(
    const velox::VectorPtr& vector,
    const velox::common::Ranges& ranges,
    RawSizeContext& context);

uint64_t getRawSizeFromRowVector(
    const velox::VectorPtr& vector,
    const velox::common::Ranges& ranges,
    RawSizeContext& context,
    bool topLevel = false);

uint64_t getRawSizeFromVector(
    const velox::VectorPtr& vector,
    const velox::common::Ranges& ranges);

} // namespace facebook::nimble
