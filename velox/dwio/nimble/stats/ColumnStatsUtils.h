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
#include <cstddef>
#include <optional>
#include <unordered_map>

#include "velox/dwio/nimble/common/OrderedRanges.h"
#include "velox/dwio/nimble/common/SchemaBuilder.h"
#include "velox/dwio/nimble/common/SchemaTypes.h"
#include "velox/dwio/nimble/common/StatsUtil.h"
#include "velox/dwio/nimble/writer/RawSizeContext.h"
#include "velox/vector/BaseVector.h"

namespace facebook::nimble {

struct ColumnStats {
  uint64_t logicalSize{0};
  std::optional<uint64_t> dedupedLogicalSize{std::nullopt};
  uint64_t physicalSize{0};
  uint64_t nullCount{0};
  uint64_t valueCount{0};
};

using OrderedRanges = range_helper::OrderedRanges<velox::vector_size_t>;

uint64_t getRawSizeFromVector(
    const velox::VectorPtr& vector,
    const OrderedRanges& ranges,
    RawSizeContext& context);

uint64_t getRawSizeFromVector(
    const velox::VectorPtr& vector,
    const OrderedRanges& ranges);

void aggregateStats(
    const TypeBuilder& builder,
    std::unordered_map<offset_size, ColumnStats>& columnStats,
    std::optional<offset_size> parentOffset = std::nullopt);
} // namespace facebook::nimble
