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
#include "velox/dwio/nimble/velox/stats/ColumnStatsUtils.h"
#include "velox/dwio/common/Range.h"
#include "velox/dwio/nimble/velox/RawSizeUtils.h"

namespace facebook::nimble {

namespace {
velox::common::Ranges toVeloxRanges(const OrderedRanges& ranges) {
  velox::common::Ranges veloxRanges;
  for (auto& range : ranges.ranges()) {
    veloxRanges.add(
        std::get<0>(range), std::get<0>(range) + std::get<1>(range));
  }
  return veloxRanges;
}
} // namespace

uint64_t getRawSizeFromVector(
    const velox::VectorPtr& vector,
    const OrderedRanges& ranges,
    RawSizeContext& context) {
  return getRawSizeFromVector(vector, toVeloxRanges(ranges), context);
}

uint64_t getRawSizeFromVector(
    const velox::VectorPtr& vector,
    const OrderedRanges& ranges) {
  return getRawSizeFromVector(vector, toVeloxRanges(ranges));
}

void aggregateStats(
    const TypeBuilder& builder,
    std::unordered_map<offset_size, ColumnStats>& columnStats,
    std::optional<offset_size> parentOffset) {
  auto updateStats = [&columnStats](auto sourceOffset, auto targetOffset) {
    const auto& sourceStats = columnStats[sourceOffset];
    auto& targetStats = columnStats[targetOffset];
    // Fetch deduplicated logical size from children. Non-deduplicated logical
    // size should already be set in the FieldWriter.
    if (targetStats.dedupedLogicalSize.has_value()) {
      targetStats.dedupedLogicalSize.value() +=
          sourceStats.dedupedLogicalSize.value_or(sourceStats.logicalSize);
    } else {
      targetStats.logicalSize += sourceStats.logicalSize;
    }
    targetStats.physicalSize += sourceStats.physicalSize;
  };

  std::optional<offset_size> offset = std::nullopt;
  switch (builder.kind()) {
    case Kind::Row: {
      const auto& row = builder.asRow();
      offset = row.nullsDescriptor().offset();
      for (auto i = 0; i < row.childrenCount(); ++i) {
        aggregateStats(row.childAt(i), columnStats, offset);
      }
      break;
    }
    case Kind::Array: {
      const auto& array = builder.asArray();
      offset = array.lengthsDescriptor().offset();
      aggregateStats(array.elements(), columnStats, offset);
      break;
    }
    case Kind::ArrayWithOffsets: {
      const auto& arrayWithOffsets = builder.asArrayWithOffsets();
      offset = arrayWithOffsets.lengthsDescriptor().offset();
      columnStats[offset.value()].dedupedLogicalSize =
          columnStats[offset.value()].nullCount * kNullSize;
      updateStats(
          arrayWithOffsets.offsetsDescriptor().offset(), offset.value());
      aggregateStats(arrayWithOffsets.elements(), columnStats, offset);
      break;
    }
    case Kind::Map: {
      const auto& map = builder.asMap();
      offset = map.lengthsDescriptor().offset();
      aggregateStats(map.keys(), columnStats, offset);
      aggregateStats(map.values(), columnStats, offset);
      break;
    }
    case Kind::FlatMap: {
      const auto& flatMap = builder.asFlatMap();
      offset = flatMap.nullsDescriptor().offset();
      // The stats for offset contains keys and nulls logical size.
      for (auto i = 0; i < flatMap.childrenCount(); ++i) {
        updateStats(flatMap.inMapDescriptorAt(i).offset(), offset.value());
        aggregateStats(flatMap.childAt(i), columnStats, offset);
      }
      break;
    }
    case Kind::SlidingWindowMap: {
      const auto& slidingWindowMap = builder.asSlidingWindowMap();
      offset = slidingWindowMap.lengthsDescriptor().offset();
      columnStats[offset.value()].dedupedLogicalSize =
          columnStats[offset.value()].nullCount * kNullSize;
      updateStats(
          slidingWindowMap.offsetsDescriptor().offset(), offset.value());
      aggregateStats(slidingWindowMap.keys(), columnStats, offset);
      aggregateStats(slidingWindowMap.values(), columnStats, offset);
      break;
    }
    case Kind::Scalar: {
      const auto& scalar = builder.asScalar();
      offset = scalar.scalarDescriptor().offset();
      break;
    }
    case Kind::TimestampMicroNano: {
      const auto& timestamp = builder.asTimestampMicroNano();
      offset = timestamp.microsDescriptor().offset();
      updateStats(timestamp.nanosDescriptor().offset(), offset.value());
      break;
    }
    default:
      NIMBLE_UNSUPPORTED("Unsupported type: {}.", toString(builder.kind()));
  }

  NIMBLE_DCHECK(offset.has_value(), "Offset should always be set.");
  if (parentOffset.has_value()) {
    updateStats(offset.value(), parentOffset.value());
  }
}
} // namespace facebook::nimble
