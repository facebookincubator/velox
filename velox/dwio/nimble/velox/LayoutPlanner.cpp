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

// Plans the physical order of streams in a stripe. Readers address streams by
// id, so changing this order affects locality rather than file semantics.
//
//   column names --------> resolved top-level columns ----+
//                                                       |
//   FlatMap features ----> validated feature orders -----+--> priority offsets
//                                                              |
//   schema tree ----------> schema-order offsets ---------------+---> layout
//                                                                    root
//                                                                    priority
//                                                                    remainder
//
// Streams already emitted by a priority offset are removed before the schema
// remainder is appended. Empty configuration therefore preserves schema order.

#include "velox/dwio/nimble/velox/LayoutPlanner.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>

#include "folly/container/F14Map.h"
#include "velox/common/Casts.h"

namespace facebook::nimble {

namespace {

void appendAllNestedStreams(
    const TypeBuilder& type,
    std::vector<offset_size>& childrenOffsets) {
  switch (type.kind()) {
    case Kind::Scalar: {
      childrenOffsets.push_back(type.asScalar().scalarDescriptor().offset());
      break;
    }
    case Kind::TimestampMicroNano: {
      auto& timestamp = type.asTimestampMicroNano();
      childrenOffsets.push_back(timestamp.microsDescriptor().offset());
      childrenOffsets.push_back(timestamp.nanosDescriptor().offset());
      break;
    }
    case Kind::Row: {
      auto& row = type.asRow();
      childrenOffsets.push_back(row.nullsDescriptor().offset());
      for (auto i = 0; i < row.childrenCount(); ++i) {
        appendAllNestedStreams(row.childAt(i), childrenOffsets);
      }
      break;
    }
    case Kind::Array: {
      auto& array = type.asArray();
      childrenOffsets.push_back(array.lengthsDescriptor().offset());
      appendAllNestedStreams(array.elements(), childrenOffsets);
      break;
    }
    case Kind::ArrayWithOffsets: {
      auto& arrayWithOffsets = type.asArrayWithOffsets();
      childrenOffsets.push_back(arrayWithOffsets.offsetsDescriptor().offset());
      childrenOffsets.push_back(arrayWithOffsets.lengthsDescriptor().offset());
      appendAllNestedStreams(arrayWithOffsets.elements(), childrenOffsets);
      break;
    }
    case Kind::Map: {
      auto& map = type.asMap();
      childrenOffsets.push_back(map.lengthsDescriptor().offset());
      appendAllNestedStreams(map.keys(), childrenOffsets);
      appendAllNestedStreams(map.values(), childrenOffsets);
      break;
    }
    case Kind::SlidingWindowMap: {
      auto& map = type.asSlidingWindowMap();
      childrenOffsets.push_back(map.offsetsDescriptor().offset());
      childrenOffsets.push_back(map.lengthsDescriptor().offset());
      appendAllNestedStreams(map.keys(), childrenOffsets);
      appendAllNestedStreams(map.values(), childrenOffsets);
      break;
    }
    case Kind::FlatMap: {
      auto& flatMap = type.asFlatMap();
      childrenOffsets.push_back(flatMap.nullsDescriptor().offset());
      for (auto i = 0; i < flatMap.childrenCount(); ++i) {
        childrenOffsets.push_back(flatMap.inMapDescriptorAt(i).offset());
        appendAllNestedStreams(flatMap.childAt(i), childrenOffsets);
      }
      break;
    }
    case Kind::HybridFlatMap: {
      const auto& hybridMap = type.asHybridFlatMap();
      childrenOffsets.push_back(hybridMap.nullsDescriptor().offset());
      for (size_t i = 0; i < hybridMap.groupCount(); ++i) {
        const auto groupStreams = hybridMap.groupAt(i);
        childrenOffsets.push_back(groupStreams.keyPresenceDescriptor.offset());
        childrenOffsets.push_back(groupStreams.inMapDescriptor.offset());
        appendAllNestedStreams(groupStreams.valueType, childrenOffsets);
      }
      break;
    }
  }
}

// References a validated feature-order entry owned by the planner options.
struct OrderedFlatMap {
  size_t ordinal;
  const std::vector<int64_t>& features;
};

// Retains feature orders that refer to top-level FlatMap columns.
std::vector<OrderedFlatMap> validFlatMapFeatureOrders(
    const RowTypeBuilder& root,
    const std::vector<std::tuple<size_t, std::vector<int64_t>>>&
        flatMapFeatureOrder) {
  std::vector<OrderedFlatMap> orderedFlatMaps;
  orderedFlatMaps.reserve(flatMapFeatureOrder.size());
  for (const auto& [ordinal, features] : flatMapFeatureOrder) {
    if (ordinal >= root.childrenCount()) {
      LOG(WARNING)
          << "Column ordinal " << ordinal
          << " for feature ordering is out of range. Top-level row has "
          << root.childrenCount() << " columns.";
      continue;
    }

    if (root.childAt(ordinal).kind() != Kind::FlatMap) {
      LOG(WARNING) << "Column '" << root.nameAt(ordinal)
                   << "' for feature ordering is not a flat map.";
      continue;
    }

    orderedFlatMaps.push_back({.ordinal = ordinal, .features = features});
  }
  return orderedFlatMaps;
}

// Appends a FlatMap header followed by each configured feature's streams.
void appendOrderedFeatures(
    const FlatMapTypeBuilder& flatMap,
    const std::vector<int64_t>& features,
    std::vector<offset_size>& offsets) {
  offsets.push_back(flatMap.nullsDescriptor().offset());

  folly::F14FastMap<std::string, offset_size> flatMapNamedOrdinals;
  flatMapNamedOrdinals.reserve(flatMap.childrenCount());
  for (auto i = 0; i < flatMap.childrenCount(); ++i) {
    flatMapNamedOrdinals.insert({flatMap.nameAt(i), i});
  }

  for (const auto& feature : features) {
    auto it = flatMapNamedOrdinals.find(folly::to<std::string>(feature));
    if (it == flatMapNamedOrdinals.end()) {
      continue;
    }

    auto ordinal = it->second;
    auto& inMapDescriptor = flatMap.inMapDescriptorAt(ordinal);
    offsets.push_back(inMapDescriptor.offset());
    appendAllNestedStreams(flatMap.childAt(ordinal), offsets);
  }
}

// Appends all feature orders for a column and reports whether one was found.
bool appendFeatureOrdersOf(
    const RowTypeBuilder& root,
    size_t ordinal,
    const std::vector<OrderedFlatMap>& orderedFlatMaps,
    std::vector<offset_size>& offsets) {
  bool found{false};
  for (const auto& orderedFlatMap : orderedFlatMaps) {
    if (orderedFlatMap.ordinal == ordinal) {
      appendOrderedFeatures(
          root.childAt(ordinal).asFlatMap(), orderedFlatMap.features, offsets);
      found = true;
    }
  }
  return found;
}

// Resolves unique column names to top-level ordinals in configured order.
std::vector<size_t> resolveColumnOrder(
    const RowTypeBuilder& root,
    const std::vector<std::string>& columnOrder,
    std::vector<std::string>& unknownNames) {
  if (columnOrder.empty()) {
    return {};
  }

  folly::F14FastMap<std::string_view, size_t> ordinalsByName;
  ordinalsByName.reserve(root.childrenCount());
  for (size_t i = 0; i < root.childrenCount(); ++i) {
    ordinalsByName.emplace(root.nameAt(i), i);
  }

  std::vector<bool> seen(root.childrenCount());
  std::vector<size_t> ordinals;
  ordinals.reserve(columnOrder.size());
  for (const auto& name : columnOrder) {
    const auto it = ordinalsByName.find(name);
    if (it == ordinalsByName.end()) {
      unknownNames.push_back(name);
      continue;
    }
    if (!seen.at(it->second)) {
      seen.at(it->second) = true;
      ordinals.push_back(it->second);
    }
  }
  return ordinals;
}

} // namespace

DefaultLayoutPlanner::DefaultLayoutPlanner(
    const SchemaBuilder* schemaBuilder,
    LayoutPlannerOptions options)
    : schemaBuilder_{*velox::checkedNotNull(schemaBuilder)},
      options_{std::move(options)} {}

DefaultLayoutPlanner::DefaultLayoutPlanner(
    const SchemaBuilder* schemaBuilder,
    const std::optional<std::vector<std::tuple<size_t, std::vector<int64_t>>>>&
        flatMapFeatureOrder)
    : DefaultLayoutPlanner{schemaBuilder, [&] {
                             LayoutPlannerOptions options;
                             if (flatMapFeatureOrder.has_value()) {
                               options.flatMapFeatureOrder =
                                   *flatMapFeatureOrder;
                             }
                             return options;
                           }()} {}

const std::vector<size_t>& DefaultLayoutPlanner::resolvedColumnOrder(
    const RowTypeBuilder& root) {
  if (resolvedRoot_ == &root &&
      resolvedChildrenCount_ == root.childrenCount()) {
    return orderedColumns_;
  }

  std::vector<std::string> unknownNames;
  orderedColumns_ =
      resolveColumnOrder(root, options_.columnOrder, unknownNames);
  for (const auto& name : unknownNames) {
    LOG(WARNING) << "Column '" << name
                 << "' for column ordering is not a top-level column.";
  }
  isOrderedColumn_.assign(root.childrenCount(), false);
  for (const auto ordinal : orderedColumns_) {
    isOrderedColumn_[ordinal] = true;
  }
  resolvedRoot_ = &root;
  resolvedChildrenCount_ = root.childrenCount();
  return orderedColumns_;
}

std::vector<Stream> DefaultLayoutPlanner::getLayout(
    std::vector<Stream>&& streams) {
  const auto& type = schemaBuilder_.root();
  NIMBLE_CHECK_EQ(
      type->kind(),
      Kind::Row,
      "Layout planner requires row as the schema root.");
  auto& root = type->asRow();

  const auto orderedFlatMaps =
      validFlatMapFeatureOrders(root, options_.flatMapFeatureOrder);
  const auto& orderedColumns = resolvedColumnOrder(root);

  std::vector<offset_size> priorityOffsets;
  priorityOffsets.reserve(
      orderedColumns.size() + options_.flatMapFeatureOrder.size() * 3);

  for (const auto ordinal : orderedColumns) {
    if (!appendFeatureOrdersOf(
            root, ordinal, orderedFlatMaps, priorityOffsets)) {
      appendAllNestedStreams(root.childAt(ordinal), priorityOffsets);
    }
  }

  for (const auto& orderedFlatMap : orderedFlatMaps) {
    if (!orderedColumns.empty() && isOrderedColumn_[orderedFlatMap.ordinal]) {
      continue;
    }
    appendOrderedFeatures(
        root.childAt(orderedFlatMap.ordinal).asFlatMap(),
        orderedFlatMap.features,
        priorityOffsets);
  }

  std::vector<offset_size> schemaOffsets;
  appendAllNestedStreams(root, schemaOffsets);

  folly::F14FastMap<uint32_t, Stream*> offsetsToStreams;
  offsetsToStreams.reserve(streams.size());
  std::transform(
      streams.begin(),
      streams.end(),
      std::inserter(offsetsToStreams, offsetsToStreams.begin()),
      [](auto& stream) { return std::make_pair(stream.offset, &stream); });

  std::vector<Stream> layout;
  layout.reserve(streams.size());

  auto tryAppendStream = [&offsetsToStreams, &layout](uint32_t offset) {
    auto it = offsetsToStreams.find(offset);
    if (it == offsetsToStreams.end()) {
      return false;
    }
    layout.emplace_back(std::move(*it->second));
    offsetsToStreams.erase(it);
    return true;
  };
  auto appendStream = [&](uint32_t offset) {
    if (!tryAppendStream(offset)) {
      return;
    }
    const auto dictionaryStreamOffset =
        schemaBuilder_.sharedDictionaryStreamOffset(offset);
    if (dictionaryStreamOffset.has_value()) {
      tryAppendStream(dictionaryStreamOffset.value());
    }
  };

  appendStream(root.nullsDescriptor().offset());

  for (const auto offset : priorityOffsets) {
    appendStream(offset);
  }

  for (const auto offset : schemaOffsets) {
    appendStream(offset);
  }

  NIMBLE_CHECK_EQ(streams.size(), layout.size(), "Stream count mismatch.");

  return layout;
}
} // namespace facebook::nimble
