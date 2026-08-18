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
#include "velox/dwio/nimble/velox/LayoutPlanner.h"
#include <cstdint>

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
  }
}

} // namespace

DefaultLayoutPlanner::DefaultLayoutPlanner(
    std::function<std::shared_ptr<const TypeBuilder>()> typeResolver,
    const std::optional<std::vector<std::tuple<size_t, std::vector<int64_t>>>>&
        flatMapFeatureOrder)
    : typeResolver_{std::move(typeResolver)},
      flatMapFeatureOrder_{
          flatMapFeatureOrder.has_value()
              ? std::move(flatMapFeatureOrder.value())
              : std::vector<std::tuple<size_t, std::vector<int64_t>>>{}} {
  NIMBLE_CHECK_NOT_NULL(typeResolver_, "typeResolver is not supplied");
}

std::vector<Stream> DefaultLayoutPlanner::getLayout(
    std::vector<Stream>&& streams) {
  auto type = typeResolver_();
  NIMBLE_CHECK_EQ(
      type->kind(),
      Kind::Row,
      "Layout planner requires row as the schema root.");
  auto& root = type->asRow();

  // Layout logic:
  // 1. Root stream (Row nulls) is always first
  // 2. Later, all flat maps included in config are layed out.
  //    For each map, we layout all the features included in the config for
  //    that map, in the order they appeared in the config. For each feature, we
  //    first add its in-map stream and then all the value streams for that
  //    feature (if the value is a complex type, we add all the nested streams
  //    for this complex type together).
  // 3. We then layout all the other "leftover" streams, in "schema order". This
  //    guarantees that all "related" streams are next to each other.
  //    Leftover streams include all streams belonging to other columns, and all
  //    flat map features not included in the config.

  // This vector is going to hold all the ordered flat-map streams contained in
  // the config
  std::vector<offset_size> orderedFlatMapOffsets;
  orderedFlatMapOffsets.reserve(flatMapFeatureOrder_.size() * 3);

  for (const auto& flatMapFeatures : flatMapFeatureOrder_) {
    if (std::get<0>(flatMapFeatures) >= root.childrenCount()) {
      LOG(WARNING)
          << "Column ordinal " << std::get<0>(flatMapFeatures)
          << " for feature ordering is out of range. Top-level row has "
          << root.childrenCount() << " columns.";
      continue;
    }

    auto& column = root.childAt(std::get<0>(flatMapFeatures));

    if (column.kind() != Kind::FlatMap) {
      LOG(WARNING) << "Column '" << root.nameAt(std::get<0>(flatMapFeatures))
                   << "' for feature ordering is not a flat map.";
      continue;
    }

    auto& flatMap = column.asFlatMap();

    // For each flat map, first we push the flat map nulls stream.
    orderedFlatMapOffsets.push_back(flatMap.nullsDescriptor().offset());

    // Build a lookup table from feature name to its schema offset.
    std::unordered_map<std::string, offset_size> flatMapNamedOrdinals;
    flatMapNamedOrdinals.reserve(flatMap.childrenCount());
    for (auto i = 0; i < flatMap.childrenCount(); ++i) {
      flatMapNamedOrdinals.insert({flatMap.nameAt(i), i});
    }

    // Add feature's inMap stream along with all its nested streams to the
    // ordered stream list.
    for (const auto& feature : std::get<1>(flatMapFeatures)) {
      auto it = flatMapNamedOrdinals.find(folly::to<std::string>(feature));
      if (it == flatMapNamedOrdinals.end()) {
        continue;
      }

      auto ordinal = it->second;
      auto& inMapDescriptor = flatMap.inMapDescriptorAt(ordinal);
      orderedFlatMapOffsets.push_back(inMapDescriptor.offset());
      appendAllNestedStreams(flatMap.childAt(ordinal), orderedFlatMapOffsets);
    }
  }

  // This vector is going to hold all the streams, ordered  based on schema
  // order. This will include streams that already appear in the
  // 'orderedFlatMapOffsets'. Later, while laying out the final stream oder,
  // we'll de-dup these streams.
  std::vector<offset_size> orderedAllOffsets;
  appendAllNestedStreams(root, orderedAllOffsets);

  // Build a lookup table from type builders to their streams.
  std::unordered_map<uint32_t, Stream*> offsetsToStreams;
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
    if (it != offsetsToStreams.end()) {
      layout.emplace_back(std::move(*it->second));
      offsetsToStreams.erase(it);
    }
  };

  // At this point we have ordered all the type builders, so now we are going to
  // try and find matching streams for each type builder and append them to the
  // final ordered stream list.

  // First add the root's null stream
  tryAppendStream(root.nullsDescriptor().offset());

  // Then, add all ordered flat maps
  for (auto offset : orderedFlatMapOffsets) {
    tryAppendStream(offset);
  }

  // Then add all remaining streams in the schema order.
  // 'tryAppendStream' will de-dup streams that were already added in previous
  // steps.
  for (auto offset : orderedAllOffsets) {
    tryAppendStream(offset);
  }

  NIMBLE_CHECK_EQ(streams.size(), layout.size(), "Stream count mismatch.");

  return layout;
}
} // namespace facebook::nimble
