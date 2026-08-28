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
#include "velox/dwio/nimble/common/SchemaUtils.h"

#include <set>

#include "folly/container/F14Map.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaBuilder.h"

namespace facebook::nimble {

using SubfieldKind = velox::common::SubfieldKind;
using Subfield = velox::common::Subfield;

//
// Common helpers shared by multiple public APIs.
//

namespace {

velox::TypePtr convertToVeloxScalarType(ScalarKind scalarKind) {
  switch (scalarKind) {
    case ScalarKind::Int8:
      return velox::ScalarType<velox::TypeKind::TINYINT>::create();
    case ScalarKind::Int16:
      return velox::ScalarType<velox::TypeKind::SMALLINT>::create();
    case ScalarKind::Int32:
      return velox::ScalarType<velox::TypeKind::INTEGER>::create();
    case ScalarKind::Int64:
      return velox::ScalarType<velox::TypeKind::BIGINT>::create();
    case ScalarKind::Float:
      return velox::ScalarType<velox::TypeKind::REAL>::create();
    case ScalarKind::Double:
      return velox::ScalarType<velox::TypeKind::DOUBLE>::create();
    case ScalarKind::Bool:
      return velox::ScalarType<velox::TypeKind::BOOLEAN>::create();
    case ScalarKind::String:
      return velox::ScalarType<velox::TypeKind::VARCHAR>::create();
    case ScalarKind::Binary:
      return velox::ScalarType<velox::TypeKind::VARBINARY>::create();
    case ScalarKind::UInt8:
    case ScalarKind::UInt16:
    case ScalarKind::UInt32:
    case ScalarKind::UInt64:
    case ScalarKind::Undefined:
      NIMBLE_UNSUPPORTED(
          "Scalar kind {} is not supported by Velox.", toString(scalarKind));
  }
  NIMBLE_UNREACHABLE("Unknown scalarKind: {}.", toString(scalarKind));
}

std::shared_ptr<TypeBuilder> convertToNimbleType(
    SchemaBuilder& builder,
    const velox::Type& type) {
  switch (type.kind()) {
    case velox::TypeKind::BOOLEAN:
      return builder.createScalarTypeBuilder(ScalarKind::Bool);
    case velox::TypeKind::TINYINT:
      return builder.createScalarTypeBuilder(ScalarKind::Int8);
    case velox::TypeKind::SMALLINT:
      return builder.createScalarTypeBuilder(ScalarKind::Int16);
    case velox::TypeKind::INTEGER:
      return builder.createScalarTypeBuilder(ScalarKind::Int32);
    case velox::TypeKind::BIGINT:
      return builder.createScalarTypeBuilder(ScalarKind::Int64);
    case velox::TypeKind::REAL:
      return builder.createScalarTypeBuilder(ScalarKind::Float);
    case velox::TypeKind::DOUBLE:
      return builder.createScalarTypeBuilder(ScalarKind::Double);
    case velox::TypeKind::VARCHAR:
      return builder.createScalarTypeBuilder(ScalarKind::String);
    case velox::TypeKind::VARBINARY:
      return builder.createScalarTypeBuilder(ScalarKind::Binary);
    case velox::TypeKind::TIMESTAMP:
      return builder.createTimestampMicroNanoTypeBuilder();
    case velox::TypeKind::ARRAY: {
      auto& arrayType = type.asArray();
      auto nimbleType = builder.createArrayTypeBuilder();
      nimbleType->setChildren(
          convertToNimbleType(builder, *arrayType.elementType()));
      return nimbleType;
    }
    case velox::TypeKind::MAP: {
      auto& mapType = type.asMap();
      auto nimbleType = builder.createMapTypeBuilder();
      auto key = convertToNimbleType(builder, *mapType.keyType());
      auto value = convertToNimbleType(builder, *mapType.valueType());
      nimbleType->setChildren(key, value);
      return nimbleType;
    }
    case velox::TypeKind::ROW: {
      auto& rowType = type.asRow();
      auto nimbleType = builder.createRowTypeBuilder(rowType.size());
      for (size_t i = 0; i < rowType.size(); ++i) {
        nimbleType->addChild(
            rowType.nameOf(i),
            convertToNimbleType(builder, *rowType.childAt(i)));
      }
      return nimbleType;
    }
    default:
      NIMBLE_UNSUPPORTED("Unsupported type kind {}.", type.kind());
  }
}

} // namespace

//
// Simple type conversions.
//

velox::TypePtr convertToVeloxType(const Type& type) {
  switch (type.kind()) {
    case Kind::Scalar: {
      return convertToVeloxScalarType(
          type.asScalar().scalarDescriptor().scalarKind());
    }
    case Kind::TimestampMicroNano: {
      return velox::ScalarType<velox::TypeKind::TIMESTAMP>::create();
    }
    case Kind::Row: {
      const auto& rowType = type.asRow();
      std::vector<std::string> names;
      std::vector<velox::TypePtr> children;
      names.reserve(rowType.childrenCount());
      children.reserve(rowType.childrenCount());
      for (size_t i = 0; i < rowType.childrenCount(); ++i) {
        names.push_back(rowType.nameAt(i));
        children.push_back(convertToVeloxType(*rowType.childAt(i)));
      }
      return std::make_shared<const velox::RowType>(
          std::move(names), std::move(children));
    }
    case Kind::Array: {
      const auto& arrayType = type.asArray();
      return std::make_shared<const velox::ArrayType>(
          convertToVeloxType(*arrayType.elements()));
    }
    case Kind::ArrayWithOffsets: {
      const auto& arrayWithOffsetsType = type.asArrayWithOffsets();
      return std::make_shared<const velox::ArrayType>(
          convertToVeloxType(*arrayWithOffsetsType.elements()));
    }
    case Kind::Map: {
      const auto& mapType = type.asMap();
      return std::make_shared<const velox::MapType>(
          convertToVeloxType(*mapType.keys()),
          convertToVeloxType(*mapType.values()));
    }
    case Kind::SlidingWindowMap: {
      const auto& mapType = type.asSlidingWindowMap();
      return std::make_shared<const velox::MapType>(
          convertToVeloxType(*mapType.keys()),
          convertToVeloxType(*mapType.values()));
    }
    case Kind::FlatMap: {
      const auto& flatMapType = type.asFlatMap();
      return std::make_shared<const velox::MapType>(
          convertToVeloxScalarType(flatMapType.keyScalarKind()),
          // When Flatmap is empty, we always insert dummy key/value
          // to it, so it is guaranteed that flatMapType.childAt(0)
          // is always valid.
          convertToVeloxType(*flatMapType.childAt(0)));
    }
    default:
      NIMBLE_UNREACHABLE("Unknown type kind {}.", toString(type.kind()));
  }
}

std::shared_ptr<const Type> convertToNimbleType(const velox::Type& type) {
  SchemaBuilder builder;
  convertToNimbleType(builder, type);
  return SchemaReader::getSchema(builder.schemaNodes());
}

//
// buildProjectedNimbleType and its helpers.
//

namespace {

ScalarKind toScalarKind(velox::TypeKind kind) {
  switch (kind) {
    case velox::TypeKind::BOOLEAN:
      return ScalarKind::Bool;
    case velox::TypeKind::TINYINT:
      return ScalarKind::Int8;
    case velox::TypeKind::SMALLINT:
      return ScalarKind::Int16;
    case velox::TypeKind::INTEGER:
      return ScalarKind::Int32;
    case velox::TypeKind::BIGINT:
      return ScalarKind::Int64;
    case velox::TypeKind::REAL:
      return ScalarKind::Float;
    case velox::TypeKind::DOUBLE:
      return ScalarKind::Double;
    case velox::TypeKind::VARCHAR:
      return ScalarKind::String;
    case velox::TypeKind::VARBINARY:
      return ScalarKind::Binary;
    default:
      NIMBLE_FAIL("Unsupported velox TypeKind for nimble ScalarKind: {}", kind);
  }
}

// Tracks which children to include at each Row/Map node during velox type
// projection. Same pattern as NimbleSelectedChildrenMap (nimble type) but keyed
// by velox type pointer. When a node appears in this map, only its selected
// children are included. When absent, all children are included.
//
// For ROW types, children are indexed by position (matching velox RowType).
// For MAP types, children are keyed by name (stored as strings in a separate
// map) — since Map children don't have positional indices, we use a name-based
// map and convert to FlatMap during nimble type construction.
using VeloxSelectedChildrenMap =
    folly::F14FastMap<const velox::Type*, std::set<std::string>>;

// Resolves a single subfield path against a velox type tree, populating
// selectedChildren with which children to include at each Row/Map node.
// Similar to resolveSubfield for nimble types but walks velox types instead.
void resolveVeloxSubfield(
    const velox::Type* type,
    const velox::common::Subfield& subfield,
    VeloxSelectedChildrenMap& selectedChildren) {
  const auto& path = subfield.path();
  NIMBLE_CHECK(!path.empty(), "Empty subfield path");

  const auto* current = type;

  for (const auto& element : path) {
    const auto kind = element->kind();
    if (kind == SubfieldKind::kNestedField) {
      const auto& name = element->asChecked<Subfield::NestedField>()->name();

      NIMBLE_CHECK_EQ(
          current->kind(),
          velox::TypeKind::ROW,
          "Cannot access nested field '{}' on non-ROW type in subfield path '{}'",
          name,
          subfield.toString());
      const auto& row = current->asRow();
      auto childIdx = row.getChildIdxIfExists(name);
      NIMBLE_CHECK(
          childIdx.has_value(),
          "Field '{}' not found in RowType: {}",
          name,
          row.toString());

      selectedChildren[current].insert(name);
      current = row.childAt(*childIdx).get();
    } else if (
        kind == SubfieldKind::kStringSubscript ||
        kind == SubfieldKind::kLongSubscript) {
      const auto keyName = (kind == SubfieldKind::kStringSubscript)
          ? element->asChecked<Subfield::StringSubscript>()->index()
          : std::to_string(
                element->asChecked<Subfield::LongSubscript>()->index());

      NIMBLE_CHECK_EQ(
          current->kind(),
          velox::TypeKind::MAP,
          "Subscript '{}' only supported on MAP type in subfield path '{}'",
          keyName,
          subfield.toString());

      selectedChildren[current].insert(keyName);
      // All map keys share the same value type, so descend into it.
      current = current->asMap().valueType().get();
    } else {
      NIMBLE_FAIL(
          "Unsupported subfield kind {} in subfield path '{}'",
          kind,
          subfield.toString());
    }
  }
}

// Recursively converts a velox type to a projected nimble type. Handles both
// projected nodes (in selectedChildren) and full subtree conversion, with
// encoding-hint-aware type creation for top-level children.
//
// - If the node is in selectedChildren: ROW creates a projected Row with only
//   selected children; MAP creates a FlatMap with selected keys sorted
//   alphabetically.
// - If the node is NOT in selectedChildren: converts the full subtree.
//   Encoding hints (useDictionaryArray, useDeduplicatedMap, useFlatMap) create
//   ArrayWithOffsets, SlidingWindowMap, or FlatMap respectively. These flags
//   are only set for top-level children by the caller; recursive calls always
//   pass false.
//
// When useFlatMap is true and the MAP column is projected without key
// subscripts, fails because key names/order cannot be recovered from the
// velox type alone.
std::shared_ptr<TypeBuilder> buildProjectedNimbleType(
    SchemaBuilder& builder,
    const velox::Type& type,
    const VeloxSelectedChildrenMap& selectedChildren,
    bool useDictionaryArray = false,
    bool useDeduplicatedMap = false,
    bool useFlatMap = false) {
  const auto it = selectedChildren.find(&type);
  const bool hasSelectedChildren = (it != selectedChildren.end());

  switch (type.kind()) {
    case velox::TypeKind::BOOLEAN:
      return builder.createScalarTypeBuilder(ScalarKind::Bool);
    case velox::TypeKind::TINYINT:
      return builder.createScalarTypeBuilder(ScalarKind::Int8);
    case velox::TypeKind::SMALLINT:
      return builder.createScalarTypeBuilder(ScalarKind::Int16);
    case velox::TypeKind::INTEGER:
      return builder.createScalarTypeBuilder(ScalarKind::Int32);
    case velox::TypeKind::BIGINT:
      return builder.createScalarTypeBuilder(ScalarKind::Int64);
    case velox::TypeKind::REAL:
      return builder.createScalarTypeBuilder(ScalarKind::Float);
    case velox::TypeKind::DOUBLE:
      return builder.createScalarTypeBuilder(ScalarKind::Double);
    case velox::TypeKind::VARCHAR:
      return builder.createScalarTypeBuilder(ScalarKind::String);
    case velox::TypeKind::VARBINARY:
      return builder.createScalarTypeBuilder(ScalarKind::Binary);
    case velox::TypeKind::TIMESTAMP:
      return builder.createTimestampMicroNanoTypeBuilder();

    case velox::TypeKind::ARRAY: {
      const auto& arrayType = type.asArray();
      if (useDictionaryArray) {
        auto nimbleType = builder.createArrayWithOffsetsTypeBuilder();
        nimbleType->setChildren(buildProjectedNimbleType(
            builder, *arrayType.elementType(), selectedChildren));
        return nimbleType;
      }
      auto nimbleType = builder.createArrayTypeBuilder();
      nimbleType->setChildren(buildProjectedNimbleType(
          builder,
          *arrayType.elementType(),
          selectedChildren,
          /*useDictionaryArray=*/false,
          /*useDeduplicatedMap=*/false,
          /*useFlatMap=*/false));
      return nimbleType;
    }

    case velox::TypeKind::MAP: {
      const auto& mapType = type.asMap();
      if (hasSelectedChildren) {
        NIMBLE_CHECK(
            useFlatMap,
            "FlatMap key-level projection requires the column to be in "
            "flatMapColumns and must be a top-level MAP column: {}",
            type.toString());
        auto keyScalarKind = toScalarKind(mapType.keyType()->kind());
        auto flatMap = builder.createFlatMapTypeBuilder(keyScalarKind);
        // std::set is already sorted alphabetically.
        for (const auto& key : it->second) {
          flatMap->addChild(
              key,
              buildProjectedNimbleType(
                  builder,
                  *mapType.valueType(),
                  selectedChildren,
                  /*useDictionaryArray=*/false,
                  /*useDeduplicatedMap=*/false,
                  /*useFlatMap=*/false));
        }
        return flatMap;
      }

      NIMBLE_CHECK(
          !useFlatMap,
          "Cannot project entire FlatMap column without key subscripts. "
          "FlatMap key names and order are not available from the velox type. "
          "Use key-level projection (e.g., map[\"key\"]) or the "
          "nimble-schema-based buildProjectedNimbleType overload instead.");

      if (useDeduplicatedMap) {
        auto nimbleType = builder.createSlidingWindowMapTypeBuilder();
        auto key = buildProjectedNimbleType(
            builder,
            *mapType.keyType(),
            selectedChildren,
            /*useDictionaryArray=*/false,
            /*useDeduplicatedMap=*/false,
            /*useFlatMap=*/false);
        auto value = buildProjectedNimbleType(
            builder,
            *mapType.valueType(),
            selectedChildren,
            /*useDictionaryArray=*/false,
            /*useDeduplicatedMap=*/false,
            /*useFlatMap=*/false);
        nimbleType->setChildren(key, value);
        return nimbleType;
      }

      auto nimbleType = builder.createMapTypeBuilder();
      auto key = buildProjectedNimbleType(
          builder,
          *mapType.keyType(),
          selectedChildren,
          /*useDictionaryArray=*/false,
          /*useDeduplicatedMap=*/false,
          /*useFlatMap=*/false);
      auto value = buildProjectedNimbleType(
          builder,
          *mapType.valueType(),
          selectedChildren,
          /*useDictionaryArray=*/false,
          /*useDeduplicatedMap=*/false,
          /*useFlatMap=*/false);
      nimbleType->setChildren(key, value);
      return nimbleType;
    }

    case velox::TypeKind::ROW: {
      const auto& rowType = type.asRow();
      if (hasSelectedChildren) {
        const auto& selectedChildNames = it->second;
        auto nimbleRow =
            builder.createRowTypeBuilder(selectedChildNames.size());
        // Iterate in velox type order for consistent child ordering.
        for (size_t i = 0; i < rowType.size(); ++i) {
          const auto& childName = rowType.nameOf(i);
          if (!selectedChildNames.contains(childName)) {
            continue;
          }
          nimbleRow->addChild(
              childName,
              buildProjectedNimbleType(
                  builder,
                  *rowType.childAt(i),
                  selectedChildren,
                  /*useDictionaryArray=*/false,
                  /*useDeduplicatedMap=*/false,
                  /*useFlatMap=*/false));
        }
        return nimbleRow;
      }

      auto nimbleType = builder.createRowTypeBuilder(rowType.size());
      for (size_t i = 0; i < rowType.size(); ++i) {
        nimbleType->addChild(
            rowType.nameOf(i),
            buildProjectedNimbleType(
                builder,
                *rowType.childAt(i),
                selectedChildren,
                /*useDictionaryArray=*/false,
                /*useDeduplicatedMap=*/false,
                /*useFlatMap=*/false));
      }
      return nimbleType;
    }

    default:
      NIMBLE_UNSUPPORTED("Unsupported type kind {}.", type.kind());
  }
}

} // namespace

std::shared_ptr<const Type> buildProjectedNimbleType(
    const velox::RowType& type,
    const std::vector<velox::common::Subfield>& projectedSubfields,
    const ColumnEncodings& columnEncodings) {
  NIMBLE_CHECK(
      !projectedSubfields.empty(), "projectedSubfields must not be empty");

  VeloxSelectedChildrenMap selectedChildren;
  for (const auto& subfield : projectedSubfields) {
    resolveVeloxSubfield(&type, subfield, selectedChildren);
  }

  const auto& selectedTopLevelColumns = selectedChildren[&type];
  NIMBLE_CHECK(
      !selectedTopLevelColumns.empty(),
      "No top-level columns resolved from projectedSubfields");

  SchemaBuilder builder;
  auto root = builder.createRowTypeBuilder(selectedTopLevelColumns.size());

  // Iterate in velox type order for consistent child ordering.
  for (size_t i = 0; i < type.size(); ++i) {
    const auto& name = type.nameOf(i);
    if (!selectedTopLevelColumns.contains(name)) {
      continue;
    }

    const bool useDictionaryArray =
        columnEncodings.dictionaryArrayColumns.contains(name);
    const bool useDeduplicatedMap =
        columnEncodings.deduplicatedMapColumns.contains(name);
    const bool useFlatMap = columnEncodings.flatMapColumns.contains(name);

    root->addChild(
        name,
        buildProjectedNimbleType(
            builder,
            *type.childAt(i),
            selectedChildren,
            useDictionaryArray,
            useDeduplicatedMap,
            useFlatMap));
  }

  return SchemaReader::getSchema(builder.schemaNodes());
}

//
// deriveColumnEncodings, computeProjectedStreamMetadata.
//

namespace {

// Tracks the source children the projection touches at each Row or FlatMap
// node, stored as the child's source-side index. For a Row, the indices are
// Row child positions. For a FlatMap, the indices are positions of source
// keys that match a requested subscript. FlatMap subscripts whose key is NOT
// in the source go into `missingChildren` below instead.
using SelectedChildrenMap = folly::F14FastMap<const Type*, std::set<size_t>>;

// Tracks the FlatMap subscript keys the projection requested that do NOT
// exist in the source. The offset walker iterates them alphabetically and
// emits UINT32_MAX placeholder slots so the projected blob lines up with the
// projected schema's synthetic children (created by the velox-source
// `buildProjectedNimbleType`).
using MissingChildrenMap =
    folly::F14FastMap<const Type*, std::set<std::string>>;

inline void appendProjectedStream(
    std::vector<uint32_t>& projectedStreamOffsets,
    std::vector<bool>& rowOrFlatMapNullStreams,
    uint32_t sourceStreamOffset,
    bool isRowOrFlatMapNullStream) {
  projectedStreamOffsets.emplace_back(sourceStreamOffset);
  rowOrFlatMapNullStreams.emplace_back(isRowOrFlatMapNullStream);
}

// Resolves a single subfield path against a source nimble schema, populating
// selectedChildren for present Row children + present FlatMap keys (by
// source index) and missingChildren for FlatMap keys absent from the source
// (by name).
void resolveSubfield(
    const Type* type,
    const velox::common::Subfield& subfield,
    SelectedChildrenMap& selectedChildren,
    MissingChildrenMap& missingChildren) {
  const auto& path = subfield.path();
  NIMBLE_CHECK(!path.empty(), "Empty subfield path");

  const auto* current = type;

  for (const auto& element : path) {
    const auto kind = element->kind();
    if (kind == SubfieldKind::kNestedField) {
      const auto* nested = element->asChecked<Subfield::NestedField>();
      const auto& name = nested->name();

      NIMBLE_CHECK(
          current->isRow(),
          "Cannot access nested field '{}' on non-Row type in subfield path '{}'",
          name,
          subfield.toString());
      const auto& row = current->asRow();
      const auto childIdx = row.findChild(name);
      NIMBLE_CHECK(
          childIdx.has_value(), "Field '{}' not found in RowType", name);

      selectedChildren[current].insert(*childIdx);
      current = row.childAt(*childIdx).get();
    } else if (
        kind == SubfieldKind::kStringSubscript ||
        kind == SubfieldKind::kLongSubscript) {
      // Convert subscript to string key name. FlatMap keys are always strings
      // internally, so LongSubscript is converted via std::to_string.
      const auto keyName = (kind == SubfieldKind::kStringSubscript)
          ? element->asChecked<Subfield::StringSubscript>()->index()
          : std::to_string(
                element->asChecked<Subfield::LongSubscript>()->index());

      NIMBLE_CHECK(
          current->isFlatMap(),
          "Subscript '{}' only supported on FlatMap in subfield path '{}'",
          keyName,
          subfield.toString());
      const auto& flatMap = current->asFlatMap();
      const auto childIdx = flatMap.findChild(keyName);
      if (!childIdx.has_value()) {
        // Key not in source — recorded by name for placeholder emission. Any
        // remaining subfield path elements are discarded; the whole synthetic
        // value subtree decodes to null.
        missingChildren[current].insert(keyName);
        return;
      }
      selectedChildren[current].insert(*childIdx);
      current = flatMap.childAt(*childIdx).get();
    } else {
      NIMBLE_FAIL(
          "Unsupported subfield kind {} in subfield path '{}'",
          kind,
          subfield.toString());
    }
  }
}

// Walks a value-type (typically `flatMap.childAt(0)` from a source FlatMap
// whose requested key is missing) and emits placeholder metadata for every
// stream descriptor the subtree contains.
// The number of UINT32_MAX entries matches the slot count the velox-source
// `buildProjectedNimbleType` would allocate for the corresponding synthetic
// FlatMap value subtree.
void emitPlaceholderStreamOffsets(
    const Type* valueType,
    std::vector<uint32_t>& projectedStreamOffsets,
    std::vector<bool>& rowOrFlatMapNullStreams) {
  switch (valueType->kind()) {
    case Kind::Scalar:
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/false);
      return;
    case Kind::TimestampMicroNano:
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/false);
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/false);
      return;
    case Kind::Row: {
      const auto& row = valueType->asRow();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/true);
      for (size_t i = 0; i < row.childrenCount(); ++i) {
        emitPlaceholderStreamOffsets(
            row.childAt(i).get(),
            projectedStreamOffsets,
            rowOrFlatMapNullStreams);
      }
      return;
    }
    case Kind::Array: {
      const auto& array = valueType->asArray();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/false);
      emitPlaceholderStreamOffsets(
          array.elements().get(),
          projectedStreamOffsets,
          rowOrFlatMapNullStreams);
      return;
    }
    case Kind::ArrayWithOffsets: {
      const auto& array = valueType->asArrayWithOffsets();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/false);
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/false);
      emitPlaceholderStreamOffsets(
          array.elements().get(),
          projectedStreamOffsets,
          rowOrFlatMapNullStreams);
      return;
    }
    case Kind::Map: {
      const auto& map = valueType->asMap();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/false);
      emitPlaceholderStreamOffsets(
          map.keys().get(), projectedStreamOffsets, rowOrFlatMapNullStreams);
      emitPlaceholderStreamOffsets(
          map.values().get(), projectedStreamOffsets, rowOrFlatMapNullStreams);
      return;
    }
    case Kind::SlidingWindowMap: {
      const auto& map = valueType->asSlidingWindowMap();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/false);
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/false);
      emitPlaceholderStreamOffsets(
          map.keys().get(), projectedStreamOffsets, rowOrFlatMapNullStreams);
      emitPlaceholderStreamOffsets(
          map.values().get(), projectedStreamOffsets, rowOrFlatMapNullStreams);
      return;
    }
    case Kind::FlatMap:
      // Value subtrees of a FlatMap are not themselves FlatMaps per the
      // encoding invariant.
      NIMBLE_FAIL(
          "Nested FlatMap inside synthetic value subtree is not supported");
  }
  NIMBLE_UNREACHABLE("Unknown type kind: {}", valueType->kind());
}

// Forward declaration: projectStreamOffsets and projectFlatmapStreamOffsets
// recurse mutually (a FlatMap's value subtree may contain Rows that recurse
// back through the general walker).
void projectStreamOffsets(
    const Type* type,
    const SelectedChildrenMap& selectedChildren,
    const MissingChildrenMap& missingChildren,
    std::vector<uint32_t>& projectedStreamOffsets,
    std::vector<bool>& rowOrFlatMapNullStreams);

// Emits the FlatMap branch of `projectStreamOffsets`. Merges present keys
// (from `selectedChildren[flatMap]`, by source index) and missing keys (from
// `missingChildren[flatMap]`, by name) into a single alphabetically-sorted
// list — matching the velox-source builder's iteration over its
// `std::set<std::string>` of requested keys — and emits per-key stream
// offsets: the source's value-subtree offsets + inMap offset for present
// keys, or UINT32_MAX placeholders for missing keys.
void projectFlatmapStreamOffsets(
    const FlatMapType& flatMap,
    const SelectedChildrenMap& selectedChildren,
    const MissingChildrenMap& missingChildren,
    std::vector<uint32_t>& projectedStreamOffsets,
    std::vector<bool>& rowOrFlatMapNullStreams) {
  const auto* flatMapPtr = static_cast<const Type*>(&flatMap);
  const auto selectedIt = selectedChildren.find(flatMapPtr);
  const auto missingIt = missingChildren.find(flatMapPtr);
  // FlatMap must have at least one requested key (real or missing).
  // Projecting an entire FlatMap without subscripts is rejected — the
  // velox-source builder cannot emit a FlatMap from a velox MAP<K,V>
  // without the explicit key list.
  NIMBLE_CHECK(
      selectedIt != selectedChildren.end() ||
          missingIt != missingChildren.end(),
      "Cannot project entire FlatMap column without key subscripts. "
      "Use key-level projection (e.g., map[\"key\"]).");

  struct Entry {
    std::string keyName;
    // Missing keys use nullopt and emit UINT32_MAX placeholders for the value
    // subtree and inMap stream.
    std::optional<size_t> valueIndex;
  };
  std::vector<Entry> entries;
  if (selectedIt != selectedChildren.end()) {
    entries.reserve(selectedIt->second.size());
    for (size_t idx : selectedIt->second) {
      entries.push_back({std::string(flatMap.nameAt(idx)), idx});
    }
  }
  if (missingIt != missingChildren.end()) {
    entries.reserve(entries.size() + missingIt->second.size());
    for (const auto& keyName : missingIt->second) {
      entries.push_back({keyName, std::nullopt});
    }
  }
  std::sort(
      entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
        return lhs.keyName < rhs.keyName;
      });

  appendProjectedStream(
      projectedStreamOffsets,
      rowOrFlatMapNullStreams,
      flatMap.nullsDescriptor().offset(),
      /*isRowOrFlatMapNullStream=*/true);
  const auto* valueType = flatMap.childAt(0).get();
  for (const auto& entry : entries) {
    if (entry.valueIndex.has_value()) {
      projectStreamOffsets(
          flatMap.childAt(*entry.valueIndex).get(),
          selectedChildren,
          missingChildren,
          projectedStreamOffsets,
          rowOrFlatMapNullStreams);
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          flatMap.inMapDescriptorAt(*entry.valueIndex).offset(),
          /*isRowOrFlatMapNullStream=*/false);
    } else {
      emitPlaceholderStreamOffsets(
          valueType, projectedStreamOffsets, rowOrFlatMapNullStreams);
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          UINT32_MAX,
          /*isRowOrFlatMapNullStream=*/false);
    }
  }
}

// Walks `type` (a node within the source nimble schema) in DFS pre-order
// and appends one source stream offset plus one Row/FlatMap null stream bit per
// stream descriptor. At Row nodes appearing in `selectedChildren`, only the
// selected child indices are descended into (in source order). Top-level
// FlatMap columns are handled by the root walker before this helper is called,
// so any FlatMap encountered here is nested and unsupported. The traversal
// order matches the velox-source overload of `buildProjectedNimbleType` so the
// emitted metadata lines up positionally with the projected schema it returns.
void projectStreamOffsets(
    const Type* type,
    const SelectedChildrenMap& selectedChildren,
    const MissingChildrenMap& missingChildren,
    std::vector<uint32_t>& projectedStreamOffsets,
    std::vector<bool>& rowOrFlatMapNullStreams) {
  switch (type->kind()) {
    case Kind::Scalar:
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          type->asScalar().scalarDescriptor().offset(),
          /*isRowOrFlatMapNullStream=*/false);
      return;
    case Kind::TimestampMicroNano: {
      const auto& ts = type->asTimestampMicroNano();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          ts.microsDescriptor().offset(),
          /*isRowOrFlatMapNullStream=*/false);
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          ts.nanosDescriptor().offset(),
          /*isRowOrFlatMapNullStream=*/false);
      return;
    }
    case Kind::Row: {
      const auto& row = type->asRow();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          row.nullsDescriptor().offset(),
          /*isRowOrFlatMapNullStream=*/true);
      const auto it = selectedChildren.find(type);
      if (it != selectedChildren.end()) {
        // std::set<size_t> iterates ascending — matches source child order.
        for (size_t idx : it->second) {
          projectStreamOffsets(
              row.childAt(idx).get(),
              selectedChildren,
              missingChildren,
              projectedStreamOffsets,
              rowOrFlatMapNullStreams);
        }
      } else {
        for (size_t i = 0; i < row.childrenCount(); ++i) {
          projectStreamOffsets(
              row.childAt(i).get(),
              selectedChildren,
              missingChildren,
              projectedStreamOffsets,
              rowOrFlatMapNullStreams);
        }
      }
      return;
    }
    case Kind::Array: {
      const auto& array = type->asArray();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          array.lengthsDescriptor().offset(),
          /*isRowOrFlatMapNullStream=*/false);
      projectStreamOffsets(
          array.elements().get(),
          selectedChildren,
          missingChildren,
          projectedStreamOffsets,
          rowOrFlatMapNullStreams);
      return;
    }
    case Kind::ArrayWithOffsets: {
      const auto& array = type->asArrayWithOffsets();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          array.offsetsDescriptor().offset(),
          /*isRowOrFlatMapNullStream=*/false);
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          array.lengthsDescriptor().offset(),
          /*isRowOrFlatMapNullStream=*/false);
      projectStreamOffsets(
          array.elements().get(),
          selectedChildren,
          missingChildren,
          projectedStreamOffsets,
          rowOrFlatMapNullStreams);
      return;
    }
    case Kind::Map: {
      const auto& map = type->asMap();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          map.lengthsDescriptor().offset(),
          /*isRowOrFlatMapNullStream=*/false);
      projectStreamOffsets(
          map.keys().get(),
          selectedChildren,
          missingChildren,
          projectedStreamOffsets,
          rowOrFlatMapNullStreams);
      projectStreamOffsets(
          map.values().get(),
          selectedChildren,
          missingChildren,
          projectedStreamOffsets,
          rowOrFlatMapNullStreams);
      return;
    }
    case Kind::SlidingWindowMap: {
      const auto& map = type->asSlidingWindowMap();
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          map.offsetsDescriptor().offset(),
          /*isRowOrFlatMapNullStream=*/false);
      appendProjectedStream(
          projectedStreamOffsets,
          rowOrFlatMapNullStreams,
          map.lengthsDescriptor().offset(),
          /*isRowOrFlatMapNullStream=*/false);
      projectStreamOffsets(
          map.keys().get(),
          selectedChildren,
          missingChildren,
          projectedStreamOffsets,
          rowOrFlatMapNullStreams);
      projectStreamOffsets(
          map.values().get(),
          selectedChildren,
          missingChildren,
          projectedStreamOffsets,
          rowOrFlatMapNullStreams);
      return;
    }
    case Kind::FlatMap:
      NIMBLE_FAIL("FlatMap projection is supported only for top-level columns");
  }
  NIMBLE_UNREACHABLE("Unknown type kind: {}", type->kind());
}

// Classifies each top-level column of the source nimble Row by `Kind` to
// build the encoding hints the velox-source `buildProjectedNimbleType`
// overload requires: FlatMap → flatMapColumns, ArrayWithOffsets →
// dictionaryArrayColumns, SlidingWindowMap → deduplicatedMapColumns. Other
// kinds produce no entry (plain encoding). Internal helper of
// `buildProjectedNimbleType` (nimble-source overload).
ColumnEncodings getColumnEncodings(const RowType& nimbleType) {
  ColumnEncodings encodings;
  for (size_t i = 0; i < nimbleType.childrenCount(); ++i) {
    const auto& name = nimbleType.nameAt(i);
    switch (nimbleType.childAt(i)->kind()) {
      case Kind::FlatMap:
        encodings.flatMapColumns.insert(std::string(name));
        break;
      case Kind::ArrayWithOffsets:
        encodings.dictionaryArrayColumns.insert(std::string(name));
        break;
      case Kind::SlidingWindowMap:
        encodings.deduplicatedMapColumns.insert(std::string(name));
        break;
      default:
        // Plain encoding; no entry needed.
        break;
    }
  }
  return encodings;
}

} // namespace

NimbleTypeProjection buildProjectedNimbleType(
    const Type* type,
    const std::vector<velox::common::Subfield>& projectedSubfields) {
  NIMBLE_CHECK(
      !projectedSubfields.empty(), "projectedSubfields must not be empty");
  NIMBLE_CHECK_NOT_NULL(type, "type must not be null");
  NIMBLE_CHECK(type->isRow(), "Root type must be a Row, got: {}", type->kind());

  NimbleTypeProjection projection;

  // Resolve subfields against the source schema once. Real selections (Row
  // children + present FlatMap keys) land in `selectedChildren` keyed by
  // source index; FlatMap keys absent from the source go into
  // `missingChildren` keyed by name. The offset walker merges the two per
  // FlatMap node before emitting.
  SelectedChildrenMap selectedChildren;
  MissingChildrenMap missingChildren;
  for (const auto& subfield : projectedSubfields) {
    resolveSubfield(type, subfield, selectedChildren, missingChildren);
  }

  // An all-missing-keys projection is allowed: the projected FlatMap contains
  // a placeholder child per requested key, and the byte-copy pipeline emits
  // 0-byte slots that the deserializer's gap-fill turns into null columns.
  // Callers that need to surface "typo on every key" as an error must do
  // their own presence check against the source schema before projecting.

  const auto& rootRow = type->asRow();
  const auto& selectedColumnIndices = selectedChildren[type];
  NIMBLE_CHECK(
      !selectedColumnIndices.empty(),
      "No top-level columns resolved from projectedSubfields");

  // Build the projected schema from the velox view of the source. The velox
  // MAP<K,V> view discards FlatMap key inventory, so the velox-source
  // builder produces one alphabetically-sorted child per requested subscript
  // key regardless of source presence — exactly the shape we need for the
  // source-offset walker below to align positionally.
  auto veloxSource = convertToVeloxType(*type);
  const auto encodings = getColumnEncodings(rootRow);
  projection.nimbleType = buildProjectedNimbleType(
      veloxSource->asRow(), projectedSubfields, encodings);

  // Walk the source nimble in the same DFS pre-order + FlatMap-alphabetical
  // traversal the velox-source builder uses, emitting one source stream offset
  // and Row/FlatMap null stream bit per projected stream position (UINT32_MAX
  // for missing keys).
  appendProjectedStream(
      projection.streamOffsets,
      projection.rowOrFlatMapNullStreams,
      rootRow.nullsDescriptor().offset(),
      /*isRowOrFlatMapNullStream=*/true);
  for (size_t columnIdx : selectedColumnIndices) {
    const auto* child = rootRow.childAt(columnIdx).get();
    if (child->kind() == Kind::FlatMap) {
      projectFlatmapStreamOffsets(
          child->asFlatMap(),
          selectedChildren,
          missingChildren,
          projection.streamOffsets,
          projection.rowOrFlatMapNullStreams);
    } else {
      projectStreamOffsets(
          child,
          selectedChildren,
          missingChildren,
          projection.streamOffsets,
          projection.rowOrFlatMapNullStreams);
    }
  }
  NIMBLE_DCHECK_EQ(
      projection.streamOffsets.size(),
      projection.rowOrFlatMapNullStreams.size());

  return projection;
}

} // namespace facebook::nimble
