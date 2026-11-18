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
#include "velox/dwio/nimble/reader/FieldIdResolver.h"

#include <folly/Conv.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include "velox/dwio/nimble/common/SchemaUtils.h"

namespace {
// Reserved prefix for the synthetic name given to a dropped (unmatched) file
// column whose on-disk name collides with a requested (table) column name.
// Renaming such columns keeps the exposed schema positionally aligned with the
// physical tablet while ensuring a re-added same-name column (new field id) is
// not shadowed and is null-filled by the connector. Dropped columns are never
// selected or projected, so this name is never user-visible.
constexpr std::string_view kDroppedColumnNamePrefix{"$nimble_dropped_"};
} // namespace

namespace facebook::velox::nimble {

std::optional<int32_t> fieldIdOf(
    const facebook::nimble::Type& type,
    std::string_view fieldIdAttributeKey) {
  for (const auto& [key, value] : type.attributes()) {
    if (key == fieldIdAttributeKey) {
      try {
        return folly::to<int32_t>(value);
      } catch (const folly::ConversionError&) {
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

std::optional<TypePtr> resolveExposedType(
    const TypePtr& requestedType,
    const dwio::common::ParquetFieldId& requestedFieldId,
    const facebook::nimble::Type& fileType,
    std::string_view fieldIdAttributeKey) {
  switch (requestedType->kind()) {
    case TypeKind::ROW: {
      if (!fileType.isRow()) {
        return std::nullopt;
      }
      const auto& requestedRow = requestedType->asRow();
      const auto& fileRow = fileType.asRow();
      if (requestedFieldId.children.size() != requestedRow.size()) {
        return std::nullopt;
      }
      // Index the requested Row children by their `iceberg.id`, so each file
      // child (walked in file order below) can find its requested
      // counterpart. Requested children whose id has no match in the file are
      // never visited -- they are added columns the connector null-fills.
      folly::F14FastMap<int32_t, size_t> requestedChildrenById;
      requestedChildrenById.reserve(requestedRow.size());
      for (size_t i = 0; i < requestedRow.size(); ++i) {
        requestedChildrenById.emplace(requestedFieldId.children[i].fieldId, i);
      }
      // Set of requested (table) names at this level. Used to detect when a
      // dropped file column's on-disk name collides with a re-added requested
      // column so it can be renamed to a non-colliding sentinel.
      folly::F14FastSet<std::string_view> requestedNames;
      requestedNames.reserve(requestedRow.size());
      for (size_t i = 0; i < requestedRow.size(); ++i) {
        requestedNames.insert(requestedRow.nameOf(i));
      }
      // Emit every file child in file order so the result stays positionally
      // aligned with the physical tablet. Matched children adopt the
      // requested (table) name and recurse; unmatched (dropped) file children
      // keep their file name and file-derived type, unless that name collides
      // with a requested column (drop + re-add same name with a new field id),
      // in which case the dropped column is renamed to a sentinel so it does
      // not shadow the re-added column in the connector's by-name null-fill.
      std::vector<std::string> names;
      std::vector<TypePtr> children;
      names.reserve(fileRow.childrenCount());
      children.reserve(fileRow.childrenCount());
      for (size_t f = 0; f < fileRow.childrenCount(); ++f) {
        const auto& fileChild = *fileRow.childAt(f);
        const auto fileId = fieldIdOf(fileChild, fieldIdAttributeKey);
        std::optional<size_t> requestedIdx;
        if (fileId.has_value()) {
          const auto it = requestedChildrenById.find(*fileId);
          if (it != requestedChildrenById.end()) {
            requestedIdx = it->second;
          }
        }
        if (requestedIdx.has_value()) {
          auto resolvedChild = resolveExposedType(
              requestedRow.childAt(*requestedIdx),
              requestedFieldId.children[*requestedIdx],
              fileChild,
              fieldIdAttributeKey);
          if (!resolvedChild.has_value()) {
            return std::nullopt;
          }
          names.push_back(requestedRow.nameOf(*requestedIdx));
          children.push_back(std::move(resolvedChild.value()));
        } else {
          // Unmatched file child. Keep its on-disk name so the reader's
          // positional walk stays aligned -- except a genuinely dropped column
          // (one that carries an `iceberg.id` which did not match any requested
          // id) whose name collides with a requested column. Exposing such a
          // column under that name would shadow the re-added same-name column
          // (new field id) in the connector's by-name null-fill, so rename it
          // to a reserved, non-projectable sentinel. Untagged file columns
          // (no `iceberg.id`) are left under their file name for legacy
          // name-based resolution.
          const auto& fileName = fileRow.nameAt(f);
          if (fileId.has_value() && requestedNames.count(fileName) > 0) {
            names.push_back(
                folly::to<std::string>(kDroppedColumnNamePrefix, f));
          } else {
            names.push_back(fileName);
          }
          children.push_back(facebook::nimble::convertToVeloxType(fileChild));
        }
      }
      return ROW(std::move(names), std::move(children));
    }
    case TypeKind::ARRAY: {
      if (!fileType.isArray()) {
        return std::nullopt;
      }
      // The Java `ColumnIdentity` for an Iceberg ListType always emits
      // exactly one child (the element identity). Mirrors the upstream
      // protocol shape encoded by `toParquetField` in
      // `IcebergPrestoToVeloxConnector.cpp`.
      if (requestedFieldId.children.size() != 1) {
        return std::nullopt;
      }
      auto resolvedElement = resolveExposedType(
          requestedType->asArray().elementType(),
          requestedFieldId.children[0],
          *fileType.asArray().elements(),
          fieldIdAttributeKey);
      if (!resolvedElement.has_value()) {
        return std::nullopt;
      }
      return ARRAY(std::move(resolvedElement.value()));
    }
    case TypeKind::MAP: {
      if (!fileType.isMap()) {
        return std::nullopt;
      }
      // The Java `ColumnIdentity` for an Iceberg MapType emits two
      // children in [key, value] order. Mirrors the upstream protocol
      // shape encoded by `toParquetField`.
      if (requestedFieldId.children.size() != 2) {
        return std::nullopt;
      }
      auto resolvedKey = resolveExposedType(
          requestedType->asMap().keyType(),
          requestedFieldId.children[0],
          *fileType.asMap().keys(),
          fieldIdAttributeKey);
      if (!resolvedKey.has_value()) {
        return std::nullopt;
      }
      auto resolvedValue = resolveExposedType(
          requestedType->asMap().valueType(),
          requestedFieldId.children[1],
          *fileType.asMap().values(),
          fieldIdAttributeKey);
      if (!resolvedValue.has_value()) {
        return std::nullopt;
      }
      return MAP(
          std::move(resolvedKey.value()), std::move(resolvedValue.value()));
    }
    default:
      // Scalar / primitive. The file column was already resolved by
      // `iceberg.id` at the parent Row level; nothing to rename at the
      // leaf, so it carries the requested leaf type. A value-shape mismatch
      // (e.g. INTEGER vs BIGINT) is surfaced later by the selective reader's
      // type-coercion check.
      return requestedType;
  }
}

std::optional<FieldIdSchemaResolution> resolveSchema(
    const RowTypePtr& requestedSchema,
    const std::vector<dwio::common::ParquetFieldId>& requestedFieldIds,
    const facebook::nimble::Type& fileSchema,
    std::string_view fieldIdAttributeKey) {
  // Synthesize a top-level ParquetFieldId wrapping the per-column
  // field-id tree, mirroring the wrapper shape produced by
  // `IcebergDataSink::createWriterOptions`. The wrapper's own `fieldId`
  // is never matched -- only its `children` are consumed by the Row
  // dispatch arm below.
  dwio::common::ParquetFieldId rootFieldId{
      .fieldId = -1, .children = requestedFieldIds};
  auto exposed = resolveExposedType(
      requestedSchema, rootFieldId, fileSchema, fieldIdAttributeKey);
  if (!exposed.has_value()) {
    return std::nullopt;
  }
  auto tableNamedSchema =
      std::dynamic_pointer_cast<const RowType>(exposed.value());
  if (tableNamedSchema == nullptr) {
    return std::nullopt;
  }
  // The reading schema is just the file schema in its on-disk names.
  auto fileNamedSchema = std::dynamic_pointer_cast<const RowType>(
      facebook::nimble::convertToVeloxType(fileSchema));
  if (fileNamedSchema == nullptr) {
    return std::nullopt;
  }
  return FieldIdSchemaResolution{
      .tableNamedSchema = std::move(tableNamedSchema),
      .fileNamedSchema = std::move(fileNamedSchema)};
}

} // namespace facebook::velox::nimble
