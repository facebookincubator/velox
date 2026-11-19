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

#include <optional>
#include <string_view>
#include <vector>

#include "velox/dwio/common/ParquetFieldId.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/type/Type.h"

// Field-id based schema resolution for NIMBLE files.
//
// A NIMBLE file can encode stable column identity via a per-type string
// attribute (the "field-id attribute"), independent of column names. The
// concrete attribute key is supplied by the caller (for example, the Iceberg
// connector passes "iceberg.id"), so this resolver carries no format-specific
// knowledge. At read time, when the caller opts in via
// `dwio::common::ColumnMappingMode::kFieldId`, the NIMBLE reader walks the
// file's `facebook::nimble::Type` tree in lockstep with the requested (table)
// Velox schema and the per-column field-id tree, producing two parallel
// schemas, both the full file schema in file column order:
//   - an "exposed" schema: the file schema with columns matched by field id
//     renamed to their requested (table) names, surfaced as the reader's
//     `rowType()` so a caller's by-table-name column matching / null-fill
//     matches projected columns and null-fills requested columns absent from
//     the file. Unmatched file columns (dropped from the table) are retained
//     under their file names so the reader's positional column walk stays
//     aligned with the physical tablet; they are never selected or projected;
//     and
//   - a "reading" schema: the file schema under its on-disk names, kept as the
//     NIMBLE `BatchReader`'s `fileSchema()`.
// This mirrors DWRF's `renameByFieldId` and resolves projected columns even
// after column renames at any nesting level (top-level columns, struct fields,
// array element fields, and map key/value fields), preserves drops, and lets
// the caller null-fill table columns added after the file was written.
//
// These helpers are exposed as a separate library so the recursive walk is
// directly unit-testable without going through a full writer/reader file
// round-trip. They are not intended for callers outside the NIMBLE reader
// + its tests.
namespace facebook::velox::nimble {

// Returns the value of the field-id attribute (named `fieldIdAttributeKey`) on
// a NIMBLE Type, or std::nullopt if the attribute is absent or not parseable
// as an int32.
std::optional<int32_t> fieldIdOf(
    const facebook::nimble::Type& type,
    std::string_view fieldIdAttributeKey);

// Recursively builds the table-named Velox type for `fileType`: a structural
// copy of the file type in file order, with every Row field that matches a
// requested column (by field id) renamed to its requested (table) name, and
// every unmatched file field kept under its on-disk file name (renamed to a
// reserved sentinel only when that name collides with a requested column, so a
// dropped + re-added same-name column is not shadowed). Requested columns with
// no matching file id are simply absent from the result (the caller null-fills
// them downstream); file columns with no requested match are retained under
// their file name so the reader's positional column walk stays aligned with
// the physical schema (they are never selected or projected). Returns
// std::nullopt only on a genuine structural error: a kind mismatch between the
// requested and file type, or a `requestedFieldId.children` count that does
// not match the requested type's shape. The caller falls back to the legacy
// name-based path on std::nullopt.
//
// Dispatches on `requestedType->kind()`:
//   - ROW: file kind must be Row. Iterate the file Row's children in file
//         order; for each file child with a field id matching a requested
//         child, recurse and emit the requested (table) name. Unmatched file
//         children are emitted under their file name with their file-derived
//         type, except a genuinely dropped column (one carrying a field id
//         that matched no requested id) whose file name collides with a
//         requested column name -- that column is emitted under a reserved
//         sentinel name so it cannot shadow a re-added same-name column in the
//         caller's by-name null-fill. Untagged file columns are always kept
//         under their file name. Bails when
//         `requestedFieldId.children.size() != requestedRow.size()`.
//   - ARRAY: file kind must be Array. `requestedFieldId.children` must have
//         exactly 1 child (the element fieldId). Recurse into the file's
//         element type.
//   - MAP: file kind must be Map. `requestedFieldId.children` must have
//         exactly 2 children ([key, value]). Recurse into both.
//   - default (scalar): the file column was already resolved by field id at
//         the parent Row level; nothing to rename at the leaf, so it carries
//         the requested leaf type. A value-shape mismatch (e.g. INTEGER vs
//         BIGINT) is surfaced later by the selector.
std::optional<TypePtr> resolveExposedType(
    const TypePtr& requestedType,
    const dwio::common::ParquetFieldId& requestedFieldId,
    const facebook::nimble::Type& fileType,
    std::string_view fieldIdAttributeKey);

// Result of resolving a full file schema against a requested table schema.
// Both schemas are the full file schema in file column order; they differ
// only in naming. Requested (table) columns absent from the file are not
// present in either schema and are null-filled downstream by the caller.
struct FieldIdSchemaResolution {
  // The file schema with columns matched by field id renamed to their
  // requested (table) names. Becomes the reader's `rowType()`, consumed by the
  // caller for by-name matching and null-fill. Because it preserves file order
  // and shape, the reader's positional column walk stays aligned with the
  // physical tablet.
  RowTypePtr tableNamedSchema;
  // The file schema with its on-disk names. Kept as
  // `ReaderOptions::fileSchema()` (the file-named view of the physical
  // tablet).
  RowTypePtr fileNamedSchema;
};

// Top-level entry point. `requestedFieldIds` is a vector of per-input-column
// field-id sub-trees (one per top-level requested column), as produced by the
// caller from its per-column field metadata. `fieldIdAttributeKey` is the
// attribute key under which the file encodes field ids (e.g. "iceberg.id").
// Returns std::nullopt under the same conditions as
// `resolveExposedType`.
std::optional<FieldIdSchemaResolution> resolveSchema(
    const RowTypePtr& requestedSchema,
    const std::vector<dwio::common::ParquetFieldId>& requestedFieldIds,
    const facebook::nimble::Type& fileSchema,
    std::string_view fieldIdAttributeKey);

} // namespace facebook::velox::nimble
