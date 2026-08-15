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

// Adapted from Apache Arrow.

#pragma once

#include <cassert>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"

#include "velox/dwio/parquet/common/LevelConversion.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"

namespace facebook::velox::parquet::arrow {

class ArrowReaderProperties;
class ArrowWriterProperties;
class WriterProperties;

namespace arrow {

/// \defgroup arrow-to-parquet-schema-conversion Functions to convert an Arrow
/// schema into a Parquet schema.
///
/// @{

PARQUET_EXPORT
::arrow::Status fieldToNode(
    const std::shared_ptr<::arrow::Field>& field,
    const WriterProperties& properties,
    const ArrowWriterProperties& arrowProperties,
    schema::NodePtr* out);

PARQUET_EXPORT
::arrow::Status toParquetSchema(
    const ::arrow::Schema* arrowSchema,
    const WriterProperties& properties,
    const ArrowWriterProperties& arrowProperties,
    std::shared_ptr<SchemaDescriptor>* out);

PARQUET_EXPORT
::arrow::Status toParquetSchema(
    const ::arrow::Schema* arrowSchema,
    const WriterProperties& properties,
    std::shared_ptr<SchemaDescriptor>* out);

/// @}

/// \defgroup parquet-to-arrow-schema-conversion Functions to convert a Parquet
/// schema into an arrow schema.
///
/// @{

PARQUET_EXPORT
::arrow::Status fromParquetSchema(
    const SchemaDescriptor* parquetSchema,
    const ArrowReaderProperties& properties,
    const std::shared_ptr<const ::arrow::KeyValueMetadata>& keyValueMetadata,
    std::shared_ptr<::arrow::Schema>* out);

PARQUET_EXPORT
::arrow::Status fromParquetSchema(
    const SchemaDescriptor* parquetSchema,
    const ArrowReaderProperties& properties,
    std::shared_ptr<::arrow::Schema>* out);

PARQUET_EXPORT
::arrow::Status fromParquetSchema(
    const SchemaDescriptor* parquetSchema,
    std::shared_ptr<::arrow::Schema>* out);

/// @}

/// \brief Bridge between an arrow::Field and Parquet column indices.
struct PARQUET_EXPORT SchemaField {
  std::shared_ptr<::arrow::Field> field;
  std::vector<SchemaField> children;

  // Only set for leaf nodes.
  int columnIndex = -1;

  LevelInfo levelInfo;

  bool isLeaf() const {
    return columnIndex != -1;
  }
};

/// \brief Bridge between a parquet Schema and an arrow Schema.
///
/// Expose Parquet columns as a tree structure. Useful to traverse and link
/// between Arrow and Parquet schemas.
struct PARQUET_EXPORT SchemaManifest {
  static ::arrow::Status make(
      const SchemaDescriptor* schema,
      const std::shared_ptr<const ::arrow::KeyValueMetadata>& metadata,
      const ArrowReaderProperties& properties,
      SchemaManifest* manifest);

  const SchemaDescriptor* descr;
  std::shared_ptr<::arrow::Schema> originSchema;
  std::shared_ptr<const ::arrow::KeyValueMetadata> schemaMetadata;
  std::vector<SchemaField> schemaFields;

  std::unordered_map<int, const SchemaField*> columnIndexToField;
  std::unordered_map<const SchemaField*, const SchemaField*> childToParent;

  ::arrow::Status getColumnField(int columnIndex, const SchemaField** out)
      const {
    auto it = columnIndexToField.find(columnIndex);
    if (it == columnIndexToField.end()) {
      return ::arrow::Status::KeyError(
          "Column index ",
          columnIndex,
          " not found in schema manifest, may be malformed");
    }
    *out = it->second;
    return ::arrow::Status::OK();
  }

  const SchemaField* getParent(const SchemaField* field) const {
    // Returns nullptr also if not found.
    auto it = childToParent.find(field);
    if (it == childToParent.end()) {
      return NULLPTR;
    }
    return it->second;
  }

  /// Coalesce a list of field indices (relative to the equivalent
  /// Arrow schema) which correspond to the column root (first node below the
  /// Parquet schema's root group) of each leaf referenced in columnIndices.
  ///
  /// For example, for leaves `a.b.c`, `a.b.d.e`, and `i.j.k`.
  /// (Column_indices=[0,1,3]) the roots are `a` and `i` (return=[0,2]).
  ///
  /// Root.
  /// -- A  <------.
  /// -- -- B  |  |.
  /// -- -- -- C  |.
  /// -- -- -- D  |.
  /// -- -- -- -- E.
  /// -- F.
  /// -- -- G.
  /// -- -- -- H.
  /// -- I  <---.
  /// -- -- J  |.
  /// -- -- -- K.
  ::arrow::Result<std::vector<int>> getFieldIndices(
      const std::vector<int>& columnIndices) const {
    const schema::GroupNode* group = descr->groupNode();
    std::unordered_set<int> alreadyAdded;

    std::vector<int> out;
    for (int columnIdx : columnIndices) {
      if (columnIdx < 0 || columnIdx >= descr->numColumns()) {
        return ::arrow::Status::IndexError(
            "Column index ", columnIdx, " is not valid");
      }

      auto fieldNode = descr->getColumnRoot(columnIdx);
      auto fieldIdx = group->fieldIndex(*fieldNode);
      if (fieldIdx == -1) {
        return ::arrow::Status::IndexError(
            "Column index ", columnIdx, " is not valid");
      }

      if (alreadyAdded.insert(fieldIdx).second) {
        out.push_back(fieldIdx);
      }
    }
    return out;
  }
};

std::shared_ptr<::arrow::KeyValueMetadata> fieldIdMetadata(int32_t fieldId);

} // namespace arrow
} // namespace facebook::velox::parquet::arrow
