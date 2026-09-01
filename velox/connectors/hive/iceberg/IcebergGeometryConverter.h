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

#include <string>

#include "velox/functions/prestosql/types/GeometryType.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SelectivityVector.h"

/// Iceberg geometry read support (initial two-dimensional / XY support).
///
/// Per the Iceberg specification a `geometry` column is stored as a Parquet
/// `binary` column with the `GEOMETRY` logical type (ORC: `binary` with
/// `iceberg.binary-type=GEOMETRY`), and the payload is standard ISO WKB.
/// Velox's internal encoding for GEOMETRY() is instead the ESRI-shape derived
/// encoding produced by `common::geospatial::GeometrySerializer`, so the bytes
/// must be re-encoded exactly once on read.
///
/// Conversion ownership: the Iceberg connector owns this conversion, and only
/// the Iceberg connector performs it. `IcebergSplitReader` resolves the
/// affected output channels once per split from the Iceberg-derived reader
/// output type and calls `convertIcebergGeometry` on those channels alone. The
/// format-generic readers (`velox/dwio/parquet`, `velox/dwio/dwrf`, ...) have
/// no knowledge of GEOMETRY: they read the column by its physical file type,
/// which is VARBINARY. Consequently
///   * an unannotated Parquet/ORC `binary` column is never reinterpreted as
///   WKB,
///   * a Hive (non-Iceberg) table is never converted, even if the query type is
///   GEOMETRY,
///   * a file written from an existing Velox GEOMETRY vector is never
///   re-parsed, because such a
///     column is not an Iceberg `geometry` column, and
///   * Gluten/Spark Parquet reads are byte-for-byte unaffected.
namespace facebook::velox::connector::hive::iceberg {

/// True when 'type' is GEOMETRY() or transitively contains it. Cheap and used
/// to decide, once per split, whether any conversion work is needed at all.
inline bool containsGeometry(const TypePtr& type) {
  if (isGeometryType(type)) {
    return true;
  }
  for (auto i = 0; i < type->size(); ++i) {
    if (containsGeometry(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

/// Re-encodes the Iceberg WKB payload of 'input' into Velox's internal geometry
/// encoding and returns a vector whose type is 'targetType' (i.e. carries
/// GEOMETRY() at the geometry positions).
///
/// 'input' holds the bytes as read from the file: a VARBINARY-typed scalar
/// vector for a top-level geometry column, or a ROW/ARRAY/MAP whose leaves are
/// such vectors.
///
/// 'rows' selects the positions of 'input' that are live. Only those positions
/// are parsed; everything else is left null in the result. Nothing about the
/// input's layout is assumed: for an ARRAY or MAP the live element positions
/// are derived from its offsets and sizes (so non-zero offsets, gaps and slices
/// are handled), and for a dictionary only the referenced dictionary entries
/// are parsed. Pass an all-selected vector of size 'input->size()' when every
/// position is live, which is what a table scan output is.
///
/// Encodings are preserved where that is safe and cheaper:
///   * FLAT      -> a new flat GEOMETRY vector; nulls are preserved.
///   * DICTIONARY-> the referenced dictionary values are converted once into a
///   *new* dictionary (the
///                  shared Parquet dictionary is never mutated) and the
///                  indices/nulls are re-wrapped, so a repeated value is parsed
///                  once per dictionary entry, not once per row.
///   * CONSTANT  -> a converted constant, or a null constant of 'targetType'.
///                  The single value is parsed once rather than once per row,
///                  and is not parsed at all when 'rows' selects nothing, which
///                  yields a null constant. Because the result is a constant,
///                  positions outside 'rows' carry the converted value rather
///                  than a null.
///   * ROW/ARRAY/MAP -> only the children that contain geometry are rebuilt;
///   offsets, sizes and
///                  nulls are shared with the input.
///
/// 'columnPath' is used verbatim in error messages, e.g. `geom`, `nested.geom`,
/// `arr[]`, `m[value]`.
///
/// Throws a user error when a live value is not WKB, when it uses Z/M
/// coordinates or EWKB (which Velox's two-dimensional GEOMETRY cannot
/// represent), or when the geometry kind is unsupported. Dimensions are never
/// silently dropped.
VectorPtr convertIcebergGeometry(
    const VectorPtr& input,
    const TypePtr& targetType,
    const SelectivityVector& rows,
    memory::MemoryPool* pool,
    const std::string& columnPath);

/// Convenience overload for the table-scan case, where every position of
/// 'input' is live.
VectorPtr convertIcebergGeometry(
    const VectorPtr& input,
    const TypePtr& targetType,
    memory::MemoryPool* pool,
    const std::string& columnPath);

} // namespace facebook::velox::connector::hive::iceberg
