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

#include "velox/connectors/hive/iceberg/IcebergGeometryConverter.h"

#include <cstdint>
#include <memory>

#include <fmt/format.h>

#define USE_UNSTABLE_GEOS_CPP_API 1
#include <geos/io/WKBReader.h>

#include "velox/common/geospatial/GeometrySerde.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

// EWKB (PostGIS) flags in the 32-bit WKB geometry type word. Iceberg mandates
// ISO WKB, which never sets them, so their presence means the payload does not
// honor the Iceberg contract.
constexpr uint32_t kEwkbFlagZ = 0x8000'0000;
constexpr uint32_t kEwkbFlagM = 0x4000'0000;
constexpr uint32_t kEwkbFlagSrid = 0x2000'0000;

constexpr int32_t kWkbHeaderLength = 5;
constexpr uint8_t kWkbBigEndian = 0;
constexpr uint8_t kWkbLittleEndian = 1;
// ISO WKB geometry codes 1..7 are Point .. GeometryCollection. Higher codes
// (CircularString, CompoundCurve, CurvePolygon, MultiCurve, MultiSurface,
// PolyhedralSurface, TIN, Triangle) belong to ISO SQL/MM and are not
// representable by Velox's GEOMETRY.
constexpr uint32_t kMaxSupportedWkbGeometryCode = 7;

const char* dimensionName(uint32_t dimensions) {
  switch (dimensions) {
    case 1:
      return "Z";
    case 2:
      return "M";
    case 3:
      return "Z and M";
    default:
      return "unknown extra";
  }
}

// Rejects payloads that Velox's two-dimensional, SRID-less GEOMETRY cannot
// represent faithfully, rather than silently dropping dimensions. Only the
// outermost header is inspected: in conforming ISO WKB the dimensionality of a
// collection is declared on the collection itself, so a 2D header implies 2D
// children.
void validateIsoWkb(StringView wkb, const std::string& columnPath) {
  VELOX_USER_CHECK_GE(
      wkb.size(),
      kWkbHeaderLength,
      "Invalid well-known binary (WKB) in Iceberg geometry column '{}': expected at least {} bytes, found {}",
      columnPath,
      kWkbHeaderLength,
      wkb.size());

  const auto* data = reinterpret_cast<const uint8_t*>(wkb.data());
  uint32_t typeCode;
  if (data[0] == kWkbLittleEndian) {
    typeCode = static_cast<uint32_t>(data[1]) |
        (static_cast<uint32_t>(data[2]) << 8) |
        (static_cast<uint32_t>(data[3]) << 16) |
        (static_cast<uint32_t>(data[4]) << 24);
  } else if (data[0] == kWkbBigEndian) {
    typeCode = (static_cast<uint32_t>(data[1]) << 24) |
        (static_cast<uint32_t>(data[2]) << 16) |
        (static_cast<uint32_t>(data[3]) << 8) | static_cast<uint32_t>(data[4]);
  } else {
    VELOX_USER_FAIL(
        "Invalid well-known binary (WKB) in Iceberg geometry column '{}': unknown byte order marker {}",
        columnPath,
        static_cast<int32_t>(data[0]));
  }

  VELOX_USER_CHECK_EQ(
      typeCode & (kEwkbFlagZ | kEwkbFlagM | kEwkbFlagSrid),
      0,
      "Iceberg geometry column '{}' is encoded as extended WKB (EWKB); the Iceberg specification requires ISO WKB",
      columnPath);

  const uint32_t dimensions = typeCode / 1000;
  VELOX_USER_CHECK_EQ(
      dimensions,
      0,
      "Iceberg geometry column '{}' contains {} coordinates; Velox GEOMETRY supports only two-dimensional (XY) geometries",
      columnPath,
      dimensionName(dimensions));

  const uint32_t geometryCode = typeCode % 1000;
  VELOX_USER_CHECK(
      geometryCode >= 1 && geometryCode <= kMaxSupportedWkbGeometryCode,
      "Iceberg geometry column '{}' contains an unsupported WKB geometry type code {}",
      columnPath,
      geometryCode);
}

// Parses one WKB value and appends Velox's internal geometry encoding to
// 'out'.
void wkbToVeloxGeometry(
    StringView wkb,
    geos::io::WKBReader& wkbReader,
    std::string& out,
    const std::string& columnPath) {
  validateIsoWkb(wkb, columnPath);
  std::unique_ptr<geos::geom::Geometry> geometry;
  try {
    geometry = wkbReader.read(
        reinterpret_cast<const unsigned char*>(wkb.data()), wkb.size());
  } catch (const std::exception& e) {
    VELOX_USER_FAIL(
        "Invalid well-known binary (WKB) in Iceberg geometry column '{}': {}",
        columnPath,
        e.what());
  }
  // Standard WKB carries no SRID and Velox's GEOMETRY has no SRID concept:
  // never invent one. GeometrySerializer must not run inside a GEOS_TRY: it
  // needs its exceptions to bubble up.
  common::geospatial::GeometrySerializer::serialize(*geometry, out);
}

// Converts the live positions of a flat scalar vector of WKB payloads into a
// new flat GEOMETRY vector. Positions outside 'rows', and null positions, are
// left null; their bytes are never read.
VectorPtr convertGeometryLeaf(
    const VectorPtr& input,
    const SelectivityVector& rows,
    memory::MemoryPool* pool,
    const std::string& columnPath) {
  const auto size = input->size();
  auto* flatInput = input->asFlatVector<StringView>();
  VELOX_CHECK_NOT_NULL(
      flatInput,
      "Expected a flat scalar vector for Iceberg geometry column '{}', got {}",
      columnPath,
      input->encoding());

  auto result =
      BaseVector::create<FlatVector<StringView>>(GEOMETRY(), size, pool);
  // Start from all-null so unselected positions carry no bytes at all.
  for (vector_size_t i = 0; i < size; ++i) {
    result->setNull(i, true);
  }

  geos::io::WKBReader wkbReader;
  std::string serialized;
  rows.applyToSelected([&](vector_size_t i) {
    if (flatInput->isNullAt(i)) {
      return;
    }
    serialized.clear();
    wkbToVeloxGeometry(
        flatInput->valueAt(i), wkbReader, serialized, columnPath);
    // set() copies non-inline values into the result's own string buffers, so
    // the result never aliases the reader's (recycled) page buffers.
    result->set(i, StringView(serialized));
  });
  return result;
}

// Selects the element positions referenced by the live rows of an ARRAY or
// MAP. Offsets and sizes are read per row, so non-zero offsets, unreferenced
// gaps and sliced vectors are all handled and no packed layout is assumed.
SelectivityVector selectReferencedElements(
    const ArrayVectorBase& vector,
    const SelectivityVector& rows,
    vector_size_t elementsSize) {
  SelectivityVector selected(elementsSize, false);
  const auto* rawOffsets = vector.rawOffsets();
  const auto* rawSizes = vector.rawSizes();
  rows.applyToSelected([&](vector_size_t row) {
    if (vector.isNullAt(row)) {
      return;
    }
    const auto offset = rawOffsets[row];
    const auto size = rawSizes[row];
    VELOX_CHECK_LE(offset + size, elementsSize);
    for (vector_size_t i = 0; i < size; ++i) {
      selected.setValid(offset + i, true);
    }
  });
  selected.updateBounds();
  return selected;
}

// Selects the dictionary entries referenced by the live rows of a dictionary
// vector.
SelectivityVector selectReferencedDictionaryEntries(
    const BaseVector& dictionary,
    const SelectivityVector& rows,
    vector_size_t baseSize) {
  SelectivityVector selected(baseSize, false);
  const auto* indices = dictionary.wrapInfo()->as<vector_size_t>();
  rows.applyToSelected([&](vector_size_t row) {
    if (dictionary.isNullAt(row)) {
      return;
    }
    const auto index = indices[row];
    VELOX_CHECK_LT(index, baseSize);
    selected.setValid(index, true);
  });
  selected.updateBounds();
  return selected;
}

} // namespace

VectorPtr convertIcebergGeometry(
    const VectorPtr& input,
    const TypePtr& targetType,
    const SelectivityVector& rows,
    memory::MemoryPool* pool,
    const std::string& columnPath) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK(containsGeometry(targetType));

  const auto& loaded = BaseVector::loadedVectorShared(input);

  switch (loaded->encoding()) {
    case VectorEncoding::Simple::CONSTANT: {
      if (loaded->isNullAt(0)) {
        return BaseVector::createNullConstant(targetType, loaded->size(), pool);
      }
      // No live position, so the constant's single value is unreachable and
      // must not be parsed.
      if (!rows.hasSelections()) {
        return BaseVector::createNullConstant(targetType, loaded->size(), pool);
      }
      // Every live row of a constant reads the same value, so parse it exactly
      // once and re-wrap, rather than materializing and re-parsing it per row.
      // Copying position 0 into a one-row vector keeps this agnostic to whether
      // the constant is scalar or complex: the recursive call lands in the leaf
      // or in the ROW/ARRAY/MAP handling below exactly as a flat input would.
      //
      // A non-null constant geometry does not arise from a scan today (per the
      // Iceberg spec a geometry column can be neither a partition source nor a
      // non-null initial default), but preserving the encoding keeps this
      // correct and O(1) if a scan later emits CONSTANT for a uniform-value
      // column.
      auto base = BaseVector::create(loaded->type(), 1, pool);
      base->copy(loaded.get(), 0, 0, 1);
      SelectivityVector singleRow(1);
      auto converted =
          convertIcebergGeometry(base, targetType, singleRow, pool, columnPath);
      // The re-encoded value outlives this call either way: for a scalar
      // geometry ConstantVector copies the string into its own buffer and drops
      // the base, and for a complex type it retains the one-row base vector.
      return BaseVector::wrapInConstant(
          loaded->size(), 0, std::move(converted));
    }

    case VectorEncoding::Simple::DICTIONARY: {
      // Convert only the referenced dictionary entries, once each rather than
      // once per row, and never in place: the Parquet dictionary is shared
      // across batches and column readers.
      const auto& base = loaded->valueVector();
      auto baseRows =
          selectReferencedDictionaryEntries(*loaded, rows, base->size());
      auto convertedValues =
          convertIcebergGeometry(base, targetType, baseRows, pool, columnPath);
      return BaseVector::wrapInDictionary(
          loaded->nulls(),
          loaded->wrapInfo(),
          loaded->size(),
          std::move(convertedValues));
    }

    default:
      break;
  }

  if (isGeometryType(targetType)) {
    return convertGeometryLeaf(loaded, rows, pool, columnPath);
  }

  switch (targetType->kind()) {
    case TypeKind::ROW: {
      auto* row = loaded->as<RowVector>();
      VELOX_CHECK_NOT_NULL(
          row,
          "Expected a RowVector for Iceberg geometry column '{}'",
          columnPath);
      const auto& rowType = targetType->asRow();
      std::vector<VectorPtr> children = row->children();
      for (auto i = 0; i < rowType.size(); ++i) {
        if (containsGeometry(rowType.childAt(i))) {
          // A row's children are positionally aligned with the row itself, so
          // the live rows carry over unchanged.
          children[i] = convertIcebergGeometry(
              children[i],
              rowType.childAt(i),
              rows,
              pool,
              fmt::format("{}.{}", columnPath, rowType.nameOf(i)));
        }
      }
      return std::make_shared<RowVector>(
          pool, targetType, row->nulls(), row->size(), std::move(children));
    }

    case TypeKind::ARRAY: {
      auto* array = loaded->as<ArrayVector>();
      VELOX_CHECK_NOT_NULL(
          array,
          "Expected an ArrayVector for Iceberg geometry column '{}'",
          columnPath);
      const auto& inputElements = array->elements();
      auto elementRows =
          selectReferencedElements(*array, rows, inputElements->size());
      auto elements = convertIcebergGeometry(
          inputElements,
          targetType->childAt(0),
          elementRows,
          pool,
          fmt::format("{}[]", columnPath));
      return std::make_shared<ArrayVector>(
          pool,
          targetType,
          array->nulls(),
          array->size(),
          array->offsets(),
          array->sizes(),
          std::move(elements));
    }

    case TypeKind::MAP: {
      auto* map = loaded->as<MapVector>();
      VELOX_CHECK_NOT_NULL(
          map,
          "Expected a MapVector for Iceberg geometry column '{}'",
          columnPath);
      auto keys = map->mapKeys();
      auto values = map->mapValues();
      const auto entryRows =
          selectReferencedElements(*map, rows, values->size());
      if (containsGeometry(targetType->childAt(0))) {
        keys = convertIcebergGeometry(
            keys,
            targetType->childAt(0),
            entryRows,
            pool,
            fmt::format("{}[key]", columnPath));
      }
      if (containsGeometry(targetType->childAt(1))) {
        values = convertIcebergGeometry(
            values,
            targetType->childAt(1),
            entryRows,
            pool,
            fmt::format("{}[value]", columnPath));
      }
      return std::make_shared<MapVector>(
          pool,
          targetType,
          map->nulls(),
          map->size(),
          map->offsets(),
          map->sizes(),
          std::move(keys),
          std::move(values));
    }

    default:
      VELOX_UNREACHABLE(
          "Unexpected type {} for Iceberg geometry column '{}'",
          targetType->toString(),
          columnPath);
  }
}

VectorPtr convertIcebergGeometry(
    const VectorPtr& input,
    const TypePtr& targetType,
    memory::MemoryPool* pool,
    const std::string& columnPath) {
  SelectivityVector allRows(input->size());
  return convertIcebergGeometry(input, targetType, allRows, pool, columnPath);
}

} // namespace facebook::velox::connector::hive::iceberg
