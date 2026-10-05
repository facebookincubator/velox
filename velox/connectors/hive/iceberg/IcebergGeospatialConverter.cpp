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

#include "velox/connectors/hive/iceberg/IcebergGeospatialConverter.h"

#include <cstdint>
#include <memory>
#include <string_view>

#include <fmt/format.h>

#define USE_UNSTABLE_GEOS_CPP_API 1
#include <geos/geom/CoordinateFilter.h>
#include <geos/io/WKBReader.h>

#include "velox/common/geospatial/GeometrySerde.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

// Names the geospatial type a column is read as, for error messages.
struct GeospatialKind {
  // Iceberg type name, e.g. "geometry".
  std::string_view columnKind;
  // Velox type name, e.g. "GEOMETRY".
  std::string_view typeName;
  // Whether coordinates must be WGS84 longitude/latitude.
  bool sphericalCoordinates;
};

constexpr GeospatialKind kGeometryKind{"geometry", "GEOMETRY", false};
constexpr GeospatialKind kGeographyKind{
    "geography",
    "SPHERICALGEOGRAPHY",
    true,
};

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

// ISO WKB geometry type codes.
constexpr uint32_t kWkbPoint = 1;
constexpr uint32_t kWkbLineString = 2;
constexpr uint32_t kWkbPolygon = 3;
constexpr uint32_t kWkbMultiPoint = 4;
constexpr uint32_t kWkbMultiLineString = 5;
constexpr uint32_t kWkbMultiPolygon = 6;
constexpr uint32_t kWkbGeometryCollection = 7;

// An XY coordinate: two IEEE-754 doubles.
constexpr uint32_t kXyCoordinateSize = 16;
constexpr uint32_t kUint32Size = 4;

// Maximum nesting depth accepted for a WKB payload, counting the outermost
// geometry as depth 1. This mirrors the MAX_PARSE_DEPTH cap that newer GEOS
// releases apply in WKBReader::readGeometry().
//
// The GEOS version Velox pins (3.10.7, see CMake/resolve_dependency_modules/
// geos.cmake) has no such cap and recurses once per nesting level, so a
// deeply nested GEOMETRYCOLLECTION -- roughly nine bytes per level -- can
// exhaust the stack inside GEOS from a payload small enough to look
// unremarkable. Bounding the depth here rejects such a payload before it
// reaches the parser, and simultaneously bounds this validator's own
// recursion.
constexpr uint32_t kMaxParseDepth = 100;

// Walks a WKB payload and validates the header of every geometry in it,
// including the nested geometries of MULTIPOINT, MULTILINESTRING, MULTIPOLYGON
// and GEOMETRYCOLLECTION, which each carry their own independent WKB header.
//
// Checking only the outermost header is not sufficient: a collection whose own
// type word says XY may contain a Z, M, ZM or EWKB child. GEOS parses such a
// payload happily and GeometrySerializer then writes X and Y only, silently
// discarding the extra ordinate. Validating recursively before parsing is what
// keeps the XY-only contract honest, and it is why the dimensionality is not
// re-derived from the parsed GEOS geometry afterwards.
//
// Only headers and element counts are read; coordinate runs are skipped
// arithmetically, so the cost is proportional to the number of nested
// geometries rather than to the number of coordinates. Every read is bounds
// checked, so a truncated or malformed payload produces a user error rather
// than an out-of-bounds read.
class WkbHeaderValidator {
 public:
  WkbHeaderValidator(
      StringView wkb,
      const GeospatialKind& kind,
      const std::string& columnPath)
      : wkb_{wkb}, kind_{kind}, columnPath_{columnPath} {}

  void validate() {
    validateGeometry(/*depth=*/1);
    // Trailing bytes mean the payload does not describe exactly one geometry.
    VELOX_USER_CHECK_EQ(
        position_,
        size(),
        "Invalid well-known binary (WKB) in Iceberg {} column '{}': {} trailing byte(s) after the geometry",
        kind_.columnKind,
        columnPath_,
        size() - position_);
  }

 private:
  const uint8_t* data() const {
    return reinterpret_cast<const uint8_t*>(wkb_.data());
  }

  uint64_t size() const {
    return wkb_.size();
  }

  // Validates one geometry at the current position and advances past it.
  // 'depth' is the nesting level of this geometry, with the outermost at 1.
  void validateGeometry(uint32_t depth) {
    VELOX_USER_CHECK_LE(
        depth,
        kMaxParseDepth,
        "Iceberg {} column '{}' is nested more than {} levels deep; deeper nesting is rejected because it can exhaust the stack while parsing",
        kind_.columnKind,
        columnPath_,
        kMaxParseDepth);

    const bool littleEndian = readByteOrder();
    const uint32_t typeCode = readUint32(littleEndian);

    VELOX_USER_CHECK_EQ(
        typeCode & (kEwkbFlagZ | kEwkbFlagM | kEwkbFlagSrid),
        0,
        "Iceberg {} column '{}' is encoded as extended WKB (EWKB); the Iceberg specification requires ISO WKB",
        kind_.columnKind,
        columnPath_);

    const uint32_t dimensions = typeCode / 1000;
    VELOX_USER_CHECK_EQ(
        dimensions,
        0,
        "Iceberg {} column '{}' contains {} coordinates; Velox {} supports only two-dimensional (XY) geometries",
        kind_.columnKind,
        columnPath_,
        dimensionName(dimensions),
        kind_.typeName);

    const uint32_t geometryCode = typeCode % 1000;
    VELOX_USER_CHECK(
        geometryCode >= 1 && geometryCode <= kMaxSupportedWkbGeometryCode,
        "Iceberg {} column '{}' contains an unsupported WKB geometry type code {}",
        kind_.columnKind,
        columnPath_,
        geometryCode);

    switch (geometryCode) {
      case kWkbPoint:
        // A lone point has no count; an empty point is written as NaN NaN.
        skip(kXyCoordinateSize);
        break;
      case kWkbLineString:
        skipCoordinates(readUint32(littleEndian));
        break;
      case kWkbPolygon: {
        const uint32_t numRings = readUint32(littleEndian);
        for (uint32_t i = 0; i < numRings; ++i) {
          skipCoordinates(readUint32(littleEndian));
        }
        break;
      }
      case kWkbMultiPoint:
      case kWkbMultiLineString:
      case kWkbMultiPolygon:
      case kWkbGeometryCollection: {
        // Every child is a complete WKB geometry with its own byte order and
        // type word, so recurse rather than skipping coordinates. This is the
        // path that catches a Z/M/ZM/EWKB child under an XY parent.
        const uint32_t numChildren = readUint32(littleEndian);
        for (uint32_t i = 0; i < numChildren; ++i) {
          validateGeometry(depth + 1);
        }
        break;
      }
      default:
        VELOX_UNREACHABLE();
    }
  }

  // Returns true for little endian. Rejects any other marker.
  bool readByteOrder() {
    require(1);
    const uint8_t marker = data()[position_++];
    if (marker == kWkbLittleEndian) {
      return true;
    }
    if (marker == kWkbBigEndian) {
      return false;
    }
    VELOX_USER_FAIL(
        "Invalid well-known binary (WKB) in Iceberg {} column '{}': unknown byte order marker {}",
        kind_.columnKind,
        columnPath_,
        static_cast<int32_t>(marker));
  }

  uint32_t readUint32(bool littleEndian) {
    require(kUint32Size);
    const uint8_t* p = data() + position_;
    position_ += kUint32Size;
    if (littleEndian) {
      return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
          (static_cast<uint32_t>(p[2]) << 16) |
          (static_cast<uint32_t>(p[3]) << 24);
    }
    return (static_cast<uint32_t>(p[0]) << 24) |
        (static_cast<uint32_t>(p[1]) << 16) |
        (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
  }

  // Skips 'count' XY coordinates. The multiplication is done in 64 bits so a
  // corrupt count cannot overflow into a small value that passes the check.
  void skipCoordinates(uint32_t count) {
    skip(static_cast<uint64_t>(count) * kXyCoordinateSize);
  }

  void skip(uint64_t numBytes) {
    require(numBytes);
    position_ += numBytes;
  }

  void require(uint64_t numBytes) {
    VELOX_USER_CHECK_LE(
        numBytes,
        size() - position_,
        "Invalid well-known binary (WKB) in Iceberg {} column '{}': truncated payload, need {} more byte(s) at offset {} of {}",
        kind_.columnKind,
        columnPath_,
        numBytes,
        position_,
        size());
  }

  // Held by value rather than as a cached data()/size() pair: a StringView of
  // 12 bytes or fewer stores its characters inline, so data() points into the
  // StringView object itself. Caching the pointer from a by-value constructor
  // parameter left it dangling as soon as the constructor returned, which ASAN
  // reported as a stack-use-after-scope for short payloads such as the EMPTY
  // forms. A member copy owns its inline bytes for the validator's lifetime.
  const StringView wkb_;
  const GeospatialKind& kind_;
  const std::string& columnPath_;
  uint64_t position_{0};
};

// Rejects payloads that Velox's two-dimensional, SRID-less GEOMETRY cannot
// represent faithfully, rather than silently dropping dimensions. Every nested
// header is checked, not just the outermost one; see WkbHeaderValidator.
void validateIsoWkb(
    StringView wkb,
    const GeospatialKind& kind,
    const std::string& columnPath) {
  VELOX_USER_CHECK_GE(
      wkb.size(),
      kWkbHeaderLength,
      "Invalid well-known binary (WKB) in Iceberg {} column '{}': expected at least {} bytes, found {}",
      kind.columnKind,
      columnPath,
      kWkbHeaderLength,
      wkb.size());

  WkbHeaderValidator{wkb, kind, columnPath}.validate();
}

// Rejects a coordinate that is not a valid WGS84 longitude/latitude, with the
// messages of Presto's to_spherical_geography. Every coordinate is visited
// rather than the envelope, because a GEOS envelope ignores NaN ordinates. An
// empty point, which WKB encodes as NaN ordinates, has no coordinates to visit.
class SphericalCoordinateValidator : public geos::geom::CoordinateFilter {
 public:
  explicit SphericalCoordinateValidator(const std::string& columnPath)
      : columnPath_{columnPath} {}

  void filter_ro(const geos::geom::Coordinate* coordinate) override {
    VELOX_USER_CHECK(
        isInRange(
            coordinate->x,
            common::geospatial::kMinLongitude,
            common::geospatial::kMaxLongitude),
        "Invalid value in Iceberg geography column '{}': Longitude must be between -180 and 180; found {}",
        columnPath_,
        coordinate->x);
    VELOX_USER_CHECK(
        isInRange(
            coordinate->y,
            common::geospatial::kMinLatitude,
            common::geospatial::kMaxLatitude),
        "Invalid value in Iceberg geography column '{}': Latitude must be between -90 and 90; found {}",
        columnPath_,
        coordinate->y);
  }

 private:
  // False for NaN, and for infinities since the bounds are finite.
  static bool isInRange(double value, double min, double max) {
    return value >= min && value <= max;
  }

  const std::string& columnPath_;
};

// Parses one WKB value and appends Velox's internal geometry encoding to
// 'out'. A geography value is validated before it is encoded.
void wkbToVeloxGeometry(
    StringView wkb,
    geos::io::WKBReader& wkbReader,
    std::string& out,
    const GeospatialKind& kind,
    const std::string& columnPath) {
  validateIsoWkb(wkb, kind, columnPath);
  std::unique_ptr<geos::geom::Geometry> geometry;
  try {
    geometry = wkbReader.read(
        reinterpret_cast<const unsigned char*>(wkb.data()), wkb.size());
  } catch (const std::exception& e) {
    VELOX_USER_FAIL(
        "Invalid well-known binary (WKB) in Iceberg {} column '{}': {}",
        kind.columnKind,
        columnPath,
        e.what());
  }
  // Z and M are already rejected by validateIsoWkb(), so only the ranges of
  // Presto's to_spherical_geography remain to be checked. Its geometry kind
  // check needs no counterpart: every kind WKB can encode passes it.
  if (kind.sphericalCoordinates) {
    SphericalCoordinateValidator validator{columnPath};
    geometry->apply_ro(&validator);
  }
  // Standard WKB carries no SRID and Velox's GEOMETRY has no SRID concept:
  // never invent one. GeometrySerializer must not run inside a GEOS_TRY: it
  // needs its exceptions to bubble up.
  common::geospatial::GeometrySerializer::serialize(*geometry, out);
}

// Converts the live positions of a flat scalar vector of WKB payloads into a
// new flat vector of 'targetType', GEOMETRY() or SPHERICAL_GEOGRAPHY().
// Positions outside 'rows', and null positions, are left null; their bytes are
// never read.
VectorPtr convertGeospatialLeaf(
    const VectorPtr& input,
    const TypePtr& targetType,
    const SelectivityVector& rows,
    memory::MemoryPool* pool,
    const std::string& columnPath) {
  const auto& kind =
      isSphericalGeographyType(targetType) ? kGeographyKind : kGeometryKind;
  const auto size = input->size();
  auto* flatInput = input->asFlatVector<StringView>();
  VELOX_CHECK_NOT_NULL(
      flatInput,
      "Expected a flat scalar vector for Iceberg {} column '{}', got {}",
      kind.columnKind,
      columnPath,
      input->encoding());

  auto result =
      BaseVector::create<FlatVector<StringView>>(targetType, size, pool);
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
        flatInput->valueAt(i), wkbReader, serialized, kind, columnPath);
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

VectorPtr convertIcebergGeospatial(
    const VectorPtr& input,
    const TypePtr& targetType,
    const SelectivityVector& rows,
    memory::MemoryPool* pool,
    const std::string& columnPath) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK(containsGeospatial(targetType));

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
      auto converted = convertIcebergGeospatial(
          base, targetType, singleRow, pool, columnPath);
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
      auto convertedValues = convertIcebergGeospatial(
          base, targetType, baseRows, pool, columnPath);
      return BaseVector::wrapInDictionary(
          loaded->nulls(),
          loaded->wrapInfo(),
          loaded->size(),
          std::move(convertedValues));
    }

    default:
      break;
  }

  if (isGeospatialType(targetType)) {
    return convertGeospatialLeaf(loaded, targetType, rows, pool, columnPath);
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
        if (containsGeospatial(rowType.childAt(i))) {
          // A row's children are positionally aligned with the row itself, so
          // the live rows carry over unchanged.
          children[i] = convertIcebergGeospatial(
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
      auto elements = convertIcebergGeospatial(
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
      if (containsGeospatial(targetType->childAt(0))) {
        keys = convertIcebergGeospatial(
            keys,
            targetType->childAt(0),
            entryRows,
            pool,
            fmt::format("{}[key]", columnPath));
      }
      if (containsGeospatial(targetType->childAt(1))) {
        values = convertIcebergGeospatial(
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

VectorPtr convertIcebergGeospatial(
    const VectorPtr& input,
    const TypePtr& targetType,
    memory::MemoryPool* pool,
    const std::string& columnPath) {
  SelectivityVector allRows(input->size());
  return convertIcebergGeospatial(input, targetType, allRows, pool, columnPath);
}

} // namespace facebook::velox::connector::hive::iceberg
