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

#include <string>
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/GeometryFunctions.h"
#include "velox/functions/prestosql/GooglePolylineFunctions.h"
#include "velox/functions/prestosql/types/BingTileType.h"
#include "velox/functions/prestosql/types/GeometryRegistration.h"
#include "velox/functions/prestosql/types/SphericalGeographyRegistration.h"

namespace facebook::velox::functions {

namespace {

void registerConstructors(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<StGeometryFromTextFunction, Geometry, Varchar>(
      {{prefix + "ST_GeometryFromText"}}, {}, true, defaultOwner);
  registerFunction<StGeomFromBinaryFunction, Geometry, Varbinary>(
      {{prefix + "ST_GeomFromBinary"}}, {}, true, defaultOwner);
  registerFunction<StAsTextFunction, Varchar, Geometry>(
      {{prefix + "ST_AsText"}}, {}, true, defaultOwner);
  registerFunction<SphericalAsTextFunction, Varchar, SphericalGeography>(
      {{prefix + "ST_AsText"}}, {}, true, defaultOwner);
  registerFunction<StAsBinaryFunction, Varbinary, Geometry>(
      {{prefix + "ST_AsBinary"}}, {}, true, defaultOwner);
  registerFunction<StPointFunction, Geometry, double, double>(
      {{prefix + "ST_Point"}}, {}, true, defaultOwner);
  registerFunction<StLineFromTextFunction, Geometry, Varchar>(
      {{prefix + "ST_LineFromText"}}, {}, true, defaultOwner);
  registerFunction<StLineStringFunction, Geometry, Array<Geometry>>(
      {{prefix + "ST_LineString"}}, {}, true, defaultOwner);
  registerFunction<StMultiPointFunction, Geometry, Array<Geometry>>(
      {{prefix + "ST_MultiPoint"}}, {}, true, defaultOwner);
  registerFunction<ToSphericalGeographyFunction, SphericalGeography, Geometry>(
      {{prefix + "to_spherical_geography"}}, {}, true, defaultOwner);
  registerFunction<ToGeometryFunction, Geometry, SphericalGeography>(
      {{prefix + "to_geometry"}}, {}, true, defaultOwner);
  registerFunction<
      StSphericalCentroidFunction,
      SphericalGeography,
      SphericalGeography>({{prefix + "st_centroid"}}, {}, true, defaultOwner);
  registerFunction<
      StSphericalDistanceFunction,
      double,
      SphericalGeography,
      SphericalGeography>({{prefix + "st_distance"}}, {}, true, defaultOwner);
  registerFunction<StSphericalLengthFunction, double, SphericalGeography>(
      {{prefix + "st_length"}}, {}, true, defaultOwner);
  registerFunction<StSphericalAreaFunction, double, SphericalGeography>(
      {{prefix + "st_area"}}, {}, true, defaultOwner);
}

void registerRelationPredicates(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<StRelateFunction, bool, Geometry, Geometry, Varchar>(
      {{prefix + "ST_Relate"}}, {}, true, defaultOwner);

  registerFunction<StContainsFunction, bool, Geometry, Geometry>(
      {{prefix + "ST_Contains"}}, {}, true, defaultOwner);
  registerFunction<StCrossesFunction, bool, Geometry, Geometry>(
      {{prefix + "ST_Crosses"}}, {}, true, defaultOwner);
  registerFunction<StDisjointFunction, bool, Geometry, Geometry>(
      {{prefix + "ST_Disjoint"}}, {}, true, defaultOwner);
  registerFunction<StEqualsFunction, bool, Geometry, Geometry>(
      {{prefix + "ST_Equals"}}, {}, true, defaultOwner);
  registerFunction<StIntersectsFunction, bool, Geometry, Geometry>(
      {{prefix + "ST_Intersects"}}, {}, true, defaultOwner);
  registerFunction<StOverlapsFunction, bool, Geometry, Geometry>(
      {{prefix + "ST_Overlaps"}}, {}, true, defaultOwner);
  registerFunction<StTouchesFunction, bool, Geometry, Geometry>(
      {{prefix + "ST_Touches"}}, {}, true, defaultOwner);
  registerFunction<StWithinFunction, bool, Geometry, Geometry>(
      {{prefix + "ST_Within"}}, {}, true, defaultOwner);
}

void registerOverlayOperations(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<StBoundaryFunction, Geometry, Geometry>(
      {{prefix + "St_Boundary"}}, {}, true, defaultOwner);
  registerFunction<StDifferenceFunction, Geometry, Geometry, Geometry>(
      {{prefix + "ST_Difference"}}, {}, true, defaultOwner);
  registerFunction<StIntersectionFunction, Geometry, Geometry, Geometry>(
      {{prefix + "ST_Intersection"}}, {}, true, defaultOwner);
  registerFunction<StSymDifferenceFunction, Geometry, Geometry, Geometry>(
      {{prefix + "ST_SymDifference"}}, {}, true, defaultOwner);
  registerFunction<StUnionFunction, Geometry, Geometry, Geometry>(
      {{prefix + "ST_Union"}}, {}, true, defaultOwner);
  registerFunction<StEnvelopeAsPtsFunction, Array<Geometry>, Geometry>(
      {{prefix + "ST_EnvelopeAsPts"}}, {}, true, defaultOwner);
  registerFunction<ExpandEnvelopeFunction, Geometry, Geometry, double>(
      {{prefix + "expand_envelope"}}, {}, true, defaultOwner);
}

void registerAccessors(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<StIsValidFunction, bool, Geometry>(
      {{prefix + "ST_IsValid"}}, {}, true, defaultOwner);
  registerFunction<StIsSimpleFunction, bool, Geometry>(
      {{prefix + "ST_IsSimple"}}, {}, true, defaultOwner);
  registerFunction<GeometryInvalidReasonFunction, Varchar, Geometry>(
      {{prefix + "geometry_invalid_reason"}}, {}, true, defaultOwner);
  registerFunction<SimplifyGeometryFunction, Geometry, Geometry, double>(
      {{prefix + "simplify_geometry"}}, {}, true, defaultOwner);

  registerFunction<StAreaFunction, double, Geometry>(
      {{prefix + "ST_Area"}}, {}, true, defaultOwner);
  registerFunction<StCentroidFunction, Geometry, Geometry>(
      {{prefix + "ST_Centroid"}}, {}, true, defaultOwner);
  registerFunction<StXFunction, double, Geometry>(
      {{prefix + "ST_X"}}, {}, true, defaultOwner);
  registerFunction<StYFunction, double, Geometry>(
      {{prefix + "ST_Y"}}, {}, true, defaultOwner);
  registerFunction<StXMinFunction, double, Geometry>(
      {{prefix + "ST_XMin"}}, {}, true, defaultOwner);
  registerFunction<StYMinFunction, double, Geometry>(
      {{prefix + "ST_YMin"}}, {}, true, defaultOwner);
  registerFunction<StXMaxFunction, double, Geometry>(
      {{prefix + "ST_XMax"}}, {}, true, defaultOwner);
  registerFunction<StYMaxFunction, double, Geometry>(
      {{prefix + "ST_YMax"}}, {}, true, defaultOwner);
  registerFunction<StGeometryTypeFunction, Varchar, Geometry>(
      {{prefix + "ST_GeometryType"}}, {}, true, defaultOwner);
  registerFunction<StDistanceFunction, double, Geometry, Geometry>(
      {{prefix + "ST_Distance"}}, {}, true, defaultOwner);
  registerFunction<StPolygonFunction, Geometry, Varchar>(
      {{prefix + "ST_Polygon"}}, {}, true, defaultOwner);
  registerFunction<StIsClosedFunction, bool, Geometry>(
      {{prefix + "ST_IsClosed"}}, {}, true, defaultOwner);
  registerFunction<StIsEmptyFunction, bool, Geometry>(
      {{prefix + "ST_IsEmpty"}}, {}, true, defaultOwner);
  registerFunction<StIsRingFunction, bool, Geometry>(
      {{prefix + "ST_IsRing"}}, {}, true, defaultOwner);
  registerFunction<StLengthFunction, double, Geometry>(
      {{prefix + "ST_Length"}}, {}, true, defaultOwner);
  registerFunction<StPointNFunction, Geometry, Geometry, int32_t>(
      {{prefix + "ST_PointN"}}, {}, true, defaultOwner);
  registerFunction<StStartPointFunction, Geometry, Geometry>(
      {{prefix + "ST_StartPoint"}}, {}, true, defaultOwner);
  registerFunction<StEndPointFunction, Geometry, Geometry>(
      {{prefix + "ST_EndPoint"}}, {}, true, defaultOwner);
  registerFunction<StGeometryNFunction, Geometry, Geometry, int32_t>(
      {{prefix + "ST_GeometryN"}}, {}, true, defaultOwner);
  registerFunction<StInteriorRingNFunction, Geometry, Geometry, int32_t>(
      {{prefix + "ST_InteriorRingN"}}, {}, true, defaultOwner);
  registerFunction<StNumGeometriesFunction, int32_t, Geometry>(
      {{prefix + "ST_NumGeometries"}}, {}, true, defaultOwner);
  registerFunction<StNumInteriorRingFunction, int64_t, Geometry>(
      {{prefix + "ST_NumInteriorRing"}}, {}, true, defaultOwner);
  registerFunction<StConvexHullFunction, Geometry, Geometry>(
      {{prefix + "ST_ConvexHull"}}, {}, true, defaultOwner);
  registerFunction<StDimensionFunction, int8_t, Geometry>(
      {{prefix + "ST_Dimension"}}, {}, true, defaultOwner);
  registerFunction<StExteriorRingFunction, Geometry, Geometry>(
      {{prefix + "ST_ExteriorRing"}}, {}, true, defaultOwner);
  registerFunction<StEnvelopeFunction, Geometry, Geometry>(
      {{prefix + "ST_Envelope"}}, {}, true, defaultOwner);
  registerFunction<StBufferFunction, Geometry, Geometry, double>(
      {{prefix + "ST_Buffer"}}, {}, true, defaultOwner);
  registerFunction<LineLocatePointFunction, double, Geometry, Geometry>(
      {{prefix + "line_locate_point"}}, {}, true, defaultOwner);
  registerFunction<LineInterpolatePointFunction, Geometry, Geometry, double>(
      {{prefix + "line_interpolate_point"}}, {}, true, defaultOwner);

  velox::exec::registerVectorFunction(
      prefix + "ST_CoordDim",
      StCoordDimFunction::signatures(),
      std::make_unique<StCoordDimFunction>(),
      {},
      /*overwrite=*/true,
      defaultOwner);
  registerFunction<StPointsFunction, Array<Geometry>, Geometry>(
      {{prefix + "ST_Points"}}, {}, true, defaultOwner);
  registerFunction<StNumPointsFunction, int64_t, Geometry>(
      {{prefix + "ST_NumPoints"}}, {}, true, defaultOwner);
  registerFunction<StInteriorRingsFunction, Array<Geometry>, Geometry>(
      {{prefix + "ST_InteriorRings"}}, {}, true, defaultOwner);
  registerFunction<StGeometriesFunction, Array<Geometry>, Geometry>(
      {{prefix + "ST_Geometries"}}, {}, true, defaultOwner);
  registerFunction<GeometryAsGeoJsonFunction, Varchar, Geometry>(
      {{prefix + "geometry_as_geojson"}}, {}, true, defaultOwner);
  registerFunction<GeometryFromGeoJsonFunction, Geometry, Varchar>(
      {{prefix + "geometry_from_geojson"}}, {}, true, defaultOwner);
  registerFunction<GeometryUnionFunction, Geometry, Array<Geometry>>(
      {{prefix + "geometry_union"}}, {}, true, defaultOwner);
  registerFunction<
      GeometryNearestPointsFunction,
      Array<Geometry>,
      Geometry,
      Geometry>({{prefix + "geometry_nearest_points"}}, {}, true, defaultOwner);
  registerFunction<
      FlattenGeometryCollectionsFunction,
      Array<Geometry>,
      Geometry>(
      {{prefix + "flatten_geometry_collections"}}, {}, true, defaultOwner);
  registerFunction<
      GreatCircleDistanceFunction,
      double,
      double,
      double,
      double,
      double>({{prefix + "great_circle_distance"}}, {}, true, defaultOwner);
}

void registerBingTileGeometryFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<BingTilePolygonFunction, Geometry, BingTile>(
      {{prefix + "bing_tile_polygon"}}, {}, true, defaultOwner);
  registerFunction<
      GeometryToBingTilesFunction,
      Array<BingTile>,
      Geometry,
      int32_t>({{prefix + "geometry_to_bing_tiles"}}, {}, true, defaultOwner);
  registerFunction<
      GeometryToDissolvedBingTilesFunction,
      Array<BingTile>,
      Geometry,
      int32_t>(
      {{prefix + "geometry_to_dissolved_bing_tiles"}}, {}, true, defaultOwner);
}

void registerGooglePolylineFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<GooglePolylineEncodeFunction, Varchar, Array<Geometry>>(
      {{prefix + "google_polyline_encode"}}, {}, true, defaultOwner);

  registerFunction<
      GooglePolylineEncodeFunction,
      Varchar,
      Array<Geometry>,
      int64_t>({{prefix + "google_polyline_encode"}}, {}, true, defaultOwner);

  registerFunction<GooglePolylineDecodeFunction, Array<Geometry>, Varchar>(
      {{prefix + "google_polyline_decode"}}, {}, true, defaultOwner);

  registerFunction<
      GooglePolylineDecodeFunction,
      Array<Geometry>,
      Varchar,
      int64_t>({{prefix + "google_polyline_decode"}}, {}, true, defaultOwner);
}

} // namespace

void registerGeometryFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerGeometryType();
  registerSphericalGeographyType();
  registerConstructors(prefix, defaultOwner);
  registerRelationPredicates(prefix, defaultOwner);
  registerOverlayOperations(prefix, defaultOwner);
  registerAccessors(prefix, defaultOwner);
  registerBingTileGeometryFunctions(prefix, defaultOwner);
  registerGooglePolylineFunctions(prefix, defaultOwner);
}

} // namespace facebook::velox::functions
