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

#include "velox/functions/prestosql/aggregates/GeometryAggregate.h"

#include <geos/geom/CoordinateSequence.h>
#include <geos/geom/Geometry.h>

#include "velox/common/geospatial/GeometrySerde.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/exec/Strings.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/types/GeometryRegistration.h"
#include "velox/functions/prestosql/types/GeometryType.h"

using namespace facebook::velox::exec;
using facebook::velox::common::geospatial::GeometryDeserializer;

namespace facebook::velox::aggregate::prestosql {

namespace {

// Take the convex hull of all the geometries.
// This is more efficient than taking the pairwise convex hull.
template <typename T>
class ConvexHullAggregate {
 public:
  using InputType = Row<Geometry>;
  using IntermediateType = Geometry;
  using OutputType = Geometry;

  struct AccumulatorType {
    static constexpr bool use_external_memory_ = true;
    static constexpr bool is_fixed_size_ = false;
    static constexpr bool is_aligned_ = true;

    ConvexHullAggregate* fn_;
    std::vector<StringView> geometryData_{};
    Strings strings_;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        ConvexHullAggregate* /*fn*/) {}

    void destroy(HashStringAllocator* allocator) {
      geometryData_.clear();
      geometryData_.shrink_to_fit();
      strings_.free(*allocator);
    }

    void addInput(
        HashStringAllocator* allocator,
        exec::arg_type<Geometry> value) {
      addGeometry(allocator, value);
    }

    void combine(
        HashStringAllocator* allocator,
        arg_type<IntermediateType> other) {
      addGeometry(allocator, other);
    }

    // If there are geometries, write to `out` and return true.
    // Otherwise, return false.
    bool writeGeometry(exec::out_type<OutputType>& out) {
      std::unique_ptr<geos::geom::Geometry> hull = makeHull();
      if (hull != nullptr) {
        common::geospatial::GeometrySerializer::serialize(*hull, out);
      }
      return hull != nullptr;
    }

    bool writeFinalResult(exec::out_type<OutputType>& out) {
      return writeGeometry(out);
    }

    bool writeIntermediateResult(out_type<IntermediateType>& out) {
      return writeGeometry(out);
    }

   private:
    void addGeometry(
        HashStringAllocator* allocator,
        exec::arg_type<Geometry> value) {
      // Must add empty geometries here, because geometryData_.empty() will
      // tell us if there has been at least one input (and thus will have
      // non-null output).
      if (!value.isInline()) {
        value = strings_.append(value, *allocator);
      }
      geometryData_.push_back(value);
    }

    // Create a convex hull from all the stored coordinates.
    // Returns a nullptr if there are no geometries to make a hull from.
    std::unique_ptr<geos::geom::Geometry> makeHull() {
      // No geometryData_ means no inputs, so return NULL
      if (geometryData_.empty()) {
        return nullptr;
      }

      std::vector<std::unique_ptr<geos::geom::Geometry>> geometries;
      geometries.reserve(geometryData_.size());
      for (const auto& data : geometryData_) {
        auto geom = common::geospatial::GeometryDeserializer::deserialize(data);
        VELOX_DCHECK_NOT_NULL(geom);
        if (!geom->isEmpty()) {
          geometries.push_back(std::move(geom));
        }
      }
      if (geometries.empty()) {
        return GeometryDeserializer::getGeometryFactory()
            ->createEmptyGeometry();
      }

      // Calculate total number of points for efficient memory allocation
      size_t totalPoints = 0;
      for (const auto& geom : geometries) {
        totalPoints += geom->getNumPoints();
      }

      // Create a single CoordinateSequence from all coordinates
      auto coords =
          std::make_unique<geos::geom::CoordinateArraySequence>(totalPoints, 2);
      size_t index = 0;

      // Add all coordinates from coordinate sequences
      for (const auto& geom : geometries) {
        auto coordSeq = geom->getCoordinates();
        for (size_t i = 0; i < coordSeq->size(); ++i) {
          VELOX_DCHECK_GT(coords->size(), index);
          coords->setAt(coordSeq->getAt(i), index++);
        }
      }

      auto points = std::unique_ptr<geos::geom::MultiPoint>(
          GeometryDeserializer::getGeometryFactory()->createMultiPoint(
              *coords));
      return points->convexHull();
    }
  };
};

void registerConvexHullAggregate(
    const std::vector<std::string>& names,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType("geometry")
          .intermediateType("geometry")
          .argumentType("geometry")
          .build());

  exec::registerAggregateFunction(
      names,
      signatures,
      [names](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        const std::string& name = names.front();
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        const auto& inputType = argTypes[0];
        if (!isGeometryType(inputType)) {
          VELOX_FAIL(
              "Unexpected input type for {} aggregation {}",
              name,
              inputType->kindName());
        }
        return std::make_unique<
            exec::SimpleAggregateAdapter<ConvexHullAggregate<Geometry>>>(
            step, argTypes, resultType);
      },
      true /*registerCompanionFunctions*/,
      overwrite);
}

template <typename T>
class GeometryUnionAggregate {
 public:
  using InputType = Row<Geometry>;
  using IntermediateType = Geometry;
  using OutputType = Geometry;

  struct AccumulatorType {
    static constexpr bool use_external_memory_ = true;
    static constexpr bool is_fixed_size_ = false;
    static constexpr bool is_aligned_ = true;

    GeometryUnionAggregate* fn_;
    std::vector<StringView> geometryData_{};
    Strings strings_;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        GeometryUnionAggregate* /*fn*/) {}

    void destroy(HashStringAllocator* allocator) {
      geometryData_.clear();
      geometryData_.shrink_to_fit();
      strings_.free(*allocator);
    }

    void addInput(
        HashStringAllocator* allocator,
        exec::arg_type<Geometry> value) {
      addGeometry(allocator, value);
    }

    void combine(
        HashStringAllocator* allocator,
        arg_type<IntermediateType> other) {
      addGeometry(allocator, other);
    }

    // If there are geometries, write to `out` and return true.
    // Otherwise, return false.
    bool writeGeometry(exec::out_type<OutputType>& out) {
      std::unique_ptr<geos::geom::Geometry> geomUnion = makeUnion();
      if (geomUnion != nullptr) {
        common::geospatial::GeometrySerializer::serialize(*geomUnion, out);
      }
      return geomUnion != nullptr;
    }

    bool writeFinalResult(exec::out_type<OutputType>& out) {
      return writeGeometry(out);
    }

    bool writeIntermediateResult(out_type<IntermediateType>& out) {
      return writeGeometry(out);
    }

   private:
    void addGeometry(
        HashStringAllocator* allocator,
        exec::arg_type<Geometry> value) {
      // Must add empty geometries here, because geometryData_.empty() will
      // tell us if there has been at least one input (and thus will have
      // non-null output).
      if (!value.isInline()) {
        value = strings_.append(value, *allocator);
      }
      geometryData_.push_back(value);
    }

    // Create the geometries union from all the stored coordinates.
    // Returns a nullptr if there has been no input.
    std::unique_ptr<geos::geom::Geometry> makeUnion() {
      // No geometryData_ means no inputs, so return NULL
      if (geometryData_.empty()) {
        return nullptr;
      }

      std::vector<std::unique_ptr<geos::geom::Geometry>> geometries;
      geometries.reserve(geometryData_.size());
      for (const auto& data : geometryData_) {
        auto geom = common::geospatial::GeometryDeserializer::deserialize(data);
        VELOX_DCHECK_NOT_NULL(geom);
        if (!geom->isEmpty()) {
          geometries.push_back(std::move(geom));
        }
      }
      if (geometries.empty()) {
        return GeometryDeserializer::getGeometryFactory()
            ->createEmptyGeometry();
      }

      auto collection =
          GeometryDeserializer::getGeometryFactory()->createGeometryCollection(
              std::move(geometries));
      return collection->Union();
    }
  };
};

void registerGeometryUnionAggregate(
    const std::vector<std::string>& names,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType("geometry")
          .intermediateType("geometry")
          .argumentType("geometry")
          .build());

  exec::registerAggregateFunction(
      names,
      signatures,
      [names](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        const std::string& name = names.front();
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        const auto& inputType = argTypes[0];
        if (!isGeometryType(inputType)) {
          VELOX_FAIL(
              "Unexpected input type for {} aggregation {}",
              name,
              inputType->kindName());
        }
        return std::make_unique<
            exec::SimpleAggregateAdapter<GeometryUnionAggregate<Geometry>>>(
            step, argTypes, resultType);
      },
      true /*registerCompanionFunctions*/,
      overwrite);
}

} // namespace

void registerGeometryAggregate(
    const std::vector<std::string>& convexHullNames,
    const std::vector<std::string>& geometryUnionNames,
    bool overwrite) {
  registerGeometryType();
  registerConvexHullAggregate(convexHullNames, overwrite);
  registerGeometryUnionAggregate(geometryUnionNames, overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
