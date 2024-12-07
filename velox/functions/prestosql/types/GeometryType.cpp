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
#include <geos/io/WKBReader.h>
#include <geos/io/WKBWriter.h>
#include <geos/io/WKTReader.h>
#include <geos/io/WKTWriter.h>

#include "velox/functions/prestosql/types/GeometryType.h"
#include "velox/type/Type.h"

namespace facebook::velox {

class GeometryCastOperator : public exec::CastOperator {
 public:
  bool isSupportedFromType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::VARBINARY:
      case TypeKind::VARCHAR:
        return true;
      default:
        return false;
    }
  }

  bool isSupportedToType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::VARBINARY:
      case TypeKind::VARCHAR:
        return true;
      default:
        return false;
    }
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (input.typeKind() == TypeKind::VARCHAR) {
      castFromString(input, context, rows, *result);
    } else if (input.typeKind() == TypeKind::VARBINARY) {
      castFromVarbinary(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from {} to IPAddress not supported", resultType->toString());
    }
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (resultType->kind() == TypeKind::VARCHAR) {
      castToString(input, context, rows, *result);
    } else if (resultType->kind() == TypeKind::VARBINARY) {
      castToVarbinary(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from IPAddress to {} not supported", resultType->toString());
    }
  }

 private:
  static void castToString(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* geometries = input.as<SimpleVector<StringView>>();
    geos::io::WKTWriter writer;

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto geometry = geometries->valueAt(row);
      auto geosGeometry = functions::GeosGeometrySerde::deserialize(geometry);
      exec::StringWriter<false> wktString(flatResult, row);
      wktString.append(writer.write(geosGeometry.get()));
      wktString.finalize();
    });
  }

  static void castFromString(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* geometryStrings = input.as<SimpleVector<StringView>>();
    geos::io::WKTReader reader;

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto wktString = geometryStrings->valueAt(row);
      auto geosGeometry = reader.read(wktString);
      exec::StringWriter<> geometry(flatResult, row);
      functions::GeosGeometrySerde::serialize(geosGeometry.get(), geometry);
      geometry.finalize();
    });
  }

  static void castToVarbinary(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* geometries = input.as<SimpleVector<StringView>>();
    geos::io::WKBWriter writer;

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto geometry = geometries->valueAt(row);
      auto geosGeometry = functions::GeosGeometrySerde::deserialize(geometry);
      std::ostringstream os;
      writer.write(*geosGeometry, os);
      exec::StringWriter<> wkbString(flatResult, row);
      wkbString.append(os.str()); // append
      wkbString.finalize();
    });
  }

  static void castFromVarbinary(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* geometryStrings = input.as<SimpleVector<StringView>>();
    geos::io::WKBReader reader;

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto wkbString = geometryStrings->valueAt(row);
      auto geosGeometry = reader.read(
          reinterpret_cast<const unsigned char*>(wkbString.data()),
          wkbString.size());
      exec::StringWriter<> geometry(flatResult, row);
      functions::GeosGeometrySerde::serialize(geosGeometry.get(), geometry);
      geometry.finalize();
    });
  }
};

class GeometryTypeFactories : public CustomTypeFactories {
 public:
  GeometryTypeFactories() = default;

  TypePtr getType() const override {
    return GEOMETRY();
  }
  exec::CastOperatorPtr getCastOperator() const override {
    return std::make_shared<GeometryCastOperator>();
  }
};

void registerGeometryType() {
  // Register the geometry type with the type registry.
  registerCustomType(
      "geometry", std::make_unique<const GeometryTypeFactories>());
}

} // namespace facebook::velox
