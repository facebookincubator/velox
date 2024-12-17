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

#include "velox/exec/fuzzer/PrestoQueryRunnerIntermediateTypeTransforms.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/parse/Expressions.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::exec::test {
namespace {
// Defines a transform for an intermediate type or a complex type that can
// contain an intermediate type.
class IntermediateTypeTransform {
 public:
  virtual ~IntermediateTypeTransform() = default;

  virtual VectorPtr transform(const VectorPtr& vector) const = 0;
  virtual core::ExprPtr projectionExpr(
      const TypePtr& type,
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const = 0;
};

class ArrayTransform : public IntermediateTypeTransform {
 public:
  VectorPtr transform(const VectorPtr& vector) const override;
  core::ExprPtr projectionExpr(
      const TypePtr& type,
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const override;
};

class MapTransform : public IntermediateTypeTransform {
 public:
  VectorPtr transform(const VectorPtr& vector) const override;
  core::ExprPtr projectionExpr(
      const TypePtr& type,
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const override;
};

class RowTransform : public IntermediateTypeTransform {
 public:
  VectorPtr transform(const VectorPtr& vector) const override;
  core::ExprPtr projectionExpr(
      const TypePtr& type,
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const override;
};

class TimestampWithTimeZoneTransform : public IntermediateTypeTransform {
 private:
  static constexpr const char* kMillisColumnName = "millis";
  static constexpr const char* kTimeZoneColumnName = "time_zone";

 public:
  VectorPtr transform(const VectorPtr& vector) const override;
  core::ExprPtr projectionExpr(
      const TypePtr& type,
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const override;
};

// Convert a Vector of TimestampWithTimeZone to a RowVector of BIGINT and
// VARCHAR (millisUtc and time_zone).
VectorPtr TimestampWithTimeZoneTransform::transform(
    const VectorPtr& vector) const {
  VELOX_CHECK(isTimestampWithTimeZoneType(vector->type()));
  DecodedVector decoded(*vector);
  const auto* base = decoded.base()->as<SimpleVector<int64_t>>();

  VectorPtr millisVector;
  VectorPtr timeZoneVector;

  if (base->isFlatEncoding()) {
    auto millisVectorFlat = BaseVector::create<FlatVector<double>>(
        DOUBLE(), base->size(), base->pool());
    auto timeZoneVectorFlat = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), base->size(), base->pool());
    millisVector = millisVectorFlat;
    timeZoneVector = timeZoneVectorFlat;

    for (int i = 0; i < base->size(); ++i) {
      if (base->isNullAt(i)) {
        millisVectorFlat->setNull(i, true);
        timeZoneVectorFlat->setNull(i, true);
      } else {
        millisVectorFlat->set(i, (double)unpackMillisUtc(base->valueAt(i)));
        std::string tzName =
            tz::getTimeZoneName(unpackZoneKeyId(base->valueAt(i)));
        timeZoneVectorFlat->set(i, StringView(tzName));
      }
    }
  } else {
    VELOX_CHECK(base->isConstantEncoding());
    if (base->isNullAt(0)) {
      millisVector =
          BaseVector::createNullConstant(DOUBLE(), base->size(), base->pool());
      timeZoneVector =
          BaseVector::createNullConstant(VARCHAR(), base->size(), base->pool());
    } else {
      millisVector = BaseVector::createConstant(
          DOUBLE(),
          (double)unpackMillisUtc(base->valueAt(0)),
          base->size(),
          base->pool());
      std::string tzName =
          tz::getTimeZoneName(unpackZoneKeyId(base->valueAt(0)));
      timeZoneVector = BaseVector::createConstant(
          VARCHAR(), StringView(tzName), base->size(), base->pool());
    }
  }

  VectorPtr row = std::make_shared<RowVector>(
      base->pool(),
      ROW({kMillisColumnName, kTimeZoneColumnName}, {DOUBLE(), VARCHAR()}),
      nullptr,
      base->size(),
      std::vector<VectorPtr>{millisVector, timeZoneVector});

  if (!decoded.isIdentityMapping()) {
    row = decoded.wrap(row, *vector, vector->size());
  }

  return row;
}

// Applies from_unixtime to a RowVector of BIGINT and VARCHAR (millisUtc and
// time_zone) to produce values of type TimestampWithTimeZone.
core::ExprPtr TimestampWithTimeZoneTransform::projectionExpr(
    const TypePtr& type,
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  VELOX_CHECK(isTimestampWithTimeZoneType(type));

  return std::make_shared<core::CallExpr>(
      "from_unixtime",
      std::vector<core::ExprPtr>{
          std::make_shared<core::FieldAccessExpr>(
              kMillisColumnName,
              kMillisColumnName,
              std::vector<core::ExprPtr>{inputExpr}),
          std::make_shared<core::FieldAccessExpr>(
              kTimeZoneColumnName,
              kTimeZoneColumnName,
              std::vector<core::ExprPtr>{inputExpr})},
      columnAlias);
}

// Converts an ArrayVector so that any intermediate only types in the elements
// are transformed.
VectorPtr ArrayTransform::transform(const VectorPtr& vector) const {
  VELOX_CHECK(vector->type()->isArray());
  DecodedVector decoded(*vector);
  const auto* base = decoded.base()->as<ArrayVector>();

  VectorPtr elementsVector = transformIntermediateOnlyType(base->elements());

  VectorPtr array = std::make_shared<ArrayVector>(
      base->pool(),
      ARRAY(elementsVector->type()),
      base->nulls(),
      base->size(),
      base->offsets(),
      base->sizes(),
      elementsVector);

  if (!decoded.isIdentityMapping()) {
    array = decoded.wrap(array, *vector, vector->size());
  }

  return array;
}

// Applies a lambda transform to the elements of an array to convert input types
// to intermediate only types where necessary.
core::ExprPtr ArrayTransform::projectionExpr(
    const TypePtr& type,
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  VELOX_CHECK(type->isArray());

  return std::make_shared<core::CallExpr>(
      "transform",
      std::vector<core::ExprPtr>{
          inputExpr,
          std::make_shared<core::LambdaExpr>(
              std::vector<std::string>{"x"},
              getIntermediateOnlyTypeProjectionExpr(
                  type->asArray().elementType(),
                  std::make_shared<core::FieldAccessExpr>("x", "x"),
                  "x"))},
      columnAlias);
}

// Converts an MapVector so that any intermediate only types in the keys and
// values are transformed.
VectorPtr MapTransform::transform(const VectorPtr& vector) const {
  VELOX_CHECK(vector->type()->isMap());
  DecodedVector decoded(*vector);
  const auto* base = decoded.base()->as<MapVector>();

  VectorPtr keysVector = base->mapKeys();
  VectorPtr valuesVector = base->mapValues();
  const auto& keysType = keysVector->type();
  const auto& valuesType = valuesVector->type();

  if (isIntermediateOnlyType(keysType)) {
    keysVector = transformIntermediateOnlyType(keysVector);
  }

  if (isIntermediateOnlyType(valuesType)) {
    valuesVector = transformIntermediateOnlyType(valuesVector);
  }

  VectorPtr map = std::make_shared<MapVector>(
      base->pool(),
      MAP(keysVector->type(), valuesVector->type()),
      base->nulls(),
      base->size(),
      base->offsets(),
      base->sizes(),
      keysVector,
      valuesVector);

  if (!decoded.isIdentityMapping()) {
    map = decoded.wrap(map, *vector, vector->size());
  }

  return map;
}

// Applies a lambda transform to the keys and values of a map to convert input
// types to intermediate only types where necessary.
core::ExprPtr MapTransform::projectionExpr(
    const TypePtr& type,
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  VELOX_CHECK(type->isMap());
  const auto& mapType = type->asMap();
  const auto& keysType = mapType.keyType();
  const auto& valuesType = mapType.valueType();

  core::ExprPtr expr = inputExpr;

  if (isIntermediateOnlyType(keysType)) {
    expr = std::make_shared<core::CallExpr>(
        "transform_keys",
        std::vector<core::ExprPtr>{
            expr,
            std::make_shared<core::LambdaExpr>(
                std::vector<std::string>{"k", "v"},
                getIntermediateOnlyTypeProjectionExpr(
                    keysType,
                    std::make_shared<core::FieldAccessExpr>("k", "k"),
                    "k"))},
        columnAlias);
  }

  if (isIntermediateOnlyType(valuesType)) {
    expr = std::make_shared<core::CallExpr>(
        "transform_values",
        std::vector<core::ExprPtr>{
            expr,
            std::make_shared<core::LambdaExpr>(
                std::vector<std::string>{"k", "v"},
                getIntermediateOnlyTypeProjectionExpr(
                    valuesType,
                    std::make_shared<core::FieldAccessExpr>("v", "v"),
                    "v"))},
        columnAlias);
  }

  return expr;
}

// Converts an RowVector so that any intermediate only types in the children are
// transformed.
VectorPtr RowTransform::transform(const VectorPtr& vector) const {
  VELOX_CHECK(vector->type()->isRow());
  DecodedVector decoded(*vector);
  const auto* base = decoded.base()->as<RowVector>();

  std::vector<VectorPtr> children;
  std::vector<TypePtr> childrenTypes;
  std::vector<std::string> childrenNames = base->type()->asRow().names();
  for (const auto& child : base->children()) {
    if (isIntermediateOnlyType(child->type())) {
      children.push_back(transformIntermediateOnlyType(child));
      childrenTypes.push_back(children.back()->type());
    } else {
      children.push_back(child);
      childrenTypes.push_back(child->type());
    }
  }

  VectorPtr row = std::make_shared<RowVector>(
      base->pool(),
      ROW(std::move(childrenNames), std::move(childrenTypes)),
      base->nulls(),
      base->size(),
      std::move(children));

  if (!decoded.isIdentityMapping()) {
    row = decoded.wrap(row, *vector, vector->size());
  }

  return row;
}

// Applies transforms to the children of a row to convert input types to
// intermediate only types where necessary, and reconstructs the row via
// row_constructor.
core::ExprPtr RowTransform::projectionExpr(
    const TypePtr& type,
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  VELOX_CHECK(type->isRow());
  const auto& rowType = type->asRow();

  std::vector<core::ExprPtr> children;
  for (int i = 0; i < rowType.size(); i++) {
    if (isIntermediateOnlyType(rowType.childAt(i))) {
      children.push_back(getIntermediateOnlyTypeProjectionExpr(
          rowType.childAt(i),
          std::make_shared<core::FieldAccessExpr>(
              rowType.nameOf(i),
              rowType.nameOf(i),
              std::vector<core::ExprPtr>{inputExpr}),
          rowType.nameOf(i)));
    } else {
      children.push_back(std::make_shared<core::FieldAccessExpr>(
          rowType.nameOf(i),
          rowType.nameOf(i),
          std::vector<core::ExprPtr>{inputExpr}));
    }
  }

  return std::make_shared<core::CallExpr>(
      "row_constructor", std::move(children), columnAlias);
}

const ArrayTransform kArrayTransform;
const MapTransform kMapTransform;
const RowTransform kRowTransform;
const TimestampWithTimeZoneTransform kTimestampWithTimeZoneTransform;
} // namespace

bool isIntermediateOnlyType(const TypePtr& type) {
  if (isTimestampWithTimeZoneType(type)) {
    return true;
  }

  for (auto i = 0; i < type->size(); ++i) {
    if (isIntermediateOnlyType(type->childAt(i))) {
      return true;
    }
  }

  return false;
}

VectorPtr transformIntermediateOnlyType(const VectorPtr& vector) {
  const auto& type = vector->type();
  if (type->isArray()) {
    return kArrayTransform.transform(vector);
  } else if (type->isMap()) {
    return kMapTransform.transform(vector);
  } else if (type->isRow()) {
    return kRowTransform.transform(vector);
  } else if (isTimestampWithTimeZoneType(type)) {
    return kTimestampWithTimeZoneTransform.transform(vector);
  }

  VELOX_UNREACHABLE();
}

core::ExprPtr getIntermediateOnlyTypeProjectionExpr(
    const TypePtr& type,
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) {
  if (type->isArray()) {
    return kArrayTransform.projectionExpr(type, inputExpr, columnAlias);
  } else if (type->isMap()) {
    return kMapTransform.projectionExpr(type, inputExpr, columnAlias);
  } else if (type->isRow()) {
    return kRowTransform.projectionExpr(type, inputExpr, columnAlias);
  } else if (isTimestampWithTimeZoneType(type)) {
    return kTimestampWithTimeZoneTransform.projectionExpr(
        type, inputExpr, columnAlias);
  }

  VELOX_UNREACHABLE();
}
} // namespace facebook::velox::exec::test
