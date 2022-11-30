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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <velox/buffer/StringViewBufferHolder.h>
#include <velox/type/Type.h>
#include <velox/type/Variant.h>
#include <velox/vector/FlatVector.h>
#include "folly/json.h"

namespace facebook::velox::py {

std::string serializeType(const std::shared_ptr<const velox::Type>& type);

inline velox::variant pyToVariant(const pybind11::handle& obj) {
  namespace py = pybind11;
  if (py::isinstance<py::bool_>(obj))
    return velox::variant::create<velox::TypeKind::BOOLEAN>(
        py::cast<bool>(obj));
  else if (py::isinstance<py::int_>(obj))
    return velox::variant::create<velox::TypeKind::BIGINT>(
        py::cast<int64_t>(obj));
  else if (py::isinstance<py::float_>(obj))
    return velox::variant::create<velox::TypeKind::DOUBLE>(
        py::cast<double>(obj));
  else if (py::isinstance<py::str>(obj))
    return velox::variant::create<velox::TypeKind::VARCHAR>(
        py::cast<std::string>(obj));
  else if (obj.is_none())
    return velox::variant();
  throw std::domain_error("Invalid type of object");
}

template <typename T>
inline VectorPtr variantToFlatVector(
    const TypePtr& type,
    const std::vector<velox::variant>& variants,
    std::shared_ptr<facebook::velox::memory::MemoryPool>& pool) {
  std::shared_ptr<FlatVector<T>> result =
      BaseVector::create<FlatVector<T>>(type, variants.size(), pool.get());
  for (int i = 0; i < variants.size(); i++) {
    if (variants[i].isNull())
      result->setNull(i, true);
    else
      result->set(i, variants[i].value<T>());
  }
  return result;
}

template <>
inline VectorPtr variantToFlatVector<velox::StringView>(
    const TypePtr& type,
    const std::vector<velox::variant>& variants,
    std::shared_ptr<facebook::velox::memory::MemoryPool>& pool) {
  velox::StringViewBufferHolder holder(pool.get());
  std::shared_ptr<FlatVector<velox::StringView>> result =
      BaseVector::create<FlatVector<velox::StringView>>(
          type, variants.size(), pool.get());
  for (int i = 0; i < variants.size(); i++) {
    if (variants[i].isNull()) {
      result->setNull(i, true);
    } else {
      velox::StringView view =
          holder.getOwnedValue(variants[i].value<std::string>());
      result->set(i, view);
    }
  }
  result->setStringBuffers(holder.moveBuffers());
  return result;
}

inline VectorPtr pyListToVector(
    const pybind11::list& list,
    std::shared_ptr<facebook::velox::memory::MemoryPool>& pool) {
  std::vector<velox::variant> variants;
  for (auto item : list)
    variants.push_back(pyToVariant(item));

  if (variants.empty())
    return nullptr;

  velox::TypeKind first_kind = velox::TypeKind::INVALID;
  for (velox::variant& var : variants) {
    if (var.hasValue()) {
      if (first_kind == velox::TypeKind::INVALID)
        first_kind = var.kind();
      else if (var.kind() != first_kind)
        throw std::domain_error(
            "Velox Vector must consist of items of the same type");
    }
  }

  TypePtr type = fromKindToScalerType(first_kind);
  switch (first_kind) {
    case velox::TypeKind::BOOLEAN:
      return variantToFlatVector<bool>(type, variants, pool);
    case velox::TypeKind::BIGINT:
      return variantToFlatVector<int64_t>(type, variants, pool);
    case velox::TypeKind::DOUBLE:
      return variantToFlatVector<double>(type, variants, pool);
    case velox::TypeKind::VARCHAR:
      return variantToFlatVector<velox::StringView>(type, variants, pool);
    default:
      throw std::runtime_error("Impossible type detected for velox::variant");
  }
}

inline void addDataTypeBindings(
    pybind11::module& m,
    bool asModuleLocalDefinitions = true) {
  // Inlining these bindings since adding them to the cpp file results in a
  // ASAN error.
  using namespace velox;
  namespace py = pybind11;

  // Add TypeKind enum.
  py::enum_<velox::TypeKind>(
      m, "TypeKind", py::module_local(asModuleLocalDefinitions))
      .value("BOOLEAN", velox::TypeKind::BOOLEAN)
      .value("TINYINT", velox::TypeKind::TINYINT)
      .value("SMALLINT", velox::TypeKind::SMALLINT)
      .value("INTEGER", velox::TypeKind::INTEGER)
      .value("BIGINT", velox::TypeKind::BIGINT)
      .value("REAL", velox::TypeKind::REAL)
      .value("DOUBLE", velox::TypeKind::DOUBLE)
      .value("VARCHAR", velox::TypeKind::VARCHAR)
      .value("VARBINARY", velox::TypeKind::VARBINARY)
      .value("TIMESTAMP", velox::TypeKind::TIMESTAMP)
      .value("OPAQUE", velox::TypeKind::OPAQUE)
      .value("ARRAY", velox::TypeKind::ARRAY)
      .value("MAP", velox::TypeKind::MAP)
      .value("ROW", velox::TypeKind::ROW)
      .export_values();

  // Create VeloxType bound to velox::Type.
  py::class_<Type, std::shared_ptr<Type>> type(
      m, "VeloxType", py::module_local(asModuleLocalDefinitions));

  // Adding all the derived types of Type here.
  py::class_<BooleanType, Type, std::shared_ptr<BooleanType>> booleanType(
      m, "BooleanType", py::module_local(asModuleLocalDefinitions));
  py::class_<IntegerType, Type, std::shared_ptr<IntegerType>> integerType(
      m, "IntegerType", py::module_local(asModuleLocalDefinitions));
  py::class_<BigintType, Type, std::shared_ptr<BigintType>> bigintType(
      m, "BigintType", py::module_local(asModuleLocalDefinitions));
  py::class_<SmallintType, Type, std::shared_ptr<SmallintType>> smallintType(
      m, "SmallintType", py::module_local(asModuleLocalDefinitions));
  py::class_<TinyintType, Type, std::shared_ptr<TinyintType>> tinyintType(
      m, "TinyintType", py::module_local(asModuleLocalDefinitions));
  py::class_<RealType, Type, std::shared_ptr<RealType>> realType(
      m, "RealType", py::module_local(asModuleLocalDefinitions));
  py::class_<DoubleType, Type, std::shared_ptr<DoubleType>> doubleType(
      m, "DoubleType", py::module_local(asModuleLocalDefinitions));
  py::class_<TimestampType, Type, std::shared_ptr<TimestampType>> timestampType(
      m, "TimestampType", py::module_local(asModuleLocalDefinitions));
  py::class_<VarcharType, Type, std::shared_ptr<VarcharType>> varcharType(
      m, "VarcharType", py::module_local(asModuleLocalDefinitions));
  py::class_<VarbinaryType, Type, std::shared_ptr<VarbinaryType>> varbinaryType(
      m, "VarbinaryType", py::module_local(asModuleLocalDefinitions));
  py::class_<ArrayType, Type, std::shared_ptr<ArrayType>> arrayType(
      m, "ArrayType", py::module_local(asModuleLocalDefinitions));
  py::class_<MapType, Type, std::shared_ptr<MapType>> mapType(
      m, "MapType", py::module_local(asModuleLocalDefinitions));
  py::class_<RowType, Type, std::shared_ptr<RowType>> rowType(
      m, "RowType", py::module_local(asModuleLocalDefinitions));
  py::class_<FixedSizeArrayType, Type, std::shared_ptr<FixedSizeArrayType>>
      fixedArrayType(
          m, "FixedSizeArrayType", py::module_local(asModuleLocalDefinitions));

  // Basic operations on Type.
  type.def("__str__", &Type::toString);
  // Gcc doesnt support the below kind of templatization.
#if defined(__clang__)
  // Adds equality and inequality comparison operators.
  type.def(py::self == py::self);
  type.def(py::self != py::self);
#endif
  type.def(
      "cpp_size_in_bytes",
      &Type::cppSizeInBytes,
      "Return the C++ size in bytes");
  type.def(
      "is_fixed_width",
      &Type::isFixedWidth,
      "Check if the type is fixed width");
  type.def(
      "is_primitive_type",
      &Type::isPrimitiveType,
      "Check if the type is a primitive type");
  type.def("kind", &Type::kind, "Returns the kind of the type");
  type.def("serialize", &serializeType, "Serializes the type as JSON");
  type.def("__eq__", &Type::equivalent);

  booleanType.def(py::init());
  tinyintType.def(py::init());
  smallintType.def(py::init());
  integerType.def(py::init());
  bigintType.def(py::init());
  realType.def(py::init());
  doubleType.def(py::init());
  varcharType.def(py::init());
  varbinaryType.def(py::init());
  timestampType.def(py::init());
  arrayType.def(py::init<std::shared_ptr<Type>>());
  arrayType.def(
      "element_type", &ArrayType::elementType, "Return the element type");
  fixedArrayType.def(py::init<int, velox::TypePtr>())
      .def("element_type", &velox::FixedSizeArrayType::elementType)
      .def("fixed_width", &velox::FixedSizeArrayType::fixedElementsWidth);
  mapType.def(py::init<std::shared_ptr<Type>, std::shared_ptr<Type>>());
  mapType.def("key_type", &MapType::keyType, "Return the key type");
  mapType.def("value_type", &MapType::valueType, "Return the value type");

  rowType.def(py::init<
              std::vector<std::string>,
              std::vector<std::shared_ptr<const Type>>>());
  rowType.def("size", &RowType::size, "Return the number of columns");
  rowType.def(
      "child_at",
      &RowType::childAt,
      "Return the type of the column at a given index",
      py::arg("idx"));
  rowType.def(
      "find_child",
      [](const std::shared_ptr<RowType>& type, const std::string& name) {
        return type->findChild(name);
      },
      "Return the type of the column with the given name",
      py::arg("name"));
  rowType.def(
      "get_child_idx",
      &RowType::getChildIdx,
      "Return the index of the column with the given name",
      py::arg("name"));
  rowType.def(
      "name_of",
      &RowType::nameOf,
      "Return the name of the column at the given index",
      py::arg("idx"));
  rowType.def("names", &RowType::names, "Return the names of the columns");
}

inline void addVectorBindings(
    pybind11::module& m,
    bool asModuleLocalDefinitions = true) {
  namespace py = pybind11;
  std::shared_ptr<facebook::velox::memory::MemoryPool> pool =
      facebook::velox::memory::getDefaultMemoryPool();
  py::class_<
      facebook::velox::BaseVector,
      std::shared_ptr<facebook::velox::BaseVector>>(
      m, "BaseVector", py::module_local(asModuleLocalDefinitions))
      .def_static(
          "from_list",
          [pool](const py::list& list) mutable {
            return pyListToVector(list, pool);
          })
      .def("__str__", [](std::shared_ptr<facebook::velox::BaseVector>& v) {
        return v->toString();
      });
}

///  Adds Velox Python Bindings to the module m.
///
/// This function adds the following bindings:
///   * velox::TypeKind enum
///   * velox::Type and its derived types
///   * Basic functions on Type and its derived types.
///
///  @param m Module to add bindings too.
///  @param asModuleLocalDefinitions If true then these bindings are only
///  visible inside the module. Refer to
///  https://pybind11.readthedocs.io/en/stable/advanced/classes.html#module-local-class-bindings
///  for further details.
inline void addVeloxBindings(
    pybind11::module& m,
    bool asModuleLocalDefinitions = true) {
  addDataTypeBindings(m, asModuleLocalDefinitions);
  addVectorBindings(m, asModuleLocalDefinitions);
}

} // namespace facebook::velox::py
