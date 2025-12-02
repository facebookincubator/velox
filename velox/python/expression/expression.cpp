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

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "velox/python/expression/PyExpression.h"
#include "velox/type/Variant.h"

namespace py = pybind11;
namespace {

using namespace facebook;

/// Converts a given Python object into a Velox constant expression. Throws
/// TypeError if it cannot convert.
velox::py::PyExpr toConstantExpr(const py::object& arg) {
  // Bool (True|False).
  if (py::isinstance<py::bool_>(arg)) {
    return velox::py::PyExpr::createConstant(
        velox::BOOLEAN(), velox::Variant(arg.cast<bool>()));
  }
  // None.
  else if (py::isinstance<py::none>(arg)) {
    return velox::py::PyExpr::createConstant(
        velox::UNKNOWN(), velox::Variant::null(velox::TypeKind::UNKNOWN));
  }
  // Integers. For now we assume always bigint (64 bits).
  else if (py::isinstance<py::int_>(arg)) {
    return velox::py::PyExpr::createConstant(
        velox::BIGINT(), velox::Variant(arg.cast<int64_t>()));
  }
  // Floats. Assume always double (64 bits) for now.
  else if (py::isinstance<py::float_>(arg)) {
    return velox::py::PyExpr::createConstant(
        velox::DOUBLE(), velox::Variant(arg.cast<double>()));
  }
  // Strings.
  else if (py::isinstance<py::str>(arg)) {
    return velox::py::PyExpr::createConstant(
        velox::VARCHAR(), velox::Variant(arg.cast<std::string>()));
  }
  // TODO: Support container types (arrays/list, maps, and structs).
  throw py::type_error(
      fmt::format("Unsupported PyVelox literal type: {}", py::type::of(arg)));
}

velox::py::PyExpr toExpr(const py::object& arg) {
  // If the object is already an expression, just return it.
  if (py::isinstance<velox::py::PyExpr>(arg)) {
    return arg.cast<velox::py::PyExpr>();
  }
  // Otherwise, try to convert to a literal/constant.
  return toConstantExpr(arg);
}

} // namespace

PYBIND11_MODULE(expression, m) {
  using velox::py::PyExpr;

  py::class_<PyExpr>(m, "Expr")
      .def("__str__", &PyExpr::toString)
      // Binary arithmetic operators.
      .def(
          "__add__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall(
                "plus", {self.expr(), toExpr(arg).expr()});
          })
      .def(
          "__radd__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall(
                "plus", {toExpr(arg).expr(), self.expr()});
          })
      .def(
          "__sub__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall(
                "minus", {self.expr(), toExpr(arg).expr()});
          })
      .def(
          "__rsub__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall(
                "minus", {toExpr(arg).expr(), self.expr()});
          })
      .def(
          "__mul__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall(
                "multiply", {self.expr(), toExpr(arg).expr()});
          })
      .def(
          "__rmul__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall(
                "multiply", {toExpr(arg).expr(), self.expr()});
          })
      .def(
          "__truediv__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall(
                "divide", {self.expr(), toExpr(arg).expr()});
          })
      .def(
          "__rtruediv__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall(
                "divide", {toExpr(arg).expr(), self.expr()});
          })
      // Conjunts.
      .def(
          "__and__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall("and", {self.expr(), toExpr(arg).expr()});
          })
      .def(
          "__rand__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall("and", {toExpr(arg).expr(), self.expr()});
          })
      .def(
          "__or__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall("or", {self.expr(), toExpr(arg).expr()});
          })
      .def(
          "__ror__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall("or", {toExpr(arg).expr(), self.expr()});
          })
      .def(
          "__invert__",
          [](const PyExpr& self) {
            return PyExpr::createCall("not", {self.expr()});
          })
      // Comparisons don't need a "right-hand" version of the method as python
      // assumes them to be perfectly symmetrical.
      .def(
          "__gt__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall("gt", {self.expr(), toExpr(arg).expr()});
          })
      .def(
          "__lt__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall("lt", {self.expr(), toExpr(arg).expr()});
          })
      .def(
          "__ge__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall("gte", {self.expr(), toExpr(arg).expr()});
          })
      .def(
          "__le__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall("lte", {self.expr(), toExpr(arg).expr()});
          })
      .def(
          "__eq__",
          [](const PyExpr& self, const py::object& arg) {
            return PyExpr::createCall("eq", {self.expr(), toExpr(arg).expr()});
          })
      .def("__ne__", [](const PyExpr& self, const py::object& arg) {
        return PyExpr::createCall("neq", {self.expr(), toExpr(arg).expr()});
      });

  // TODO: Add more arithmetics binary and unary operators.

  m.def("col", &PyExpr::createFieldAccess, py::doc(R"(
    Create a reference to a column. To be used when building expressions.
  )"));
  m.def("lit", &toConstantExpr, py::doc(R"(
    Create a literal to be used when building expressions.
  )"));
}
