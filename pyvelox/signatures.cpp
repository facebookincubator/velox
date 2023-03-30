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

#include "signatures.h" // @manual
#include "velox/functions/FunctionRegistry.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/sparksql/Register.h"

namespace facebook::velox::py {

namespace py = pybind11;

void registerPrestoFunctions(const std::string& prefix) {
  facebook::velox::functions::prestosql::registerAllScalarFunctions(prefix);
}

void registerSparkFunctions(const std::string& prefix) {
  facebook::velox::functions::sparksql::registerFunctions(prefix);
}

void addSignatureBindings(py::module& m, bool asModuleLocalDefinitions) {
  // TypeSignature
  py::class_<exec::TypeSignature> typeSignature(
      m, "TypeSignature", py::module_local(asModuleLocalDefinitions));
  typeSignature.def("__str__", &exec::TypeSignature::toString);
  typeSignature.def("base_name", &exec::TypeSignature::baseName);
  typeSignature.def("parameters", &exec::TypeSignature::parameters);

  // FunctionSignature
  py::class_<exec::FunctionSignature> functionSignature(
      m, "FunctionSignature", py::module_local(asModuleLocalDefinitions));

  functionSignature.def("__str__", &exec::FunctionSignature::toString);
  functionSignature.def("return_type", &exec::FunctionSignature::returnType);
  functionSignature.def(
      "argument_types", &exec::FunctionSignature::argumentTypes);
  functionSignature.def(
      "variable_arity", &exec::FunctionSignature::variableArity);
  functionSignature.def("variables", &exec::FunctionSignature::variables);
  functionSignature.def(
      "constant_arguments", &exec::FunctionSignature::constantArguments);

  m.def(
      "clear_signatures",
      &clearFunctionRegistry,
      "Clears the function registry.");

  m.def(
      "register_spark_signatures",
      &registerSparkFunctions,
      "Adds Spark signatures to the function registry.",
      py::arg("prefix") = "");

  m.def(
      "register_presto_signatures",
      &registerPrestoFunctions,
      "Adds Presto signatures to the function registry.",
      py::arg("prefix") = "");

  m.def(
      "get_function_signatures",
      &getFunctionSignatures,
      py::return_value_policy::reference,
      "Returns a dictionary of the current signatures.");
}
} // namespace facebook::velox::py
