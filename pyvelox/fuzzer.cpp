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

#include "fuzzer.h" // @manual
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/sparksql/Register.h"
#include "velox/functions/FunctionRegistry.h"

namespace facebook::velox::py {
using namespace velox;
namespace py = pybind11;

FunctionSignatureMap getPrestoSignatures() {
  clearFunctionRegistry();
  facebook::velox::functions::prestosql::registerAllScalarFunctions();
  return getFunctionSignatures();
}

FunctionSignatureMap getSparkSignatures() {
  clearFunctionRegistry();
  facebook::velox::functions::sparksql::registerFunctions("");
  return getFunctionSignatures();
}

void addSignatureBindings(py::module& m,
                          bool asModuleLocalDefinitions) {
  //TypeSignature
  py::class_<exec::TypeSignature>  typeSignature(m, "TypeSignature",
                                                py::module_local(asModuleLocalDefinitions));
  typeSignature.def("__str__", &exec::TypeSignature::toString);
  typeSignature.def("base_name", &exec::TypeSignature::baseName);
  typeSignature.def("parameters", &exec::TypeSignature::parameters);

  //FunctionSignature
  py::class_<exec::FunctionSignature> functionSignature(m, "FunctionSignature",
                                                        py::module_local(asModuleLocalDefinitions));

  functionSignature.def("__str__", &exec::FunctionSignature::toString);
  functionSignature.def("return_type", &exec::FunctionSignature::returnType);
  functionSignature.def("argument_types", &exec::FunctionSignature::argumentTypes);
  functionSignature.def("variable_arity", &exec::FunctionSignature::variableArity);
  functionSignature.def("variables", &exec::FunctionSignature::variables);

  m.def("spark_signatures", &getSparkSignatures, py::return_value_policy::reference,
        "Returns a dictionary of spark function signatures.");
  m.def("presto_signatures", &getPrestoSignatures, py::return_value_policy::reference,
        "Returns a dictionary of presto function signatures.");
}
}

