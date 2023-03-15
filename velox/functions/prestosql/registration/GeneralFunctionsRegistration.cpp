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
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/IsNull.h"
#include "velox/functions/prestosql/Cardinality.h"

namespace facebook::velox::functions {

// Special form functions don't have any prefix.
void registerAllSpecialFormGeneralFunctions() {
  VELOX_REGISTER_VECTOR_FUNCTION(udf_in, "in");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_concat_row, "row_constructor");
  registerIsNullFunction("is_null");
}

void registerGeneralFunctions(const std::string& prefix) {
  VELOX_REGISTER_VECTOR_FUNCTION(udf_element_at, prefix + "element_at");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_subscript, prefix + "subscript");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_transform, prefix + "transform");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_reduce, prefix + "reduce");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_filter, prefix + "filter");

  VELOX_REGISTER_VECTOR_FUNCTION(udf_least, prefix + "least");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_greatest, prefix + "greatest");

  registerFunction<CardinalityFunction, int64_t, Array<Any>>(
      {prefix + "cardinality"});
  registerFunction<CardinalityFunction, int64_t, Map<Any, Any>>(
      {prefix + "cardinality"});

  registerAllSpecialFormGeneralFunctions();
}

} // namespace facebook::velox::functions
