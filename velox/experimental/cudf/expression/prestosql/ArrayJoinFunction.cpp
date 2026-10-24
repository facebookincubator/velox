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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/prestosql/ArrayJoinFunction.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>

#include <string>

namespace facebook::velox::cudf_velox::prestosql {

bool ArrayJoinFunction::canEvaluate(const core::TypedExprPtr& expr) {
  // The registered signature, matched before this runs, fixes the arity and
  // the argument types, and the CPU resolves array_join to VARCHAR. The
  // binder matches the array(varchar) element type by name, so a JSON array,
  // whose elements the CPU unescapes before joining, stays on the CPU.
  return !expr->inputs()[0]->isConstantKind() &&
      constantVarcharValue(expr->inputs()[1]).has_value();
}

ArrayJoinFunction::ArrayJoinFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* /*pool*/) {
  VELOX_CHECK(
      canEvaluate(expr),
      "array_join expression cannot be evaluated by "
      "prestosql::ArrayJoinFunction");
  separator_ = std::string(*constantVarcharValue(expr->inputs()[1]));
}

ColumnOrView ArrayJoinFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  const cudf::string_scalar separator(separator_, true, stream, mr);
  // Presto writes nothing for a null element, not even the separator.
  const cudf::string_scalar nullElement(std::string{}, true, stream, mr);
  return cudf::strings::join_list_elements(
      cudf::lists_column_view(asView(inputColumns[0])),
      separator,
      nullElement,
      cudf::strings::separator_on_nulls::NO,
      cudf::strings::output_if_empty_list::EMPTY_STRING,
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox::prestosql
