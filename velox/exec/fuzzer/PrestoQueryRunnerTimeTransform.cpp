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

#include "velox/exec/fuzzer/PrestoQueryRunnerTimeTransform.h"
#include "velox/parse/Expressions.h"

namespace facebook::velox::exec::test {

core::ExprPtr TimeTransform::projectToTargetType(
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  // TIME -> BIGINT: This is supported by the regular cast system
  return std::make_shared<core::CastExpr>(
      targetType(), inputExpr, false, columnAlias);
}

core::ExprPtr TimeTransform::projectToIntermediateType(
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  // BIGINT -> TIME: For fuzzer-internal transformation, we can directly use
  // the input expression since TIME is internally represented as BIGINT
  // milliseconds. This is safe because the fuzzer ensures valid TIME values are
  // generated. The type system will interpret the BIGINT as TIME in this
  // context.
  return inputExpr;
}

} // namespace facebook::velox::exec::test
