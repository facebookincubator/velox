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

// Converts TIME values to BIGINT (milliseconds since start of day)
core::ExprPtr TimeTransform::projectToTargetType(
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  // TIME values are stored as milliseconds since midnight
  // Convert BIGINT back to TIME by casting
  return std::make_shared<core::CastExpr>(
      TIME(), inputExpr, false, columnAlias);
}

// Converts BIGINT (milliseconds) back to TIME values
core::ExprPtr TimeTransform::projectToIntermediateType(
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  // Extract milliseconds from TIME and store as BIGINT
  return std::make_shared<core::CastExpr>(
      BIGINT(), inputExpr, false, columnAlias);
}

} // namespace facebook::velox::exec::test
