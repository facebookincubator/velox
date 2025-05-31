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

#include "velox/exec/fuzzer/PrestoQueryRunnerHyperLogLogTransform.h"

#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/parse/Expressions.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::exec::test {

variant HyperLogLogTransform::transform(
    const BaseVector* const vector,
    vector_size_t row) const {
  VELOX_CHECK(isHyperLogLogType(vector->type()));
  VELOX_CHECK(!vector->isNullAt(row));

  return variant::create<TypeKind::VARBINARY>(
      vector->asChecked<SimpleVector<StringView>>()->valueAt(row));
}

core::ExprPtr HyperLogLogTransform::projectionExpr(
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  return std::make_shared<core::CastExpr>(
      HYPERLOGLOG(), inputExpr, false, columnAlias);
}

} // namespace facebook::velox::exec::test
