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

#include "velox/experimental/cudf/exec/CustomComparison.h"

#include <algorithm>

namespace facebook::velox::cudf_velox {

bool containsCustomComparison(const TypePtr& type) {
  if (type->providesCustomComparison()) {
    return true;
  }
  for (uint32_t i = 0; i < type->size(); ++i) {
    if (containsCustomComparison(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

bool keysUseCustomComparison(
    const std::vector<core::FieldAccessTypedExprPtr>& keys) {
  return std::any_of(keys.begin(), keys.end(), [](const auto& key) {
    return containsCustomComparison(key->type());
  });
}

bool exprUsesCustomComparison(const core::TypedExprPtr& expr) {
  if (containsCustomComparison(expr->type())) {
    return true;
  }
  const auto& inputs = expr->inputs();
  return std::any_of(inputs.begin(), inputs.end(), exprUsesCustomComparison);
}

} // namespace facebook::velox::cudf_velox
