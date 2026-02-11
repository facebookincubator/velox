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

 #include "velox/experimental/cudf/expression/DecimalUtils.h"

namespace facebook::velox::cudf_velox {

bool containsDecimalType(const std::shared_ptr<velox::exec::Expr>& expr) {
  if (!expr) {
    return false;
  }
  if (expr->type() && expr->type()->isDecimal()) {
    return true;
  }
  for (const auto& input : expr->inputs()) {
    if (containsDecimalType(input)) {
      return true;
    }
  }
  return false;
}

} // namespace facebook::velox::cudf_velox
