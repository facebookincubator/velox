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

#include "velox/functions/lib/aggregates/VarianceAggregatesBase.h"

namespace facebook::velox::functions::aggregate {

void checkSumCountRowType(
    const TypePtr& type,
    const std::string& errorMessage) {
  VELOX_CHECK_EQ(type->kind(), TypeKind::ROW, "{}", errorMessage);
  VELOX_CHECK_EQ(
      type->childAt(kCountIdx)->kind(), TypeKind::BIGINT, "{}", errorMessage);
  VELOX_CHECK_EQ(
      type->childAt(kMeanIdx)->kind(), TypeKind::DOUBLE, "{}", errorMessage);
  VELOX_CHECK_EQ(
      type->childAt(kM2Idx)->kind(), TypeKind::DOUBLE, "{}", errorMessage);
}

} // namespace facebook::velox::functions::aggregate
