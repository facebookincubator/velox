/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#pragma once

#include "velox/common/EnumDeclare.h"

namespace facebook::velox::common::vectorindex {

/// Distance metric of a vector index.
enum class VectorIndexMetric {
  /// The angle between vectors, ignoring their length. Vectors are normalized
  /// to length 1 and compared by dot product.
  kCosine,
};

VELOX_DECLARE_ENUM_NAME(VectorIndexMetric);

} // namespace facebook::velox::common::vectorindex
