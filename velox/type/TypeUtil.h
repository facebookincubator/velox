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

#pragma once

#include <vector>

#include "velox/type/Type.h"

namespace facebook::velox::type {

RowTypePtr concatRowTypes(const std::vector<RowTypePtr>& rowTypes);

/// Returns the common child type if 'type' is a Row where all children are
/// the same. Returns nullptr otherwise. Empty row returns nullptr.
/// @pre 'type' must not be null.
TypePtr tryGetHomogeneousRowChild(const TypePtr& type);

} // namespace facebook::velox::type
