/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/type/TypeUtil.h"

namespace facebook::velox::type {

RowTypePtr concatRowTypes(const std::vector<RowTypePtr>& rowTypes) {
  std::vector<std::string> columnNames;
  std::vector<TypePtr> columnTypes;
  for (auto& rowType : rowTypes) {
    columnNames.insert(
        columnNames.end(), rowType->names().begin(), rowType->names().end());
    columnTypes.insert(
        columnTypes.end(),
        rowType->children().begin(),
        rowType->children().end());
  }
  return ROW(std::move(columnNames), std::move(columnTypes));
}

TypePtr tryGetHomogeneousRowChild(const TypePtr& type) {
  VELOX_DCHECK(type != nullptr);
  if (type->kind() != TypeKind::ROW) {
    return nullptr;
  }

  const auto childCount = type->size();
  if (childCount == 0) {
    return nullptr; // No child type to infer
  }

  const auto& first = type->childAt(0);
  for (size_t i = 1; i < childCount; ++i) {
    const auto& child = type->childAt(i);
    if (!(*first == *child)) {
      return nullptr;
    }
  }
  return first;
}

} // namespace facebook::velox::type
