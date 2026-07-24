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

#include "velox/connectors/hive/iceberg/IcebergFieldIdUtils.h"

#include <folly/String.h>

namespace facebook::velox::connector::hive::iceberg {

// NOTE: The flat name->fieldId map used here does not handle schemas where the
// same field name appears at multiple nesting levels (e.g.
// struct<a: struct<id: int>, b: struct<id: int>>). In that case both inner
// `id` fields collide to the same map key. This is acceptable for the common
// case (Iceberg tables with distinct field names across levels) but is a known
// limitation. A future improvement could scope the map by parent field ID.
void extractNestedFieldIds(
    const parquet::ParquetFieldId& field,
    const RowTypePtr& rowType,
    std::unordered_map<std::string, int32_t>& mapping) {
  if (!rowType || field.children.empty()) {
    return;
  }

  const auto numChildren =
      std::min<size_t>(field.children.size(), rowType->size());
  for (size_t i = 0; i < numChildren; ++i) {
    const auto& child = field.children[i];
    std::string lowerName = rowType->nameOf(i);
    folly::toLowerAscii(lowerName);
    mapping[lowerName] = child.fieldId;

    if (!child.children.empty()) {
      auto childRowType = asRowType(rowType->childAt(i));
      if (childRowType) {
        extractNestedFieldIds(child, childRowType, mapping);
      }
    }
  }
}

} // namespace facebook::velox::connector::hive::iceberg
