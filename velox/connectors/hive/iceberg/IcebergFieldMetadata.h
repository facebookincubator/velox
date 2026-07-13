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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::connector::hive::iceberg {

/// Iceberg V3 type-disambiguation attributes for a single schema node,
/// carried alongside (not folded into) the format-agnostic
/// dwio::common::ParquetFieldId so the Iceberg V3 vocabulary does not leak
/// into the 23+ Parquet call sites that consume ParquetFieldId.
///
/// Each field is optional and only set when the planner derived it from the
/// Iceberg schema; an all-empty instance stamps nothing, keeping output
/// byte-identical to a writer that only emits `iceberg.id`. The `children`
/// vector parallels `ParquetFieldId::children` so the two trees are walked in
/// lockstep. Values map to the Iceberg V3 ORC Appendix A attribute keys.
struct IcebergFieldMetadata {
  /// Maps to `iceberg.required` ("true"/"false").
  std::optional<bool> required;

  /// Maps to `iceberg.long-type` (e.g. "LONG").
  std::optional<std::string> longType;

  /// Maps to `iceberg.timestamp-unit` (e.g. "MICROS"/"NANOS").
  std::optional<std::string> timestampUnit;

  /// Maps to `iceberg.binary-type` (e.g. "FIXED"/"UUID"/"Variant").
  std::optional<std::string> binaryType;

  /// Maps to `iceberg.struct-type` (e.g. the Variant struct marker).
  std::optional<std::string> structType;

  /// Maps to `iceberg.length` (e.g. FixedType length, UUID length 16).
  std::optional<int32_t> length;

  /// Per-child metadata, parallel to ParquetFieldId::children.
  std::vector<IcebergFieldMetadata> children;

  /// True when no attribute on this node is set (children are not
  /// considered).
  bool empty() const {
    return !required.has_value() && !longType.has_value() &&
        !timestampUnit.has_value() && !binaryType.has_value() &&
        !structType.has_value() && !length.has_value();
  }
};

} // namespace facebook::velox::connector::hive::iceberg
