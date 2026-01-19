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

namespace facebook::velox::connector::hive::iceberg {

/// Config for collecting Iceberg parquet field statistics.
/// Holds the Iceberg source field id and whether to skip bounds
/// collection for this field. For nested field, it contains child fields.
struct IcebergFieldStatsConfig {
  int32_t fieldId;

  /// Whether to skip collecting min/max bounds statistics for this field and
  /// all its descendants. When true, lower and upper bounds are NOT collected,
  /// This is automatically set to true for MAP and ARRAY types and propagated
  /// to all their nested children.
  bool skipBounds;

  std::vector<IcebergFieldStatsConfig> children;

  IcebergFieldStatsConfig(int32_t _id, bool _skip)
      : fieldId(_id), skipBounds(_skip) {}
};

} // namespace facebook::velox::connector::hive::iceberg
