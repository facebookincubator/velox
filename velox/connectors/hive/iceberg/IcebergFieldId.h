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
#include <vector>

namespace facebook::velox::connector::hive::iceberg {

/// Connector-local representation of Iceberg field IDs, decoupled from
/// dwio::common::ParquetFieldId to avoid exposing Parquet-specific types in
/// public Iceberg connector APIs. Mirrors the hierarchical structure needed for
/// nested schema support.
struct IcebergFieldId {
  int32_t fieldId{0};
  std::vector<IcebergFieldId> children;
};

} // namespace facebook::velox::connector::hive::iceberg
