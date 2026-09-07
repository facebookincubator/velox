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
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "velox/dwio/common/FileMetadata.h"
#include "velox/type/Type.h"

namespace facebook::nimble {

/// File-level metadata returned by the NIMBLE writer when a file is closed.
/// Carries a deep copy of the per-column statistics (the writer's own
/// statistics views are non-owning and become invalid once the writer is
/// destroyed) together with the file schema, so downstream consumers such as
/// the Iceberg connector can aggregate field-id-keyed manifest statistics.
class NimbleFileMetadata : public velox::dwio::common::FileMetadata {
 public:
  /// Snapshot of a single schema node's statistics. The min/max optionals are
  /// mutually exclusive by physical category: at most one of the integral,
  /// floating-point, or string pairs is populated, matching the concrete
  /// NIMBLE ColumnStatistics subtype. Container nodes (ROW/ARRAY/MAP) carry
  /// only counts.
  struct ColumnStats {
    uint64_t valueCount{0};
    uint64_t nullCount{0};
    uint64_t physicalSize{0};
    std::optional<int64_t> integralMin;
    std::optional<int64_t> integralMax;
    std::optional<double> floatingMin;
    std::optional<double> floatingMax;
    std::optional<std::string> stringMin;
    std::optional<std::string> stringMax;
  };

  NimbleFileMetadata(
      velox::RowTypePtr schema,
      int64_t numRows,
      std::vector<ColumnStats> columnStats)
      : schema_{std::move(schema)},
        numRows_{numRows},
        columnStats_{std::move(columnStats)} {}

  /// File schema. Used to rebuild a TypeWithId tree whose pre-order ids align
  /// with the columnStats() vector.
  const velox::RowTypePtr& schema() const {
    return schema_;
  }

  /// Total number of rows in the file.
  int64_t numRows() const {
    return numRows_;
  }

  /// Per-node statistics indexed by TypeWithId::id() (schema pre-order DFS,
  /// root at index 0).
  const std::vector<ColumnStats>& columnStats() const {
    return columnStats_;
  }

 private:
  const velox::RowTypePtr schema_;
  const int64_t numRows_;
  const std::vector<ColumnStats> columnStats_;
};

} // namespace facebook::nimble
