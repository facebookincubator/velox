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

#include "velox/common/base/Fs.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/dwrf/writer/StatisticsBuilder.h"

namespace facebook::velox::exec {

struct LocalColumn {
  LocalColumn(const std::string& name, TypePtr type) : name(name), type(type) {}

  void addStats(std::unique_ptr<dwio::common::ColumnStatistics> stats);

  std::string name;
  TypePtr type;
  std::unique_ptr<dwio::common::ColumnStatistics> stats;
  int64_t numDistinct;
};

class LocalSchema;

struct LocalTable {
  LocalTable(
      const std::string& name,
      dwio::common::FileFormat format,
      LocalSchema* schema)
      : name(name), format(format), schema(schema) {}

  const RowTypePtr& rowType() const {
    return type;
  }

  /// Samples 'pct' percent of rows for 'fields'. Applies 'filters'
  /// before sampling. Returns {count of sampled, count matching filters}.
  /// Returns statistics for the post-filtering values in 'stats' for each of
  /// 'fields'. If 'fields' is empty, simply returns the number of
  /// rows matching 'filter' in a sample of 'pct'% of the table.
  std::pair<int64_t, int64_t> sample(
      float pct,
      const std::vector<common::Subfield>& columns,
      connector::hive::SubfieldFilters filters,
      const core::TypedExprPtr& remainingFilter,
      HashStringAllocator* allocator = nullptr,
      std::vector<std::unique_ptr<dwrf::StatisticsBuilder>>* stats = nullptr);

  std::string name;
  dwio::common::FileFormat format;
  RowTypePtr type;
  std::vector<std::string> files;
  std::unordered_map<std::string, std::unique_ptr<LocalColumn>> columns;
  int64_t numRows{0};
  LocalSchema* schema;
  int64_t numSampledRows{0};
};

class LocalSchema {
 public:
  LocalSchema(
      const std::string& path,
      dwio::common::FileFormat format,
      connector::hive::HiveConnector* hiveConector,
      std::shared_ptr<connector::ConnectorQueryCtx> ctx);

  const std::unordered_map<std::string, std::unique_ptr<LocalTable>>& tables() {
    return tables_;
  }

  LocalTable* findTable(const std::string& name) {
    auto it = tables_.find(name);
    VELOX_CHECK(it != tables_.end(), "Table {} not found", name);
    return it->second.get();
  }

  connector::Connector* connector() const {
    return hiveConnector_;
  }

  const std::shared_ptr<connector::ConnectorQueryCtx>& connectorQueryCtx()
      const {
    return connectorQueryCtx_;
  }

  memory::MemoryPool* pool() {
    return pool_.get();
  }

 private:
  void initialize(const std::string& path);

  void readTable(const std::string& tableName, const fs::path& tablePath);

  connector::hive::HiveConnector* hiveConnector_;
  std::string connectorId_;
  std::shared_ptr<memory::MemoryPool> pool_;

  std::shared_ptr<connector::ConnectorQueryCtx> connectorQueryCtx_;
  dwio::common::FileFormat format_;

  std::unordered_map<std::string, std::unique_ptr<LocalTable>> tables_;
};

} // namespace facebook::velox::exec
