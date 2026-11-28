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
#include <unordered_map>
#include <vector>

#include "velox/connectors/Connector.h"

namespace facebook::velox::connector::lance {

/// Parameters of an approximate-nearest-neighbor (k-NN) search pushed down
/// to the Lance scanner. Produced on the coordinator by the vector_search
/// table function and carried here so LanceDataSource can configure the
/// scanner's nearest query. Lance appends a float32 `_distance` column
/// (smaller is always nearer, for every metric) to each batch of a nearest
/// scan.
struct LanceVectorSearchParams {
  /// Vector column to search (FixedSizeList<float> in the dataset schema).
  std::string column;
  /// The query vector; its size is the vector dimension.
  std::vector<float> queryVector;
  /// Number of nearest neighbors to return.
  uint32_t k{0};
  /// Distance metric: "l2", "cosine" or "dot".
  std::string metric;
  /// IVF partitions to probe; unset uses the Lance default.
  std::optional<uint32_t> nprobes;
  /// HNSW search list size; unset uses the Lance default.
  std::optional<uint32_t> ef;
  /// Re-rank factor over raw index results; unset disables refinement.
  std::optional<uint32_t> refineFactor;
};

/// Represents a handle to a Lance dataset for table scan operations.
///
/// Holds the logical table name along with the settings that are resolved
/// once during planning and shared by every split of the scan: the dataset
/// version, the object store options used to open it, and the predicate
/// pushed to the Lance scanner. The physical dataset path varies per split
/// and is carried by LanceConnectorSplit.
class LanceTableHandle : public ConnectorTableHandle {
 public:
  /// @param datasetVersion Dataset version to open. 0 opens the latest
  /// version. Resolving the version during planning keeps every split of a
  /// query on the same snapshot even while the dataset is being written to.
  /// @param storageOptions Object store options such as region, endpoint and
  /// credentials, forwarded to the Lance dataset open call.
  /// @param filter Predicate pushed to the Lance scanner, written in the SQL
  /// dialect Lance accepts, for example "age > 30 AND name IS NOT NULL". Empty
  /// means no filter. Lance evaluates this predicate in full, so the planner
  /// must push only predicates Lance can express and keep the rest above the
  /// scan.
  /// @param vectorSearch Optional k-NN search to run instead of a plain
  /// scan. When set, the scan returns the k nearest rows to the query
  /// vector plus a `_distance` column.
  LanceTableHandle(
      const std::string& connectorId,
      const std::string& tableName,
      uint64_t datasetVersion = 0,
      std::unordered_map<std::string, std::string> storageOptions = {},
      const std::string& filter = "",
      std::optional<LanceVectorSearchParams> vectorSearch = std::nullopt)
      : ConnectorTableHandle(connectorId),
        tableName_(tableName),
        datasetVersion_(datasetVersion),
        storageOptions_(std::move(storageOptions)),
        filter_(filter),
        vectorSearch_(std::move(vectorSearch)) {}

  const std::string& name() const override {
    return tableName_;
  }

  const std::string& tableName() const {
    return tableName_;
  }

  /// Returns the dataset version to open, or 0 for the latest version.
  uint64_t datasetVersion() const {
    return datasetVersion_;
  }

  const std::unordered_map<std::string, std::string>& storageOptions() const {
    return storageOptions_;
  }

  /// Returns the pushed predicate, or an empty string when none was pushed.
  const std::string& filter() const {
    return filter_;
  }

  /// Returns the k-NN search to run, or std::nullopt for a plain scan.
  const std::optional<LanceVectorSearchParams>& vectorSearch() const {
    return vectorSearch_;
  }

  /// Renders the handle for logging. Reports only the number of storage
  /// options, never their values, because they can hold credentials.
  std::string toString() const override {
    std::string vectorSearchText;
    if (vectorSearch_.has_value()) {
      vectorSearchText = fmt::format(
          ", vectorSearch: {{column: {}, dim: {}, k: {}, metric: {}}}",
          vectorSearch_->column,
          vectorSearch_->queryVector.size(),
          vectorSearch_->k,
          vectorSearch_->metric);
    }
    return fmt::format(
        "LanceTableHandle [table: {}, version: {}, storageOptions: {}, filter: {}{}]",
        tableName_,
        datasetVersion_ == 0 ? "latest" : std::to_string(datasetVersion_),
        storageOptions_.size(),
        filter_.empty() ? "<none>" : filter_,
        vectorSearchText);
  }

 private:
  const std::string tableName_;
  const uint64_t datasetVersion_;
  const std::unordered_map<std::string, std::string> storageOptions_;
  const std::string filter_;
  const std::optional<LanceVectorSearchParams> vectorSearch_;
};

} // namespace facebook::velox::connector::lance
