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

#include <string>
#include <vector>

#include "velox/connectors/clp/search_lib/ClpQueryRunner.h"

namespace clp_s {
enum class InputSource : uint8_t;
class ArchiveReader;
class BaseColumnReader;
} // namespace clp_s

namespace clp_s::search {
class Projection;
class SchemaMatch;
} // namespace clp_s::search

namespace clp_s::search::ast {
class Expression;
} // namespace clp_s::search::ast

namespace facebook::velox::connector::clp::search_lib {

enum class ErrorCode {
  DictionaryNotFound,
  InternalError,
  InvalidQuerySyntax,
  InvalidTimestampRange,
  LogicalError,
  QueryNotInitialized,
  SchemaNotFound,
  Success
};

enum class ColumnType {
  Array,
  Boolean,
  Float,
  Integer,
  String,
  Timestamp,
  Unknown = -1
};

struct Field {
  ColumnType type;
  std::string name;
};

/// A query execution interface that manages the lifecycle of a query on a CLP-S
/// archive, including parsing and validating the query, loading the relevant
/// schemas and archives, applying filters, and iterating over the results. It
/// abstracts away the low-level details of archive access and schema matching
/// while supporting projection and batch-oriented retrieval of filtered rows.
class ClpCursor {
 public:
  explicit ClpCursor(clp_s::InputSource inputSource, std::string archivePath);
  ~ClpCursor();

  /// Executes a query. This function parses, validates, and prepares the given
  /// query for execution.
  ///
  /// @param query The KQL query to execute.
  /// @param outputColumns A vector specifying the columns to be included in the
  /// query result.
  void executeQuery(
      const std::string& query,
      const std::vector<Field>& outputColumns);

  /// Fetches the next set of rows from the cursor. If the archive and schema
  /// are not yet loaded, this function will perform the necessary loading.
  ///
  /// @param numRows The maximum number of rows to fetch.
  /// @param filteredRowIndices A vector of row indices that match the filter.
  /// @return The number of rows scanned.
  uint64_t fetchNext(
      uint64_t numRows,
      const std::shared_ptr<std::vector<uint64_t>>& filteredRowIndices);

  /// Retrieves the projected columns.
  ///
  /// @return A vector of BaseColumnReader pointers representing the projected
  /// columns.
  const std::vector<clp_s::BaseColumnReader*>& getProjectedColumns() const;

 private:
  /// Preprocesses the query, performing parsing, validation, and optimization.
  ///
  /// @return The error code.
  ErrorCode preprocessQuery();

  /// Loads the archive at the current index.
  ///
  /// @return The error code.
  ErrorCode loadArchive();

  ErrorCode errorCode_;

  clp_s::InputSource inputSource_{clp_s::InputSource::Filesystem};
  std::string archivePath_;
  std::string query_;
  std::vector<Field> outputColumns_;
  std::vector<int32_t> matchedSchemas_;
  size_t currentSchemaIndex_{0};
  int32_t currentSchemaId_{-1};
  bool currentSchemaTableLoaded_{false};
  bool currentArchiveLoaded_{false};

  std::shared_ptr<clp_s::search::ast::Expression> expr_;
  std::shared_ptr<clp_s::search::SchemaMatch> schemaMatch_;
  std::shared_ptr<ClpQueryRunner> queryRunner_;
  std::shared_ptr<clp_s::search::Projection> projection_;
  std::shared_ptr<clp_s::ArchiveReader> archiveReader_;
};

} // namespace facebook::velox::connector::clp::search_lib
