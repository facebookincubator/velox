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

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "arrow-adbc/adbc.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::adbc::test {

/// Entrypoint of an in-process fake ADBC driver that serves configured Velox
/// vectors as Arrow data. Pass to the AdbcConnector constructor that accepts
/// an AdbcDriverInitFunc.
extern "C" AdbcStatusCode
adbcTestDriverInit(int version, void* driver, struct AdbcError* error);

/// Process-global state backing the fake driver: the result the next query
/// returns and a record of what the driver was asked to do. The driver
/// entrypoint is a plain function pointer, so tests configure the driver
/// through this singleton.
class AdbcTestDriverState {
 public:
  static AdbcTestDriverState& instance();

  /// Clears the configured result, recorded SQL, options, and error.
  void reset();

  /// Sets the result served by subsequent queries. 'rowType' also determines
  /// the stream schema when 'batches' is empty. 'pool' is used to export the
  /// batches to Arrow; it must stay alive until reset() is called.
  void setResult(
      RowTypePtr rowType,
      std::vector<RowVectorPtr> batches,
      memory::MemoryPool* pool);

  /// Makes subsequent AdbcStatementExecuteQuery calls fail with 'message'.
  void setExecuteError(std::string message);

  /// Returns the SQL text of the last executed statement.
  std::string lastSql();

  /// Returns the database options set via AdbcDatabaseSetOption.
  std::unordered_map<std::string, std::string> databaseOptions();

  // The methods below are used by the driver callbacks.

  void recordSql(std::string sql);

  void recordDatabaseOption(std::string key, std::string value);

  // Returns the configured execute error, or an empty string.
  std::string executeError();

  // Returns the configured result.
  std::pair<RowTypePtr, std::vector<RowVectorPtr>> result();

  // Returns the memory pool used to export vectors to Arrow.
  memory::MemoryPool* pool();

 private:
  AdbcTestDriverState() = default;

  std::mutex mutex_;
  RowTypePtr rowType_;
  std::vector<RowVectorPtr> batches_;
  std::string lastSql_;
  std::string executeError_;
  std::unordered_map<std::string, std::string> databaseOptions_;
  memory::MemoryPool* pool_{nullptr};
};

} // namespace facebook::velox::connector::adbc::test
