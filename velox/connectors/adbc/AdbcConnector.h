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

#include "arrow-adbc/adbc.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/adbc/AdbcConfig.h"
#include "velox/connectors/adbc/AdbcConnectorSplit.h"

namespace facebook::velox::connector::adbc {

/// Owns the AdbcDatabase shared by all data sources of a connector. Defined
/// in AdbcConnector.cpp.
class AdbcDatabaseHolder;

/// Names a column of the remote table or query result. The handle name is
/// both the plan-side name and the remote column name.
class AdbcColumnHandle : public ColumnHandle {
 public:
  explicit AdbcColumnHandle(const std::string& name) : name_(name) {}

  const std::string& name() const override {
    return name_;
  }

  folly::dynamic serialize() const override;

  static std::shared_ptr<AdbcColumnHandle> create(const folly::dynamic& obj);

  static void registerSerDe();

  VELOX_DEFINE_CLASS_NAME(AdbcColumnHandle)

 private:
  const std::string name_;
};

/// Identifies what to scan: either a remote table by name or the result of a
/// raw SQL query. Exactly one of 'tableName' and 'query' must be non-empty.
///
/// For a table name, the connector generates 'SELECT <columns> FROM
/// <tableName>' with the projected columns quoted using the configured
/// identifier quote. The table name is inserted verbatim so it may carry
/// schema qualification or pre-applied quoting. A query is sent to the driver
/// verbatim; its result schema must cover the projected columns.
class AdbcTableHandle : public ConnectorTableHandle {
 public:
  AdbcTableHandle(
      std::string connectorId,
      std::string tableName,
      std::string query);

  const std::string& name() const override {
    return name_;
  }

  const std::string& tableName() const {
    return tableName_;
  }

  const std::string& query() const {
    return query_;
  }

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);

  static void registerSerDe();

  VELOX_DEFINE_CLASS_NAME(AdbcTableHandle)

 private:
  const std::string tableName_;
  const std::string query_;
  // Display name: the table name, or a fixed label for query handles.
  std::string name_;
};

/// Streams the result of one ADBC query as Velox vectors. Executes the SQL on
/// addSplit and imports each Arrow chunk produced by the driver through the
/// Arrow C-ABI bridge; imported vectors take ownership of the driver's Arrow
/// buffers without copying them.
class AdbcDataSource : public DataSource {
 public:
  AdbcDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& columnHandles,
      const AdbcConfig& config,
      std::shared_ptr<AdbcDatabaseHolder> database,
      ConnectorQueryCtx* connectorQueryCtx);

  ~AdbcDataSource() override;

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  std::optional<RowVectorPtr> next(uint64_t size, velox::ContinueFuture& future)
      override;

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) override {
    VELOX_NYI("Dynamic filters not supported by AdbcConnector.");
  }

  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

 private:
  // Holds the ADBC connection, statement, and result stream of the current
  // split. Defined in AdbcConnector.cpp.
  struct SplitState;

  // Returns the SQL to execute: the handle's raw query, or a SELECT over the
  // projected columns generated from the handle's table name.
  std::string buildSql() const;

  // Executes the SQL and validates the result schema against 'outputType_',
  // filling the stream-column index for every output column.
  void openStream();

  const RowTypePtr outputType_;
  std::shared_ptr<const AdbcTableHandle> tableHandle_;
  const std::string identifierQuote_;
  const std::shared_ptr<AdbcDatabaseHolder> database_;
  memory::MemoryPool* const pool_;

  // Remote column name for each column of 'outputType_'.
  std::vector<std::string> remoteColumnNames_;

  std::unique_ptr<SplitState> splitState_;

  uint64_t completedRows_{0};
  uint64_t completedBytes_{0};
};

/// Connector that scans relational databases (MySQL, PostgreSQL, SQLite, ...)
/// through Arrow ADBC drivers. The database handle is initialized eagerly so
/// driver misconfiguration fails at connector creation rather than at first
/// query.
class AdbcConnector final : public Connector {
 public:
  /// Loads the ADBC driver shared library named by the 'adbc.driver' config.
  AdbcConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* executor);

  /// Uses 'driverInitFunc' as an in-process driver entry point instead of
  /// loading a shared library. Serves embedders that link an ADBC driver
  /// statically, and tests.
  AdbcConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      AdbcDriverInitFunc driverInitFunc);

  ~AdbcConnector() override;

  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override;

  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr /*inputType*/,
      ConnectorInsertTableHandlePtr /*connectorInsertTableHandle*/,
      ConnectorQueryCtx* /*connectorQueryCtx*/,
      CommitStrategy /*commitStrategy*/) override {
    VELOX_NYI("AdbcConnector does not support data sink.");
  }

  static void registerSerDe();

 private:
  const AdbcConfig adbcConfig_;
  std::shared_ptr<AdbcDatabaseHolder> database_;
};

class AdbcConnectorFactory : public ConnectorFactory {
 public:
  static constexpr const char* kAdbcConnectorName{"adbc"};

  AdbcConnectorFactory() : ConnectorFactory(kAdbcConnectorName) {}

  explicit AdbcConnectorFactory(const char* connectorName)
      : ConnectorFactory(connectorName) {}

  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* ioExecutor = nullptr,
      folly::Executor* cpuExecutor = nullptr) override {
    return std::make_shared<AdbcConnector>(id, config, ioExecutor);
  }
};

} // namespace facebook::velox::connector::adbc
