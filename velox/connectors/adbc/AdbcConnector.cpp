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
#include "velox/connectors/adbc/AdbcConnector.h"

#include <cstdlib>
#include <cstring>

#include <folly/String.h>

#include "arrow-adbc/adbc_driver_manager.h"
#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::connector::adbc {

namespace {

// Throws if 'status' is not ADBC_STATUS_OK, using and releasing 'error'.
// 'userError' selects a user error for failures caused by configuration or
// the remote database rather than by Velox internals.
void checkAdbcStatus(
    AdbcStatusCode status,
    AdbcError& error,
    const char* operation,
    bool userError) {
  if (status == ADBC_STATUS_OK) {
    return;
  }
  std::string errorMessage{
      error.message != nullptr ? error.message : "no details"};
  if (error.release != nullptr) {
    error.release(&error);
  }
  if (userError) {
    VELOX_USER_FAIL(
        "ADBC call failed: {} ({}): {}",
        operation,
        AdbcStatusCodeMessage(status),
        errorMessage);
  }
  VELOX_FAIL(
      "ADBC call failed: {} ({}): {}",
      operation,
      AdbcStatusCodeMessage(status),
      errorMessage);
}

// Throws if an ArrowArrayStream callback returned a non-zero errno-style
// code, including the stream's last error text if available.
void checkStreamCall(
    int code,
    ArrowArrayStream& stream,
    const char* operation) {
  if (code == 0) {
    return;
  }
  const char* detail = stream.get_last_error != nullptr
      ? stream.get_last_error(&stream)
      : nullptr;
  VELOX_FAIL(
      "Arrow stream call failed: {} (error code {}): {}",
      operation,
      code,
      detail != nullptr ? detail : "no details");
}

char* copyString(const char* source) {
  return source != nullptr ? ::strdup(source) : nullptr;
}

// Copies an ArrowSchema metadata blob. The blob is length-prefixed binary:
// an int32 pair count followed, for each pair, by an int32 length and bytes
// for the key and the same for the value.
char* copyMetadata(const char* source) {
  if (source == nullptr) {
    return nullptr;
  }
  const char* cursor = source;
  int32_t numPairs;
  ::memcpy(&numPairs, cursor, sizeof(int32_t));
  cursor += sizeof(int32_t);
  for (int32_t pair = 0; pair < numPairs; ++pair) {
    for (int part = 0; part < 2; ++part) {
      int32_t length;
      ::memcpy(&length, cursor, sizeof(int32_t));
      cursor += sizeof(int32_t) + length;
    }
  }
  const size_t totalSize = cursor - source;
  char* copy = static_cast<char*>(::malloc(totalSize));
  ::memcpy(copy, source, totalSize);
  return copy;
}

void releaseCopiedSchema(ArrowSchema* schema) {
  if (schema->release == nullptr) {
    return;
  }
  ::free(const_cast<char*>(schema->format));
  ::free(const_cast<char*>(schema->name));
  ::free(const_cast<char*>(schema->metadata));
  for (int64_t i = 0; i < schema->n_children; ++i) {
    ArrowSchema* child = schema->children[i];
    if (child != nullptr) {
      if (child->release != nullptr) {
        child->release(child);
      }
      ::free(child);
    }
  }
  ::free(schema->children);
  if (schema->dictionary != nullptr) {
    if (schema->dictionary->release != nullptr) {
      schema->dictionary->release(schema->dictionary);
    }
    ::free(schema->dictionary);
  }
  schema->release = nullptr;
}

// Deep-copies 'source' into 'out'. Needed because importFromArrowAsOwner
// consumes the schema it is given, while the stream schema must describe
// every subsequent chunk of the result.
void deepCopySchema(const ArrowSchema& source, ArrowSchema& out) {
  out.format = copyString(source.format);
  out.name = copyString(source.name);
  out.metadata = copyMetadata(source.metadata);
  out.flags = source.flags;
  out.n_children = source.n_children;
  out.children = nullptr;
  if (source.n_children > 0) {
    out.children = static_cast<ArrowSchema**>(
        ::calloc(source.n_children, sizeof(ArrowSchema*)));
    for (int64_t i = 0; i < source.n_children; ++i) {
      out.children[i] =
          static_cast<ArrowSchema*>(::calloc(1, sizeof(ArrowSchema)));
      deepCopySchema(*source.children[i], *out.children[i]);
    }
  }
  out.dictionary = nullptr;
  if (source.dictionary != nullptr) {
    out.dictionary =
        static_cast<ArrowSchema*>(::calloc(1, sizeof(ArrowSchema)));
    deepCopySchema(*source.dictionary, *out.dictionary);
  }
  out.private_data = nullptr;
  out.release = releaseCopiedSchema;
}

std::string toLowerAscii(const std::string& value) {
  std::string result = value;
  for (auto& character : result) {
    if (character >= 'A' && character <= 'Z') {
      character += 'a' - 'A';
    }
  }
  return result;
}

// Quotes 'name' with 'quote', doubling embedded occurrences of the quote.
std::string quoteIdentifier(const std::string& name, const std::string& quote) {
  if (quote.empty()) {
    return name;
  }
  std::string body = name;
  size_t position = 0;
  while ((position = body.find(quote, position)) != std::string::npos) {
    body.insert(position, quote);
    position += 2 * quote.size();
  }
  return quote + body + quote;
}

} // namespace

class AdbcDatabaseHolder {
 public:
  AdbcDatabaseHolder(
      const AdbcConfig& config,
      AdbcDriverInitFunc driverInitFunc) {
    try {
      init(config, driverInitFunc);
    } catch (...) {
      releaseDatabase();
      throw;
    }
  }

  ~AdbcDatabaseHolder() {
    releaseDatabase();
  }

  AdbcDatabase* database() {
    return &database_;
  }

 private:
  void init(const AdbcConfig& config, AdbcDriverInitFunc driverInitFunc) {
    AdbcError error{};
    checkAdbcStatus(
        AdbcDatabaseNew(&database_, &error), error, "AdbcDatabaseNew", false);
    created_ = true;
    if (driverInitFunc != nullptr) {
      AdbcError setInitFuncError{};
      checkAdbcStatus(
          AdbcDriverManagerDatabaseSetInitFunc(
              &database_, driverInitFunc, &setInitFuncError),
          setInitFuncError,
          "AdbcDriverManagerDatabaseSetInitFunc",
          false);
    } else {
      auto driver = config.driver();
      VELOX_USER_CHECK(
          driver.has_value(),
          "ADBC connector requires the driver config: {}",
          AdbcConfig::kDriver);
      setOption("driver", driver.value());
      auto entrypoint = config.entrypoint();
      if (entrypoint.has_value()) {
        setOption("entrypoint", entrypoint.value());
      }
    }
    for (const auto& [key, value] : config.databaseOptions()) {
      setOption(key, value);
    }
    AdbcError initError{};
    checkAdbcStatus(
        AdbcDatabaseInit(&database_, &initError),
        initError,
        "AdbcDatabaseInit",
        true);
  }

  void setOption(const std::string& key, const std::string& value) {
    AdbcError error{};
    checkAdbcStatus(
        AdbcDatabaseSetOption(&database_, key.c_str(), value.c_str(), &error),
        error,
        "AdbcDatabaseSetOption",
        true);
  }

  void releaseDatabase() {
    if (created_) {
      AdbcDatabaseRelease(&database_, nullptr);
      created_ = false;
    }
  }

  AdbcDatabase database_{};
  bool created_{false};
};

folly::dynamic AdbcColumnHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = AdbcColumnHandle::getClassName();
  obj["columnName"] = name_;
  return obj;
}

// static
std::shared_ptr<AdbcColumnHandle> AdbcColumnHandle::create(
    const folly::dynamic& obj) {
  return std::make_shared<AdbcColumnHandle>(obj["columnName"].asString());
}

// static
void AdbcColumnHandle::registerSerDe() {
  registerDeserializer<AdbcColumnHandle>();
}

AdbcTableHandle::AdbcTableHandle(
    std::string connectorId,
    std::string tableName,
    std::string query)
    : ConnectorTableHandle(std::move(connectorId)),
      tableName_(std::move(tableName)),
      query_(std::move(query)) {
  VELOX_USER_CHECK(
      tableName_.empty() != query_.empty(),
      "Exactly one of table name and query must be set");
  name_ = tableName_.empty() ? "(adbc query)" : tableName_;
}

std::string AdbcTableHandle::toString() const {
  if (!tableName_.empty()) {
    return fmt::format("adbc table: {}", tableName_);
  }
  return fmt::format("adbc query: {}", query_);
}

folly::dynamic AdbcTableHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = AdbcTableHandle::getClassName();
  obj["connectorId"] = connectorId();
  obj["tableName"] = tableName_;
  obj["query"] = query_;
  return obj;
}

// static
ConnectorTableHandlePtr AdbcTableHandle::create(
    const folly::dynamic& obj,
    void* /*context*/) {
  return std::make_shared<AdbcTableHandle>(
      obj["connectorId"].asString(),
      obj["tableName"].asString(),
      obj["query"].asString());
}

// static
void AdbcTableHandle::registerSerDe() {
  registerDeserializerWithContext<AdbcTableHandle>();
}

struct AdbcDataSource::SplitState {
  AdbcConnection connection{};
  AdbcStatement statement{};
  ArrowArrayStream stream{};
  // Result schema of 'stream'.
  ArrowSchema schema{};
  // Index of each output column in the stream schema.
  std::vector<column_index_t> streamChildIndices;
  bool hasConnection{false};
  bool hasStatement{false};

  ~SplitState() {
    if (schema.release != nullptr) {
      schema.release(&schema);
    }
    if (stream.release != nullptr) {
      stream.release(&stream);
    }
    if (hasStatement) {
      AdbcStatementRelease(&statement, nullptr);
    }
    if (hasConnection) {
      AdbcConnectionRelease(&connection, nullptr);
    }
  }
};

AdbcDataSource::AdbcDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& columnHandles,
    const AdbcConfig& config,
    std::shared_ptr<AdbcDatabaseHolder> database,
    ConnectorQueryCtx* connectorQueryCtx)
    : outputType_(outputType),
      tableHandle_(
          std::dynamic_pointer_cast<const AdbcTableHandle>(tableHandle)),
      identifierQuote_(config.identifierQuote()),
      database_(std::move(database)),
      pool_(connectorQueryCtx->memoryPool()) {
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of AdbcTableHandle");
  remoteColumnNames_.reserve(outputType_->size());
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_USER_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);
    auto handle = std::dynamic_pointer_cast<const AdbcColumnHandle>(it->second);
    VELOX_CHECK_NOT_NULL(
        handle,
        "ColumnHandle must be an instance of AdbcColumnHandle: {}",
        outputName);
    remoteColumnNames_.push_back(handle->name());
  }
}

AdbcDataSource::~AdbcDataSource() = default;

std::string AdbcDataSource::buildSql() const {
  if (!tableHandle_->query().empty()) {
    return tableHandle_->query();
  }
  if (remoteColumnNames_.empty()) {
    // Zero projected columns, e.g. count(*). Select a constant to obtain the
    // row count.
    return fmt::format("SELECT 1 FROM {}", tableHandle_->tableName());
  }
  std::vector<std::string> quotedColumns;
  quotedColumns.reserve(remoteColumnNames_.size());
  for (const auto& columnName : remoteColumnNames_) {
    quotedColumns.push_back(quoteIdentifier(columnName, identifierQuote_));
  }
  return fmt::format(
      "SELECT {} FROM {}",
      folly::join(", ", quotedColumns),
      tableHandle_->tableName());
}

void AdbcDataSource::openStream() {
  splitState_ = std::make_unique<SplitState>();
  auto& state = *splitState_;

  AdbcError connectionNewError{};
  checkAdbcStatus(
      AdbcConnectionNew(&state.connection, &connectionNewError),
      connectionNewError,
      "AdbcConnectionNew",
      false);
  state.hasConnection = true;

  AdbcError connectionInitError{};
  checkAdbcStatus(
      AdbcConnectionInit(
          &state.connection, database_->database(), &connectionInitError),
      connectionInitError,
      "AdbcConnectionInit",
      true);

  AdbcError statementNewError{};
  checkAdbcStatus(
      AdbcStatementNew(&state.connection, &state.statement, &statementNewError),
      statementNewError,
      "AdbcStatementNew",
      false);
  state.hasStatement = true;

  const auto sql = buildSql();
  AdbcError setQueryError{};
  checkAdbcStatus(
      AdbcStatementSetSqlQuery(&state.statement, sql.c_str(), &setQueryError),
      setQueryError,
      "AdbcStatementSetSqlQuery",
      false);

  AdbcError executeError{};
  int64_t rowsAffected = -1;
  checkAdbcStatus(
      AdbcStatementExecuteQuery(
          &state.statement, &state.stream, &rowsAffected, &executeError),
      executeError,
      "AdbcStatementExecuteQuery",
      true);

  checkStreamCall(
      state.stream.get_schema(&state.stream, &state.schema),
      state.stream,
      "get_schema");

  // Map every output column onto the stream schema by case-insensitive name
  // and validate its type.
  auto streamType = importFromArrow(state.schema);
  VELOX_USER_CHECK(streamType->isRow(), "ADBC result schema must be a struct");
  const auto& streamRowType = streamType->asRow();
  std::unordered_map<std::string, column_index_t> streamColumnIndices;
  for (auto i = 0; i < streamRowType.size(); ++i) {
    streamColumnIndices.emplace(toLowerAscii(streamRowType.nameOf(i)), i);
  }
  state.streamChildIndices.reserve(outputType_->size());
  for (auto i = 0; i < outputType_->size(); ++i) {
    auto it = streamColumnIndices.find(toLowerAscii(remoteColumnNames_[i]));
    VELOX_USER_CHECK(
        it != streamColumnIndices.end(),
        "Column not found in ADBC result set: {}",
        remoteColumnNames_[i]);
    const auto& streamChildType = streamRowType.childAt(it->second);
    VELOX_USER_CHECK(
        outputType_->childAt(i)->equivalent(*streamChildType),
        "Column type mismatch in ADBC result set: {}: expected {}, got {}",
        remoteColumnNames_[i],
        outputType_->childAt(i)->toString(),
        streamChildType->toString());
    state.streamChildIndices.push_back(it->second);
  }
}

void AdbcDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  VELOX_CHECK(
      splitState_ == nullptr,
      "Previous split must be fully processed before adding a new one: {}",
      split->toString());
  auto adbcSplit = std::dynamic_pointer_cast<AdbcConnectorSplit>(split);
  VELOX_CHECK_NOT_NULL(
      adbcSplit, "Split must be an instance of AdbcConnectorSplit");
  openStream();
}

std::optional<RowVectorPtr> AdbcDataSource::next(
    uint64_t /*size*/,
    velox::ContinueFuture& /*future*/) {
  VELOX_CHECK_NOT_NULL(
      splitState_, "No split to process. Call addSplit first.");
  auto& state = *splitState_;
  for (;;) {
    ArrowArray chunk{};
    checkStreamCall(
        state.stream.get_next(&state.stream, &chunk), state.stream, "get_next");
    if (chunk.release == nullptr) {
      // End of stream: the split is fully processed.
      splitState_.reset();
      return nullptr;
    }
    if (chunk.length == 0) {
      chunk.release(&chunk);
      continue;
    }
    // importFromArrowAsOwner consumes the schema, so hand it a copy; each
    // imported buffer keeps the chunk alive through shared releasers.
    ArrowSchema schemaCopy;
    deepCopySchema(state.schema, schemaCopy);
    auto imported = importFromArrowAsOwner(schemaCopy, chunk, pool_);
    auto importedRow = std::dynamic_pointer_cast<RowVector>(imported);
    VELOX_CHECK_NOT_NULL(
        importedRow, "ADBC result chunk must import as a row vector");
    std::vector<VectorPtr> children;
    children.reserve(outputType_->size());
    for (auto streamIndex : state.streamChildIndices) {
      children.push_back(importedRow->childAt(streamIndex));
    }
    auto result = std::make_shared<RowVector>(
        pool_,
        outputType_,
        BufferPtr(nullptr),
        importedRow->size(),
        std::move(children));
    completedRows_ += result->size();
    completedBytes_ += result->estimateFlatSize();
    return result;
  }
}

AdbcConnector::AdbcConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* /*executor*/)
    : Connector(id, config),
      adbcConfig_(config),
      database_(std::make_shared<AdbcDatabaseHolder>(adbcConfig_, nullptr)) {}

AdbcConnector::AdbcConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    AdbcDriverInitFunc driverInitFunc)
    : Connector(id, config),
      adbcConfig_(config),
      database_(
          std::make_shared<AdbcDatabaseHolder>(adbcConfig_, driverInitFunc)) {}

AdbcConnector::~AdbcConnector() = default;

std::unique_ptr<DataSource> AdbcConnector::createDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  return std::make_unique<AdbcDataSource>(
      outputType,
      tableHandle,
      columnHandles,
      adbcConfig_,
      database_,
      connectorQueryCtx);
}

// static
void AdbcConnector::registerSerDe() {
  AdbcTableHandle::registerSerDe();
  AdbcColumnHandle::registerSerDe();
  AdbcConnectorSplit::registerSerDe();
}

} // namespace facebook::velox::connector::adbc
