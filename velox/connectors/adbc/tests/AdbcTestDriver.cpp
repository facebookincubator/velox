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
#include "velox/connectors/adbc/tests/AdbcTestDriver.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::connector::adbc::test {

namespace {

void releaseDriverError(AdbcError* error) {
  std::free(error->message);
  error->message = nullptr;
  error->release = nullptr;
}

void setDriverError(AdbcError* error, const std::string& message) {
  if (error == nullptr) {
    return;
  }
  error->message = ::strdup(message.c_str());
  error->vendor_code = 0;
  std::memset(error->sqlstate, 0, sizeof(error->sqlstate));
  error->release = releaseDriverError;
}

AdbcStatusCode driverRelease(AdbcDriver* driver, AdbcError* /*error*/) {
  driver->private_data = nullptr;
  return ADBC_STATUS_OK;
}

AdbcStatusCode databaseNew(AdbcDatabase* database, AdbcError* /*error*/) {
  database->private_data = nullptr;
  return ADBC_STATUS_OK;
}

AdbcStatusCode databaseSetOption(
    AdbcDatabase* /*database*/,
    const char* key,
    const char* value,
    AdbcError* /*error*/) {
  AdbcTestDriverState::instance().recordDatabaseOption(key, value);
  return ADBC_STATUS_OK;
}

AdbcStatusCode databaseInit(AdbcDatabase* /*database*/, AdbcError* /*error*/) {
  return ADBC_STATUS_OK;
}

AdbcStatusCode databaseRelease(AdbcDatabase* database, AdbcError* /*error*/) {
  database->private_data = nullptr;
  return ADBC_STATUS_OK;
}

AdbcStatusCode connectionNew(AdbcConnection* connection, AdbcError* /*error*/) {
  connection->private_data = nullptr;
  return ADBC_STATUS_OK;
}

AdbcStatusCode connectionInit(
    AdbcConnection* /*connection*/,
    AdbcDatabase* /*database*/,
    AdbcError* /*error*/) {
  return ADBC_STATUS_OK;
}

AdbcStatusCode connectionRelease(
    AdbcConnection* connection,
    AdbcError* /*error*/) {
  connection->private_data = nullptr;
  return ADBC_STATUS_OK;
}

// SQL text of one statement.
struct StatementState {
  std::string sql;
};

AdbcStatusCode statementNew(
    AdbcConnection* /*connection*/,
    AdbcStatement* statement,
    AdbcError* /*error*/) {
  statement->private_data = new StatementState();
  return ADBC_STATUS_OK;
}

AdbcStatusCode statementSetSqlQuery(
    AdbcStatement* statement,
    const char* query,
    AdbcError* /*error*/) {
  static_cast<StatementState*>(statement->private_data)->sql = query;
  return ADBC_STATUS_OK;
}

AdbcStatusCode statementRelease(
    AdbcStatement* statement,
    AdbcError* /*error*/) {
  delete static_cast<StatementState*>(statement->private_data);
  statement->private_data = nullptr;
  return ADBC_STATUS_OK;
}

// Result batches served by one ArrowArrayStream.
struct StreamState {
  RowTypePtr rowType;
  std::vector<RowVectorPtr> batches;
  size_t nextBatch{0};
  std::string lastError;
};

int streamGetSchema(ArrowArrayStream* stream, ArrowSchema* out) {
  auto* state = static_cast<StreamState*>(stream->private_data);
  try {
    auto emptyVector = BaseVector::create(
        state->rowType, 0, AdbcTestDriverState::instance().pool());
    exportToArrow(emptyVector, *out);
    return 0;
  } catch (const std::exception& e) {
    state->lastError = e.what();
    return EINVAL;
  }
}

int streamGetNext(ArrowArrayStream* stream, ArrowArray* out) {
  auto* state = static_cast<StreamState*>(stream->private_data);
  try {
    if (state->nextBatch >= state->batches.size()) {
      // Signal end of stream.
      std::memset(out, 0, sizeof(ArrowArray));
      return 0;
    }
    exportToArrow(
        state->batches[state->nextBatch],
        *out,
        AdbcTestDriverState::instance().pool());
    ++state->nextBatch;
    return 0;
  } catch (const std::exception& e) {
    state->lastError = e.what();
    return EINVAL;
  }
}

const char* streamGetLastError(ArrowArrayStream* stream) {
  auto* state = static_cast<StreamState*>(stream->private_data);
  return state->lastError.empty() ? nullptr : state->lastError.c_str();
}

void streamRelease(ArrowArrayStream* stream) {
  delete static_cast<StreamState*>(stream->private_data);
  stream->private_data = nullptr;
  stream->release = nullptr;
}

AdbcStatusCode statementExecuteQuery(
    AdbcStatement* statement,
    ArrowArrayStream* out,
    int64_t* rowsAffected,
    AdbcError* error) {
  auto& driverState = AdbcTestDriverState::instance();
  driverState.recordSql(
      static_cast<StatementState*>(statement->private_data)->sql);
  auto executeError = driverState.executeError();
  if (!executeError.empty()) {
    setDriverError(error, executeError);
    return ADBC_STATUS_INTERNAL;
  }
  auto [rowType, batches] = driverState.result();
  auto* streamState = new StreamState();
  streamState->rowType = std::move(rowType);
  streamState->batches = std::move(batches);
  out->private_data = streamState;
  out->get_schema = streamGetSchema;
  out->get_next = streamGetNext;
  out->get_last_error = streamGetLastError;
  out->release = streamRelease;
  if (rowsAffected != nullptr) {
    *rowsAffected = -1;
  }
  return ADBC_STATUS_OK;
}

} // namespace

extern "C" AdbcStatusCode
adbcTestDriverInit(int version, void* rawDriver, struct AdbcError* error) {
  if (version != ADBC_VERSION_1_0_0) {
    setDriverError(error, "Unsupported ADBC version requested by the manager");
    return ADBC_STATUS_NOT_IMPLEMENTED;
  }
  auto* driver = static_cast<AdbcDriver*>(rawDriver);
  std::memset(driver, 0, ADBC_DRIVER_1_0_0_SIZE);
  driver->release = driverRelease;
  driver->DatabaseNew = databaseNew;
  driver->DatabaseSetOption = databaseSetOption;
  driver->DatabaseInit = databaseInit;
  driver->DatabaseRelease = databaseRelease;
  driver->ConnectionNew = connectionNew;
  driver->ConnectionInit = connectionInit;
  driver->ConnectionRelease = connectionRelease;
  driver->StatementNew = statementNew;
  driver->StatementSetSqlQuery = statementSetSqlQuery;
  driver->StatementExecuteQuery = statementExecuteQuery;
  driver->StatementRelease = statementRelease;
  return ADBC_STATUS_OK;
}

AdbcTestDriverState& AdbcTestDriverState::instance() {
  static AdbcTestDriverState state;
  return state;
}

void AdbcTestDriverState::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  rowType_.reset();
  batches_.clear();
  lastSql_.clear();
  executeError_.clear();
  databaseOptions_.clear();
  pool_ = nullptr;
}

void AdbcTestDriverState::setResult(
    RowTypePtr rowType,
    std::vector<RowVectorPtr> batches,
    memory::MemoryPool* pool) {
  std::lock_guard<std::mutex> lock(mutex_);
  rowType_ = std::move(rowType);
  batches_ = std::move(batches);
  pool_ = pool;
}

void AdbcTestDriverState::setExecuteError(std::string message) {
  std::lock_guard<std::mutex> lock(mutex_);
  executeError_ = std::move(message);
}

std::string AdbcTestDriverState::lastSql() {
  std::lock_guard<std::mutex> lock(mutex_);
  return lastSql_;
}

std::unordered_map<std::string, std::string>
AdbcTestDriverState::databaseOptions() {
  std::lock_guard<std::mutex> lock(mutex_);
  return databaseOptions_;
}

void AdbcTestDriverState::recordSql(std::string sql) {
  std::lock_guard<std::mutex> lock(mutex_);
  lastSql_ = std::move(sql);
}

void AdbcTestDriverState::recordDatabaseOption(
    std::string key,
    std::string value) {
  std::lock_guard<std::mutex> lock(mutex_);
  databaseOptions_[std::move(key)] = std::move(value);
}

std::string AdbcTestDriverState::executeError() {
  std::lock_guard<std::mutex> lock(mutex_);
  return executeError_;
}

std::pair<RowTypePtr, std::vector<RowVectorPtr>> AdbcTestDriverState::result() {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK_NOT_NULL(
      rowType_, "AdbcTestDriverState has no configured result");
  return {rowType_, batches_};
}

memory::MemoryPool* AdbcTestDriverState::pool() {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK_NOT_NULL(pool_, "AdbcTestDriverState has no configured pool");
  return pool_;
}

} // namespace facebook::velox::connector::adbc::test
