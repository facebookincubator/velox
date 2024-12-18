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

#include "velox/experimental/cudf/connectors/parquet/ParquetReaderConfig.h"
#include "velox/common/config/Config.h"
#include "velox/core/QueryConfig.h"

#include <boost/algorithm/string.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <optional>
#include <string>

namespace facebook::velox::cudf_velox::connector::parquet {

namespace {

ParquetReaderConfig::InsertExistingPartitionsBehavior
stringToInsertExistingPartitionsBehavior(const std::string& strValue) {
  auto upperValue = boost::algorithm::to_upper_copy(strValue);
  if (upperValue == "ERROR") {
    return ParquetReaderConfig::InsertExistingPartitionsBehavior::kError;
  }
  if (upperValue == "OVERWRITE") {
    return ParquetReaderConfig::InsertExistingPartitionsBehavior::kOverwrite;
  }
  VELOX_UNSUPPORTED(
      "Unsupported insert existing partitions behavior: {}.", strValue);
}

} // namespace

// static
std::string ParquetReaderConfig::insertExistingPartitionsBehaviorString(
    InsertExistingPartitionsBehavior behavior) {
  switch (behavior) {
    case InsertExistingPartitionsBehavior::kError:
      return "ERROR";
    case InsertExistingPartitionsBehavior::kOverwrite:
      return "OVERWRITE";
    default:
      return fmt::format("UNKNOWN BEHAVIOR {}", static_cast<int>(behavior));
  }
}

ParquetReaderConfig::InsertExistingPartitionsBehavior
ParquetReaderConfig::insertExistingPartitionsBehavior(
    const config::ConfigBase* session) const {
  return stringToInsertExistingPartitionsBehavior(session->get<std::string>(
      kInsertExistingPartitionsBehaviorSession,
      config_->get<std::string>(kInsertExistingPartitionsBehavior, "ERROR")));
}

int64_t ParquetReaderConfig::skipRows() const {
  return config_->get<int64_t>(kSkipRows, 0);
}

std::optional<cudf::size_type> ParquetReaderConfig::numRows() const {
  auto numRows = config_->get<cudf::size_type>(kNumRows);
  if (numRows.has_value()) {
    return numRows.value();
  }
  return std::nullopt;
}

std::size_t ParquetReaderConfig::maxChunkReadLimit() const {
  // chunk read limit = 0 means no limit
  return config_->get<std::size_t>(kMaxChunkReadLimit, 0);
}

std::size_t ParquetReaderConfig::maxChunkReadLimitSession(
    const config::ConfigBase* session) const {
  // pass read limit = 0 means no limit
  return session->get<std::size_t>(
      kMaxChunkReadLimitSession,
      config_->get<std::size_t>(kMaxChunkReadLimit, 0));
}

std::size_t ParquetReaderConfig::maxPassReadLimit() const {
  // pass read limit = 0 means no limit
  return config_->get<std::size_t>(kMaxPassReadLimit, 0);
}

std::size_t ParquetReaderConfig::maxPassReadLimitSession(
    const config::ConfigBase* session) const {
  // pass read limit = 0 means no limit
  return session->get<std::size_t>(
      kMaxPassReadLimitSession,
      config_->get<std::size_t>(kMaxPassReadLimit, 0));
}

bool ParquetReaderConfig::isConvertStringsToCategories() const {
  return config_->get<bool>(kConvertStringsToCategories, false);
}

bool ParquetReaderConfig::isConvertStringsToCategoriesSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kConvertStringsToCategoriesSession,
      config_->get<bool>(kConvertStringsToCategories, false));
}

bool ParquetReaderConfig::isUsePandasMetadata() const {
  return config_->get<bool>(kUsePandasMetadata, true);
}

bool ParquetReaderConfig::isUsePandasMetadataSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kUsePandasMetadataSession, config_->get<bool>(kUsePandasMetadata, true));
}

bool ParquetReaderConfig::isUseArrowSchema() const {
  return config_->get<bool>(kUseArrowSchema, true);
}

bool ParquetReaderConfig::isUseArrowSchemaSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kUseArrowSchemaSession, config_->get<bool>(kUseArrowSchema, true));
}

bool ParquetReaderConfig::isAllowMismatchedParquetSchemas() const {
  return config_->get<bool>(kAllowMismatchedParquetSchemas, false);
}

bool ParquetReaderConfig::isAllowMismatchedParquetSchemasSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kAllowMismatchedParquetSchemasSession,
      config_->get<bool>(kAllowMismatchedParquetSchemas, false));
}

cudf::data_type ParquetReaderConfig::timestampType() const {
  const auto unit = config_->get<cudf::type_id>(
      kTimestampType, cudf::type_id::TIMESTAMP_MILLISECONDS /*milli*/);
  VELOX_CHECK(
      unit == cudf::type_id::TIMESTAMP_DAYS /*days*/ ||
          unit == cudf::type_id::TIMESTAMP_SECONDS /*seconds*/ ||
          unit == cudf::type_id::TIMESTAMP_MILLISECONDS /*milli*/ ||
          unit == cudf::type_id::TIMESTAMP_MICROSECONDS /*micro*/ ||
          unit == cudf::type_id::TIMESTAMP_NANOSECONDS /*nano*/,
      "Invalid timestamp unit.");
  return cudf::data_type(cudf::type_id{unit});
}

cudf::data_type ParquetReaderConfig::timestampTypeSession(
    const config::ConfigBase* session) const {
  const auto unit = session->get<cudf::type_id>(
      kTimestampTypeSession,
      config_->get<cudf::type_id>(
          kTimestampType, cudf::type_id::TIMESTAMP_MILLISECONDS /*milli*/));
  VELOX_CHECK(
      unit == cudf::type_id::TIMESTAMP_DAYS /*days*/ ||
          unit == cudf::type_id::TIMESTAMP_SECONDS /*seconds*/ ||
          unit == cudf::type_id::TIMESTAMP_MILLISECONDS /*milli*/ ||
          unit == cudf::type_id::TIMESTAMP_MICROSECONDS /*micro*/ ||
          unit == cudf::type_id::TIMESTAMP_NANOSECONDS /*nano*/,
      "Invalid timestamp unit.");
  return cudf::data_type(cudf::type_id{unit});
}

} // namespace facebook::velox::cudf_velox::connector::parquet
