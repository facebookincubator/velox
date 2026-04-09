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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfEqualityDeleteFileReader.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

CudfEqualityDeleteFileReader::CudfEqualityDeleteFileReader(
    const velox_iceberg::IcebergDeleteFile& deleteFile,
    const std::vector<std::string>& equalityColumnNames,
    const std::vector<TypePtr>& equalityColumnTypes,
    const std::string& baseFilePath,
    FileHandleFactory* fileHandleFactory,
    const ConnectorQueryCtx* connectorQueryCtx,
    folly::Executor* executor,
    const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    dwio::common::RuntimeStatistics& runtimeStats,
    const std::string& connectorId)
    : upstreamReader_(std::make_unique<velox_iceberg::EqualityDeleteFileReader>(
          deleteFile,
          equalityColumnNames,
          equalityColumnTypes,
          baseFilePath,
          fileHandleFactory,
          connectorQueryCtx,
          executor,
          hiveConfig,
          ioStatistics,
          ioStats,
          runtimeStats,
          connectorId)),
      equalityColumnNames_(equalityColumnNames),
      equalityColumnTypes_(equalityColumnTypes) {}

void CudfEqualityDeleteFileReader::applyDeletes(
    cudf::table_view /*table*/,
    std::shared_ptr<rmm::device_buffer> /*rowMask*/,
    rmm::cuda_stream_view /*stream*/) {
  // TODO: Implement GPU-accelerated equality delete application using
  // cudf::ast::tree and cudf::compute_column. See Milestone 2.
}

void CudfEqualityDeleteFileReader::applyDeletes(
    const RowVectorPtr& output,
    BufferPtr deleteBitmap) {
  upstreamReader_->applyDeletes(output, deleteBitmap);
}

size_t CudfEqualityDeleteFileReader::numDeleteKeys() const {
  return upstreamReader_->numDeleteKeys();
}

bool CudfEqualityDeleteFileReader::empty() const {
  return upstreamReader_->empty();
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
