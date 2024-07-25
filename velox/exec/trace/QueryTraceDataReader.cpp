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

#include "velox/exec/trace/QueryTraceDataReader.h"

#include "velox/common/file/File.h"
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec {

QueryTraceDataReader::QueryTraceDataReader(std::string path)
    : path_(std::move(path)) {
  Type::registerSerDe();
  if (!isRegisteredVectorSerde()) {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }

  const auto fs = filesystems::getFileSystem(path_, nullptr);
  type_ = traceDataType(fs);
  stream_ = getFileInputStream(fs);
}

bool QueryTraceDataReader::read(RowVectorPtr& batch) const {
  if (stream_->atEnd()) {
    return false;
  }

  VectorStreamGroup::read(stream_.get(), pool_, type_, &batch, &readOptions_);
  return true;
}

RowTypePtr QueryTraceDataReader::traceDataType(
    const std::shared_ptr<filesystems::FileSystem>& fs) const {
  const auto file = fs->openFileForRead(fmt::format("{}/summary.json", path_));
  const auto summary = file->pread(0, file->size());
  VELOX_USER_CHECK(!summary.empty());
  folly::dynamic obj = folly::parseJson(summary);
  return ISerializable::deserialize<RowType>(obj["rowType"]);
}

std::unique_ptr<common::FileInputStream>
QueryTraceDataReader::getFileInputStream(
    const std::shared_ptr<filesystems::FileSystem>& fs) const {
  auto file = fs->openFileForRead(fmt::format("{}/trace.data", path_));
  // TODO: Add this item to the trace config.
  constexpr uint64_t kMaxReadBufferSize =
      (1 << 20) - AlignedBuffer::kPaddedSize; // 1MB - padding.
  return std::make_unique<common::FileInputStream>(
      std::move(file), kMaxReadBufferSize, pool_);
}

} // namespace facebook::velox::exec
