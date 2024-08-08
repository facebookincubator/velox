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

#include <utility>
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec {
void TraceInputStream::next(bool /*throwIfPastEnd*/) {
  const auto readBytes =
      static_cast<int32_t>(std::min(size_ - offset_, buffer_->capacity()));
  VELOX_CHECK_LT(0, readBytes, "Reading past end of trace data file.");
  setRange({buffer_->asMutable<uint8_t>(), readBytes, 0});
  file_->pread(offset_, readBytes, buffer_->asMutable<char>());
  offset_ += readBytes;
}

QueryTraceDataReader::QueryTraceDataReader(std::string path)
    : path_(std::move(path)) {
  Type::registerSerDe();
  if (!isRegisteredVectorSerde()) {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }
  loadSummary();
  constexpr uint64_t kMaxReadBufferSize =
      (1 << 20) - AlignedBuffer::kPaddedSize; // 1MB - padding.
  const auto fs = filesystems::getFileSystem(path_, nullptr);
  auto file = fs->openFileForRead(fmt::format("{}/trace.data", path_));
  auto buffer = AlignedBuffer::allocate<char>(
      std::min<uint64_t>(file->size(), kMaxReadBufferSize), pool_);
  stream_ =
      std::make_unique<TraceInputStream>(std::move(file), std::move(buffer));
}

bool QueryTraceDataReader::read(RowVectorPtr& batch) const {
  if (stream_->atEnd()) {
    return false;
  }

  VectorStreamGroup::read(stream_.get(), pool_, type_, &batch, &readOptions_);
  return true;
}

void QueryTraceDataReader::loadSummary() {
  const auto fs = filesystems::getFileSystem(path_, nullptr);
  const auto file = fs->openFileForRead(fmt::format("{}/summary.json", path_));
  const auto summary = file->pread(0, file->size());
  VELOX_USER_CHECK(!summary.empty());
  folly::dynamic obj = folly::parseJson(summary);
  type_ = ISerializable::deserialize<RowType>(obj["rowType"]);
}
} // namespace facebook::velox::exec
