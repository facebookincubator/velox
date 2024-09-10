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

#include "velox/exec/trace/QueryDataWriter.h"
#include "velox/common/base/SpillStats.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/trace/QueryTraceTraits.h"
#include "velox/serializers/PrestoSerializer.h"

namespace facebook::velox::exec::trace {

QueryDataWriter::QueryDataWriter(
    const std::string& path,
    memory::MemoryPool* pool)
    : dirPath_(path),
      fs_(filesystems::getFileSystem(dirPath_, nullptr)),
      pool_(pool) {
  dataFile_ = fs_->openFileForWrite(
      fmt::format("{}/{}", dirPath_, QueryTraceTraits::kDataFileName));
  VELOX_CHECK_NOT_NULL(dataFile_);
}

void QueryDataWriter::write(const RowVectorPtr& rows) {
  if (batch_ == nullptr) {
    batch_ = std::make_unique<VectorStreamGroup>(pool_);
    batch_->createStreamTree(
        std::static_pointer_cast<const RowType>(rows->type()),
        1'000,
        &options_);
  }
  batch_->append(rows);
  dataType_ = rows->type();

  // Serialize and write out each batch.
  IOBufOutputStream out(
      *pool_, nullptr, std::max<int64_t>(64 * 1024, batch_->size()));
  batch_->flush(&out);
  batch_->clear();
  auto iobuf = out.getIOBuf();
  dataFile_->append(std::move(iobuf));
}

void QueryDataWriter::finish() {
  VELOX_CHECK_NOT_NULL(
      dataFile_, "The query data writer has already been finished");
  dataFile_->close();
  dataFile_.reset();
  batch_.reset();
  writeSummary();
}

void QueryDataWriter::writeSummary() const {
  const auto summaryFilePath =
      fmt::format("{}/{}", dirPath_, QueryTraceTraits::kDataSummaryFileName);
  const auto file = fs_->openFileForWrite(summaryFilePath);
  folly::dynamic obj = folly::dynamic::object;
  obj[QueryTraceTraits::kDataTypeKey] = dataType_->serialize();
  file->append(folly::toJson(obj));
  file->close();
}

} // namespace facebook::velox::exec::trace
