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

#include "velox/exec/trace/QueryTraceDataWriter.h"
#include "velox/common/base/SpillStats.h"
#include "velox/common/compression/Compression.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/TreeOfLosers.h"
#include "velox/exec/UnorderedStreamReader.h"
#include "velox/serializers/PrestoSerializer.h"

namespace facebook::velox::exec {

// Static
std::unique_ptr<TraceWriteFile> TraceWriteFile::create(
    const std::string& path) {
  return std::make_unique<TraceWriteFile>(path);
}

uint64_t TraceWriteFile::write(std::unique_ptr<folly::IOBuf> iobuf) const {
  const auto writtenBytes = iobuf->computeChainCapacity();
  file_->append(std::move(iobuf));
  return writtenBytes;
}

void TraceWriteFile::close() {
  VELOX_CHECK_NOT_NULL(file_);
  file_->close();
  file_.reset();
}

TraceWriteFile::TraceWriteFile(std::string path) : path_(std::move(path)) {
  const auto fs = filesystems::getFileSystem(path_, nullptr);
  file_ = fs->openFileForWrite(path_);
}

namespace {
// Query tracer currently uses the default PrestoSerializer which by default
// serializes timestamp with millisecond precision to maintain compatibility
// with presto. Since velox's native timestamp implementation supports
// nanosecond precision, we use this serde option to ensure the serializer
// preserves precision.
constexpr bool kDefaultUseLosslessTimestamp = true;
const std::string kSummaryFileName = "summary.json";
const std::string kRowTypeKey = "rowType";
} // namespace

void TraceWriter::write(const RowVectorPtr& rows) {
  if (batch_ == nullptr) {
    const serializer::presto::PrestoVectorSerde::PrestoOptions options = {
        kDefaultUseLosslessTimestamp,
        common::CompressionKind::CompressionKind_ZSTD,
        true /*nullsFirst*/};
    batch_ = std::make_unique<VectorStreamGroup>(pool_);
    batch_->createStreamTree(
        std::static_pointer_cast<const RowType>(rows->type()), 1'000, &options);
  }
  batch_->append(rows);
  type_ = rows->type();

  if (batch_->size() < writeBufferSize_) {
    return;
  }

  flush();
}

uint64_t TraceWriter::flush() {
  if (batch_ == nullptr) {
    return 0;
  }

  if (file_ == nullptr) {
    file_ = TraceWriteFile::create(fmt::format("{}/trace.data", path_));
  }

  IOBufOutputStream out(
      *pool_, nullptr, std::max<int64_t>(64 * 1024, batch_->size()));
  batch_->flush(&out);
  batch_.reset();

  auto iobuf = out.getIOBuf();
  return file_->write(std::move(iobuf));
}

void TraceWriter::finish() {
  flush();
  writeSummary();
  file_->close();
  file_.reset();
}

void TraceWriter::writeSummary() const {
  const auto filePath = fmt::format("{}/{}", path_, kSummaryFileName);
  const auto fs = filesystems::getFileSystem(filePath, nullptr);
  const auto file = fs->openFileForWrite(filePath);
  folly::dynamic obj = folly::dynamic::object;
  obj[kRowTypeKey] = type_->serialize();
  file->append(folly::toJson(obj));
  file->close();
}

} // namespace facebook::velox::exec
