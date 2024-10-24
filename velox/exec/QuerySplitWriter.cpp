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

#include "velox/exec/QuerySplitWriter.h"

#include <folly/hash/Checksum.h>
#include <folly/io/Cursor.h>

#include "QueryTraceUtil.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/exec/QueryTraceTraits.h"

using namespace facebook::velox::connector::hive;

namespace facebook::velox::exec::trace {
QuerySplitWriter::QuerySplitWriter(std::string traceDir)
    : traceDir_(std::move(traceDir)),
      fs_(filesystems::getFileSystem(traceDir_, nullptr)) {
  VELOX_CHECK_NOT_NULL(fs_);
  splitFile_ = fs_->openFileForWrite(
      fmt::format("{}/{}", traceDir_, QueryTraceTraits::kSplitFileName));
  VELOX_CHECK_NOT_NULL(splitFile_);
}

void QuerySplitWriter::write(const exec::Split& split) const {
  VELOX_CHECK(!split.hasGroup(), "Do not support grouped execution");
  VELOX_CHECK(split.hasConnectorSplit());
  const auto splitObj = split.connectorSplit->serialize();
  const auto splitJson = folly::toJson(splitObj);
  auto ioBuf = serialize(splitJson);
  splitFile_->append(std::move(ioBuf));
}

void QuerySplitWriter::finish() {
  if (finished_) {
    return;
  }

  VELOX_CHECK_NOT_NULL(
      splitFile_, "The query data writer has already been finished");
  splitFile_->close();
  splitFile_.reset();
  finished_ = true;
}

// static
std::unique_ptr<folly::IOBuf> QuerySplitWriter::serialize(
    const std::string& split) {
  const uint32_t length = split.length();
  const uint32_t crc32 = folly::crc32(
      reinterpret_cast<const uint8_t*>(split.data()), split.size());
  auto ioBuf =
      folly::IOBuf::create(sizeof(length) + split.size() + sizeof(crc32));
  folly::io::Appender appender(ioBuf.get(), 0);
  appender.writeLE(length);
  appender.push(reinterpret_cast<const uint8_t*>(split.data()), length);
  appender.writeLE(crc32);
  return ioBuf;
}
} // namespace facebook::velox::exec::trace
