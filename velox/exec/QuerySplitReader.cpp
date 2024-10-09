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

#include "velox/exec/QuerySplitReader.h"

#include <folly/hash/Checksum.h>

#include "velox/common/file/FileInputStream.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/exec/QueryTraceTraits.h"
#include "velox/exec/QueryTraceUtil.h"

using namespace facebook::velox::connector::hive;

namespace facebook::velox::exec::trace {

QuerySplitReader::QuerySplitReader(
    std::string traceDir,
    memory::MemoryPool* pool)
    : traceDir_(std::move(traceDir)),
      fs_(filesystems::getFileSystem(traceDir_, nullptr)),
      pool_(pool),
      splitInfoStream_(getSplitInputStream()) {
  VELOX_CHECK_NOT_NULL(fs_);
  VELOX_CHECK_NOT_NULL(splitInfoStream_);
}

std::vector<exec::Split> QuerySplitReader::read() const {
  const auto splitStrings = getSplitInfos(splitInfoStream_.get());
  std::vector<exec::Split> splits;
  for (const auto& splitString : splitStrings) {
    folly::dynamic splitInfoObj = folly::parseJson(splitString);
    const auto split =
        ISerializable::deserialize<HiveConnectorSplit>(splitInfoObj);
    splits.emplace_back(
        std::make_shared<HiveConnectorSplit>(
            split->connectorId,
            split->filePath,
            split->fileFormat,
            split->start,
            split->length,
            split->partitionKeys,
            split->tableBucketNumber,
            split->customSplitInfo,
            split->extraFileInfo,
            split->serdeParameters,
            split->splitWeight,
            split->infoColumns,
            split->properties),
        -1);
  }
  return splits;
}

std::unique_ptr<common::FileInputStream> QuerySplitReader::getSplitInputStream()
    const {
  auto splitInfoFile = fs_->openFileForRead(
      fmt::format("{}/{}", traceDir_, QueryTraceTraits::kSplitInfoFileName));
  // TODO: Make the buffer size configurable.
  return std::make_unique<common::FileInputStream>(
      std::move(splitInfoFile), 1 << 20, pool_);
}

// static
std::vector<std::string> QuerySplitReader::getSplitInfos(
    common::FileInputStream* stream) {
  std::vector<std::string> splits;
  try {
    while (!stream->atEnd()) {
      const auto length = stream->read<uint32_t>();
      std::string splitInfoString(length, '\0');
      stream->readBytes(
          reinterpret_cast<uint8_t*>(splitInfoString.data()), length);
      const auto crc32 = stream->read<uint32_t>();
      const auto actualCrc32 = folly::crc32(
          reinterpret_cast<const uint8_t*>(splitInfoString.data()),
          splitInfoString.size());
      if (crc32 != actualCrc32) {
        LOG(ERROR) << "Fails to verify the checksum " << crc32
                   << " does not equal to the actual checksum " << actualCrc32;
        break;
      }
      splits.push_back(std::move(splitInfoString));
    }
  } catch (const VeloxException& e) {
    LOG(ERROR) << "Fails to deserialize split string from the stream for "
               << e.message();
  }
  return splits;
}
} // namespace facebook::velox::exec::trace
