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

#include "velox/exec/QuerySplitTracer.h"
#include "QueryTraceUtil.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/exec/QueryTraceTraits.h"

using namespace facebook::velox::connector::hive;

namespace facebook::velox::exec::trace {
/// Used to record and load the input splits from a tracing 'TableScan'
/// operator, and for getting the traced splits when relaying 'TableScan'.
///
/// Currently, it only works with 'HiveConnectorSplit'. In the future, it will
/// be extended to handle more types of splits, such as
/// 'IcebergHiveConnectorSplit'.
QuerySplitTracer::QuerySplitTracer(std::string traceDir)
    : traceDir_(std::move(traceDir)),
      fs_(filesystems::getFileSystem(traceDir_, nullptr)) {
  VELOX_CHECK_NOT_NULL(fs_);
}

void QuerySplitTracer::write(const exec::Split& split) {
  VELOX_CHECK(!split.hasGroup());
  VELOX_CHECK(split.hasConnectorSplit());
  const auto splitInfoFile = fs_->openFileForWrite(fmt::format(
      "{}/trace{}.{}",
      traceDir_,
      QueryTraceTraits::kSplitInfoFileSuffix,
      fileId_));
  const auto splitObj = split.connectorSplit->serialize();
  const auto splitJson = folly::toJson(splitObj);
  splitInfoFile->append(splitJson);
  splitInfoFile->flush();
  splitInfoFile->close();
  ++fileId_;
}

// static
int32_t QuerySplitTracer::extractFileIndex(const std::string& str) {
  std::string capturedStr;
  if (!RE2::FullMatch(str, kFileRegExp, &capturedStr)) {
    return -1;
  }
  return std::stoul(capturedStr);
}

std::vector<exec::Split> QuerySplitTracer::read() const {
  std::vector<exec::Split> splits;
  std::map<int32_t, std::string> fileMap;
  for (const auto& filePath : fs_->list(traceDir_)) {
    const auto index = extractFileIndex(filePath);
    if (index == -1) {
      continue;
    }
    fileMap[index] = filePath;
  }

  for (const auto& [_, filePath] : fileMap) {
    const auto splitInfoFilePath = fs_->openFileForRead(filePath);
    folly::dynamic splitInfoObj = getMetadata(filePath, fs_);
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
} // namespace facebook::velox::exec::trace
