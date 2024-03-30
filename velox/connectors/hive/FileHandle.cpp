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

#include "velox/connectors/hive/FileHandle.h"
#include "velox/common/base/Counters.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/time/Timer.h"

#include <atomic>

namespace facebook::velox {

namespace {
// The group tracking is at the level of the directory, i.e. Hive partition.
std::string groupName(const std::string& filename) {
  const char* slash = strrchr(filename.c_str(), '/');
  return slash ? std::string(filename.data(), slash - filename.data())
               : filename;
}
} // namespace

std::shared_ptr<FileHandle> FileHandleGenerator::operator()(
    std::tuple<const std::string, const std::string, const std::string> params) {
  // We have seen cases where drivers are stuck when creating file handles.
  // Adding a trace here to spot this more easily in future.
  process::TraceContext trace("FileHandleGenerator::operator()");
  uint64_t elapsedTimeUs{0};
  std::shared_ptr<FileHandle> fileHandle;
  {
    MicrosecondTimer timer(&elapsedTimeUs);
    fileHandle = std::make_shared<FileHandle>();
    filesystems::FileOptions options;
    options.values["fileSize"] = std::get<1>(params);
    // Add length and modification time here to create a unique handle?
    fileHandle->file = filesystems::getFileSystem(std::get<0>(params), properties_)
                           ->openFileForRead(std::get<0>(params), options);
    fileHandle->uuid = StringIdLease(fileIds(), std::get<0>(params));
    fileHandle->groupId = StringIdLease(fileIds(), groupName(std::get<0>(params)));
    VLOG(1) << "Generating file handle for: " << std::get<0>(params)
            << " uuid: " << fileHandle->uuid.id();
  }
  RECORD_HISTOGRAM_METRIC_VALUE(
      kMetricHiveFileHandleGenerateLatencyMs, elapsedTimeUs / 1000);
  // TODO: build the hash map/etc per file type -- presumably after reading
  // the appropriate magic number from the file, or perhaps we include the file
  // type in the file handle key.
  return fileHandle;
}

} // namespace facebook::velox
