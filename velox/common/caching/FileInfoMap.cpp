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

#include "velox/common/caching/FileInfoMap.h"

namespace facebook::velox::cache {

std::shared_ptr<FileInfoMap> FileInfoMap::instance_ = nullptr;

// static
void FileInfoMap::create() {
  if (!instance_) {
    instance_ = std::make_shared<FileInfoMap>();
  }
}

// static
void FileInfoMap::release() {
  if (instance_ != nullptr) {
    instance_ = nullptr;
  }
}

bool FileInfoMap::addOpenFileInfo(uint64_t fileNum, int64_t openTimeSec) {
  return fileMap_.emplace(fileNum, RawFileInfo{openTimeSec, false}).second;
}

const RawFileInfo* FileInfoMap::find(uint64_t fileNum) const {
  const auto it = fileMap_.find(fileNum);
  if (it == fileMap_.cend()) {
    return nullptr;
  }
  return &it->second;
}

RawFileInfo* FileInfoMap::find(uint64_t fileNum) {
  auto it = fileMap_.find(fileNum);
  if (it == fileMap_.end()) {
    return nullptr;
  }
  return &it->second;
}

void FileInfoMap::resetInCache() {
  for (auto& [_, fileInfo] : fileMap_) {
    fileInfo.inCache = false;
  }
}

void FileInfoMap::deleteNotInCache() {
  for (auto it = fileMap_.begin(); it != fileMap_.end();) {
    if (!it->second.inCache) {
      it = fileMap_.erase(it);
    } else {
      it++;
    }
  }
}

void FileInfoMap::forEach(
    std::function<void(uint64_t, RawFileInfo&)> function) {
  for (auto& [fileNum, rawFileInfo] : fileMap_) {
    function(fileNum, rawFileInfo);
  }
}

folly::SharedMutex& FileInfoMap::mutex() {
  return fileMapMutex_;
}

} // namespace facebook::velox::cache
