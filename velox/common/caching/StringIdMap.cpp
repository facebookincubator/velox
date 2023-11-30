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

#include "velox/common/caching/StringIdMap.h"

namespace facebook::velox {

uint64_t StringIdMap::id(std::string_view string) {
  std::lock_guard<std::mutex> l(mutex_);
  auto it = stringToId_.find(string);
  if (it != stringToId_.end()) {
    return it->second;
  }
  return kNoId;
}

void StringIdMap::release(uint64_t id) {
  std::lock_guard<std::mutex> l(mutex_);
  auto it = idToString_.find(id);
  if (it != idToString_.end()) {
    VELOX_CHECK_LT(
        0, it->second.numInUse, "Extra release of id in StringIdMap");
    if (--it->second.numInUse == 0) {
      VLOG(1) << "[StringIdMap] removed id: " << id;
      pinnedSize_ -= it->second.string.size();
      auto strIter = stringToId_.find(it->second.string);
      assert(strIter != stringToId_.end());
      stringToId_.erase(strIter);
      idToString_.erase(it);
    }
  }
}

void StringIdMap::checkAndReportIntegrity(uint64_t id) const {
  auto reportError = [](const std::string& msg) {
    LOG(ERROR) << "[StringIdMap] " << msg;
  };

  if (idToString_.size() != stringToId_.size()) {
    reportError(fmt::format(
        "idToString size {}. stringToId size {}. ",
        idToString_.size(),
        stringToId_.size()));
  }

  std::vector<std::pair<uint64_t, std::string>> missingFiles;
  uint64_t pinnedSize{0};
  for (auto it = stringToId_.begin(); it != stringToId_.end(); ++it) {
    pinnedSize += it->first.length();
    if (it->second == id) {
      reportError(fmt::format(
          "Found missing id {} in stringToId with file \"{}\".",
          id,
          it->first));
    }
    auto idEntry = idToString_.find(it->second);
    if (idEntry == idToString_.end()) {
      missingFiles.push_back(
          std::pair<uint64_t, std::string>(it->second, it->first));
    }
  }

  if (pinnedSize != pinnedSize_) {
    reportError(fmt::format(
        "stringToId pinnedSize {} does not match actual pinnedSize {}.",
        pinnedSize_,
        pinnedSize));
  }

  for (auto fileIt = missingFiles.begin(); fileIt != missingFiles.end();
       ++fileIt) {
    if (fileIt->first != id) {
      reportError(fmt::format(
          "Extra file {} in stringToId with id {}. ",
          fileIt->second,
          fileIt->first));
    }
    for (auto it = idToString_.begin(); it != idToString_.end(); ++it) {
      if (fileIt->second == it->second.string) {
        reportError(fmt::format(
            " Extra file \"{}\" maps to id {} in idToString with entryId {} and numInUse {}. ",
            fileIt->second,
            it->first,
            it->second.id,
            it->second.numInUse));
      }
    }
  }
  missingFiles.clear();
  pinnedSize = 0;

  for (auto idIt = idToString_.begin(); idIt != idToString_.end(); ++idIt) {
    pinnedSize += idIt->second.string.length();
    if (idIt->first != idIt->second.id) {
      reportError(fmt::format(
          " idToString id does {} not match corresponding entry id {} file {} numInUse {}. ",
          idIt->first,
          idIt->second.id,
          idIt->second.string,
          idIt->second.numInUse));
    }

    auto stringEntry = stringToId_.find(idIt->second.string);
    if (stringEntry == stringToId_.end()) {
      missingFiles.push_back(
          std::pair<uint64_t, std::string>(idIt->first, idIt->second.string));
    }
  }

  if (pinnedSize != pinnedSize_) {
    reportError(fmt::format(
        "idToString pinnedSize {} does not match actual pinnedSize {}.",
        pinnedSize_,
        pinnedSize));
  }

  for (auto fileIt = missingFiles.begin(); fileIt != missingFiles.end();
       ++fileIt) {
    reportError(fmt::format(
        "Extra id {} in idToString with file \"{}\". ",
        fileIt->first,
        fileIt->second));
    for (auto stringIt = stringToId_.begin(); stringIt != stringToId_.end();
         ++stringIt) {
      // Find if the missing id has an entry (which would map to a different
      // file).
      if (fileIt->first == stringIt->second) {
        reportError(fmt::format(
            " Extra id {} maps to file \"{}\" in stringToId . ",
            fileIt->first,
            stringIt->first));
      }
    }
  }
}

void StringIdMap::addReference(uint64_t id) {
  std::lock_guard<std::mutex> l(mutex_);
  auto it = idToString_.find(id);
  if (it == idToString_.end()) {
    checkAndReportIntegrity(id);
    VELOX_FAIL(
        "Trying to add a reference to id {} that is not in StringIdMap", id);
  }

  ++it->second.numInUse;
}

uint64_t StringIdMap::makeId(std::string_view string) {
  std::lock_guard<std::mutex> l(mutex_);
  auto it = stringToId_.find(string);
  if (it != stringToId_.end()) {
    auto entry = idToString_.find(it->second);
    VELOX_CHECK(entry != idToString_.end());
    if (++entry->second.numInUse == 1) {
      pinnedSize_ += entry->second.string.size();
    }

    return it->second;
  }
  Entry entry;
  entry.string = std::string(string);
  // Check that we do not use an id twice. In practice this never
  // happens because the int64 counter would have to wrap around for
  // this. Even if this happened, the time spent in the loop would
  // have a low cap since the number of mappings would in practice
  // be in the 100K range.
  do {
    entry.id = ++lastId_;
  } while (idToString_.find(entry.id) != idToString_.end());
  entry.numInUse = 1;
  pinnedSize_ += entry.string.size();
  auto id = entry.id;
  auto& entryInTable = idToString_[id] = std::move(entry);
  stringToId_[entryInTable.string] = entry.id;
  return lastId_;
}

} // namespace facebook::velox
