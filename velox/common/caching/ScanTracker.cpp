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

#include "velox/common/caching/ScanTracker.h"
#include "velox/common/caching/FileGroupStats.h"

#include <sstream>

namespace facebook::velox::cache {

// Marks that 'bytes' worth of data may be accessed in the
// future. See TrackingData for meaning of quantum.
void ScanTracker::recordReference(
    const TrackingId id,
    uint64_t bytes,
    uint64_t fileId,
    uint64_t groupId) {
  if (fileGroupStats_) {
    fileGroupStats_->recordReference(fileId, groupId, id, bytes);
  }
  std::lock_guard<std::mutex> l(mutex_);
  data_[id].incrementReference(bytes, loadQuantum_);
  sum_.incrementReference(bytes, loadQuantum_);
}

void ScanTracker::recordRead(
    const TrackingId id,
    uint64_t bytes,
    uint64_t fileId,
    uint64_t groupId) {
  if (fileGroupStats_) {
    fileGroupStats_->recordRead(fileId, groupId, id, bytes);
  }
  std::lock_guard<std::mutex> l(mutex_);
  data_[id].incrementRead(bytes);
  sum_.incrementRead(bytes);
}

std::string ScanTracker::toString() const {
  std::stringstream out;
  out << "ScanTracker for " << id_ << std::endl;
  for (auto& pair : data_) {
    int pct = 100 * pair.second.readBytes / (1 + pair.second.referencedBytes);
    out << pair.first.id() << ": " << pct << "% " << pair.second.readBytes
        << "/" << pair.second.numReads << std::endl;
  }
  return out.str();
}
} // namespace facebook::velox::cache
