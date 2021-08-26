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

#include "velox/common/memory/MemoryUsageTracker.h"
#include "velox/common/memory/Memory.h"

namespace facebook::velox::memory {
std::shared_ptr<MemoryUsageTracker> MemoryUsageTracker::create(
    const std::shared_ptr<MemoryUsageTracker>& parent,
    UsageType type,
    const MemoryUsageConfig& config) {
  struct SharedMemoryUsageTracker : public MemoryUsageTracker {
    SharedMemoryUsageTracker(
        const std::shared_ptr<MemoryUsageTracker>& parent,
        UsageType type,
        const MemoryUsageConfig& config)
        : MemoryUsageTracker(parent, type, config) {}
  };

  // If it is not forMemoryManager and parent is null, set the parent
  // to MemoryManager's usage tracker.
  auto parentTracker = (!config.forMemoryManager && parent == nullptr)
      ? MemoryManager<>::getProcessDefaultManager().getMemoryUsageTracker()
      : parent;
  return std::make_shared<SharedMemoryUsageTracker>(
      parentTracker, type, config);
}

void MemoryUsageTracker::update(UsageType type, int64_t size) {
  if (size > 0) {
    ++numAllocs_[static_cast<int>(type)];
    cumulativeBytes_[static_cast<int>(type)] += size;
    ++numAllocs_[static_cast<int>(UsageType::kTotalMem)];
    cumulativeBytes_[static_cast<int>(UsageType::kTotalMem)] += size;
  }

  auto currentOfType =
      currentUsageInBytes_[static_cast<int32_t>(type)].fetch_add(size);
  auto currentTotal =
      currentUsageInBytes_[static_cast<int32_t>(UsageType::kTotalMem)]
          .fetch_add(size);

  try {
    if (size > 0 &&
        currentTotal > maxMemory_[static_cast<int>(UsageType::kTotalMem)]) {
      if (growCallback_) {
        if (growCallback_(type, size, this)) {
          return;
        }
      }

      VELOX_MEM_CAP_EXCEEDED();
    }
    if (parent_) {
      parent_->update(type, size);
    }
  } catch (const std::exception& e) {
    currentUsageInBytes_[static_cast<int32_t>(type)].fetch_sub(size);
    currentUsageInBytes_[static_cast<int32_t>(UsageType::kTotalMem)].fetch_sub(
        size);
    std::rethrow_exception(std::current_exception());
  }
  maySetMax(type, currentOfType);
  maySetMax(UsageType::kTotalMem, currentTotal);
}

namespace {

std::string usageString(const std::array<std::atomic<int64_t>, 4>& usage) {
  std::stringstream out;
  const char* heading[] = {"user", "sys", "total", "revocable"};
  for (auto i = 0; i < 4; ++i) {
    if (usage[i]) {
      out << heading[i] << usage[i] << " ";
    }
  }
  return out.str();
}

} // namespace

std::string MemoryUsageTracker::toString() const {
  std::stringstream out;
  out << "<tracker current:" << usageString(currentUsageInBytes_);
  if (reservation_) {
    out << " used/reserved " << usedReservation_ << "/" << reservation_;
  }
  auto max = getMaxTotalBytes();
  if (max < kMaxMemory) {
    out << " max " << max;
  }
  out << ">";
  return out.str();
}

} // namespace facebook::velox::memory
