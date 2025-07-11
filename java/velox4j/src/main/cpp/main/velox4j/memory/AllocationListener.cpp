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
#include "velox4j/memory/AllocationListener.h"

namespace facebook::velox4j {

class NoopAllocationListener : public AllocationListener {
 public:
  void allocationChanged(int64_t diff) override {
    // no-op
  }

  const int64_t currentBytes() const override {
    return 0;
  }
  const int64_t peakBytes() const override {
    return 0;
  }
};

std::unique_ptr<AllocationListener> AllocationListener::noop() {
  return std::make_unique<NoopAllocationListener>();
}

void BlockAllocationListener::allocationChanged(int64_t diff) {
  if (diff == 0) {
    return;
  }
  int64_t granted = reserve(diff);
  delegated_->allocationChanged(granted);
}

int64_t BlockAllocationListener::reserve(int64_t diff) {
  std::lock_guard<std::mutex> lock(mutex_);
  usedBytes_ += diff;
  int64_t newBlockCount;
  if (usedBytes_ == 0) {
    newBlockCount = 0;
  } else {
    // ceil to get the required block number
    newBlockCount = (usedBytes_ - 1) / blockSize_ + 1;
  }
  int64_t bytesGranted = (newBlockCount - blocksReserved_) * blockSize_;
  blocksReserved_ = newBlockCount;
  peakBytes_ = std::max(peakBytes_, usedBytes_);
  return bytesGranted;
}
} // namespace facebook::velox4j
