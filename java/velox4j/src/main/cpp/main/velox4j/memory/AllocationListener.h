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
#pragma once

#include <algorithm>
#include <memory>
#include <mutex>

namespace velox4j {

class AllocationListener {
 public:
  static std::unique_ptr<AllocationListener> noop();

  virtual ~AllocationListener() = default;

  // Delete copy/move CTORs.
  AllocationListener(AllocationListener&&) = delete;
  AllocationListener(const AllocationListener&) = delete;
  AllocationListener& operator=(const AllocationListener&) = delete;
  AllocationListener& operator=(AllocationListener&&) = delete;

  // Value of diff can be either positive or negative
  virtual void allocationChanged(int64_t diff) = 0;

  virtual const int64_t currentBytes() const = 0;

  virtual const int64_t peakBytes() const = 0;

 protected:
  AllocationListener() = default;
};

/// Memory changes will be round to specified block size which aim to decrease
/// delegated listener calls.
// The class must be thread safe.
class BlockAllocationListener final : public AllocationListener {
 public:
  BlockAllocationListener(
      std::unique_ptr<AllocationListener> delegated,
      int64_t blockSize)
      : delegated_(std::move(delegated)), blockSize_(blockSize) {}

  void allocationChanged(int64_t diff) override;

  const int64_t currentBytes() const override {
    return reservationBytes_;
  }

  const int64_t peakBytes() const override {
    return peakBytes_;
  }

 private:
  inline int64_t reserve(int64_t diff);

  const std::unique_ptr<AllocationListener> delegated_;
  const int64_t blockSize_;
  int64_t blocksReserved_{0L};
  int64_t usedBytes_{0L};
  int64_t peakBytes_{0L};
  int64_t reservationBytes_{0L};

  mutable std::mutex mutex_;
};
} // namespace velox4j
