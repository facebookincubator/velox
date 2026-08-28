/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <cstdint>
#include <map>

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {
class InputBufferGrowthPolicy {
 public:
  virtual ~InputBufferGrowthPolicy() = default;

  virtual uint64_t getExtendedCapacity(uint64_t newSize, uint64_t capacity)
      const = 0;
};

// Default growth policy for input buffering is a step function across
// the ranges.
// The default growth policy is based on item count in the vectors because
// it tend to be more uniform (tied together by row count) across types.
// However, bytes based policies are worth evaluation as well.
class DefaultInputBufferGrowthPolicy : public InputBufferGrowthPolicy {
 public:
  explicit DefaultInputBufferGrowthPolicy(
      std::map<uint64_t, float> rangeConfigs)
      : rangeConfigs_{std::move(rangeConfigs)} {
    NIMBLE_CHECK(
        !rangeConfigs_.empty() && rangeConfigs_.begin()->first > 0,
        "Invalid range config supplied for default buffer input growth policy.");
    minCapacity_ = rangeConfigs_.begin()->first;
  }

  ~DefaultInputBufferGrowthPolicy() override = default;

  uint64_t getExtendedCapacity(uint64_t newSize, uint64_t capacity)
      const override;

  static std::unique_ptr<InputBufferGrowthPolicy> withDefaultRanges() {
    return std::make_unique<DefaultInputBufferGrowthPolicy>(
        std::map<uint64_t, float>{
            {32UL, 4.0}, {512UL, 1.414}, {4096UL, 1.189}});
  }

  // Growth policy for per-stream string buffers when disableSharedStringBuffers
  // is enabled. Uses byte-based thresholds since string data is stored as raw
  // bytes.
  static std::unique_ptr<InputBufferGrowthPolicy> withStringBufferRanges() {
    return std::make_unique<DefaultInputBufferGrowthPolicy>(
        std::map<uint64_t, float>{
            {4096UL, 2.0}, {4194304UL, 1.414}, {12582912UL, 1.189}});
  }

 private:
  // Map of range lowerbounds and the growth factor for the range.
  // The first lowerbound is the smallest unit of allocation and the last range
  // extends uint64_t max.
  std::map<uint64_t, float> rangeConfigs_;
  uint64_t minCapacity_;
};

class ExactGrowthPolicy : public InputBufferGrowthPolicy {
 public:
  uint64_t getExtendedCapacity(uint64_t newSize, uint64_t /* capacity */)
      const override {
    return newSize;
  }
};

} // namespace facebook::nimble
