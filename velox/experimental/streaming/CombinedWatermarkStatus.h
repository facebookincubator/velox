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

#include <climits>
#include <vector>

namespace facebook::velox::streaming {

class PartialWatermark;

/**
 * This class is used to calculate the watermark of an streaming operator.
 * An operator may have several inputs, the watermark is the minimal one
 * pushed down from all inputs.
 */
class CombinedWatermarkStatus {
 public:
  CombinedWatermarkStatus(int numWatermarks) {
    partialWatermarks_.resize(numWatermarks);
  }

  bool updateWatermark(int index, long timestamp);

  long getCombinedWatermark();

 private:
  bool updateCombinedWatermark();

  std::vector<PartialWatermark> partialWatermarks_;
  bool idle_ = false;
  long combinedWatermark_ = LONG_MIN;
};

class PartialWatermark {
 public:
  bool setWatermark(long watermark);

  bool idle() const {
    return idle_;
  }

  long watermark() const {
    return watermark_;
  }

  void setIdle(bool idle) {
    idle_ = idle;
  }

 private:
  long watermark_ = LONG_MIN;
  bool idle_ = false;
};

} // namespace facebook::velox::streaming
