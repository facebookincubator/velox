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
#include "velox/experimental/streaming/CombinedWatermarkStatus.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::streaming {

bool CombinedWatermarkStatus::updateWatermark(int index, long timestamp) {
  VELOX_CHECK(index < partialWatermarks_.size(), "Index out of range");
  auto& watermark = partialWatermarks_[index];

  if (!watermark.setWatermark(timestamp)) {
    return false;
  }

  // If the watermark is already set, we do not update it.
  return updateCombinedWatermark();
}

long CombinedWatermarkStatus::getCombinedWatermark() {
  return combinedWatermark_;
}

bool CombinedWatermarkStatus::updateCombinedWatermark() {
  long minimumOverAll = LONG_MIN;
  bool allIdle = true;
  for (const auto& watermark : partialWatermarks_) {
    if (!watermark.idle()) {
      minimumOverAll = std::max(minimumOverAll, watermark.watermark());
      allIdle = false;
    }
  }

  idle_ = allIdle;
  if (!allIdle && minimumOverAll > combinedWatermark_) {
    combinedWatermark_ = minimumOverAll;
    return true;
  }

  // If the new combined watermark is not greater, we do not update it.
  return false;
}

bool PartialWatermark::setWatermark(long watermark) {
  if (watermark < watermark_) {
    // If the new watermark is less than or equal to the current one, we do not update it.
    return false;
  }

  watermark_ = watermark;
  idle_ = false;
  return true;
}

} // namespace facebook::velox::streaming
