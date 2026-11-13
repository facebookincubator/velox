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
#include "velox/dwio/nimble/tablet/StripeGroupFlushPolicy.h"

namespace facebook::nimble {

namespace {

// Estimate of an entry until a written group calibrates it: 8 bytes for the
// offset, 4 for the size and 1 for the compression type.
constexpr double kInitialBytesPerEntry{13};

} // namespace

StripeGroupFlushPolicy::StripeGroupFlushPolicy(uint32_t thresholdBytes)
    : thresholdBytes_{thresholdBytes},
      estimatedBytesPerEntry_{kInitialBytesPerEntry} {}

bool StripeGroupFlushPolicy::shouldFlush(uint64_t numEntries) const {
  return 4 + static_cast<double>(numEntries) * estimatedBytesPerEntry_ >=
      static_cast<double>(thresholdBytes_);
}

void StripeGroupFlushPolicy::onFlush(
    uint64_t numEntries,
    uint64_t encodedBytes) {
  // A group without entries has no bytes per entry to calibrate with.
  if (numEntries == 0) {
    return;
  }
  estimatedBytesPerEntry_ =
      static_cast<double>(encodedBytes) / static_cast<double>(numEntries);
}

} // namespace facebook::nimble
