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
#include <string>

#include <fmt/format.h>

namespace facebook::nimble {

/// Range of stripes of a file [startStripe, endStripe), by stripe index.
struct StripeRange {
  uint32_t startStripe{0};
  uint32_t endStripe{0};

  StripeRange() = default;
  StripeRange(uint32_t start, uint32_t end)
      : startStripe(start), endStripe(end) {}

  inline bool empty() const {
    return startStripe >= endStripe;
  }

  /// Returns how many stripes the range holds, none when it ends at or before
  /// its start.
  inline uint32_t numStripes() const {
    return empty() ? 0 : endStripe - startStripe;
  }

  inline bool contains(uint32_t stripe) const {
    return stripe >= startStripe && stripe < endStripe;
  }

  inline bool operator==(const StripeRange& other) const {
    return startStripe == other.startStripe && endStripe == other.endStripe;
  }

  inline std::string toString() const {
    return fmt::format("[{}, {})", startStripe, endStripe);
  }
};

} // namespace facebook::nimble
