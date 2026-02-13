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
#include <string_view>

#include <fmt/format.h>

#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox::common {

/// Defines a disk region to read.
struct Region {
  uint64_t offset;
  uint64_t length;
  /// Optional label used by lower layers for cache warm up.
  std::string_view label;

  Region(uint64_t offset = 0, uint64_t length = 0, std::string_view label = {})
      : offset{offset}, length{length}, label{label} {}

  bool operator<(const Region& other) const {
    return offset < other.offset ||
        (offset == other.offset && length < other.length);
  }

  std::string toString() const {
    return fmt::format(
        "Region{{offset: {}, length: {}, label: {}}}",
        offset,
        succinctBytes(length),
        label);
  }
};

} // namespace facebook::velox::common
