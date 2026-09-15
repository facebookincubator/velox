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

#include <cudf/utilities/error.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace facebook::velox::cudf_velox::compression::detail {

[[nodiscard]] inline bool tryAddSizes(std::size_t left,
                                      std::size_t right,
                                      std::size_t& result) noexcept {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

[[nodiscard]] inline std::size_t checkedAddSizes(std::size_t left,
                                                 std::size_t right,
                                                 const char* message) {
  std::size_t result = 0;
  CUDF_EXPECTS(tryAddSizes(left, right, result), message, std::overflow_error);
  return result;
}

[[nodiscard]] inline std::size_t checkedMultiplySizes(std::size_t left,
                                                      std::size_t right,
                                                      const char* message) {
  CUDF_EXPECTS(
      left == 0 || right <= std::numeric_limits<std::size_t>::max() / left,
      message,
      std::overflow_error);
  return left * right;
}

// nvCOMP ANS currently requires at least 8-byte alignment. This format uses a
// conservative 16-byte frame boundary, which also keeps the stable wire layout
// compatible with nvCOMP's documented higher-alignment performance guidance.
// Descriptors retain each frame's true byte size.
inline constexpr std::size_t kNvcompFrameAlignment = 16;

[[nodiscard]] inline bool tryNvcompAlignedSize(
    std::size_t size,
    std::size_t& alignedSize) noexcept {
  constexpr auto mask = kNvcompFrameAlignment - 1;
  static_assert((kNvcompFrameAlignment & mask) == 0);
  std::size_t padded = 0;
  if (!tryAddSizes(size, mask, padded)) {
    return false;
  }
  alignedSize = padded & ~mask;
  return true;
}

[[nodiscard]] inline std::size_t nvcompAlignedSize(std::size_t size) {
  std::size_t alignedSize = 0;
  CUDF_EXPECTS(tryNvcompAlignedSize(size, alignedSize),
               "nvCOMP aligned size overflow",
               std::overflow_error);
  return alignedSize;
}

} // namespace facebook::velox::cudf_velox::compression::detail
