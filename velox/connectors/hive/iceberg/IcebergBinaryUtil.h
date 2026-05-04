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

#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox::connector::hive::iceberg {

/// Computes an upper bound for binary data by truncating to a specified length
/// and incrementing the last byte that is not 0xFF.
///
/// This function is used in Apache Iceberg for computing upper bounds on
/// binary statistics (e.g., for Parquet file metadata). It follows the
/// algorithm described in Iceberg's BinaryUtil.truncateBinaryMax().
///
/// The algorithm:
/// 1. If the input is shorter than or equal to truncateLength, return it as-is.
/// 2. Otherwise, truncate to truncateLength bytes.
/// 3. Starting from the last byte, find the first byte that is not 0xFF.
/// 4. Increment that byte and truncate everything after it.
/// 5. If all bytes are 0xFF, return std::nullopt (no valid upper bound).
///
/// @param input The binary data as a string_view.
/// @param truncateLength Maximum number of bytes to retain before incrementing.
/// @return An optional string containing the upper bound, or std::nullopt if
///         no valid upper bound exists (e.g., all bytes are 0xFF).
inline std::optional<std::string> roundUpBinary(
    std::string_view input,
    int32_t truncateLength) {
  if (truncateLength <= 0) {
    return std::nullopt;
  }

  const size_t length = static_cast<size_t>(truncateLength);
  if (input.size() <= length) {
    return std::string(input);
  }

  // Create a mutable copy of the truncated input.
  std::string result(input.data(), length);

  // Try incrementing bytes from the end.
  for (size_t i = length; i-- > 0;) {
    unsigned char byte = static_cast<unsigned char>(result[i]);

    if (byte != 0xFF) { // Can increment without overflow.
      result[i] = static_cast<char>(byte + 1);
      // Truncate to i + 1 bytes (remove trailing bytes after increment point).
      result.resize(i + 1);
      return result;
    }
    // If byte == 0xFF, it will overflow, continue to previous byte.
  }

  // All bytes were 0xFF and overflowed - no valid upper bound.
  return std::nullopt;
}

} // namespace facebook::velox::connector::hive::iceberg
