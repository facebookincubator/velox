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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox::parquet::arrow {

/// Truncates a UTF-8 encoded string to at most 'numCodePoints' Unicode code
/// points. Returns a string_view pointing to the truncated portion of the
/// input string. This is used for computing lower bound statistics,
/// as the truncated string is guaranteed to be less than or equal to the
/// original string in lexicographic order.
///
/// @param input The UTF-8 encoded input string.
/// @param numCodePoints Maximum number of Unicode code points to retain.
/// @return A string_view of the truncated string.
std::string_view truncateUtf8(std::string_view input, int32_t numCodePoints);

/// Rounds up a UTF-8 encoded string to produce an exclusive upper bound.
/// The result is guaranteed to be greater than any string that shares the
/// same prefix up to 'numCodePoints' code points. This is used for computing
/// upper bound statistics.
///
/// The function behaves as follows:
/// - If the string has fewer than or equal to 'numCodePoints' code points,
///   returns the original string unchanged.
/// - Otherwise, truncates to 'numCodePoints' code points and increments
///   code points from the last to the first, returning immediately on the
///   first successful increment.
/// - If no code point can be incremented (e.g., all are at max value
///   U+10FFFF), returns std::nullopt.
///
/// @param input The UTF-8 encoded input string.
/// @param numCodePoints Maximum number of Unicode code points to retain.
/// @return A new string containing the rounded-up result, or std::nullopt if
///         no valid upper bound can be computed.
std::optional<std::string> roundUpUtf8(
    std::string_view input,
    int32_t numCodePoints);

/// Computes an upper bound for binary data by truncating to a specified length
/// and incrementing the last byte that is not 0xFF.
///
/// This function is used for computing upper bounds on binary statistics
/// (e.g., for Parquet file metadata). It follows the algorithm described in
/// Apache Iceberg's BinaryUtil.truncateBinaryMax().
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
std::optional<std::string> roundUpBinary(
    std::string_view input,
    int32_t truncateLength);

} // namespace facebook::velox::parquet::arrow
