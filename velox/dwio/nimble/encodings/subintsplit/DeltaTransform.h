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
#include <span>
#include <type_traits>

// Zigzag-delta pre-transform applied to a stream before it is split into bit
// ranges.
//
// Zigzag maps signed deltas onto unsigned values so that small negative steps
// stay small instead of wrapping to near-2^64. Mirrors the mapping OpenZL's
// delta path uses, and is the difference between ReverseSorted compressing and
// actively regressing.

namespace facebook::nimble::subintsplit {

template <typename U>
inline U zigzagEncodeDelta(U current, U previous) noexcept {
  using Signed = std::make_signed_t<U>;
  constexpr int kSignShift = static_cast<int>(sizeof(U) * 8 - 1);
  const Signed difference = static_cast<Signed>(current - previous);
  return static_cast<U>(
      (static_cast<U>(difference) << 1) ^
      static_cast<U>(difference >> kSignShift));
}

template <typename U>
inline U zigzagDecodeDelta(U encoded, U previous) noexcept {
  using Signed = std::make_signed_t<U>;
  const Signed difference =
      static_cast<Signed>((encoded >> 1) ^ (~(encoded & U{1}) + U{1}));
  return static_cast<U>(previous + static_cast<U>(difference));
}

/// Writes the zigzag-delta residuals of `values` into `residuals`, which must
/// be the same length. The first residual is the first value verbatim, so the
/// stream can be reconstructed without external state.
template <typename U>
inline void encodeDeltas(
    std::span<const U> values,
    std::span<U> residuals) noexcept {
  residuals[0] = values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    residuals[i] = zigzagEncodeDelta<U>(values[i], values[i - 1]);
  }
}

/// Turns the zigzag residuals in `values` back into absolute values in place,
/// continuing from `accumulator` and leaving the last value there.
///
/// `isStreamStart` marks the very first value of the stream, which is stored
/// verbatim rather than as a delta.
template <typename U>
inline void
decodeDeltas(std::span<U> values, U& accumulator, bool isStreamStart) noexcept {
  size_t index = 0;
  if (isStreamStart) {
    accumulator = values[0];
    index = 1;
  }
  for (; index < values.size(); ++index) {
    accumulator = zigzagDecodeDelta<U>(values[index], accumulator);
    values[index] = accumulator;
  }
}

} // namespace facebook::nimble::subintsplit
