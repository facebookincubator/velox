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

// A whole-column linear predictor of a value from its row number, removed
// before SubIntSplit plans its sections and added back on every read.
//
// A bit-range section cannot see a packed ID's per-row counter, since the
// counter's carries cross whichever bit boundaries the planner picks.
// Subtracting slope * row + base first leaves only the residual distance
// from that line, which sections do encode well, at the cost of one
// multiply-add per row with no dependence on neighbouring rows.

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace facebook::nimble::subintsplit {

/// The predictor subtracted from every value of a SubIntSplit stream before
/// its sections were planned: value = residual + slope * row + base, in the
/// physical type's modular arithmetic. Inactive (the default) means the
/// stream stores values as they are.
struct RowFrame {
  /// Amount the predictor grows per row.
  uint64_t slope{0};
  /// Predictor at row zero.
  uint64_t base{0};

  /// Whether the stream carries a frame at all.
  bool active() const {
    return slope != 0 || base != 0;
  }
};

/// First byte of the frame block. A reader that predates frames reads this
/// byte as a key section index, and no stream has 254 sections, so such a
/// reader rejects the stream instead of returning residuals as values.
inline constexpr uint8_t kRowFrameGuard = 0xFE;
/// Guard byte, slope and base.
inline constexpr uint32_t kRowFrameHeaderSize = 17;

/// Adds the frame back to `count` residuals of rows firstRow, firstRow + 1, ...
template <typename PhysicalType>
inline void addRowFrame(
    const RowFrame& frame,
    uint64_t firstRow,
    PhysicalType* values,
    uint32_t count) {
  // Physical types are unsigned, so the truncation to a 32-bit type below is
  // the modular arithmetic the frame is defined in, not a loss.
  const auto slope = static_cast<PhysicalType>(frame.slope);
  auto predicted =
      static_cast<PhysicalType>(frame.slope * firstRow + frame.base);
  for (uint32_t i = 0; i < count; ++i) {
    values[i] = static_cast<PhysicalType>(values[i] + predicted);
    predicted = static_cast<PhysicalType>(predicted + slope);
  }
}

/// The frame's prediction for one row.
template <typename PhysicalType>
inline PhysicalType rowFramePrediction(const RowFrame& frame, uint64_t row) {
  return static_cast<PhysicalType>(frame.slope * row + frame.base);
}

/// Rows between the two values a growth sample compares.
inline constexpr uint32_t kRowFrameStride = 1'024;
/// A second stride the fitted slope must also hold over, sharing no large
/// factor with the first: sampling at one stride alone cannot see what a
/// column does between samples, so a field whose period divides that stride
/// can alias to a slope it does not have.
inline constexpr uint32_t kRowFrameCheckStride = 1'000;
/// Fewest growth samples a frame is fitted on. Below this a median of strides
/// says too little about the column to be worth a planner pass.
inline constexpr uint32_t kRowFrameMinStrides = 16;
/// Share of strides whose growth must agree with the fitted slope, to within
/// half a stride, for the column to count as tracking a line.
inline constexpr double kRowFrameMinAgreement = 0.9;

/// Fits a row frame to `values`, or returns an inactive one when the column
/// does not follow a line through its rows.
///
/// The slope is read off the low `width` bits for the widest width at which
/// nearly every stride grows by the same whole multiple, checked again at a
/// second stride to rule out aliasing. Low bits rather than the whole word,
/// since a packed ID usually keeps a field above its counters that moves
/// independently of the row.
///
/// The base is the most negative residual of those low bits, so the fields
/// below `width` stay non-negative and do not borrow from the fields above
/// them.
template <typename PhysicalType>
RowFrame fitRowFrame(std::span<const PhysicalType> values) {
  const int kBits = static_cast<int>(sizeof(PhysicalType) * 8);
  const uint64_t numStrides =
      values.size() < 2 ? 0 : (values.size() - 1) / kRowFrameStride;
  if (numStrides < kRowFrameMinStrides) {
    return {};
  }

  const auto signExtend = [](uint64_t value, int width) -> int64_t {
    if (width >= 64) {
      return static_cast<int64_t>(value);
    }
    const uint64_t signBit = uint64_t{1} << (width - 1);
    const uint64_t mask = (uint64_t{1} << width) - 1;
    return static_cast<int64_t>(((value & mask) ^ signBit) - signBit);
  };

  // Share of strides of `stride` rows over which the low `width` bits grow by
  // slope * stride, to within half a stride.
  const auto agreement = [&](int width, __int128 slope, uint64_t stride) {
    const uint64_t numSamples = (values.size() - 1) / stride;
    const auto length = static_cast<__int128>(stride);
    uint64_t agreeing = 0;
    for (uint64_t s = 0; s < numSamples; ++s) {
      const uint64_t from = values[s * stride];
      const uint64_t to = values[(s + 1) * stride];
      __int128 deviation =
          static_cast<__int128>(signExtend(to - from, width)) - slope * length;
      if (deviation < 0) {
        deviation = -deviation;
      }
      agreeing += deviation <= length / 2 ? 1 : 0;
    }
    return static_cast<double>(agreeing) / static_cast<double>(numSamples);
  };

  std::vector<int64_t> growth(numStrides);
  const auto strideLength = static_cast<__int128>(kRowFrameStride);
  for (int width = kBits; width > 0; --width) {
    for (uint64_t s = 0; s < numStrides; ++s) {
      const uint64_t from = values[s * kRowFrameStride];
      const uint64_t to = values[(s + 1) * kRowFrameStride];
      growth[s] = signExtend(to - from, width);
    }
    const auto middle = growth.begin() + numStrides / 2;
    std::nth_element(growth.begin(), middle, growth.end());
    const __int128 median = *middle;
    // Rounded to the nearest whole slope, ties away from zero.
    const __int128 slope = median >= 0
        ? (median + strideLength / 2) / strideLength
        : -((-median + strideLength / 2) / strideLength);
    if (slope == 0 ||
        agreement(width, slope, kRowFrameStride) < kRowFrameMinAgreement ||
        agreement(width, slope, kRowFrameCheckStride) < kRowFrameMinAgreement) {
      continue;
    }

    RowFrame frame;
    frame.slope = static_cast<uint64_t>(static_cast<int64_t>(slope));
    int64_t lowest = 0;
    for (size_t row = 0; row < values.size(); ++row) {
      const uint64_t residual =
          static_cast<uint64_t>(values[row]) - frame.slope * row;
      lowest = std::min(lowest, signExtend(residual, width));
    }
    frame.base = static_cast<uint64_t>(lowest);
    if (kBits < 64) {
      frame.slope &= (uint64_t{1} << kBits) - 1;
      frame.base &= (uint64_t{1} << kBits) - 1;
    }
    return frame;
  }
  return {};
}

/// Share of adjacent row pairs that must step by one common non-zero amount for
/// a step frame to be fitted.
inline constexpr double kStepFrameMinShare = 0.25;

/// Fits a row frame whose slope is the non-zero step that adjacent rows most
/// often take, or returns an inactive one when no step is common enough.
///
/// Unlike fitRowFrame, which asks whether a column follows one line across
/// the whole stream, this asks whether it follows many short lines of the
/// same slope, broken wherever something else in the value moves. The base
/// is zero, since the frame is kept on encoded bytes and there is no
/// borrow-free residual to protect.
template <typename PhysicalType>
RowFrame fitStepFrame(std::span<const PhysicalType> values) {
  if (values.size() < 2 ||
      (values.size() - 1) / kRowFrameStride < kRowFrameMinStrides) {
    return {};
  }
  const auto stepAt = [&values](size_t row) {
    return static_cast<PhysicalType>(values[row + 1] - values[row]);
  };

  // Misra-Gries with three counters keeps every step taken by more than a
  // quarter of the pairs among its candidates, in one pass and constant
  // memory. A second pass counts the candidates exactly.
  constexpr size_t kCandidates = 3;
  std::array<PhysicalType, kCandidates> candidates{};
  std::array<uint64_t, kCandidates> weights{};
  const size_t numPairs = values.size() - 1;
  for (size_t row = 0; row < numPairs; ++row) {
    const PhysicalType step = stepAt(row);
    if (step == 0) {
      continue;
    }
    bool placed = false;
    for (size_t i = 0; i < kCandidates && !placed; ++i) {
      if (weights[i] > 0 && candidates[i] == step) {
        ++weights[i];
        placed = true;
      }
    }
    for (size_t i = 0; i < kCandidates && !placed; ++i) {
      if (weights[i] == 0) {
        candidates[i] = step;
        weights[i] = 1;
        placed = true;
      }
    }
    if (!placed) {
      for (auto& weight : weights) {
        --weight;
      }
    }
  }

  std::array<uint64_t, kCandidates> counts{};
  for (size_t row = 0; row < numPairs; ++row) {
    const PhysicalType step = stepAt(row);
    for (size_t i = 0; i < kCandidates; ++i) {
      counts[i] += (weights[i] > 0 && candidates[i] == step) ? 1 : 0;
    }
  }
  const auto best = static_cast<size_t>(
      std::max_element(counts.begin(), counts.end()) - counts.begin());
  if (static_cast<double>(counts[best]) <
      kStepFrameMinShare * static_cast<double>(numPairs)) {
    return {};
  }
  return {.slope = static_cast<uint64_t>(candidates[best]), .base = 0};
}

/// Writes value - (slope * row + base) for every row of `values` into
/// `residuals`.
template <typename PhysicalType>
void subtractRowFrame(
    const RowFrame& frame,
    std::span<const PhysicalType> values,
    std::vector<PhysicalType>& residuals) {
  residuals.resize(values.size());
  auto predicted = static_cast<PhysicalType>(frame.base);
  const auto slope = static_cast<PhysicalType>(frame.slope);
  for (size_t row = 0; row < values.size(); ++row) {
    residuals[row] = static_cast<PhysicalType>(values[row] - predicted);
    predicted = static_cast<PhysicalType>(predicted + slope);
  }
}

} // namespace facebook::nimble::subintsplit
