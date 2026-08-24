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
#include "velox/dwio/parquet/writer/arrow/SizeStatistics.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"

namespace facebook::velox::parquet::arrow {
namespace {
void checkedAdd(int64_t& value, int64_t increment) {
  if (increment < 0 ||
      value > std::numeric_limits<int64_t>::max() - increment) {
    throw ParquetException("Size statistics overflow");
  }
  value += increment;
}

void mergeLevelHistogram(
    std::span<int64_t> histogram,
    std::span<const int64_t> other) {
  VELOX_DCHECK_EQ(histogram.size(), other.size());
  for (size_t i = 0; i < histogram.size(); ++i) {
    checkedAdd(histogram[i], other[i]);
  }
}

} // namespace

bool SizeStatistics::isSet() const {
  return !definitionLevelHistogram.empty() ||
      !repetitionLevelHistogram.empty() ||
      unencodedByteArrayDataBytes.has_value();
}

void SizeStatistics::incrementUnencodedByteArrayDataBytes(int64_t value) {
  VELOX_DCHECK(unencodedByteArrayDataBytes.has_value());
  checkedAdd(*unencodedByteArrayDataBytes, value);
}

void SizeStatistics::merge(const SizeStatistics& other) {
  if (repetitionLevelHistogram.size() !=
      other.repetitionLevelHistogram.size()) {
    throw ParquetException("Repetition level histogram size mismatch");
  }
  if (definitionLevelHistogram.size() !=
      other.definitionLevelHistogram.size()) {
    throw ParquetException("Definition level histogram size mismatch");
  }
  if (unencodedByteArrayDataBytes.has_value() !=
      other.unencodedByteArrayDataBytes.has_value()) {
    throw ParquetException(
        "Unencoded byte array data bytes are not consistent");
  }
  mergeLevelHistogram(repetitionLevelHistogram, other.repetitionLevelHistogram);
  mergeLevelHistogram(definitionLevelHistogram, other.definitionLevelHistogram);
  if (unencodedByteArrayDataBytes.has_value()) {
    checkedAdd(
        *unencodedByteArrayDataBytes, *other.unencodedByteArrayDataBytes);
  }
}

void SizeStatistics::reset() {
  repetitionLevelHistogram.assign(repetitionLevelHistogram.size(), 0);
  definitionLevelHistogram.assign(definitionLevelHistogram.size(), 0);
  if (unencodedByteArrayDataBytes.has_value()) {
    unencodedByteArrayDataBytes = 0;
  }
}

void SizeStatistics::validate(const ColumnDescriptor* descriptor) const {
  auto validateHistogram = [](const std::vector<int64_t>& histogram,
                              int16_t maxLevel,
                              const std::string& name) {
    if (histogram.empty()) {
      // A levels histogram is always allowed to be missing.
      return;
    }
    if (histogram.size() != static_cast<size_t>(maxLevel + 1)) {
      std::stringstream ss;
      ss << name << " level histogram size mismatch, size: " << histogram.size()
         << ", expected: " << (maxLevel + 1);
      throw ParquetException(ss.str());
    }
  };
  validateHistogram(
      repetitionLevelHistogram, descriptor->maxRepetitionLevel(), "Repetition");
  validateHistogram(
      definitionLevelHistogram, descriptor->maxDefinitionLevel(), "Definition");
  if (unencodedByteArrayDataBytes.has_value() &&
      descriptor->physicalType() != Type::kByteArray) {
    throw ParquetException(
        "Unencoded byte array data bytes does not support " +
        typeToString(descriptor->physicalType()));
  }
}

std::unique_ptr<SizeStatistics> SizeStatistics::make(
    const ColumnDescriptor* descriptor) {
  auto sizeStats = std::make_unique<SizeStatistics>();
  // If the max level is 0, the level histogram can be omitted because it
  // contains only single level (a.k.a. 0) and its count is equivalent to
  // `num_values` of the column chunk or data page.
  if (descriptor->maxRepetitionLevel() != 0) {
    sizeStats->repetitionLevelHistogram.resize(
        descriptor->maxRepetitionLevel() + 1, 0);
  }
  if (descriptor->maxDefinitionLevel() != 0) {
    sizeStats->definitionLevelHistogram.resize(
        descriptor->maxDefinitionLevel() + 1, 0);
  }
  if (descriptor->physicalType() == Type::kByteArray) {
    sizeStats->unencodedByteArrayDataBytes = 0;
  }
  return sizeStats;
}

void updateLevelHistogram(
    std::span<const int16_t> levels,
    std::span<int64_t> histogram) {
  const int64_t numLevels = static_cast<int64_t>(levels.size());
  VELOX_DCHECK_GE(histogram.size(), 1);
  const int16_t maxLevel = static_cast<int16_t>(histogram.size() - 1);
  if (maxLevel == 0) {
    checkedAdd(histogram[0], numLevels);
    return;
  }

#ifndef NDEBUG
  for (auto level : levels) {
    VELOX_DCHECK_LE(level, maxLevel);
  }
#endif

  if (maxLevel == 1) {
    // Specialize the common case for non-repeated non-nested columns.
    // Summing the levels gives us the number of 1s, and the number of 0s
    // follows. We do repeated sums in the int16_t space, which the compiler is
    // likely to vectorize efficiently.
    constexpr int64_t kChunkSize = 1 << 14;
    int64_t histogramOne = 0;
    auto it = levels.begin();
    while (it != levels.end()) {
      const auto chunkSize = std::min<int64_t>(levels.end() - it, kChunkSize);
      histogramOne += std::accumulate(it, it + chunkSize, int16_t{0});
      it += chunkSize;
    }
    checkedAdd(histogram[0], numLevels - histogramOne);
    checkedAdd(histogram[1], histogramOne);
    return;
  }

  // The generic implementation issues a series of histogram load-stores.
  // However, it limits store-to-load dependencies by interleaving partial
  // histogram updates.
  constexpr int kUnroll = 4;
  std::array<std::vector<int64_t>, kUnroll> partialHistograms;
  for (auto& partialHistogram : partialHistograms) {
    partialHistogram.assign(histogram.size(), 0);
  }
  int64_t i = 0;
  for (; i <= numLevels - kUnroll; i += kUnroll) {
    for (int j = 0; j < kUnroll; ++j) {
      ++partialHistograms[j][levels[i + j]];
    }
  }
  for (; i < numLevels; ++i) {
    ++partialHistograms[0][levels[i]];
  }
  for (const auto& partialHistogram : partialHistograms) {
    mergeLevelHistogram(histogram, partialHistogram);
  }
}
} // namespace facebook::velox::parquet::arrow
