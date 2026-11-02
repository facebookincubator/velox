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
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "velox/dwio/parquet/writer/arrow/Platform.h"

namespace facebook::velox::parquet::arrow {
class ColumnDescriptor;

/// Statistics for estimating the unencoded, uncompressed size of a column chunk
/// or data page.
struct PARQUET_EXPORT SizeStatistics {
  /// Counts by definition level, indexed by level. Empty if not collected;
  /// otherwise has maxDefinitionLevel + 1 entries.
  std::vector<int64_t> definitionLevelHistogram;

  /// Counts by repetition level, indexed by level. Empty if not collected;
  /// otherwise has maxRepetitionLevel + 1 entries.
  std::vector<int64_t> repetitionLevelHistogram;

  /// Unencoded BYTE_ARRAY payload bytes, excluding length prefixes.
  /// Unset for other types or when byte counts are not collected.
  std::optional<int64_t> unencodedByteArrayDataBytes;

  /// Returns whether any histogram or byte count is set.
  bool isSet() const;

  /// Adds value to the BYTE_ARRAY byte count, which must already be set.
  void incrementUnencodedByteArrayDataBytes(int64_t value);

  /// Merges counts from other, which must have matching histogram sizes and
  /// byte count presence.
  void merge(const SizeStatistics& other);

  /// Zeros counts while preserving histogram sizes and byte count presence.
  void reset();

  /// Throws if set statistics do not match the column descriptor.
  void validate(const ColumnDescriptor* descriptor) const;

  /// Creates zeroed statistics for descriptor, omitting histograms for maximum
  /// levels of zero and byte counts for non-BYTE_ARRAY columns.
  static std::unique_ptr<SizeStatistics> make(
      const ColumnDescriptor* descriptor);
};

/// Accumulates level counts into a non-empty histogram. Each level must be in
/// [0, histogram.size()).
PARQUET_EXPORT void updateLevelHistogram(
    std::span<const int16_t> levels,
    std::span<int64_t> histogram);
} // namespace facebook::velox::parquet::arrow
