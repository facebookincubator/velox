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

struct PARQUET_EXPORT SizeStatistics {
  std::vector<int64_t> definitionLevelHistogram;
  std::vector<int64_t> repetitionLevelHistogram;
  std::optional<int64_t> unencodedByteArrayDataBytes;

  bool isSet() const;
  void incrementUnencodedByteArrayDataBytes(int64_t value);
  void merge(const SizeStatistics& other);
  void reset();
  void validate(const ColumnDescriptor* descriptor) const;
  static std::unique_ptr<SizeStatistics> make(
      const ColumnDescriptor* descriptor);
};

PARQUET_EXPORT void updateLevelHistogram(
    std::span<const int16_t> levels,
    std::span<int64_t> histogram);
} // namespace facebook::velox::parquet::arrow
