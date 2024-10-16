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

#include <stdint.h>

namespace facebook::velox::common {

/// Specifies the config for prefix-sort.
struct PrefixSortConfig {
  PrefixSortConfig() = default;

  PrefixSortConfig(
      int64_t _maxNormalizedKeySize,
      int32_t _threshold,
      double _minAllowedAvgToMaxLenRatio)
      : maxNormalizedKeySize(_maxNormalizedKeySize),
        threshold(_threshold),
        minAllowedAvgToMaxLenRatio(_minAllowedAvgToMaxLenRatio) {}

  /// Max number of bytes can store normalized keys in prefix-sort buffer per
  /// entry. Same with QueryConfig kPrefixSortNormalizedKeyMaxBytes.
  int64_t maxNormalizedKeySize{128};

  /// PrefixSort will have performance regression when the dateset is too small.
  int32_t threshold{130};

  /// Minimum allowed ratio between the average and maximum length for a
  /// variable-length column in prefix sorting. Only variable-length columns
  /// with an average-to-maximum length ratio greater than this value will be
  /// stored in prefix-sort buffer.
  double minAllowedAvgToMaxLenRatio{0.3};
};
} // namespace facebook::velox::common
