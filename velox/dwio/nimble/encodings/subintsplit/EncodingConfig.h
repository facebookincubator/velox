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

#include <map>
#include <string>
#include <vector>

#include "velox/dwio/nimble/encodings/common/EncodingType.h"

namespace facebook::nimble::subintsplit {

/// Describes the split boundaries and ordered child encodings for one column.
struct ColumnEncodingConfig {
  /// Covers every bit of the column's physical type in LSB-first order.
  std::string boundaries;

  /// Selects one encoding for each boundary section in the same order.
  std::vector<EncodingType> childEncodings;

  bool operator==(const ColumnEncodingConfig&) const = default;
};

/// Groups SubIntSplit layouts by caller-defined column name.
struct EncodingConfig {
  /// Stores independently configurable layouts for selected columns.
  std::map<std::string, ColumnEncodingConfig> columns;

  bool operator==(const EncodingConfig&) const = default;
};

} // namespace facebook::nimble::subintsplit
