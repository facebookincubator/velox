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

#include <memory>

#include "velox/dwio/common/MetadataFilter.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/type/Filter.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector {

class FileScanRequirements;

/// Mutable scan state. FileDataSource reuses its state across serial splits.
struct FileScanState {
  /// Owns the immutable requirements and handles borrowed by the reader.
  std::shared_ptr<const FileScanRequirements> scanRequirements;

  /// Filters from the table handle and extracted from the remaining expression,
  /// also used for split validation. Filter objects can be shared with the
  /// table handle and other data sources and must not be modified in place.
  /// Dynamic filters are added to scanSpec, not to this map.
  common::SubfieldFilters filters;

  /// Mutable column access tree, including file constants, filters, and
  /// cross-split selectivity statistics.
  std::shared_ptr<common::ScanSpec> scanSpec;

  /// Optional statistics filter prepared for the remaining expression.
  std::shared_ptr<common::MetadataFilter> metadataFilter;

  /// Actual reader output type when native extraction changes the physical
  /// schema type. Null when no separate produced type is needed.
  RowTypePtr readerProducedType;
};

} // namespace facebook::velox::connector
