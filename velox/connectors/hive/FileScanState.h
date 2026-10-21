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

namespace facebook::velox::connector::hive {

/// Mutable scan preparation owned by a FileDataSource. Serial splits reuse
/// this state and its ScanSpec, rebuilding the tree only when their physical
/// column requirements change. Different data sources own separate states.
///
/// A preloaded reader and its state are transferred together so borrowed
/// validation filters keep their address after the original source is gone.
/// The reader must be released before its state. This object does not own the
/// reader, column handles, query context, or remaining-filter ExprSet.
struct FileScanState {
  /// Original and extracted subfield predicates, also used for split
  /// validation. Dynamic filters are installed on scanSpec, not in this map.
  /// Published filter payloads are shared and must not be mutated in place.
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

} // namespace facebook::velox::connector::hive
