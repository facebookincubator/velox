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

/// Mutable connector-internal scan preparation for one active physical reader.
/// FileScanSpec::newFileScanState() creates an independent instance using the
/// reader's context and memory pool. Readers may share an immutable
/// FileScanSpec, but concurrently active readers must use separate states.
///
/// Owns cloned filters and the DWIO ScanSpec that readers adapt with file
/// constants, filter ordering, and caches. MetadataFilter is bound to this
/// exact tree; changes to physical column demands require rebuilding the state
/// so metadata filtering and extraction continue to describe the same inputs.
///
/// A data source and its adapter may jointly own this state for one physical
/// reader, including serial takeover of a preloaded data source. Joint
/// ownership keeps borrowed inputs alive and does not make mutation thread
/// safe. Release the physical reader before releasing its state and the
/// FileScanSpec backing its immutable inputs. This object owns neither the
/// physical reader nor the FileScanSpec or query context.
struct FileScanState {
  /// Clones of the physical predicates, also used for split validation.
  common::SubfieldFilters filters;

  /// Mutable column access tree for the physical reader.
  std::shared_ptr<common::ScanSpec> scanSpec;

  /// Optional statistics filter whose leaves belong to scanSpec.
  std::shared_ptr<common::MetadataFilter> metadataFilter;

  /// Actual reader output type when native extraction changes the physical
  /// schema type. Null when no separate produced type is needed.
  RowTypePtr readerProducedType;
};

} // namespace facebook::velox::connector::hive
