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

#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::hive {

/// Apply an extraction chain to a vector, producing the output vector.
/// The input vector must have a type compatible with the chain's expected
/// input type.  The output vector has the type derived by the chain.
///
/// For multiple named extractions from the same column, call this once
/// per extraction chain and assemble the results into a RowVector.
VectorPtr applyExtractionChain(
    const VectorPtr& input,
    const std::vector<ExtractionPathElementPtr>& chain,
    memory::MemoryPool* pool);

/// Configure a ScanSpec for a column that has extraction chains.
/// Analyzes the extraction chains and marks unneeded sub-streams as
/// constant null so the reader can skip them (DWRF/Nimble pushdown).
///
/// This is an optimization — correctness is guaranteed by the
/// post-read extraction in applyExtractionChain even without these hints.
void configureExtractionScanSpec(
    const TypePtr& hiveType,
    const std::vector<NamedExtraction>& extractions,
    common::ScanSpec& spec,
    memory::MemoryPool* pool);

} // namespace facebook::velox::connector::hive
