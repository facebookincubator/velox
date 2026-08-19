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

#include <folly/container/F14Set.h>

#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

namespace facebook::nimble {

velox::TypePtr convertToVeloxType(const Type& type);

std::shared_ptr<const Type> convertToNimbleType(const velox::Type& type);

/// Encoding types for top-level columns extracted from a nimble file schema.
/// Records which columns use encoding-specific types (ArrayWithOffsets,
/// SlidingWindowMap, FlatMap), enabling the Serializer to produce output
/// consistent with the file's encoding.
struct ColumnEncodings {
  folly::F14FastSet<std::string> dictionaryArrayColumns;
  folly::F14FastSet<std::string> deduplicatedMapColumns;
  folly::F14FastSet<std::string> flatMapColumns;
};

/// Converts a projected velox RowType to a nimble schema, including only the
/// projected columns. Supports arbitrary-depth subfield projections:
/// - Full column projection: subfield is just the column name (e.g., "col").
/// - FlatMap key projection: subfield has a subscript (e.g., "map[\"key\"]").
/// - Row child projection: subfield has a nested field (e.g., "struct.field").
/// - Deep projection: "struct.nested_struct.field", "map[\"key\"].field", etc.
///
/// When columnEncodings is provided, creates encoding-specific nimble types
/// (ArrayWithOffsets, SlidingWindowMap, FlatMap) to match data serialized with
/// these encodings. When empty (default), uses plain nimble types.
std::shared_ptr<const Type> buildProjectedNimbleType(
    const velox::RowType& type,
    const std::vector<velox::common::Subfield>& projectedSubfields,
    const ColumnEncodings& columnEncodings = {});

/// Contains a projected Nimble schema and its mapping to source streams.
struct NimbleTypeProjection {
  std::shared_ptr<const Type> nimbleType;
  std::vector<uint32_t> streamOffsets;
  std::vector<bool> rowOrFlatMapNullStreams;
};

/// Builds a projected Nimble schema and its source-stream mapping.
NimbleTypeProjection buildProjectedNimbleType(
    const Type* type,
    const std::vector<velox::common::Subfield>& projectedSubfields);

} // namespace facebook::nimble
