/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include <string>
#include <vector>

namespace facebook::velox::aggregate::prestosql {

/// Registers the vector_index_build aggregate:
///
///   (embedding ARRAY(REAL), id BIGINT, index_type VARCHAR, metric VARCHAR,
///    options MAP(VARCHAR, VARCHAR)) -> VARBINARY
///
/// Builds a local index over the group's rows (see LocalVectorIndex).
///  - Embeddings must all have the same, non-zero length.
///  - Embeddings must not contain null, NaN or infinite elements.
///  - Rows with a null embedding or id are ignored.
///  - 'index_type' is a constant: 'flat', 'ivf_flat', 'ivf_sq' or 'hnsw_sq'.
///  - 'metric' is a constant: 'cosine'.
///  - 'options' is a constant: the vector index's settings (see
///    LocalVectorIndexOptions).
void registerVectorIndexBuildAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

} // namespace facebook::velox::aggregate::prestosql
