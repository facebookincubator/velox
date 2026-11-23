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

/// Registers the vector_count aggregate:
///
///   (index VECTOR_INDEX) -> BIGINT
///
/// Returns the number of embeddings in the group's vector indexes: the sum of
/// cardinality(index). Rows with a null index are ignored. A group with no
/// non-null index gets null.
void registerVectorCountAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

} // namespace facebook::velox::aggregate::prestosql
