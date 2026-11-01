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

#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions {

/// Returns the number of embeddings in a vector index.
template <typename TExec>
struct VectorIndexCardinalityFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(int64_t& result, const arg_type<VectorIndex>& index) {
    result = common::vectorindex::LocalVectorIndex::size(
        std::string_view(index.data(), index.size()));
  }
};

} // namespace facebook::velox::functions
