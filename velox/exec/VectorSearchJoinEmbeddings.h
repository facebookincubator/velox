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

#include <span>

#include "velox/common/vectorindex/VectorIndexMetric.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::exec {

/// The embeddings of an ARRAY(REAL) column of query rows of a vector search
/// join. Every non-null embedding has dimension() elements, none of them null,
/// NaN or infinite.
class VectorSearchJoinEmbeddings {
 public:
  /// Decodes and checks 'column', whose embeddings are compared under
  /// 'metric'. Throws a user error if an embedding is empty, differs in length
  /// from another, or has a null, NaN or infinite element.
  VectorSearchJoinEmbeddings(
      const BaseVector& column,
      common::vectorindex::VectorIndexMetric metric);

  /// The length of the non-null embeddings. 0 if all are null.
  int32_t dimension() const {
    return dimension_;
  }

  /// Deselects from 'rows' the rows with a null embedding.
  void deselectNulls(SelectivityVector& rows) const {
    if (nulls_ != nullptr) {
      rows.deselectNulls(nulls_, rows.begin(), rows.end());
    }
  }

  /// Copies the embedding of non-null 'row', normalized for the metric, into
  /// 'query', which has dimension() elements.
  void copyNormalized(vector_size_t row, std::span<float> query) const;

 private:
  const common::vectorindex::VectorIndexMetric metric_;
  DecodedVector embeddings_;
  // The nulls of 'embeddings_', or null if there are none.
  const uint64_t* nulls_{nullptr};
  // The arrays under 'embeddings_', and their elements. Null and empty if all
  // embeddings are null.
  const ArrayVector* arrays_{nullptr};
  DecodedVector elements_;
  int32_t dimension_{0};
};

} // namespace facebook::velox::exec
