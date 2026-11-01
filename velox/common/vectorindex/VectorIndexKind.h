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

#include <cstdint>

#include "velox/common/EnumDeclare.h"

namespace facebook::velox::common::vectorindex {

/// Kind of a local index (see LocalVectorIndex): the FAISS index that holds
/// its embeddings. The values are stored in intermediate results, so new types
/// take new values.
enum class VectorIndexKind {
  /// Every embedding stored in full; a search compares the query with each
  /// one.
  kFlat = 0,
  /// The embeddings clustered into 'ivf.nlist' lists; a search scans the
  /// 'ivf.nprobe' lists nearest to the query.
  kIvfFlat = 1,
  /// As kIvfFlat, with each dimension compressed to 8 bits.
  kIvfSq = 2,
  /// A layered graph of near neighbors over embeddings with each dimension
  /// compressed to 8 bits; a search walks it keeping 'hnsw.efSearch'
  /// candidates.
  kHnswSq = 3,
};

VELOX_DECLARE_ENUM_NAME(VectorIndexKind);

/// The integer form of VectorIndexKind, stored in intermediate results.
class VectorIndexKindValue {
 public:
  static int32_t toValue(VectorIndexKind kind) {
    return static_cast<int32_t>(kind);
  }

  /// Throws a user error for a value that names no index kind.
  static VectorIndexKind fromValue(int32_t value);
};

} // namespace facebook::velox::common::vectorindex
