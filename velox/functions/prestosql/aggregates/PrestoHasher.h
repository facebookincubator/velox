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

#include "velox/vector/DecodedVector.h"

namespace facebook::velox::aggregate {

class PrestoHasher {
 public:
  explicit PrestoHasher(
      const BaseVector& vector,
      const SelectivityVector& rows) {
    vector_ = std::make_shared<DecodedVector>(vector, rows);
    createChildren();
  }

  /// Function that computes a Presto equivalent hash.
  /// Takes SelectivityVector of rows to be hashed , These rows must be a subset
  /// of the rows passed to the constructor. The resultant hashes are saved in
  /// BufferPtr hashes, which must have as much capacity as sizeof(int64_t) *
  /// rows.end().
  void hash(const SelectivityVector& rows, BufferPtr& hashes);

 private:
  template <TypeKind kind>
  void hash(const SelectivityVector& rows, BufferPtr& hashes);

  void createChildren();

  std::shared_ptr<DecodedVector> vector_;
  std::vector<std::unique_ptr<PrestoHasher>> children_;
};

} // namespace facebook::velox::aggregate
