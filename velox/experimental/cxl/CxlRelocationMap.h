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

#include <algorithm>
#include <utility>
#include <vector>

#include "velox/exec/RowContainer.h"

namespace facebook::velox::cxl {

/// Address translator over the affine pieces returned by
/// RowContainer::relocateTo. Maps a moved row to its new address, or returns it
/// unchanged. Pieces are sorted by source address, so translate() binary-
/// searches a handful of entries even when millions of rows moved.
class RowRelocationMap {
 public:
  explicit RowRelocationMap(std::vector<exec::RowRelocation> pieces)
      : pieces_(std::move(pieces)) {}

  bool empty() const {
    return pieces_.empty();
  }

  size_t numPieces() const {
    return pieces_.size();
  }

  char* translate(char* from) const {
    auto it = std::upper_bound(
        pieces_.begin(),
        pieces_.end(),
        from,
        [](char* key, const exec::RowRelocation& piece) {
          return key < piece.srcBegin;
        });
    if (it != pieces_.begin()) {
      --it;
      if (from <= it->srcLast) {
        return from + it->delta;
      }
    }
    return from;
  }

 private:
  std::vector<exec::RowRelocation> pieces_;
};

} // namespace facebook::velox::cxl
