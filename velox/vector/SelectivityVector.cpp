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
#include "velox/vector/SelectivityVector.h"

#include "velox/common/base/Nulls.h"

namespace facebook::velox {

// static
const SelectivityVector& SelectivityVector::empty() {
  static SelectivityVector kEmpty{SelectivityVector(0, false)};
  return kEmpty;
}

SelectivityVector SelectivityVector::empty(vector_size_t size) {
  return SelectivityVector{size, false};
}

std::string SelectivityVector::toString(
    vector_size_t maxSelectedRowsToPrint) const {
  const auto selectedCnt = countSelected();

  VELOX_CHECK_GE(maxSelectedRowsToPrint, 0);

  std::stringstream out;
  out << selectedCnt << " out of " << size() << " rows selected between "
      << begin() << " and " << end();

  if (selectedCnt > 0 && maxSelectedRowsToPrint > 0) {
    out << ": ";
    int cnt = 0;
    testSelected([&](auto row) {
      if (cnt > 0) {
        out << ", ";
      }
      out << row;
      ++cnt;
      return cnt < maxSelectedRowsToPrint;
    });
  }
  return out.str();
}

void translateToInnerRows(
    const SelectivityVector& outerRows,
    const vector_size_t* indices,
    const uint64_t* nulls,
    SelectivityVector& innerRows) {
  outerRows.applyToSelected([&](vector_size_t row) {
    if (!(nulls && bits::isBitNull(nulls, row))) {
      innerRows.setValid(indices[row], true);
    }
  });
  innerRows.updateBounds();
}

} // namespace facebook::velox
