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

#include "velox/vector/SimpleVector.h"
#include "velox/functions/lib/string/StringCore.h"

namespace facebook::velox {

template <>
void SimpleVector<StringView>::validate(
    const VectorValidateOptions& options) const {
  BaseVector::validate(options);

  // We only validate the right size of ascii info, if it has any selection.
  auto rlockedAsciiComputedRows{asciiInfo.readLockedAsciiComputedRows()};
  if (rlockedAsciiComputedRows->hasSelections()) {
    VELOX_CHECK_GE(rlockedAsciiComputedRows->size(), size());
  }
}

template <>
template <>
bool SimpleVector<StringView>::computeAndSetIsAscii<StringView>(
    const SelectivityVector& rows) {
  if (rows.isSubset(*asciiInfo.readLockedAsciiComputedRows())) {
    return asciiInfo.isAllAscii();
  }
  ensureIsAsciiCapacity();
  bool isAllAscii = true;
  rows.applyToSelected([&](auto row) {
    if (!isNullAt(row)) {
      auto string = valueAt(row);
      isAllAscii &=
          functions::stringCore::isAscii(string.data(), string.size());
    }
  });

  auto wlockedAsciiComputedRows = asciiInfo.writeLockedAsciiComputedRows();
  if (!wlockedAsciiComputedRows->hasSelections()) {
    asciiInfo.setIsAllAscii(isAllAscii);
  } else {
    asciiInfo.setIsAllAscii(asciiInfo.isAllAscii() & isAllAscii);
  }

  wlockedAsciiComputedRows->select(rows);
  asciiInfo.setAsciiComputedRowsEmpty(
      !wlockedAsciiComputedRows->hasSelections());
  return asciiInfo.isAllAscii();
}

} // namespace facebook::velox
