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

#include "PrefixSortAlgorithm.h"
#include "RowContainer.h"
#include "string.h"
#include "velox/common/memory/Allocation.h"
#include "velox/common/memory/AllocationPool.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/common/memory/MemoryAllocator.h"

namespace facebook::velox::exec {

struct PrefixSortLayout {
  PrefixSortLayout(
      const std::vector<TypePtr>& types,
      const std::vector<CompareFlags>& keyCompareFlags)
      : keySize(0),
        numPrefixKeys_(0),
        numSortKeys_(types.size()),
        keyCompareFlags_(keyCompareFlags) {
    VELOX_CHECK(types.size() > 0);
    for (TypePtr type : types) {
      if (type->kind() == TypeKind::BIGINT) {
        numPrefixKeys_++;
        keySize += sizeof(TypeTraits<TypeKind::BIGINT>::NativeType);
      } else {
        break;
      }
    }
    entrySize = keySize + sizeof(char*);
    if (numPrefixKeys_ < numSortKeys_) {
      needSortData = true;
    }
  }

  // prefix size is fixed.
  uint32_t keySize;
  uint32_t entrySize;
  int32_t numPrefixKeys_;
  const int32_t numSortKeys_;
  std::vector<CompareFlags> keyCompareFlags_;
  bool needSortData = false;
};

class PrefixSort {
 public:
  PrefixSort(
      RowContainer* FOLLY_NONNULL rowContainer,
      const std::vector<CompareFlags>& keyCompareFlags,
      size_t numInputRows)
      : sortLayout_(rowContainer->keyTypes(), keyCompareFlags),
        rowContainer_(rowContainer),
        numInputRows_(numInputRows) {}

  // Implement the prepare and sort methods separately to
  // facilitate the collection of metrics.
  void preparePrefix();

  void sort(std::vector<char*>& rows);

  int compare(const PrefixSortIterator& left, const PrefixSortIterator& right);

 private:
  void extractRowToPrefix(char* row, char* prefix);

  template <TypeKind typeKind>
  inline void rowToPrefix(
      uint32_t index,
      const RowColumn& rowColumn,
      char* FOLLY_NONNULL row,
      char* FOLLY_NONNULL prefix) {
    VELOX_UNSUPPORTED("prefix sort not support the type.");
  }

  uint32_t prefixKeySize(const TypeKind& typeKind) {
    if (typeKind == TypeKind::BIGINT) {
      return sizeof(TypeTraits<TypeKind::BIGINT>::NativeType);
    }
    // TODO support varchar later
    VELOX_UNSUPPORTED("prefix sort not support the type.");
  }

  inline char* getAddressFromPrefix(const PrefixSortIterator& iter) {
    return *reinterpret_cast<char**>((*iter) + sortLayout_.keySize);
  }

  // Store prefix and address for sort data.
  memory::ContiguousAllocation prefixAllocation;
  char* prefixes_;
  PrefixSortLayout sortLayout_;
  std::vector<int32_t> prefixOffsets;
  RowContainer* rowContainer_;
  size_t numInputRows_;
};

template <>
inline void PrefixSort::rowToPrefix<TypeKind::BIGINT>(
    uint32_t index,
    const RowColumn& rowColumn,
    char* FOLLY_NONNULL row,
    char* FOLLY_NONNULL prefix) {
  using T = TypeTraits<TypeKind::BIGINT>::NativeType;
  // store null as min/max value according compare flags.
  if (RowContainer::isNullAt(row, rowColumn.nullByte(), rowColumn.nullMask())) {
    CompareFlags compareFlags = sortLayout_.keyCompareFlags_[index];
    EncodeData(
        prefix + prefixOffsets[index],
        ((compareFlags.ascending && compareFlags.nullsFirst) ||
         (!compareFlags.ascending && !compareFlags.nullsFirst))
            ? std::numeric_limits<T>::min()
            : std::numeric_limits<T>::max());
  } else {
    EncodeData(
        prefix + prefixOffsets[index],
        *(reinterpret_cast<T*>(row + rowColumn.offset())));
  }
  // invert bits if desc
  if (!sortLayout_.keyCompareFlags_[index].ascending) {
    for (idx_t s = 0; s < sizeof(T); s++) {
      *(prefix + prefixOffsets[index] + s) =
          ~*(prefix + prefixOffsets[index] + s);
    }
  }
}
} // namespace facebook::velox::exec
