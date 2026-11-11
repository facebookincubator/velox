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
#include "velox/serializers/PrestoSerializerEstimationUtils.h"

#include <algorithm>
#include <array>
#include <limits>

#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::serializer::presto::detail {
namespace {
template <TypeKind Kind>
void estimateFlattenedConstantSerializedSize(
    const BaseVector* vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    Scratch& scratch) {
  VELOX_CHECK_EQ(vector->encoding(), VectorEncoding::Simple::CONSTANT);
  using T = typename KindToFlatVector<Kind>::WrapperType;
  auto* constantVector = vector->as<ConstantVector<T>>();
  if (constantVector->valueVector()) {
    estimateWrapperSerializedSize(ranges, sizes, vector, scratch);
    return;
  }

  int32_t elementSize = sizeof(T);
  if (constantVector->isNullAt(0)) {
    elementSize = 1;
  } else if constexpr (std::is_same_v<T, StringView>) {
    elementSize = saturatedSize(constantVector->valueAt(0).size());
  }
  for (int32_t i = 0; i < ranges.size(); ++i) {
    addToSize(sizes[i], saturatedProduct(elementSize, ranges[i].size));
  }
}

void estimateBiasedSerializedSize(
    const BaseVector* vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes) {
  auto valueSize = static_cast<vector_size_t>(vector->type()->cppSizeInBytes());
  if (vector->mayHaveNulls()) {
    auto rawNulls = vector->rawNulls();
    for (int32_t i = 0; i < ranges.size(); ++i) {
      auto end = ranges[i].begin + ranges[i].size;
      int32_t numValues = bits::countBits(rawNulls, ranges[i].begin, end);
      addToSize(
          sizes[i],
          static_cast<uint64_t>(numValues) * valueSize +
              bits::nbytes(ranges[i].size));
    }
  } else {
    for (int32_t i = 0; i < ranges.size(); ++i) {
      addToSize(sizes[i], saturatedProduct(ranges[i].size, valueSize));
    }
  }
}

template <TypeKind Kind>
void estimateFlatSerializedSize(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  const auto valueSize =
      static_cast<vector_size_t>(vector->type()->cppSizeInBytes());
  const auto numRows = rows.size();
  if (vector->mayHaveNulls()) {
    auto rawNulls = vector->rawNulls();
    ScratchPtr<uint64_t, 4> nullsHolder(scratch);
    ScratchPtr<int32_t, 64> nonNullsHolder(scratch);
    auto nulls = nullsHolder.get(bits::nwords(numRows));
    simd::gatherBits(rawNulls, rows, nulls);
    auto nonNulls = nonNullsHolder.get(numRows);
    const auto numNonNull = simd::indicesOfSetBits(nulls, 0, numRows, nonNulls);
    for (int32_t i = 0; i < numNonNull; ++i) {
      addToSize(sizes[nonNulls[i]], valueSize);
    }
  } else {
    VELOX_UNREACHABLE("Non null fixed width case handled before this");
  }
}

void estimateFlatSerializedSizeVarcharOrVarbinary(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  const auto numRows = rows.size();
  auto strings = static_cast<const FlatVector<StringView>*>(vector);
  auto rawNulls = strings->rawNulls();
  auto rawValues = strings->rawValues();
  if (!rawNulls) {
    for (auto i = 0; i < rows.size(); ++i) {
      // Add the size of the length and the string data.
      addToSize(
          sizes[i],
          static_cast<uint64_t>(sizeof(int32_t) + rawValues[rows[i]].size()));
    }
  } else {
    for (auto i = 0; i < numRows; ++i) {
      addToSize(sizes[i], kInt32Bytes);
    }

    ScratchPtr<uint64_t, 4> nullsHolder(scratch);
    ScratchPtr<int32_t, 64> nonNullsHolder(scratch);
    auto nulls = nullsHolder.get(bits::nwords(numRows));
    simd::gatherBits(rawNulls, rows, nulls);
    auto* nonNulls = nonNullsHolder.get(numRows);
    auto numNonNull = simd::indicesOfSetBits(nulls, 0, numRows, nonNulls);

    for (int32_t i = 0; i < numNonNull; ++i) {
      addToSize(
          sizes[nonNulls[i]],
          static_cast<uint64_t>(rawValues[rows[nonNulls[i]]].size()));
    }
  }
}

template <>
void estimateFlatSerializedSize<TypeKind::VARCHAR>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  estimateFlatSerializedSizeVarcharOrVarbinary(vector, rows, sizes, scratch);
}

template <>
void estimateFlatSerializedSize<TypeKind::VARBINARY>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  estimateFlatSerializedSizeVarcharOrVarbinary(vector, rows, sizes, scratch);
}

template <>
void estimateFlatSerializedSize<TypeKind::OPAQUE>(
    const BaseVector*,
    const folly::Range<const vector_size_t*>&,
    vector_size_t**,
    Scratch&) {
  VELOX_FAIL("Opaque type support is not implemented.");
}

template <TypeKind Kind>
void estimateFlattenedConstantSerializedSize(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  VELOX_CHECK_EQ(vector->encoding(), VectorEncoding::Simple::CONSTANT);

  using T = typename KindToFlatVector<Kind>::WrapperType;
  auto* constantVector = vector->as<ConstantVector<T>>();
  int32_t elementSize = vector->valueVector() ? 0 : sizeof(T);
  if (constantVector->isNullAt(0)) {
    // A flattened null string still has a length.
    elementSize = std::is_same_v<T, StringView> ? sizeof(int32_t) : 0;
  } else if (vector->valueVector()) {
    const auto* values = constantVector->wrappedVector();
    vector_size_t* sizePtr = &elementSize;
    const vector_size_t singleRow = constantVector->wrappedIndex(0);
    estimateSerializedSizeInt(
        values,
        folly::Range<const vector_size_t*>(&singleRow, 1),
        &sizePtr,
        scratch);
  } else if constexpr (std::is_same_v<T, StringView>) {
    // Add the size of the length and the string data.
    elementSize =
        saturatedSize(sizeof(int32_t) + constantVector->valueAt(0).size());
  }
  for (int32_t i = 0; i < rows.size(); ++i) {
    addToSize(sizes[i], elementSize);
  }
}

void estimateWrapperSerializedSize(
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    const BaseVector* wrapper,
    Scratch& scratch) {
  ScratchPtr<vector_size_t, 1> innerRowsHolder(scratch);
  ScratchPtr<vector_size_t*, 1> innerSizesHolder(scratch);
  const int32_t numRows = rows.size();
  int32_t numInner = 0;
  auto* innerRows = innerRowsHolder.get(numRows);
  auto* innerSizes = sizes;
  const BaseVector* wrapped;
  if (wrapper->encoding() == VectorEncoding::Simple::DICTIONARY &&
      !wrapper->rawNulls()) {
    // Dictionary with no nulls.
    auto* indices = wrapper->wrapInfo()->as<vector_size_t>();
    wrapped = wrapper->valueVector().get();
    simd::transpose(indices, rows, innerRows);
    numInner = numRows;
  } else {
    wrapped = wrapper->wrappedVector();
    innerSizes = innerSizesHolder.get(numRows);
    for (int32_t i = 0; i < rows.size(); ++i) {
      if (!wrapper->isNullAt(rows[i])) {
        innerRows[numInner] = wrapper->wrappedIndex(rows[i]);
        innerSizes[numInner] = sizes[i];
        ++numInner;
      }
    }
  }
  if (numInner == 0) {
    return;
  }

  estimateSerializedSizeInt(
      wrapped,
      folly::Range<const vector_size_t*>(innerRows, numInner),
      innerSizes,
      scratch);
}

void estimateBiasedSerializedSize(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes) {
  const auto valueSize =
      static_cast<vector_size_t>(vector->type()->cppSizeInBytes());
  if (!vector->mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      addToSize(sizes[i], valueSize);
    }
    return;
  }

  for (auto i = 0; i < rows.size(); ++i) {
    if (!vector->isNullAt(rows[i])) {
      addToSize(sizes[i], valueSize);
    }
  }
}

// Non-null rows of a collection and their size accumulators.
struct ParentRows {
  const vector_size_t* rows;
  vector_size_t** sizes;
  int32_t numRows;
};

// Returns the non-null 'rows' of a collection with 'rawNulls' and adds the
// 4-byte length of each. Uses the holders only when some rows are null.
ParentRows collectNonNullRows(
    const folly::Range<const vector_size_t*>& rows,
    const uint64_t* rawNulls,
    vector_size_t** sizes,
    ScratchPtr<vector_size_t, 64>& nonNullRowsHolder,
    ScratchPtr<vector_size_t*, 64>& nonNullSizesHolder,
    Scratch& scratch) {
  const auto numRows = static_cast<int32_t>(rows.size());
  ParentRows parents{rows.data(), sizes, numRows};
  if (rawNulls != nullptr) {
    ScratchPtr<uint64_t, 4> nullsHolder(scratch);
    ScratchPtr<vector_size_t, 64> positionsHolder(scratch);
    auto* nulls = nullsHolder.get(static_cast<int32_t>(bits::nwords(numRows)));
    simd::gatherBits(rawNulls, rows, nulls);
    auto* positions = positionsHolder.get(numRows);
    const auto numNonNull =
        simd::indicesOfSetBits(nulls, 0, numRows, positions);
    auto* nonNullRows = nonNullRowsHolder.get(numNonNull);
    auto* nonNullSizes = nonNullSizesHolder.get(numNonNull);
    for (auto i = 0; i < numNonNull; ++i) {
      nonNullRows[i] = rows[positions[i]];
      nonNullSizes[i] = sizes[positions[i]];
    }
    parents = {nonNullRows, nonNullSizes, numNonNull};
  }
  for (auto i = 0; i < parents.numRows; ++i) {
    addToSize(parents.sizes[i], kInt32Bytes);
  }
  return parents;
}

// Returns the total number of elements of 'rows' of a collection.
int64_t countElements(
    const vector_size_t* rows,
    int32_t numRows,
    const vector_size_t* rawLengths) {
  int64_t numElements{0};
  for (auto i = 0; i < numRows; ++i) {
    numElements += rawLengths[rows[i]];
  }
  return numElements;
}

// Most element rows estimateElementsSerializedSize() expands at a time.
constexpr int32_t kMaxElementRowsPerBatch{1 << 16};

// Estimates the 'totalElements' elements of 'parents' of a collection into the
// parents' size accumulators. A fixed-width child adds its width per non-null
// element and a flat string child 4 bytes plus the string per element, which
// the element ranges give without expanding them. Other children get the
// element rows a bounded batch at a time. A batch can split a parent because
// per-row estimates have no per-batch part.
void estimateElementsSerializedSize(
    const ParentRows& parents,
    int64_t totalElements,
    const vector_size_t* rawOffsets,
    const vector_size_t* rawLengths,
    std::initializer_list<const BaseVector*> children,
    Scratch& scratch) {
  if (totalElements == 0) {
    return;
  }
  std::array<const BaseVector*, 2> expandedChildren;
  int32_t numExpandedChildren{0};
  for (const auto* child : children) {
    const auto& type = *child->type();
    if (type.isFixedWidth() && !child->mayHaveNullsRecursive()) {
      const auto width = static_cast<vector_size_t>(type.cppSizeInBytes());
      for (auto i = 0; i < parents.numRows; ++i) {
        addToSize(
            parents.sizes[i],
            saturatedProduct(rawLengths[parents.rows[i]], width));
      }
    } else if (
        type.isFixedWidth() &&
        child->encoding() == VectorEncoding::Simple::FLAT) {
      const auto width = static_cast<vector_size_t>(type.cppSizeInBytes());
      const auto* rawNulls = child->rawNulls();
      for (auto i = 0; i < parents.numRows; ++i) {
        const auto begin = rawOffsets[parents.rows[i]];
        const auto numNonNull = bits::countBits(
            rawNulls, begin, begin + rawLengths[parents.rows[i]]);
        addToSize(parents.sizes[i], saturatedProduct(numNonNull, width));
      }
    } else if (
        child->encoding() == VectorEncoding::Simple::FLAT &&
        (type.kind() == TypeKind::VARCHAR ||
         type.kind() == TypeKind::VARBINARY)) {
      const auto* rawValues =
          child->asUnchecked<FlatVector<StringView>>()->rawValues();
      const auto* rawNulls = child->rawNulls();
      for (auto i = 0; i < parents.numRows; ++i) {
        const auto begin = rawOffsets[parents.rows[i]];
        const auto end = begin + rawLengths[parents.rows[i]];
        uint64_t bytes{0};
        for (auto row = begin; row < end; ++row) {
          bytes += sizeof(int32_t);
          if (rawNulls == nullptr || !bits::isBitNull(rawNulls, row)) {
            bytes += rawValues[row].size();
          }
        }
        addToSize(parents.sizes[i], bytes);
      }
    } else {
      VELOX_CHECK_LT(numExpandedChildren, expandedChildren.size());
      expandedChildren[numExpandedChildren++] = child;
    }
  }
  if (numExpandedChildren == 0) {
    return;
  }

  const auto batchSize = static_cast<int32_t>(
      std::min<int64_t>(totalElements, kMaxElementRowsPerBatch));
  ScratchPtr<vector_size_t> elementRowsHolder(scratch);
  ScratchPtr<vector_size_t*> elementSizesHolder(scratch);
  auto* elementRows = elementRowsHolder.get(batchSize);
  auto* elementSizes = elementSizesHolder.get(batchSize);
  int32_t numFilled{0};
  auto estimateBatch = [&]() {
    const folly::Range<const vector_size_t*> batch(elementRows, numFilled);
    for (auto i = 0; i < numExpandedChildren; ++i) {
      estimateSerializedSizeInt(
          expandedChildren[i], batch, elementSizes, scratch);
    }
    numFilled = 0;
  };
  for (auto i = 0; i < parents.numRows; ++i) {
    const auto begin = rawOffsets[parents.rows[i]];
    const auto end = begin + rawLengths[parents.rows[i]];
    for (auto elementRow = begin; elementRow < end; ++elementRow) {
      elementRows[numFilled] = elementRow;
      elementSizes[numFilled] = parents.sizes[i];
      if (++numFilled == batchSize) {
        estimateBatch();
      }
    }
  }
  if (numFilled > 0) {
    estimateBatch();
  }
}

// Estimates ARRAY or MAP 'rows'. Each row serializes all of its elements, and a
// dictionary can repeat one row many times, as a join does. When per-row
// estimates would expand more elements than the collection holds, the elements
// of each distinct row are estimated once and added to every repeat, so the
// work follows the elements of the distinct rows, not rows x elements. Distinct
// rows that share elements through overlapping offsets still count them each.
void estimateCollectionSerializedSize(
    const BaseVector& collection,
    const vector_size_t* rawOffsets,
    const vector_size_t* rawLengths,
    vector_size_t numCollectionElements,
    std::initializer_list<const BaseVector*> children,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  ScratchPtr<vector_size_t, 64> nonNullRowsHolder(scratch);
  ScratchPtr<vector_size_t*, 64> nonNullSizesHolder(scratch);
  const auto parents = collectNonNullRows(
      rows,
      collection.rawNulls(),
      sizes,
      nonNullRowsHolder,
      nonNullSizesHolder,
      scratch);
  const auto totalElements =
      countElements(parents.rows, parents.numRows, rawLengths);

  if (totalElements <= numCollectionElements) {
    estimateElementsSerializedSize(
        parents, totalElements, rawOffsets, rawLengths, children, scratch);
    return;
  }

  ScratchPtr<vector_size_t, 64> distinctRowsHolder(scratch);
  auto* distinctRows = distinctRowsHolder.get(parents.numRows);
  std::copy(parents.rows, parents.rows + parents.numRows, distinctRows);
  std::sort(distinctRows, distinctRows + parents.numRows);
  const auto numDistinct = static_cast<int32_t>(
      std::unique(distinctRows, distinctRows + parents.numRows) - distinctRows);

  ScratchPtr<vector_size_t, 64> distinctSizesHolder(scratch);
  ScratchPtr<vector_size_t*, 64> distinctSizePointersHolder(scratch);
  auto* distinctSizes = distinctSizesHolder.get(numDistinct);
  auto* distinctSizePointers = distinctSizePointersHolder.get(numDistinct);
  for (auto i = 0; i < numDistinct; ++i) {
    distinctSizes[i] = 0;
    distinctSizePointers[i] = &distinctSizes[i];
  }
  estimateElementsSerializedSize(
      {distinctRows, distinctSizePointers, numDistinct},
      countElements(distinctRows, numDistinct, rawLengths),
      rawOffsets,
      rawLengths,
      children,
      scratch);

  for (auto i = 0; i < parents.numRows; ++i) {
    const auto index =
        std::lower_bound(
            distinctRows, distinctRows + numDistinct, parents.rows[i]) -
        distinctRows;
    addToSize(parents.sizes[i], distinctSizes[index]);
  }
}
} // namespace

void estimateFlatSerializedSizeVarcharOrVarbinary(
    const BaseVector* vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes) {
  auto strings = static_cast<const FlatVector<StringView>*>(vector);
  auto rawNulls = strings->rawNulls();
  auto rawValues = strings->rawValues();
  for (int32_t i = 0; i < ranges.size(); ++i) {
    auto end = ranges[i].begin + ranges[i].size;
    int32_t numNulls = 0;
    uint64_t bytes = 0;
    for (int32_t offset = ranges[i].begin; offset < end; ++offset) {
      if (rawNulls && bits::isBitNull(rawNulls, offset)) {
        ++numNulls;
      } else {
        bytes += sizeof(int32_t) + rawValues[offset].size();
      }
    }
    addToSize(sizes[i], bytes + bits::nbytes(numNulls) + 4 * numNulls);
  }
}

void estimateSerializedSizeInt(
    const BaseVector* vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    Scratch& scratch) {
  switch (vector->encoding()) {
    case VectorEncoding::Simple::FLAT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          estimateFlatSerializedSize,
          vector->typeKind(),
          vector,
          ranges,
          sizes);
      break;
    case VectorEncoding::Simple::CONSTANT:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          estimateFlattenedConstantSerializedSize,
          vector->typeKind(),
          vector,
          ranges,
          sizes,
          scratch);
      break;
    case VectorEncoding::Simple::DICTIONARY:
    case VectorEncoding::Simple::SEQUENCE:
      estimateWrapperSerializedSize(ranges, sizes, vector, scratch);
      break;
    case VectorEncoding::Simple::BIASED:
      estimateBiasedSerializedSize(vector, ranges, sizes);
      break;
    case VectorEncoding::Simple::ROW: {
      std::vector<IndexRange> childRanges;
      std::vector<vector_size_t*> childSizes;
      for (int32_t i = 0; i < ranges.size(); ++i) {
        auto begin = ranges[i].begin;
        auto end = begin + ranges[i].size;
        for (auto offset = begin; offset < end; ++offset) {
          addToSize(sizes[i], kInt32Bytes);
          if (!vector->isNullAt(offset)) {
            childRanges.push_back(IndexRange{offset, 1});
            childSizes.push_back(sizes[i]);
          }
        }
      }
      auto rowVector = vector->as<RowVector>();
      auto& children = rowVector->children();
      for (auto& child : children) {
        if (child) {
          estimateSerializedSizeInt(
              child.get(),
              folly::Range(childRanges.data(), childRanges.size()),
              childSizes.data(),
              scratch);
        }
      }
      break;
    }
    case VectorEncoding::Simple::MAP: {
      auto mapVector = vector->as<MapVector>();
      std::vector<IndexRange> childRanges;
      std::vector<vector_size_t*> childSizes;
      expandRepeatedRanges(
          mapVector,
          mapVector->rawOffsets(),
          mapVector->rawSizes(),
          ranges,
          sizes,
          &childRanges,
          &childSizes);
      estimateSerializedSizeInt(
          mapVector->mapKeys().get(), childRanges, childSizes.data(), scratch);
      estimateSerializedSizeInt(
          mapVector->mapValues().get(),
          childRanges,
          childSizes.data(),
          scratch);
      break;
    }
    case VectorEncoding::Simple::ARRAY: {
      auto arrayVector = vector->as<ArrayVector>();
      std::vector<IndexRange> childRanges;
      std::vector<vector_size_t*> childSizes;
      expandRepeatedRanges(
          arrayVector,
          arrayVector->rawOffsets(),
          arrayVector->rawSizes(),
          ranges,
          sizes,
          &childRanges,
          &childSizes);
      estimateSerializedSizeInt(
          arrayVector->elements().get(),
          childRanges,
          childSizes.data(),
          scratch);
      break;
    }
    case VectorEncoding::Simple::LAZY:
      estimateSerializedSizeInt(vector->loadedVector(), ranges, sizes, scratch);
      break;
    default:
      VELOX_UNSUPPORTED("Unsupported vector encoding {}", vector->encoding());
  }
}

void estimateWrapperSerializedSize(
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    const BaseVector* wrapper,
    Scratch& scratch) {
  std::vector<IndexRange> newRanges;
  std::vector<vector_size_t*> newSizes;
  const BaseVector* wrapped = wrapper->wrappedVector();
  if (!wrapper->mayHaveNulls()) {
    int64_t totalSize = 0;
    for (int32_t i = 0; i < ranges.size(); ++i) {
      totalSize += ranges[i].size;
    }
    newRanges.reserve(totalSize);
    newSizes.reserve(totalSize);
  }
  for (int32_t i = 0; i < ranges.size(); ++i) {
    int32_t numNulls = 0;
    auto end = ranges[i].begin + ranges[i].size;
    for (int32_t offset = ranges[i].begin; offset < end; ++offset) {
      if (!wrapper->isNullAt(offset)) {
        newRanges.push_back(IndexRange{wrapper->wrappedIndex(offset), 1});
        newSizes.push_back(sizes[i]);
      } else {
        ++numNulls;
      }
    }
    addToSize(sizes[i], static_cast<vector_size_t>(bits::nbytes(numNulls)));
  }
  estimateSerializedSizeInt(wrapped, newRanges, newSizes.data(), scratch);
}

void expandRepeatedRanges(
    const BaseVector* vector,
    const vector_size_t* rawOffsets,
    const vector_size_t* rawSizes,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    std::vector<IndexRange>* childRanges,
    std::vector<vector_size_t*>* childSizes) {
  if (!vector->mayHaveNulls()) {
    int64_t totalSize = 0;
    for (int32_t i = 0; i < ranges.size(); ++i) {
      totalSize += ranges[i].size;
    }
    childRanges->reserve(totalSize);
    childSizes->reserve(totalSize);
  }
  for (int32_t i = 0; i < ranges.size(); ++i) {
    int32_t begin = ranges[i].begin;
    int32_t end = begin + ranges[i].size;
    bool hasNull = false;
    for (int32_t offset = begin; offset < end; ++offset) {
      if (vector->isNullAt(offset)) {
        hasNull = true;
      } else {
        // Add the size of the length.
        addToSize(sizes[i], kInt32Bytes);
        childRanges->push_back(
            IndexRange{rawOffsets[offset], rawSizes[offset]});
        childSizes->push_back(sizes[i]);
      }
    }

    if (hasNull) {
      // Add the size of the null bit mask.
      addToSize(
          sizes[i], static_cast<vector_size_t>(bits::nbytes(ranges[i].size)));
    }
  }
}

void estimateSerializedSizeInt(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  const auto numRows = rows.size();
  if (vector->type()->isFixedWidth() && !vector->mayHaveNullsRecursive()) {
    const auto elementSize =
        static_cast<vector_size_t>(vector->type()->cppSizeInBytes());
    for (auto i = 0; i < numRows; ++i) {
      addToSize(sizes[i], elementSize);
    }
    return;
  }
  switch (vector->encoding()) {
    case VectorEncoding::Simple::FLAT: {
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          estimateFlatSerializedSize,
          vector->typeKind(),
          vector,
          rows,
          sizes,
          scratch);
      break;
    }
    case VectorEncoding::Simple::CONSTANT:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          estimateFlattenedConstantSerializedSize,
          vector->typeKind(),
          vector,
          rows,
          sizes,
          scratch);
      break;
    case VectorEncoding::Simple::DICTIONARY:
    case VectorEncoding::Simple::SEQUENCE:
      estimateWrapperSerializedSize(rows, sizes, vector, scratch);
      break;
    case VectorEncoding::Simple::BIASED:
      estimateBiasedSerializedSize(vector, rows, sizes);
      break;
    case VectorEncoding::Simple::ROW: {
      ScratchPtr<vector_size_t, 1> innerRowsHolder(scratch);
      ScratchPtr<vector_size_t*, 1> innerSizesHolder(scratch);
      ScratchPtr<uint64_t, 1> nullsHolder(scratch);
      auto* innerRows = rows.data();
      auto* innerSizes = sizes;
      const auto numRows = rows.size();
      for (auto i = 0; i < numRows; ++i) {
        addToSize(sizes[i], kInt32Bytes);
      }
      int32_t numInner = numRows;
      if (vector->mayHaveNulls()) {
        auto nulls = nullsHolder.get(bits::nwords(numRows));
        simd::gatherBits(vector->rawNulls(), rows, nulls);
        auto mutableInnerRows = innerRowsHolder.get(numRows);
        numInner = simd::indicesOfSetBits(nulls, 0, numRows, mutableInnerRows);
        innerSizes = innerSizesHolder.get(numInner);
        for (auto i = 0; i < numInner; ++i) {
          innerSizes[i] = sizes[mutableInnerRows[i]];
        }
        simd::transpose(
            rows.data(),
            folly::Range<const vector_size_t*>(mutableInnerRows, numInner),
            mutableInnerRows);
        innerRows = mutableInnerRows;
      }
      auto* rowVector = vector->as<RowVector>();
      auto& children = rowVector->children();
      for (auto& child : children) {
        if (child) {
          estimateSerializedSizeInt(
              child.get(),
              folly::Range(innerRows, numInner),
              innerSizes,
              scratch);
        }
      }
      break;
    }
    case VectorEncoding::Simple::MAP: {
      auto* mapVector = vector->asUnchecked<MapVector>();
      estimateCollectionSerializedSize(
          *mapVector,
          mapVector->rawOffsets(),
          mapVector->rawSizes(),
          mapVector->mapKeys()->size(),
          {mapVector->mapKeys().get(), mapVector->mapValues().get()},
          rows,
          sizes,
          scratch);
      break;
    }
    case VectorEncoding::Simple::ARRAY: {
      auto* arrayVector = vector->as<ArrayVector>();
      estimateCollectionSerializedSize(
          *arrayVector,
          arrayVector->rawOffsets(),
          arrayVector->rawSizes(),
          arrayVector->elements()->size(),
          {arrayVector->elements().get()},
          rows,
          sizes,
          scratch);
      break;
    }
    case VectorEncoding::Simple::LAZY:
      estimateSerializedSizeInt(vector->loadedVector(), rows, sizes, scratch);
      break;
    default:
      VELOX_UNSUPPORTED("Unsupported vector encoding {}", vector->encoding());
  }
}
} // namespace facebook::velox::serializer::presto::detail
