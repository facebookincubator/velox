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

#include "velox/serializers/PrestoSerializerSerializationUtils.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::serializer::presto::detail {
namespace {
void estimateSerializedSizeByRows(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch,
    bool flatten,
    bool includeNullBitmap);

// Returns 'bytes' as a size estimate, saturated at the largest vector_size_t
// so that a size of 2GB or more cannot wrap to a small or negative one.
vector_size_t saturatedSize(uint64_t bytes) {
  return static_cast<vector_size_t>(
      std::min<uint64_t>(bytes, std::numeric_limits<vector_size_t>::max()));
}

// Adds 'bytes' to the size estimate '*size', saturating like saturatedSize().
void addToSize(vector_size_t* size, uint64_t bytes) {
  *size = saturatedSize(*size + bytes);
}

// Only range-based estimates include bitmaps, and all size pointers within a
// range refer to the same accumulator.
void addNullBitmapSize(vector_size_t** sizes, int32_t numRows) {
  addToSize(sizes[0], bits::nbytes(numRows));
}

template <TypeKind Kind>
void estimateFlatSerializedSize(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch,
    bool includeNullBitmap) {
  const auto valueSize = vector->type()->cppSizeInBytes();
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
    if (includeNullBitmap && numNonNull != numRows) {
      addNullBitmapSize(sizes, numRows);
    }
  } else {
    VELOX_UNREACHABLE("Non null fixed width case handled before this");
  }
}

void estimateFlatSerializedSizeVarcharOrVarbinary(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch,
    bool includeNullBitmap) {
  const auto numRows = rows.size();
  auto strings = static_cast<const FlatVector<StringView>*>(vector);
  auto rawNulls = strings->rawNulls();
  auto rawValues = strings->rawValues();
  if (!rawNulls) {
    for (auto i = 0; i < rows.size(); ++i) {
      // Add the size of the length and the string data.
      addToSize(sizes[i], sizeof(int32_t) + rawValues[rows[i]].size());
    }
  } else {
    for (auto i = 0; i < numRows; ++i) {
      addToSize(sizes[i], sizeof(int32_t));
    }

    ScratchPtr<uint64_t, 4> nullsHolder(scratch);
    ScratchPtr<int32_t, 64> nonNullsHolder(scratch);
    auto nulls = nullsHolder.get(bits::nwords(numRows));
    simd::gatherBits(rawNulls, rows, nulls);
    auto* nonNulls = nonNullsHolder.get(numRows);
    auto numNonNull = simd::indicesOfSetBits(nulls, 0, numRows, nonNulls);

    for (int32_t i = 0; i < numNonNull; ++i) {
      addToSize(sizes[nonNulls[i]], rawValues[rows[nonNulls[i]]].size());
    }
    if (includeNullBitmap && numNonNull != numRows) {
      addNullBitmapSize(sizes, numRows);
    }
  }
}

template <>
void estimateFlatSerializedSize<TypeKind::VARCHAR>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch,
    bool includeNullBitmap) {
  estimateFlatSerializedSizeVarcharOrVarbinary(
      vector, rows, sizes, scratch, includeNullBitmap);
}

template <>
void estimateFlatSerializedSize<TypeKind::VARBINARY>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch,
    bool includeNullBitmap) {
  estimateFlatSerializedSizeVarcharOrVarbinary(
      vector, rows, sizes, scratch, includeNullBitmap);
}

template <>
void estimateFlatSerializedSize<TypeKind::OPAQUE>(
    const BaseVector*,
    const folly::Range<const vector_size_t*>&,
    vector_size_t**,
    Scratch&,
    bool) {
  VELOX_FAIL("Opaque type support is not implemented.");
}

template <TypeKind Kind>
void estimateFlattenedConstantSerializedSize(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch,
    bool flatten,
    bool includeNullBitmap) {
  VELOX_CHECK_EQ(vector->encoding(), VectorEncoding::Simple::CONSTANT);

  using T = typename KindToFlatVector<Kind>::WrapperType;
  auto* constantVector = vector->as<ConstantVector<T>>();
  int32_t elementSize = vector->valueVector() ? 0 : sizeof(T);
  if (constantVector->isNullAt(0)) {
    elementSize = includeNullBitmap ? 1 : 0;
  } else if (vector->valueVector()) {
    const auto* values = constantVector->wrappedVector();
    vector_size_t* sizePtr = &elementSize;
    const vector_size_t singleRow = constantVector->wrappedIndex(0);
    estimateSerializedSizeByRows(
        values,
        folly::Range<const vector_size_t*>(&singleRow, 1),
        &sizePtr,
        scratch,
        flatten,
        includeNullBitmap);
  } else if constexpr (std::is_same_v<T, StringView>) {
    elementSize = saturatedSize(constantVector->valueAt(0).size());
  }
  if (flatten) {
    for (int32_t i = 0; i < rows.size(); ++i) {
      addToSize(sizes[i], elementSize);
    }
  } else if (!rows.empty()) {
    addToSize(sizes[0], elementSize);
  }
}

void estimateWrapperSerializedSize(
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    const BaseVector* wrapper,
    Scratch& scratch,
    bool flatten,
    bool includeNullBitmap) {
  ScratchPtr<vector_size_t, 1> innerRowsHolder(scratch);
  ScratchPtr<vector_size_t*, 1> innerSizesHolder(scratch);
  const int32_t numRows = rows.size();
  int32_t numInner = 0;
  auto* innerRows = innerRowsHolder.get(numRows);
  auto* innerSizes = sizes;
  const BaseVector* wrapped;
  auto hasNulls = false;
  if (!flatten && wrapper->encoding() == VectorEncoding::Simple::DICTIONARY &&
      !wrapper->rawNulls()) {
    // A dictionary serializes an index for every row plus each selected
    // dictionary entry once.
    auto* indices = wrapper->wrapInfo()->as<vector_size_t>();
    wrapped = wrapper->valueVector().get();
    ScratchPtr<uint64_t, 64> usedIndicesHolder(scratch);
    auto* usedIndices = usedIndicesHolder.get(bits::nwords(wrapped->size()));
    simd::memset(usedIndices, 0, usedIndicesHolder.size() * sizeof(uint64_t));
    for (int32_t i = 0; i < numRows; ++i) {
      addToSize(sizes[i], sizeof(int32_t));
      bits::setBit(usedIndices, indices[rows[i]]);
    }
    numInner =
        simd::indicesOfSetBits(usedIndices, 0, wrapped->size(), innerRows);
    innerSizes = innerSizesHolder.get(numInner);
    // In range overload all rows of a range share the same size accumulator.
    for (int32_t i = 0; i < numInner; ++i) {
      innerSizes[i] = sizes[0];
    }
  } else if (
      wrapper->encoding() == VectorEncoding::Simple::DICTIONARY &&
      !wrapper->rawNulls()) {
    // Per-row estimation does not preserve dictionary encoding.
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
      } else {
        hasNulls = true;
      }
    }
  }
  if (includeNullBitmap && hasNulls) {
    addNullBitmapSize(sizes, numRows);
  }
  if (numInner == 0) {
    return;
  }

  estimateSerializedSizeByRows(
      wrapped,
      folly::Range<const vector_size_t*>(innerRows, numInner),
      innerSizes,
      scratch,
      flatten,
      includeNullBitmap);
}

void estimateBiasedSerializedSize(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    bool includeNullBitmap) {
  const auto valueSize = vector->type()->cppSizeInBytes();
  if (!vector->mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      addToSize(sizes[i], valueSize);
    }
    return;
  }

  auto numNonNull = 0;
  for (auto i = 0; i < rows.size(); ++i) {
    if (!vector->isNullAt(rows[i])) {
      addToSize(sizes[i], valueSize);
      ++numNonNull;
    }
  }
  if (includeNullBitmap && numNonNull != rows.size()) {
    addNullBitmapSize(sizes, rows.size());
  }
}

constexpr vector_size_t kMaxElementRowsPerBatch{1 << 20};

// Adds the null bitmap and lengths of the ARRAY or MAP rows in 'rows' to their
// estimates, then calls 'estimateElements' with the element rows of the
// non-null ones and, per element row, the estimate of its parent row. Rows
// that repeat a collection, through a dictionary or overlapping offsets, can
// select more element rows than vector_size_t counts, so the element rows go
// in batches of whole collections of at most kMaxElementRowsPerBatch, or one
// larger collection. A batch of a range estimate adds its own null bitmap.
template <typename EstimateElements>
void estimateCollectionsSerializedSize(
    const folly::Range<const vector_size_t*>& rows,
    const uint64_t* rawNulls,
    const vector_size_t* offsets,
    const vector_size_t* sizes,
    vector_size_t** sizePtrs,
    Scratch& scratch,
    bool includeNullBitmap,
    EstimateElements estimateElements) {
  const vector_size_t* nonNullPositions = rows.data();
  auto numNonNull = rows.size();
  ScratchPtr<uint64_t, 4> nullsHolder(scratch);
  ScratchPtr<vector_size_t, 64> nonNullPositionsHolder(scratch);
  if (rawNulls) {
    auto* nulls = nullsHolder.get(bits::nwords(rows.size()));
    simd::gatherBits(rawNulls, rows, nulls);
    auto* mutableNonNullPositions = nonNullPositionsHolder.get(rows.size());
    numNonNull =
        simd::indicesOfSetBits(nulls, 0, rows.size(), mutableNonNullPositions);
    if (includeNullBitmap && numNonNull != rows.size()) {
      addNullBitmapSize(sizePtrs, rows.size());
    }
    nonNullPositions = mutableNonNullPositions;
  }

  uint64_t numElements{0};
  vector_size_t maxParentElements{0};
  for (auto i = 0; i < numNonNull; ++i) {
    const auto parentPosition = rawNulls ? nonNullPositions[i] : i;
    addToSize(sizePtrs[parentPosition], sizeof(int32_t));
    const auto numParentElements = sizes[rows[parentPosition]];
    numElements += numParentElements;
    maxParentElements = std::max(maxParentElements, numParentElements);
  }
  if (numElements == 0) {
    return;
  }

  const auto batchCapacity = static_cast<vector_size_t>(std::min<uint64_t>(
      numElements, std::max(kMaxElementRowsPerBatch, maxParentElements)));
  ScratchPtr<vector_size_t> elementRowsHolder(scratch);
  ScratchPtr<vector_size_t*> elementSizesHolder(scratch);
  auto* elementRows = elementRowsHolder.get(batchCapacity);
  auto* elementSizes = elementSizesHolder.get(batchCapacity);
  vector_size_t fill{0};
  for (auto i = 0; i < numNonNull; ++i) {
    const auto parentPosition = rawNulls ? nonNullPositions[i] : i;
    const auto parentRow = rows[parentPosition];
    if (sizes[parentRow] > batchCapacity - fill) {
      estimateElements(
          folly::Range<const vector_size_t*>(elementRows, fill), elementSizes);
      fill = 0;
    }
    const auto end = offsets[parentRow] + sizes[parentRow];
    for (auto elementRow = offsets[parentRow]; elementRow < end; ++elementRow) {
      elementRows[fill] = elementRow;
      elementSizes[fill] = sizePtrs[parentPosition];
      ++fill;
    }
  }
  estimateElements(
      folly::Range<const vector_size_t*>(elementRows, fill), elementSizes);
}
} // namespace

void estimateSerializedSizeInt(
    const BaseVector* vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    Scratch& scratch) {
  const auto totalSize = rangesTotalSize(ranges);
  ScratchPtr<vector_size_t, 64> rowsHolder(scratch);
  ScratchPtr<vector_size_t*, 64> rowSizesHolder(scratch);
  auto* allRows = rowsHolder.get(totalSize);
  auto* allRowSizes = rowSizesHolder.get(totalSize);
  auto offset = 0;
  for (auto i = 0; i < ranges.size(); ++i) {
    const auto numRows = ranges[i].size;
    if (numRows == 0) {
      continue;
    }
    auto* rows = allRows + offset;
    auto* rowSizes = allRowSizes + offset;
    for (auto j = 0; j < numRows; ++j) {
      rows[j] = ranges[i].begin + j;
      rowSizes[j] = sizes[i];
    }
    estimateSerializedSizeByRows(
        vector,
        folly::Range<const vector_size_t*>(rows, numRows),
        rowSizes,
        scratch,
        /*flatten=*/false,
        /*includeNullBitmap=*/true);
    offset += numRows;
  }
}

void estimateSerializedSizeInt(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  // A bitmap belongs to a batch, not to an individual row. Omitting it keeps
  // per-row estimates invariant under splitting or reordering the input.
  estimateSerializedSizeByRows(
      vector,
      rows,
      sizes,
      scratch,
      /*flatten=*/true,
      /*includeNullBitmap=*/false);
}

namespace {
void estimateSerializedSizeByRows(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch,
    bool flatten,
    bool includeNullBitmap) {
  const auto numRows = rows.size();
  if (vector->encoding() == VectorEncoding::Simple::FLAT &&
      vector->type()->isFixedWidth() && !vector->mayHaveNullsRecursive()) {
    const auto elementSize = vector->type()->cppSizeInBytes();
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
          scratch,
          includeNullBitmap);
      break;
    }
    case VectorEncoding::Simple::CONSTANT:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          estimateFlattenedConstantSerializedSize,
          vector->typeKind(),
          vector,
          rows,
          sizes,
          scratch,
          flatten,
          includeNullBitmap);
      break;
    case VectorEncoding::Simple::DICTIONARY:
    case VectorEncoding::Simple::SEQUENCE:
      estimateWrapperSerializedSize(
          rows, sizes, vector, scratch, flatten, includeNullBitmap);
      break;
    case VectorEncoding::Simple::BIASED:
      estimateBiasedSerializedSize(vector, rows, sizes, includeNullBitmap);
      break;
    case VectorEncoding::Simple::ROW: {
      ScratchPtr<vector_size_t, 1> innerRowsHolder(scratch);
      ScratchPtr<vector_size_t*, 1> innerSizesHolder(scratch);
      ScratchPtr<uint64_t, 1> nullsHolder(scratch);
      auto* innerRows = rows.data();
      auto* innerSizes = sizes;
      const auto numRows = rows.size();
      for (auto i = 0; i < numRows; ++i) {
        addToSize(sizes[i], sizeof(int32_t));
      }
      int32_t numInner = numRows;
      if (vector->mayHaveNulls()) {
        auto nulls = nullsHolder.get(bits::nwords(numRows));
        simd::gatherBits(vector->rawNulls(), rows, nulls);
        auto mutableInnerRows = innerRowsHolder.get(numRows);
        numInner = simd::indicesOfSetBits(nulls, 0, numRows, mutableInnerRows);
        if (includeNullBitmap && numInner != numRows) {
          addNullBitmapSize(sizes, numRows);
        }
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
          estimateSerializedSizeByRows(
              child.get(),
              folly::Range(innerRows, numInner),
              innerSizes,
              scratch,
              flatten,
              includeNullBitmap);
        }
      }
      break;
    }
    case VectorEncoding::Simple::MAP: {
      auto* mapVector = vector->asUnchecked<MapVector>();
      estimateCollectionsSerializedSize(
          rows,
          mapVector->rawNulls(),
          mapVector->rawOffsets(),
          mapVector->rawSizes(),
          sizes,
          scratch,
          includeNullBitmap,
          [&](const folly::Range<const vector_size_t*>& elementRows,
              vector_size_t** elementSizes) {
            for (const auto& child :
                 {mapVector->mapKeys(), mapVector->mapValues()}) {
              // Map children are serialized as flattened streams because
              // VectorStream only descends into ROW.
              estimateSerializedSizeByRows(
                  child.get(),
                  elementRows,
                  elementSizes,
                  scratch,
                  true,
                  includeNullBitmap);
            }
          });
      break;
    }
    case VectorEncoding::Simple::ARRAY: {
      auto* arrayVector = vector->as<ArrayVector>();
      estimateCollectionsSerializedSize(
          rows,
          arrayVector->rawNulls(),
          arrayVector->rawOffsets(),
          arrayVector->rawSizes(),
          sizes,
          scratch,
          includeNullBitmap,
          [&](const folly::Range<const vector_size_t*>& elementRows,
              vector_size_t** elementSizes) {
            // ARRAY elements are serialized as flattened stream because
            // VectorStream only descends into ROW.
            estimateSerializedSizeByRows(
                arrayVector->elements().get(),
                elementRows,
                elementSizes,
                scratch,
                true,
                includeNullBitmap);
          });
      break;
    }
    case VectorEncoding::Simple::LAZY:
      estimateSerializedSizeByRows(
          vector->loadedVector(),
          rows,
          sizes,
          scratch,
          flatten,
          includeNullBitmap);
      break;
    default:
      VELOX_UNSUPPORTED("Unsupported vector encoding {}", vector->encoding());
  }
}
} // namespace
} // namespace facebook::velox::serializer::presto::detail
