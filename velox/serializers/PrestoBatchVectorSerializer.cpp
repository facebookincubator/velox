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

#include "velox/serializers/PrestoBatchVectorSerializer.h"

#include "velox/serializers/PrestoSerializerEstimationUtils.h"
#include "velox/serializers/PrestoSerializerSerializationUtils.h"
#include "velox/serializers/VectorStream.h"

namespace facebook::velox::serializer::presto::detail {
namespace {
// Populates mutableOffsets with the starting offset of each collection and the
// total size of all collections.
//
// Populates mutableSelectedRanges with the ranges to write from the children.
//
// Populates numSelectedRanges with the number of ranges to write from the
// children.
template <typename VectorType, typename RangeType>
void computeCollectionRangesAndOffsets(
    const VectorType* vector,
    const folly::Range<const RangeType*>& ranges,
    bool hasNulls,
    int32_t* mutableOffsets,
    IndexRange* mutableSelectedRanges,
    size_t& numSelectedRanges) {
  auto* rawSizes = vector->rawSizes();
  auto* rawOffsets = vector->rawOffsets();

  // The first offset is always 0.
  mutableOffsets[0] = 0;
  // The index of the next offset to write in mutableOffsets.
  size_t offsetsIndex = 1;
  // The length all the collections in ranges seen so far. This is the offset to
  // write for the next collection.
  int32_t totalLength = 0;
  if (hasNulls) {
    for (const auto& range : ranges) {
      if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
        if (range.isNull) {
          std::fill_n(&mutableOffsets[offsetsIndex], range.size, totalLength);
          offsetsIndex += range.size;

          continue;
        }
      }

      for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
        if (vector->isNullAt(i)) {
          mutableOffsets[offsetsIndex++] = totalLength;
        } else {
          auto length = rawSizes[i];
          totalLength += length;
          mutableOffsets[offsetsIndex++] = totalLength;

          // We only have to write anything from the children if the collection
          // is non-empty.
          if (length > 0) {
            mutableSelectedRanges[numSelectedRanges++] = {
                rawOffsets[i], rawSizes[i]};
          }
        }
      }
    }
  } else {
    for (const auto& range : ranges) {
      for (auto i = range.begin; i < range.begin + range.size; ++i) {
        auto length = rawSizes[i];
        totalLength += length;
        mutableOffsets[offsetsIndex++] = totalLength;

        // We only have to write anything from the children if the collection is
        // non-empty.
        if (length > 0) {
          mutableSelectedRanges[numSelectedRanges++] = {rawOffsets[i], length};
        }
      }
    }
  }
}
} // namespace

void PrestoBatchVectorSerializer::serialize(
    const RowVectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    Scratch& scratch,
    OutputStream* stream) {
  const auto numRows = rangesTotalSize(ranges);
  const auto rowType = vector->type();
  const auto numChildren = vector->childrenSize();

  std::vector<VectorStream> streams;
  streams.reserve(numChildren);
  for (int i = 0; i < numChildren; i++) {
    streams.emplace_back(
        rowType->childAt(i),
        std::nullopt,
        vector->childAt(i),
        &arena_,
        numRows,
        opts_);

    if (numRows > 0) {
      velox::serializer::presto::detail::serializeColumn(
          vector->childAt(i), ranges, &streams[i], scratch);
    }
  }

  flushStreams(
      streams, numRows, arena_, *codec_, opts_.minCompressionRatio, stream);

  arena_.clear();
}

void PrestoBatchVectorSerializer::estimateSerializedSizeImpl(
    const VectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    Scratch& scratch) {
  switch (vector->encoding()) {
    case VectorEncoding::Simple::FLAT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          estimateFlatSerializedSize,
          vector->typeKind(),
          vector.get(),
          ranges,
          sizes);
      break;
    case VectorEncoding::Simple::CONSTANT:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          estimateConstantSerializedSize,
          vector->typeKind(),
          vector,
          ranges,
          sizes,
          scratch);
      break;
    case VectorEncoding::Simple::DICTIONARY:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          estimateDictionarySerializedSize,
          vector->typeKind(),
          vector,
          ranges,
          sizes,
          scratch);
      break;
    case VectorEncoding::Simple::ROW: {
      if (!vector->mayHaveNulls()) {
        // Add the size of the offsets in the Row encoding.
        for (int32_t i = 0; i < ranges.size(); ++i) {
          *sizes[i] += ranges[i].size * sizeof(int32_t);
        }

        auto rowVector = vector->as<RowVector>();
        auto& children = rowVector->children();
        for (auto& child : children) {
          if (child) {
            estimateSerializedSizeImpl(child, ranges, sizes, scratch);
          }
        }

        break;
      }

      std::vector<IndexRange> childRanges;
      std::vector<vector_size_t*> childSizes;
      for (int32_t i = 0; i < ranges.size(); ++i) {
        // Add the size of the nulls bit mask.
        *sizes[i] += bits::nbytes(ranges[i].size);

        auto begin = ranges[i].begin;
        auto end = begin + ranges[i].size;
        for (auto offset = begin; offset < end; ++offset) {
          // Add the size of the offset.
          *sizes[i] += sizeof(int32_t);
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
          estimateSerializedSizeImpl(
              child,
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
      estimateSerializedSizeImpl(
          mapVector->mapKeys(), childRanges, childSizes.data(), scratch);
      estimateSerializedSizeImpl(
          mapVector->mapValues(), childRanges, childSizes.data(), scratch);
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
      estimateSerializedSizeImpl(
          arrayVector->elements(), childRanges, childSizes.data(), scratch);
      break;
    }
    case VectorEncoding::Simple::LAZY:
      estimateSerializedSizeImpl(
          vector->as<LazyVector>()->loadedVectorShared(),
          ranges,
          sizes,
          scratch);
      break;
    default:
      VELOX_UNSUPPORTED("Unsupported vector encoding {}", vector->encoding());
  }
}

void PrestoBatchVectorSerializer::writeHeader(
    BufferedOutputStream* stream,
    const TypePtr& type) {
  auto encoding = typeToEncodingName(type);
  writeInt32(stream, encoding.size());
  stream->write(encoding.data(), encoding.size());
}

template <>
bool PrestoBatchVectorSerializer::hasNulls(
    const VectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges) {
  if (vector->nulls()) {
    for (auto& range : ranges) {
      if (!bits::isAllSet(
              vector->rawNulls(), range.begin, range.begin + range.size)) {
        return true;
      }
    }
  }

  return false;
}

template <>
bool PrestoBatchVectorSerializer::hasNulls(
    const VectorPtr& vector,
    const folly::Range<const IndexRangeWithNulls*>& ranges) {
  if (vector->nulls()) {
    for (auto& range : ranges) {
      if (range.isNull ||
          !bits::isAllSet(
              vector->rawNulls(), range.begin, range.begin + range.size)) {
        return true;
      }
    }
  } else {
    for (auto& range : ranges) {
      if (range.isNull) {
        return true;
      }
    }
  }

  return false;
}

template <>
void PrestoBatchVectorSerializer::writeNulls(
    BufferedOutputStream* stream,
    const VectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    const vector_size_t numRows) {
  VELOX_DCHECK_EQ(numRows, rangesTotalSize(ranges));

  nulls_.startWrite(bits::nbytes(numRows));
  for (auto& range : ranges) {
    nulls_.appendBits(
        vector->rawNulls(), range.begin, range.begin + range.size);
  }
  nulls_.flush(stream);
}

template <>
void PrestoBatchVectorSerializer::writeNulls(
    BufferedOutputStream* stream,
    const VectorPtr& vector,
    const folly::Range<const IndexRangeWithNulls*>& ranges,
    const vector_size_t numRows) {
  VELOX_DCHECK_EQ(numRows, rangesTotalSize(ranges));

  nulls_.startWrite(bits::nbytes(numRows));
  for (auto& range : ranges) {
    if (range.isNull) {
      nulls_.appendBool(bits::kNull, range.size);
    } else if (vector->mayHaveNulls()) {
      nulls_.appendBits(
          vector->rawNulls(), range.begin, range.begin + range.size);
    } else {
      nulls_.appendBool(bits::kNotNull, range.size);
    }
  }
  nulls_.flush(stream);
}

template <typename RangeType>
void PrestoBatchVectorSerializer::serializeRowVector(
    BufferedOutputStream* stream,
    const VectorPtr& vector,
    const folly::Range<const RangeType*>& ranges) {
  const auto* rowVector = vector->as<RowVector>();
  const auto numRows = rangesTotalSize(ranges);

  // Write out the header.
  writeHeader(stream, vector->type());

  const bool hasNulls = this->hasNulls(vector, ranges);

  // The ranges to write of the child Vectors, this is the same as the ranges
  // of this RowVector to write except positions where the row is null.
  folly::Range<const IndexRange*> childRanges;
  // PrestoPage requires us to write out for each row 0 if the row is null or
  // i if the row is the i'th non-null row. We track these values here.
  ScratchPtr<int32_t, 64> offsetsHolder(scratch_);
  int32_t* mutableOffsets = offsetsHolder.get(numRows + 1);
  // The first offset is always 0, this in addition to the offset per row.
  mutableOffsets[0] = 0;
  // The index at which we should write the next value in mutableOffsets.
  size_t offsetsIndex = 1;
  // The value of "offset" to write for the next non-null row.
  int32_t rowOffset = 1;

  // We use this to construct contiguous ranges to write for the children,
  // excluding any null rows.
  ScratchPtr<IndexRange, 64> selectedRangesHolder(scratch_);

  if (hasNulls) {
    IndexRange* mutableSelectedRanges = selectedRangesHolder.get(numRows);
    // The index in mutableSelectedRanges to write the next range.
    size_t rangeIndex = 0;

    for (const auto& range : ranges) {
      if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
        if (range.isNull) {
          std::fill_n(&mutableOffsets[offsetsIndex], range.size, 0);
          offsetsIndex += range.size;

          continue;
        }
      }

      if (vector->mayHaveNulls() &&
          !bits::isAllSet(
              vector->rawNulls(), range.begin, range.begin + range.size)) {
        // The start of the current contiguous range.
        int rangeStart = -1;
        // The length of the current contiguous range.
        int rangeSize = 0;
        for (auto i = range.begin; i < range.begin + range.size; ++i) {
          if (!vector->isNullAt(i)) {
            mutableOffsets[offsetsIndex++] = rowOffset++;

            // If we aren't already in a contiguous range, mark the beginning.
            if (rangeStart == -1) {
              rangeStart = i;
            }
            // Continue the contiguous range.
            rangeSize++;
          } else {
            mutableOffsets[offsetsIndex++] = 0;

            // If we were in a contiguous range, write it out to the scratch
            // buffer and indicate we are no longer in one.
            if (rangeStart != -1) {
              mutableSelectedRanges[rangeIndex++] =
                  IndexRange{rangeStart, rangeSize};
              rangeStart = -1;
              rangeSize = 0;
            }
          }
        }

        // If we were in a contigous range, write out the last one.
        if (rangeStart != -1) {
          mutableSelectedRanges[rangeIndex++] =
              IndexRange{rangeStart, rangeSize};
        }
      } else {
        // There are no nulls in this range, write out the offsets and copy
        // the range to the scratch buffer.
        std::iota(
            &mutableOffsets[offsetsIndex],
            &mutableOffsets[offsetsIndex + range.size],
            rowOffset);
        rowOffset += range.size;
        offsetsIndex += range.size;

        mutableSelectedRanges[rangeIndex++] =
            IndexRange{range.begin, range.size};
      }
    }

    // Lastly update child ranges to exclude any null rows.
    childRanges =
        folly::Range<const IndexRange*>(mutableSelectedRanges, rangeIndex);
  } else {
    // There are no null rows, so offsets is just an incrementing series and
    // we can reuse ranges for the children.
    std::iota(&mutableOffsets[1], &mutableOffsets[numRows + 1], rowOffset);

    if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
      IndexRange* mutableSelectedRanges = selectedRangesHolder.get(numRows);
      // The index in mutableSelectedRanges to write the next range.
      size_t rangeIndex = 0;
      for (const auto& range : ranges) {
        mutableSelectedRanges[rangeIndex++] = {range.begin, range.size};
      }

      childRanges =
          folly::Range<const IndexRange*>(mutableSelectedRanges, ranges.size());
    } else {
      childRanges = ranges;
    }
  }

  if (opts_.nullsFirst) {
    // Write out the number of rows.
    writeInt32(stream, numRows);
    // Write out the hasNull and isNull flags.
    writeNullsSegment(stream, hasNulls, vector, ranges, numRows);
  }

  // Write out the number of children.
  writeInt32(stream, vector->type()->size());
  // Write out the children.
  for (int32_t i = 0; i < rowVector->childrenSize(); ++i) {
    serializeColumn(stream, rowVector->childAt(i), childRanges);
  }

  if (!opts_.nullsFirst) {
    // Write out the number of rows.
    writeInt32(stream, numRows);
    // Write out the offsets.
    stream->write(
        reinterpret_cast<char*>(mutableOffsets),
        (numRows + 1) * sizeof(int32_t));
    // Write out the hasNull and isNull flags.
    writeNullsSegment(stream, hasNulls, vector, ranges, numRows);
  }
}

template <typename RangeType>
void PrestoBatchVectorSerializer::serializeColumn(
    BufferedOutputStream* stream,
    const VectorPtr& vector,
    const folly::Range<const RangeType*>& ranges) {
  switch (vector->encoding()) {
    case VectorEncoding::Simple::FLAT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          serializeFlatVector, vector->typeKind(), stream, vector, ranges);
      break;
    case VectorEncoding::Simple::CONSTANT:
      // VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      //     serializeConstantVector,
      //     vector->typeKind(),
      //     stream,
      //     vector,
      //     ranges);
      break;
    case VectorEncoding::Simple::DICTIONARY:
      // VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      //     serializeDictionaryVector,
      //     vector->typeKind(),
      //     stream,
      //     vector,
      //     ranges);
      break;
    case VectorEncoding::Simple::ROW:
      serializeRowVector(stream, vector, ranges);
      break;
    case VectorEncoding::Simple::ARRAY:
      serializeArrayVector(stream, vector, ranges);
      break;
    case VectorEncoding::Simple::MAP:
      serializeMapVector(stream, vector, ranges);
      break;
    case VectorEncoding::Simple::LAZY:
      serializeColumn(stream, BaseVector::loadedVectorShared(vector), ranges);
      break;
    default:
      VELOX_UNSUPPORTED();
  }
}

template <typename RangeType>
void PrestoBatchVectorSerializer::serializeArrayVector(
    BufferedOutputStream* stream,
    const VectorPtr& vector,
    const folly::Range<const RangeType*>& ranges) {
  const auto* arrayVector = vector->as<ArrayVector>();
  const auto numRows = rangesTotalSize(ranges);

  // Write out the header.
  writeHeader(stream, vector->type());

  const bool hasNulls = this->hasNulls(vector, ranges);

  // This is used to hold the ranges of the elements Vector to write out.
  ScratchPtr<IndexRange, 64> selectedRangesHolder(scratch_);
  IndexRange* mutableSelectedRanges = selectedRangesHolder.get(numRows);
  // This is used to hold the offsets of the arrays which we write out towards
  // the end.
  ScratchPtr<int32_t, 64> offsetsHolder(scratch_);
  int32_t* mutableOffsets = offsetsHolder.get(numRows + 1);
  // The number of ranges to write out from the elements Vector. This is equal
  // to the number of non-empty, non-null arrays in ranges.
  size_t numSelectedRanges = 0;
  computeCollectionRangesAndOffsets(
      arrayVector,
      ranges,
      hasNulls,
      mutableOffsets,
      mutableSelectedRanges,
      numSelectedRanges);

  // Write out the elements.
  serializeColumn(
      stream,
      arrayVector->elements(),
      folly::Range<const IndexRange*>(
          mutableSelectedRanges, numSelectedRanges));

  // Write out the number of rows.
  writeInt32(stream, numRows);
  // Write out the offsets.
  stream->write(
      reinterpret_cast<char*>(mutableOffsets), (numRows + 1) * sizeof(int32_t));

  // Write out the hasNull and isNUll flags.
  writeNullsSegment(stream, hasNulls, vector, ranges, numRows);
}

template <typename RangeType>
void PrestoBatchVectorSerializer::serializeMapVector(
    BufferedOutputStream* stream,
    const VectorPtr& vector,
    const folly::Range<const RangeType*>& ranges) {
  const auto* mapVector = vector->as<MapVector>();
  const auto numRows = rangesTotalSize(ranges);

  // Write out the header.
  writeHeader(stream, vector->type());

  const bool hasNulls = this->hasNulls(vector, ranges);

  // This is used to hold the ranges of the keys/values Vectors to write out.
  ScratchPtr<IndexRange, 64> selectedRangesHolder(scratch_);
  IndexRange* mutableSelectedRanges = selectedRangesHolder.get(numRows);
  // This is used to hold the offsets of the maps which we write out towards the
  // end.
  ScratchPtr<int32_t, 64> offsetsHolder(scratch_);
  int32_t* mutableOffsets = offsetsHolder.get(numRows + 1);
  // The number of ranges to write out from the keys/values Vectors. This is
  // equal to the number of non-empty, non-null mpas in ranges.
  size_t rangesSize = 0;
  computeCollectionRangesAndOffsets(
      mapVector,
      ranges,
      hasNulls,
      mutableOffsets,
      mutableSelectedRanges,
      rangesSize);

  // Write out the keys.
  serializeColumn(
      stream,
      mapVector->mapKeys(),
      folly::Range<const IndexRange*>(mutableSelectedRanges, rangesSize));
  // Write out the values.
  serializeColumn(
      stream,
      mapVector->mapValues(),
      folly::Range<const IndexRange*>(mutableSelectedRanges, rangesSize));

  // Write out the hash table size (we don't write out the optional hash table,
  // so use -1 as the size).
  writeInt32(stream, -1);
  // Write out the number of rows.
  writeInt32(stream, numRows);
  // Write out the offsets.
  stream->write(
      reinterpret_cast<char*>(mutableOffsets), (numRows + 1) * sizeof(int32_t));

  // Wirte out the hasNull and isNull flags.
  writeNullsSegment(stream, hasNulls, vector, ranges, numRows);
}
} // namespace facebook::velox::serializer::presto::detail
