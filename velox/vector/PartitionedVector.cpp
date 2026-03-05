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
#include "velox/vector/PartitionedVector.h"

#include "velox/vector/FlatVector.h"

namespace facebook::velox {

using Byte = uint8_t;
using BitIndex = uint8_t;

namespace {

inline void countPartitionSizes(
    const std::vector<uint32_t>& partitions,
    vector_size_t* rowCounts) {
  VELOX_DCHECK_NOT_NULL(rowCounts);

  for (vector_size_t i = 0; i < partitions.size(); i++) {
    rowCounts[partitions[i]]++;
  }
}

inline void prefixSum(vector_size_t* offsets, uint32_t numPartitions) {
  for (uint32_t i = 1; i < numPartitions; i++) {
    offsets[i] += offsets[i - 1];
  }
}

inline void calculateOffsets(
    const std::vector<uint32_t>& partitions,
    uint32_t numPartitions,
    vector_size_t* endPartitionOffsets) {
  VELOX_DCHECK_NOT_NULL(endPartitionOffsets);

  if (numPartitions > 1) {
    std::fill_n(endPartitionOffsets, numPartitions, 0);
    countPartitionSizes(partitions, endPartitionOffsets);
    prefixSum(endPartitionOffsets, numPartitions);
  } else {
    endPartitionOffsets[0] = static_cast<vector_size_t>(partitions.size());
  }
}

// endPartitionOffsets is an array of length numPartitions where each entry i is
// the exclusive end position of partition i. cursorPartitionOffsets is
// initialized such that cursorPartitionOffsets[0] = 0 and for i>0,
// cursorPartitionOffsets[i] = endPartitionOffsets[i-1], i.e., the inclusive
// begin positions.
void initializeCursorPartitionOffsets(
    BufferPtr& cursorPartitionOffsets,
    const BufferPtr& endPartitionOffsets,
    uint32_t numPartitions,
    velox::memory::MemoryPool* pool) {
  VELOX_DCHECK_NOT_NULL(endPartitionOffsets);
  VELOX_DCHECK_EQ(
      endPartitionOffsets->size(), numPartitions * sizeof(vector_size_t));

  ensureCapacity<vector_size_t>(cursorPartitionOffsets, numPartitions, pool);
  cursorPartitionOffsets->asMutable<vector_size_t>()[0] = 0;
  std::memcpy(
      &cursorPartitionOffsets->asMutable<vector_size_t>()[1],
      endPartitionOffsets->as<vector_size_t>(),
      sizeof(vector_size_t) * (numPartitions - 1));
  cursorPartitionOffsets->setSize(numPartitions * sizeof(vector_size_t));
}

// In-place partitioning algorithm for fixed-width values
// This algorithm rearranges elements so that each element ends up in its target
// partition by repeatedly swapping elements until the current element belongs
// to the current partition
template <typename T>
void partitionFixedWidthValuesInPlace(
    T* values,
    const std::vector<uint32_t>& partitions,
    uint32_t numPartitions,
    vector_size_t* cursorPartitionOffsets,
    const vector_size_t* endPartitionOffsets) {
  VELOX_DCHECK_NOT_NULL(values);
  VELOX_DCHECK_NOT_NULL(cursorPartitionOffsets);
  VELOX_DCHECK_NOT_NULL(endPartitionOffsets);

  for (auto currentPartition = 0; currentPartition < numPartitions;
       currentPartition++) {
    vector_size_t& offset = cursorPartitionOffsets[currentPartition];
    vector_size_t endOffset = endPartitionOffsets[currentPartition];

    while (offset < endOffset) {
      uint32_t targetPartition = partitions[offset];

      while (targetPartition != currentPartition) {
        auto destinationOffset = cursorPartitionOffsets[targetPartition]++;
        std::swap(values[destinationOffset], values[offset]);
        targetPartition = partitions[destinationOffset];
      }
      offset = ++cursorPartitionOffsets[currentPartition];
    }
  }
}

template <typename T>
void partitionFixedWidthValues(
    BufferPtr& inputBuffer,
    const std::vector<uint32_t>& partitions,
    const BufferPtr& endPartitionOffsets,
    uint32_t numPartitions,
    PartitionBuildContext& ctx,
    velox::memory::MemoryPool* pool) {
  VELOX_DCHECK_NOT_NULL(inputBuffer);
  VELOX_DCHECK_NOT_NULL(endPartitionOffsets);

  auto input = inputBuffer->asMutable<T>();

  initializeCursorPartitionOffsets(
      ctx.cursorPartitionOffsets, endPartitionOffsets, numPartitions, pool);

  vector_size_t* rawCursorOffsets =
      ctx.cursorPartitionOffsets->asMutable<vector_size_t>();
  const vector_size_t* rawEndOffsets =
      endPartitionOffsets->asMutable<vector_size_t>();

  partitionFixedWidthValuesInPlace<T>(
      input, partitions, numPartitions, rawCursorOffsets, rawEndOffsets);
}

// Swap two bits between two bytes
void swapBit(Byte& byte1, BitIndex bit1, Byte& byte2, BitIndex bit2) {
  // Calculate the difference between the bits
  char bitDiff = ((byte1 >> bit1) & 1) ^ ((byte2 >> bit2) & 1);

  // Apply the difference to toggle the bits
  byte1 ^= (bitDiff << bit1);
  byte2 ^= (bitDiff << bit2);
}

void partitionBitsInPlace(
    Byte* bits,
    const std::vector<uint32_t>& partitions,
    uint32_t numPartitions,
    PartitionBuildContext& ctx,
    const BufferPtr& endPartitionOffsets,
    velox::memory::MemoryPool* pool) {
  initializeCursorPartitionOffsets(
      ctx.cursorPartitionOffsets, endPartitionOffsets, numPartitions, pool);

  auto rawCursorOffsets =
      ctx.cursorPartitionOffsets->asMutable<vector_size_t>();
  auto rawEndOffsets = endPartitionOffsets->asMutable<vector_size_t>();

  for (uint32_t partition = 0; partition < numPartitions; partition++) {
    auto& offset = rawCursorOffsets[partition];
    auto endOffset = rawEndOffsets[partition];
    while (offset < endOffset) {
      uint32_t p = partitions[offset];
      while (p != partition) {
        vector_size_t destinationOffset = rawCursorOffsets[p]++;

        // Calculate the byte address and bit index within the byte for the
        // source and destination bits. Since each byte contains 8 bits, we
        // divide the offset by 8 to get the byte address and take the modulus
        // by 8 to get the bit index within that byte.
        vector_size_t destinationAddr = destinationOffset >> 3;
        int8_t destinationBitInByte = destinationOffset & 7;
        vector_size_t fromAddr = offset >> 3;
        int8_t fromBitInByte = offset & 7;

        swapBit(
            bits[destinationAddr],
            destinationBitInByte,
            bits[fromAddr],
            fromBitInByte);
        p = partitions[destinationOffset];
      }
      offset = ++rawCursorOffsets[partition];
    }
  }
}

template <TypeKind typeKind>
PartitionedVectorPtr createPartitionedFlatVector(
    VectorPtr vector,
    const std::vector<uint32_t>& partitions,
    uint32_t numPartitions,
    const BufferPtr& endPartitionOffsets,
    PartitionBuildContext& ctx,
    velox::memory::MemoryPool* pool) {
  using T = typename TypeTraits<typeKind>::NativeType;
  auto flatVector = std::dynamic_pointer_cast<FlatVector<T>>(vector);
  VELOX_CHECK_NOT_NULL(flatVector);

  auto partitionedFlatVector = std::make_shared<PartitionedFlatVector<T>>(
      flatVector, numPartitions, endPartitionOffsets, pool);

  if (numPartitions > 1) {
    partitionedFlatVector->partition(partitions, ctx);
  }

  return partitionedFlatVector;
}

PartitionedVectorPtr createPartitionedRowVector(
    VectorPtr vector,
    const std::vector<uint32_t>& partitions,
    uint32_t numPartitions,
    const BufferPtr& endPartitionOffsets,
    PartitionBuildContext& ctx,
    velox::memory::MemoryPool* pool) {
  auto rowVector = std::dynamic_pointer_cast<RowVector>(vector);
  VELOX_CHECK_NOT_NULL(rowVector);

  auto partitionedRowVector = std::make_shared<PartitionedRowVector>(
      rowVector, numPartitions, endPartitionOffsets, pool);

  // Always call partition() to initialize partitionedChildren_, even when
  // numPartitions == 1, so that partitionAt() can reconstruct the RowVector.
  partitionedRowVector->partition(partitions, ctx);

  return partitionedRowVector;
}

} // namespace

PartitionedVector::~PartitionedVector() = default;

PartitionedVectorPtr PartitionedVector::create(
    const VectorPtr& vector,
    const std::vector<uint32_t>& partitions,
    uint32_t numPartitions,
    PartitionBuildContext& ctx,
    velox::memory::MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(vector);
  VELOX_CHECK_EQ(vector->size(), partitions.size());
  VELOX_CHECK_GT(numPartitions, 0);
  VELOX_CHECK_NOT_NULL(pool);

  // Calculate the end offsets for each partition. For example, if there are 3
  // partitions with 2, 3, and 1 rows respectively, then endPartitionOffsets[0]
  // = 2, endPartitionOffsets[1] = 5, and endPartitionOffsets[2] = 6.
  BufferPtr endPartitionOffsets;
  ensureCapacity<vector_size_t>(endPartitionOffsets, numPartitions, pool);
  calculateOffsets(
      partitions,
      numPartitions,
      endPartitionOffsets->asMutable<vector_size_t>());
  endPartitionOffsets->setSize(numPartitions * sizeof(vector_size_t));

  auto raw = endPartitionOffsets->as<vector_size_t>();
  VELOX_DCHECK_EQ(raw[numPartitions - 1], partitions.size());

  return create(
      vector, partitions, numPartitions, endPartitionOffsets, ctx, pool);
}

PartitionedVectorPtr PartitionedVector::create(
    const VectorPtr& vector,
    const std::vector<uint32_t>& partitions,
    uint32_t numPartitions,
    const BufferPtr& endPartitionOffsets,
    PartitionBuildContext& ctx,
    velox::memory::MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(endPartitionOffsets);
  VELOX_CHECK_EQ(
      endPartitionOffsets->size(), numPartitions * sizeof(vector_size_t));

  auto encoding = vector->encoding();
  auto typeKind = vector->typeKind();

  switch (encoding) {
    case VectorEncoding::Simple::FLAT: {
      auto partitionedFlatVector = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createPartitionedFlatVector,
          typeKind,
          vector,
          partitions,
          numPartitions,
          endPartitionOffsets,
          ctx,
          pool);
      return partitionedFlatVector;
    }

    case VectorEncoding::Simple::ROW: {
      return createPartitionedRowVector(
          vector, partitions, numPartitions, endPartitionOffsets, ctx, pool);
    }

    case VectorEncoding::Simple::ARRAY:
    case VectorEncoding::Simple::MAP:
    case VectorEncoding::Simple::DICTIONARY:
    case VectorEncoding::Simple::BIASED:
    case VectorEncoding::Simple::SEQUENCE:
    case VectorEncoding::Simple::CONSTANT:
    case VectorEncoding::Simple::LAZY:
      VELOX_UNSUPPORTED(
          "Unsupported vector encoding for PartitionedVector: {}",
          mapSimpleToName(encoding));
    default:
      VELOX_UNREACHABLE(
          "Invalid vector encoding for PartitionedVector: {}", encoding);
  }
}

VectorPtr PartitionedVector::baseVector() const {
  return vector_;
}

std::string PartitionedVector::toString() const {
  std::string offsets;
  for (vector_size_t i = 0; i < numPartitions_; ++i) {
    if (i > 0) {
      offsets += ',';
    }
    offsets += fmt::format("{}", rawEndPartitionOffsets_[i]);
  }

  return fmt::format(
      "PartitionedVector[numPartitions: {}, offsets: {}]",
      numPartitions_,
      offsets);
}

template <typename T>
void PartitionedFlatVector<T>::partition(
    const std::vector<uint32_t>& partitions,
    PartitionBuildContext& ctx) {
  Byte* rawNulls = reinterpret_cast<Byte*>(vector_->mutableRawNulls());
  if (rawNulls) {
    partitionBitsInPlace(
        rawNulls, partitions, numPartitions_, ctx, endPartitionOffsets_, pool_);
  }

  auto valuesBuffer = vector_->as<FlatVector<T>>()->values();
  partitionFixedWidthValues<T>(
      valuesBuffer,
      partitions,
      endPartitionOffsets_,
      numPartitions_,
      ctx,
      pool_);
}

template <typename T>
VectorPtr PartitionedFlatVector<T>::partitionAt(uint32_t partition) const {
  VELOX_CHECK_LT(partition, numPartitions_);

  vector_size_t beginOffset =
      partition == 0 ? 0 : rawEndPartitionOffsets_[partition - 1];
  vector_size_t numRowsInPartition =
      rawEndPartitionOffsets_[partition] - beginOffset;

  return vector_->slice(beginOffset, numRowsInPartition);
}

void PartitionedRowVector::partition(
    const std::vector<uint32_t>& partitions,
    PartitionBuildContext& ctx) {
  auto* rowVector = vector_->as<RowVector>();
  partitionedChildren_.reserve(rowVector->childrenSize());

  for (const auto& child : rowVector->children()) {
    partitionedChildren_.push_back(PartitionedVector::create(
        child, partitions, numPartitions_, endPartitionOffsets_, ctx, pool_));
  }

  if (numPartitions_ > 1) {
    Byte* rawNulls = reinterpret_cast<Byte*>(vector_->mutableRawNulls());
    if (rawNulls) {
      partitionBitsInPlace(
          rawNulls, partitions, numPartitions_, ctx, endPartitionOffsets_, pool_);
    }
  }
}

VectorPtr PartitionedRowVector::partitionAt(uint32_t partition) const {
  VELOX_CHECK_LT(partition, numPartitions_);

  vector_size_t beginOffset =
      partition == 0 ? 0 : rawEndPartitionOffsets_[partition - 1];
  vector_size_t numRowsInPartition =
      rawEndPartitionOffsets_[partition] - beginOffset;

  std::vector<VectorPtr> children;
  children.reserve(partitionedChildren_.size());
  for (const auto& child : partitionedChildren_) {
    children.push_back(child->partitionAt(partition));
  }

  BufferPtr nulls = nullptr;
  if (numRowsInPartition > 0 && vector_->rawNulls()) {
    nulls = AlignedBuffer::allocate<bool>(numRowsInPartition, pool_);
    bits::copyBits(
        vector_->rawNulls(),
        beginOffset,
        nulls->asMutable<uint64_t>(),
        0,
        numRowsInPartition);
  }

  return std::make_shared<RowVector>(
      pool_,
      vector_->type(),
      std::move(nulls),
      numRowsInPartition,
      std::move(children));
}

} // namespace facebook::velox
