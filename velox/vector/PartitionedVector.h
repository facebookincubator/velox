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

#include <vector>

#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox {

class PartitionedVector;
using PartitionedVectorPtr = std::shared_ptr<PartitionedVector>;

namespace {

// TODO: This was copied from dwio::common::BufferUtil.h. However the vector
// module should not depend on dwio. Move this to a common place
template <typename T>
inline void ensureCapacity(
    BufferPtr& data,
    size_t numElements,
    velox::memory::MemoryPool* pool,
    bool preserveOldData = false,
    bool clearBits = false) {
  size_t oldSize = 0;
  size_t newCapacity = BaseVector::byteSize<T>(numElements);
  if (!data) {
    data = AlignedBuffer::allocate<T>(numElements, pool);
  } else {
    oldSize = data->size();
    if (!data->isMutable() || data->capacity() < newCapacity) {
      auto newData = AlignedBuffer::allocate<T>(numElements, pool);
      if (preserveOldData) {
        std::memcpy(
            newData->template asMutable<uint8_t>(),
            data->as<uint8_t>(),
            oldSize);
      }
      data = newData;
    }
  }

  if (clearBits && newCapacity > oldSize) {
    std::memset(
        (void*)(data->asMutable<int8_t>() + oldSize),
        0L,
        newCapacity - oldSize);
  }
}

} // namespace

/// Construction-time context used to build a PartitionedVector.
///
/// This struct contains only transient execution context needed during
/// construction. None of the fields here define the logical state of
/// PartitionedVector and none are retained after create().
/// All fields are only valid during the PartitionedVector::create() call.
struct PartitionBuildContext {
  BufferPtr cursorPartitionOffsets = nullptr;

  PartitionBuildContext() = default;
};

/// PartitionedVector provides an in-place, partition-aware layout of a vector
/// based on per-row partition IDs.
///
/// This is a low-level execution abstraction, analogous to DecodedVector:
/// - it owns partitioning metadata (offsets, indices)
/// - it does not encode operator-specific semantics
/// - it is intended to be reused by multiple exec components
///   (aggregation, sorting, shuffle, etc.)
///
/// The partitioning operation rearranges rows so that rows belonging to the
/// same partition occupy a contiguous range.
///
/// Thread-safety:
///   This class is NOT thread-safe. All methods must be called from a single
///   thread. Internal buffers are mutated during create().
class PartitionedVector {
 public:
  /// Disable default constructor.
  PartitionedVector() = delete;

  /// Disable copy constructor and assignment.
  PartitionedVector(const PartitionedVector& other) = delete;
  PartitionedVector& operator=(const PartitionedVector& other) = delete;

  // Use default move constructor and move assignment operator.
  PartitionedVector(PartitionedVector&&) noexcept = default;
  PartitionedVector& operator=(PartitionedVector&&) noexcept = default;

  /// Virtual destructor.
  virtual ~PartitionedVector();

  /// Factory method to create a PartitionedVector. This is the main entry point
  /// for constructing a PartitionedVector. The partitioning operation
  /// rearranges rows in the base vector so that rows belonging to the same
  /// partition occupy a contiguous range.
  ///
  /// Params:
  /// - vector: the base vector to be partitioned. This is modified during
  ///   partitioning, and becomes the underlying vector of the created
  ///   PartitionedVector.
  /// - partitions: a vector of partition IDs for each row in the base vector.
  ///   The length of this vector must be the same as the number of rows in the
  ///   base vector. Each entry must be a value between 0 and numPartitions - 1.
  /// - numPartitions: the total number of partitions. This must be greater than
  ///   0.
  /// - ctx: the context object for building the partitioned vector. This
  ///   contains transient execution context needed during construction, such as
  ///   intermediate buffers. None of the fields in this context define the
  ///   logical state of the PartitionedVector, and none are retained after
  ///   create(). All fields in this context are only valid during the create()
  ///   call.
  /// - pool: the memory pool for allocating any necessary buffers during the
  ///   creation of the PartitionedVector.
  static PartitionedVectorPtr create(
      const VectorPtr& vector,
      const std::vector<uint32_t>& partitions,
      uint32_t numPartitions,
      PartitionBuildContext& ctx,
      velox::memory::MemoryPool* pool);

  /// Returns the underlying vector.
  VectorPtr baseVector() const;

  /// Returns the partitioned vector at partition p. If the number of rows in
  /// that partition is 0, returns an empty vector.
  virtual VectorPtr partitionAt(uint32_t partition) const = 0;

  template <typename T>
  T* as() {
    static_assert(std::is_base_of_v<PartitionedVector, T>);
    return dynamic_cast<T*>(this);
  }

  TypeKind typeKind() const {
    return vector_->typeKind();
  }

  vector_size_t* rawPartitionOffsets() {
    return rawEndPartitionOffsets_;
  }

  virtual const vector_size_t* rawSizes() = 0;

  /// Returns string representation of the value in the specified row.
  virtual std::string toString() const;

 protected:
  // Internal create method that accepts pre-computed endPartitionOffsets
  // buffer.
  static PartitionedVectorPtr create(
      const VectorPtr& vector,
      const std::vector<uint32_t>& partitions,
      uint32_t numPartitions,
      const BufferPtr& partitionOffsetsBuffer,
      PartitionBuildContext& ctx,
      velox::memory::MemoryPool* pool);

  PartitionedVector(
      const VectorPtr& vector,
      uint32_t numPartitions,
      const BufferPtr& endPartitionOffsets,
      velox::memory::MemoryPool* pool)
      : vector_(vector),
        numPartitions_(numPartitions),
        endPartitionOffsets_(endPartitionOffsets),
        pool_(pool) {
    VELOX_CHECK_NOT_NULL(vector_);
    VELOX_CHECK_GT(numPartitions_, 0);
    VELOX_CHECK_NOT_NULL(endPartitionOffsets_);
    VELOX_CHECK_EQ(
        endPartitionOffsets_->size(), numPartitions_ * sizeof(vector_size_t));
    VELOX_CHECK_NOT_NULL(pool_);

    rawEndPartitionOffsets_ = endPartitionOffsets_->asMutable<vector_size_t>();
  }

  virtual void partition(
      const std::vector<uint32_t>& partitions,
      PartitionBuildContext& ctx) = 0;

  // The base vector that is being partitioned. This is modified during
  // partitioning.
  VectorPtr vector_;

  // Total number of partitions. This is set at construction and does not change
  // during partitioning. It doesn't have const quantifier because we want to
  // allow move assignment operator.
  uint32_t numPartitions_;

  // The cumulative end row offsets for each partition. For example, if there
  // are 3 partitions with 2, 3, and 1 rows respectively, then
  // endPartitionOffsets_[0] = 2, endPartitionOffsets_[1] = 5, and
  // endPartitionOffsets_[2] = 6.
  BufferPtr endPartitionOffsets_;

  // The raw pointer to the endPartitionOffsets_ buffer for easy access during
  // partitioning.
  vector_size_t* rawEndPartitionOffsets_;

  velox::memory::MemoryPool* pool_;
};

using PartitionedVectorPtr = std::shared_ptr<PartitionedVector>;

template <typename T>
class PartitionedFlatVector : public PartitionedVector {
 public:
  PartitionedFlatVector(
      const VectorPtr& flatVector,
      uint32_t numPartitions,
      const BufferPtr& partitionOffsets,
      velox::memory::MemoryPool* pool)
      : PartitionedVector(flatVector, numPartitions, partitionOffsets, pool) {}

  void partition(
      const std::vector<uint32_t>& partitions,
      PartitionBuildContext& ctx) override;

  VectorPtr partitionAt(uint32_t partition) const override;

  const vector_size_t* rawSizes() override {
    VELOX_UNREACHABLE("PartitionedFlatVector does not implement rawSizes()");
  }
};

} // namespace facebook::velox
