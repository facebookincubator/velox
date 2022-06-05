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

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/vector/BaseVector.h"

#include "velox/dwio/common/AbstractByteRleDecoder.h"

namespace facebook::velox::dwio::common::reader {
/**
 * Expand an array of bytes in place to the corresponding bigger.
 * Has to work backwards so that they data isn't clobbered during the
 * expansion.
 * @param buffer the array of chars and array of longs that need to be
 *        expanded
 * @param numValues the number of bytes to convert to longs
 */
template <typename From, typename To>
std::enable_if_t<std::is_same_v<From, bool>> expandBytes(
    To* buffer,
    uint64_t numValues) {
  for (size_t i = numValues - 1; i < numValues; --i) {
    buffer[i] = static_cast<To>(bits::isBitSet(buffer, i));
  }
}

template <typename From, typename To>
std::enable_if_t<std::is_same_v<From, int8_t>> expandBytes(
    To* buffer,
    uint64_t numValues) {
  auto from = reinterpret_cast<int8_t*>(buffer);
  for (size_t i = numValues - 1; i < numValues; --i) {
    buffer[i] = static_cast<To>(from[i]);
  }
}





/**
 * The interface for reading ORC data types.
 */
class ColumnReader {
 protected:
  explicit ColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& type, memory::MemoryPool& memoryPool) : notNullDecoder_{}, nodeType_{type}, memoryPool_{memoryPool} {}
  // Reads nulls, if any. Sets '*nulls' to nullptr if void
  // the reader has no nulls and there are no incoming
  //          nulls.Takes 'nulls' from 'result' if '*result' is non -
  //      null.Otherwise ensures that 'nulls' has a buffer of sufficient
  //          size and uses this.
  virtual void readNulls(
      vector_size_t numValues,
      const uint64_t* incomingNulls,
      VectorPtr* result,
      BufferPtr& nulls);

  virtual BufferPtr readNulls(
      vector_size_t numValues,
      VectorPtr& result,
      const uint64_t* incomingNulls);

  std::shared_ptr<AbstractByteRleDecoder> notNullDecoder_;

  // We use AbstractByteRleDecoder as an interface wrapper for ByteRleDecoder so that readNulls can be put in ColumnReader
  const std::shared_ptr<const dwio::common::TypeWithId> nodeType_;
  memory::MemoryPool& memoryPool_;

 public:

  virtual ~ColumnReader() = default;

  /**
   * Skip number of specified rows.
   * @param numValues the number of values to skip
   * @return the number of non-null values skipped
   */
  virtual uint64_t skip(uint64_t numValues);


  /**
   * Read the next group of values into a RowVector.
   * @param numValues the number of values to read
   * @param vector to read into
   */
  virtual void next(
      uint64_t numValues,
      VectorPtr& result,
      const uint64_t* nulls = nullptr) = 0;

};

namespace detail {

template <typename T>
inline void ensureCapacity(
    BufferPtr& data,
    size_t capacity,
    velox::memory::MemoryPool* pool) {
  if (!data || !data->unique() ||
      data->capacity() < BaseVector::byteSize<T>(capacity)) {
    data = AlignedBuffer::allocate<T>(capacity, pool);
  }
}

template <typename T>
inline T* resetIfWrongVectorType(VectorPtr& result) {
  if (result) {
    auto casted = result->as<T>();
    // We only expect vector to be used by a single thread.
    if (casted && result.use_count() == 1) {
      return casted;
    }
    result.reset();
  }
  return nullptr;
}

template <typename... T>
inline void resetIfNotWritable(VectorPtr& result, T&... buffer) {
  // The result vector and the buffer both hold reference, so refCount is at
  // least 2
  auto resetIfShared = [](auto& buffer) {
    const bool reset = buffer->refCount() > 2;
    if (reset) {
      buffer.reset();
    }
    return reset;
  };

  if ((... | resetIfShared(buffer))) {
    result.reset();
  }
}

// Helper method to build timestamps based on nulls/seconds/nanos
void fillTimestamps(
    Timestamp* timestamps,
    const uint64_t* nulls,
    const int64_t* seconds,
    const uint64_t* nanos,
    vector_size_t numValues);

} // namespace detail
} // namespace facebook::velox::dwio::common::reader
