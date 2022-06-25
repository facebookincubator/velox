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
#include "velox/dwio/dwrf/common/ByteRLE.h"
#include "velox/dwio/dwrf/common/Compression.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/reader/AbstractColumnReader.h"
#include "velox/dwio/dwrf/reader/EncodingContext.h"
#include "velox/dwio/dwrf/reader/StripeStream.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::dwrf {

/**
 * The interface for reading ORC data types.
 */
class ColumnReader : public AbstractColumnReader {
 public:
  explicit ColumnReader(
      memory::MemoryPool& memoryPool,
      const std::shared_ptr<const dwio::common::TypeWithId>& type)
      : AbstractColumnReader(memoryPool, type),
        flatMapContext_(FlatMapContext::nonFlatMapContext()) {}

  ColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> nodeId,
      StripeStreams& stripe,
      FlatMapContext flatMapContext = FlatMapContext::nonFlatMapContext());

  /**
   * Skip number of specified rows.
   * @param numValues the number of values to skip
   * @return the number of non-null values skipped
   */
  virtual uint64_t skip(uint64_t numValues) override;

  /**
   * Create a reader for the given stripe.
   */
  static std::unique_ptr<ColumnReader> build(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      StripeStreams& stripe,
      FlatMapContext flatMapContext = FlatMapContext::nonFlatMapContext());

 protected:
  // Reads nulls, if any. Sets '*nulls' to nullptr if void
  // the reader has no nulls and there are no incoming
  //          nulls.Takes 'nulls' from 'result' if '*result' is non -
  //      null.Otherwise ensures that 'nulls' has a buffer of sufficient
  //          size and uses this.
  void readNulls(
      vector_size_t numValues,
      const uint64_t* incomingNulls,
      VectorPtr* result,
      BufferPtr& nulls);

  // Shorthand for long form of readNulls for use in next().
  BufferPtr readNulls(
      vector_size_t numValues,
      VectorPtr& result,
      const uint64_t* incomingNulls);

  std::unique_ptr<ByteRleDecoder> notNullDecoder_;
  FlatMapContext flatMapContext_;
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
} // namespace facebook::velox::dwrf
