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

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/common/memory/MappedMemory.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/StreamArena.h"
#include "velox/type/Type.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox {

class BaseVector;
class ByteStream;
class RowVector;

struct IndexRange {
  vector_size_t begin;
  vector_size_t size;
};

class VectorSerializer {
 public:
  virtual ~VectorSerializer() = default;

  virtual void append(
      std::shared_ptr<RowVector> vector,
      const folly::Range<const IndexRange*>& ranges) = 0;

  // Writes the contents to 'stream' in wire format
  virtual void flush(OutputStream* stream) = 0;
};

class VectorSerde {
 public:
  virtual ~VectorSerde() = default;

  virtual void estimateSerializedSize(
      std::shared_ptr<BaseVector> vector,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes) = 0;

  virtual std::unique_ptr<VectorSerializer> createSerializer(
      std::shared_ptr<const RowType> type,
      int32_t numRows,
      StreamArena* streamArena) = 0;

  virtual void deserialize(
      ByteStream* source,
      velox::memory::MemoryPool* pool,
      std::shared_ptr<const RowType> type,
      std::shared_ptr<RowVector>* result) = 0;
};

bool registerVectorSerde(std::unique_ptr<VectorSerde> serde);

bool isRegisteredVectorSerde();

#define _VELOX_REGISTER_VECTOR_SERDE_NAME(serde) registerVectorSerde_##serde

#define VELOX_DECLARE_VECTOR_SERDE(serde)             \
  void _VELOX_REGISTER_VECTOR_SERDE_NAME(serde)() {   \
    registerVectorSerde((std::make_unique<serde>())); \
  }

#define VELOX_REGISTER_VECTOR_SERDE(serde)                  \
  {                                                         \
    extern void _VELOX_REGISTER_VECTOR_SERDE_NAME(serde)(); \
    _VELOX_REGISTER_VECTOR_SERDE_NAME(serde)();             \
  }

class VectorStreamGroup : public StreamArena {
 public:
  explicit VectorStreamGroup(memory::MappedMemory* mappedMemory)
      : StreamArena(mappedMemory) {}

  void createStreamTree(std::shared_ptr<const RowType> type, int32_t numRows);

  static void estimateSerializedSize(
      std::shared_ptr<BaseVector> vector,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes);

  void append(
      std::shared_ptr<RowVector> vector,
      const folly::Range<const IndexRange*>& ranges);

  // Writes the contents to 'stream' in wire format.
  void flush(OutputStream* stream);

  uint64_t size() const override {
    if (iobuf_) {
      return iobufBytes_;
    }
    return StreamArena::size();
  }

  // Returns the IOBuf containing the final serialized data. The IOBuf chain has
  // shared ownership of 'this'. May be called several times and will return a
  // clone of the same chain.
  std::unique_ptr<folly::IOBuf> getIOBuf();

  // Reads data in wire format. Returns the RowVector in 'result'.
  static void read(
      ByteStream* source,
      velox::memory::MemoryPool* pool,
      std::shared_ptr<const RowType> type,
      std::shared_ptr<RowVector>* result);

 private:
  // Flushes the gathered content into a chain of IOBuf and frees any
  // outstanding state/memory. size() will hereafter return the byte
  // size of the payload in the IOBufs. Retrieve the data with
  // getIOBuf(). The IOBufs own have shared ownership of 'this'.
  void makeIOBuf();

  std::unique_ptr<VectorSerializer> serializer_;
  std::unique_ptr<folly::IOBuf> iobuf_;
  int64_t iobufBytes_{0};
};

} // namespace facebook::velox
