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
#include "velox/serializers/UnsafeRowSerializer.h"
#include "velox/row/UnsafeRowDeserializers.h"
#include "velox/row/UnsafeRowSerializers.h"

namespace facebook::velox::serializer::spark {

void UnsafeRowVectorSerde::estimateSerializedSize(
    VectorPtr /* vector */,
    const folly::Range<const IndexRange*>& /* ranges */,
    vector_size_t** /* sizes */) {
  VELOX_UNSUPPORTED();
}

namespace {
class UnsafeRowVectorSerializer : public VectorSerializer {
 public:
  explicit UnsafeRowVectorSerializer(StreamArena* streamArena)
      : pool_{streamArena->pool()} {}

  void append(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges) override {
    size_t totalSize = 0;
    for (auto& range : ranges) {
      for (auto i = range.begin; i < range.begin + range.size; ++i) {
        auto rowSize = velox::row::UnsafeRowDynamicSerializer::getSizeRow(
            vector->type(), vector.get(), i);
        totalSize += rowSize + sizeof(size_t);
      }
    }

    if (totalSize == 0) {
      return;
    }

    auto* buffer = (char*)pool_->allocate(totalSize);
    buffers_.push_back(
        ByteRange{(uint8_t*)buffer, (int32_t)totalSize, (int32_t)totalSize});

    size_t offset = 0;
    for (auto& range : ranges) {
      for (auto i = range.begin; i < range.begin + range.size; ++i) {
        // Write row data.
        auto rowSize = velox::row::UnsafeRowDynamicSerializer::getSizeRow(
            vector->type(), vector.get(), i);
        auto size =
            velox::row::UnsafeRowDynamicSerializer::serialize(
                vector->type(), vector, buffer + offset + sizeof(size_t), i)
                .value_or(0);

        // Sanity check.
        VELOX_CHECK_EQ(rowSize, size);

        // Write raw size.
        *(size_t*)(buffer + offset) = size;

        offset += sizeof(size_t) + size;
      }
    }
  }

  void flush(OutputStream* stream) override {
    for (auto& buffer : buffers_) {
      stream->write((char*)buffer.buffer, buffer.position);
      pool_->free(buffer.buffer, buffer.size);
    }
    buffers_.clear();
  }

 private:
  memory::MemoryPool* const FOLLY_NONNULL pool_;
  std::vector<ByteRange> buffers_;
};
} // namespace

std::unique_ptr<VectorSerializer> UnsafeRowVectorSerde::createSerializer(
    RowTypePtr /* type */,
    int32_t /* numRows */,
    StreamArena* streamArena,
    const Options* /* options */) {
  return std::make_unique<UnsafeRowVectorSerializer>(streamArena);
}

void UnsafeRowVectorSerde::deserialize(
    ByteStream* source,
    velox::memory::MemoryPool* pool,
    RowTypePtr type,
    RowVectorPtr* result,
    const Options* /* options */) {
  std::vector<std::optional<std::string_view>> serializedRows;
  while (!source->atEnd()) {
    auto rowSize = source->read<size_t>();
    auto row = source->nextView(rowSize);
    VELOX_CHECK_EQ(row.size(), rowSize);
    serializedRows.push_back(row);
  }

  if (serializedRows.empty()) {
    *result = BaseVector::create<RowVector>(type, 0, pool);
    return;
  }

  *result = std::dynamic_pointer_cast<RowVector>(
      velox::row::UnsafeRowDynamicVectorBatchDeserializer::deserializeComplex(
          serializedRows, type, pool));
}

// static
void UnsafeRowVectorSerde::registerVectorSerde() {
  velox::registerVectorSerde(std::make_unique<UnsafeRowVectorSerde>());
}

} // namespace facebook::velox::serializer::spark
