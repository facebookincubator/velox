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
#include <folly/lang/Bits.h>
#include "velox/row/UnsafeRowDeserializers.h"
#include "velox/row/UnsafeRowFast.h"

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
  using TRowSize = uint32_t;

  explicit UnsafeRowVectorSerializer(StreamArena* streamArena)
      : pool_{streamArena->pool()} {}

  void append(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges) override {
    size_t totalSize = 0;
    row::UnsafeRowFast unsafeRow(vector);
    if (auto fixedRowSize =
            row::UnsafeRowFast::fixedRowSize(asRowType(vector->type()))) {
      for (const auto& range : ranges) {
        totalSize += (fixedRowSize.value() + sizeof(TRowSize)) * range.size;
      }

    } else {
      for (const auto& range : ranges) {
        for (auto i = range.begin; i < range.begin + range.size; ++i) {
          totalSize += unsafeRow.rowSize(i) + sizeof(TRowSize);
        }
      }
    }

    if (totalSize == 0) {
      return;
    }

    BufferPtr buffer = AlignedBuffer::allocate<char>(totalSize, pool_, 0);
    auto rawBuffer = buffer->asMutable<char>();
    buffers_.push_back(std::move(buffer));

    size_t offset = 0;
    for (auto& range : ranges) {
      for (auto i = range.begin; i < range.begin + range.size; ++i) {
        // Write row data.
        TRowSize size =
            unsafeRow.serialize(i, rawBuffer + offset + sizeof(TRowSize));

        // Write raw size. Needs to be in big endian order.
        *(TRowSize*)(rawBuffer + offset) = folly::Endian::big(size);
        offset += sizeof(TRowSize) + size;
      }
    }
  }

  void flush(OutputStream* stream) override {
    for (const auto& buffer : buffers_) {
      stream->write(buffer->as<char>(), buffer->size());
    }
    buffers_.clear();
  }

 private:
  memory::MemoryPool* const FOLLY_NONNULL pool_;
  std::vector<BufferPtr> buffers_;
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
    // First read row size in big endian order.
    auto rowSize =
        folly::Endian::big(source->read<UnsafeRowVectorSerializer::TRowSize>());
    auto row = source->nextView(rowSize);
    VELOX_CHECK_EQ(row.size(), rowSize);
    serializedRows.push_back(row);
  }

  if (serializedRows.empty()) {
    *result = BaseVector::create<RowVector>(type, 0, pool);
    return;
  }

  *result = std::dynamic_pointer_cast<RowVector>(
      velox::row::UnsafeRowDeserializer::deserialize(
          serializedRows, type, pool));
}

// static
void UnsafeRowVectorSerde::registerVectorSerde() {
  velox::registerVectorSerde(std::make_unique<UnsafeRowVectorSerde>());
}

} // namespace facebook::velox::serializer::spark
