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

#include "velox/serializers/CompactRowSerializer.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::serializer {

using TRowSize = uint32_t;

namespace detail {
struct RowHeader {
  int32_t uncompressedSize;
  int32_t compressedSize;
  bool compressed;

  static RowHeader read(ByteInputStream* source) {
    RowHeader header;
    header.uncompressedSize = source->read<int32_t>();
    header.compressedSize = source->read<int32_t>();
    header.compressed = source->read<char>();

    VELOX_CHECK_GE(header.uncompressedSize, 0);
    VELOX_CHECK_GE(header.compressedSize, 0);

    return header;
  }

  void write(OutputStream* out) {
    out->write(reinterpret_cast<char*>(&uncompressedSize), sizeof(int32_t));
    out->write(reinterpret_cast<char*>(&compressedSize), sizeof(int32_t));
    const char writeValue = compressed ? 1 : 0;
    out->write(reinterpret_cast<const char*>(&writeValue), sizeof(char));
  }

  static size_t size() {
    return sizeof(int32_t) * 2 + sizeof(char);
  }

  std::string debugString() const {
    return fmt::format(
        "uncompressedSize: {}, compressedSize: {}, compressed: {}",
        succinctBytes(uncompressedSize),
        succinctBytes(compressedSize),
        compressed);
  }
};
} // namespace detail

template <class Serializer>
class RowSerializer : public IterativeVectorSerializer {
 public:
  RowSerializer(memory::MemoryPool* pool, const VectorSerde::Options* options)
      : pool_(pool),
        options_(options == nullptr ? VectorSerde::Options() : *options),
        codec_(common::compressionKindToCodec(options_.compressionKind)) {}

  void append(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      Scratch& /*scratch*/) override {
    size_t totalSize = 0;
    const auto totalRows = std::accumulate(
        ranges.begin(),
        ranges.end(),
        0,
        [](vector_size_t sum, const auto& range) { return sum + range.size; });

    if (totalRows == 0) {
      return;
    }

    Serializer row(vector);
    std::vector<vector_size_t> rowSize(totalRows);
    if (auto fixedRowSize =
            Serializer::fixedRowSize(asRowType(vector->type()))) {
      totalSize += (fixedRowSize.value() + sizeof(TRowSize)) * totalRows;
      std::fill(rowSize.begin(), rowSize.end(), fixedRowSize.value());
    } else {
      vector_size_t index = 0;
      for (const auto& range : ranges) {
        for (auto i = 0; i < range.size; ++i, ++index) {
          rowSize[index] = row.rowSize(range.begin + i);
          totalSize += rowSize[index] + sizeof(TRowSize);
        }
      }
    }

    if (totalSize == 0) {
      return;
    }

    BufferPtr buffer = AlignedBuffer::allocate<char>(totalSize, pool_, 0);
    auto* rawBuffer = buffer->asMutable<char>();
    buffers_.push_back(std::move(buffer));

    serializeRanges(row, ranges, rawBuffer, rowSize);
  }

  void append(
      const Serializer& compactRow,
      const folly::Range<const vector_size_t*>& rows,
      const std::vector<vector_size_t>& sizes) override {
    size_t totalSize = 0;
    for (const auto row : rows) {
      totalSize += sizes[row];
    }
    if (totalSize == 0) {
      return;
    }

    BufferPtr buffer = AlignedBuffer::allocate<char>(totalSize, pool_, 0);
    auto* rawBuffer = buffer->asMutable<char>();
    buffers_.push_back(std::move(buffer));

    size_t offset = 0;
    for (auto& row : rows) {
      // Write row data.
      const TRowSize size =
          compactRow.serialize(row, rawBuffer + offset + sizeof(TRowSize));

      // Write raw size. Needs to be in big endian order.
      *(TRowSize*)(rawBuffer + offset) = folly::Endian::big(size);
      offset += sizeof(TRowSize) + size;
    }
  }

  size_t maxSerializedSize() const override {
    const auto size = uncompressedSize();
    if (!needCompression()) {
      return detail::RowHeader::size() + size;
    }
    VELOX_CHECK_LE(
        size,
        codec_->maxUncompressedLength(),
        "UncompressedSize exceeds limit");
    return detail::RowHeader::size() + codec_->maxCompressedLength(size);
  }

  /// The serialization format is | uncompressedSize | compressedSize |
  /// compressed | data.
  void flush(OutputStream* stream) override {
    constexpr int32_t kMaxCompressionAttemptsToSkip = 30;
    const auto size = uncompressedSize();
    if (!needCompression()) {
      flushUncompressed(size, stream);
    } else if (numCompressionToSkip_ > 0) {
      flushUncompressed(size, stream);
      stats_.compressionSkippedBytes += size;
      --numCompressionToSkip_;
      ++stats_.numCompressionSkipped;
    } else {
      // Compress the buffer if satisfied condition.
      const auto toCompress = toIOBuf(buffers_);
      const auto compressedBuffer = codec_->compress(toCompress.get());
      const int32_t compressedSize = compressedBuffer->length();
      stats_.compressionInputBytes += size;
      stats_.compressedBytes += compressedSize;
      if (compressedSize > options_.minCompressionRatio * size) {
        // Skip this compression.
        numCompressionToSkip_ = std::min<int64_t>(
            kMaxCompressionAttemptsToSkip, 1 + stats_.numCompressionSkipped);
        flushUncompressed(size, stream);
      } else {
        // Do the compression.
        detail::RowHeader header = {size, compressedSize, true};
        header.write(stream);
        for (auto range : *compressedBuffer) {
          stream->write(
              reinterpret_cast<const char*>(range.data()), range.size());
        }
      }
    }

    buffers_.clear();
  }

  std::unordered_map<std::string, RuntimeCounter> runtimeStats() override {
    std::unordered_map<std::string, RuntimeCounter> map;
    map.insert(
        {{"compressedBytes",
          RuntimeCounter(stats_.compressedBytes, RuntimeCounter::Unit::kBytes)},
         {"compressionInputBytes",
          RuntimeCounter(
              stats_.compressionInputBytes, RuntimeCounter::Unit::kBytes)},
         {"compressionSkippedBytes",
          RuntimeCounter(
              stats_.compressionSkippedBytes, RuntimeCounter::Unit::kBytes)}});
    return map;
  }

  void clear() override {}

 protected:
  virtual void serializeRanges(
      const Serializer& rowSerializer,
      const folly::Range<const IndexRange*>& ranges,
      char* rawBuffer,
      const std::vector<vector_size_t>& /*rowSize*/) {
    size_t offset = 0;
    for (auto& range : ranges) {
      for (auto row = range.begin; row < range.begin + range.size; ++row) {
        // Write row data.
        TRowSize size =
            rowSerializer.serialize(row, rawBuffer + offset + sizeof(TRowSize));

        // Write raw size. Needs to be in big endian order.
        *(TRowSize*)(rawBuffer + offset) = folly::Endian::big(size);
        offset += sizeof(TRowSize) + size;
      }
    }
  }

  memory::MemoryPool* const pool_;
  std::vector<BufferPtr> buffers_;

 private:
  std::unique_ptr<folly::IOBuf> toIOBuf(const std::vector<BufferPtr>& buffers) {
    std::unique_ptr<folly::IOBuf> iobuf;
    for (const auto& buffer : buffers) {
      auto newBuf =
          folly::IOBuf::wrapBuffer(buffer->asMutable<char>(), buffer->size());
      if (iobuf) {
        iobuf->prev()->appendChain(std::move(newBuf));
      } else {
        iobuf = std::move(newBuf);
      }
    }
    return iobuf;
  }

  int32_t uncompressedSize() const {
    int32_t totalSize = 0;
    for (const auto& buffer : buffers_) {
      totalSize += buffer->size();
    }
    return totalSize;
  }

  bool needCompression() const {
    return codec_->type() != folly::io::CodecType::NO_COMPRESSION;
  }

  void flushUncompressed(int32_t size, OutputStream* stream) {
    detail::RowHeader header = {size, size, false};
    header.write(stream);
    for (const auto& buffer : buffers_) {
      stream->write(buffer->template asMutable<char>(), buffer->size());
    }
  }

  const VectorSerde::Options options_;
  const std::unique_ptr<folly::io::Codec> codec_;
  // Count of forthcoming compressions to skip.
  int32_t numCompressionToSkip_{0};
  CompressionStats stats_;
};

template <typename SerializeView>
class RowDeserializer {
 public:
  static void deserialize(
      ByteInputStream* source,
      std::vector<SerializeView>& serializedRows,
      std::vector<std::unique_ptr<std::string>>& serializedBuffers,
      const VectorSerde::Options* options) {
    const auto compressionKind = options == nullptr
        ? VectorSerde::Options().compressionKind
        : options->compressionKind;
    while (!source->atEnd()) {
      std::unique_ptr<folly::IOBuf> uncompressedBuf;
      const auto header = detail::RowHeader::read(source);
      if (header.compressed) {
        VELOX_DCHECK_NE(
            compressionKind, common::CompressionKind::CompressionKind_NONE);
        auto compressBuf = folly::IOBuf::create(header.compressedSize);
        source->readBytes(compressBuf->writableData(), header.compressedSize);
        compressBuf->append(header.compressedSize);

        // Process chained uncompressed results IOBufs.
        const auto codec = common::compressionKindToCodec(compressionKind);
        uncompressedBuf =
            codec->uncompress(compressBuf.get(), header.uncompressedSize);
      }

      std::unique_ptr<ByteInputStream> uncompressedStream;
      ByteInputStream* uncompressedSource{nullptr};
      if (uncompressedBuf == nullptr) {
        uncompressedSource = source;
      } else {
        uncompressedStream = std::make_unique<BufferInputStream>(
            byteRangesFromIOBuf(uncompressedBuf.get()));
        uncompressedSource = uncompressedStream.get();
      }
      const std::streampos initialSize = uncompressedSource->tellp();
      while (uncompressedSource->tellp() - initialSize <
             header.uncompressedSize) {
        // First read row size in big endian order.
        const auto rowSize =
            folly::Endian::big(uncompressedSource->read<TRowSize>());

        auto serializedBuffer = std::make_unique<std::string>();
        serializedBuffer->reserve(rowSize);

        const auto row = uncompressedSource->nextView(rowSize);
        serializedBuffer->append(row.data(), row.size());
        // If we couldn't read the entire row at once, we need to concatenate it
        // in a different buffer.
        if (serializedBuffer->size() < rowSize) {
          concatenatePartialRow(uncompressedSource, rowSize, *serializedBuffer);
        }

        VELOX_CHECK_EQ(serializedBuffer->size(), rowSize);
        serializedBuffers.emplace_back(std::move(serializedBuffer));
        serializedRows.push_back(std::string_view(
            serializedBuffers.back()->data(),
            serializedBuffers.back()->size()));
      }
    }
  }

 private:
  // Read from the stream until the full row is concatenated.
  static void concatenatePartialRow(
      ByteInputStream* source,
      TRowSize rowSize,
      std::string& rowBuffer) {
    while (rowBuffer.size() < rowSize) {
      const std::string_view rowFragment =
          source->nextView(rowSize - rowBuffer.size());
      VELOX_CHECK_GT(
          rowFragment.size(),
          0,
          "Unable to read full serialized row. Needed {} but read {} bytes.",
          rowSize - rowBuffer.size(),
          rowFragment.size());
      rowBuffer.append(rowFragment.data(), rowFragment.size());
    }
  }
};

} // namespace facebook::velox::serializer
