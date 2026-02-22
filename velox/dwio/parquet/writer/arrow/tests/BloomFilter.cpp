/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// Adapted from Apache Arrow.

#include <cstdint>
#include <cstring>
#include <memory>

#include "arrow/result.h"
#include "arrow/util/macros.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"
#include "velox/dwio/parquet/writer/arrow/tests/BloomFilter.h"
#include "velox/dwio/parquet/writer/arrow/tests/XxHasher.h"

namespace facebook::velox::parquet::arrow {

BlockSplitBloomFilter::BlockSplitBloomFilter(::arrow::MemoryPool* pool)
    : pool_(pool),
      hashStrategy_(HashStrategy::XXHASH),
      algorithm_(Algorithm::BLOCK),
      compressionStrategy_(CompressionStrategy::UNCOMPRESSED) {}

void BlockSplitBloomFilter::init(uint32_t numBytes) {
  if (numBytes < kMinimumBloomFilterBytes) {
    numBytes = kMinimumBloomFilterBytes;
  }

  // Get next power of 2 if it is not power of 2.
  if ((numBytes & (numBytes - 1)) != 0) {
    numBytes = static_cast<uint32_t>(::arrow::bit_util::NextPower2(numBytes));
  }

  if (numBytes > kMaximumBloomFilterBytes) {
    numBytes = kMaximumBloomFilterBytes;
  }

  numBytes_ = numBytes;
  PARQUET_ASSIGN_OR_THROW(data_, ::arrow::AllocateBuffer(numBytes_, pool_));
  memset(data_->mutable_data(), 0, numBytes_);

  this->hasher_ = std::make_unique<XxHasher>();
}

void BlockSplitBloomFilter::init(const uint8_t* bitset, uint32_t numBytes) {
  VELOX_DCHECK_NOT_NULL(bitset);

  if (numBytes < kMinimumBloomFilterBytes ||
      numBytes > kMaximumBloomFilterBytes || (numBytes & (numBytes - 1)) != 0) {
    throw ParquetException("Given length of bitset is illegal");
  }

  numBytes_ = numBytes;
  PARQUET_ASSIGN_OR_THROW(data_, ::arrow::AllocateBuffer(numBytes_, pool_));
  memcpy(data_->mutable_data(), bitset, numBytes_);

  this->hasher_ = std::make_unique<XxHasher>();
}

static constexpr uint32_t kBloomFilterHeaderSizeGuess = 256;

static ::arrow::Status validateBloomFilterHeader(
    const facebook::velox::parquet::thrift::BloomFilterHeader& header) {
  if (!header.algorithm.__isset.BLOCK) {
    return ::arrow::Status::Invalid(
        "Unsupported Bloom filter algorithm: ", header.algorithm, ".");
  }

  if (!header.hash.__isset.XXHASH) {
    return ::arrow::Status::Invalid(
        "Unsupported Bloom filter hash: ", header.hash, ".");
  }

  if (!header.compression.__isset.UNCOMPRESSED) {
    return ::arrow::Status::Invalid(
        "Unsupported Bloom filter compression: ", header.compression, ".");
  }

  if (header.numBytes <= 0 ||
      static_cast<uint32_t>(header.numBytes) >
          BloomFilter::kMaximumBloomFilterBytes) {
    std::stringstream ss;
    ss << "Bloom filter size is incorrect: " << header.numBytes
       << ". Must be in range (" << 0 << ", "
       << BloomFilter::kMaximumBloomFilterBytes << "].";
    return ::arrow::Status::Invalid(ss.str());
  }

  return ::arrow::Status::OK();
}

BlockSplitBloomFilter BlockSplitBloomFilter::deserialize(
    const ReaderProperties& properties,
    ArrowInputStream* input) {
  // NOTE: we don't know the bloom filter header size upfront, and we can't
  // rely. On InputStream::Peek() which isn't always implemented. Therefore, we
  // must. First Read() with an upper bound estimate of the header size, then
  // once we. Know the bloom filter data size, we can Read() the exact number
  // of. Remaining data bytes.
  ThriftDeserializer deserializer(properties);
  facebook::velox::parquet::thrift::BloomFilterHeader header;

  // Read and deserialize bloom filter header.
  PARQUET_ASSIGN_OR_THROW(
      auto headerBuf, input->Read(kBloomFilterHeaderSizeGuess));
  // This gets used, then set by DeserializeThriftMsg.
  uint32_t headerSize = static_cast<uint32_t>(headerBuf->size());
  try {
    deserializer.deserializeMessage(
        reinterpret_cast<const uint8_t*>(headerBuf->data()),
        &headerSize,
        &header);
    VELOX_DCHECK_LE(headerSize, headerBuf->size());
  } catch (std::exception& e) {
    std::stringstream ss;
    ss << "Deserializing bloom filter header failed.\n" << e.what();
    throw ParquetException(ss.str());
  }
  PARQUET_THROW_NOT_OK(validateBloomFilterHeader(header));

  const int32_t bloomFilterSize = header.numBytes;
  if (bloomFilterSize + headerSize <= headerBuf->size()) {
    // The bloom filter data is entirely contained in the buffer we just read.
    // => Just return it.
    BlockSplitBloomFilter bloomFilter(properties.memoryPool());
    bloomFilter.init(headerBuf->data() + headerSize, bloomFilterSize);
    return bloomFilter;
  }
  // We have read a part of the bloom filter already, copy it to the target.
  // Buffer and read the remaining part from the InputStream.
  auto buffer = allocateBuffer(properties.memoryPool(), bloomFilterSize);

  const auto bloomFilterBytesInHeader = headerBuf->size() - headerSize;
  if (bloomFilterBytesInHeader > 0) {
    std::memcpy(
        buffer->mutable_data(),
        headerBuf->data() + headerSize,
        bloomFilterBytesInHeader);
  }

  const auto requiredReadSize = bloomFilterSize - bloomFilterBytesInHeader;
  PARQUET_ASSIGN_OR_THROW(
      auto readSize,
      input->Read(
          requiredReadSize, buffer->mutable_data() + bloomFilterBytesInHeader));
  if (ARROW_PREDICT_FALSE(readSize < requiredReadSize)) {
    throw ParquetException("Bloom Filter read failed: not enough data");
  }
  BlockSplitBloomFilter bloomFilter(properties.memoryPool());
  bloomFilter.init(buffer->data(), bloomFilterSize);
  return bloomFilter;
}

void BlockSplitBloomFilter::writeTo(ArrowOutputStream* sink) const {
  VELOX_DCHECK_NOT_NULL(sink);

  facebook::velox::parquet::thrift::BloomFilterHeader header;
  if (ARROW_PREDICT_FALSE(algorithm_ != BloomFilter::Algorithm::BLOCK)) {
    throw ParquetException(
        "BloomFilter does not support Algorithm other than BLOCK");
  }
  header.algorithm.__set_BLOCK(
      facebook::velox::parquet::thrift::SplitBlockAlgorithm());
  if (ARROW_PREDICT_FALSE(hashStrategy_ != HashStrategy::XXHASH)) {
    throw ParquetException(
        "BloomFilter does not support Hash other than XXHASH");
  }
  header.hash.__set_XXHASH(facebook::velox::parquet::thrift::XxHash());
  if (ARROW_PREDICT_FALSE(
          compressionStrategy_ != CompressionStrategy::UNCOMPRESSED)) {
    throw ParquetException(
        "BloomFilter does not support Compression other than UNCOMPRESSED");
  }
  header.compression.__set_UNCOMPRESSED(
      facebook::velox::parquet::thrift::Uncompressed());
  header.__set_numBytes(numBytes_);

  ThriftSerializer serializer;
  serializer.serialize(&header, sink);

  PARQUET_THROW_NOT_OK(sink->Write(data_->data(), numBytes_));
}

bool BlockSplitBloomFilter::findHash(uint64_t hash) const {
  const uint32_t bucketIndex = static_cast<uint32_t>(
      ((hash >> 32) * (numBytes_ / kBytesPerFilterBlock)) >> 32);
  const uint32_t key = static_cast<uint32_t>(hash);
  const uint32_t* bitset32 = reinterpret_cast<const uint32_t*>(data_->data());

  for (int i = 0; i < kBitsSetPerBlock; ++i) {
    // Calculate mask for key in the given bitset.
    const uint32_t mask = UINT32_C(0x1) << ((key * SALT[i]) >> 27);
    if (ARROW_PREDICT_FALSE(
            0 == (bitset32[kBitsSetPerBlock * bucketIndex + i] & mask))) {
      return false;
    }
  }
  return true;
}

void BlockSplitBloomFilter::insertHashImpl(uint64_t hash) {
  const uint32_t bucketIndex = static_cast<uint32_t>(
      ((hash >> 32) * (numBytes_ / kBytesPerFilterBlock)) >> 32);
  const uint32_t key = static_cast<uint32_t>(hash);
  uint32_t* bitset32 = reinterpret_cast<uint32_t*>(data_->mutable_data());

  for (int i = 0; i < kBitsSetPerBlock; i++) {
    // Calculate mask for key in the given bitset.
    const uint32_t mask = UINT32_C(0x1) << ((key * SALT[i]) >> 27);
    bitset32[bucketIndex * kBitsSetPerBlock + i] |= mask;
  }
}

void BlockSplitBloomFilter::insertHash(uint64_t hash) {
  insertHashImpl(hash);
}

void BlockSplitBloomFilter::insertHashes(
    const uint64_t* hashes,
    int numValues) {
  for (int i = 0; i < numValues; ++i) {
    insertHashImpl(hashes[i]);
  }
}

} // namespace facebook::velox::parquet::arrow
