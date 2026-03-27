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

#include <fmt/format.h>
#include <thrift/lib/cpp2/protocol/ProtocolReaderWithRefill.h>
#include <ostream>
#include <string_view>

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/parquet/thrift/gen-cpp2/parquet_types.h"
#include "velox/dwio/parquet/thrift/gen-cpp2/parquet_types_custom_protocol.h"

namespace facebook::velox::parquet::thrift {
template <
    typename Enum,
    bool IsEnum =
        std::is_same_v<typename apache::thrift::TEnumTraits<Enum>::type, Enum>>
fmt::underlying_t<Enum> format_as(Enum value) {
  return fmt::underlying(value);
}

template <
    typename Enum,
    bool IsEnum =
        std::is_same_v<typename apache::thrift::TEnumTraits<Enum>::type, Enum>>
std::ostream& operator<<(std::ostream& os, const Enum& value) {
  std::string_view name;
  if (apache::thrift::TEnumTraits<Enum>::findName(value, &name)) {
    os.write(name.data(), name.size());
  } else {
    os << static_cast<int>(value);
  }
  return os;
}

template <typename ThriftStruct>
unsigned long deserialize(ThriftStruct* thriftStruct, std::string_view data) {
  apache::thrift::CompactProtocolReader reader;
  folly::IOBuf buffer(
      folly::IOBuf::WRAP_BUFFER,
      folly::ByteRange(
          reinterpret_cast<const uint8_t*>(data.data()), data.size()));
  reader.setInput(&buffer);
  try {
    return thriftStruct->read(&reader);
  } catch (apache::thrift::protocol::TProtocolException& e) {
    VELOX_FAIL("Thrift deserialize error: {}", e.what());
  }
}

struct DeserializeResult {
  unsigned long readBytes;
  const uint8_t* remainedData;
  size_t remainedDataBytes;
  uint64_t readUs;
  // Holds the last buffer read from the refiller to keep remainedData valid.
  std::unique_ptr<folly::IOBuf> lastBuffer;
  // Track if we consumed data from the initial buffer or needed refills.
  bool usedRefiller;
  // If we used the refiller, store the actual stream position data.
  const void* streamData;
  int32_t streamDataBytes;
};

struct StreamReader {
  facebook::velox::dwio::common::SeekableInputStream* input;
  uint64_t& totalReadUs;
  const void*& lastStreamData;
  int32_t& lastStreamDataBytes;

  bool readNext(const void** data, int32_t* dataBytes) {
    bool haveData;
    uint64_t readUs{0};
    {
      MicrosecondTimer timer(&readUs);
      haveData = input->Next(data, dataBytes);
    }
    totalReadUs += readUs;
    // Track the last data read from stream
    if (haveData) {
      lastStreamData = *data;
      lastStreamDataBytes = *dataBytes;
    }
    return haveData;
  }
};

// Ensures we have initial data to start deserialization.
// If no initial data is provided, reads from the stream.
// Returns pair of (data pointer, size).
inline std::pair<const uint8_t*, size_t> ensureInitialData(
    StreamReader& reader,
    const uint8_t* initialData,
    size_t initialDataBytes) {
  if (initialDataBytes > 0) {
    return {initialData, initialDataBytes};
  }

  const void* buffer;
  int32_t size;
  reader.readNext(&buffer, &size);
  return {reinterpret_cast<const uint8_t*>(buffer), static_cast<size_t>(size)};
}

inline size_t calculateConsumedBytes(
    bool usedRefiller,
    size_t readBytes,
    int32_t totalBytesReadBeforeRefill,
    const uint8_t* coalescedBufferStart,
    size_t coalescedBufferSize,
    const uint8_t* remainedData) {
  if (!usedRefiller) {
    return readBytes;
  }

  if (!coalescedBufferStart || coalescedBufferSize == 0) {
    return readBytes;
  }

  const auto coalescedEnd = coalescedBufferStart + coalescedBufferSize;
  VELOX_CHECK(
      remainedData >= coalescedBufferStart && remainedData < coalescedEnd,
      "Cursor not in coalesced buffer range");

  size_t bytesConsumedFromCoalesced = remainedData - coalescedBufferStart;

  return totalBytesReadBeforeRefill + bytesConsumedFromCoalesced;
}

// Manages buffer refilling for Thrift deserialization with
// CompactProtocolReaderWithRefill. Ensures all deserialized data points to a
// single contiguous buffer by coalescing unconsumed bytes with newly read data.
//
// When the protocol reader needs more data, this refiller:
// 1. Reads new data from the stream
// 2. Creates a contiguous buffer containing unconsumed bytes + new data
// 3. Continues reading until requested bytes are available
//
// The coalesced buffer is necessary because Thrift deserialization may create
// pointers into the buffer that must remain valid throughout deserialization.
//
// Tracks metrics to calculate total bytes consumed from the stream:
// - totalBytesReadBeforeRefill: Bytes consumed from initial buffer
// - currentDataBytesInRefill: Unconsumed bytes when refiller was called
// - coalescedBufferStart/Size: Address range of the coalesced buffer
class ThriftStreamRefiller {
 public:
  ThriftStreamRefiller(
      StreamReader& streamReader,
      bool& usedRefiller,
      int32_t& totalBytesReadBeforeRefill,
      int32_t& currentDataBytesInRefill,
      const uint8_t*& coalescedBufferStart,
      size_t& coalescedBufferSize,
      std::unique_ptr<folly::IOBuf>& lastRefillBuffer)
      : streamReader_(streamReader),
        usedRefiller_(usedRefiller),
        totalBytesReadBeforeRefill_(totalBytesReadBeforeRefill),
        currentDataBytesInRefill_(currentDataBytesInRefill),
        coalescedBufferStart_(coalescedBufferStart),
        coalescedBufferSize_(coalescedBufferSize),
        lastRefillBuffer_(lastRefillBuffer) {}

  std::unique_ptr<folly::IOBuf> operator()(
      const uint8_t* currentData,
      int32_t currentDataBytes,
      int32_t totalBytesRead,
      int32_t requestedBytes) {
    usedRefiller_ = true;
    totalBytesReadBeforeRefill_ = totalBytesRead;
    currentDataBytesInRefill_ = currentDataBytes;

    const void* data;
    int32_t dataBytes{0};
    if (!streamReader_.readNext(&data, &dataBytes) || dataBytes == 0) {
      // Return nullptr to signal end of stream
      return nullptr;
    }

    auto coalescedBuffer = createCoalescedBuffer(
        currentData, currentDataBytes, data, dataBytes, requestedBytes);

    coalescedBufferStart_ = coalescedBuffer->data();
    coalescedBufferSize_ = coalescedBuffer->length();

    lastRefillBuffer_ = std::move(coalescedBuffer);
    return lastRefillBuffer_->clone();
  }

 private:
  static void appendToContiguousBuffer(
      folly::IOBuf* buffer,
      const void* data,
      size_t dataBytes) {
    buffer->reserve(0, dataBytes);
    memcpy(buffer->writableTail(), data, dataBytes);
    buffer->append(dataBytes);
  }

  // Creates a contiguous buffer that includes:
  // 1. The unconsumed bytes from currentData
  // 2. The new data just read from the stream
  // 3. Additional data read until requestedBytes is satisfied
  // This ensures all deserialized data points to a single stable buffer.
  std::unique_ptr<folly::IOBuf> createCoalescedBuffer(
      const uint8_t* currentData,
      int32_t currentDataBytes,
      const void* initialData,
      int32_t initialDataBytes,
      int32_t requestedBytes) {
    std::unique_ptr<folly::IOBuf> coalescedBuffer;
    size_t totalSize = currentDataBytes + initialDataBytes;

    if (currentDataBytes > 0) {
      coalescedBuffer = folly::IOBuf::copyBuffer(currentData, currentDataBytes);
      appendToContiguousBuffer(
          coalescedBuffer.get(), initialData, initialDataBytes);
    } else {
      coalescedBuffer = folly::IOBuf::copyBuffer(initialData, initialDataBytes);
    }

    while (totalSize < requestedBytes) {
      const void* data = nullptr;
      int32_t dataBytes = 0;
      if (!streamReader_.readNext(&data, &dataBytes) || dataBytes == 0) {
        break;
      }

      appendToContiguousBuffer(coalescedBuffer.get(), data, dataBytes);
      totalSize += dataBytes;
    }

    return coalescedBuffer;
  }

  StreamReader& streamReader_;
  bool& usedRefiller_;
  int32_t& totalBytesReadBeforeRefill_;
  int32_t& currentDataBytesInRefill_;
  const uint8_t*& coalescedBufferStart_;
  size_t& coalescedBufferSize_;
  std::unique_ptr<folly::IOBuf>& lastRefillBuffer_;
};

template <typename ThriftStruct>
DeserializeResult deserialize(
    ThriftStruct* thriftStruct,
    facebook::velox::dwio::common::SeekableInputStream* input,
    const uint8_t* initialData,
    size_t initialDataBytes) {
  uint64_t totalReadUs{0};
  std::unique_ptr<folly::IOBuf> lastRefillBuffer;
  bool usedRefiller = false;
  const void* lastStreamData = initialData;
  int32_t lastStreamDataBytes = initialDataBytes;
  int totalBytesReadBeforeRefill = 0;
  int currentDataBytesInRefill = 0;
  const uint8_t* coalescedBufferStart = nullptr;
  size_t coalescedBufferSize = 0;

  StreamReader streamReader{
      input, totalReadUs, lastStreamData, lastStreamDataBytes};

  auto [data, size] =
      ensureInitialData(streamReader, initialData, initialDataBytes);
  initialData = data;
  initialDataBytes = size;

  ThriftStreamRefiller refiller(
      streamReader,
      usedRefiller,
      totalBytesReadBeforeRefill,
      currentDataBytesInRefill,
      coalescedBufferStart,
      coalescedBufferSize,
      lastRefillBuffer);

  apache::thrift::CompactProtocolReaderWithRefill reader(std::ref(refiller));
  folly::IOBuf initialBuffer(
      folly::IOBuf::WRAP_BUFFER, initialData, initialDataBytes);

  reader.setInput(&initialBuffer);
  try {
    DeserializeResult result;
    result.readBytes = thriftStruct->read(&reader);

    auto cursor = reader.getCursor();
    result.remainedData = cursor.data();
    result.remainedDataBytes = cursor.length();
    result.readUs = totalReadUs;
    result.lastBuffer = std::move(lastRefillBuffer);
    result.usedRefiller = usedRefiller;
    result.streamData = lastStreamData;
    result.streamDataBytes = lastStreamDataBytes;

    result.readBytes = calculateConsumedBytes(
        usedRefiller,
        result.readBytes,
        totalBytesReadBeforeRefill,
        coalescedBufferStart,
        coalescedBufferSize,
        result.remainedData);

    return result;
  } catch (const std::exception& e) {
    VELOX_FAIL("Thrift deserialize error: {}", e.what());
  }
}

template <typename ThriftStruct>
uint32_t serialize(
    const ThriftStruct& thriftStruct,
    folly::IOBufQueue* buffer) {
  apache::thrift::CompactProtocolWriter writer;
  writer.setOutput(buffer);
  return thriftStruct.write(&writer);
}
}; // namespace facebook::velox::parquet::thrift
