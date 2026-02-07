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

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/parquet/thrift/gen-cpp2/parquet_types.h"
#include "velox/dwio/parquet/thrift/gen-cpp2/parquet_types_custom_protocol.h"

#include <fmt/format.h>
#include <thrift/lib/cpp2/protocol/ProtocolReaderWithRefill.h>
#include <ostream>
#include <string_view>

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
};

template <typename ThriftStruct>
DeserializeResult deserialize(
    ThriftStruct* thriftStruct,
    facebook::velox::dwio::common::SeekableInputStream* input,
    const uint8_t* initialData,
    size_t initialDataBytes) {
  uint64_t totalReadUs{0};
  auto readData = [&](const void** data, int32_t* dataBytes) {
    bool haveData;
    uint64_t readUs{0};
    {
      MicrosecondTimer timer(&readUs);
      haveData = input->Next(data, dataBytes);
    }
    totalReadUs += readUs;
    return haveData;
  };
  auto refiller = [&](const uint8_t* currentData,
                      int currentDataBytes,
                      int totalBytesRead,
                      int requestedBytes) -> std::unique_ptr<folly::IOBuf> {
    std::unique_ptr<folly::IOBuf> buffer;
    if (currentDataBytes == 0) {
      const void* data;
      int32_t dataBytes;
      if (!readData(&data, &dataBytes)) {
        return folly::IOBuf::wrapBuffer(nullptr, 0);
      }
      if (dataBytes >= requestedBytes) {
        return folly::IOBuf::wrapBuffer(data, dataBytes);
      }
      buffer = folly::IOBuf::copyBuffer(data, dataBytes);
    } else {
      buffer = folly::IOBuf::copyBuffer(currentData, currentDataBytes);
    }
    while (true) {
      const void* data;
      int32_t dataBytes;
      if (!readData(&data, &dataBytes)) {
        break;
      }
      std::unique_ptr<folly::IOBuf> moreBuffer;
      auto currentBytes = buffer->computeChainCapacity();
      if (currentBytes + dataBytes >= requestedBytes) {
        moreBuffer = folly::IOBuf::wrapBuffer(data, dataBytes);
      } else {
        moreBuffer = folly::IOBuf::copyBuffer(data, dataBytes);
      }
      buffer->appendToChain(std::move(moreBuffer));
      if (currentBytes + dataBytes >= requestedBytes) {
        break;
      }
    }
    return buffer;
  };
  apache::thrift::CompactProtocolReaderWithRefill reader(std::move(refiller));
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
    return result;
  } catch (apache::thrift::protocol::TProtocolException& e) {
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
