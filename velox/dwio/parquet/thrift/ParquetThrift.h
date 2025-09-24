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
#include "velox/dwio/parquet/thrift/gen-cpp2/parquet_types.h"
#include "velox/dwio/parquet/thrift/gen-cpp2/parquet_types_custom_protocol.h"

#include <fmt/format.h>
#include <thrift/lib/cpp2/protocol/CompactProtocol.h>
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

template <typename ThriftStruct>
uint32_t serialize(
    const ThriftStruct& thriftStruct,
    folly::IOBufQueue* buffer) {
  apache::thrift::CompactProtocolWriter writer;
  writer.setOutput(buffer);
  return thriftStruct.write(&writer);
}
}; // namespace facebook::velox::parquet::thrift
