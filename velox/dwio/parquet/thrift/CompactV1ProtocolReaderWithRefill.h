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

#include <thrift/lib/cpp2/protocol/CompactV1Protocol.h>
#include <thrift/lib/cpp2/protocol/ProtocolReaderWithRefill.h>

namespace apache::thrift {

class CompactV1ProtocolReaderWithRefill
    : public CompactProtocolReaderWithRefill {
 public:
  using ProtocolWriter = CompactV1ProtocolWriter;
  using CompactProtocolReaderWithRefill::CompactProtocolReaderWithRefill;

  inline void readDouble(double& dub) override {
    static_assert(sizeof(double) == sizeof(uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    ensureBuffer(sizeof(double));
    uint64_t bits = readLEFromBuffer<int64_t>();
    dub = folly::bit_cast<double>(bits);
  }
};

template <>
inline bool canReadNElements(
    CompactV1ProtocolReaderWithRefill& /* prot */,
    uint32_t /* n */,
    std::initializer_list<TType> /* types */) {
  return true;
}

} // namespace apache::thrift
