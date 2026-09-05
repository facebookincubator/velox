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

#include <cstdint>
#include <string>
#include <string_view>

#include <folly/lang/Bits.h>

namespace facebook::velox::exec::file_exchange {

/// Identifies one committed partition file in backend-private exchange
/// metadata.
struct ExchangeOutputFile {
  std::string path;
  uint64_t size;
  uint32_t checksum;

  std::string serialize() const;

  static bool serialized(std::string_view value);

  static ExchangeOutputFile deserialize(std::string_view value);
};

// Each page is a big-endian payload size followed by that many payload bytes.
using PageSize = uint64_t;

inline PageSize encodePageSize(uint64_t size) {
  return folly::Endian::big(size);
}

inline uint64_t decodePageSize(PageSize size) {
  return folly::Endian::big(size);
}

} // namespace facebook::velox::exec::file_exchange
