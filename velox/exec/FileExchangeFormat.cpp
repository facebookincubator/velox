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

#include "velox/exec/FileExchangeFormat.h"

#include <limits>

#include <folly/dynamic.h>
#include <folly/json.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec::file_exchange {
namespace {

constexpr std::string_view kExchangeOutputFilePrefix{"file-exchange:"};

} // namespace

std::string ExchangeOutputFile::serialize() const {
  VELOX_CHECK(!path.empty(), "Exchange output file path cannot be empty");
  VELOX_CHECK_LE(size, std::numeric_limits<int64_t>::max());
  return std::string{kExchangeOutputFilePrefix} +
      folly::toJson(
             folly::dynamic::object("path", path)(
                 "size", static_cast<int64_t>(size))(
                 "checksum", static_cast<int64_t>(checksum)));
}

bool ExchangeOutputFile::serialized(std::string_view value) {
  return value.starts_with(kExchangeOutputFilePrefix);
}

ExchangeOutputFile ExchangeOutputFile::deserialize(std::string_view value) {
  VELOX_CHECK(serialized(value), "Invalid exchange output file: {}", value);

  folly::dynamic metadata;
  try {
    metadata = folly::parseJson(
        std::string{value.substr(kExchangeOutputFilePrefix.size())});
  } catch (const std::exception& error) {
    VELOX_FAIL("Invalid exchange output file: {}: {}", value, error.what());
  }

  VELOX_CHECK(
      metadata.isObject() && metadata.count("path") == 1 &&
          metadata.count("size") == 1 && metadata.count("checksum") == 1,
      "Incomplete exchange output file: {}",
      value);
  VELOX_CHECK(
      metadata["path"].isString() && !metadata["path"].asString().empty() &&
          metadata["size"].isInt() && metadata["size"].asInt() >= 0 &&
          metadata["checksum"].isInt() && metadata["checksum"].asInt() >= 0 &&
          metadata["checksum"].asInt() <= std::numeric_limits<uint32_t>::max(),
      "Invalid exchange output file metadata: {}",
      value);

  return {
      .path = metadata["path"].asString(),
      .size = static_cast<uint64_t>(metadata["size"].asInt()),
      .checksum = static_cast<uint32_t>(metadata["checksum"].asInt()),
  };
}

} // namespace facebook::velox::exec::file_exchange
