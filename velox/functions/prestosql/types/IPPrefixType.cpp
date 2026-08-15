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

#include "velox/functions/prestosql/types/IPPrefixType.h"

#include <fmt/format.h>

namespace facebook::velox {

std::string_view IPPrefixType::valueToString(
    int128_t ip,
    int8_t prefixLength,
    char* buffer) const {
  auto ipStr = IPADDRESS()->valueToString(ip);
  auto result = fmt::format_to_n(
      buffer,
      kMaxStringSize,
      "{}/{}",
      ipStr,
      static_cast<uint8_t>(prefixLength));
  return {buffer, result.size};
}

} // namespace facebook::velox
