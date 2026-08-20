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

#include "velox/dwio/nimble/encodings/SharedDictionaryTypes.h"

#include <algorithm>

#include "velox/common/EnumDefine.h"
#include "velox/dwio/nimble/common/Varint.h"

namespace facebook::nimble {

namespace {
const auto& sharedDictionaryScopeNames() {
  static const folly::F14FastMap<SharedDictionaryScope, std::string_view>
      kNames = {
          {SharedDictionaryScope::Stripe, "Stripe"},
          {SharedDictionaryScope::File, "File"},
          {SharedDictionaryScope::External, "External"},
      };
  return kNames;
}

size_t checkedSharedDictionaryOffset(std::string_view data, const char* pos) {
  NIMBLE_CHECK(
      pos >= data.data() && pos <= data.data() + data.size(),
      "Shared dictionary cursor is outside the encoding.");
  return static_cast<size_t>(pos - data.data());
}
} // namespace

VELOX_DEFINE_ENUM_NAME(SharedDictionaryScope, sharedDictionaryScopeNames)

SharedDictionaryScope readSharedDictionaryScope(
    std::string_view data,
    const char*& pos) {
  const auto offset = checkedSharedDictionaryOffset(data, pos);
  NIMBLE_CHECK_LT(
      offset, data.size(), "Shared dictionary encoding is missing its scope.");
  return toSharedDictionaryScope(static_cast<uint8_t>(*pos++));
}

uint32_t readSharedDictionaryId(std::string_view data, const char*& pos) {
  const auto offset = checkedSharedDictionaryOffset(data, pos);
  NIMBLE_CHECK_LT(
      offset, data.size(), "Truncated shared dictionary ID varint.");

  const auto remaining = data.size() - offset;
  const auto bytesToCheck = std::min<size_t>(
      remaining, varint::maxVarintSizeForBitWidth(/*bitWidth=*/32));
  for (size_t i = 0; i < bytesToCheck; ++i) {
    if ((static_cast<uint8_t>(pos[i]) & 0x80) == 0) {
      return varint::readVarint32(&pos);
    }
  }

  NIMBLE_CHECK_GE(
      remaining,
      varint::maxVarintSizeForBitWidth(/*bitWidth=*/32),
      "Truncated shared dictionary ID varint.");
  NIMBLE_UNSUPPORTED("Shared dictionary ID varint is too long.");
}

} // namespace facebook::nimble
