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

#include "velox/functions/prestosql/types/UuidType.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox {

std::string_view UuidType::valueToString(int128_t value, char* buffer) const {
  auto bigEndianValue = DecimalUtil::bigEndian(value);
  const auto* bytes = reinterpret_cast<const uint8_t*>(&bigEndianValue);

  // Do not use boost::lexical_cast. It is very slow.

  // 2 hex digits per byte value. Lookup table avoids per-nibble branching.
  static const char* const kHexTable =
      "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
      "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
      "404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f"
      "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f"
      "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f"
      "a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebf"
      "c0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedf"
      "e0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff";

  size_t offset = 0;
  for (auto i = 0; i < 16; ++i) {
    buffer[offset] = kHexTable[bytes[i] * 2];
    buffer[offset + 1] = kHexTable[bytes[i] * 2 + 1];
    offset += 2;
    if (i == 3 || i == 5 || i == 7 || i == 9) {
      buffer[offset++] = '-';
    }
  }
  return {buffer, kStringSize};
}

} // namespace facebook::velox
