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
#include "velox/dwio/nimble/encodings/common/Encoding.h"

#include <algorithm>

namespace facebook::nimble {

Encoding::Encoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const Options& options)
    : pool_{&pool},
      data_{data},
      encodingType_{data_[EncodingPrefix::kEncodingTypeOffset]},
      options_{options},
      dataType_{EncodingPrefix::readDataType(data)},
      rowCount_{EncodingPrefix::readRowCount(data, options_.useVarintRowCount)},
      prefixSize_{
          EncodingPrefix::prefixSize(data, options_.useVarintRowCount)} {}

/* static */ void Encoding::copyIOBuf(char* pos, const folly::IOBuf& buf) {
  [[maybe_unused]] size_t length = buf.computeChainDataLength();
  for (auto data : buf) {
    std::copy(data.cbegin(), data.cend(), pos);
    pos += data.size();
    length -= data.size();
  }
  NIMBLE_DCHECK_EQ(length, 0, "IOBuf chain length corruption");
}

std::string Encoding::debugString(int offset) const {
  return fmt::format(
      "{}{}<{}> rowCount={}",
      std::string(offset, ' '),
      toString(encodingType()),
      toString(dataType()),
      rowCount());
}

} // namespace facebook::nimble
