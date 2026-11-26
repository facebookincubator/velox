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
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"

#include <algorithm>

#include "velox/common/base/Nulls.h"

namespace facebook::nimble {

Encoding::Encoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const Options& options)
    : pool_{&pool},
      encodingType_{EncodingPrefix::encodingType(data)},
      options_{options},
      dataType_{EncodingPrefix::readDataType(data)},
      data_{data},
      rowCount_{EncodingPrefix::readRowCount(data, options_.useVarintRowCount)},
      prefixSize_{
          EncodingPrefix::prefixSize(data, options_.useVarintRowCount)} {}

bool Encoding::rebind(
    std::string_view data,
    const std::function<void*(uint32_t)>& stringBufferFactory) {
  auto body = data;
  const auto prefix = EncodingPrefix::consume(body, options_.useVarintRowCount);
  if (EncodingPrefix::encodingType(prefix) != encodingType_ ||
      EncodingPrefix::dataType(prefix) != dataType_) {
    return false;
  }
  const auto previousData = data_;
  const auto previousRowCount = rowCount_;
  const auto previousPrefixSize = prefixSize_;
  data_ = data;
  rowCount_ = EncodingPrefix::readRowCount(prefix, options_.useVarintRowCount);
  prefixSize_ = static_cast<uint32_t>(prefix.size());
  if (rebindImpl(data, stringBufferFactory)) {
    return true;
  }
  // rebindImpl() returns false only for an encoding that does not support
  // rebinding, before it reads `data`, so restoring the prefix fields leaves
  // the encoding bound to its segment.
  data_ = previousData;
  rowCount_ = previousRowCount;
  prefixSize_ = previousPrefixSize;
  return false;
}

std::unique_ptr<Encoding> Encoding::rebindChild(
    std::string_view data,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    std::unique_ptr<Encoding> child) {
  if (child != nullptr && child->rebind(data, stringBufferFactory)) {
    return child;
  }
  return EncodingFactory().create(*pool_, data, stringBufferFactory, options_);
}

/* static */ void Encoding::copyIOBuf(char* pos, const folly::IOBuf& buf) {
  NIMBLE_CHECK_NOT_NULL(pos);
  [[maybe_unused]] size_t length = buf.computeChainDataLength();
  for (auto data : buf) {
    std::copy(data.cbegin(), data.cend(), pos);
    pos += data.size();
    length -= data.size();
  }
  NIMBLE_DCHECK_EQ(length, 0, "IOBuf chain length corruption");
}

void Encoding::readNulls(uint32_t rowCount, uint64_t* nulls, int begin) {
  NIMBLE_CHECK(
      !isNullable(),
      "Nullable encoding must override readNulls: {}",
      toString(encodingType()));
  velox::bits::fillBits(
      nulls, begin, begin + static_cast<int>(rowCount), velox::bits::kNotNull);
  skip(rowCount);
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
