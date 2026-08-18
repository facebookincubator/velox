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

#include "velox/dwio/nimble/encodings/ConstantEncoding.h"

namespace facebook::nimble {

ConstantEncoding<std::string_view>::ConstantEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : ConstantEncodingBase<std::string_view>(pool, data, options) {
  const char* pos = data.data() + this->dataOffset();
  value_ = encoding::read<physicalType>(pos);
  NIMBLE_CHECK(pos == data.end(), "Unexpected constant encoding end");
  auto stringBuffer = static_cast<char*>(stringBufferFactory(value_.size()));
  std::memcpy(stringBuffer, value_.begin(), value_.size());
  value_ = std::string_view{stringBuffer, value_.size()};
}

} // namespace facebook::nimble
