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
#pragma once

#include <cstdint>
#include <functional>
#include "velox/buffer/Buffer.h"
#include "velox/common/base/BitUtil.h"

namespace facebook::nimble {

class Encoding;

class Decoder {
 public:
  virtual ~Decoder() = default;

  /// Decode next 'count' rows into 'output' and return the number of non-null
  /// rows materialized.
  ///
  /// For string types, stringBuffers will be populated with buffers holding
  /// the string data. For non-string types, stringBuffers will remain empty.
  ///
  /// If the stream is nullable, getOutputNulls must return the mutable null
  /// bitmap for the output vector. The callback is lazy so callers only
  /// allocate nulls when the encoded stream actually contains nulls.
  ///
  /// If scatterOutputBitmap is provided, decoded rows are written into the
  /// selected output positions and the null bitmap follows the same scattered
  /// layout. A null scatterOutputBitmap means dense output.
  virtual uint32_t next(
      uint32_t count,
      void* output,
      std::vector<velox::BufferPtr>& stringBuffers,
      std::function<void*()> getOutputNulls = nullptr,
      const velox::bits::Bitmap* scatterOutputBitmap = nullptr) = 0;

  virtual void skip(uint32_t count) = 0;

  virtual void reset() = 0;

  virtual const Encoding* encoding() const = 0;
};

} // namespace facebook::nimble
