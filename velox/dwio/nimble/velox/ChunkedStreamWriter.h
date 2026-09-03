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

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble {

class ChunkedStreamWriter {
 public:
  explicit ChunkedStreamWriter(
      Buffer& buffer,
      CompressionParams compressionParams = {
          .type = CompressionType::Uncompressed});

  std::vector<std::string_view> encode(std::string_view chunk);

 private:
  Buffer& buffer_;
  CompressionParams compressionParams_;
};

} // namespace facebook::nimble
