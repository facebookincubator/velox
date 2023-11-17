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
#include <memory>

#include <qpl/qpl.h>
#include "velox/common/compression/v2/Compression.h"
#include "velox/common/compression/v2/iaa/QplJobPool.h"

namespace facebook::velox::common::iaa {

class IaaGzipCodecOptions : public CodecOptions {
 public:
  explicit IaaGzipCodecOptions(
      int32_t compressionLevel,
      uint32_t maxJobNumber = kMaxQplJobNumber)
      : CodecOptions(compressionLevel), maxJobNumber(maxJobNumber) {}

  uint32_t maxJobNumber;
};

std::unique_ptr<Codec> makeIaaGzipCodec(
    int32_t compressionLevel = qpl_default_level,
    uint32_t maxJobNumber = kMaxQplJobNumber);

std::unique_ptr<AsyncCodec> makeIaaGzipAsyncCodec(
    int32_t compressionLevel = qpl_default_level,
    uint32_t maxJobNumber = kMaxQplJobNumber);
} // namespace facebook::velox::common::iaa
