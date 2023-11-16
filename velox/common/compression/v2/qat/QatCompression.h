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

#include <qatzip.h>
#include <vector>
#include "velox/common/compression/v2/Compression.h"

namespace facebook::velox::common::qat {

class QatGzipCodecOptions : public CodecOptions {
 public:
  QatGzipCodecOptions(
      int32_t compressionLevel = kUseDefaultCompressionLevel,
      QzPollingMode_T pollingMode = QZ_BUSY_POLLING)
      : CodecOptions(compressionLevel), pollingMode(pollingMode) {}

  QzPollingMode_T pollingMode;
};

std::unique_ptr<Codec> makeQatGzipCodec(
    int32_t compressionLevel = QZ_COMP_LEVEL_DEFAULT,
    QzPollingMode_T pollingMode = QZ_BUSY_POLLING);

} // namespace facebook::velox::common::qat