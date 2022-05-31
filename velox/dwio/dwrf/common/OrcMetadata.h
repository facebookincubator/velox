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

#include "velox/dwio/dwrf/common/Common.h"

namespace facebook::velox::dwrf {

static CompressionKind convertCompressionKind(proto::orc::CompressionKind
    compression) {
  auto compressionUint = static_cast<uint32_t>(compression);
  if (compressionUint >= 4 && compressionUint <= 5) {
    compressionUint = 9 - compressionUint;
  }
  return static_cast<CompressionKind>(compressionUint);
}

class ORCPostScript : public PostScript {
  public:
    ORCPostScript(
      uint64_t footerLength,
      proto::orc::CompressionKind compression,
      uint64_t compressionBlockSize,
      uint64_t metadataLength,
      uint32_t writerVersion,
      uint64_t stripeStatisticsLength)
    : PostScript(footerLength, convertCompressionKind(compression),
        compressionBlockSize, writerVersion)
    , metadataLength_{metadataLength}
    , stripeStatisticsLength_{stripeStatisticsLength} {
    }

  private:
    uint64_t metadataLength_;
    uint64_t stripeStatisticsLength_;
};


} // namespace facebook::velox::dwrf