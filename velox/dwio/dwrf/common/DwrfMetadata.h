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

class DWRFPostScript : public PostScript {
 public:
  DWRFPostScript() : PostScript() {}
  DWRFPostScript(
      uint64_t footerLength,
      proto::CompressionKind compression,
      uint64_t compressionBlockSize,
      uint32_t writerVersion,
      proto::StripeCacheMode cacheMode,
      uint32_t cacheSize)
      : PostScript(
            footerLength,
            static_cast<CompressionKind>(compression),
            compressionBlockSize,
            writerVersion),
        cacheMode_{static_cast<StripeCacheMode>(cacheMode)},
        cacheSize_{cacheSize} {}

  StripeCacheMode cachemode() const {
    return cacheMode_;
  }

  bool has_cachemode() const {
    return cacheMode_ != StripeCacheMode::NA;
  }

  uint32_t cachesize() const {
    return cacheSize_;
  }

  bool has_cachesize() const {
    return cacheSize_ != 0;
  }

 private:
  StripeCacheMode cacheMode_;
  uint32_t cacheSize_;
};

} // namespace facebook::velox::dwrf