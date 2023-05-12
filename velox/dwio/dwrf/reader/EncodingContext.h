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

#include "velox/dwio/dwrf/common/ByteRLE.h"

namespace facebook::velox::dwrf {
struct FlatMapContext {
 public:
  static FlatMapContext nonFlatMapContext() {
    return FlatMapContext{0, nullptr, nullptr};
  }

  uint32_t sequence;
  // Kept alive by key nodes
  BooleanRleDecoder* inMapDecoder{nullptr};

  std::function<void(uint64_t totalKeys, uint64_t selectedKeys)>
      keySelectionCallback;
};

} // namespace facebook::velox::dwrf
