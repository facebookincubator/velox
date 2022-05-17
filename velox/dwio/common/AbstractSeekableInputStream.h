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

// This class is an interface for SeekableInputStream. It will be removed
// after SeekableInputStream Class being migrated to velox/dwio/common

#pragma once

#include "velox/dwio/dwrf/common/wrap/zero-copy-stream-wrapper.h"

namespace facebook::velox::dwio::common {

class AbstractSeekableInputStream
    : public google::protobuf::io::ZeroCopyInputStream{
 public:
  virtual bool Next(const void** data, int* size) = 0;
  virtual bool Skip(int count) = 0;
};

} // namespace facebook::velox::dwio::common