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

#include <memory>
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"

namespace facebook::velox::dwio::common {

class AbstractByteRleDecoder {
  // TODO: inputStream or AbstractInputStream can be moved here later
  //  std::unique_ptr<SeekableInputStream> inputStream;
 public:
  AbstractByteRleDecoder()
      : remainingValues{0},
        value{0},
        bufferStart{nullptr},
        bufferEnd{nullptr},
        repeating{false} {}
  size_t remainingValues;
  char value;
  const char* bufferStart;
  const char* bufferEnd;
  bool repeating;

  /**
   * Read a number of values into the batch.
   * @param data the array to read into
   * @param numValues the number of values to read
   * @param nulls If the pointer is null, all values are read. If the
   *    pointer is not null, positions that are true are skipped.
   */
  virtual void next(char* data, uint64_t numValues, const uint64_t* nulls) = 0;
};

} // namespace facebook::velox::dwio::common
