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

#include "velox/common/base/ByteInputStream.h"

namespace facebook::velox {

std::vector<ByteRange> byteRangesFromIOBuf(folly::IOBuf* iobuf) {
  if (iobuf == nullptr) {
    return {};
  }
  std::vector<ByteRange> byteRanges;
  auto* current = iobuf;
  do {
    byteRanges.push_back(
        {current->writableData(), (int32_t)current->length(), 0});
    current = current->next();
  } while (current != iobuf);
  return byteRanges;
}

uint32_t ByteRange::availableBytes() const {
  return std::max(0, size - position);
}

std::string ByteRange::toString() const {
  return fmt::format("[{} starting at {}]", succinctBytes(size), position);
}

} // namespace facebook::velox
