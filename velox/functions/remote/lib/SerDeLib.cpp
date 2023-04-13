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

#include "velox/functions/remote/lib/SerDeLib.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::functions {

folly::IOBuf serializeIOBuf(
    const RowVectorPtr rowVector,
    vector_size_t rangeEnd,
    memory::MemoryPool& pool) {
  auto streamGroup = std::make_unique<VectorStreamGroup>(&pool);
  streamGroup->createStreamTree(asRowType(rowVector->type()), 1000);

  IndexRange range{0, rangeEnd};
  streamGroup->append(rowVector, folly::Range<IndexRange*>(&range, 1));

  IOBufOutputStream stream(pool);
  streamGroup->flush(&stream);
  return std::move(*stream.getIOBuf());
}

RowVectorPtr deserializeIOBuf(
    const folly::IOBuf& ioBuf,
    const RowTypePtr& outputType,
    memory::MemoryPool& pool) {
  std::vector<ByteRange> ranges;
  ranges.reserve(4);

  for (const auto& range : ioBuf) {
    ranges.emplace_back(ByteRange{
        const_cast<uint8_t*>(range.data()), (int32_t)range.size(), 0});
  }

  ByteStream byteStream;
  byteStream.resetInput(std::move(ranges));
  RowVectorPtr outputVector;
  VectorStreamGroup::read(&byteStream, &pool, outputType, &outputVector);
  return outputVector;
}

} // namespace facebook::velox::functions
