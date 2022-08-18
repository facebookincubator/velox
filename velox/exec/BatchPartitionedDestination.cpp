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
#include <folly/io/IOBuf.h>
#include <velox/exec/BatchPartitionedDestination.h>

namespace facebook::velox::exec {

BlockingReason BatchDestination::advance(
    uint64_t maxBytes,
    const std::vector<vector_size_t>& sizes,
    const RowVectorPtr& output,
    PartitionedOutputBufferManager& bufferManager,
    bool* atEnd,
    ContinueFuture* future) {
  if (row_ >= rows_.size()) {
    *atEnd = true;
    return BlockingReason::kNotBlocked;
  }

  // In the batch mode, the destination writes each row directly
  // to the shuffle write client
  VELOX_CHECK(shuffleWriter_s != nullptr);
  VELOX_CHECK(serializer_s != nullptr);

  serializer_s->initialize(output);
  for (; row_ < rows_.size(); ++row_) {
    std::string_view serializedRow;
    std::string_view serializedKeys;

    // Serializing one row at a time and converting to IOBuf and sending to the
    // shuffle service
    VELOX_CHECK(
        serializer_s->serializeRow(
            output, row_, serializedKeys, serializedRow) ==
        velox::batch::BatchSerdeStatus::Success);
    auto rowBuffer =
        folly::IOBuf::wrapBuffer(serializedRow.data(), serializedRow.size());
    auto keysBuffer =
        folly::IOBuf::wrapBuffer(serializedKeys.data(), serializedKeys.size());
    shuffleWriter_s->collect(*rowBuffer.get(), *keysBuffer.get(), destination_);
  }
  *atEnd = true;
  return BlockingReason::kNotBlocked;
}
} // namespace facebook::velox::exec