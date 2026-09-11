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

#include "velox/serializers/PrestoBatchVectorSerializer.h"

#include "velox/serializers/PrestoSerializerSerializationUtils.h"
#include "velox/serializers/VectorStream.h"

namespace facebook::velox::serializer::presto::detail {
void PrestoBatchVectorSerializer::serialize(
    const RowVectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    Scratch& scratch,
    OutputStream* stream) {
  VELOX_CHECK_NOT_NULL(vector, "Vector to serialize is null.");
  VELOX_CHECK_NOT_NULL(stream, "Stream to serialize out to is null.");

#ifndef NDEBUG
  for (int i = 0; i < ranges.size(); i++) {
    VELOX_CHECK_GE(ranges[i].begin, 0, "Invalid range at index {}", i);
    VELOX_CHECK_LE(
        ranges[i].begin + ranges[i].size,
        vector->size(),
        "Invalid range at index {}",
        i);
  }
#endif

  SCOPE_EXIT {
    inUse.store(false);
  };

  VELOX_CHECK(
      !inUse.exchange(true),
      "PrestoBatchVectorSerializer::serialize being called concurrently on the same object.");

  common::testutil::TestValue::adjust(
      "facebook::velox::serializers::PrestoBatchVectorSerializer::serialize",
      this);

  const auto numRows = rangesTotalSize(ranges);
  const auto rowType = vector->type();
  const auto numChildren = vector->childrenSize();

  StreamArena arena(pool_);
  std::vector<VectorStream> streams;
  streams.reserve(numChildren);
  for (int i = 0; i < numChildren; i++) {
    streams.emplace_back(
        rowType->childAt(i),
        std::nullopt,
        vector->childAt(i),
        &arena,
        numRows,
        opts_);

    if (numRows > 0) {
      serializeColumn(vector->childAt(i), ranges, &streams[i], scratch);
    }
  }

  flushStreams(
      streams, numRows, arena, *codec_, opts_.minCompressionRatio, stream);
}
} // namespace facebook::velox::serializer::presto::detail
