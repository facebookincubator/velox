/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/index/StripeStreams.h"

#include <limits>

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

RowRange StripeSelection::selectedRowRange() const {
  NIMBLE_CHECK_LE(
      rowRange.startRow, rowRange.endRow, "Selected row range is reversed");
  NIMBLE_CHECK_LE(
      rowRange.endRow, numRows, "Selected row range exceeds stripe row count");
  NIMBLE_CHECK_LE(
      static_cast<uint64_t>(startRow) + rowRange.endRow,
      std::numeric_limits<uint32_t>::max(),
      "Selected row range exceeds file row limit");
  return {startRow + rowRange.startRow, startRow + rowRange.endRow};
}

bool StripeSelection::contains(uint32_t row) const {
  return selectedRowRange().contains(row);
}

namespace {

// Keeps the data handle of loaded streams, which owns their bytes, while a
// buffer over the bytes lives. The buffer keeps a copy of the releaser, whose
// handle does the counting, so addRef() and release() do nothing.
struct DataHandleReleaser {
  DataInput::Handle dataHandle;

  void addRef() const {}

  void release() const {}
};

} // namespace

std::vector<velox::BufferPtr> makeStreamBufferHolders(
    std::span<const std::string_view> streams,
    const std::vector<bool>& stringStreams,
    const DataInput::Handle& dataHandle) {
  NIMBLE_CHECK_EQ(
      stringStreams.size(),
      streams.size(),
      "String stream flags must align with the streams");
  std::vector<velox::BufferPtr> holders;
  for (size_t i{0}; i < streams.size(); ++i) {
    const auto stream = streams[i];
    if (!stringStreams[i] || stream.empty()) {
      continue;
    }
    NIMBLE_CHECK_NOT_NULL(
        dataHandle, "Loaded streams require the handle that keeps them alive");
    if (holders.empty()) {
      holders.resize(streams.size());
    }
    // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
    holders[i] = velox::BufferView<DataHandleReleaser>::create(
        reinterpret_cast<const uint8_t*>(stream.data()),
        stream.size(),
        DataHandleReleaser{dataHandle});
  }
  return holders;
}

} // namespace facebook::nimble
