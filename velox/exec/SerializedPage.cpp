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
#include "velox/exec/SerializedPage.h"

#include <utility>

namespace facebook::velox::exec {

PrestoSerializedPage::PrestoSerializedPage(
    std::unique_ptr<folly::IOBuf> iobuf,
    std::function<void(folly::IOBuf&)> onDestructionCb,
    std::optional<int64_t> numRows)
    : iobuf_(std::move(iobuf)),
      iobufBytes_(chainBytes(*iobuf_.get())),
      numRows_(numRows),
      onDestructionCb_(std::move(onDestructionCb)) {
  VELOX_CHECK_NOT_NULL(iobuf_);
  for (auto& buf : *iobuf_) {
    int32_t bufSize = buf.size();
    ranges_.push_back(
        ByteRange{
            const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(buf.data())),
            bufSize,
            0});
  }
}

PrestoSerializedPage::~PrestoSerializedPage() {
  if (onDestructionCb_) {
    onDestructionCb_(*iobuf_.get());
  }
}

std::unique_ptr<ByteInputStream>
PrestoSerializedPage::prepareStreamForDeserialize() {
  return std::make_unique<BufferInputStream>(std::move(ranges_));
}

} // namespace facebook::velox::exec
