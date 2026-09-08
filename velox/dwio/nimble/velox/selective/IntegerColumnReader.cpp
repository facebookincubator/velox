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

#include "velox/dwio/nimble/velox/selective/IntegerColumnReader.h"

#include "velox/dwio/nimble/encodings/legacy/EncodingUtils.h"

namespace facebook::nimble {

using namespace facebook::velox;

uint64_t IntegerColumnReader::skip(uint64_t numValues) {
  numValues = SelectiveIntegerColumnReader::skip(numValues);
  decoder_.skip(numValues);
  return numValues;
}

void IntegerColumnReader::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  VELOX_WIDTH_DISPATCH(
      dwio::common::sizeOfIntKind(fileType_->type()->kind()),
      prepareRead,
      offset,
      rows,
      incomingNulls);
  readCommon<IntegerColumnReader, false>(rows);
  readOffset_ += rows.back() + 1;
}

bool IntegerColumnReader::estimateMaterializedSize(
    size_t& byteSize,
    size_t& rowCount) const {
  auto decoderRowCount = decoder_.estimateRowCount();
  if (!decoderRowCount.has_value()) {
    return false;
  }
  rowCount = *decoderRowCount;
  byteSize = (1 << ((int)requestedType_->kind() - 1)) * rowCount;
  return true;
}

} // namespace facebook::nimble
