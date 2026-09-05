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

#include "velox/dwio/dwrf/reader/SelectiveByteRleColumnReader.h"

namespace facebook::velox::dwrf {

SelectiveByteRleColumnReader::SelectiveByteRleColumnReader(
    const TypePtr& requestedType,
    std::shared_ptr<const dwio::common::TypeWithId> fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec,
    bool isBool)
    : dwio::common::SelectiveByteRleColumnReader(
          requestedType,
          std::move(fileType),
          params,
          scanSpec) {
  const EncodingKey encodingKey{
      fileType_->id(), params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  if (isBool) {
    boolRle_ = createBooleanRleDecoder(
        stripe.getStream(
            StripeStreamsUtil::getStreamForKind(
                stripe,
                encodingKey,
                proto::Stream_Kind_DATA,
                proto::orc::Stream_Kind_DATA),
            params.streamLabels().label(),
            true),
        encodingKey);
  } else {
    byteRle_ = createByteRleDecoder(
        stripe.getStream(
            StripeStreamsUtil::getStreamForKind(
                stripe,
                encodingKey,
                proto::Stream_Kind_DATA,
                proto::orc::Stream_Kind_DATA),
            params.streamLabels().label(),
            true),
        encodingKey);
  }
}

void SelectiveByteRleColumnReader::seekToRowGroup(int64_t index) {
  dwio::common::SelectiveByteRleColumnReader::seekToRowGroup(index);
  auto positionsProvider = formatData_->seekToRowGroup(index);
  if (boolRle_) {
    boolRle_->seekToRowGroup(positionsProvider);
  } else {
    byteRle_->seekToRowGroup(positionsProvider);
  }

  VELOX_CHECK(!positionsProvider.hasNext());
}

uint64_t SelectiveByteRleColumnReader::skip(uint64_t numValues) {
  numValues = formatData_->skipNulls(numValues);
  if (byteRle_) {
    byteRle_->skip(numValues);
  } else {
    boolRle_->skip(numValues);
  }
  return numValues;
}

void SelectiveByteRleColumnReader::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  readCommon<SelectiveByteRleColumnReader, true>(offset, rows, incomingNulls);
  readOffset_ += rows.back() + 1;
}

void SelectiveByteRleColumnReader::getValues(
    const RowSet& rows,
    VectorPtr* result) {
  if (requestedType_->kind() != TypeKind::VARCHAR) {
    dwio::common::SelectiveByteRleColumnReader::getValues(rows, result);
    return;
  }

  VectorPtr integers;
  getFlatValues<int8_t, int8_t>(rows, &integers, TINYINT());
  *result = convertIntegerToVarchar<int8_t>(integers, pool_);
}

} // namespace facebook::velox::dwrf
