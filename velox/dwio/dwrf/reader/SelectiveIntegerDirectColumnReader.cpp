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

#include "velox/dwio/dwrf/reader/SelectiveIntegerDirectColumnReader.h"

namespace facebook::velox::dwrf {

void SelectiveIntegerDirectColumnReader::getValues(
    const RowSet& rows,
    VectorPtr* result) {
  if (!requestedType_->isVarchar()) {
    SelectiveIntegerColumnReader::getValues(rows, result);
    return;
  }

  VectorPtr integers;
  getIntValues(rows, fileType_->type(), &integers);
  switch (fileType_->type()->kind()) {
    case TypeKind::SMALLINT:
      *result = convertIntegerToVarchar<int16_t>(integers, pool_);
      return;
    case TypeKind::INTEGER:
      *result = convertIntegerToVarchar<int32_t>(integers, pool_);
      return;
    case TypeKind::BIGINT:
      *result = convertIntegerToVarchar<int64_t>(integers, pool_);
      return;
    default:
      VELOX_UNREACHABLE(
          "Unexpected integer type: {}", fileType_->type()->toString());
  }
}

uint64_t SelectiveIntegerDirectColumnReader::skip(uint64_t numValues) {
  numValues = SelectiveColumnReader::skip(numValues);
  intDecoder_->skip(numValues);
  return numValues;
}

void SelectiveIntegerDirectColumnReader::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  VELOX_WIDTH_DISPATCH(
      dwio::common::sizeOfIntKind(fileType_->type()->kind()),
      prepareRead,
      offset,
      rows,
      incomingNulls);
  readCommon<SelectiveIntegerDirectColumnReader, true>(rows);
  readOffset_ += rows.back() + 1;
}

} // namespace facebook::velox::dwrf
