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

#include "velox/dwio/nimble/velox/selective/StructColumnReader.h"

#include "velox/dwio/nimble/velox/selective/ChunkedDecoder.h"
#include "velox/dwio/nimble/velox/selective/ColumnReader.h"

namespace facebook::nimble {

using namespace facebook::velox;

void StructColumnReaderBase::seekTo(int64_t offset, bool /*readsNullsOnly*/) {
  if (offset == readOffset_) {
    return;
  }
  VELOX_CHECK_LT(readOffset_, offset);
  if (numParentNulls_ > 0) {
    VELOX_CHECK_LE(
        parentNullsRecordedTo_,
        offset,
        "Must not seek to before parentNullsRecordedTo_");
  }
  const auto distance = offset - readOffset_ - numParentNulls_;
  const auto numNonNulls = formatData_->skipNulls(distance);
  for (auto& child : children_) {
    if (child != nullptr) {
      child->addSkippedParentNulls(
          readOffset_, offset, numParentNulls_ + distance - numNonNulls);
    }
  }
  numParentNulls_ = 0;
  parentNullsRecordedTo_ = 0;
  readOffset_ = offset;
}

StructColumnReader::StructColumnReader(
    const velox::TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    common::ScanSpec& scanSpec,
    bool isRoot)
    : StructColumnReaderBase(
          requestedType,
          fileType,
          params,
          scanSpec,
          isRoot) {
  NIMBLE_CHECK(
      !params.hasLazyIoColumns() || isRoot,
      "Lazy I/O columns should only be set at root level");
  const auto& rowType = params.nimbleType()->asRow();
  const auto& fileRowType = fileType_->type()->asRow();
  if (params.hasLazyIoColumns()) {
    lazyInput_ = params.streams().createLazyInput();
  }
  const auto stableChildren = scanSpec.stableChildren();
  for (const auto& childSpec : *stableChildren) {
    if (childSpec->isConstant() || isChildMissing(*childSpec)) {
      childSpec->setSubscript(kConstantChildSpecSubscript);
      continue;
    }
    if (!childSpec->readFromFile()) {
      continue;
    }
    const auto childIdx = fileRowType.getChildIdx(childSpec->fieldName());
    const auto& childFileType = fileType_->childAt(childIdx);
    const auto& childRequestedType =
        requestedType_->asRow().findChild(childSpec->fieldName());
    auto childParams = params.makeChildParams(rowType.childAt(childIdx));
    if (params.lazyIoColumn(childSpec->fieldName())) {
      childParams.setLazyColumnIo(true);
    }
    addChild(buildColumnReader(
        childRequestedType, childFileType, childParams, *childSpec, false));
    childSpec->setSubscript(children_.size() - 1);
  }
}

bool StructColumnReader::estimateMaterializedSize(
    size_t& byteSize,
    size_t& rowCount) const {
  auto* nulls = formatData().as<NimbleData>().nullsDecoder();
  if (nulls != nullptr) {
    const auto nullsRowCount = nulls->estimateRowCount();
    if (!nullsRowCount.has_value()) {
      return false;
    }
    rowCount = *nullsRowCount;
  } else {
    rowCount = 0;
  }
  // When lazy columns exist, return false to fall through to RowSizeTracker.
  // A partial estimate from eager columns only would underestimate row size
  // (lazy columns are often large FlatMaps), causing oversized batches.
  // RowSizeTracker learns the actual row size after the first batch.
  if (lazyInput_ != nullptr) {
    return false;
  }
  size_t rowSize{0};
  for (const auto& child : children_) {
    size_t childByteSize{0};
    size_t childRowCount{0};
    if (!child->estimateMaterializedSize(childByteSize, childRowCount)) {
      return false;
    }
    if (nulls == nullptr) {
      rowCount = childRowCount;
    }
    if (childRowCount > 0) {
      rowSize += childByteSize / childRowCount;
    }
  }
  byteSize = rowSize * rowCount;
  if (nulls != nullptr) {
    byteSize += rowCount / 8;
  }
  return true;
}

} // namespace facebook::nimble
