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

#include "velox/dwio/dwrf/reader/SelectiveRepeatedColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveDwrfReader.h"

namespace facebook::velox::dwrf {
namespace {
std::unique_ptr<dwio::common::IntDecoder</*isSigned*/ false>> makeLengthDecoder(
    const dwio::common::TypeWithId& nodeType,
    DwrfParams& params,
    memory::MemoryPool& pool) {
  EncodingKey encodingKey{nodeType.id, params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  auto rleVersion = convertRleVersion(stripe.getEncoding(encodingKey).kind());
  auto lenId = encodingKey.forKind(proto::Stream_Kind_LENGTH);
  bool lenVints = stripe.getUseVInts(lenId);
  return createRleDecoder</*isSigned*/ false>(
      stripe.getStream(lenId, true),
      rleVersion,
      pool,
      lenVints,
      dwio::common::INT_BYTE_SIZE);
}
} // namespace

SelectiveListColumnReader::SelectiveListColumnReader(
    const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
    DwrfParams& params,
    common::ScanSpec& scanSpec)
    : dwio::common::SelectiveListColumnReader(
          requestedType,
          dataType,
          params,
          scanSpec),
      length_(makeLengthDecoder(*nodeType_, params, memoryPool_)) {
  DWIO_ENSURE_EQ(nodeType_->id, dataType->id, "working on the same node");
  EncodingKey encodingKey{nodeType_->id, params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  // count the number of selected sub-columns
  const auto& cs = stripe.getColumnSelector();
  auto& childType = requestedType_->childAt(0);
  VELOX_CHECK(
      cs.shouldReadNode(childType->id),
      "SelectiveListColumnReader must select the values stream");
  if (scanSpec_->children().empty()) {
    scanSpec.getOrCreateChild(
        common::Subfield(common::ScanSpec::kArrayElementsFieldName));
  }
  scanSpec_->children()[0]->setProjectOut(true);
  scanSpec_->children()[0]->setExtractValues(true);

  auto childParams =
      DwrfParams(stripe, FlatMapContext{encodingKey.sequence, nullptr});
  child_ = SelectiveDwrfReader::build(
      childType, nodeType_->childAt(0), childParams, *scanSpec_->children()[0]);
  children_ = {child_.get()};
}

SelectiveMapColumnReader::SelectiveMapColumnReader(
    const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
    DwrfParams& params,
    common::ScanSpec& scanSpec)
    : dwio::common::SelectiveMapColumnReader(
          requestedType,
          dataType,
          params,
          scanSpec),
      length_(makeLengthDecoder(*nodeType_, params, memoryPool_)) {
  DWIO_ENSURE_EQ(nodeType_->id, dataType->id, "working on the same node");
  EncodingKey encodingKey{nodeType_->id, params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  if (scanSpec_->children().empty()) {
    scanSpec_->getOrCreateChild(
        common::Subfield(common::ScanSpec::kMapKeysFieldName));
    scanSpec_->getOrCreateChild(
        common::Subfield(common::ScanSpec::kMapValuesFieldName));
  }
  scanSpec_->children()[0]->setProjectOut(true);
  scanSpec_->children()[0]->setExtractValues(true);
  scanSpec_->children()[1]->setProjectOut(true);
  scanSpec_->children()[1]->setExtractValues(true);

  const auto& cs = stripe.getColumnSelector();
  auto& keyType = requestedType_->childAt(0);
  VELOX_CHECK(
      cs.shouldReadNode(keyType->id),
      "Map key must be selected in SelectiveMapColumnReader");
  auto keyParams =
      DwrfParams(stripe, FlatMapContext{encodingKey.sequence, nullptr});
  keyReader_ = SelectiveDwrfReader::build(
      keyType,
      nodeType_->childAt(0),
      keyParams,
      *scanSpec_->children()[0].get());

  auto& valueType = requestedType_->childAt(1);
  VELOX_CHECK(
      cs.shouldReadNode(valueType->id),
      "Map Values must be selected in SelectiveMapColumnReader");
  auto elementParams =
      DwrfParams(stripe, FlatMapContext{encodingKey.sequence, nullptr});
  elementReader_ = SelectiveDwrfReader::build(
      valueType,
      nodeType_->childAt(1),
      elementParams,
      *scanSpec_->children()[1]);
  children_ = {keyReader_.get(), elementReader_.get()};

  VLOG(1) << "[Map] Initialized map column reader for node " << nodeType_->id;
}

} // namespace facebook::velox::dwrf
