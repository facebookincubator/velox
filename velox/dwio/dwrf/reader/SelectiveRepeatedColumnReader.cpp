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
    const dwio::common::TypeWithId& fileType,
    DwrfParams& params,
    memory::MemoryPool& pool) {
  EncodingKey encodingKey{fileType.id(), params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  const auto rleVersion = convertRleVersion(stripe, encodingKey);
  const auto lenId = StripeStreamsUtil::getStreamForKind(
      stripe,
      encodingKey,
      proto::Stream_Kind_LENGTH,
      proto::orc::Stream_Kind_LENGTH);
  const bool lenVints = stripe.getUseVInts(lenId);
  return createRleDecoder</*isSigned=*/false>(
      stripe.getStream(lenId, params.streamLabels().label(), true),
      rleVersion,
      pool,
      lenVints,
      dwio::common::INT_BYTE_SIZE);
}

// Returns true if the MAP extraction type needs the key child reader.
// When deltaUpdate is set, all child readers are needed regardless of
// ExtractionType because delta updates (e.g., MAP_CONCAT) operate on the
// full map.
bool needsKeyReader(
    common::ScanSpec::ExtractionType extractionType,
    bool hasDeltaUpdate) {
  return hasDeltaUpdate ||
      extractionType == common::ScanSpec::ExtractionType::kNone ||
      extractionType == common::ScanSpec::ExtractionType::kKeys;
}

// Returns true if the MAP extraction type needs the value child reader.
// When deltaUpdate is set, all child readers are needed regardless of
// ExtractionType because delta updates (e.g., MAP_CONCAT) operate on the
// full map.
bool needsElementReader(
    common::ScanSpec::ExtractionType extractionType,
    bool hasDeltaUpdate) {
  return hasDeltaUpdate ||
      extractionType == common::ScanSpec::ExtractionType::kNone ||
      extractionType == common::ScanSpec::ExtractionType::kValues;
}
} // namespace

FlatMapContext flatMapContextFromEncodingKey(const EncodingKey& encodingKey) {
  return FlatMapContext{
      .sequence = encodingKey.sequence(),
      .inMapDecoder = nullptr,
      .keySelectionCallback = nullptr};
}

SelectiveListColumnReader::SelectiveListColumnReader(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec)
    : dwio::common::SelectiveListColumnReader(
          requestedType,
          fileType,
          params,
          scanSpec),
      length_(makeLengthDecoder(*fileType_, params, *pool_)) {
  EncodingKey encodingKey{fileType_->id(), params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  // count the number of selected sub-columns
  auto& childType = requestedType_->childAt(0);
  if (scanSpec_->children().empty()) {
    scanSpec.getOrCreateChild(common::ScanSpec::kArrayElementsFieldName);
  }
  scanSpec_->children()[0]->setProjectOut(true);

  // For kSize extraction we only need the length stream, so skip creating
  // the element reader entirely.  This avoids registering its streams and
  // reduces IO.  When deltaUpdate is set, we still need the element reader
  // because delta updates operate on the full array.
  if (scanSpec.extractionType() == common::ScanSpec::ExtractionType::kSize &&
      !scanSpec.deltaUpdate()) {
    return;
  }

  auto childParams = DwrfParams(
      stripe,
      params.streamLabels(),
      params.runtimeStatistics(),
      flatMapContextFromEncodingKey(encodingKey));
  child_ = SelectiveDwrfReader::build(
      columnReaderOptions,
      childType,
      fileType_->childAt(0),
      childParams,
      *scanSpec_->children()[0]);
  children_ = {child_.get()};
}

namespace {

void makeMapChildrenReaders(
    const dwio::common::TypeWithId& fileType,
    const Type& requestedType,
    DwrfParams& params,
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const common::ScanSpec& scanSpec,
    common::ScanSpec::ExtractionType extractionType,
    bool hasDeltaUpdate,
    std::unique_ptr<dwio::common::SelectiveColumnReader>& keyReader,
    std::unique_ptr<dwio::common::SelectiveColumnReader>& elementReader) {
  const EncodingKey encodingKey{
      fileType.id(), params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  // Skip creating child readers that extraction pushdown doesn't need.
  // This avoids registering their streams and reduces IO.
  if (needsKeyReader(extractionType, hasDeltaUpdate)) {
    DwrfParams keyParams(
        stripe,
        params.streamLabels(),
        params.runtimeStatistics(),
        flatMapContextFromEncodingKey(encodingKey));
    keyReader = SelectiveDwrfReader::build(
        columnReaderOptions,
        requestedType.childAt(0),
        fileType.childAt(0),
        keyParams,
        *scanSpec.children()[0]);
  }
  if (needsElementReader(extractionType, hasDeltaUpdate)) {
    DwrfParams elementParams = DwrfParams(
        stripe,
        params.streamLabels(),
        params.runtimeStatistics(),
        flatMapContextFromEncodingKey(encodingKey));
    elementReader = SelectiveDwrfReader::build(
        columnReaderOptions,
        requestedType.childAt(1),
        fileType.childAt(1),
        elementParams,
        *scanSpec.children()[1]);
  }
}

} // namespace

SelectiveMapColumnReader::SelectiveMapColumnReader(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec)
    : dwio::common::SelectiveMapColumnReader(
          requestedType,
          fileType,
          params,
          scanSpec),
      length_(makeLengthDecoder(*fileType_, params, *pool_)) {
  makeMapChildrenReaders(
      *fileType_,
      *requestedType_,
      params,
      columnReaderOptions,
      *scanSpec_,
      scanSpec.extractionType(),
      scanSpec.deltaUpdate() != nullptr,
      keyReader_,
      elementReader_);
  if (keyReader_) {
    children_.push_back(keyReader_.get());
  }
  if (elementReader_) {
    children_.push_back(elementReader_.get());
  }
}

SelectiveMapAsStructColumnReader::SelectiveMapAsStructColumnReader(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec)
    : dwio::common::SelectiveMapAsStructColumnReader(
          requestedType,
          fileType,
          params,
          scanSpec),
      length_(makeLengthDecoder(*fileType_, params, *pool_)) {
  // MapAsStruct never uses extraction pushdown (asserted in base class),
  // so always create both readers.
  makeMapChildrenReaders(
      *fileType_,
      *requestedType_,
      params,
      columnReaderOptions,
      mapScanSpec_,
      common::ScanSpec::ExtractionType::kNone,
      /*hasDeltaUpdate=*/false,
      keyReader_,
      elementReader_);
  children_ = {keyReader_.get(), elementReader_.get()};
}

} // namespace facebook::velox::dwrf
