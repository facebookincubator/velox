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

#include "velox/experimental/wave/dwio/nimble/NimbleSplitReader.h"

#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/BufferedInput.h"

namespace facebook::velox::wave {

using common::Subfield;

NimbleSplitReader::NimbleSplitReader(
    const std::shared_ptr<connector::ConnectorSplit>& split,
    const SplitReaderParams& params,
    const DefinesMap* defines)
    : StripedSplitReader(params), defines_(defines) {
  auto* hiveSplit =
      dynamic_cast<connector::hive::HiveConnectorSplit*>(split.get());
  VELOX_CHECK_NOT_NULL(hiveSplit);
  fileHandle_ = params_.fileHandleFactory->generate(
      FileHandleKey{hiveSplit->filePath},
      hiveSplit->properties.has_value() ? &*hiveSplit->properties : nullptr);
  auto* pool = params_.connectorQueryCtx->memoryPool();
  auto input =
      std::make_unique<dwio::common::BufferedInput>(fileHandle_->file, *pool);
  readerBase_ = nimble::ReaderBase::create(
      std::move(input), dwio::common::ReaderOptions(pool));
  const auto& tablet = readerBase_->tablet();
  const uint64_t end = hiveSplit->start + hiveSplit->length;
  for (uint32_t i = 0; i < tablet.stripeCount(); ++i) {
    const auto offset = tablet.stripeOffset(i);
    if (offset >= hiveSplit->start && offset < end) {
      stripes_.push_back(i);
    }
  }
  fileInfo_.file = fileHandle_->file.get();
  fileInfo_.fileId = &fileHandle_->uuid;
  fileInfo_.cache = params_.connectorQueryCtx->cache();
}

void NimbleSplitReader::loadStripe(uint32_t stripe) {
  const auto& tablet = readerBase_->tablet();
  const auto& nimbleRow = readerBase_->nimbleSchema()->asRow();
  std::vector<uint32_t> streamIds;
  const auto children = params_.scanSpec->stableChildren();
  for (const auto& childSpec : *children) {
    auto idx = nimbleRow.findChild(childSpec->fieldName());
    if (!idx.has_value() || childSpec->isConstant()) {
      // The column is not in the file.
      continue;
    }
    NimbleFormatReader::addStreamIds(
        *nimbleRow.childAt(idx.value()), childSpec.get(), streamIds);
  }
  const auto stripeIdentifier = tablet.stripeIdentifier(stripe);
  std::vector<nimble::TabletReader::StreamMetadata> locations(streamIds.size());
  tablet.streamLocations(stripeIdentifier, streamIds, locations);
  const auto stripeOffset = tablet.stripeOffset(stripe);
  auto& input = readerBase_->input();
  std::vector<
      std::pair<uint32_t, std::unique_ptr<dwio::common::SeekableInputStream>>>
      inputs;
  for (auto i = 0; i < streamIds.size(); ++i) {
    if (locations[i].size == 0) {
      continue;
    }
    dwio::common::StreamIdentifier sid(static_cast<int32_t>(streamIds[i]));
    inputs.emplace_back(
        streamIds[i],
        input.enqueue(
            {stripeOffset + locations[i].offset, locations[i].size}, &sid));
  }
  input.load(dwio::common::LogType::STREAM);
  stripe_ = std::make_unique<NimbleStripe>();
  stripe_->numRows = static_cast<int32_t>(tablet.stripeRowCount(stripe));
  for (auto& [streamId, stream] : inputs) {
    auto& bytes = stripe_->streams[streamId];
    const void* data;
    int32_t size;
    while (stream->Next(&data, &size)) {
      bytes.append(reinterpret_cast<const char*>(data), size);
    }
    bytes.resize(bytes.size() + NimbleStripe::kStreamPadding, 0);
  }
}

std::unique_ptr<ColumnReader> NimbleSplitReader::startStripe(
    int32_t index,
    int32_t& numRows) {
  if (stripe_) {
    previousStripes_[stripeIndex_] = std::move(stripe_);
  }
  stripeIndex_ = index;
  loadStripe(stripes_[index]);
  VELOX_CHECK_NOT_NULL(stripe_);
  // NOLINTNEXTLINE(facebook-hte-MovedUniquePointer)
  numRows = stripe_->numRows;
  NimbleFormatParams formatParams(
      *params_.connectorQueryCtx->memoryPool(), readerStats_, *stripe_);
  std::vector<std::unique_ptr<Subfield::PathElement>> path;
  return NimbleFormatReader::build(
      params_.readerOutputType,
      readerBase_->fileSchemaWithId(),
      readerBase_->nimbleSchema().get(),
      formatParams,
      *params_.scanSpec,
      path,
      *defines_,
      true);
}

void NimbleSplitReader::releaseStripe(int32_t index) {
  previousStripes_.erase(index);
}

namespace {

class NimbleSplitReaderFactory : public WaveSplitReaderFactory {
 public:
  std::shared_ptr<WaveSplitReader> create(
      const std::shared_ptr<connector::ConnectorSplit>& split,
      const SplitReaderParams& params,
      const DefinesMap* defines) override {
    auto* hiveSplit =
        dynamic_cast<connector::hive::HiveConnectorSplit*>(split.get());
    if (!hiveSplit ||
        hiveSplit->fileFormat != dwio::common::FileFormat::NIMBLE) {
      return nullptr;
    }
    return std::make_shared<NimbleSplitReader>(split, params, defines);
  }
};

} // namespace

// static
void NimbleSplitReader::registerFactory() {
  static bool registered = false;
  if (registered) {
    return;
  }
  registered = true;
  WaveSplitReader::registerFactory(
      std::make_unique<NimbleSplitReaderFactory>());
}

} // namespace facebook::velox::wave
