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

#include "velox/experimental/wave/exec/tests/utils/WaveTestSplitReader.h"
#include "velox/experimental/wave/exec/tests/utils/TestFormatReader.h"

namespace facebook::velox::wave::test {

using common::Subfield;

WaveTestSplitReader::WaveTestSplitReader(
    const std::shared_ptr<connector::ConnectorSplit>& split,
    const SplitReaderParams& params,
    const DefinesMap* defines)
    : StripedSplitReader(params), defines_(defines) {
  auto hiveSplit =
      dynamic_cast<connector::hive::HiveConnectorSplit*>(split.get());
  VELOX_CHECK_NOT_NULL(hiveSplit);

  stripe_ = test::Table::getStripe(hiveSplit->filePath);
  if (!stripe_->isLoaded()) {
    try {
      FileHandleKey key{stripe_->path};
      fileHandle_ = params_.fileHandleFactory->generate(
          key,
          hiveSplit->properties.has_value() ? &*hiveSplit->properties
                                            : nullptr);
      VELOX_CHECK_NOT_NULL(fileHandle_.get());
    } catch (const VeloxRuntimeError& e) {
      if (e.errorCode() == error_code::kFileNotFound &&
          params_.hiveConfig->ignoreMissingFiles(
              params_.connectorQueryCtx->sessionProperties())) {
        emptySplit_ = true;
        return;
      }
      throw;
    }
  }
  VELOX_CHECK_NOT_NULL(stripe_);
  if (fileHandle_.get()) {
    fileInfo_.file = fileHandle_->file.get();
    fileInfo_.fileId = &fileHandle_->uuid;
    fileInfo_.cache = params_.connectorQueryCtx->cache();
  }
}

std::unique_ptr<ColumnReader> WaveTestSplitReader::startStripe(
    int32_t /*index*/,
    int32_t& numRows) {
  numRows = stripe_->columns[0]->numValues;
  TestFormatParams formatParams(
      *params_.connectorQueryCtx->memoryPool(), readerStats_, stripe_);
  std::vector<std::unique_ptr<Subfield::PathElement>> empty;
  return TestFormatReader::build(
      params_.readerOutputType,
      stripe_->typeWithId,
      formatParams,
      *params_.scanSpec,
      empty,
      *defines_,
      true);
}

namespace {
class WaveTestSplitReaderFactory : public WaveSplitReaderFactory {
 public:
  std::shared_ptr<WaveSplitReader> create(
      const std::shared_ptr<connector::ConnectorSplit>& split,
      const SplitReaderParams& params,
      const DefinesMap* defines) override {
    auto hiveSplit =
        dynamic_cast<connector::hive::HiveConnectorSplit*>(split.get());
    if (!hiveSplit) {
      return nullptr;
    }
    if (hiveSplit->filePath.size() > 11 &&
        memcmp(hiveSplit->filePath.data(), "wavemock://", 11) == 0) {
      return std::make_shared<WaveTestSplitReader>(split, params, defines);
    }
    return nullptr;
  }
};
} // namespace

//  static
void WaveTestSplitReader::registerTestSplitReader() {
  WaveSplitReader::registerFactory(
      std::make_unique<WaveTestSplitReaderFactory>());
}

} // namespace facebook::velox::wave::test
