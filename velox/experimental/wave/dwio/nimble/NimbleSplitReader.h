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

#pragma once

#include <folly/container/F14Map.h>

#include "velox/dwio/nimble/selective/ReaderBase.h"
#include "velox/experimental/wave/dwio/ColumnReader.h"
#include "velox/experimental/wave/dwio/StripedSplitReader.h"
#include "velox/experimental/wave/dwio/nimble/NimbleFormatReader.h"

namespace facebook::velox::wave {

/// A WaveSplitReader that decodes Nimble files on the GPU. Reads the stripes
/// that start in the split, one stripe at a time. The encoded streams of a
/// stripe are read through the BufferedInput of the file and parsed on the
/// host. The decoding runs on the device.
class NimbleSplitReader : public StripedSplitReader {
 public:
  NimbleSplitReader(
      const std::shared_ptr<connector::ConnectorSplit>& split,
      const SplitReaderParams& params,
      const DefinesMap* defines);

  bool emptySplit() override {
    return stripes_.empty();
  }

  /// Registers a factory for splits of Nimble files.
  static void registerFactory();

 protected:
  int32_t numStripes() const override {
    return static_cast<int32_t>(stripes_.size());
  }

  std::unique_ptr<ColumnReader> startStripe(int32_t index, int32_t& numRows)
      override;

  void releaseStripe(int32_t index) override;

 private:
  // Reads the streams of the scanned columns of stripe 'stripe'.
  void loadStripe(uint32_t stripe);

  const DefinesMap* const defines_;
  FileHandleCachedPtr fileHandle_;
  std::shared_ptr<nimble::ReaderBase> readerBase_;
  // Stripes that start in the split.
  std::vector<uint32_t> stripes_;
  std::unique_ptr<NimbleStripe> stripe_;
  // Index of 'stripe_' in 'stripes_'.
  int32_t stripeIndex_{-1};
  // Data of previous stripes by index in 'stripes_'. Batches of a previous
  // stripe may still be decoding.
  folly::F14FastMap<int32_t, std::unique_ptr<NimbleStripe>> previousStripes_;
  dwio::common::ColumnRuntimeStats readerStats_{TypeKind::ROW};
};

} // namespace facebook::velox::wave
