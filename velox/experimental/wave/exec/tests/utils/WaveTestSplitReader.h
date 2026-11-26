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

#include "velox/experimental/wave/dwio/ColumnReader.h"
#include "velox/experimental/wave/dwio/StripedSplitReader.h"
#include "velox/experimental/wave/exec/tests/utils/FileFormat.h"

namespace facebook::velox::wave::test {

/// A WaveSplitReader that decodes mock Wave tables. A table file is one
/// stripe.
class WaveTestSplitReader : public StripedSplitReader {
 public:
  WaveTestSplitReader(
      const std::shared_ptr<connector::ConnectorSplit>& split,
      const SplitReaderParams& params,
      const DefinesMap* defines);

  bool emptySplit() override {
    return numStripes() == 0 || stripe_->columns[0]->numValues == 0;
  }

  static void registerTestSplitReader();

 protected:
  int32_t numStripes() const override {
    return stripe_ && !emptySplit_ ? 1 : 0;
  }

  std::unique_ptr<ColumnReader> startStripe(int32_t index, int32_t& numRows)
      override;

 private:
  const DefinesMap* const defines_;
  test::Stripe* stripe_{nullptr};
  dwio::common::ColumnRuntimeStats readerStats_{TypeKind::ROW};
  FileHandleCachedPtr fileHandle_;
  bool emptySplit_{false};
};

} // namespace facebook::velox::wave::test
