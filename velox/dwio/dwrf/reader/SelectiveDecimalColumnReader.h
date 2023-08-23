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

#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/common/SelectiveColumnReaderInternal.h"
#include "velox/dwio/dwrf/common/DecoderUtil.h"
#include "velox/dwio/dwrf/reader/DwrfData.h"

namespace facebook::velox::dwrf {

using namespace dwio::common;

template <typename DataT>
class SelectiveDecimalColumnReader : public SelectiveColumnReader {
 public:
  SelectiveDecimalColumnReader(
      const std::shared_ptr<const TypeWithId>& nodeType,
      DwrfParams& params,
      common::ScanSpec& scanSpec);

  void seekToRowGroup(uint32_t index) override;

  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* nulls) override;

  void getValues(RowSet rows, VectorPtr* result) override;

  bool hasBulkPath() const override {
    return bulkPathEnable_;
  }

 private:
  /**
   * helper method used by processFilter and processValueHook
   * @param rows target rows to read
   * @param decodeFilter Filter during decoding phase
   *        maybe AlwaysTrue or IsNotNull
   * @param extractValues Function class to extract values
   * @param valuesFilter Filter during getValues phase
   */
  template <bool isDense, typename TFilter, typename ExtractValues>
  void readHelper(
      RowSet rows,
      velox::common::Filter* decodeFilter,
      ExtractValues extractValues,
      const velox::common::Filter& valuesFilter);

  template <bool isDense, typename ExtractValues>
  void processFilter(
      velox::common::Filter* filter,
      ExtractValues extractValues,
      RowSet rows);

  template <bool isDence>
  void processValueHook(RowSet rows, ValueHook* hook);

  std::unique_ptr<IntDecoder<true>> valueDecoder_;
  std::unique_ptr<IntDecoder<true>> scaleDecoder_;

  BufferPtr scaleBuffer_;
  RleVersion version_;
  int32_t scale_ = 0;

  // Will be false during prepare phase, in order to ensure that
  // resultNulls_ will be allocated by prepareNulls(), or addNull() will fail.
  bool bulkPathEnable_ = false;
};

} // namespace facebook::velox::dwrf
