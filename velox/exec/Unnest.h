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
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {
class Unnest : public Operator {
 public:
  Unnest(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::UnnestNode>& unnestNode);

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return BlockingReason::kNotBlocked;
  }

  bool needsInput() const override {
    return input_ == nullptr;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool isFinished() override;

 private:
  // Represents the range of rows to process and indicates that the first and
  // last
  // rows may need to be processed partially to match the configured output
  // batch size. When processing a single row, the range is from
  // 'firstRowStart_' to 'lastRowEnd'. For multiple rows, the range for the
  // first row is from 'firstRowStart_' to 'rawMaxSizes_[firstRow]', and for the
  // last row, it is from 0 to 'lastRowEnd', unless the last row is processed
  // fully, in which case' rawMaxSizes_[lastRow]' is used as the end of the last
  // row.
  //
  // Single row:
  //    firstRowStart_  firstRowEnd = lastRowEnd
  //---|----------------|--- start, size = 1
  //
  // Multiple rows:
  //    firstRowStart_     firstRowEnd = rawMaxSizes_[start]
  //---|-------------------| start
  //-----------------------
  //-----------------------
  //-----------------|----- end - 1
  //                 lastRowEnd
  // The size is end - start, the range is [start, end).
  struct RowRange {
    // Invokes a function on each row represented by the RowRange.
    // @param func The function to call for each row. 'row' is the current row
    // number in the '[start, start + size)' range, 'start' is the row number to
    // start processing, and 'size' is the number of rows to process..
    // @param rawMaxSizes Used to compute the end of each row.
    // @param firstRowStart The index to start processing the first row. Same
    // with Unnest member firstRowStart_.
    void forEachRow(
        std::function<void(
            vector_size_t /*row*/,
            vector_size_t /*start*/,
            vector_size_t /*size*/)> func,
        const vector_size_t* const rawMaxSizes,
        vector_size_t firstRowStart) const;

    // First input row to be included in the output.
    const vector_size_t start;

    // Number of input rows to be included in the output.
    const vector_size_t size;

    // The processing of the last input row starts at index 'firstRowStart_' or
    // 0, depending on whether it is the first row being processed, and ends at
    // 'lastRowEnd'. It is nullopt when the last row is processed completely.
    const std::optional<vector_size_t> lastRowEnd;

    // Total number of inner rows in the range.
    const vector_size_t numElements;
  };

  // Extract the range of rows to process.
  // @param size The size of input RowVector.
  RowRange extractRowRange(vector_size_t size) const;

  // Generate output for 'rowRange' represented rows.
  // @param rowRange Range of rows to process.
  RowVectorPtr generateOutput(const RowRange& rowRange);

  // Invoked by generateOutput function above to generate the repeated output
  // columns.
  void generateRepeatedColumns(
      const RowRange& rowRange,
      std::vector<VectorPtr>& outputs);

  struct UnnestChannelEncoding {
    BufferPtr indices;
    BufferPtr nulls;
    bool identityMapping;

    VectorPtr wrap(const VectorPtr& base, vector_size_t wrapSize) const;
  };

  // Invoked by generateOutput above to generate the encoding for the unnested
  // Array or Map.
  const UnnestChannelEncoding generateEncodingForChannel(
      column_index_t channel,
      const RowRange& rowRange);

  // Invoked by generateOutput for the ordinality column.
  VectorPtr generateOrdinalityVector(const RowRange& rowRange);

  const bool withOrdinality_;
  std::vector<column_index_t> unnestChannels_;

  std::vector<DecodedVector> unnestDecoded_;

  // The maximum number of output batch rows.
  const uint32_t maxOutputSize_;
  BufferPtr maxSizes_;
  vector_size_t* rawMaxSizes_{nullptr};

  // The index to start processing the first row.
  vector_size_t firstRowStart_ = 0;

  std::vector<const vector_size_t*> rawSizes_;
  std::vector<const vector_size_t*> rawOffsets_;
  std::vector<const vector_size_t*> rawIndices_;

  // Next 'input_' row to process in getOutput().
  vector_size_t nextInputRow_{0};
};
} // namespace facebook::velox::exec
