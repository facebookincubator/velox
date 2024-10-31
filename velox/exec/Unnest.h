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
  /// Represents the range of rows to process and indicates that first and last
  /// row may need to be processed partially to match the output batch size.
  /// The row range is firstRowStart_ to lasRowEnd when there is
  /// single row to process, moreover, the row range for first row is
  /// firstRowStart_ to rawMaxSizes_[firstRow] and the row range for last row is
  /// 0 to lastRowEnd when there are several rows to process.
  struct RowRange {
    // First input row to be included in the output.
    const vector_size_t start;

    // Number of input rows to be included in the output.
    const vector_size_t size;

    /// Processing of the last input row begins at index firstRowStart_ or 0
    /// depending on whether the row to process is first row and ends at
    /// 'lastRowEnd'.
    const vector_size_t lastRowEnd;
  };

  /// Extract the range of rows to process.
  /// @param size The size of input RowVector.
  /// @param numElements Records the number of output rows.
  /// @param lastRowPartial True when processing last row partially, the
  /// firstRowStart_ is lastRowEnd, otherwise, the firstRowStart_ is 0.
  const RowRange extractRowRange(
      vector_size_t size,
      vector_size_t& numElements,
      bool& lastRowPartial);

  // Generate output for 'rowRange' represented rows.
  // @param rowRange Range of rows to process.
  // @param outputSize Pre-computed number of output rows.
  RowVectorPtr generateOutput(
      const RowRange& rowRange,
      vector_size_t outputSize);

  // Invoked by generateOutput function above to generate the repeated output
  // columns.
  void generateRepeatedColumns(
      const RowRange& rowRange,
      vector_size_t numElements,
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
      const RowRange& rowRange,
      vector_size_t numElements);

  // Invoked by generateOutput for the ordinality column.
  VectorPtr generateOrdinalityVector(
      const RowRange& rowRange,
      vector_size_t numElements);

  const bool withOrdinality_;
  std::vector<column_index_t> unnestChannels_;

  std::vector<DecodedVector> unnestDecoded_;

  // The maximum number of output batch rows.
  const uint32_t maxOutputSize_;
  BufferPtr maxSizes_;
  vector_size_t* rawMaxSizes_{nullptr};

  // Start processing the first row input from `firstRowStart_`.
  vector_size_t firstRowStart_ = 0;

  std::vector<const vector_size_t*> rawSizes_;
  std::vector<const vector_size_t*> rawOffsets_;
  std::vector<const vector_size_t*> rawIndices_;

  // Next 'input_' row to process in getOutput().
  vector_size_t nextInputRow_{0};
};
} // namespace facebook::velox::exec
