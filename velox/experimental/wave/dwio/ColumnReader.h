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

#include "velox/dwio/common/TypeWithId.h"
#include "velox/experimental/wave/dwio/FormatData.h"
#include "velox/experimental/wave/exec/Wave.h"

namespace facebook::velox::wave {

class ReadStream;
class StructColumnReader;

/// dwio::SelectiveColumnReader for Wave
class ColumnReader {
 public:
  ColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      OperandId operand,
      FormatParams& params,
      velox::common::ScanSpec& scanSpec)
      : requestedType_(requestedType),
        fileType_(fileType),
        operand_(operand),
        formatData_(params.toFormatData(fileType_, scanSpec, operand)),
        scanSpec_(&scanSpec) {}

  virtual ~ColumnReader() = default;

  const common::ScanSpec& scanSpec() const {
    return *scanSpec_;
  }

  const std::vector<ColumnReader*> children() const {
    return children_;
  }

  int32_t totalRows() const {
    return formatData_->totalRows();
  }

  OperandId operand() const {
    return operand_;
  }

  virtual void makeOp(
      ReadStream* readStream,
      ColumnAction action,
      int32_t offset,
      RowSet rows,
      ColumnOp& op);

  FormatData* formatData() const {
    return formatData_.get();
  }

 protected:
  TypePtr requestedType_;
  std::shared_ptr<const dwio::common::TypeWithId> fileType_;
  const OperandId operand_;
  std::unique_ptr<FormatData> formatData_;
  // Specification of filters, value extraction, pruning etc. The
  // spec is assigned at construction and the contents may change at
  // run time based on adaptation. Owned by caller.
  velox::common::ScanSpec* scanSpec_;

  std::vector<ColumnReader*> children_;

  // Row number after last read row, relative to the ORC stripe or Parquet
  // Rowgroup start.
  vector_size_t readOffset_ = 0;
};

class ReadStream : public Executable {
 public:
  ReadStream(
      StructColumnReader* columnReader,
      vector_size_t offset,
      RowSet rows,
      WaveStream& waveStream,
      const OperandSet* firstColumns = nullptr);

  /// Runs a sequence of kernel invocations until all eagerly produced columns
  /// have their last kernel in flight. Transfers ownership of 'readStream' to
  /// its WaveStream.
  static void launch(std::unique_ptr<ReadStream>&& readStream);

  DecodePrograms& programs() {
    return programs_;
  }

  // Prepares the next kernel launch in 'programs_'. Returns true if
  // all non-lazy activity will be complete after the program kernel
  // completes. Sets needSync if the next step(s) depend on the stream
  // being synced first, i.e. a device to host transfer must have
  // completed so that the next step can decide based on data received
  // from device.
  bool makePrograms(bool& needSync);

 private:
  /// Makes column dependencies.
  void makeOps();

  StructColumnReader* reader_;
  int32_t offset_;
  RowSet rows_;
  std::vector<ColumnOp> ops_;
  std::vector<std::unique_ptr<SplitStaging>> staging_;
  SplitStaging* currentStaging_;

  // Data to be copied from device, e.g. filter selectivities.
  ResultStaging resultStaging_;
  // Intermediate data to stay on device, e.g. selected rows.
  ResultStaging deviceStaging_;
  // Reusable control block for launching decode kernels.
  DecodePrograms programs_;
};

} // namespace facebook::velox::wave
