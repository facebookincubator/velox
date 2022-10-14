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

#include "velox/connectors/WriteProtocol.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

class TableWriterWriteInfo : public connector::WriteInfo {
 public:
  TableWriterWriteInfo(
      RowTypePtr outputType,
      const std::string& taskId,
      vector_size_t numWrittenRows,
      std::vector<std::shared_ptr<const connector::WriterParameters>>
          writeParameters)
      : outputType_(std::move(outputType)),
        taskId_(taskId),
        numWrittenRows_(numWrittenRows),
        writeParameters_(std::move(writeParameters)) {}

  ~TableWriterWriteInfo() override {}

  RowTypePtr outputType() const {
    return outputType_;
  }

  const std::string& taskId() const {
    return taskId_;
  }

  vector_size_t numWrittenRows() const {
    return numWrittenRows_;
  }

  const std::vector<std::shared_ptr<const connector::WriterParameters>>&
  writeParameters() const {
    return writeParameters_;
  }

 private:
  const RowTypePtr outputType_;
  const std::string taskId_;
  const vector_size_t numWrittenRows_;
  const std::vector<std::shared_ptr<const connector::WriterParameters>>
      writeParameters_;
};

/**
 * The class implements a simple table writer VELOX operator
 */
class TableWriter : public Operator {
 public:
  TableWriter(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::TableWriteNode>& tableWriteNode,
      connector::WriteProtocol::CommitStrategy commitStrategy);

  BlockingReason isBlocked(ContinueFuture* /* future */) override {
    return BlockingReason::kNotBlocked;
  }

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override {
    Operator::noMoreInput();
    close();
  }

  virtual bool needsInput() const override {
    return true;
  }

  void close() override {
    if (!closed_) {
      if (dataSink_) {
        dataSink_->close();
      }
      closed_ = true;
    }
  }

  RowVectorPtr getOutput() override;

  bool isFinished() override {
    return finished_;
  }

  RowTypePtr outputType() const {
    return outputType_;
  }

  vector_size_t numWrittenRows() const {
    return numWrittenRows_;
  }

  const DriverCtx* driverCtx() const {
    return driverCtx_;
  }

  std::shared_ptr<connector::DataSink> dataSink() const {
    return dataSink_;
  }

 private:
  void createDataSink();

  std::vector<column_index_t> inputMapping_;
  std::shared_ptr<const RowType> mappedType_;
  vector_size_t numWrittenRows_;
  bool finished_;
  bool closed_;
  DriverCtx* driverCtx_;
  std::shared_ptr<connector::Connector> connector_;
  std::shared_ptr<connector::ConnectorQueryCtx> connectorQueryCtx_;
  std::shared_ptr<connector::DataSink> dataSink_;
  std::shared_ptr<connector::WriteProtocol> writeProtocol_;
  std::shared_ptr<connector::ConnectorInsertTableHandle> insertTableHandle_;
};
} // namespace facebook::velox::exec
