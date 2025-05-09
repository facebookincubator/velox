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

#include <velox/connectors/Connector.h>
#include <velox/exec/Driver.h>
#include <velox/exec/Task.h>
#include "velox4j/lifecycle/ObjectStore.h"

namespace velox4j {
class SuspendedSection {
 public:
  explicit SuspendedSection(facebook::velox::exec::Driver* driver);

  virtual ~SuspendedSection();

 private:
  facebook::velox::exec::Driver* const driver_;
};

class ExternalStream {
 public:
  ExternalStream() = default;

  // Delete copy/move CTORs.
  ExternalStream(ExternalStream&&) = delete;
  ExternalStream(const ExternalStream&) = delete;
  ExternalStream& operator=(const ExternalStream&) = delete;
  ExternalStream& operator=(ExternalStream&&) = delete;

  virtual ~ExternalStream() = default;

  virtual std::optional<facebook::velox::RowVectorPtr> read(
      facebook::velox::ContinueFuture& future) = 0;
};

class ExternalStreamConnectorSplit
    : public facebook::velox::connector::ConnectorSplit {
 public:
  ExternalStreamConnectorSplit(
      const std::string& connectorId,
      ObjectHandle esId);

  const ObjectHandle esId() const;

  folly::dynamic serialize() const override;

  static void registerSerDe();

  static std::shared_ptr<ExternalStreamConnectorSplit> create(
      const folly::dynamic& obj,
      void* context);

 private:
  const ObjectHandle esId_;
};

class ExternalStreamTableHandle
    : public facebook::velox::connector::ConnectorTableHandle {
 public:
  explicit ExternalStreamTableHandle(const std::string& connectorId);

  std::string toString() const override {
    return "ExternalStreamTableHandle";
  }

  folly::dynamic serialize() const override;

  static void registerSerDe();

  static facebook::velox::connector::ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);
};

class ExternalStreamDataSource : public facebook::velox::connector::DataSource {
 public:
  explicit ExternalStreamDataSource(
      const std::shared_ptr<facebook::velox::connector::ConnectorTableHandle>&
          tableHandle);

  void addSplit(std::shared_ptr<facebook::velox::connector::ConnectorSplit>
                    split) override;

  std::optional<facebook::velox::RowVectorPtr> next(
      uint64_t size,
      facebook::velox::ContinueFuture& future) override;

  void addDynamicFilter(
      facebook::velox::column_index_t outputChannel,
      const std::shared_ptr<facebook::velox::common::Filter>& filter) override {
    // TODO.
    VELOX_NYI();
  }

  uint64_t getCompletedBytes() override {
    // TODO.
    return 0;
  }

  uint64_t getCompletedRows() override {
    // TODO.
    return 0;
  }

  std::unordered_map<std::string, facebook::velox::RuntimeCounter>
  runtimeStats() override {
    // TODO.
    return {};
  }

  void cancel() override;

 private:
  std::shared_ptr<ExternalStreamTableHandle> tableHandle_;
  std::queue<std::shared_ptr<ExternalStream>> streams_{};
  std::shared_ptr<ExternalStream> current_{nullptr};
};

class ExternalStreamConnector : public facebook::velox::connector::Connector {
 public:
  ExternalStreamConnector(
      const std::string& id,
      const std::shared_ptr<const facebook::velox::config::ConfigBase>& config);

  std::unique_ptr<facebook::velox::connector::DataSource> createDataSource(
      const facebook::velox::RowTypePtr& outputType,
      const std::shared_ptr<facebook::velox::connector::ConnectorTableHandle>&
          tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<facebook::velox::connector::ColumnHandle>>&
          columnHandles,
      facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx)
      override;

  std::unique_ptr<facebook::velox::connector::DataSink> createDataSink(
      facebook::velox::RowTypePtr inputType,
      std::shared_ptr<facebook::velox::connector::ConnectorInsertTableHandle>
          connectorInsertTableHandle,
      facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx,
      facebook::velox::connector::CommitStrategy commitStrategy) override {
    VELOX_NYI();
  }

 private:
  std::shared_ptr<const facebook::velox::config::ConfigBase> config_;
};

class ExternalStreamConnectorFactory
    : public facebook::velox::connector::ConnectorFactory {
 public:
  static constexpr const char* kConnectorName = "external-stream";

  ExternalStreamConnectorFactory();

  std::shared_ptr<facebook::velox::connector::Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const facebook::velox::config::ConfigBase> config,
      folly::Executor* ioExecutor,
      folly::Executor* cpuExecutor) override;
};

} // namespace velox4j
