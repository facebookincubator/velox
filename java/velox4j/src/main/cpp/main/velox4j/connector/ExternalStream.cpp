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
#include "velox4j/connector/ExternalStream.h"

#include <folly/json/dynamic-inl.h>
#include <glog/logging.h>
#include <ostream>

#include <velox/common/serialization/DeserializationRegistry.h>
#include <velox/connectors/Connector.h>
#include <velox/exec/Driver.h>
#include <velox/exec/Task.h>

namespace facebook::velox4j {
using namespace facebook::velox;

SuspendedSection::SuspendedSection(facebook::velox::exec::Driver* driver)
    : driver_(driver) {
  if (driver_->task()->enterSuspended(driver_->state()) !=
      facebook::velox::exec::StopReason::kNone) {
    VELOX_FAIL(
        "Terminate detected when entering suspended section for driver "
        "{} from task {}",
        driver_->driverCtx()->driverId,
        driver_->task()->taskId());
  }
}

SuspendedSection::~SuspendedSection() {
  if (driver_->task()->leaveSuspended(driver_->state()) !=
      facebook::velox::exec::StopReason::kNone) {
    LOG(WARNING)
        << "Terminate detected when leaving suspended section for driver "
        << driver_->driverCtx()->driverId << " from task "
        << driver_->task()->taskId();
  }
}

ExternalStreamConnectorSplit::ExternalStreamConnectorSplit(
    const std::string& connectorId,
    ObjectHandle externalStreamId)
    : ConnectorSplit(connectorId), externalStreamId_(externalStreamId) {}

const ObjectHandle ExternalStreamConnectorSplit::externalStreamId() const {
  return externalStreamId_;
}

folly::dynamic ExternalStreamConnectorSplit::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "ExternalStreamConnectorSplit";
  obj["connectorId"] = connectorId;
  obj["externalStreamId"] = externalStreamId_;
  return obj;
}

void ExternalStreamConnectorSplit::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("ExternalStreamConnectorSplit", create);
}

std::shared_ptr<ExternalStreamConnectorSplit>
ExternalStreamConnectorSplit::create(const folly::dynamic& obj, void* context) {
  const auto connectorId = obj["connectorId"].asString();
  const auto externalStreamId = obj["externalStreamId"].asInt();
  return std::make_shared<ExternalStreamConnectorSplit>(
      connectorId, externalStreamId);
}

ExternalStreamTableHandle::ExternalStreamTableHandle(
    const std::string& connectorId)
    : ConnectorTableHandle(connectorId) {}

folly::dynamic ExternalStreamTableHandle::serialize() const {
  folly::dynamic obj =
      ConnectorTableHandle::serializeBase("ExternalStreamTableHandle");
  return obj;
}

void ExternalStreamTableHandle::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("ExternalStreamTableHandle", create);
}

connector::ConnectorTableHandlePtr ExternalStreamTableHandle::create(
    const folly::dynamic& obj,
    void* context) {
  auto connectorId = obj["connectorId"].asString();
  return std::make_shared<const ExternalStreamTableHandle>(connectorId);
}

ExternalStreamDataSource::ExternalStreamDataSource(
    const facebook::velox::connector::ConnectorTableHandlePtr& tableHandle)
    : DataSource() {
  tableHandle_ =
      std::dynamic_pointer_cast<const ExternalStreamTableHandle>(tableHandle);
}

void ExternalStreamDataSource::addSplit(
    std::shared_ptr<connector::ConnectorSplit> split) {
  VELOX_CHECK_EQ(
      split->connectorId,
      tableHandle_->connectorId(),
      "Split's connector ID doesn't match table handle's connector ID");
  auto externalStreamSplit =
      std::dynamic_pointer_cast<ExternalStreamConnectorSplit>(split);
  auto externalStream = ObjectStore::retrieve<ExternalStream>(
      externalStreamSplit->externalStreamId());
  streams_.push(externalStream);
}

std::optional<RowVectorPtr> ExternalStreamDataSource::next(
    uint64_t size,
    ContinueFuture& future) {
  // TODO: Sizing the output row vectors with the size argument.
  while (true) {
    if (current_ == nullptr) {
      if (streams_.empty()) {
        // End of all streams.
        return nullptr;
      }
      current_ = streams_.front();
      streams_.pop();
      continue;
    }
    {
      // We are leaving Velox task execution and are entering external code.
      // Suspend the current driver to make the current task open to spilling.
      //
      // When a task is getting spilled, it should have been suspended so has
      // zero running threads, otherwise there's possibility that this spill
      // call hangs. See
      // https://github.com/apache/incubator-gluten/issues/7243. As of now,
      // non-zero running threads usually happens when:
      // 1. Task A spills task B;
      // 2. Task A tries to grow buffers created by task B, during which spill
      // is requested on task A again.
      const exec::DriverThreadContext* driverThreadCtx =
          exec::driverThreadContext();
      VELOX_CHECK_NOT_NULL(
          driverThreadCtx,
          "ExternalStreamDataSource::next() is not called "
          "from a driver thread");
      SuspendedSection ss(driverThreadCtx->driverCtx()->driver);
      const std::optional<RowVectorPtr> vector = current_->read(future);
      if (vector == nullptr) {
        // End of the current stream.
        current_ = nullptr;
        continue;
      }
      return vector;
    }
  }
}

void ExternalStreamDataSource::cancel() {
  // Reset the pending streams because it may hold pointer to the resident
  // driver that causes reference cycle eventually.
  // See https://github.com/facebookincubator/velox/pull/12701.
  current_.reset();
  while (!streams_.empty()) {
    streams_.pop();
  }
}

ExternalStreamConnector::ExternalStreamConnector(
    const std::string& id,
    const std::shared_ptr<const config::ConfigBase>& config)
    : connector::Connector(id), config_(config) {}

std::unique_ptr<connector::DataSource>
ExternalStreamConnector::createDataSource(
    const RowTypePtr& outputType,
    const connector::ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& columnHandles,
    connector::ConnectorQueryCtx* connectorQueryCtx) {
  VELOX_CHECK(
      columnHandles.empty(),
      "ExternalStreamConnector doesn't accept column handles");
  return std::make_unique<ExternalStreamDataSource>(tableHandle);
}

ExternalStreamConnectorFactory::ExternalStreamConnectorFactory()
    : ConnectorFactory(kConnectorName) {}

std::shared_ptr<connector::Connector>
ExternalStreamConnectorFactory::newConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* ioExecutor,
    folly::Executor* cpuExecutor) {
  return std::make_shared<ExternalStreamConnector>(id, config);
}
} // namespace facebook::velox4j
