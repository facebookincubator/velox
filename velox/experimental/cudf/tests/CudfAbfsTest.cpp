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

// IMPORTANT: Includes that reference `exec::Driver` / `exec::DriverFactory`
// from inside `namespace facebook::velox::cudf_velox` (e.g. ToCudf.h) must be
// included before `CudfAbfsTest.h`, which opens `cudf_velox::exec::test` and
// would otherwise shadow the upstream `velox::exec` namespace during qualified
// name lookup. This mirrors the pattern already used by `S3ReadTest.cpp`.
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/CudfAbfsTest.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"

namespace facebook::velox::cudf_velox::exec::test {

namespace velox_filesystems = ::facebook::velox::filesystems;

namespace {

// Local Azurite blob endpoint port. Matches the upstream
// AbfsFileSystemTest pattern of using a fixed port per test binary.
constexpr int kAzuritePort{12'345};

// Connector ID used by tests built on this fixture. Matches the value used
// by the rest of the cudf test suite so split builders and table handles
// remain interchangeable.
constexpr const char* kCudfHiveConnectorId{"test-cudf-hive"};

} // namespace

void CudfAbfsTest::SetUpTestCase() {
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
}

void CudfAbfsTest::SetUp() {
  ioExecutor_ = std::make_shared<folly::IOThreadPoolExecutor>(3);

  azuriteServer_ =
      std::make_unique<velox_filesystems::AzuriteServer>(kAzuritePort);
  azuriteServer_->start();

  velox_filesystems::registerLocalFileSystem();
  velox_filesystems::registerAbfsFileSystem();

  // Force the cudf hive connector down the BufferedInput path so reads route
  // through the upstream AbfsFileSystem rather than kvikIO.
  auto hiveConfig = azuriteServer_->hiveConfig(
      {{cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInput,
        "true"}});

  velox_filesystems::registerAzureClientProvider(*hiveConfig);

  cudf_velox::registerCudf();

  cudf_velox::connector::hive::CudfHiveConnectorFactory factory;
  auto hiveConnector =
      factory.newConnector(kCudfHiveConnectorId, hiveConfig, ioExecutor_.get());
  ::facebook::velox::connector::ConnectorRegistry::global().insert(
      hiveConnector->connectorId(), hiveConnector);
}

void CudfAbfsTest::TearDown() {
  ::facebook::velox::connector::ConnectorRegistry::global().erase(
      kCudfHiveConnectorId);
  cudf_velox::unregisterCudf();
  if (azuriteServer_) {
    azuriteServer_->stop();
    azuriteServer_.reset();
  }
  ioExecutor_.reset();
}

std::string CudfAbfsTest::fileURI() const {
  return azuriteServer_->fileURI();
}

std::string CudfAbfsTest::uploadFile(const std::string& localPath) {
  // Upstream AzuriteServer uploads under a hardcoded blob name; the
  // resulting URI is `azuriteServer_->fileURI()`. Multi-file workflows
  // must add a separate uploader rather than mutate upstream behavior.
  azuriteServer_->addFile(localPath);
  return fileURI();
}

} // namespace facebook::velox::cudf_velox::exec::test
