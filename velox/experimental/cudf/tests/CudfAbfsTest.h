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

#include "velox/connectors/hive/storage_adapters/abfs/tests/AzuriteServer.h"

#include <folly/executors/IOThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace facebook::velox::cudf_velox::exec::test {

/// Test fixture that launches an Azurite blob server, registers the upstream
/// ABFS filesystem and Azure client provider, and registers a
/// `CudfHiveConnector` configured to read through Velox's buffered input
/// data source. Uploaded files are served at `abfs://...` URIs that the
/// fixture exposes to the test body.
class CudfAbfsTest : public ::testing::Test {
 protected:
  /// Initializes the process-wide MemoryManager once for the whole test
  /// suite. Mirrors the pattern used by `S3ReadTest`.
  static void SetUpTestCase();

  /// Starts a fresh Azurite server, registers the ABFS filesystem and Azure
  /// client provider against the Azurite-derived config, registers cudf
  /// adapters, and inserts a `CudfHiveConnector` into the global registry.
  void SetUp() override;

  /// Erases the connector, unregisters cudf adapters, and stops the Azurite
  /// server.
  void TearDown() override;

  /// Returns the `abfs://...` URI of the single test file inside the
  /// Azurite container. Valid only after `uploadFile` has been called.
  std::string fileURI() const;

  /// Uploads `localPath` into the Azurite container under the upstream
  /// `AzuriteServer`'s default blob name and returns the resulting
  /// `abfs://...` URI. Only one file per fixture instance is supported;
  /// reuploading overwrites the previous content.
  std::string uploadFile(const std::string& localPath);

  /// Owns the Azurite subprocess and the per-test blob container.
  std::unique_ptr<::facebook::velox::filesystems::AzuriteServer>
      azuriteServer_;

  /// IO executor handed to the `CudfHiveConnector` for async file work.
  std::shared_ptr<folly::IOThreadPoolExecutor> ioExecutor_;
};

} // namespace facebook::velox::cudf_velox::exec::test
