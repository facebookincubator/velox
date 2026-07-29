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

#include <gtest/gtest.h>

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include "velox/connectors/hive/iceberg/IcebergSessionCredentials.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using filesystems::DirectoryOptions;
using filesystems::FileOptions;
using filesystems::FileSystem;
using filesystems::getFileSystem;
using filesystems::registerFileSystem;

// FileSystem wrapper that records the FileOptions::fileReadOps of every
// openFileForRead call and otherwise delegates to the underlying real file
// system. Used to observe that a per-query session credential reaches the
// FileSystem boundary. Paths are prefixed with scheme() and delegated to the
// local file system after stripping the prefix.
class RecordingFileSystem : public FileSystem {
 public:
  explicit RecordingFileSystem(std::shared_ptr<const config::ConfigBase> config)
      : FileSystem(std::move(config)) {}

  static std::string scheme() {
    return "recording:";
  }

  std::string name() const override {
    return "RecordingFS";
  }

  std::string_view extractPath(std::string_view path) const override {
    return path.find(scheme()) == 0 ? path.substr(scheme().length()) : path;
  }

  std::unique_ptr<ReadFile> openFileForRead(
      std::string_view path,
      const FileOptions& options = {}) override {
    {
      std::lock_guard<std::mutex> l(mutex_);
      capturedReadOps_.push_back(options.fileReadOps);
    }
    const std::string delegated{extractPath(path)};
    return delegate(delegated)->openFileForRead(delegated, options);
  }

  std::unique_ptr<WriteFile> openFileForWrite(
      std::string_view path,
      const FileOptions& options = {}) override {
    const std::string delegated{extractPath(path)};
    return delegate(delegated)->openFileForWrite(delegated, options);
  }

  void remove(std::string_view path) override {
    const std::string delegated{extractPath(path)};
    delegate(delegated)->remove(delegated);
  }

  void rename(
      std::string_view oldPath,
      std::string_view newPath,
      bool overwrite) override {
    const std::string from{extractPath(oldPath)};
    const std::string to{extractPath(newPath)};
    delegate(from)->rename(from, to, overwrite);
  }

  bool exists(std::string_view path) override {
    const std::string delegated{extractPath(path)};
    return delegate(delegated)->exists(delegated);
  }

  bool isDirectory(std::string_view path) const override {
    const std::string delegated{extractPath(path)};
    return delegate(delegated)->isDirectory(delegated);
  }

  std::vector<std::string> list(std::string_view path) override {
    const std::string delegated{extractPath(path)};
    return delegate(delegated)->list(delegated);
  }

  void mkdir(std::string_view path, const DirectoryOptions& options = {})
      override {
    const std::string delegated{extractPath(path)};
    delegate(delegated)->mkdir(delegated, options);
  }

  void rmdir(std::string_view path) override {
    const std::string delegated{extractPath(path)};
    delegate(delegated)->rmdir(delegated);
  }

  void clear() {
    std::lock_guard<std::mutex> l(mutex_);
    capturedReadOps_.clear();
  }

  // Returns the recorded value for 'key' across all openFileForRead calls, or
  // std::nullopt if no call carried it.
  std::optional<std::string> readOpValue(const std::string& key) {
    std::lock_guard<std::mutex> l(mutex_);
    for (const auto& ops : capturedReadOps_) {
      const auto it = ops.find(key);
      if (it != ops.end()) {
        return it->second;
      }
    }
    return std::nullopt;
  }

 private:
  static std::shared_ptr<FileSystem> delegate(const std::string& path) {
    return getFileSystem(path, nullptr);
  }

  mutable std::mutex mutex_;
  std::vector<folly::F14FastMap<std::string, std::string>> capturedReadOps_{};
};

std::shared_ptr<RecordingFileSystem> recordingFileSystem() {
  static auto fs = std::make_shared<RecordingFileSystem>(nullptr);
  return fs;
}

void registerRecordingFileSystem() {
  static folly::once_flag once;
  folly::call_once(once, [] {
    registerFileSystem(
        [](std::string_view path) {
          return path.find(RecordingFileSystem::scheme()) == 0;
        },
        [](std::shared_ptr<const config::ConfigBase>, std::string_view) {
          return recordingFileSystem();
        });
  });
}

// Session-property key naming the credential and the connector config that
// declares it as forwardable.
constexpr const char* kCredentialKey = "delegated_cat";
constexpr const char* kCredentialValue = "cat-token-xyz";

class IcebergSessionCredentialsE2ETest : public test::IcebergTestBase {
 protected:
  void SetUp() override {
    test::IcebergTestBase::SetUp();
    // Use DWRF so the file is written/read with the always-registered DWRF
    // factories (Parquet is not registered in this test binary).
    fileFormat_ = dwio::common::FileFormat::DWRF;
    registerRecordingFileSystem();
    recordingFileSystem()->clear();

    // Re-register the Iceberg connector so its build-time config declares
    // 'delegated_cat' as a session-credential key to forward.
    ConnectorRegistry::global().erase(test::kIcebergConnectorId);
    IcebergConnectorFactory factory;
    auto connector = factory.newConnector(
        test::kIcebergConnectorId,
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{
                {std::string(kSessionCredentialKeysConfig), kCredentialKey}}));
    ConnectorRegistry::global().insert(connector->connectorId(), connector);
  }
};

// Drives a real Iceberg table scan whose data file is opened through the
// recording file system, and asserts the per-query session credential
// (delivered via connector session properties) is forwarded all the way to
// FileOptions::fileReadOps at the file-system boundary.
TEST_F(IcebergSessionCredentialsE2ETest, forwardsSessionCredentialToReadPath) {
  const auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  const auto vectors =
      createTestData(rowType, /*numBatches=*/1, /*rowsPerBatch=*/20);

  const auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(dataFile->getPath(), vectors);

  // Route the read through the recording file system by prefixing its scheme.
  const auto splits =
      makeIcebergSplits(RecordingFileSystem::scheme() + dataFile->getPath());

  const auto plan = exec::test::PlanBuilder()
                        .startTableScan(test::kIcebergConnectorId)
                        .outputType(rowType)
                        .endTableScan()
                        .planNode();

  exec::test::AssertQueryBuilder(plan)
      .connectorSessionProperties(
          {{test::kIcebergConnectorId, {{kCredentialKey, kCredentialValue}}}})
      .splits(splits)
      .assertResults(vectors);

  EXPECT_EQ(
      recordingFileSystem()->readOpValue(kCredentialKey),
      std::optional<std::string>(kCredentialValue));
}

// Without the session property set, nothing is forwarded (no-op path).
TEST_F(IcebergSessionCredentialsE2ETest, noSessionCredentialForwardsNothing) {
  const auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  const auto vectors =
      createTestData(rowType, /*numBatches=*/1, /*rowsPerBatch=*/20);

  const auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(dataFile->getPath(), vectors);

  const auto splits =
      makeIcebergSplits(RecordingFileSystem::scheme() + dataFile->getPath());

  const auto plan = exec::test::PlanBuilder()
                        .startTableScan(test::kIcebergConnectorId)
                        .outputType(rowType)
                        .endTableScan()
                        .planNode();

  exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);

  EXPECT_EQ(recordingFileSystem()->readOpValue(kCredentialKey), std::nullopt);
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
