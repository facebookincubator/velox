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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include "velox/connectors/hive/iceberg/IcebergSessionCredentials.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using filesystems::DirectoryOptions;
using filesystems::FileOptions;
using filesystems::FileSystem;
using filesystems::getFileSystem;
using filesystems::registerFileSystem;

// ReadFile wrapper that passes the FileIoContext::fileOpts of every pread to
// 'recordRead' and otherwise delegates. The default preadv overloads route
// through pread, so vectored reads are recorded too.
class RecordingReadFile : public ReadFile {
 public:
  RecordingReadFile(
      std::unique_ptr<ReadFile> file,
      std::function<void(const folly::F14FastMap<std::string, std::string>&)>
          recordRead)
      : file_(std::move(file)), recordRead_(std::move(recordRead)) {}

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& context = {}) const override {
    recordRead_(context.fileOpts);
    return file_->pread(offset, length, buf, context);
  }

  bool shouldCoalesce() const override {
    return file_->shouldCoalesce();
  }

  uint64_t size() const override {
    return file_->size();
  }

  uint64_t memoryUsage() const override {
    return file_->memoryUsage();
  }

  std::string getName() const override {
    return file_->getName();
  }

  uint64_t getNaturalReadSize() const override {
    return file_->getNaturalReadSize();
  }

 private:
  const std::unique_ptr<ReadFile> file_;
  const std::function<void(const folly::F14FastMap<std::string, std::string>&)>
      recordRead_;
};

// FileSystem wrapper that records the FileOptions::fileReadOps of every
// openFileForRead call and the FileIoContext::fileOpts of every read from the
// opened files, and otherwise delegates to the underlying real file system.
// Used to observe that a per-query session credential reaches the FileSystem
// boundary. Paths are prefixed with scheme() and delegated to the local file
// system after stripping the prefix.
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
    return std::make_unique<RecordingReadFile>(
        delegate(delegated)->openFileForRead(delegated, options),
        [this](const folly::F14FastMap<std::string, std::string>& fileOpts) {
          std::lock_guard<std::mutex> l(mutex_);
          capturedReadContextOps_.push_back(fileOpts);
        });
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
    capturedReadContextOps_.clear();
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

  // Returns the recorded value for 'key' on each read, in read order, with an
  // empty string for a read that did not carry it.
  std::vector<std::string> readContextValues(const std::string& key) {
    std::lock_guard<std::mutex> l(mutex_);
    std::vector<std::string> values;
    values.reserve(capturedReadContextOps_.size());
    for (const auto& ops : capturedReadContextOps_) {
      const auto it = ops.find(key);
      if (it == ops.end()) {
        values.emplace_back();
      } else {
        values.push_back(it->second);
      }
    }
    return values;
  }

 private:
  static std::shared_ptr<FileSystem> delegate(const std::string& path) {
    return getFileSystem(path, nullptr);
  }

  mutable std::mutex mutex_;
  std::vector<folly::F14FastMap<std::string, std::string>> capturedReadOps_{};
  std::vector<folly::F14FastMap<std::string, std::string>>
      capturedReadContextOps_{};
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
    dwio::common::LocalFileSink::registerFactory();
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

  // Writes a data file whose c0 value equals its row position (0..3).
  std::shared_ptr<common::testutil::TempFilePath> writeRowPositionDataFile() {
    return writeDataFile(
        {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3})})});
  }

  // Scans 'dataFilePath' with 'deleteFile' applied and the session credential
  // set, and checks that rows 1 and 3 are deleted.
  void assertScanDeletesOddRows(
      const std::string& dataFilePath,
      const IcebergDeleteFile& deleteFile) {
    // Equality deletes resolve their field IDs against the data columns.
    const auto plan = makeIcebergTableScanPlan(ROW("c0", BIGINT()));

    exec::test::AssertQueryBuilder(plan)
        .connectorSessionProperties(
            {{test::kIcebergConnectorId, {{kCredentialKey, kCredentialValue}}}})
        .splits(makeIcebergSplits(dataFilePath, {deleteFile}))
        .assertResults(makeRowVector({makeFlatVector<int64_t>({0, 2})}));
  }

  // Asserts that the credential reached the open of, and every read from, the
  // files opened through the recording file system.
  static void assertCredentialOnOpenAndReads() {
    EXPECT_EQ(
        recordingFileSystem()->readOpValue(kCredentialKey),
        std::optional<std::string>(kCredentialValue));
    EXPECT_THAT(
        recordingFileSystem()->readContextValues(kCredentialKey),
        testing::AllOf(
            testing::Not(testing::IsEmpty()),
            testing::Each(std::string(kCredentialValue))));
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

  const auto plan = makeIcebergTableScanPlan(rowType);

  exec::test::AssertQueryBuilder(plan)
      .connectorSessionProperties(
          {{test::kIcebergConnectorId, {{kCredentialKey, kCredentialValue}}}})
      .splits(splits)
      .assertResults(vectors);

  EXPECT_EQ(
      recordingFileSystem()->readOpValue(kCredentialKey),
      std::optional<std::string>(kCredentialValue));
}

// Keeps the data file local and opens only the positional delete file through
// the recording file system, so everything recorded comes from the delete file
// reader.
TEST_F(
    IcebergSessionCredentialsE2ETest,
    forwardsSessionCredentialToPositionalDeleteFile) {
  const auto dataFile = writeRowPositionDataFile();

  const auto deleteFile = common::testutil::TempFilePath::create();
  writeToFile(
      deleteFile->getPath(),
      {makeRowVector(
          {"file_path", "pos"},
          {makeFlatVector<std::string>(
               {dataFile->getPath(), dataFile->getPath()}),
           makeFlatVector<int64_t>({1, 3})})});

  assertScanDeletesOddRows(
      dataFile->getPath(),
      IcebergDeleteFile(
          FileContent::kPositionalDeletes,
          RecordingFileSystem::scheme() + deleteFile->getPath(),
          fileFormat_,
          /*recordCount=*/2,
          getFileSize(deleteFile->getPath())));

  assertCredentialOnOpenAndReads();
}

// Keeps the data file local and opens only the equality delete file through
// the recording file system.
TEST_F(
    IcebergSessionCredentialsE2ETest,
    forwardsSessionCredentialToEqualityDeleteFile) {
  const auto dataFile = writeRowPositionDataFile();

  const auto deleteFile = common::testutil::TempFilePath::create();
  writeToFile(
      deleteFile->getPath(),
      {makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 3})})});

  assertScanDeletesOddRows(
      dataFile->getPath(),
      IcebergDeleteFile(
          FileContent::kEqualityDeletes,
          RecordingFileSystem::scheme() + deleteFile->getPath(),
          fileFormat_,
          /*recordCount=*/2,
          getFileSize(deleteFile->getPath()),
          // Field ID 1 is 'c0'.
          /*equalityFieldIds=*/{1}));

  assertCredentialOnOpenAndReads();
}

// Keeps the data file local and opens only the deletion vector's Puffin file
// through the recording file system.
TEST_F(
    IcebergSessionCredentialsE2ETest,
    forwardsSessionCredentialToDeletionVector) {
  const auto dataFile = writeRowPositionDataFile();
  const auto [puffinDir, dvFile] =
      writeDeletionVector(dataFile->getPath(), {1, 3});

  assertScanDeletesOddRows(
      dataFile->getPath(),
      IcebergDeleteFile(
          dvFile.content,
          RecordingFileSystem::scheme() + dvFile.filePath,
          dvFile.fileFormat,
          dvFile.recordCount,
          dvFile.fileSizeInBytes,
          /*equalityFieldIds=*/{},
          /*lowerBounds=*/{},
          /*upperBounds=*/{},
          dvFile.dataSequenceNumber,
          dvFile.contentOffset,
          dvFile.contentLength,
          dvFile.referencedDataFile));

  assertCredentialOnOpenAndReads();
}

// A delete file's own fileReadOps reach its open and every read from it. On a
// shared key the reader's value, here the session credential, wins.
TEST_F(IcebergSessionCredentialsE2ETest, deleteFileReadOpsAddKeys) {
  const auto dataFile = writeRowPositionDataFile();
  const auto assertDeleteFileReadOps = [&](IcebergDeleteFile deleteFile) {
    recordingFileSystem()->clear();
    deleteFile.fileReadOps = {
        {"delete_file_key", "delete_file_value"},
        {kCredentialKey, "delete_file_credential"},
    };
    assertScanDeletesOddRows(dataFile->getPath(), deleteFile);

    EXPECT_EQ(
        recordingFileSystem()->readOpValue("delete_file_key"),
        std::optional<std::string>("delete_file_value"));
    EXPECT_THAT(
        recordingFileSystem()->readContextValues("delete_file_key"),
        testing::AllOf(
            testing::Not(testing::IsEmpty()),
            testing::Each(std::string("delete_file_value"))));
    assertCredentialOnOpenAndReads();
  };

  {
    const auto deleteFile = common::testutil::TempFilePath::create();
    writeToFile(
        deleteFile->getPath(),
        {makeRowVector(
            {"file_path", "pos"},
            {makeFlatVector<std::string>(
                 {dataFile->getPath(), dataFile->getPath()}),
             makeFlatVector<int64_t>({1, 3})})});
    assertDeleteFileReadOps(IcebergDeleteFile(
        FileContent::kPositionalDeletes,
        RecordingFileSystem::scheme() + deleteFile->getPath(),
        fileFormat_,
        /*recordCount=*/2,
        getFileSize(deleteFile->getPath())));
  }

  {
    const auto deleteFile = common::testutil::TempFilePath::create();
    writeToFile(
        deleteFile->getPath(),
        {makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 3})})});
    assertDeleteFileReadOps(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        RecordingFileSystem::scheme() + deleteFile->getPath(),
        fileFormat_,
        /*recordCount=*/2,
        getFileSize(deleteFile->getPath()),
        // Field ID 1 is 'c0'.
        /*equalityFieldIds=*/{1}));
  }

  {
    const auto [puffinDir, dvFile] =
        writeDeletionVector(dataFile->getPath(), {1, 3});
    assertDeleteFileReadOps(IcebergDeleteFile(
        dvFile.content,
        RecordingFileSystem::scheme() + dvFile.filePath,
        dvFile.fileFormat,
        dvFile.recordCount,
        dvFile.fileSizeInBytes,
        /*equalityFieldIds=*/{},
        /*lowerBounds=*/{},
        /*upperBounds=*/{},
        dvFile.dataSequenceNumber,
        dvFile.contentOffset,
        dvFile.contentLength,
        dvFile.referencedDataFile));
  }
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

  const auto plan = makeIcebergTableScanPlan(rowType);

  exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);

  EXPECT_EQ(recordingFileSystem()->readOpValue(kCredentialKey), std::nullopt);
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
