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

#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include "velox/connectors/hive/iceberg/IcebergSessionCredentials.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/connectors/hive/iceberg/tests/RecordingFileSystem.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using test::RecordingFileSystem;
using test::recordingFileSystem;
using test::registerRecordingFileSystem;

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
