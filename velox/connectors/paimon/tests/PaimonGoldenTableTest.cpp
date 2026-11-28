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

// Reads tables that Paimon Java wrote, with the splits Paimon Java planned for
// them, and checks the rows against what Paimon Java read from the same
// splits. The cases live in examples/golden; see the README there.

#include <fstream>
#include <sstream>

#include <folly/json.h>
#include <gtest/gtest.h>

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/paimon/PaimonColumnHandle.h"
#include "velox/connectors/paimon/PaimonConnector.h"
#include "velox/connectors/paimon/PaimonConnectorSplit.h"
#include "velox/connectors/paimon/PaimonTableHandle.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/type/fbhive/HiveTypeParser.h"

namespace facebook::velox::connector::paimon {
namespace {

const std::string kPaimonConnectorId{"test-paimon"};

folly::dynamic readJson(const std::string& path) {
  std::ifstream file(path);
  VELOX_CHECK(file.good(), "Cannot open golden file: {}", path);
  std::stringstream contents;
  contents << file.rdbuf();
  return folly::parseJson(contents.str());
}

// Returns the value of 'json', a value of expected.json, as a Variant of
// 'type'.
Variant toVariant(const folly::dynamic& json, const TypePtr& type) {
  if (json.isNull()) {
    return Variant::null(type->kind());
  }
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return Variant(json.asBool());
    case TypeKind::TINYINT:
      return Variant(static_cast<int8_t>(json.asInt()));
    case TypeKind::SMALLINT:
      return Variant(static_cast<int16_t>(json.asInt()));
    case TypeKind::INTEGER:
      return Variant(static_cast<int32_t>(json.asInt()));
    case TypeKind::BIGINT:
      return Variant(static_cast<int64_t>(json.asInt()));
    case TypeKind::REAL:
      return Variant(static_cast<float>(json.asDouble()));
    case TypeKind::DOUBLE:
      return Variant(json.asDouble());
    case TypeKind::VARCHAR:
      return Variant(json.asString());
    case TypeKind::ARRAY: {
      std::vector<Variant> elements;
      for (const auto& element : json) {
        elements.emplace_back(toVariant(element, type->childAt(0)));
      }
      return Variant::array(std::move(elements));
    }
    case TypeKind::MAP: {
      std::map<Variant, Variant> entries;
      for (const auto& entry : json) {
        entries.emplace(
            toVariant(entry[0], type->childAt(0)),
            toVariant(entry[1], type->childAt(1)));
      }
      return Variant::map(std::move(entries));
    }
    case TypeKind::ROW: {
      std::vector<Variant> children;
      for (size_t i = 0; i < type->size(); ++i) {
        children.emplace_back(toVariant(json[i], type->childAt(i)));
      }
      return Variant::row(std::move(children));
    }
    default:
      VELOX_NYI("Unsupported type in a golden table: {}", type->toString());
  }
}

// One table that Paimon Java wrote, the splits it planned and the rows it read.
struct GoldenTable {
  RowTypePtr rowType;
  std::shared_ptr<PaimonTableHandle> tableHandle;
  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  exec::test::MaterializedRowMultiset expectedRows;
};

GoldenTable readGoldenTable(const std::string& name) {
  const auto directory = test::getDataFilePath(
      "velox/connectors/paimon/tests", "examples/golden/" + name);

  GoldenTable golden;
  const auto table = readJson(directory + "/table.json");
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  std::vector<int32_t> fieldIds;
  type::fbhive::HiveTypeParser typeParser;
  for (const auto& column : table["columns"]) {
    names.emplace_back(column["name"].asString());
    types.emplace_back(typeParser.parse(column["type"].asString()));
    fieldIds.emplace_back(column["id"].asInt());
  }
  golden.rowType = ROW(std::move(names), std::move(types));
  std::vector<std::string> primaryKeys;
  for (const auto& key : table["primaryKeys"]) {
    primaryKeys.emplace_back(key.asString());
  }
  std::vector<std::string> partitionKeys;
  for (const auto& key : table["partitionKeys"]) {
    partitionKeys.emplace_back(key.asString());
  }
  std::unordered_map<std::string, std::string> options;
  for (const auto& [key, value] : table["options"].items()) {
    options.emplace(key.asString(), value.asString());
  }
  const auto tableType = primaryKeys.empty() ? PaimonTableType::kAppendOnly
                                             : PaimonTableType::kPrimaryKey;
  golden.tableHandle = std::make_shared<PaimonTableHandle>(
      kPaimonConnectorId,
      name,
      table["schemaId"].asInt(),
      golden.rowType,
      std::move(fieldIds),
      tableType,
      std::move(partitionKeys),
      std::move(options),
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      std::vector<connector::FileColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"",
      std::move(primaryKeys));

  for (auto split : readJson(directory + "/splits.json")) {
    split["connectorId"] = kPaimonConnectorId;
    for (auto& file : split["dataFiles"]) {
      file["filePath"] = directory + "/" + file["filePath"].asString();
      if (auto* deletionFile = file.get_ptr("deletionFile");
          deletionFile != nullptr) {
        (*deletionFile)["path"] =
            directory + "/" + (*deletionFile)["path"].asString();
      }
    }
    golden.splits.emplace_back(PaimonConnectorSplit::create(split));
  }

  for (const auto& row : readJson(directory + "/expected.json")) {
    golden.expectedRows.emplace(toVariant(row, golden.rowType).row());
  }
  return golden;
}

class PaimonGoldenTableTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    nimble::registerSelectiveNimbleReaderFactory();
    auto connector = PaimonConnectorFactory().newConnector(
        kPaimonConnectorId,
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{}));
    ConnectorRegistry::global().insert(connector->connectorId(), connector);
  }

  void TearDown() override {
    ConnectorRegistry::global().erase(kPaimonConnectorId);
    nimble::unregisterSelectiveNimbleReaderFactory();
    HiveConnectorTestBase::TearDown();
  }

  // Reads all columns of 'golden' with Velox and compares the rows with what
  // Paimon Java read.
  void assertGoldenRows(const GoldenTable& golden) {
    connector::ColumnHandleMap assignments;
    for (column_index_t i = 0; i < golden.rowType->size(); ++i) {
      const auto& name = golden.rowType->nameOf(i);
      assignments[name] = std::make_shared<PaimonColumnHandle>(
          name,
          golden.tableHandle->fieldIds()[i],
          connector::FileColumnHandle::ColumnType::kRegular,
          golden.rowType->childAt(i));
    }
    const auto plan = exec::test::PlanBuilder()
                          .startTableScan()
                          .connectorId(kPaimonConnectorId)
                          .outputType(golden.rowType)
                          .tableHandle(golden.tableHandle)
                          .assignments(assignments)
                          .endTableScan()
                          .planNode();
    const auto result = exec::test::AssertQueryBuilder(plan)
                            .splits(golden.splits)
                            .copyResults(pool());
    EXPECT_TRUE(
        exec::test::assertEqualResults(
            golden.expectedRows, golden.rowType, {result}));
  }
};

TEST_F(PaimonGoldenTableTest, javaTables) {
  // Each case names the layout of the splits Paimon Java planned.
  const std::vector<std::string> cases{
      // A raw split of two append files.
      "append_only",
      // A merge split of three overlapping level-0 files with updates and
      // deletes.
      "deduplicate",
      // A merge split of a level-0 file over a fully compacted file.
      "deduplicate_compacted",
      // A raw split of a compacted first-row table.
      "first_row",
      // A merge split ordered by a sequence field, with a null and a tie.
      "sequence_field",
      // A raw split whose compacted file has a deletion vector.
      "deletion_vectors",
      // A merge split of an aggregation table with retracts and a key that
      // has only an UPDATE_BEFORE row.
      "aggregation",
      // A merge split of an aggregation table whose DELETE rows replace the
      // merged row.
      "aggregation_remove_record_on_delete",
      // A merge split of an aggregation table whose compaction looks up keys.
      "aggregation_lookup",
  };
  for (const auto& name : cases) {
    SCOPED_TRACE(name);
    assertGoldenRows(readGoldenTable(name));
  }
}

} // namespace
} // namespace facebook::velox::connector::paimon
