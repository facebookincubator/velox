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
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include "velox/common/Casts.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/common/ExtractionUtils.h"
#include "velox/connectors/common/FileDataSource.h"

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprToSubfieldFilter.h"

namespace facebook::velox::connector::hive {
namespace {

using namespace facebook::velox::common;
using namespace facebook::velox::exec::test;

class HiveConnectorTest : public exec::test::HiveConnectorTestBase {
 protected:
  // Records the options of every file opened for read through the faulty file
  // system in 'fileOpens_'. HiveConnectorTestBase::TearDown removes the hook.
  void recordFileOpens() {
    tests::utils::faultyFileSystem()->setFilesystemInjectionHook(
        [this](tests::utils::FaultFileSystemOperation* op) {
          if (op->type ==
              tests::utils::FaultFileSystemOperation::Type::kOpenFileForRead) {
            fileOpens_.push_back(
                static_cast<
                    tests::utils::FaultFileSystemOpenFileForReadOperation*>(op)
                    ->options);
          }
        });
  }

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();

  // Options of the files opened since recordFileOpens(), in open order.
  std::vector<filesystems::FileOptions> fileOpens_;
};

void validateNullConstant(const ScanSpec& spec, const Type& type) {
  ASSERT_TRUE(spec.isConstant());
  auto constant = spec.constantValue();
  ASSERT_TRUE(constant->isConstantEncoding());
  ASSERT_EQ(*constant->type(), type);
  ASSERT_TRUE(constant->isNullAt(0));
}

std::vector<Subfield> makeSubfields(const std::vector<std::string>& paths) {
  std::vector<Subfield> subfields;
  for (auto& path : paths) {
    subfields.emplace_back(path);
  }
  return subfields;
}

folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
groupSubfields(const std::vector<Subfield>& subfields) {
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>> grouped;
  for (auto& subfield : subfields) {
    auto& name =
        static_cast<const common::Subfield::NestedField&>(*subfield.path()[0])
            .name();
    grouped[name].push_back(&subfield);
  }
  return grouped;
}

bool mapKeyIsNotNull(const ScanSpec& mapSpec) {
  return dynamic_cast<const IsNotNull*>(
      mapSpec.childByName(ScanSpec::kMapKeysFieldName)->filter());
}

TEST_F(HiveConnectorTest, hiveConfig) {
  ASSERT_EQ(
      HiveConfig::toName(HiveConfig::InsertExistingPartitionsBehavior::kError),
      "ERROR");
  ASSERT_EQ(
      HiveConfig::toName(
          HiveConfig::InsertExistingPartitionsBehavior::kOverwrite),
      "OVERWRITE");
  ASSERT_EQ(
      HiveConfig::toInsertExistingPartitionsBehavior("OVERWRITE"),
      HiveConfig::InsertExistingPartitionsBehavior::kOverwrite);
  VELOX_ASSERT_THROW(
      HiveConfig::toName(
          static_cast<HiveConfig::InsertExistingPartitionsBehavior>(100)),
      "Invalid enum value: 100");
}

TEST_F(HiveConnectorTest, makeScanSpecRequiredSubfieldsMultilevel) {
  auto columnType = ROW(
      {{"c0c0", BIGINT()},
       {"c0c1",
        ARRAY(MAP(
            VARCHAR(), ROW({{"c0c1c0", BIGINT()}, {"c0c1c1", BIGINT()}})))}});
  auto rowType = ROW({{"c0", columnType}});
  auto subfields = makeSubfields({"c0.c0c1[3][\"foo\"].c0c1c0"});
  for (bool statsBasedFilterReorderDisabled : {false, true}) {
    SCOPED_TRACE(
        fmt::format(
            "statsBasedFilterReorderDisabled {}",
            statsBasedFilterReorderDisabled));

    auto scanSpec = connector::FileDataSource::makeScanSpec(
        rowType,
        groupSubfields(subfields),
        /*subfieldFilters=*/{},
        /*dataColumns=*/nullptr,
        /*partitionKeys=*/{},
        /*infoColumns=*/{},
        /*specialColumns=*/{},
        statsBasedFilterReorderDisabled,
        pool_.get());
    ASSERT_EQ(
        scanSpec->statsBasedFilterReorderDisabled(),
        statsBasedFilterReorderDisabled);

    auto* c0c0 = scanSpec->childByName("c0")->childByName("c0c0");
    validateNullConstant(*c0c0, *BIGINT());
    auto* c0c1 = scanSpec->childByName("c0")->childByName("c0c1");
    ASSERT_EQ(c0c1->maxArrayElementsCount(), 3);
    auto* elements = c0c1->childByName(ScanSpec::kArrayElementsFieldName);
    auto* keysFilter =
        elements->childByName(ScanSpec::kMapKeysFieldName)->filter();
    ASSERT_TRUE(keysFilter);
    ASSERT_TRUE(applyFilter(*keysFilter, "foo"_sv));
    ASSERT_FALSE(applyFilter(*keysFilter, "bar"_sv));
    ASSERT_FALSE(keysFilter->testNull());
    auto* values = elements->childByName(ScanSpec::kMapValuesFieldName);
    auto* c0c1c0 = values->childByName("c0c1c0");
    ASSERT_FALSE(c0c1c0->isConstant());
    ASSERT_FALSE(c0c1c0->filter());
    validateNullConstant(*values->childByName("c0c1c1"), *BIGINT());
  }
}

TEST_F(HiveConnectorTest, makeScanSpecRequiredSubfieldsMergeFields) {
  auto columnType = ROW(
      {{"c0c0",
        ROW(
            {{"c0c0c0", BIGINT()},
             {"c0c0c1", BIGINT()},
             {"c0c0c2", BIGINT()}})},
       {"c0c1", ROW({{"c0c1c0", BIGINT()}, {"c0c1c1", BIGINT()}})}});
  auto rowType = ROW({{"c0", columnType}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields(
          {"c0.c0c0.c0c0c0", "c0.c0c0.c0c0c2", "c0.c0c1", "c0.c0c1.c0c1c0"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  auto* c0c0 = scanSpec->childByName("c0")->childByName("c0c0");
  ASSERT_FALSE(c0c0->childByName("c0c0c0")->isConstant());
  ASSERT_FALSE(c0c0->childByName("c0c0c2")->isConstant());
  validateNullConstant(*c0c0->childByName("c0c0c1"), *BIGINT());
  auto* c0c1 = scanSpec->childByName("c0")->childByName("c0c1");
  ASSERT_FALSE(c0c1->isConstant());
  ASSERT_FALSE(c0c1->hasFilter());
  ASSERT_FALSE(c0c1->childByName("c0c1c0")->isConstant());
  ASSERT_FALSE(c0c1->childByName("c0c1c1")->isConstant());
}

TEST_F(HiveConnectorTest, makeScanSpecRequiredSubfieldsMergeArray) {
  auto columnType =
      ARRAY(ROW({{"c0c0", BIGINT()}, {"c0c1", BIGINT()}, {"c0c2", BIGINT()}}));
  auto rowType = ROW({{"c0", columnType}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields({"c0[1].c0c0", "c0[2].c0c2"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  auto* c0 = scanSpec->childByName("c0");
  ASSERT_EQ(c0->maxArrayElementsCount(), 2);
  ASSERT_TRUE(c0->flatMapFeatureSelection().empty());
  auto* elements = c0->childByName(ScanSpec::kArrayElementsFieldName);
  ASSERT_FALSE(elements->childByName("c0c0")->isConstant());
  ASSERT_FALSE(elements->childByName("c0c2")->isConstant());
  validateNullConstant(*elements->childByName("c0c1"), *BIGINT());
}

TEST_F(HiveConnectorTest, makeScanSpecRequiredSubfieldsMergeArrayNegative) {
  auto columnType =
      ARRAY(ROW({{"c0c0", BIGINT()}, {"c0c1", BIGINT()}, {"c0c2", BIGINT()}}));
  auto rowType = ROW({{"c0", columnType}});
  auto subfields = makeSubfields({"c0[1].c0c0", "c0[-1].c0c2"});
  auto groupedSubfields = groupSubfields(subfields);
  VELOX_ASSERT_USER_THROW(
      connector::FileDataSource::makeScanSpec(
          rowType,
          groupedSubfields,
          /*subfieldFilters=*/{},
          /*dataColumns=*/nullptr,
          /*partitionKeys=*/{},
          /*infoColumns=*/{},
          /*specialColumns=*/{},
          /*disableStatsBasedFilterReorder=*/false,
          pool_.get()),
      "Non-positive array subscript cannot be push down");
}

TEST_F(HiveConnectorTest, makeScanSpecRequiredSubfieldsMergeMap) {
  auto columnType =
      MAP(BIGINT(),
          ROW({{"c0c0", BIGINT()}, {"c0c1", BIGINT()}, {"c0c2", BIGINT()}}));
  auto rowType = ROW({{"c0", columnType}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields({"c0[10].c0c0", "c0[20].c0c2"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  auto* c0 = scanSpec->childByName("c0");
  ASSERT_EQ(
      c0->flatMapFeatureSelection(), std::vector<std::string>({"10", "20"}));
  auto* keysFilter = c0->childByName(ScanSpec::kMapKeysFieldName)->filter();
  ASSERT_TRUE(keysFilter);
  ASSERT_TRUE(applyFilter(*keysFilter, 10));
  ASSERT_TRUE(applyFilter(*keysFilter, 20));
  ASSERT_FALSE(applyFilter(*keysFilter, 15));
  auto* values = c0->childByName(ScanSpec::kMapValuesFieldName);
  auto c0c0 = values->childByName("c0c0");
  ASSERT_FALSE(c0c0->isConstant());
  ASSERT_TRUE(c0c0->projectOut());
  auto c0c1 = values->childByName("c0c1");
  validateNullConstant(*c0c1, *BIGINT());
  ASSERT_FALSE(values->childByName("c0c2")->isConstant());
}

TEST_F(
    HiveConnectorTest,
    makeScanSpecRequiredSubfieldsFlatMapFeatureSelectionStringKeys) {
  auto rowType = ROW({{"c0", MAP(VARCHAR(), BIGINT())}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields({"c0[\"foo\"]"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  auto* c0 = scanSpec->childByName("c0");
  ASSERT_EQ(c0->flatMapFeatureSelection(), std::vector<std::string>({"foo"}));
  auto cs = dwio::common::ColumnSelector::fromScanSpec(*scanSpec, rowType);
  ASSERT_EQ(cs->findNode("c0")->getNode().expression, "[\"foo\"]");
}

TEST_F(HiveConnectorTest, makeScanSpecRequiredSubfieldsAllSubscripts) {
  auto columnType =
      MAP(BIGINT(), ARRAY(ROW({{"c0c0", BIGINT()}, {"c0c1", BIGINT()}})));
  auto rowType = ROW({{"c0", columnType}});
  for (auto* path : {"c0", "c0[*]", "c0[*][*]"}) {
    SCOPED_TRACE(path);
    auto scanSpec = connector::FileDataSource::makeScanSpec(
        rowType,
        groupSubfields(makeSubfields({path})),
        /*subfieldFilters=*/{},
        /*dataColumns=*/nullptr,
        /*partitionKeys=*/{},
        /*infoColumns=*/{},
        /*specialColumns=*/{},
        /*disableStatsBasedFilterReorder=*/false,
        pool_.get());
    auto* c0 = scanSpec->childByName("c0");
    ASSERT_TRUE(c0->flatMapFeatureSelection().empty());
    ASSERT_TRUE(mapKeyIsNotNull(*c0));
    auto* values = c0->childByName(ScanSpec::kMapValuesFieldName);
    ASSERT_EQ(
        values->maxArrayElementsCount(),
        std::numeric_limits<vector_size_t>::max());
    auto* elements = values->childByName(ScanSpec::kArrayElementsFieldName);
    ASSERT_FALSE(elements->hasFilter());
    ASSERT_FALSE(elements->childByName("c0c0")->isConstant());
    ASSERT_FALSE(elements->childByName("c0c1")->isConstant());
  }
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields({"c0[*][*].c0c0"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  auto* c0 = scanSpec->childByName("c0");
  ASSERT_TRUE(mapKeyIsNotNull(*c0));
  auto* values = c0->childByName(ScanSpec::kMapValuesFieldName);
  ASSERT_EQ(
      values->maxArrayElementsCount(),
      std::numeric_limits<vector_size_t>::max());
  auto* elements = values->childByName(ScanSpec::kArrayElementsFieldName);
  ASSERT_FALSE(elements->hasFilter());
  ASSERT_FALSE(elements->childByName("c0c0")->isConstant());
  validateNullConstant(*elements->childByName("c0c1"), *BIGINT());
}

TEST_F(HiveConnectorTest, makeScanSpecRequiredSubfieldsDoubleMapKey) {
  auto rowType =
      ROW({{"c0", MAP(REAL(), BIGINT())}, {"c1", MAP(DOUBLE(), BIGINT())}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields({"c0[0]", "c1[-1]"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  auto* keysFilter = scanSpec->childByName("c0")
                         ->childByName(ScanSpec::kMapKeysFieldName)
                         ->filter();
  ASSERT_TRUE(keysFilter);
  ASSERT_TRUE(applyFilter(*keysFilter, 0.0f));
  ASSERT_TRUE(applyFilter(*keysFilter, 0.99f));
  ASSERT_FALSE(applyFilter(*keysFilter, 1.0f));
  ASSERT_TRUE(applyFilter(*keysFilter, -0.99f));
  ASSERT_FALSE(applyFilter(*keysFilter, -1.0f));
  keysFilter = scanSpec->childByName("c1")
                   ->childByName(ScanSpec::kMapKeysFieldName)
                   ->filter();
  ASSERT_TRUE(keysFilter);
  ASSERT_FALSE(applyFilter(*keysFilter, 0.0));
  ASSERT_TRUE(applyFilter(*keysFilter, -1.0));
  ASSERT_TRUE(applyFilter(*keysFilter, -1.99));
  ASSERT_FALSE(applyFilter(*keysFilter, -2.0));

  // Integer min and max means infinities.
  scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields(
          {"c0[-9223372036854775808]", "c1[9223372036854775807]"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  keysFilter = scanSpec->childByName("c0")
                   ->childByName(ScanSpec::kMapKeysFieldName)
                   ->filter();
  ASSERT_TRUE(applyFilter(*keysFilter, -1e30f));
  ASSERT_FALSE(applyFilter(*keysFilter, -9223370000000000000.0f));
  keysFilter = scanSpec->childByName("c1")
                   ->childByName(ScanSpec::kMapKeysFieldName)
                   ->filter();
  ASSERT_TRUE(applyFilter(*keysFilter, 1e100));
  ASSERT_FALSE(applyFilter(*keysFilter, 9223372036854700000.0));
  scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields(
          {"c0[9223372036854775807]", "c0[-9223372036854775808]"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  keysFilter = scanSpec->childByName("c0")
                   ->childByName(ScanSpec::kMapKeysFieldName)
                   ->filter();
  ASSERT_TRUE(applyFilter(*keysFilter, -1e30f));
  ASSERT_FALSE(applyFilter(*keysFilter, 0.0f));
  ASSERT_TRUE(applyFilter(*keysFilter, 1e30f));

  // Unrepresentable values.
  scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields({"c0[-100000000]", "c0[100000000]"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  keysFilter = scanSpec->childByName("c0")
                   ->childByName(ScanSpec::kMapKeysFieldName)
                   ->filter();
  ASSERT_TRUE(applyFilter(*keysFilter, -100000000.0f));
  ASSERT_FALSE(applyFilter(*keysFilter, -100000008.0f));
  ASSERT_FALSE(applyFilter(*keysFilter, 0.0f));
  ASSERT_TRUE(applyFilter(*keysFilter, 100000000.0f));
  ASSERT_FALSE(applyFilter(*keysFilter, 100000008.0f));
}

TEST_F(HiveConnectorTest, makeScanSpecRequiredSubfieldsOnlyInFilters) {
  auto c0Type = ROW({
      {"c0c0", BIGINT()},
      {"c0c1", VARCHAR()},
      {"c0c2", ROW({{"c0c2c0", BIGINT()}})},
      {"c0c3", ROW({{"c0c3c0", BIGINT()}})},
      {"c0c4", BIGINT()},
  });
  auto c1c0Type = ROW({{"c1c0c0", BIGINT()}, {"c1c0c1", BIGINT()}});
  auto c1c1Type = ROW({{"c1c1c0", BIGINT()}, {"c1c1c1", BIGINT()}});
  auto c1Type = ROW({
      {"c1c0", c1c0Type},
      {"c1c1", c1c1Type},
  });
  auto readerOutputType = ROW({{"c0", c0Type}});

  SubfieldFilters filters;
  filters.emplace(Subfield("c0.c0c0"), exec::equal(42));
  filters.emplace(Subfield("c0.c0c2"), exec::isNotNull());
  filters.emplace(Subfield("c0.c0c3"), exec::isNotNull());
  filters.emplace(Subfield("c1.c1c0.c1c0c0"), exec::equal(43));

  auto scanSpec = connector::FileDataSource::makeScanSpec(
      readerOutputType,
      groupSubfields(makeSubfields({"c0.c0c1", "c0.c0c3"})),
      filters,
      /*dataColumns=*/ROW({{"c0", c0Type}, {"c1", c1Type}}),
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());

  auto c0 = scanSpec->childByName("c0");
  ASSERT_FALSE(c0->isConstant());
  ASSERT_TRUE(c0->projectOut());
  ASSERT_FALSE(c0->filter());
  ASSERT_TRUE(c0->hasFilter());

  // Filter only.
  auto* c0c0 = c0->childByName("c0c0");
  ASSERT_FALSE(c0c0->isConstant());
  ASSERT_TRUE(c0c0->projectOut());
  ASSERT_TRUE(c0c0->filter());
  ASSERT_TRUE(c0c0->hasFilter());
  // Project output.
  auto* c0c1 = c0->childByName("c0c1");
  ASSERT_FALSE(c0c1->isConstant());
  ASSERT_TRUE(c0c1->projectOut());
  ASSERT_FALSE(c0c1->filter());
  ASSERT_FALSE(c0c1->hasFilter());
  // Filter on struct, no children.
  auto* c0c2 = c0->childByName("c0c2");
  ASSERT_FALSE(c0c2->isConstant());
  ASSERT_TRUE(c0c2->projectOut());
  ASSERT_TRUE(c0c2->filter());
  ASSERT_TRUE(c0c2->hasFilter());

  auto c0c2c0 = c0c2->childByName("c0c2c0");
  validateNullConstant(*c0c2c0, *BIGINT());

  // Filtered and project out.
  auto* c0c3 = c0->childByName("c0c3");
  ASSERT_FALSE(c0c3->isConstant());
  ASSERT_TRUE(c0c3->projectOut());
  ASSERT_TRUE(c0c3->filter());
  ASSERT_TRUE(c0c3->hasFilter());

  auto c0c3c0 = c0c3->childByName("c0c3c0");
  ASSERT_FALSE(c0c3c0->isConstant());

  auto c0c4 = c0->childByName("c0c4");
  ASSERT_TRUE(c0c4->projectOut());

  // Filter only, column not projected out.
  auto* c1 = scanSpec->childByName("c1");
  ASSERT_FALSE(c1->isConstant());
  ASSERT_FALSE(c1->projectOut());
  ASSERT_FALSE(c1->filter());
  ASSERT_TRUE(c1->hasFilter());

  auto* c1c0 = c1->childByName("c1c0");
  ASSERT_FALSE(c1c0->filter());
  ASSERT_TRUE(c1c0->hasFilter());

  auto c1c0c0 = c1c0->childByName("c1c0c0");
  ASSERT_TRUE(c1c0c0);
  ASSERT_FALSE(c1c0c0->isConstant());
  ASSERT_TRUE(c1c0c0->filter());
  ASSERT_TRUE(c1c0c0->hasFilter());

  auto c1c0c1 = c1c0->childByName("c1c0c1");
  ASSERT_TRUE(c1c0c1);
  validateNullConstant(*c1c0c1, *BIGINT());

  auto c1c1 = c1->childByName("c1c1");
  validateNullConstant(*c1c1, *c1c1Type);
}

TEST_F(HiveConnectorTest, makeScanSpecDuplicateSubfields) {
  auto c0Type = MAP(BIGINT(), MAP(BIGINT(), BIGINT()));
  auto c1Type = MAP(VARCHAR(), MAP(BIGINT(), BIGINT()));
  auto rowType = ROW({{"c0", c0Type}, {"c1", c1Type}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields(
          {"c0[10][1]", "c0[10][2]", "c1[\"foo\"][1]", "c1[\"foo\"][2]"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  auto* c0 = scanSpec->childByName("c0");
  ASSERT_EQ(c0->children().size(), 2);
  auto* c1 = scanSpec->childByName("c1");
  ASSERT_EQ(c1->children().size(), 2);
}

// For TEXTFILE, partition key is not included in data columns.
TEST_F(HiveConnectorTest, makeScanSpecFilterPartitionKey) {
  auto rowType = ROW({{"c0", BIGINT()}});
  SubfieldFilters filters;
  filters.emplace(Subfield("ds"), exec::equal("2023-10-13"));
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      /*outputSubfields=*/{},
      filters,
      /*dataColumns=*/rowType,
      /*partitionKeys=*/{{"ds", nullptr}},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  ASSERT_TRUE(scanSpec->childByName("c0")->projectOut());
  ASSERT_FALSE(scanSpec->childByName("ds")->projectOut());
}

TEST_F(HiveConnectorTest, makeScanSpecPrunedMapNonNullMapKey) {
  auto rowType =
      ROW({"c0"},
          {ROW(
              {{"c0c0", MAP(BIGINT(), MAP(BIGINT(), BIGINT()))},
               {"c0c1", BIGINT()}})});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields({"c0.c0c1"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  auto* c0 = scanSpec->childByName("c0");
  ASSERT_EQ(c0->children().size(), 2);
  validateNullConstant(
      *c0->childByName("c0c0"), *MAP(BIGINT(), MAP(BIGINT(), BIGINT())));
  ASSERT_FALSE(c0->childByName("c0c1")->isConstant());

  scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      groupSubfields(makeSubfields({"c0.c0c0"})),
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());
  c0 = scanSpec->childByName("c0");
  ASSERT_EQ(c0->children().size(), 2);
  auto c0c0 = c0->childByName("c0c0");
  ASSERT_TRUE(mapKeyIsNotNull(*c0c0));
}

TEST_F(HiveConnectorTest, fileReadOpsTableIdentityPropagation) {
  recordFileOpens();

  // Write test data to a local temp file.
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto vector = makeNamedRowVector("c0", makeFlatVector<int64_t>({1, 2, 3}));
  auto filePath = writeToFile(vector);

  // Create a table handle with dbName and tableName set.
  auto tableHandle = std::make_shared<HiveTableHandle>(
      kHiveConnectorId,
      "test_table",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<HiveColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"test_db");

  // Build the split on the faulty file system so that its open hook sees the
  // fileReadOps populated by FileSplitReader.
  auto split =
      exec::test::HiveConnectorSplitBuilder(
          tests::utils::FaultyFileSystem::scheme() + filePath->getPath())
          .fileFormat(dwio::common::FileFormat::DWRF)
          .build();

  // Build and run a table scan. This exercises the full pipeline:
  // FileSplitReader::createReader() -> FileHandleGenerator ->
  // FaultyFileSystem.
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).split(split).copyResults(pool_.get());
  ASSERT_EQ(result->size(), 3);

  // Verify that FileSplitReader propagated dbName and tableName into
  // fileReadOps.
  const folly::F14FastMap<std::string, std::string> expected{
      {std::string(kDbNameKey), "test_db"},
      {std::string(kTableNameKey), "test_table"},
  };
  EXPECT_THAT(
      fileOpens_,
      testing::ElementsAre(
          testing::Field(&filesystems::FileOptions::fileReadOps, expected)));
}

TEST_F(HiveConnectorTest, setFileReadOpsForwardsToFileSystem) {
  recordFileOpens();

  const auto rowType = ROW({"c0"}, {BIGINT()});
  const auto tableHandle = makeTableHandle(
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      "test_table",
      /*dataColumns=*/nullptr,
      /*indexColumns=*/{},
      /*storageParameters=*/{},
      "test_db");
  const config::ConfigBase sessionProperties({});
  auto connector = ConnectorRegistry::tryGet(kHiveConnectorId);

  // Returns the options of each file a new data source opens for its split,
  // calling setFileReadOps() first when 'fileReadOps' is set. Each call writes
  // its own file because file handles are cached by path and a cached handle is
  // not reopened.
  auto openFileOptions =
      [&](std::optional<folly::F14FastMap<std::string, std::string>>
              fileReadOps) {
        auto filePath = TempFilePath::create();
        writeToFile(
            filePath->getPath(),
            makeNamedRowVector("c0", makeFlatVector<int64_t>({1, 2, 3})));
        auto connectorQueryCtx = ConnectorQueryCtx::Builder()
                                     .operatorPool(pool_.get())
                                     .connectorPool(pool_.get())
                                     .sessionProperties(&sessionProperties)
                                     .queryId("query.HiveConnectorTest")
                                     .taskId("task.HiveConnectorTest")
                                     .planNodeId("planNodeId.HiveConnectorTest")
                                     .build();
        auto dataSource = connector->createDataSource(
            rowType,
            tableHandle,
            allRegularColumns(rowType),
            connectorQueryCtx.get());
        if (fileReadOps.has_value()) {
          checkedPointerCast<FileDataSource>(dataSource.get())
              ->setFileReadOps(std::move(*fileReadOps));
        }
        fileOpens_.clear();
        dataSource->addSplit(
            exec::test::HiveConnectorSplitBuilder(
                tests::utils::FaultyFileSystem::scheme() + filePath->getPath())
                .fileFormat(dwio::common::FileFormat::DWRF)
                .build());
        return fileOpens_;
      };

  const folly::F14FastMap<std::string, std::string> tableIdentity{
      {std::string(kDbNameKey), "test_db"},
      {std::string(kTableNameKey), "test_table"},
  };
  EXPECT_THAT(
      openFileOptions(std::nullopt),
      testing::ElementsAre(
          testing::Field(
              &filesystems::FileOptions::fileReadOps, tableIdentity)));

  auto expected = tableIdentity;
  expected.emplace("test.key", "test.value");
  EXPECT_THAT(
      openFileOptions(
          folly::F14FastMap<std::string, std::string>{
              {"test.key", "test.value"},
          }),
      testing::ElementsAre(
          testing::Field(&filesystems::FileOptions::fileReadOps, expected)));
}

TEST_F(HiveConnectorTest, splitFileReadOpsAddKeys) {
  const auto rowType = ROW({"c0"}, {BIGINT()});
  const auto filePath = TempFilePath::create();
  writeToFile(
      filePath->getPath(),
      makeNamedRowVector("c0", makeFlatVector<int64_t>({1, 2, 3})));
  const auto tableHandle = makeTableHandle(
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      "test_table",
      /*dataColumns=*/nullptr,
      /*indexColumns=*/{},
      /*storageParameters=*/{},
      "test_db");
  const config::ConfigBase sessionProperties({});
  const auto connectorQueryCtx = ConnectorQueryCtx::Builder()
                                     .operatorPool(pool_.get())
                                     .connectorPool(pool_.get())
                                     .sessionProperties(&sessionProperties)
                                     .queryId("query.HiveConnectorTest")
                                     .taskId("task.HiveConnectorTest")
                                     .planNodeId("planNodeId.HiveConnectorTest")
                                     .build();
  auto dataSource = ConnectorRegistry::tryGet(kHiveConnectorId)
                        ->createDataSource(
                            rowType,
                            tableHandle,
                            allRegularColumns(rowType),
                            connectorQueryCtx.get());
  checkedPointerCast<FileDataSource>(dataSource.get())
      ->setFileReadOps(
          folly::F14FastMap<std::string, std::string>{
              {"readerKey", "readerValue"},
              {"sharedKey", "readerValue"},
          });

  FileProperties fileProperties;
  fileProperties.fileReadOps = {
      {"splitKey", "splitValue"},
      {"sharedKey", "splitValue"},
      {std::string(kTableNameKey), "split_table"},
  };
  recordFileOpens();
  dataSource->addSplit(
      exec::test::HiveConnectorSplitBuilder(
          tests::utils::FaultyFileSystem::scheme() + filePath->getPath())
          .fileFormat(dwio::common::FileFormat::DWRF)
          .fileProperties(fileProperties)
          .build());

  const folly::F14FastMap<std::string, std::string> expected{
      {"splitKey", "splitValue"},
      {"readerKey", "readerValue"},
      {"sharedKey", "readerValue"},
      {std::string(kDbNameKey), "test_db"},
      {std::string(kTableNameKey), "test_table"},
  };
  EXPECT_THAT(
      fileOpens_,
      testing::ElementsAre(
          testing::Field(&filesystems::FileOptions::fileReadOps, expected)));
}

/// Verifies that getRuntimeStats() merge logic allows IoStats (ReadFile-layer)
/// to override the storageReadBytes value from IoStatistics (DWIO-level).
TEST_F(HiveConnectorTest, ioStatsOverridesStorageReadBytes) {
  // Step 1: Simulate DWIO-level IoStatistics with an inaccurate estimate.
  auto ioStatistics = std::make_shared<io::IoStatistics>();
  ioStatistics->read().increment(200); // DWIO estimate (undercounts gaps)

  // Step 2: Simulate ReadFile-layer IoStats with ground-truth values.
  IoStats ioStats;
  ioStats.addCounter(
      std::string(connector::FileDataSource::kStorageReadBytes),
      RuntimeCounter(50, RuntimeCounter::Unit::kBytes));
  ioStats.addCounter(
      "extentCacheHitBytes", RuntimeCounter(250, RuntimeCounter::Unit::kBytes));

  // Replicate the merge logic from FileDataSource::getRuntimeStats().
  std::unordered_map<std::string, RuntimeMetric> res;

  // IoStatistics inserts first (Step 2 in getRuntimeStats).
  res.insert(
      {std::string(connector::FileDataSource::kStorageReadBytes),
       RuntimeMetric(
           ioStatistics->read().sum(),
           ioStatistics->read().count(),
           ioStatistics->read().min(),
           ioStatistics->read().max(),
           RuntimeCounter::Unit::kBytes)});

  // IoStats merge (Step 3 in getRuntimeStats) -- override storageReadBytes.
  const auto ioStatsMap = ioStats.stats();
  for (const auto& [key, value] : ioStatsMap) {
    if (key == connector::FileDataSource::kStorageReadBytes) {
      res[std::string(key)] = value;
    } else {
      res.emplace(key, value);
    }
  }

  // Ground-truth storageReadBytes from IoStats should override DWIO estimate.
  ASSERT_EQ(res.at("storageReadBytes").sum, 50);
  ASSERT_EQ(res.at("storageReadBytes").unit, RuntimeCounter::Unit::kBytes);
  // extentCacheHitBytes should be added as a new diagnostic counter.
  ASSERT_EQ(res.at("extentCacheHitBytes").sum, 250);
}

/// Verifies that without IoStats override, storageReadBytes retains the
/// IoStatistics value.
TEST_F(HiveConnectorTest, storageReadBytesWithoutOverride) {
  auto ioStatistics = std::make_shared<io::IoStatistics>();
  ioStatistics->read().increment(200);

  // IoStats has unrelated counters only -- no storageReadBytes.
  IoStats ioStats;
  ioStats.addCounter(
      "wsInRegionReadBytes", RuntimeCounter(300, RuntimeCounter::Unit::kBytes));

  std::unordered_map<std::string, RuntimeMetric> res;
  res.insert(
      {std::string(connector::FileDataSource::kStorageReadBytes),
       RuntimeMetric(
           ioStatistics->read().sum(),
           ioStatistics->read().count(),
           ioStatistics->read().min(),
           ioStatistics->read().max(),
           RuntimeCounter::Unit::kBytes)});

  const auto ioStatsMap = ioStats.stats();
  for (const auto& [key, value] : ioStatsMap) {
    if (key == connector::FileDataSource::kStorageReadBytes) {
      res[std::string(key)] = value;
    } else {
      res.emplace(key, value);
    }
  }

  // storageReadBytes should retain the IoStatistics value.
  ASSERT_EQ(res.at("storageReadBytes").sum, 200);
  // Unrelated IoStats counters should still be added.
  ASSERT_EQ(res.at("wsInRegionReadBytes").sum, 300);
}

// --- ScanSpec extraction pushdown tests (file reader layer) ---

TEST_F(HiveConnectorTest, extractionScanSpecMapKeepsBothChildren) {
  // Map readers read keys and values together, so extraction from maps
  // does not prune map children.  The extraction is applied post-read.
  auto hiveType = MAP(VARCHAR(), BIGINT());
  auto rowType = ROW({{"col", hiveType}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      /*outputSubfields=*/{},
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());

  // MapKeys extraction -- map children should still be readable.
  std::vector<connector::NamedExtraction> extractions = {
      {"keys",
       {connector::ExtractionPathElement::simple(
           connector::ExtractionStep::kMapKeys)},
       ARRAY(VARCHAR())}};

  auto* colSpec = scanSpec->childByName("col");
  ASSERT_NE(colSpec, nullptr);
  connector::configureExtractionScanSpec(
      hiveType, extractions, *colSpec, pool_.get());

  auto* keysSpec = colSpec->childByName(ScanSpec::kMapKeysFieldName);
  ASSERT_NE(keysSpec, nullptr);
  ASSERT_FALSE(keysSpec->isConstant());
  auto* valuesSpec = colSpec->childByName(ScanSpec::kMapValuesFieldName);
  ASSERT_NE(valuesSpec, nullptr);
  ASSERT_FALSE(valuesSpec->isConstant());
}

TEST_F(HiveConnectorTest, extractionScanSpecSizeKeepsBothChildren) {
  // Size extraction on a map -- map children should still be readable
  // (extraction is post-read).
  auto hiveType = MAP(VARCHAR(), BIGINT());
  auto rowType = ROW({{"col", hiveType}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      /*outputSubfields=*/{},
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());

  std::vector<connector::NamedExtraction> extractions = {
      {"sz",
       {connector::ExtractionPathElement::simple(
           connector::ExtractionStep::kSize)},
       BIGINT()}};

  auto* colSpec = scanSpec->childByName("col");
  connector::configureExtractionScanSpec(
      hiveType, extractions, *colSpec, pool_.get());

  auto* keysSpec = colSpec->childByName(ScanSpec::kMapKeysFieldName);
  ASSERT_FALSE(keysSpec->isConstant());
  auto* valuesSpec = colSpec->childByName(ScanSpec::kMapValuesFieldName);
  ASSERT_FALSE(valuesSpec->isConstant());
}

TEST_F(HiveConnectorTest, extractionScanSpecStructFieldPruning) {
  // Extraction with StructField should prune unneeded fields.
  auto hiveType = ROW({{"x", INTEGER()}, {"y", DOUBLE()}, {"z", VARCHAR()}});
  auto rowType = ROW({{"col", hiveType}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      /*outputSubfields=*/{},
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());

  std::vector<connector::NamedExtraction> extractions = {
      {"col_x",
       {connector::ExtractionPathElement::structField("x")},
       INTEGER()}};

  auto* colSpec = scanSpec->childByName("col");
  connector::configureExtractionScanSpec(
      hiveType, extractions, *colSpec, pool_.get());

  // x should be readable, y and z should be constant null.
  ASSERT_FALSE(colSpec->childByName("x")->isConstant());
  ASSERT_TRUE(colSpec->childByName("y")->isConstant());
  ASSERT_TRUE(colSpec->childByName("z")->isConstant());
}

TEST_F(HiveConnectorTest, scanSpecTransformApplied) {
  // Verify that a transform set on ScanSpec is callable and produces
  // the correct output type.
  auto hiveType = MAP(VARCHAR(), BIGINT());
  auto scanSpec = std::make_shared<ScanSpec>("col");
  scanSpec->addFieldRecursively("col", *hiveType, 0);

  auto* colSpec = scanSpec->childByName("col");
  ASSERT_NE(colSpec, nullptr);
  ASSERT_FALSE(colSpec->hasTransform());

  // Set a MapKeys extraction transform.
  auto chain = std::vector<connector::ExtractionPathElementPtr>{
      connector::ExtractionPathElement::simple(
          connector::ExtractionStep::kMapKeys)};
  colSpec->setTransform(
      [chain](const VectorPtr& input, memory::MemoryPool* pool) -> VectorPtr {
        return connector::applyExtractionChain(input, chain, pool);
      },
      ARRAY(VARCHAR()));

  ASSERT_TRUE(colSpec->hasTransform());

  // Create a test MapVector and apply the transform.
  auto keys = makeFlatVector<StringView>({"a", "b", "c"});
  auto values = makeFlatVector<int64_t>({1, 2, 3});
  auto mapVector = makeMapVector({0, 2}, keys, values);

  auto result = colSpec->transform()(mapVector, pool_.get());
  ASSERT_TRUE(result->type()->isArray());
  ASSERT_EQ(result->size(), 2);
  auto* array = result->as<ArrayVector>();
  ASSERT_EQ(array->sizeAt(0), 2);
  ASSERT_EQ(array->sizeAt(1), 1);
}

TEST_F(HiveConnectorTest, extractionScanSpecMapKeyFilterString) {
  // kMapKeyFilter with string keys should set an IN filter on the keys
  // ScanSpec.  ExtractionType should remain kNone since kMapKeyFilter is
  // type-preserving.
  auto hiveType = MAP(VARCHAR(), BIGINT());
  auto rowType = ROW({{"col", hiveType}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      /*outputSubfields=*/{},
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());

  std::vector<connector::NamedExtraction> extractions = {
      {"filtered",
       {connector::ExtractionPathElement::mapKeyFilter(
           std::vector<std::string>{"a", "b"})},
       hiveType}};

  auto* colSpec = scanSpec->childByName("col");
  ASSERT_NE(colSpec, nullptr);
  connector::configureExtractionScanSpec(
      hiveType, extractions, *colSpec, pool_.get());

  // ExtractionType should remain kNone.
  ASSERT_EQ(colSpec->extractionType(), ScanSpec::ExtractionType::kNone);

  // Keys ScanSpec should have an IN filter set.
  auto* keysSpec = colSpec->childByName(ScanSpec::kMapKeysFieldName);
  ASSERT_NE(keysSpec, nullptr);
  ASSERT_NE(keysSpec->filter(), nullptr);
  // Filter should pass "a" and "b" but not "c".
  ASSERT_TRUE(keysSpec->filter()->testStringView(StringView("a")));
  ASSERT_TRUE(keysSpec->filter()->testStringView(StringView("b")));
  ASSERT_FALSE(keysSpec->filter()->testStringView(StringView("c")));
}

TEST_F(HiveConnectorTest, extractionScanSpecMapKeyFilterInt) {
  // kMapKeyFilter with integer keys should set an IN filter on the keys
  // ScanSpec.
  auto hiveType = MAP(BIGINT(), VARCHAR());
  auto rowType = ROW({{"col", hiveType}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      /*outputSubfields=*/{},
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());

  std::vector<connector::NamedExtraction> extractions = {
      {"filtered",
       {connector::ExtractionPathElement::mapKeyFilter(
           std::vector<int64_t>{10, 20, 30})},
       hiveType}};

  auto* colSpec = scanSpec->childByName("col");
  ASSERT_NE(colSpec, nullptr);
  connector::configureExtractionScanSpec(
      hiveType, extractions, *colSpec, pool_.get());

  // ExtractionType should remain kNone.
  ASSERT_EQ(colSpec->extractionType(), ScanSpec::ExtractionType::kNone);

  // Keys ScanSpec should have an IN filter set.
  auto* keysSpec = colSpec->childByName(ScanSpec::kMapKeysFieldName);
  ASSERT_NE(keysSpec, nullptr);
  ASSERT_NE(keysSpec->filter(), nullptr);
  // Filter should pass 10, 20, 30 but not 5.
  ASSERT_TRUE(keysSpec->filter()->testInt64(10));
  ASSERT_TRUE(keysSpec->filter()->testInt64(20));
  ASSERT_TRUE(keysSpec->filter()->testInt64(30));
  ASSERT_FALSE(keysSpec->filter()->testInt64(5));
}

TEST_F(HiveConnectorTest, extractionScanSpecMapKeyFilterThenMapKeys) {
  // kMapKeyFilter followed by kMapKeys should set a filter on keys and then
  // configure kKeys extraction on the remaining chain.
  auto hiveType = MAP(VARCHAR(), BIGINT());
  auto rowType = ROW({{"col", hiveType}});
  auto scanSpec = connector::FileDataSource::makeScanSpec(
      rowType,
      /*outputSubfields=*/{},
      /*subfieldFilters=*/{},
      /*dataColumns=*/nullptr,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_.get());

  std::vector<connector::NamedExtraction> extractions = {
      {"keys",
       {connector::ExtractionPathElement::mapKeyFilter(
            std::vector<std::string>{"x", "y"}),
        connector::ExtractionPathElement::simple(
            connector::ExtractionStep::kMapKeys)},
       ARRAY(VARCHAR())}};

  auto* colSpec = scanSpec->childByName("col");
  ASSERT_NE(colSpec, nullptr);
  connector::configureExtractionScanSpec(
      hiveType, extractions, *colSpec, pool_.get());

  // Filter should be set on keys.
  auto* keysSpec = colSpec->childByName(ScanSpec::kMapKeysFieldName);
  ASSERT_NE(keysSpec, nullptr);
  ASSERT_NE(keysSpec->filter(), nullptr);
  ASSERT_TRUE(keysSpec->filter()->testStringView(StringView("x")));
  ASSERT_FALSE(keysSpec->filter()->testStringView(StringView("z")));

  // After stripping kMapKeyFilter, remaining chain is [kMapKeys], so
  // kKeys extraction should be set.
  ASSERT_EQ(colSpec->extractionType(), ScanSpec::ExtractionType::kKeys);
}

} // namespace
} // namespace facebook::velox::connector::hive
