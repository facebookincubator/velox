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

#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"

#include <gtest/gtest.h>
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

TEST(IcebergColumnHandleTest, primitiveTypes) {
  IcebergNestedField nestedField{10, {}};

  auto testType = [&](const TypePtr& type,
                      HiveColumnHandle::ColumnType columnType) {
    auto columnHandle = std::make_shared<IcebergColumnHandle>(
        "C1", columnType, type, type, nestedField);
    EXPECT_EQ(columnHandle->dataType(), type);
    EXPECT_EQ(columnHandle->hiveType(), type);
    EXPECT_EQ(columnHandle->nestedField().id, 10);
    EXPECT_TRUE(columnHandle->nestedField().children.empty());
    EXPECT_EQ(columnHandle->columnType(), columnType);
  };

  testType(BOOLEAN(), HiveColumnHandle::ColumnType::kRegular);
  testType(TINYINT(), HiveColumnHandle::ColumnType::kRegular);
  testType(SMALLINT(), HiveColumnHandle::ColumnType::kRegular);
  testType(INTEGER(), HiveColumnHandle::ColumnType::kPartitionKey);
  testType(BIGINT(), HiveColumnHandle::ColumnType::kPartitionKey);
  testType(REAL(), HiveColumnHandle::ColumnType::kRegular);
  testType(DOUBLE(), HiveColumnHandle::ColumnType::kPartitionKey);
  testType(VARCHAR(), HiveColumnHandle::ColumnType::kRegular);
  testType(VARBINARY(), HiveColumnHandle::ColumnType::kRegular);
  testType(TIMESTAMP(), HiveColumnHandle::ColumnType::kRegular);
  testType(DATE(), HiveColumnHandle::ColumnType::kPartitionKey);
  testType(DECIMAL(10, 2), HiveColumnHandle::ColumnType::kRegular);
}

TEST(IcebergColumnHandleTest, structType) {
  IcebergNestedField childField1{10, {}};
  IcebergNestedField childField2{11, {}};
  IcebergNestedField nestedField{1, {childField1, childField2}};

  auto structType = ROW({{"field1", INTEGER()}, {"field2", VARCHAR()}});
  auto columnHandle = std::make_shared<IcebergColumnHandle>(
      "struct_column",
      HiveColumnHandle::ColumnType::kPartitionKey,
      structType,
      structType,
      nestedField);

  EXPECT_EQ(columnHandle->name(), "struct_column");
  EXPECT_EQ(columnHandle->nestedField().id, 1);
  EXPECT_EQ(columnHandle->nestedField().children.size(), 2);
  EXPECT_EQ(columnHandle->nestedField().children[0].id, 10);
  EXPECT_EQ(columnHandle->nestedField().children[1].id, 11);
}

TEST(IcebergColumnHandleTest, arrayType) {
  IcebergNestedField elementField{101, {}};
  IcebergNestedField nestedField{1, {elementField}};

  auto arrayType = ARRAY(INTEGER());
  auto columnHandle = std::make_shared<IcebergColumnHandle>(
      "array_column",
      HiveColumnHandle::ColumnType::kRegular,
      arrayType,
      arrayType,
      nestedField);

  EXPECT_EQ(columnHandle->name(), "array_column");
  EXPECT_EQ(columnHandle->dataType()->kind(), TypeKind::ARRAY);
  EXPECT_EQ(columnHandle->nestedField().id, 1);
  EXPECT_EQ(columnHandle->nestedField().children.size(), 1);
  EXPECT_EQ(columnHandle->nestedField().children[0].id, 101);
}

TEST(IcebergColumnHandleTest, mapType) {
  IcebergNestedField keyField{201, {}};
  IcebergNestedField valueField{202, {}};
  IcebergNestedField nestedField{2, {keyField, valueField}};

  auto mapType = MAP(VARCHAR(), INTEGER());
  auto columnHandle = std::make_shared<IcebergColumnHandle>(
      "map_column",
      HiveColumnHandle::ColumnType::kRegular,
      mapType,
      mapType,
      nestedField);

  EXPECT_EQ(columnHandle->name(), "map_column");
  EXPECT_EQ(columnHandle->dataType()->kind(), TypeKind::MAP);
  EXPECT_EQ(columnHandle->nestedField().id, 2);
  EXPECT_EQ(columnHandle->nestedField().children.size(), 2);
  EXPECT_EQ(columnHandle->nestedField().children[0].id, 201); // key
  EXPECT_EQ(columnHandle->nestedField().children[1].id, 202); // value
}

TEST(IcebergColumnHandleTest, complexNestedType) {
  IcebergNestedField rowField1{401, {}};
  IcebergNestedField rowField2{402, {}};
  IcebergNestedField rowField{400, {rowField1, rowField2}};
  IcebergNestedField mapKeyField{301, {}};
  IcebergNestedField mapValueField{302, {rowField}};
  IcebergNestedField mapField{300, {mapKeyField, mapValueField}};
  IcebergNestedField arrayElementField{200, {mapField}};
  IcebergNestedField rootField{1, {arrayElementField}};

  auto complexType = ARRAY(
      MAP(VARCHAR(),
          ROW({{"nested_field1", BIGINT()}, {"nested_field2", REAL()}})));
  auto columnHandle = std::make_shared<IcebergColumnHandle>(
      "complex_column",
      HiveColumnHandle::ColumnType::kRegular,
      complexType,
      complexType,
      rootField);

  EXPECT_EQ(columnHandle->name(), "complex_column");
  EXPECT_EQ(columnHandle->nestedField().id, 1);
  EXPECT_EQ(columnHandle->nestedField().children.size(), 1);

  const auto& arrayElement = columnHandle->nestedField().children[0];
  EXPECT_EQ(arrayElement.id, 200);
  EXPECT_EQ(arrayElement.children.size(), 1);

  const auto& mapElement = arrayElement.children[0];
  EXPECT_EQ(mapElement.id, 300);
  EXPECT_EQ(mapElement.children.size(), 2);

  EXPECT_EQ(mapElement.children[0].id, 301);
  EXPECT_EQ(mapElement.children[1].id, 302);

  const auto& rowElement = mapElement.children[1].children[0];
  EXPECT_EQ(rowElement.id, 400);
  EXPECT_EQ(rowElement.children.size(), 2);
  EXPECT_EQ(rowElement.children[0].id, 401);
  EXPECT_EQ(rowElement.children[1].id, 402);
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
