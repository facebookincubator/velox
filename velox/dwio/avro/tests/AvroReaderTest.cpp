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

#include <functional>
#include <optional>
#include <sstream>

#include <avro/Compiler.hh>
#include <avro/DataFile.hh>
#include <avro/Generic.hh>

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/File.h"
#include "velox/dwio/avro/RegisterAvroReader.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/exec/tests/utils/TempFilePath.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::avro {
namespace {

std::shared_ptr<exec::test::TempFilePath> writeAvroFile(
    const std::string& schemaJson,
    std::function<void(
        ::avro::DataFileWriter<::avro::GenericDatum>&,
        const ::avro::ValidSchema&)> writeRows) {
  ::avro::ValidSchema schema;
  std::istringstream schemaStream(schemaJson);
  ::avro::compileJsonSchema(schemaStream, schema);

  auto path = exec::test::TempFilePath::create();
  ::avro::DataFileWriter<::avro::GenericDatum> writer(
      path->getPath().c_str(), schema);
  writeRows(writer, schema);
  writer.close();
  return path;
}

class AvroReaderTest : public testing::Test,
                       public velox::test::VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    registerAvroReaderFactory();
  }

  void TearDown() override {
    unregisterAvroReaderFactory();
  }

  void setScanSpec(const Type& type, dwio::common::RowReaderOptions& options) {
    auto spec = std::make_shared<common::ScanSpec>("root");
    spec->addAllChildFields(type);
    options.setScanSpec(spec);
  }

  std::unique_ptr<dwio::common::Reader> createReader(
      const std::shared_ptr<exec::test::TempFilePath>& filePath,
      std::optional<dwio::common::ReaderOptions> readerOptions = std::nullopt) {
    auto options = readerOptions.value_or(dwio::common::ReaderOptions{pool()});
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::AVRO);
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(filePath->getPath()), *pool());
    return factory->createReader(std::move(input), options);
  }

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      dwio::common::Reader& reader) {
    dwio::common::RowReaderOptions rowOptions;
    setScanSpec(*reader.rowType(), rowOptions);
    return reader.createRowReader(rowOptions);
  }

  VectorPtr readRows(
      dwio::common::Reader& reader,
      uint64_t maxRows,
      uint64_t expectedCount) {
    auto rowReader = createRowReader(reader);
    VectorPtr result;
    EXPECT_EQ(rowReader->next(maxRows, result), expectedCount);
    return result;
  }
};

TEST_F(AvroReaderTest, primitiveAndLogicalTypes) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "AllTypesRecord",
      "fields": [
        {"name": "boolCol", "type": "boolean"},
        {"name": "intCol", "type": "int"},
        {"name": "longCol", "type": "long"},
        {"name": "floatCol", "type": "float"},
        {"name": "doubleCol", "type": "double"},
        {"name": "stringCol", "type": "string"},
        {"name": "bytesCol", "type": "bytes"},
        {"name": "fixedCol", "type": {"type": "fixed", "name": "FixedTwo", "size": 2}},
        {"name": "enumCol", "type": {"type": "enum", "name": "Color", "symbols": ["RED", "GREEN"]}},
        {"name": "arrayCol", "type": {"type": "array", "items": "int"}},
        {"name": "mapCol", "type": {"type": "map", "values": "long"}},
        {"name": "dateCol", "type": {"type": "int", "logicalType": "date"}},
        {"name": "timeMillisCol", "type": {"type": "int", "logicalType": "time-millis"}},
        {"name": "timeMicrosCol", "type": {"type": "long", "logicalType": "time-micros"}},
        {"name": "tsMillisCol", "type": {"type": "long", "logicalType": "timestamp-millis"}},
        {"name": "tsMicrosCol", "type": {"type": "long", "logicalType": "timestamp-micros"}},
        {"name": "uuidCol", "type": {"type": "string", "logicalType": "uuid"}},
        {"name": "decimalBytesCol", "type": {"type": "bytes", "logicalType": "decimal", "precision": 9, "scale": 2}},
        {"name": "decimalFixedCol", "type": {"type": "fixed", "name": "FixedDecimal", "size": 4, "logicalType": "decimal", "precision": 7, "scale": 3}}
      ]
    }
  )JSON";

  auto filePath = writeAvroFile(
      schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        auto& record = datum.value<::avro::GenericRecord>();

        auto populateRow =
            [&](bool boolValue,
                int32_t intValue,
                int64_t longValue,
                float floatValue,
                double doubleValue,
                const std::string& text,
                const std::vector<uint8_t>& bytes,
                const std::vector<uint8_t>& fixed,
                size_t enumIndex,
                const std::vector<int32_t>& arrayValues,
                const std::vector<std::pair<std::string, int64_t>>& mapValues,
                int32_t dateValue,
                int32_t timeMillis,
                int64_t timeMicros,
                int64_t tsMillis,
                int64_t tsMicros,
                const std::string& uuidValue,
                const std::vector<uint8_t>& decimalBytes,
                const std::vector<uint8_t>& decimalFixed) {
              record.fieldAt(0).value<bool>() = boolValue;
              record.fieldAt(1).value<int32_t>() = intValue;
              record.fieldAt(2).value<int64_t>() = longValue;
              record.fieldAt(3).value<float>() = floatValue;
              record.fieldAt(4).value<double>() = doubleValue;
              record.fieldAt(5).value<std::string>() = text;
              record.fieldAt(6).value<std::vector<uint8_t>>() = bytes;
              record.fieldAt(7).value<::avro::GenericFixed>().value() = fixed;
              record.fieldAt(8).value<::avro::GenericEnum>().set(enumIndex);

              auto& array =
                  record.fieldAt(9).value<::avro::GenericArray>().value();
              array.clear();
              for (auto value : arrayValues) {
                array.emplace_back(value);
              }

              auto& map =
                  record.fieldAt(10).value<::avro::GenericMap>().value();
              map.clear();
              for (const auto& [key, value] : mapValues) {
                map.emplace_back(key, ::avro::GenericDatum(value));
              }

              record.fieldAt(11).value<int32_t>() = dateValue;
              record.fieldAt(12).value<int32_t>() = timeMillis;
              record.fieldAt(13).value<int64_t>() = timeMicros;
              record.fieldAt(14).value<int64_t>() = tsMillis;
              record.fieldAt(15).value<int64_t>() = tsMicros;
              record.fieldAt(16).value<std::string>() = uuidValue;
              record.fieldAt(17).value<std::vector<uint8_t>>() = decimalBytes;
              record.fieldAt(18).value<::avro::GenericFixed>().value() =
                  decimalFixed;

              writer.write(datum);
            };

        populateRow(
            true,
            123,
            7890,
            1.5F,
            3.25,
            "alpha",
            {0x01, 0x02},
            {0xAA, 0xBB},
            0,
            {1, 2},
            {{"a", 10}, {"b", 20}},
            1000,
            1234,
            5678,
            1700,
            3500,
            "123e4567-e89b-12d3-a456-426655440000",
            {0x00, 0x00, 0x30, 0x39},
            {0x00, 0x00, 0x1A, 0x85});

        populateRow(
            false,
            -456,
            -9876,
            -2.5F,
            -4.75,
            "beta",
            {0x0A, 0x0B, 0x0C},
            {0xCC, 0xDD},
            1,
            {3, 4},
            {{"c", -5}},
            2000,
            4321,
            8765,
            2700,
            4500,
            "00000000-0000-0000-0000-000000000000",
            {0xFF, 0xFF, 0xEF, 0x98},
            {0xFF, 0xFF, 0xFA, 0xB3});
      });

  auto reader = createReader(filePath);
  auto result = readRows(*reader, 5, 2);

  auto decimalBytesType = DECIMAL(9, 2);
  auto decimalFixedType = DECIMAL(7, 3);
  auto expected = makeRowVector({
      makeFlatVector<bool>({true, false}),
      makeFlatVector<int32_t>({123, -456}),
      makeFlatVector<int64_t>({7890, -9876}),
      makeFlatVector<float>({1.5F, -2.5F}),
      makeFlatVector<double>({3.25, -4.75}),
      makeFlatVector<std::string>({"alpha", "beta"}),
      makeFlatVector<std::string>({"\x01\x02", "\x0A\x0B\x0C"}, VARBINARY()),
      makeFlatVector<std::string>({"\xAA\xBB", "\xCC\xDD"}, VARBINARY()),
      makeFlatVector<std::string>({"RED", "GREEN"}),
      makeArrayVector<int32_t>({{1, 2}, {3, 4}}),
      makeMapVector<std::string, int64_t>(
          {{{"a", 10}, {"b", 20}}, {{"c", -5}}}),
      makeFlatVector<int32_t>({1000, 2000}, DATE()),
      makeFlatVector<int64_t>({1234 * 1000L, 4321 * 1000L}, TIME()),
      makeFlatVector<int64_t>({5678, 8765}, TIME()),
      makeFlatVector<Timestamp>(
          {Timestamp::fromMillis(1700), Timestamp::fromMillis(2700)}),
      makeFlatVector<Timestamp>(
          {Timestamp::fromMicros(3500), Timestamp::fromMicros(4500)}),
      makeFlatVector<std::string>(
          {"123e4567-e89b-12d3-a456-426655440000",
           "00000000-0000-0000-0000-000000000000"}),
      makeFlatVector<int64_t>({12345, -4200}, decimalBytesType),
      makeFlatVector<int64_t>({6789, -1357}, decimalFixedType),
  });

  assertEqualVectors(expected, result);
}

TEST_F(AvroReaderTest, schemaLiteralOverride) {
  const std::string writerSchemaJson = R"JSON(
    {
      "type": "record",
      "name": "OverrideRecord",
      "fields": [
        {"name": "number", "type": "int"}
      ]
    }
  )JSON";

  const std::string overrideSchemaJson = R"JSON(
    {
      "type": "record",
      "name": "OverrideRecord",
      "fields": [
        {"name": "number", "type": "long"}
      ]
    }
  )JSON";

  auto filePath = writeAvroFile(
      writerSchemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        auto& record = datum.value<::avro::GenericRecord>();

        record.fieldAt(0).value<int32_t>() = 1;
        writer.write(datum);

        record.fieldAt(0).value<int32_t>() = 2;
        writer.write(datum);
      });

  auto readerOptions = dwio::common::ReaderOptions(pool());
  readerOptions.serDeOptions().parameters["avro.schema.literal"] =
      overrideSchemaJson;
  auto reader = createReader(filePath, readerOptions);

  ASSERT_EQ(reader->rowType()->childAt(0)->kind(), TypeKind::BIGINT);

  auto result = readRows(*reader, 10, 2);

  auto expected = makeRowVector({makeFlatVector<int64_t>({1, 2})});
  assertEqualVectors(expected, result);
}

TEST_F(AvroReaderTest, scanBatchBytesRespected) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "BatchRecord",
      "fields": [
        {"name": "text", "type": "string"}
      ]
    }
  )JSON";

  auto filePath = writeAvroFile(
      schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        auto& record = datum.value<::avro::GenericRecord>();

        record.fieldAt(0).value<std::string>() = "first";
        writer.write(datum);

        record.fieldAt(0).value<std::string>() = "second";
        writer.write(datum);

        record.fieldAt(0).value<std::string>() = "third";
        writer.write(datum);
      });

  auto readerOptions = dwio::common::ReaderOptions(pool());
  readerOptions.serDeOptions().parameters["avro.scan.batch.bytes"] = "1";

  auto reader = createReader(filePath, readerOptions);
  auto rowReader = createRowReader(*reader);

  VectorPtr firstBatch;
  ASSERT_EQ(rowReader->next(1, firstBatch), 1);

  VectorPtr secondBatch;
  ASSERT_EQ(rowReader->next(10, secondBatch), 1);

  VectorPtr thirdBatch;
  ASSERT_EQ(rowReader->next(10, thirdBatch), 1);

  auto expected = makeRowVector({makeFlatVector<std::string>({"first"})});
  assertEqualVectors(expected, firstBatch);

  expected = makeRowVector({makeFlatVector<std::string>({"second"})});
  assertEqualVectors(expected, secondBatch);

  expected = makeRowVector({makeFlatVector<std::string>({"third"})});
  assertEqualVectors(expected, thirdBatch);
}

TEST_F(AvroReaderTest, nullableUnion) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "UnionRecord",
      "fields": [
        {"name": "nullableString", "type": ["null", "string"]}
      ]
    }
  )JSON";

  auto filePath = writeAvroFile(
      schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        auto& record = datum.value<::avro::GenericRecord>();

        record.fieldAt(0).selectBranch(0);
        writer.write(datum);

        record.fieldAt(0).selectBranch(1);
        record.fieldAt(0).value<std::string>() = "text";
        writer.write(datum);
      });

  auto reader = createReader(filePath);

  auto result = readRows(*reader, 2, 2);
  auto expected = makeRowVector(
      {makeNullableFlatVector<std::string>({std::nullopt, "text"})});
  assertEqualVectors(expected, result);
}

TEST_F(AvroReaderTest, numericPromotionUnion) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "NumericPromotionRecord",
      "fields": [
        {"name": "numeric", "type": ["null", "int", "long"]}
      ]
    }
  )JSON";

  auto filePath = writeAvroFile(
      schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        auto& record = datum.value<::avro::GenericRecord>();

        record.fieldAt(0).selectBranch(0);
        writer.write(datum);

        record.fieldAt(0).selectBranch(1);
        record.fieldAt(0).value<int32_t>() = 12;
        writer.write(datum);

        record.fieldAt(0).selectBranch(2);
        record.fieldAt(0).value<int64_t>() = 9'000'000'000;
        writer.write(datum);
      });

  auto reader = createReader(filePath);

  auto result = readRows(*reader, 5, 3);

  auto expected = makeRowVector(
      {makeNullableFlatVector<int64_t>({std::nullopt, 12, 9'000'000'000})});
  assertEqualVectors(expected, result);
}

TEST_F(AvroReaderTest, numericPromotionUnionFloatDouble) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "NumericPromotionFloatDoubleRecord",
      "fields": [
        {"name": "numeric", "type": ["null", "float", "double"]}
      ]
    }
  )JSON";

  auto filePath = writeAvroFile(
      schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        auto& record = datum.value<::avro::GenericRecord>();

        record.fieldAt(0).selectBranch(0);
        writer.write(datum);

        record.fieldAt(0).selectBranch(1);
        record.fieldAt(0).value<float>() = 1.5f;
        writer.write(datum);

        record.fieldAt(0).selectBranch(2);
        record.fieldAt(0).value<double>() = 3.75;
        writer.write(datum);
      });

  auto reader = createReader(filePath);

  auto result = readRows(*reader, 5, 3);

  auto expected = makeRowVector(
      {makeNullableFlatVector<double>({std::nullopt, 1.5, 3.75})});
  assertEqualVectors(expected, result);
}

TEST_F(AvroReaderTest, structUnionWithNullBranch) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "StructUnionNullableRecord",
      "fields": [
        {"name": "unionField", "type": ["null", "string", "long"]}
      ]
    }
  )JSON";

  auto filePath = writeAvroFile(
      schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        auto& record = datum.value<::avro::GenericRecord>();

        record.fieldAt(0).selectBranch(0);
        writer.write(datum);

        record.fieldAt(0).selectBranch(1);
        record.fieldAt(0).value<std::string>() = "alpha";
        writer.write(datum);

        record.fieldAt(0).selectBranch(2);
        record.fieldAt(0).value<int64_t>() = 42;
        writer.write(datum);
      });

  auto reader = createReader(filePath);

  const auto& rootType = reader->rowType()->asRow();
  ASSERT_EQ(rootType.size(), 1);
  EXPECT_EQ(rootType.nameOf(0), "unionField");

  const auto& unionType = rootType.childAt(0)->asRow();
  ASSERT_EQ(unionType.size(), 2);
  EXPECT_EQ(unionType.nameOf(0), "member0");
  EXPECT_TRUE(unionType.childAt(0)->equivalent(*VARCHAR()));
  EXPECT_EQ(unionType.nameOf(1), "member1");
  EXPECT_TRUE(unionType.childAt(1)->equivalent(*BIGINT()));

  auto result = readRows(*reader, 5, 3);

  auto expected = makeRowVector({makeRowVector(
      {makeNullableFlatVector<std::string>(
           {std::nullopt, "alpha", std::nullopt}),
       makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 42})},
      [](vector_size_t row) { return row == 0; })});

  assertEqualVectors(expected, result);
}

TEST_F(AvroReaderTest, fileColumnNamesReadAsLowerCase) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "CaseRecord",
      "fields": [
        {"name": "MixedCase", "type": "long"},
        {"name": "SECOND", "type": "int"}
      ]
    }
  )JSON";

  auto filePath = writeAvroFile(
      schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        auto& record = datum.value<::avro::GenericRecord>();

        record.fieldAt(0).value<int64_t>() = 123;
        record.fieldAt(1).value<int32_t>() = 5;
        writer.write(datum);
      });

  dwio::common::ReaderOptions readerOptions(pool());
  readerOptions.setFileColumnNamesReadAsLowerCase(true);
  auto reader = createReader(filePath, readerOptions);

  const auto& rootType = reader->rowType()->asRow();
  ASSERT_EQ(rootType.size(), 2);
  EXPECT_EQ(rootType.nameOf(0), "mixedcase");
  EXPECT_EQ(rootType.nameOf(1), "second");

  auto result = readRows(*reader, 3, 1);
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({123}), makeFlatVector<int32_t>({5})});
  assertEqualVectors(expected, result);
}

} // namespace
} // namespace facebook::velox::avro
