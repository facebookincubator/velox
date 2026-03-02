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

#include <avro/Compiler.hh>
#include <avro/DataFile.hh>
#include <avro/Generic.hh>

#include "velox/common/base/VeloxException.h"
#include "velox/dwio/avro/RegisterAvroReader.h"
#include "velox/exec/tests/utils/TempFilePath.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::test;

namespace facebook::velox::avro {
namespace {
class AvroReaderTest : public testing::Test, public VectorTestBase {
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

  void setScanSpec(const Type& type, dwio::common::RowReaderOptions& options)
      const {
    auto spec = std::make_shared<common::ScanSpec>("root");
    spec->addAllChildFields(type);
    options.setScanSpec(spec);
  }

  std::unique_ptr<dwio::common::Reader> createReader(
      const std::shared_ptr<exec::test::TempFilePath>& filePath,
      std::optional<dwio::common::ReaderOptions> readerOptions =
          std::nullopt) const {
    auto options = readerOptions.value_or(dwio::common::ReaderOptions{pool()});
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::AVRO);
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(filePath->getPath()), *pool());
    return factory->createReader(std::move(input), options);
  }

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      dwio::common::Reader& reader,
      std::optional<dwio::common::RowReaderOptions> rowOptions =
          std::nullopt) const {
    auto options = rowOptions.value_or(dwio::common::RowReaderOptions{});
    setScanSpec(*reader.rowType(), options);
    return reader.createRowReader(options);
  }

  VectorPtr readRows(
      dwio::common::Reader& reader,
      uint64_t maxRows,
      uint64_t expectedCount,
      std::optional<dwio::common::RowReaderOptions> rowOptions =
          std::nullopt) const {
    auto rowReader = createRowReader(reader, rowOptions);
    VectorPtr result;
    EXPECT_EQ(rowReader->next(maxRows, result), expectedCount);
    return result;
  }

  static std::shared_ptr<exec::test::TempFilePath> writeAvroFile(
      const std::string& schemaJson,
      const std::function<void(
          ::avro::DataFileWriter<::avro::GenericDatum>&,
          const ::avro::ValidSchema&)>& writeRows) {
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

  std::shared_ptr<exec::test::TempFilePath> writeAllTypesRecord() const {
    const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "AllTypesRecord",
      "fields": [
        {"name": "nullCol", "type": "null"},
        {"name": "boolCol", "type": "boolean"},
        {"name": "intCol", "type": "int"},
        {"name": "longCol", "type": "long"},
        {"name": "floatCol", "type": "float"},
        {"name": "doubleCol", "type": "double"},
        {"name": "stringCol", "type": "string"},
        {"name": "bytesCol", "type": "bytes"},
        {"name": "enumCol", "type": {"type": "enum", "name": "Color", "symbols": ["RED", "GREEN"]}},
        {"name": "fixedCol", "type": {"type": "fixed", "name": "FixedTwo", "size": 2}},
        {"name": "arrayCol", "type": {"type": "array", "items": "int"}},
        {"name": "mapCol", "type": {"type": "map", "values": "long"}},
        {"name": "recordCol", "type": {"type": "record", "name": "SubRecord", "fields": [
          {"name": "subInt", "type": "int"},
          {"name": "subString", "type": "string"}
        ]}},
        {"name": "unionCol", "type": ["null", "int"]},
        {"name": "dateCol", "type": {"type": "int", "logicalType": "date"}},
        {"name": "timeMillisCol", "type": {"type": "int", "logicalType": "time-millis"}},
        {"name": "timeMicrosCol", "type": {"type": "long", "logicalType": "time-micros"}},
        {"name": "tsMillisCol", "type": {"type": "long", "logicalType": "timestamp-millis"}},
        {"name": "tsMicrosCol", "type": {"type": "long", "logicalType": "timestamp-micros"}},
        {"name": "tsNanosCol", "type": {"type": "long", "logicalType": "timestamp-nanos"}},
        {"name": "uuidCol", "type": {"type": "string", "logicalType": "uuid"}},
        {"name": "uuidFixedCol", "type": {"type": "fixed", "name": "FixedUuid", "size": 16, "logicalType": "uuid"}},
        {"name": "decimalBytesCol", "type": {"type": "bytes", "logicalType": "decimal", "precision": 9, "scale": 2}},
        {"name": "decimalFixedCol", "type": {"type": "fixed", "name": "FixedDecimal", "size": 4, "logicalType": "decimal", "precision": 7, "scale": 3}}
      ]
    })JSON";

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
                  int32_t subInt,
                  const std::string& subString,
                  std::optional<int32_t> unionValue,
                  int32_t dateValue,
                  int32_t timeMillis,
                  int64_t timeMicros,
                  int64_t tsMillis,
                  int64_t tsMicros,
                  int64_t tsNanos,
                  const std::string& uuidValue,
                  const std::vector<uint8_t>& uuidFixed,
                  const std::vector<uint8_t>& decimalBytes,
                  const std::vector<uint8_t>& decimalFixed) {
                record.fieldAt(0) = ::avro::GenericDatum();
                record.fieldAt(1).value<bool>() = boolValue;
                record.fieldAt(2).value<int32_t>() = intValue;
                record.fieldAt(3).value<int64_t>() = longValue;
                record.fieldAt(4).value<float>() = floatValue;
                record.fieldAt(5).value<double>() = doubleValue;
                record.fieldAt(6).value<std::string>() = text;
                record.fieldAt(7).value<std::vector<uint8_t>>() = bytes;
                record.fieldAt(8).value<::avro::GenericEnum>().set(enumIndex);
                record.fieldAt(9).value<::avro::GenericFixed>().value() = fixed;

                auto& array =
                    record.fieldAt(10).value<::avro::GenericArray>().value();
                array.clear();
                for (auto value : arrayValues) {
                  array.emplace_back(value);
                }
                auto& map =
                    record.fieldAt(11).value<::avro::GenericMap>().value();
                map.clear();
                for (const auto& [key, value] : mapValues) {
                  map.emplace_back(key, ::avro::GenericDatum(value));
                }
                auto& subRecord =
                    record.fieldAt(12).value<::avro::GenericRecord>();
                subRecord.fieldAt(0).value<int32_t>() = subInt;
                subRecord.fieldAt(1).value<std::string>() = subString;
                auto& unionDatum = record.fieldAt(13);
                if (unionValue.has_value()) {
                  unionDatum.selectBranch(1);
                  unionDatum.value<int32_t>() = unionValue.value();
                } else {
                  unionDatum.selectBranch(0);
                }
                record.fieldAt(14).value<int32_t>() = dateValue;
                record.fieldAt(15).value<int32_t>() = timeMillis;
                record.fieldAt(16).value<int64_t>() = timeMicros;
                record.fieldAt(17).value<int64_t>() = tsMillis;
                record.fieldAt(18).value<int64_t>() = tsMicros;
                record.fieldAt(19).value<int64_t>() = tsNanos;
                record.fieldAt(20).value<std::string>() = uuidValue;
                record.fieldAt(21).value<::avro::GenericFixed>().value() =
                    uuidFixed;
                record.fieldAt(22).value<std::vector<uint8_t>>() = decimalBytes;
                record.fieldAt(23).value<::avro::GenericFixed>().value() =
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
              101,
              "sub-alpha",
              42,
              1000,
              1234,
              5678,
              1700,
              3500,
              9876543210,
              "123e4567-e89b-12d3-a456-426655440000",
              {0x10,
               0x11,
               0x12,
               0x13,
               0x14,
               0x15,
               0x16,
               0x17,
               0x18,
               0x19,
               0x1A,
               0x1B,
               0x1C,
               0x1D,
               0x1E,
               0x1F},
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
              202,
              "sub-beta",
              std::nullopt,
              2000,
              4321,
              8765,
              2700,
              4500,
              1234567890,
              "00000000-0000-0000-0000-000000000000",
              {0xFF,
               0xEE,
               0xDD,
               0xCC,
               0xBB,
               0xAA,
               0x99,
               0x88,
               0x77,
               0x66,
               0x55,
               0x44,
               0x33,
               0x22,
               0x11,
               0x00},
              {0xFF, 0xFF, 0xEF, 0x98},
              {0xFF, 0xFF, 0xFA, 0xB3});
        });
    return filePath;
  }

  // Avro schema for the file produced by the function below
  //
  // OuterRecord (record)
  // ├── meta : MetaRecord (record)
  // │   ├── ids : array<long>
  // │   └── attrs : map<string, nullable<string>>
  // │
  // └── payloads : array<PayloadRecord>
  //     └── PayloadRecord (record)
  //         ├── flags : array<nullable<boolean>>
  //         └── properties : map<string, PropertyRecord>
  //             └── PropertyRecord (record)
  //                 ├── name : string
  //                 └── values : array<int>
  std::shared_ptr<exec::test::TempFilePath> writeComplexNestedRecord() const {
    const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "OuterRecord",
      "fields": [
        {"name": "meta", "type": {
          "type": "record",
          "name": "MetaRecord",
          "fields": [
            {"name": "ids", "type": {"type": "array", "items": "long"}},
            {"name": "attrs", "type": {"type": "map", "values": ["null", "string"]}}
          ]
        }},
        {"name": "payloads", "type": {"type": "array", "items": {
          "type": "record",
          "name": "PayloadRecord",
          "fields": [
            {"name": "flags", "type": {"type": "array", "items": ["null", "boolean"]}},
            {"name": "properties", "type": {"type": "map", "values": {
              "type": "record",
              "name": "PropertyRecord",
              "fields": [
                {"name": "name", "type": "string"},
                {"name": "values", "type": {"type": "array", "items": "int"}}
              ]
            }}}
          ]
        }}}
      ]
    })JSON";

    auto filePath = writeAvroFile(
        schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
          ::avro::GenericDatum datum(schema.root());
          auto& record = datum.value<::avro::GenericRecord>();

          auto& metaRecord = record.fieldAt(0).value<::avro::GenericRecord>();
          auto& ids =
              metaRecord.fieldAt(0).value<::avro::GenericArray>().value();
          ids.emplace_back(static_cast<int64_t>(101));
          ids.emplace_back(static_cast<int64_t>(202));
          auto& attrs =
              metaRecord.fieldAt(1).value<::avro::GenericMap>().value();
          auto attrValueSchema = schema.root()->leafAt(0)->leafAt(1)->leafAt(1);
          ::avro::GenericDatum nullAttr(attrValueSchema);
          nullAttr.selectBranch(0);
          attrs.emplace_back("alpha", nullAttr);
          ::avro::GenericDatum textAttr(attrValueSchema);
          textAttr.selectBranch(1);
          textAttr.value<std::string>() = "beta";
          attrs.emplace_back("beta", textAttr);

          auto& payloadsArray =
              record.fieldAt(1).value<::avro::GenericArray>().value();
          auto payloadSchema = schema.root()->leafAt(1)->leafAt(0);
          ::avro::GenericDatum payloadDatum(payloadSchema);
          auto& payloadRecord = payloadDatum.value<::avro::GenericRecord>();
          auto& flags =
              payloadRecord.fieldAt(0).value<::avro::GenericArray>().value();
          auto flagSchema = payloadSchema->leafAt(0)->leafAt(0);
          ::avro::GenericDatum nullFlag(flagSchema);
          nullFlag.selectBranch(0);
          flags.emplace_back(nullFlag);
          ::avro::GenericDatum trueFlag(flagSchema);
          trueFlag.selectBranch(1);
          trueFlag.value<bool>() = true;
          flags.emplace_back(trueFlag);
          auto& properties =
              payloadRecord.fieldAt(1).value<::avro::GenericMap>().value();
          auto propertySchema = payloadSchema->leafAt(1)->leafAt(1);
          ::avro::GenericDatum propertyDatum(propertySchema);
          auto& propertyRecord = propertyDatum.value<::avro::GenericRecord>();
          propertyRecord.fieldAt(0).value<std::string>() = "p1";
          auto& values =
              propertyRecord.fieldAt(1).value<::avro::GenericArray>().value();
          values.emplace_back(1);
          values.emplace_back(2);
          properties.emplace_back("k1", propertyDatum);
          payloadsArray.emplace_back(payloadDatum);

          writer.write(datum);
        });
    return filePath;
  }

  std::shared_ptr<exec::test::TempFilePath> writeUnionRecord() const {
    const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "UnionRecord",
      "fields": [
        {"name": "simpleString", "type": ["string"]},
        {"name": "simpleNull", "type": ["null"]},
        {"name": "nullableInt", "type": ["null", "int"]},
        {"name": "intOrLong", "type": ["int", "long"]},
        {"name": "floatOrDouble", "type": ["float", "double"]},
        {"name": "nullableIntOrLong", "type": ["null", "int", "long"]},
        {"name": "nullableFloatOrDouble", "type": ["float", "null", "double"]},
        {"name": "mixedUnion", "type": ["int", "string"]},
        {"name": "nullableMixedUnion", "type": ["int", "long", "string", "null"]},
        {"name": "simpleNestedRecord", "type": [{
          "type": "record",
          "name": "NestedRecord",
          "fields": [
            {"name": "nestedInt", "type": "int"},
            {"name": "nestedString", "type": "string"}
          ]
        }]},
        {"name": "nullableNestedRecord", "type": ["null", "NestedRecord"]}
      ]
    })JSON";
    auto filePath = writeAvroFile(
        schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
          ::avro::GenericDatum datum(schema.root());
          auto& record = datum.value<::avro::GenericRecord>();
          auto writeRow =
              [&](std::string simpleString,
                  std::optional<int32_t> nullableInt,
                  int64_t intOrLongValue,
                  bool intOrLongIsLong,
                  double floatOrDoubleValue,
                  bool floatOrDoubleIsDouble,
                  std::optional<int64_t> nullableIntOrLongValue,
                  bool nullableIntOrLongIsLong,
                  std::optional<double> nullableFloatOrDoubleValue,
                  bool nullableFloatOrDoubleIsDouble,
                  std::optional<int32_t> mixedUnionInt,
                  std::optional<std::string> mixedUnionString,
                  std::optional<int32_t> nullableMixedUnionInt,
                  std::optional<int64_t> nullableMixedUnionLong,
                  std::optional<std::string> nullableMixedUnionString,
                  int32_t simpleNestedInt,
                  std::string simpleNestedString,
                  std::optional<int32_t> nullableNestedInt,
                  std::optional<std::string> nullableNestedString) {
                auto& simpleStringDatum = record.fieldAt(0);
                simpleStringDatum.selectBranch(0);
                simpleStringDatum.value<std::string>() = simpleString;

                auto& simpleNullDatum = record.fieldAt(1);
                simpleNullDatum.selectBranch(0);

                auto& nullableIntDatum = record.fieldAt(2);
                if (nullableInt.has_value()) {
                  nullableIntDatum.selectBranch(1);
                  nullableIntDatum.value<int32_t>() = nullableInt.value();
                } else {
                  nullableIntDatum.selectBranch(0);
                }

                auto& intOrLongDatum = record.fieldAt(3);
                if (intOrLongIsLong) {
                  intOrLongDatum.selectBranch(1);
                  intOrLongDatum.value<int64_t>() = intOrLongValue;
                } else {
                  intOrLongDatum.selectBranch(0);
                  intOrLongDatum.value<int32_t>() =
                      static_cast<int32_t>(intOrLongValue);
                }

                auto& floatOrDoubleDatum = record.fieldAt(4);
                if (floatOrDoubleIsDouble) {
                  floatOrDoubleDatum.selectBranch(1);
                  floatOrDoubleDatum.value<double>() = floatOrDoubleValue;
                } else {
                  floatOrDoubleDatum.selectBranch(0);
                  floatOrDoubleDatum.value<float>() =
                      static_cast<float>(floatOrDoubleValue);
                }

                auto& nullableIntOrLongDatum = record.fieldAt(5);
                if (nullableIntOrLongValue.has_value()) {
                  if (nullableIntOrLongIsLong) {
                    nullableIntOrLongDatum.selectBranch(2);
                    nullableIntOrLongDatum.value<int64_t>() =
                        nullableIntOrLongValue.value();
                  } else {
                    nullableIntOrLongDatum.selectBranch(1);
                    nullableIntOrLongDatum.value<int32_t>() =
                        static_cast<int32_t>(nullableIntOrLongValue.value());
                  }
                } else {
                  nullableIntOrLongDatum.selectBranch(0);
                }

                auto& nullableFloatOrDoubleDatum = record.fieldAt(6);
                if (nullableFloatOrDoubleValue.has_value()) {
                  if (nullableFloatOrDoubleIsDouble) {
                    nullableFloatOrDoubleDatum.selectBranch(2);
                    nullableFloatOrDoubleDatum.value<double>() =
                        nullableFloatOrDoubleValue.value();
                  } else {
                    nullableFloatOrDoubleDatum.selectBranch(0);
                    nullableFloatOrDoubleDatum.value<float>() =
                        static_cast<float>(nullableFloatOrDoubleValue.value());
                  }
                } else {
                  nullableFloatOrDoubleDatum.selectBranch(1);
                }

                auto& mixedUnionDatum = record.fieldAt(7);
                if (mixedUnionInt.has_value()) {
                  mixedUnionDatum.selectBranch(0);
                  mixedUnionDatum.value<int32_t>() = mixedUnionInt.value();
                } else {
                  mixedUnionDatum.selectBranch(1);
                  mixedUnionDatum.value<std::string>() =
                      mixedUnionString.value();
                }

                auto& nullableMixedUnionDatum = record.fieldAt(8);
                if (nullableMixedUnionInt.has_value()) {
                  nullableMixedUnionDatum.selectBranch(0);
                  nullableMixedUnionDatum.value<int32_t>() =
                      nullableMixedUnionInt.value();
                } else if (nullableMixedUnionLong.has_value()) {
                  nullableMixedUnionDatum.selectBranch(1);
                  nullableMixedUnionDatum.value<int64_t>() =
                      nullableMixedUnionLong.value();
                } else if (nullableMixedUnionString.has_value()) {
                  nullableMixedUnionDatum.selectBranch(2);
                  nullableMixedUnionDatum.value<std::string>() =
                      nullableMixedUnionString.value();
                } else {
                  nullableMixedUnionDatum.selectBranch(3);
                }

                auto& simpleNestedRecordDatum = record.fieldAt(9);
                simpleNestedRecordDatum.selectBranch(0);
                auto& simpleNestedRecord =
                    simpleNestedRecordDatum.value<::avro::GenericRecord>();
                simpleNestedRecord.fieldAt(0).value<int32_t>() =
                    simpleNestedInt;
                simpleNestedRecord.fieldAt(1).value<std::string>() =
                    simpleNestedString;

                auto& nullableNestedRecordDatum = record.fieldAt(10);
                if (nullableNestedInt.has_value()) {
                  nullableNestedRecordDatum.selectBranch(1);
                  auto& nullableNestedRecord =
                      nullableNestedRecordDatum.value<::avro::GenericRecord>();
                  nullableNestedRecord.fieldAt(0).value<int32_t>() =
                      nullableNestedInt.value();
                  nullableNestedRecord.fieldAt(1).value<std::string>() =
                      nullableNestedString.value();
                } else {
                  nullableNestedRecordDatum.selectBranch(0);
                }

                writer.write(datum);
              };

          writeRow(
              std::string("alpha"),
              11,
              101,
              false,
              1.5,
              false,
              1001,
              false,
              2.5,
              false,
              7,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::string("mix-a"),
              1,
              std::string("simple-a"),
              2,
              std::string("nullable-a"));
          writeRow(
              std::string("beta"),
              std::nullopt,
              10000000000L,
              true,
              9.25,
              true,
              20000000000L,
              true,
              std::nullopt,
              true,
              std::nullopt,
              std::string("mix-b"),
              std::nullopt,
              std::nullopt,
              std::nullopt,
              3,
              std::string("simple-b"),
              std::nullopt,
              std::nullopt);
        });
    return filePath;
  }
};

TEST_F(AvroReaderTest, allTypesSchemaMapping) {
  const auto filePath = writeAllTypesRecord();
  auto reader = createReader(filePath);
  auto rowType = reader->rowType();

  ASSERT_EQ(rowType->size(), 24);
  EXPECT_EQ(rowType->nameOf(0), "nullCol");
  EXPECT_EQ(rowType->childAt(0)->kind(), TypeKind::UNKNOWN);
  EXPECT_EQ(rowType->childAt(1)->kind(), TypeKind::BOOLEAN);
  EXPECT_EQ(rowType->childAt(2)->kind(), TypeKind::INTEGER);
  EXPECT_EQ(rowType->childAt(3)->kind(), TypeKind::BIGINT);
  EXPECT_EQ(rowType->childAt(4)->kind(), TypeKind::REAL);
  EXPECT_EQ(rowType->childAt(5)->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(rowType->childAt(6)->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(rowType->childAt(7)->kind(), TypeKind::VARBINARY);
  EXPECT_EQ(rowType->childAt(8)->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(rowType->childAt(9)->kind(), TypeKind::VARBINARY);
  EXPECT_EQ(rowType->childAt(10)->kind(), TypeKind::ARRAY);
  EXPECT_EQ(rowType->childAt(11)->kind(), TypeKind::MAP);
  EXPECT_EQ(rowType->childAt(12)->kind(), TypeKind::ROW);
  EXPECT_EQ(rowType->childAt(13)->kind(), TypeKind::INTEGER);
  EXPECT_TRUE(rowType->childAt(14)->isDate());
  EXPECT_TRUE(rowType->childAt(15)->isTime());
  EXPECT_TRUE(rowType->childAt(16)->isTime());
  EXPECT_EQ(rowType->childAt(17)->kind(), TypeKind::TIMESTAMP);
  EXPECT_EQ(rowType->childAt(18)->kind(), TypeKind::TIMESTAMP);
  EXPECT_EQ(rowType->childAt(19)->kind(), TypeKind::TIMESTAMP);
  EXPECT_EQ(rowType->childAt(20)->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(rowType->childAt(21)->kind(), TypeKind::VARBINARY);
  ASSERT_TRUE(rowType->childAt(22)->isDecimal());
  const auto [bytesPrecision, bytesScale] =
      getDecimalPrecisionScale(*rowType->childAt(22));
  EXPECT_EQ(bytesPrecision, 9);
  EXPECT_EQ(bytesScale, 2);
  ASSERT_TRUE(rowType->childAt(23)->isDecimal());
  const auto [fixedPrecision, fixedScale] =
      getDecimalPrecisionScale(*rowType->childAt(23));
  EXPECT_EQ(fixedPrecision, 7);
  EXPECT_EQ(fixedScale, 3);
}

TEST_F(AvroReaderTest, unionMapping) {
  const auto filePath = writeUnionRecord();

  auto reader = createReader(filePath);
  auto rowType = reader->rowType();
  ASSERT_EQ(rowType->size(), 11);
  EXPECT_EQ(rowType->childAt(0)->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(rowType->childAt(1)->kind(), TypeKind::UNKNOWN);
  EXPECT_EQ(rowType->childAt(2)->kind(), TypeKind::INTEGER);
  EXPECT_EQ(rowType->childAt(3)->kind(), TypeKind::BIGINT);
  EXPECT_EQ(rowType->childAt(4)->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(rowType->childAt(5)->kind(), TypeKind::BIGINT);
  EXPECT_EQ(rowType->childAt(6)->kind(), TypeKind::DOUBLE);
  ASSERT_EQ(rowType->childAt(7)->kind(), TypeKind::ROW);
  const auto& mixedRow = rowType->childAt(7)->asRow();
  ASSERT_EQ(mixedRow.size(), 2);
  EXPECT_EQ(mixedRow.nameOf(0), "member0");
  EXPECT_EQ(mixedRow.nameOf(1), "member1");
  EXPECT_EQ(mixedRow.childAt(0)->kind(), TypeKind::INTEGER);
  EXPECT_EQ(mixedRow.childAt(1)->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(rowType->childAt(8)->kind(), TypeKind::ROW);
  const auto& nullableMixedUnion = rowType->childAt(8)->asRow();
  ASSERT_EQ(nullableMixedUnion.size(), 3);
  EXPECT_EQ(nullableMixedUnion.nameOf(0), "member0");
  EXPECT_EQ(nullableMixedUnion.nameOf(1), "member1");
  EXPECT_EQ(nullableMixedUnion.nameOf(2), "member2");
  EXPECT_EQ(nullableMixedUnion.childAt(0)->kind(), TypeKind::INTEGER);
  EXPECT_EQ(nullableMixedUnion.childAt(1)->kind(), TypeKind::BIGINT);
  EXPECT_EQ(nullableMixedUnion.childAt(2)->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(rowType->childAt(9)->kind(), TypeKind::ROW);
  const auto& simpleNestedRecord = rowType->childAt(9)->asRow();
  ASSERT_EQ(simpleNestedRecord.size(), 2);
  EXPECT_EQ(simpleNestedRecord.nameOf(0), "nestedInt");
  EXPECT_EQ(simpleNestedRecord.nameOf(1), "nestedString");
  EXPECT_EQ(simpleNestedRecord.childAt(0)->kind(), TypeKind::INTEGER);
  EXPECT_EQ(simpleNestedRecord.childAt(1)->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(rowType->childAt(10)->kind(), TypeKind::ROW);
  const auto& nullableNestedRecord = rowType->childAt(10)->asRow();
  ASSERT_EQ(nullableNestedRecord.size(), 2);
  EXPECT_EQ(nullableNestedRecord.nameOf(0), "nestedInt");
  EXPECT_EQ(nullableNestedRecord.nameOf(1), "nestedString");
  EXPECT_EQ(nullableNestedRecord.childAt(0)->kind(), TypeKind::INTEGER);
  EXPECT_EQ(nullableNestedRecord.childAt(1)->kind(), TypeKind::VARCHAR);
}

TEST_F(AvroReaderTest, lowerCaseFieldNames) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "CaseRecord",
      "fields": [
        {"name": "FooBar", "type": "int"},
        {"name": "Baz", "type": "string"}
      ]
    })JSON";
  auto filePath = writeAvroFile(
      schemaJson,
      [](auto& /*writer*/, const ::avro::ValidSchema& /*schema*/) {});

  dwio::common::ReaderOptions options{pool()};
  options.setFileColumnNamesReadAsLowerCase(true);
  auto reader = createReader(filePath, options);
  auto rowType = reader->rowType();
  ASSERT_EQ(rowType->size(), 2);
  EXPECT_EQ(rowType->nameOf(0), "foobar");
  EXPECT_EQ(rowType->nameOf(1), "baz");
}

TEST_F(AvroReaderTest, rejectsDuplicateFieldNames) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "DupRecord",
      "fields": [
        {"name": "Foo", "type": "int"},
        {"name": "foo", "type": "long"}
      ]
    })JSON";
  auto filePath = writeAvroFile(
      schemaJson,
      [](auto& /*writer*/, const ::avro::ValidSchema& /*schema*/) {});

  dwio::common::ReaderOptions options{pool()};
  options.setFileColumnNamesReadAsLowerCase(true);
  EXPECT_THROW(createReader(filePath, options), VeloxRuntimeError);
}

TEST_F(AvroReaderTest, rejectsUnsupportedLogicalType) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "LogicalRecord",
      "fields": [
        {"name": "badLogical", "type": {"type": "long", "logicalType": "local-timestamp-millis"}}
      ]
    })JSON";
  auto filePath = writeAvroFile(
      schemaJson,
      [](auto& /*writer*/, const ::avro::ValidSchema& /*schema*/) {});

  EXPECT_THROW(createReader(filePath), VeloxUserError);
}

TEST_F(AvroReaderTest, rejectsNonRecordRootSchema) {
  const std::string schemaJson = R"JSON("int")JSON";
  auto filePath = writeAvroFile(
      schemaJson,
      [](auto& /*writer*/, const ::avro::ValidSchema& /*schema*/) {});

  EXPECT_THROW(createReader(filePath), VeloxRuntimeError);
}

TEST_F(AvroReaderTest, usesSchemaOverride) {
  const std::string writerSchema = R"JSON(
    {
      "type": "record",
      "name": "WriterRecord",
      "fields": [
        {"name": "a", "type": "float"}
        {"name": "b", "type": "long"}
      ]
    })JSON";
  auto filePath = writeAvroFile(
      writerSchema, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        datum.value<::avro::GenericRecord>().fieldAt(0).value<float>() = 0.1f;
        datum.value<::avro::GenericRecord>().fieldAt(1).value<int64_t>() = 1;
        writer.write(datum);
      });

  const std::string readerSchema = R"JSON(
    {
      "type": "record",
      "name": "ReaderRecord",
      "fields": [
        {"name": "a", "type": "double"},
        {"name": "c", "type": "int", "default": 7}
      ]
    })JSON";
  dwio::common::SerDeOptions serdeOptions;
  serdeOptions.avroSchema = readerSchema;
  dwio::common::ReaderOptions options{pool()};
  options.setSerDeOptions(serdeOptions);

  auto reader = createReader(filePath, options);
  auto rowType = reader->rowType();
  ASSERT_EQ(rowType->size(), 2);
  EXPECT_EQ(rowType->nameOf(0), "a");
  EXPECT_EQ(rowType->childAt(0)->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(rowType->nameOf(1), "c");
  EXPECT_EQ(rowType->childAt(1)->kind(), TypeKind::INTEGER);
}

TEST_F(AvroReaderTest, rejectsInvalidSchemaOverride) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "SchemaOverrideRecord",
      "fields": [
        {"name": "a", "type": "int"}
      ]
    })JSON";
  auto filePath = writeAvroFile(
      schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        datum.value<::avro::GenericRecord>().fieldAt(0).value<int32_t>() = 1;
        writer.write(datum);
      });

  dwio::common::SerDeOptions serdeOptions;
  serdeOptions.avroSchema = R"JSON({not-valid-json)JSON";
  dwio::common::ReaderOptions options{pool()};
  options.setSerDeOptions(serdeOptions);
  EXPECT_THROW(createReader(filePath, options), VeloxUserError);
}

TEST_F(AvroReaderTest, complexNestedSchemaMapping) {
  auto filePath = writeComplexNestedRecord();

  auto reader = createReader(filePath);
  auto rowType = reader->rowType();
  ASSERT_EQ(rowType->size(), 2);
  ASSERT_EQ(rowType->childAt(0)->kind(), TypeKind::ROW);
  ASSERT_EQ(rowType->childAt(1)->kind(), TypeKind::ARRAY);

  const auto& metaRow = rowType->childAt(0)->asRow();
  ASSERT_EQ(metaRow.size(), 2);
  EXPECT_EQ(metaRow.childAt(0)->kind(), TypeKind::ARRAY);
  EXPECT_EQ(metaRow.childAt(1)->kind(), TypeKind::MAP);
  const auto& attrsMap = metaRow.childAt(1)->asMap();
  EXPECT_EQ(attrsMap.keyType()->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(attrsMap.valueType()->kind(), TypeKind::VARCHAR);

  const auto& payloadArray = rowType->childAt(1)->asArray();
  const auto& payloadRow = payloadArray.elementType()->asRow();
  ASSERT_EQ(payloadRow.size(), 2);
  EXPECT_EQ(payloadRow.childAt(0)->kind(), TypeKind::ARRAY);
  EXPECT_EQ(payloadRow.childAt(1)->kind(), TypeKind::MAP);

  const auto& flags = payloadRow.childAt(0)->asArray();
  EXPECT_EQ(flags.elementType()->kind(), TypeKind::BOOLEAN);
  const auto& propertyMap = payloadRow.childAt(1)->asMap();
  const auto& propertyRow = propertyMap.valueType()->asRow();
  ASSERT_EQ(propertyRow.size(), 2);
  EXPECT_EQ(propertyRow.childAt(0)->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(propertyRow.childAt(1)->kind(), TypeKind::ARRAY);
  const auto& propertyRowValue = propertyRow.childAt(1)->asArray();
  EXPECT_EQ(propertyRowValue.elementType()->kind(), TypeKind::INTEGER);
}

TEST_F(AvroReaderTest, readsAllTypesData) {
  const auto filePath = writeAllTypesRecord();
  auto reader = createReader(filePath);
  auto result = readRows(*reader, 5, 2);

  auto expected = makeRowVector({
      makeNullConstant(TypeKind::UNKNOWN, 2),
      makeFlatVector<bool>({true, false}),
      makeFlatVector<int32_t>({123, -456}),
      makeFlatVector<int64_t>({7890, -9876}),
      makeFlatVector<float>({1.5F, -2.5F}),
      makeFlatVector<double>({3.25, -4.75}),
      makeFlatVector<std::string>({"alpha", "beta"}),
      makeFlatVector<std::string>({"\x01\x02", "\x0A\x0B\x0C"}, VARBINARY()),
      makeFlatVector<std::string>({"RED", "GREEN"}),
      makeFlatVector<std::string>({"\xAA\xBB", "\xCC\xDD"}, VARBINARY()),
      makeArrayVector<int32_t>({{1, 2}, {3, 4}}),
      makeMapVector<std::string, int64_t>(
          {{{"a", 10}, {"b", 20}}, {{"c", -5}}}),
      makeRowVector(
          {makeFlatVector<int32_t>({101, 202}),
           makeFlatVector<std::string>({"sub-alpha", "sub-beta"})}),
      makeNullableFlatVector<int32_t>({42, std::nullopt}),
      makeFlatVector<int32_t>({1000, 2000}, DATE()),
      makeFlatVector<int64_t>({1234 * 1000L, 4321 * 1000L}, TIME()),
      makeFlatVector<int64_t>({5678, 8765}, TIME()),
      makeFlatVector<Timestamp>(
          {Timestamp::fromMillis(1700), Timestamp::fromMillis(2700)}),
      makeFlatVector<Timestamp>(
          {Timestamp::fromMicros(3500), Timestamp::fromMicros(4500)}),
      makeFlatVector<Timestamp>(
          {Timestamp::fromNanos(9876543210), Timestamp::fromNanos(1234567890)}),
      makeFlatVector<std::string>(
          {"123e4567-e89b-12d3-a456-426655440000",
           "00000000-0000-0000-0000-000000000000"}),
      makeFlatVector<std::string>(
          {std::string(
               "\x10\x11\x12\x13\x14\x15\x16\x17"
               "\x18\x19\x1A\x1B\x1C\x1D\x1E\x1F",
               16),
           std::string(
               "\xFF\xEE\xDD\xCC\xBB\xAA\x99\x88"
               "\x77\x66\x55\x44\x33\x22\x11\x00",
               16)},
          VARBINARY()),
      makeFlatVector<int64_t>({12345, -4200}, DECIMAL(9, 2)),
      makeFlatVector<int64_t>({6789, -1357}, DECIMAL(7, 3)),
  });
  assertEqualVectors(expected, result);
}

TEST_F(AvroReaderTest, readsComplexNestedData) {
  const auto filePath = writeComplexNestedRecord();
  auto reader = createReader(filePath);

  auto result = readRows(*reader, 1, 1);
  auto ids = makeArrayVector<int64_t>({{101, 202}});
  auto attrsKeys = makeFlatVector<std::string>({"alpha", "beta"});
  auto attrsValues =
      makeNullableFlatVector<std::string>({std::nullopt, "beta"});
  auto attrs = makeMapVector({0, 2}, attrsKeys, attrsValues);
  auto meta = makeRowVector({ids, attrs});

  auto flagsElements = makeNullableFlatVector<bool>({std::nullopt, true});
  auto flags = makeArrayVector({0, 2}, flagsElements);
  auto propertyKeys = makeFlatVector<std::string>({"k1"});
  auto propertyNames = makeFlatVector<std::string>({"p1"});
  auto propertyValuesElements = makeFlatVector<int32_t>({1, 2});
  auto propertyValues = makeArrayVector({0, 2}, propertyValuesElements);
  auto propertyRow = makeRowVector({propertyNames, propertyValues});
  auto properties = makeMapVector({0, 1}, propertyKeys, propertyRow);

  auto payload = makeRowVector({flags, properties});
  auto payloads = makeArrayVector({0, 1}, payload);
  auto expected = makeRowVector({meta, payloads});
  assertEqualVectors(expected, result);
}

TEST_F(AvroReaderTest, readsUnionData) {
  const auto filePath = writeUnionRecord();

  auto reader = createReader(filePath);
  auto result = readRows(*reader, 2, 2);

  auto mixedUnion = makeRowVector({
      makeNullableFlatVector<int32_t>({7, std::nullopt}),
      makeNullableFlatVector<std::string>({std::nullopt, "mix-b"}),
  });
  auto nullableMixedUnion = makeRowVector(
      {
          makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt}),
          makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}),
          makeNullableFlatVector<std::string>({"mix-a", std::nullopt}),
      },
      [](vector_size_t row) { return row == 1; });
  auto simpleNestedRecord = makeRowVector({
      makeFlatVector<int32_t>({1, 3}),
      makeFlatVector<std::string>({"simple-a", "simple-b"}),
  });
  auto nullableNestedRecord = makeRowVector(
      {
          makeNullableFlatVector<int32_t>({2, std::nullopt}),
          makeNullableFlatVector<std::string>({"nullable-a", std::nullopt}),
      },
      [](vector_size_t row) { return row == 1; });

  auto expected = makeRowVector({
      makeFlatVector<std::string>({"alpha", "beta"}),
      makeNullConstant(TypeKind::UNKNOWN, 2),
      makeNullableFlatVector<int32_t>({11, std::nullopt}),
      makeFlatVector<int64_t>({101L, 10000000000L}),
      makeFlatVector<double>({1.5, 9.25}),
      makeNullableFlatVector<int64_t>({1001, 20000000000L}),
      makeNullableFlatVector<double>({2.5, std::nullopt}),
      mixedUnion,
      nullableMixedUnion,
      simpleNestedRecord,
      nullableNestedRecord,
  });
  assertEqualVectors(expected, result);
}

TEST_F(AvroReaderTest, scanBatchBytesRespected) {
  const std::string schemaJson = R"JSON(
    {
      "type": "record",
      "name": "BatchRecord",
      "fields": [
        {"name": "index", "type": "int"}
      ]
    })JSON";
  auto filePath = writeAvroFile(
      schemaJson, [](auto& writer, const ::avro::ValidSchema& schema) {
        ::avro::GenericDatum datum(schema.root());
        auto& record = datum.value<::avro::GenericRecord>();
        for (int i = 0; i < 20; ++i) {
          record.fieldAt(0).value<int>() = i;
          writer.write(datum);
        }
      });

  auto reader = createReader(filePath);
  dwio::common::RowReaderOptions rowOptions;
  rowOptions.setSerdeParameters({{"avro.scan.batch.bytes", "1"}});
  auto rowReader = createRowReader(*reader, rowOptions);
  VectorPtr firstBatch;
  ASSERT_EQ(rowReader->next(10, firstBatch), 10);
  VectorPtr secondBatch;
  ASSERT_EQ(rowReader->next(10, secondBatch), 1);
  VectorPtr thirdBatch;
  ASSERT_EQ(rowReader->next(10, thirdBatch), 1);

  auto expected = makeRowVector({makeFlatVector<int>({11})});
  assertEqualVectors(expected, thirdBatch);
}
} // namespace
} // namespace facebook::velox::avro
