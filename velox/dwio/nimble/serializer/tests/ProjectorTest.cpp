/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Exceptions.h"

#include "folly/Random.h"
#include "folly/container/F14Set.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/Projector.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
#include "velox/type/Subfield.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;

namespace facebook::nimble::serde {

// Coalesces an IOBuf chain into a single string for test comparisons.
std::string toString(const folly::IOBuf& buf) {
  std::string result;
  result.reserve(buf.computeChainDataLength());
  for (auto range : buf) {
    result.append(reinterpret_cast<const char*>(range.data()), range.size());
  }
  return result;
}

bool outputRequiresNullBarrier(const folly::IOBuf& buf) {
  const auto serialized = toString(buf);
  const char* pos = serialized.data();
  const auto header =
      readSerializationHeader(pos, serialized.data() + serialized.size(), true);
  return header.flags.requiresNullBarrier;
}

// Test parameter: input serialization version and output (project) version.
struct FormatParam {
  SerializationVersion inputVersion;
  SerializationVersion projectVersion;
  EncodingType streamSizesEncodingType{EncodingType::Trivial};
  size_t bufferPoolCapacity{velox::BufferPool::kDefaultCapacity};

  std::string name() const {
    auto base =
        fmt::format("{}To{}", toString(inputVersion), toString(projectVersion));
    if (streamSizesEncodingType != EncodingType::Trivial) {
      base += fmt::format("_{}StreamSizes", toString(streamSizesEncodingType));
    }
    if (bufferPoolCapacity == 0) {
      base += "_NoBufferPool";
    } else if (bufferPoolCapacity != velox::BufferPool::kDefaultCapacity) {
      base += "_BufferPool" + std::to_string(bufferPoolCapacity);
    }
    return base;
  }
};

// Generate all legal input × output version combinations.
// Input is kSerialization (Serializer-written blobs); output is kProjection
// (the Projector's default writer version).
std::vector<FormatParam> allFormatCombinations() {
  return {
      {SerializationVersion::kSerialization, SerializationVersion::kProjection},
      // Non-default stream sizes encodings for projected output.
      {SerializationVersion::kSerialization,
       SerializationVersion::kProjection,
       EncodingType::Delta},
      {SerializationVersion::kSerialization,
       SerializationVersion::kProjection,
       EncodingType::Varint},
      // Buffer pool disabled variants.
      {SerializationVersion::kSerialization,
       SerializationVersion::kProjection,
       EncodingType::Trivial,
       /*bufferPoolCapacity=*/0},
      {SerializationVersion::kSerialization,
       SerializationVersion::kProjection,
       EncodingType::Delta,
       /*bufferPoolCapacity=*/0},
      {SerializationVersion::kSerialization,
       SerializationVersion::kProjection,
       EncodingType::Varint,
       /*bufferPoolCapacity=*/0},
      {SerializationVersion::kSerialization,
       SerializationVersion::kProjection,
       EncodingType::FixedBitWidth,
       /*bufferPoolCapacity=*/0},
  };
}

// Base test fixture with helper methods.
class ProjectorTestBase : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = memory::memoryManager()->addRootPool("projector_test_root");
    pool_ = memory::memoryManager()->addLeafPool("projector_test_leaf");
  }

  // Serialize a vector.
  std::string serialize(
      const VectorPtr& vec,
      const TypePtr& type,
      SerializerOptions options = SerializerOptions{}) {
    Serializer serializer{std::move(options), type, pool_.get()};
    auto sv = serializer.serialize(vec, OrderedRanges::of(0, vec->size()));
    return std::string(sv);
  }

  // Serialize a vector and return both data and schema.
  // Use this for FlatMap tests where keys are discovered during serialization.
  std::pair<std::string, std::shared_ptr<const nimble::Type>>
  serializeWithSchema(
      const VectorPtr& vec,
      const TypePtr& type,
      SerializerOptions options = SerializerOptions{}) {
    Serializer serializer{std::move(options), type, pool_.get()};
    auto sv = serializer.serialize(vec, OrderedRanges::of(0, vec->size()));
    auto schema =
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
    return {std::string(sv), schema};
  }

  // Get the nimble schema from a serializer (before serialization).
  // Note: For FlatMap, use serializeWithSchema() to get schema after
  // serialization when keys are discovered.
  std::shared_ptr<const nimble::Type> getNimbleSchema(
      const TypePtr& type,
      SerializerOptions options = SerializerOptions{}) {
    Serializer serializer{std::move(options), type, pool_.get()};
    return SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  }

  // Deserialize a projected buffer with the projected schema.
  VectorPtr deserialize(
      std::string_view data,
      std::shared_ptr<const nimble::Type> nimbleSchema,
      DeserializerOptions options = DeserializerOptions{}) {
    Deserializer deserializer{nimbleSchema, pool_.get(), options};
    VectorPtr output;
    deserializer.deserialize({data}, output);
    return output;
  }

  // Create a simple row vector with scalar columns.
  RowVectorPtr makeSimpleRowVector(
      const std::vector<std::string>& names,
      const std::vector<VectorPtr>& children) {
    return std::make_shared<RowVector>(
        pool_.get(),
        ROW(std::vector<std::string>(names), extractTypes(children)),
        nullptr,
        children[0]->size(),
        children);
  }

  // Extract types from vectors.
  std::vector<TypePtr> extractTypes(const std::vector<VectorPtr>& vectors) {
    std::vector<TypePtr> types;
    types.reserve(vectors.size());
    for (const auto& v : vectors) {
      types.push_back(v->type());
    }
    return types;
  }

  // Create a flat vector of integers.
  template <typename T>
  FlatVectorPtr<T> makeIntVector(const std::vector<T>& values) {
    auto vector = BaseVector::create<FlatVector<T>>(
        CppToType<T>::create(), values.size(), pool_.get());
    for (size_t i = 0; i < values.size(); ++i) {
      vector->set(i, values[i]);
    }
    return vector;
  }

  // Create a flat vector of strings.
  FlatVectorPtr<StringView> makeStringVector(
      const std::vector<std::string>& values) {
    auto vector = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), values.size(), pool_.get());
    for (size_t i = 0; i < values.size(); ++i) {
      vector->set(i, StringView(values[i]));
    }
    return vector;
  }

  // Projects using either string_view or IOBuf API based on useIOBuf flag.
  static folly::IOBuf projectInput(
      const Projector& projector,
      const std::string& serialized,
      bool useIOBuf) {
    if (useIOBuf) {
      auto buf =
          folly::IOBuf::wrapBufferAsValue(serialized.data(), serialized.size());
      return projector.project(buf);
    }
    return projector.project(std::string_view(serialized));
  }

  // Verifies that buildProjectedNimbleType produces a schema consistent
  // with the projector's schema (from buildProjectedNimbleType). For
  // non-FlatMap and FlatMap key-level projections, schemas should match and
  // both should decode correctly. For full FlatMap column projections (without
  // key subscripts), buildProjectedNimbleType creates a regular Map instead
  // of FlatMap (because key names aren't available from velox types), so we
  // only verify the schema type mismatch.
  void verifyProjectedSchema(
      const TypePtr& veloxType,
      const std::vector<common::Subfield>& subfields,
      const std::shared_ptr<const nimble::Type>& projectorSchema,
      std::string_view projectedData,
      DeserializerOptions deserOptions,
      const nimble::ColumnEncodings& columnEncodings = {}) {
    auto convertSchema = nimble::buildProjectedNimbleType(
        veloxType->asRow(), subfields, columnEncodings);

    ASSERT_TRUE(convertSchema->isRow());
    ASSERT_TRUE(projectorSchema->isRow());
    const auto& convertRow = convertSchema->asRow();
    const auto& projRow = projectorSchema->asRow();
    ASSERT_EQ(convertRow.childrenCount(), projRow.childrenCount());

    // Check if any projected child is FlatMap (schema type mismatch expected).
    bool hasFlatMapMismatch = false;
    for (size_t i = 0; i < projRow.childrenCount(); ++i) {
      EXPECT_EQ(convertRow.nameAt(i), projRow.nameAt(i));
      if (projRow.childAt(i)->isFlatMap() && convertRow.childAt(i)->isMap()) {
        hasFlatMapMismatch = true;
      }
    }

    if (hasFlatMapMismatch) {
      // buildProjectedNimbleType can't produce FlatMap without key names.
      // Only verify with projector schema.
      Deserializer deserializer(projectorSchema, pool_.get(), deserOptions);
      VectorPtr output;
      deserializer.deserialize({projectedData}, output);
      ASSERT_NE(output, nullptr);
      ASSERT_GT(output->size(), 0);
      return;
    }

    // Schemas should match — verify both decode the same data.
    VectorPtr projResult;
    {
      Deserializer deserializer(projectorSchema, pool_.get(), deserOptions);
      deserializer.deserialize({projectedData}, projResult);
      ASSERT_NE(projResult, nullptr);
    }

    VectorPtr convResult;
    {
      Deserializer deserializer(convertSchema, pool_.get(), deserOptions);
      deserializer.deserialize({projectedData}, convResult);
      ASSERT_NE(convResult, nullptr);
    }

    ASSERT_EQ(projResult->size(), convResult->size());
    for (vector_size_t i = 0; i < projResult->size(); ++i) {
      EXPECT_TRUE(projResult->equalValueAt(convResult.get(), i, i))
          << "Schema decode mismatch at row " << i;
    }
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

// Helper to create a vector of subfields.
std::vector<common::Subfield> makeSubfields(
    std::initializer_list<const char*> paths) {
  std::vector<common::Subfield> subfields;
  subfields.reserve(paths.size());
  for (const auto* path : paths) {
    subfields.emplace_back(path);
  }
  return subfields;
}

// Parameterized test fixture for format combinations.
class ProjectorFormatTest : public ProjectorTestBase,
                            public ::testing::WithParamInterface<FormatParam> {
 protected:
  // Get serializer options for input format.
  SerializerOptions inputSerializerOptions() const {
    return SerializerOptions{.version = GetParam().inputVersion};
  }

  // Get projector options for output format.
  Projector::Options projectorOptions() const {
    return Projector::Options{
        .projectVersion = GetParam().projectVersion,
        .streamIndicesEncodingType = GetParam().streamSizesEncodingType,
        .streamSizesEncodingType = GetParam().streamSizesEncodingType};
  }

  // Get deserializer options for output format.
  DeserializerOptions outputDeserializerOptions() const {
    return {
        .hasHeader = true,
        .bufferPoolCapacity = GetParam().bufferPoolCapacity,
    };
  }
};

// Test projecting a single column from a multi-column row.
TEST_P(ProjectorFormatTest, projectSingleColumn) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
      {"c", VARCHAR()},
  });

  auto vec = makeSimpleRowVector(
      {"a", "b", "c"},
      {
          makeIntVector<int32_t>({1, 2, 3}),
          makeIntVector<int64_t>({100, 200, 300}),
          makeStringVector({"x", "y", "z"}),
      });

  auto serialized = serialize(vec, type, inputSerializerOptions());
  auto inputSchema = getNimbleSchema(type, inputSerializerOptions());

  // Project only column "b".
  auto subfields = makeSubfields({"b"});
  Projector projector(inputSchema, subfields, pool_.get(), projectorOptions());
  auto outputSchema = projector.projectedSchema();

  // Verify output schema has only one column.
  ASSERT_TRUE(outputSchema->isRow());
  ASSERT_EQ(outputSchema->asRow().childrenCount(), 1);
  ASSERT_EQ(outputSchema->asRow().nameAt(0), "b");

  // Helper to verify result.
  auto verifyResult = [&](std::string_view projected) {
    auto result =
        deserialize(projected, outputSchema, outputDeserializerOptions());
    ASSERT_EQ(result->size(), 3);

    auto resultRow = result->as<RowVector>();
    auto bCol = resultRow->childAt(0)->as<FlatVector<int64_t>>();
    EXPECT_EQ(bCol->valueAt(0), 100);
    EXPECT_EQ(bCol->valueAt(1), 200);
    EXPECT_EQ(bCol->valueAt(2), 300);
  };

  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    auto projectedStr = toString(projected);
    verifyResult(projectedStr);
    verifyProjectedSchema(
        type,
        subfields,
        outputSchema,
        projectedStr,
        outputDeserializerOptions());
  }
}

TEST_F(ProjectorTestBase, projectsInputWithoutStreamVarintRowCountFlag) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
  });
  auto vec = makeSimpleRowVector(
      {"a", "b"},
      {
          makeIntVector<int32_t>({1, 2, 3}),
          makeIntVector<int64_t>({10, 20, 30}),
      });

  SerializerOptions serializerOptions{
      .version = SerializationVersion::kSerialization};
  auto serialized = serialize(vec, type, serializerOptions);
  auto inputSchema = getNimbleSchema(type, serializerOptions);
  serialized[sizeof(uint8_t) + varint::varintSize(vec->size())] =
      static_cast<char>(facebook::nimble::serde::detail::makeFlagsByte(
          /*requiresNullBarrier=*/false,
          /*streamEncodingUsesVarintRowCount=*/false,
          /*streamHasChunkHeader=*/false));

  Projector projector{
      inputSchema,
      makeSubfields({"b"}),
      pool_.get(),
      Projector::Options{.projectVersion = SerializationVersion::kProjection}};
  const auto outputSchema = projector.projectedSchema();

  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    auto projectedStr = toString(projected);

    const char* pos = projectedStr.data();
    const auto header = readSerializationHeader(
        pos, projectedStr.data() + projectedStr.size(), true);
    EXPECT_TRUE(header.flags.streamEncodingUsesVarintRowCount);

    auto result = deserialize(
        projectedStr, outputSchema, DeserializerOptions{.hasHeader = true});
    ASSERT_EQ(result->size(), 3);
    auto resultRow = result->as<RowVector>();
    auto bCol = resultRow->childAt(0)->as<FlatVector<int64_t>>();
    EXPECT_EQ(bCol->valueAt(0), 10);
    EXPECT_EQ(bCol->valueAt(1), 20);
    EXPECT_EQ(bCol->valueAt(2), 30);
  }
}

// A Projector predating kStreamVarintRowCountFlag leaves the bit clear even
// though its stream bodies are varint-encoded. Readers must not infer fixed
// u32 row counts from the cleared bit, or every encoding prefix desyncs.
TEST_F(
    ProjectorTestBase,
    deserializesProjectionWithoutStreamVarintRowCountFlag) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
  });
  auto vec = makeSimpleRowVector(
      {"a", "b"},
      {
          makeIntVector<int32_t>({1, 2, 3}),
          makeIntVector<int64_t>({10, 20, 30}),
      });

  SerializerOptions serializerOptions{
      .version = SerializationVersion::kSerialization};
  auto serialized = serialize(vec, type, serializerOptions);
  auto inputSchema = getNimbleSchema(type, serializerOptions);

  Projector projector{
      inputSchema,
      makeSubfields({"b"}),
      pool_.get(),
      Projector::Options{.projectVersion = SerializationVersion::kProjection}};
  const auto outputSchema = projector.projectedSchema();

  auto projected = projectInput(projector, serialized, /*useIOBuf=*/false);
  auto projectedStr = toString(projected);
  projectedStr[sizeof(uint8_t) + varint::varintSize(vec->size())] =
      static_cast<char>(facebook::nimble::serde::detail::makeFlagsByte(
          /*requiresNullBarrier=*/false,
          /*streamEncodingUsesVarintRowCount=*/false,
          /*streamHasChunkHeader=*/false));

  auto result = deserialize(
      projectedStr, outputSchema, DeserializerOptions{.hasHeader = true});
  ASSERT_EQ(result->size(), 3);
  auto bCol = result->as<RowVector>()->childAt(0)->as<FlatVector<int64_t>>();
  EXPECT_EQ(bCol->valueAt(0), 10);
  EXPECT_EQ(bCol->valueAt(1), 20);
  EXPECT_EQ(bCol->valueAt(2), 30);
}

// Test projecting multiple columns.
TEST_P(ProjectorFormatTest, projectMultipleColumns) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
      {"c", VARCHAR()},
      {"d", DOUBLE()},
  });

  auto vec = makeSimpleRowVector(
      {"a", "b", "c", "d"},
      {
          makeIntVector<int32_t>({1, 2}),
          makeIntVector<int64_t>({100, 200}),
          makeStringVector({"x", "y"}),
          BaseVector::create<FlatVector<double>>(DOUBLE(), 2, pool_.get()),
      });
  vec->childAt(3)->as<FlatVector<double>>()->set(0, 1.5);
  vec->childAt(3)->as<FlatVector<double>>()->set(1, 2.5);

  auto serialized = serialize(vec, type, inputSerializerOptions());
  auto inputSchema = getNimbleSchema(type, inputSerializerOptions());

  // Project columns "a" and "c".
  auto subfields = makeSubfields({"a", "c"});
  Projector projector(inputSchema, subfields, pool_.get(), projectorOptions());
  auto outputSchema = projector.projectedSchema();

  ASSERT_EQ(outputSchema->asRow().childrenCount(), 2);
  ASSERT_EQ(outputSchema->asRow().nameAt(0), "a");
  ASSERT_EQ(outputSchema->asRow().nameAt(1), "c");

  // Helper to verify result.
  auto verifyResult = [&](std::string_view projected) {
    auto result =
        deserialize(projected, outputSchema, outputDeserializerOptions());
    ASSERT_EQ(result->size(), 2);

    auto resultRow = result->as<RowVector>();
    auto aCol = resultRow->childAt(0)->as<FlatVector<int32_t>>();
    EXPECT_EQ(aCol->valueAt(0), 1);
    EXPECT_EQ(aCol->valueAt(1), 2);

    auto cCol = resultRow->childAt(1)->as<FlatVector<StringView>>();
    EXPECT_EQ(cCol->valueAt(0).str(), "x");
    EXPECT_EQ(cCol->valueAt(1).str(), "y");
  };

  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    auto projectedStr = toString(projected);
    verifyResult(projectedStr);
    verifyProjectedSchema(
        type,
        subfields,
        outputSchema,
        projectedStr,
        outputDeserializerOptions());
  }
}

// Test projecting nested struct fields.
TEST_P(ProjectorFormatTest, projectNestedField) {
  auto type = ROW({
      {"outer",
       ROW({
           {"inner1", INTEGER()},
           {"inner2", VARCHAR()},
       })},
      {"other", BIGINT()},
  });

  auto innerVec = makeSimpleRowVector(
      {"inner1", "inner2"},
      {
          makeIntVector<int32_t>({10, 20}),
          makeStringVector({"a", "b"}),
      });

  auto vec = makeSimpleRowVector(
      {"outer", "other"},
      {
          innerVec,
          makeIntVector<int64_t>({100, 200}),
      });

  auto serialized = serialize(vec, type, inputSerializerOptions());
  auto inputSchema = getNimbleSchema(type, inputSerializerOptions());

  // Project only "outer.inner1".
  auto subfields = makeSubfields({"outer.inner1"});
  Projector projector(inputSchema, subfields, pool_.get(), projectorOptions());
  auto outputSchema = projector.projectedSchema();

  // Output schema: ROW { outer: ROW { inner1: INTEGER } }
  ASSERT_TRUE(outputSchema->isRow());
  ASSERT_EQ(outputSchema->asRow().childrenCount(), 1);
  ASSERT_EQ(outputSchema->asRow().nameAt(0), "outer");

  auto outerType = outputSchema->asRow().childAt(0);
  ASSERT_TRUE(outerType->isRow());
  ASSERT_EQ(outerType->asRow().childrenCount(), 1);
  ASSERT_EQ(outerType->asRow().nameAt(0), "inner1");

  // Helper to verify result.
  auto verifyResult = [&](std::string_view projected) {
    auto result =
        deserialize(projected, outputSchema, outputDeserializerOptions());
    ASSERT_EQ(result->size(), 2);

    auto resultRow = result->as<RowVector>();
    auto outerRow = resultRow->childAt(0)->as<RowVector>();
    auto inner1Col = outerRow->childAt(0)->as<FlatVector<int32_t>>();
    EXPECT_EQ(inner1Col->valueAt(0), 10);
    EXPECT_EQ(inner1Col->valueAt(1), 20);
  };

  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    auto projectedStr = toString(projected);
    verifyResult(projectedStr);
    verifyProjectedSchema(
        type,
        subfields,
        outputSchema,
        projectedStr,
        outputDeserializerOptions());
  }
}

// Test projecting with array type (entire array column).
TEST_P(ProjectorFormatTest, projectArrayColumn) {
  auto type = ROW({
      {"arr", ARRAY(INTEGER())},
      {"other", BIGINT()},
  });

  auto arrElements = makeIntVector<int32_t>({1, 2, 3, 4, 5, 6});
  auto arrOffsets = allocateOffsets(2, pool_.get());
  auto arrSizes = allocateSizes(2, pool_.get());
  auto rawOffsets = arrOffsets->asMutable<vector_size_t>();
  auto rawSizes = arrSizes->asMutable<vector_size_t>();
  rawOffsets[0] = 0;
  rawSizes[0] = 3; // [1, 2, 3]
  rawOffsets[1] = 3;
  rawSizes[1] = 3; // [4, 5, 6]

  auto arrVec = std::make_shared<ArrayVector>(
      pool_.get(),
      ARRAY(INTEGER()),
      nullptr,
      2,
      arrOffsets,
      arrSizes,
      arrElements);

  auto vec = makeSimpleRowVector(
      {"arr", "other"},
      {
          arrVec,
          makeIntVector<int64_t>({100, 200}),
      });

  auto serialized = serialize(vec, type, inputSerializerOptions());
  auto inputSchema = getNimbleSchema(type, inputSerializerOptions());

  // Project only "arr".
  auto subfields = makeSubfields({"arr"});
  Projector projector(inputSchema, subfields, pool_.get(), projectorOptions());
  auto outputSchema = projector.projectedSchema();

  ASSERT_EQ(outputSchema->asRow().childrenCount(), 1);
  ASSERT_EQ(outputSchema->asRow().nameAt(0), "arr");

  // Helper to verify result.
  auto verifyResult = [&](std::string_view projected) {
    auto result =
        deserialize(projected, outputSchema, outputDeserializerOptions());
    ASSERT_EQ(result->size(), 2);

    auto resultRow = result->as<RowVector>();
    auto arrResult = resultRow->childAt(0)->as<ArrayVector>();
    EXPECT_EQ(arrResult->sizeAt(0), 3);
    EXPECT_EQ(arrResult->sizeAt(1), 3);
  };

  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    auto projectedStr = toString(projected);
    verifyResult(projectedStr);
    verifyProjectedSchema(
        type,
        subfields,
        outputSchema,
        projectedStr,
        outputDeserializerOptions());
  }
}

// Test empty input (0 rows).
TEST_P(ProjectorFormatTest, emptyInput) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
  });

  auto vec = makeSimpleRowVector(
      {"a", "b"},
      {
          makeIntVector<int32_t>({}),
          makeIntVector<int64_t>({}),
      });

  auto serialized = serialize(vec, type, inputSerializerOptions());
  auto inputSchema = getNimbleSchema(type, inputSerializerOptions());

  auto subfields = makeSubfields({"a"});
  Projector projector(inputSchema, subfields, pool_.get(), projectorOptions());
  auto outputSchema = projector.projectedSchema();

  // Helper to verify result.
  auto verifyResult = [&](std::string_view projected) {
    auto result =
        deserialize(projected, outputSchema, outputDeserializerOptions());
    EXPECT_EQ(result->size(), 0);
  };

  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    auto projectedStr = toString(projected);
    verifyResult(projectedStr);
    verifyProjectedSchema(
        type,
        subfields,
        outputSchema,
        projectedStr,
        outputDeserializerOptions());
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllFormats,
    ProjectorFormatTest,
    ::testing::ValuesIn(allFormatCombinations()),
    [](const ::testing::TestParamInfo<FormatParam>& info) {
      return info.param.name();
    });

// Non-parameterized tests for special cases.
class ProjectorTest : public ProjectorTestBase {};

TEST_F(ProjectorTest, columnProjectionDeserializerBuildsProjectedType) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
      {"c", VARCHAR()},
  });

  auto vec = makeSimpleRowVector(
      {"a", "b", "c"},
      {
          makeIntVector<int32_t>({1, 2, 3}),
          makeIntVector<int64_t>({100, 200, 300}),
          makeStringVector({"x", "y", "z"}),
      });

  auto serialized =
      serialize(vec, type, {.version = SerializationVersion::kSerialization});
  auto inputSchema =
      getNimbleSchema(type, {.version = SerializationVersion::kSerialization});

  Deserializer deserializer{
      inputSchema, makeSubfields({"b"}), pool_.get(), {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), 3);
  auto* row = output->as<RowVector>();
  ASSERT_NE(row, nullptr);
  EXPECT_TRUE(output->type()->equivalent(*ROW({{"b", BIGINT()}})));
  ASSERT_EQ(row->children().size(), 1);

  auto* b = row->childAt(0)->as<FlatVector<int64_t>>();
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->valueAt(0), 100);
  EXPECT_EQ(b->valueAt(1), 200);
  EXPECT_EQ(b->valueAt(2), 300);
}

TEST_F(ProjectorTest, columnProjectionDeserializerCanBeReused) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
  });

  auto inputSchema =
      getNimbleSchema(type, {.version = SerializationVersion::kSerialization});
  Deserializer deserializer{
      inputSchema, makeSubfields({"b"}), pool_.get(), {.hasHeader = true}};

  auto makeInput = [&](const std::vector<int32_t>& aValues,
                       const std::vector<int64_t>& bValues) {
    return serialize(
        makeSimpleRowVector(
            {"a", "b"},
            {
                makeIntVector<int32_t>(aValues),
                makeIntVector<int64_t>(bValues),
            }),
        type,
        {.version = SerializationVersion::kSerialization});
  };

  VectorPtr firstOutput;
  deserializer.deserialize(makeInput({1, 2}, {100, 200}), firstOutput);
  auto* firstRow = firstOutput->as<RowVector>();
  ASSERT_NE(firstRow, nullptr);
  EXPECT_TRUE(firstOutput->type()->equivalent(*ROW({{"b", BIGINT()}})));
  auto* firstB = firstRow->childAt(0)->as<FlatVector<int64_t>>();
  ASSERT_NE(firstB, nullptr);
  EXPECT_EQ(firstB->valueAt(0), 100);
  EXPECT_EQ(firstB->valueAt(1), 200);

  VectorPtr secondOutput;
  deserializer.deserialize(makeInput({3, 4, 5}, {300, 400, 500}), secondOutput);
  auto* secondRow = secondOutput->as<RowVector>();
  ASSERT_NE(secondRow, nullptr);
  EXPECT_TRUE(secondOutput->type()->equivalent(*ROW({{"b", BIGINT()}})));
  auto* secondB = secondRow->childAt(0)->as<FlatVector<int64_t>>();
  ASSERT_NE(secondB, nullptr);
  EXPECT_EQ(secondB->valueAt(0), 300);
  EXPECT_EQ(secondB->valueAt(1), 400);
  EXPECT_EQ(secondB->valueAt(2), 500);
}

TEST_F(ProjectorTest, columnProjectionDeserializerProjectsNestedFields) {
  auto type = ROW({
      {"outer",
       ROW({
           {"inner1", INTEGER()},
           {"inner2", VARCHAR()},
       })},
      {"other", BIGINT()},
  });

  auto outer = makeSimpleRowVector(
      {"inner1", "inner2"},
      {
          makeIntVector<int32_t>({10, 20}),
          makeStringVector({"a", "b"}),
      });
  auto vec = makeSimpleRowVector(
      {"outer", "other"},
      {
          outer,
          makeIntVector<int64_t>({100, 200}),
      });

  auto serialized =
      serialize(vec, type, {.version = SerializationVersion::kSerialization});
  auto inputSchema =
      getNimbleSchema(type, {.version = SerializationVersion::kSerialization});

  Deserializer deserializer{
      inputSchema,
      makeSubfields({"outer.inner1"}),
      pool_.get(),
      {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), 2);
  auto* row = output->as<RowVector>();
  ASSERT_NE(row, nullptr);
  EXPECT_TRUE(output->type()->equivalent(
      *ROW({{"outer", ROW({{"inner1", INTEGER()}})}})));
  ASSERT_EQ(row->children().size(), 1);

  auto* outerRow = row->childAt(0)->as<RowVector>();
  ASSERT_NE(outerRow, nullptr);
  ASSERT_EQ(outerRow->children().size(), 1);

  auto* inner1 = outerRow->childAt(0)->as<FlatVector<int32_t>>();
  ASSERT_NE(inner1, nullptr);
  EXPECT_EQ(inner1->valueAt(0), 10);
  EXPECT_EQ(inner1->valueAt(1), 20);
}

TEST_F(
    ProjectorTest,
    columnProjectionDeserializerEmptySelectionReadsAllFields) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
  });

  auto vec = makeSimpleRowVector(
      {"a", "b"},
      {
          makeIntVector<int32_t>({1, 2, 3}),
          makeIntVector<int64_t>({100, 200, 300}),
      });

  auto serialized =
      serialize(vec, type, {.version = SerializationVersion::kSerialization});
  auto inputSchema =
      getNimbleSchema(type, {.version = SerializationVersion::kSerialization});

  Deserializer deserializer{inputSchema, {}, pool_.get(), {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), 3);
  auto* row = output->as<RowVector>();
  ASSERT_NE(row, nullptr);
  ASSERT_EQ(row->children().size(), 2);

  auto* a = row->childAt(0)->as<FlatVector<int32_t>>();
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->valueAt(0), 1);
  EXPECT_EQ(a->valueAt(1), 2);
  EXPECT_EQ(a->valueAt(2), 3);

  auto* b = row->childAt(1)->as<FlatVector<int64_t>>();
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->valueAt(0), 100);
  EXPECT_EQ(b->valueAt(1), 200);
  EXPECT_EQ(b->valueAt(2), 300);
}

TEST_F(ProjectorTest, columnProjectionDeserializerSelectsAllFieldsExplicitly) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
  });
  auto vec = makeSimpleRowVector(
      {"a", "b"},
      {
          makeIntVector<int32_t>({1, 2, 3}),
          makeIntVector<int64_t>({100, 200, 300}),
      });
  auto serialized =
      serialize(vec, type, {.version = SerializationVersion::kSerialization});
  auto inputSchema =
      getNimbleSchema(type, {.version = SerializationVersion::kSerialization});

  Deserializer deserializer{
      inputSchema, makeSubfields({"a", "b"}), pool_.get(), {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(output->type()->equivalent(*type));
  EXPECT_TRUE(output->equalValueAt(vec.get(), 0, 0));
  EXPECT_TRUE(output->equalValueAt(vec.get(), 1, 1));
  EXPECT_TRUE(output->equalValueAt(vec.get(), 2, 2));
}

TEST_F(ProjectorTest, columnProjectionDeserializerOrdersFieldsByName) {
  auto type = ROW({
      {"c", VARCHAR()},
      {"a", INTEGER()},
      {"b", BIGINT()},
  });
  auto vec = makeSimpleRowVector(
      {"c", "a", "b"},
      {
          makeStringVector({"x", "y"}),
          makeIntVector<int32_t>({1, 2}),
          makeIntVector<int64_t>({100, 200}),
      });
  auto serialized =
      serialize(vec, type, {.version = SerializationVersion::kSerialization});
  auto inputSchema =
      getNimbleSchema(type, {.version = SerializationVersion::kSerialization});

  Deserializer deserializer{
      inputSchema, makeSubfields({"c", "a"}), pool_.get(), {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(serialized, output);

  EXPECT_TRUE(
      output->type()->equivalent(*ROW({{"a", INTEGER()}, {"c", VARCHAR()}})));
  auto* row = output->asChecked<RowVector>();
  EXPECT_EQ(row->childAt(0)->asChecked<FlatVector<int32_t>>()->valueAt(0), 1);
  EXPECT_EQ(
      row->childAt(1)->asChecked<FlatVector<StringView>>()->valueAt(0), "x");
}

TEST_F(ProjectorTest, columnProjectionDeserializerProjectsNestedScenarios) {
  auto type = ROW({
      {"z",
       ROW({
           {"value", VARCHAR()},
           {"nested", ROW({{"right", BIGINT()}, {"left", INTEGER()}})},
       })},
      {"a", ROW({{"second", VARCHAR()}, {"first", INTEGER()}})},
      {"ignored", BIGINT()},
  });
  auto nested = makeSimpleRowVector(
      {"right", "left"},
      {
          makeIntVector<int64_t>({100, 200}),
          makeIntVector<int32_t>({10, 20}),
      });
  auto z = makeSimpleRowVector(
      {"value", "nested"}, {makeStringVector({"x", "y"}), nested});
  auto a = makeSimpleRowVector(
      {"second", "first"},
      {makeStringVector({"p", "q"}), makeIntVector<int32_t>({1, 2})});
  auto input = makeSimpleRowVector(
      {"z", "a", "ignored"}, {z, a, makeIntVector<int64_t>({15, 25})});
  auto serialized =
      serialize(input, type, {.version = SerializationVersion::kSerialization});
  auto inputSchema =
      getNimbleSchema(type, {.version = SerializationVersion::kSerialization});

  Deserializer deserializer{
      inputSchema,
      makeSubfields({"z.nested.right", "z.nested.left", "a.first"}),
      pool_.get(),
      {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(serialized, output);

  auto expectedType = ROW({
      {"a", ROW({{"first", INTEGER()}})},
      {"z", ROW({{"nested", ROW({{"left", INTEGER()}, {"right", BIGINT()}})}})},
  });
  EXPECT_TRUE(output->type()->equivalent(*expectedType));
  auto* outputRow = output->asChecked<RowVector>();
  auto* outputA = outputRow->childAt(0)->asChecked<RowVector>();
  EXPECT_EQ(
      outputA->childAt(0)->asChecked<FlatVector<int32_t>>()->valueAt(1), 2);
  auto* outputNested = outputRow->childAt(1)
                           ->asChecked<RowVector>()
                           ->childAt(0)
                           ->asChecked<RowVector>();
  EXPECT_EQ(
      outputNested->childAt(0)->asChecked<FlatVector<int32_t>>()->valueAt(0),
      10);
  EXPECT_EQ(
      outputNested->childAt(1)->asChecked<FlatVector<int64_t>>()->valueAt(1),
      200);

  Deserializer wholeFieldDeserializer{
      inputSchema,
      makeSubfields({"z.nested.left", "z"}),
      pool_.get(),
      {.hasHeader = true}};
  wholeFieldDeserializer.deserialize(serialized, output);
  EXPECT_TRUE(output->type()->equivalent(*ROW({{"z", type->childAt(0)}})));
  for (vector_size_t row = 0; row < input->size(); ++row) {
    EXPECT_TRUE(output->asChecked<RowVector>()->childAt(0)->equalValueAt(
        input->asChecked<RowVector>()->childAt(0).get(), row, row));
  }
}

TEST_F(ProjectorTest, columnProjectionDeserializerRejectsDuplicateSubfields) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
  });
  auto inputSchema =
      getNimbleSchema(type, {.version = SerializationVersion::kSerialization});

  NIMBLE_ASSERT_THROW(
      Deserializer(inputSchema, makeSubfields({"a", "a"}), pool_.get(), {}),
      "Duplicate column projection subfield");
}

TEST_F(ProjectorTest, columnProjectionDeserializerNestedFuzzer) {
  constexpr vector_size_t kBatchSize = 32;
  constexpr int kIterations = 10;
  auto type = ROW({
      {"z",
       ROW({
           {"y", VARCHAR()},
           {"x", ROW({{"b", BIGINT()}, {"a", INTEGER()}})},
       })},
      {"a", ROW({{"q", DOUBLE()}, {"p", BOOLEAN()}})},
      {"m", BIGINT()},
      {"unused", VARCHAR()},
  });
  auto inputSchema =
      getNimbleSchema(type, {.version = SerializationVersion::kSerialization});
  Deserializer deserializer{
      inputSchema,
      makeSubfields({"z.y", "z.x.b", "z.x.a", "a", "m"}),
      pool_.get(),
      {.hasHeader = true}};
  auto expectedType = ROW({
      {"a", type->childAt(1)},
      {"m", BIGINT()},
      {"z",
       ROW({
           {"x", ROW({{"a", INTEGER()}, {"b", BIGINT()}})},
           {"y", VARCHAR()},
       })},
  });

  const auto seed = folly::Random::rand32();
  LOG(INFO) << "columnProjectionDeserializerNestedFuzzer seed: " << seed;
  VectorFuzzer fuzzer(
      {.vectorSize = kBatchSize, .nullRatio = 0.2}, pool_.get(), seed);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(fmt::format("seed={} iteration={}", seed, iteration));
    std::vector<VectorPtr> inputChildren;
    inputChildren.reserve(type->size());
    for (const auto& child : type->children()) {
      inputChildren.push_back(fuzzer.fuzzFlat(child, kBatchSize));
    }
    auto input = std::make_shared<RowVector>(
        pool_.get(), type, nullptr, kBatchSize, std::move(inputChildren));
    auto serialized = serialize(
        input, type, {.version = SerializationVersion::kSerialization});
    VectorPtr output;
    deserializer.deserialize(serialized, output);

    EXPECT_TRUE(output->type()->equivalent(*expectedType));
    auto* inputRow = input->asChecked<RowVector>();
    auto* inputZ = inputRow->childAt(0)->asChecked<RowVector>();
    auto* inputX = inputZ->childAt(1)->asChecked<RowVector>();
    auto expectedX = std::make_shared<RowVector>(
        pool_.get(),
        expectedType->childAt(2)->asRow().childAt(0),
        inputX->nulls(),
        kBatchSize,
        std::vector<VectorPtr>{inputX->childAt(1), inputX->childAt(0)});
    auto expectedZ = std::make_shared<RowVector>(
        pool_.get(),
        expectedType->childAt(2),
        inputZ->nulls(),
        kBatchSize,
        std::vector<VectorPtr>{expectedX, inputZ->childAt(0)});
    auto expected = std::make_shared<RowVector>(
        pool_.get(),
        expectedType,
        inputRow->nulls(),
        kBatchSize,
        std::vector<VectorPtr>{
            inputRow->childAt(1), inputRow->childAt(2), expectedZ});
    for (vector_size_t row = 0; row < kBatchSize; ++row) {
      EXPECT_TRUE(output->equalValueAt(expected.get(), row, row))
          << "row=" << row;
    }
  }
}

TEST_F(ProjectorTest, columnProjectionDeserializerReadsTopLevelFlatMap) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(INTEGER(), DOUBLE())},
  });

  const vector_size_t numRows = 1;
  auto ids = makeIntVector<int64_t>({100});
  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  mapOffsets->asMutable<vector_size_t>()[0] = 0;
  mapSizes->asMutable<vector_size_t>()[0] = 1;

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      makeIntVector<int32_t>({1}),
      BaseVector::create<FlatVector<double>>(DOUBLE(), 1, pool_.get()));
  mapVector->mapValues()->as<FlatVector<double>>()->set(0, 1.5);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  auto [serialized, inputSchema] = serializeWithSchema(
      vec,
      type,
      {
          .version = SerializationVersion::kSerialization,
          .flatMapColumns = {{"features", {}}},
      });

  Deserializer deserializer{
      inputSchema,
      makeSubfields({"features"}),
      pool_.get(),
      {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), numRows);
  auto* row = output->as<RowVector>();
  ASSERT_NE(row, nullptr);
  EXPECT_TRUE(output->type()->equivalent(
      *ROW({{"features", MAP(INTEGER(), DOUBLE())}})));
  ASSERT_EQ(row->children().size(), 1);

  auto* features = row->childAt(0)->as<MapVector>();
  ASSERT_NE(features, nullptr);
  ASSERT_EQ(features->sizeAt(0), 1);
  const auto offset = features->offsetAt(0);
  auto* keys = features->mapKeys()->as<FlatVector<int32_t>>();
  auto* values = features->mapValues()->as<FlatVector<double>>();
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(keys->valueAt(offset), 1);
  EXPECT_DOUBLE_EQ(values->valueAt(offset), 1.5);
}

TEST_F(ProjectorTest, columnProjectionDeserializerReadsFlatMapKey) {
  auto type = ROW({
      {"features", MAP(INTEGER(), DOUBLE())},
  });

  const vector_size_t numRows = 1;
  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  mapOffsets->asMutable<vector_size_t>()[0] = 0;
  mapSizes->asMutable<vector_size_t>()[0] = 2;

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      makeIntVector<int32_t>({1, 2}),
      BaseVector::create<FlatVector<double>>(DOUBLE(), 2, pool_.get()));
  mapVector->mapValues()->as<FlatVector<double>>()->set(0, 1.5);
  mapVector->mapValues()->as<FlatVector<double>>()->set(1, 2.5);

  auto vec = std::make_shared<RowVector>(
      pool_.get(), type, nullptr, numRows, std::vector<VectorPtr>{mapVector});

  auto [serialized, inputSchema] = serializeWithSchema(
      vec,
      type,
      {
          .version = SerializationVersion::kSerialization,
          .flatMapColumns = {{"features", {}}},
      });

  Deserializer deserializer{
      inputSchema,
      makeSubfields({"features[1]"}),
      pool_.get(),
      {.hasHeader = true}};
  auto expectedType = ROW({{"features", MAP(INTEGER(), DOUBLE())}});
  VectorPtr output;
  deserializer.deserialize(serialized, output);

  EXPECT_TRUE(output->type()->equivalent(*expectedType));
  auto* row = output->as<RowVector>();
  ASSERT_NE(row, nullptr);
  ASSERT_EQ(row->childrenSize(), 1);
  auto* features = row->childAt(0)->as<MapVector>();
  ASSERT_NE(features, nullptr);
  ASSERT_EQ(features->sizeAt(0), 1);
  const auto offset = features->offsetAt(0);
  EXPECT_EQ(features->mapKeys()->as<FlatVector<int32_t>>()->valueAt(offset), 1);
  EXPECT_DOUBLE_EQ(
      features->mapValues()->as<FlatVector<double>>()->valueAt(offset), 1.5);
}

// Test that incompatible format combinations are rejected at projection time.
TEST_F(ProjectorTest, incompatibleFormatsRejected) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
  });

  auto vec = makeSimpleRowVector(
      {"a", "b"},
      {
          makeIntVector<int32_t>({1, 2}),
          makeIntVector<int64_t>({100, 200}),
      });

  auto subfields = makeSubfields({"a"});

  auto inputSchema =
      getNimbleSchema(type, {.version = SerializationVersion::kSerialization});

  // Test kLegacy output version — rejected in constructor.
  NIMBLE_ASSERT_THROW(
      Projector(
          inputSchema,
          subfields,
          pool_.get(),
          {.projectVersion = SerializationVersion::kLegacy}),
      "Projection output version must be kProjection");

  // Test kTablet output version — rejected in constructor.
  NIMBLE_ASSERT_THROW(
      Projector(
          inputSchema,
          subfields,
          pool_.get(),
          {.projectVersion = SerializationVersion::kTablet}),
      "Projection output version must be kProjection");

  // Test kSerialization output version — also rejected (Projector writes
  // kProjection, not kSerialization).
  NIMBLE_ASSERT_THROW(
      Projector(
          inputSchema,
          subfields,
          pool_.get(),
          {.projectVersion = SerializationVersion::kSerialization}),
      "Projection output version must be kProjection");

  // Test kLegacyCompact output version — silently upgraded to kProjection
  // rather than rejected, so directory-branched callers that still pass the
  // old enum value keep working while migrating.
  EXPECT_NO_THROW(Projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kLegacyCompact}));

  // Test kTablet input — rejected at projection time.
  {
    auto serialized =
        serialize(vec, type, {.version = SerializationVersion::kSerialization});
    Projector projector(inputSchema, subfields, pool_.get(), {});

    // Patch the version byte to kTablet.
    std::string tabletInput = serialized;
    tabletInput[0] = static_cast<char>(SerializationVersion::kTablet);

    NIMBLE_ASSERT_THROW(
        projector.project(std::string_view(tabletInput)),
        "Input must be kLegacyCompact, kLegacySerialization, kSerialization, or kProjection");
  }
}

// Test that empty projection is invalid.
TEST_F(ProjectorTest, emptyProjectionInvalid) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
  });

  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto inputSchema = getNimbleSchema(type, serOpts);

  // Empty subfields should throw.
  std::vector<common::Subfield> emptySubfields;
  NIMBLE_ASSERT_THROW(
      Projector(
          inputSchema,
          emptySubfields,
          pool_.get(),
          {.projectVersion = SerializationVersion::kProjection}),
      "Must project at least one subfield");
}

// Test full projection - all columns selected, structurally roundtrips.
// Note: this is no longer a byte-level pass-through because input is
// kSerialization while the Projector's output is kProjection (different
// version bytes for independent format evolution).
TEST_F(ProjectorTest, fullProjectionPassThrough) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
      {"c", VARCHAR()},
  });

  auto vec = makeSimpleRowVector(
      {"a", "b", "c"},
      {
          makeIntVector<int32_t>({1, 2, 3}),
          makeIntVector<int64_t>({100, 200, 300}),
          makeStringVector({"x", "y", "z"}),
      });

  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto serialized = serialize(vec, type, serOpts);
  auto inputSchema = getNimbleSchema(type, serOpts);

  // Full projection - all columns selected.
  auto subfields = makeSubfields({"a", "b", "c"});
  Projector projector(inputSchema, subfields, pool_.get(), {});

  auto outputSchema = projector.projectedSchema();
  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);

    // Verify can deserialize correctly.
    auto result =
        deserialize(toString(projected), outputSchema, {.hasHeader = true});
    ASSERT_EQ(result->size(), 3);

    auto resultRow = result->as<RowVector>();
    EXPECT_EQ(resultRow->childAt(0)->as<FlatVector<int32_t>>()->valueAt(0), 1);
    EXPECT_EQ(
        resultRow->childAt(1)->as<FlatVector<int64_t>>()->valueAt(0), 100);

    verifyProjectedSchema(
        type,
        subfields,
        outputSchema,
        toString(projected),
        {.hasHeader = true});
  }
}

// Test that unsupported operations throw.
TEST_F(ProjectorTest, unsupportedArraySubscript) {
  auto type = ROW({{"arr", ARRAY(INTEGER())}});

  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto inputSchema = getNimbleSchema(type, serOpts);

  // Array subscripts are not supported — subscripts on non-FlatMap source
  // nodes are rejected during subfield resolution against the source schema.
  auto subfields = makeSubfields({"arr[0]"});

  NIMBLE_ASSERT_THROW(
      Projector(
          inputSchema,
          subfields,
          pool_.get(),
          {.projectVersion = SerializationVersion::kProjection}),
      "only supported on FlatMap");
}

// Test that regular map key projection throws.
TEST_F(ProjectorTest, unsupportedMapKeyProjection) {
  auto type = ROW({{"m", MAP(VARCHAR(), INTEGER())}});

  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto inputSchema = getNimbleSchema(type, serOpts);

  // Regular map subscripts are not supported (would need re-encoding) — the
  // source's MAP is not encoded as FlatMap, so subfield resolution rejects
  // the subscript before we even reach schema building.
  auto subfields = makeSubfields({"m[\"key\"]"});

  NIMBLE_ASSERT_THROW(
      Projector(
          inputSchema,
          subfields,
          pool_.get(),
          {.projectVersion = SerializationVersion::kProjection}),
      "only supported on FlatMap");
}

// Test FlatMap serialization/deserialization without projection.
TEST_F(ProjectorTest, flatMapSerializeDeserializeNoProjction) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(INTEGER(), DOUBLE())},
  });

  // Create test data with keys 1, 2, 3.
  const vector_size_t numRows = 2;
  auto ids = makeIntVector<int64_t>({100, 200});

  const int entriesPerRow = 3;
  const int totalEntries = numRows * entriesPerRow;

  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto rawOffsets = mapOffsets->asMutable<vector_size_t>();
  auto rawSizes = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }

  auto mapKeys = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), totalEntries, pool_.get());
  auto mapValues = BaseVector::create<FlatVector<double>>(
      DOUBLE(), totalEntries, pool_.get());

  for (int i = 0; i < totalEntries; ++i) {
    mapKeys->set(i, (i % entriesPerRow) + 1); // Keys: 1, 2, 3
    mapValues->set(i, i * 1.5);
  }

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  // Serialize with FlatMap encoding.
  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Deserialize directly (no projection).
  auto result = deserialize(serialized, inputSchema, {.hasHeader = true});

  ASSERT_EQ(result->size(), 2);
  auto resultRow = result->as<RowVector>();
  auto featuresMap = resultRow->childAt(1)->as<MapVector>();

  // Each row should have 3 entries.
  for (vector_size_t i = 0; i < numRows; ++i) {
    EXPECT_EQ(featuresMap->sizeAt(i), 3);
  }
}

// Full FlatMap projection (no key subscripts) is not supported.
TEST_F(ProjectorTest, projectEntireFlatMapColumn) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(INTEGER(), DOUBLE())},
  });

  const vector_size_t numRows = 2;
  auto ids = makeIntVector<int64_t>({100, 200});

  const int entriesPerRow = 3;
  const int totalEntries = numRows * entriesPerRow;

  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto* rawOffsets = mapOffsets->asMutable<vector_size_t>();
  auto* rawSizes = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }

  auto mapKeys = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), totalEntries, pool_.get());
  auto mapValues = BaseVector::create<FlatVector<double>>(
      DOUBLE(), totalEntries, pool_.get());

  for (int i = 0; i < totalEntries; ++i) {
    mapKeys->set(i, (i % entriesPerRow) + 1);
    mapValues->set(i, i * 1.5);
  }

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Projecting entire FlatMap without key subscripts should fail.
  auto subfields = makeSubfields({"features"});
  NIMBLE_ASSERT_THROW(
      Projector(
          inputSchema,
          subfields,
          pool_.get(),
          {.projectVersion = SerializationVersion::kProjection}),
      "Cannot project entire FlatMap column without key subscripts");
}

// Test FlatMap full projection (all keys) works.
TEST_F(ProjectorTest, projectFlatMapAllKeys) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(INTEGER(), DOUBLE())},
  });

  // Create test data with keys 1, 2, 3.
  const vector_size_t numRows = 2;
  auto ids = makeIntVector<int64_t>({100, 200});

  const int entriesPerRow = 3;
  const int totalEntries = numRows * entriesPerRow;

  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto rawOffsets = mapOffsets->asMutable<vector_size_t>();
  auto rawSizes = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }

  auto mapKeys = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), totalEntries, pool_.get());
  auto mapValues = BaseVector::create<FlatVector<double>>(
      DOUBLE(), totalEntries, pool_.get());

  for (int i = 0; i < totalEntries; ++i) {
    mapKeys->set(i, (i % entriesPerRow) + 1); // Keys: 1, 2, 3
    mapValues->set(i, i * 1.5);
  }

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  // Serialize with FlatMap encoding.
  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Project ALL keys from FlatMap.
  auto subfields =
      makeSubfields({"features[\"1\"]", "features[\"2\"]", "features[\"3\"]"});
  Projector projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  auto outputSchema = projector.projectedSchema();
  DeserializerOptions deserOpts{.hasHeader = true};
  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    auto projectedStr = toString(projected);

    // Deserialize and verify.
    auto result = deserialize(projectedStr, outputSchema, deserOpts);

    ASSERT_EQ(result->size(), 2);
    auto resultRow = result->as<RowVector>();
    auto featuresMap = resultRow->childAt(0)->as<MapVector>();

    // Each row should have 3 entries (all keys).
    for (vector_size_t i = 0; i < numRows; ++i) {
      EXPECT_EQ(featuresMap->sizeAt(i), 3);
    }

    nimble::ColumnEncodings encodings;
    encodings.flatMapColumns.insert("features");
    verifyProjectedSchema(
        type, subfields, outputSchema, projectedStr, deserOpts, encodings);
  }
}

// Test that buildProjectedNimbleType schema matches projector schema for
// FlatMap key projection, and both decode correctly.
TEST_F(ProjectorTest, flatMapKeyProjectionSchemaComparison) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(INTEGER(), DOUBLE())},
  });

  const vector_size_t numRows = 2;
  auto ids = makeIntVector<int64_t>({100, 200});

  const int entriesPerRow = 3;
  const int totalEntries = numRows * entriesPerRow;

  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto* rawOffsets = mapOffsets->asMutable<vector_size_t>();
  auto* rawSizes = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }

  auto mapKeys = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), totalEntries, pool_.get());
  auto mapValues = BaseVector::create<FlatVector<double>>(
      DOUBLE(), totalEntries, pool_.get());
  for (int i = 0; i < totalEntries; ++i) {
    mapKeys->set(i, (i % entriesPerRow) + 1);
    mapValues->set(i, i * 1.5);
  }

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Project keys "1" and "3" (skip "2").
  auto subfields = makeSubfields({"features[\"1\"]", "features[\"3\"]"});
  Projector projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});
  auto projectorSchema = projector.projectedSchema();

  // buildProjectedNimbleType should produce matching FlatMap schema.
  nimble::ColumnEncodings encodings;
  encodings.flatMapColumns.insert("features");
  auto convertSchema =
      nimble::buildProjectedNimbleType(type->asRow(), subfields, encodings);

  // Both should have Row > FlatMap with keys "1", "3".
  ASSERT_TRUE(projectorSchema->isRow());
  ASSERT_TRUE(convertSchema->isRow());
  ASSERT_EQ(projectorSchema->asRow().childrenCount(), 1);
  ASSERT_EQ(convertSchema->asRow().childrenCount(), 1);

  const auto& projFlatMap = projectorSchema->asRow().childAt(0)->asFlatMap();
  const auto& convFlatMap = convertSchema->asRow().childAt(0)->asFlatMap();
  ASSERT_EQ(projFlatMap.childrenCount(), 2);
  ASSERT_EQ(convFlatMap.childrenCount(), 2);
  EXPECT_EQ(projFlatMap.nameAt(0), "1");
  EXPECT_EQ(projFlatMap.nameAt(1), "3");
  EXPECT_EQ(convFlatMap.nameAt(0), "1");
  EXPECT_EQ(convFlatMap.nameAt(1), "3");

  // Project the data.
  auto projected = projector.project(serialized);
  auto projectedStr = toString(projected);

  // Both schemas should decode correctly.
  {
    auto result =
        deserialize(projectedStr, projectorSchema, {.hasHeader = true});
    ASSERT_EQ(result->size(), numRows);
    auto* map = result->as<RowVector>()->childAt(0)->as<MapVector>();
    ASSERT_NE(map, nullptr);
    for (vector_size_t i = 0; i < numRows; ++i) {
      EXPECT_EQ(map->sizeAt(i), 2);
    }
  }
  {
    auto result = deserialize(projectedStr, convertSchema, {.hasHeader = true});
    ASSERT_EQ(result->size(), numRows);
    auto* map = result->as<RowVector>()->childAt(0)->as<MapVector>();
    ASSERT_NE(map, nullptr);
    for (vector_size_t i = 0; i < numRows; ++i) {
      EXPECT_EQ(map->sizeAt(i), 2);
    }
  }
}

// Full FlatMap projection (no key subscripts) is not supported by
// buildProjectedNimbleType — it requires explicit key selection.
TEST_F(ProjectorTest, flatMapFullProjectionSchemaTypeMismatch) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(INTEGER(), DOUBLE())},
  });

  const vector_size_t numRows = 2;
  auto ids = makeIntVector<int64_t>({100, 200});

  const int entriesPerRow = 3;
  const int totalEntries = numRows * entriesPerRow;

  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto* rawOffsets = mapOffsets->asMutable<vector_size_t>();
  auto* rawSizes = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }

  auto mapKeys = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), totalEntries, pool_.get());
  auto mapValues = BaseVector::create<FlatVector<double>>(
      DOUBLE(), totalEntries, pool_.get());
  for (int i = 0; i < totalEntries; ++i) {
    mapKeys->set(i, (i % entriesPerRow) + 1);
    mapValues->set(i, i * 1.5);
  }

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Projecting entire FlatMap without key subscripts should fail.
  auto subfields = makeSubfields({"features"});
  NIMBLE_ASSERT_THROW(
      Projector(
          inputSchema,
          subfields,
          pool_.get(),
          {.projectVersion = SerializationVersion::kProjection}),
      "Cannot project entire FlatMap column without key subscripts");
}

// Test FlatMap stream indices are correct.
TEST_F(ProjectorTest, flatMapStreamIndices) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(INTEGER(), DOUBLE())},
  });

  // Create test data with keys 1, 2, 3.
  const vector_size_t numRows = 2;
  auto ids = makeIntVector<int64_t>({100, 200});

  const int entriesPerRow = 3;
  const int totalEntries = numRows * entriesPerRow;

  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto rawOffsets = mapOffsets->asMutable<vector_size_t>();
  auto rawSizes = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }

  auto mapKeys = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), totalEntries, pool_.get());
  auto mapValues = BaseVector::create<FlatVector<double>>(
      DOUBLE(), totalEntries, pool_.get());

  for (int i = 0; i < totalEntries; ++i) {
    mapKeys->set(i, (i % entriesPerRow) + 1); // Keys: 1, 2, 3
    mapValues->set(i, i * 1.5);
  }

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  // Serialize with FlatMap encoding.
  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Verify input schema structure.
  // ROW(nulls=0) -> id(scalar=1) -> features(FlatMap, nulls=2)
  //   -> key "1": value=3, inMap=4
  //   -> key "2": value=5, inMap=6
  //   -> key "3": value=7, inMap=8
  ASSERT_TRUE(inputSchema->isRow());
  const auto& row = inputSchema->asRow();
  ASSERT_EQ(row.childrenCount(), 2);
  ASSERT_TRUE(row.childAt(1)->isFlatMap());
  const auto& flatMap = row.childAt(1)->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), 3); // 3 keys discovered

  // Print stream offsets for debugging.
  LOG(INFO) << "Row nulls offset: " << row.nullsDescriptor().offset();
  LOG(INFO) << "id offset: "
            << row.childAt(0)->asScalar().scalarDescriptor().offset();
  LOG(INFO) << "FlatMap nulls offset: " << flatMap.nullsDescriptor().offset();
  for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
    LOG(INFO) << "Key '" << flatMap.nameAt(i)
              << "' inMap offset: " << flatMap.inMapDescriptorAt(i).offset()
              << ", value offset: "
              << flatMap.childAt(i)->asScalar().scalarDescriptor().offset();
  }

  // Project only key "2".
  auto subfields = makeSubfields({"features[\"2\"]"});
  Projector projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  const auto& indices = projector.testingInputStreamIndices();
  LOG(INFO) << "Projected stream indices: ";
  for (uint32_t idx : indices) {
    LOG(INFO) << "  " << idx;
  }

  // Expected: Row nulls, FlatMap nulls, value for "2", inMap for "2"
  // Should be 4 streams total.
  ASSERT_EQ(indices.size(), 4);

  // Verify output schema has correct stream ordering.
  // FlatMap allocates value streams BEFORE inMap streams for each key.
  // This ordering must be preserved in the projected schema.
  auto outputSchema = projector.projectedSchema();
  ASSERT_TRUE(outputSchema->isRow());
  const auto& outputRow = outputSchema->asRow();
  ASSERT_EQ(outputRow.childrenCount(), 1); // Only "features" projected
  ASSERT_TRUE(outputRow.childAt(0)->isFlatMap());
  const auto& outputFlatMap = outputRow.childAt(0)->asFlatMap();
  ASSERT_EQ(outputFlatMap.childrenCount(), 1); // Only key "2" projected

  // Output schema offsets should be: Row nulls=0, FlatMap nulls=1, value=2,
  // inMap=3
  EXPECT_EQ(outputRow.nullsDescriptor().offset(), 0);
  EXPECT_EQ(outputFlatMap.nullsDescriptor().offset(), 1);
  // Value stream must come BEFORE inMap stream (this is the key ordering
  // check).
  EXPECT_EQ(
      outputFlatMap.childAt(0)->asScalar().scalarDescriptor().offset(), 2);
  EXPECT_EQ(outputFlatMap.inMapDescriptorAt(0).offset(), 3);
}

// Test projecting FlatMap with single key.
TEST_F(ProjectorTest, projectFlatMapSingleKey) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(INTEGER(), DOUBLE())},
  });

  // Create test data with keys 1, 2, 3.
  const vector_size_t numRows = 3;
  auto ids = makeIntVector<int64_t>({100, 200, 300});

  // Build map: each row has 3 entries with keys 1, 2, 3.
  const int entriesPerRow = 3;
  const int totalEntries = numRows * entriesPerRow;

  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto rawOffsets = mapOffsets->asMutable<vector_size_t>();
  auto rawSizes = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }

  auto mapKeys = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), totalEntries, pool_.get());
  auto mapValues = BaseVector::create<FlatVector<double>>(
      DOUBLE(), totalEntries, pool_.get());

  for (int i = 0; i < totalEntries; ++i) {
    mapKeys->set(i, (i % entriesPerRow) + 1); // Keys: 1, 2, 3
    mapValues->set(i, i * 1.5);
  }

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  // Serialize with FlatMap encoding.
  // Use serializeWithSchema to get schema AFTER serialization (when FlatMap
  // keys are discovered).
  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Project only key "2" from FlatMap.
  auto subfields = makeSubfields({"features[\"2\"]"});
  Projector projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  auto outputSchema = projector.projectedSchema();

  // Verify output schema: ROW { features: FlatMap with only key "2" }
  ASSERT_TRUE(outputSchema->isRow());
  ASSERT_EQ(outputSchema->asRow().childrenCount(), 1);
  ASSERT_EQ(outputSchema->asRow().nameAt(0), "features");
  ASSERT_TRUE(outputSchema->asRow().childAt(0)->isFlatMap());
  ASSERT_EQ(outputSchema->asRow().childAt(0)->asFlatMap().childrenCount(), 1);
  ASSERT_EQ(outputSchema->asRow().childAt(0)->asFlatMap().nameAt(0), "2");

  // Project and deserialize.
  DeserializerOptions deserOpts{.hasHeader = true};
  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    auto projectedStr = toString(projected);
    auto result = deserialize(projectedStr, outputSchema, deserOpts);

    ASSERT_EQ(result->size(), 3);
    auto resultRow = result->as<RowVector>();
    auto featuresMap = resultRow->childAt(0)->as<MapVector>();

    // Each row should have 1 entry (key 2).
    for (vector_size_t i = 0; i < numRows; ++i) {
      EXPECT_EQ(featuresMap->sizeAt(i), 1);
    }

    nimble::ColumnEncodings encodings;
    encodings.flatMapColumns.insert("features");
    verifyProjectedSchema(
        type, subfields, outputSchema, projectedStr, deserOpts, encodings);
  }
}

// Test projecting multiple FlatMap keys.
TEST_F(ProjectorTest, projectFlatMapMultipleKeys) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(VARCHAR(), INTEGER())},
  });

  // Create test data with string keys "a", "b", "c".
  const vector_size_t numRows = 2;
  auto ids = makeIntVector<int64_t>({100, 200});

  const int entriesPerRow = 3;
  const int totalEntries = numRows * entriesPerRow;

  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto rawOffsets = mapOffsets->asMutable<vector_size_t>();
  auto rawSizes = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }

  auto mapKeys = BaseVector::create<FlatVector<StringView>>(
      VARCHAR(), totalEntries, pool_.get());
  auto mapValues = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), totalEntries, pool_.get());

  std::vector<std::string> keyNames = {"a", "b", "c"};
  for (int i = 0; i < totalEntries; ++i) {
    mapKeys->set(i, StringView(keyNames[i % entriesPerRow]));
    mapValues->set(i, i * 10);
  }

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(VARCHAR(), INTEGER()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  // Serialize with FlatMap encoding.
  // Use serializeWithSchema to get schema AFTER serialization.
  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Project keys "a" and "c" from FlatMap (skip "b").
  auto subfields = makeSubfields({"features[\"a\"]", "features[\"c\"]"});
  Projector projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  auto outputSchema = projector.projectedSchema();

  // Verify output schema has 2 keys.
  ASSERT_TRUE(outputSchema->asRow().childAt(0)->isFlatMap());
  ASSERT_EQ(outputSchema->asRow().childAt(0)->asFlatMap().childrenCount(), 2);
  ASSERT_EQ(outputSchema->asRow().childAt(0)->asFlatMap().nameAt(0), "a");
  ASSERT_EQ(outputSchema->asRow().childAt(0)->asFlatMap().nameAt(1), "c");

  // Project and deserialize.
  DeserializerOptions deserOpts{.hasHeader = true};
  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    auto projectedStr = toString(projected);
    auto result = deserialize(projectedStr, outputSchema, deserOpts);

    ASSERT_EQ(result->size(), 2);
    auto resultRow = result->as<RowVector>();
    auto featuresMap = resultRow->childAt(0)->as<MapVector>();

    // Each row should have 2 entries (keys a and c).
    for (vector_size_t i = 0; i < numRows; ++i) {
      EXPECT_EQ(featuresMap->sizeAt(i), 2);
    }

    nimble::ColumnEncodings encodings;
    encodings.flatMapColumns.insert("features");
    verifyProjectedSchema(
        type, subfields, outputSchema, projectedStr, deserOpts, encodings);
  }
}

// Test projecting FlatMap with only non-existent keys: the projected schema
// contains a placeholder child per requested key, and the round-trip decoded
// output produces null values for those keys.
TEST_F(ProjectorTest, projectFlatMapNonExistentKey) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(INTEGER(), DOUBLE())},
  });

  // Create test data with keys 1, 2, 3.
  const vector_size_t numRows = 2;
  auto ids = makeIntVector<int64_t>({100, 200});

  const int entriesPerRow = 3;
  const int totalEntries = numRows * entriesPerRow;

  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto rawOffsets = mapOffsets->asMutable<vector_size_t>();
  auto rawSizes = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }

  auto mapKeys = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), totalEntries, pool_.get());
  auto mapValues = BaseVector::create<FlatVector<double>>(
      DOUBLE(), totalEntries, pool_.get());

  for (int i = 0; i < totalEntries; ++i) {
    mapKeys->set(i, (i % entriesPerRow) + 1); // Keys: 1, 2, 3
    mapValues->set(i, i * 1.5);
  }

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  // Serialize with FlatMap encoding to discover keys 1, 2, 3.
  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Project a key that does not exist in the source schema. The Projector
  // succeeds: the projected FlatMap contains "999" as a synthetic child,
  // and the byte-copy pipeline writes a 0-byte placeholder slot for it.
  auto subfields = makeSubfields({"features[\"999\"]"});
  Projector projector{
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection}};

  // The projected schema has one child for the requested (missing) key.
  const auto& projectedSchema = projector.projectedSchema();
  ASSERT_EQ(Kind::Row, projectedSchema->kind());
  ASSERT_EQ(1, projectedSchema->asRow().childrenCount());
  ASSERT_EQ(Kind::FlatMap, projectedSchema->asRow().childAt(0)->kind());
  const auto& projectedFlatMap =
      projectedSchema->asRow().childAt(0)->asFlatMap();
  ASSERT_EQ(1, projectedFlatMap.childrenCount());
  EXPECT_EQ("999", projectedFlatMap.nameAt(0));

  // Round-trip: deserialize the projected blob against the projected schema
  // and confirm key 999 decodes to null for every row.
  auto projected = projectInput(projector, serialized, /*useIOBuf=*/false);
  auto projectedStr = toString(projected);
  DeserializerOptions deserOpts{.hasHeader = true};
  auto result = deserialize(projectedStr, projectedSchema, deserOpts);
  ASSERT_EQ(numRows, result->size());
  auto* features = result->as<RowVector>()->childAt(0)->as<MapVector>();
  ASSERT_NE(nullptr, features);
  for (vector_size_t i = 0; i < numRows; ++i) {
    EXPECT_EQ(0, features->sizeAt(i))
        << "row " << i << " should have an empty map (all requested keys "
        << "missing in source)";
  }
}

TEST_F(ProjectorTest, flatMapMissingKeyDeserializesAsNullField) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(VARCHAR(), BIGINT())},
  });

  constexpr vector_size_t numRows = 4;
  constexpr vector_size_t entriesPerRow = 2;
  auto offsets = allocateOffsets(numRows, pool_.get());
  auto sizes = allocateSizes(numRows, pool_.get());
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto* rawSizes = sizes->asMutable<vector_size_t>();
  for (vector_size_t row = 0; row < numRows; ++row) {
    rawOffsets[row] = row * entriesPerRow;
    rawSizes[row] = entriesPerRow;
  }

  auto features = std::make_shared<MapVector>(
      pool_.get(),
      MAP(VARCHAR(), BIGINT()),
      nullptr,
      numRows,
      offsets,
      sizes,
      makeStringVector({"a", "b", "a", "b", "a", "b", "a", "b"}),
      makeIntVector<int64_t>({10, 11, 20, 21, 30, 31, 40, 41}));
  auto input = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{makeIntVector<int64_t>({1, 2, 3, 4}), features});

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(input, type, serOpts);

  Projector projector{
      inputSchema,
      makeSubfields({"features[\"a\"]", "features[\"x\"]"}),
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection}};
  auto projected = projectInput(projector, serialized, /*useIOBuf=*/false);

  DeserializerOptions deserOpts{
      .hasHeader = true,
      .outputType = ROW({"features"}, {ROW({"a", "x"}, {BIGINT(), BIGINT()})})};
  auto output =
      deserialize(toString(projected), projector.projectedSchema(), deserOpts);

  ASSERT_EQ(output->size(), numRows);
  auto* featuresStruct = output->as<RowVector>()->childAt(0)->as<RowVector>();
  ASSERT_NE(featuresStruct, nullptr);
  ASSERT_EQ(featuresStruct->childrenSize(), 2);
  EXPECT_EQ(featuresStruct->type()->asRow().nameOf(0), "a");
  EXPECT_EQ(featuresStruct->type()->asRow().nameOf(1), "x");

  const auto* presentKey = featuresStruct->childAt(0)->asFlatVector<int64_t>();
  const auto* missingKey = featuresStruct->childAt(1)->asFlatVector<int64_t>();
  ASSERT_NE(presentKey, nullptr);
  ASSERT_NE(missingKey, nullptr);

  const std::vector<int64_t> expectedPresentValues{10, 20, 30, 40};
  for (vector_size_t row = 0; row < numRows; ++row) {
    SCOPED_TRACE(fmt::format("row={}", row));
    EXPECT_FALSE(presentKey->isNullAt(row));
    EXPECT_EQ(presentKey->valueAt(row), expectedPresentValues[row]);
    EXPECT_TRUE(missingKey->isNullAt(row));
  }
}

// All-missing-keys projection on a FlatMap whose value subtree is a Row.
// Exercises the `emitPlaceholderOffsets` Row branch end-to-end (Row.nulls +
// 2 inner scalars = 3 UINT32_MAX value slots + 1 inMap slot per missing key).
TEST_F(ProjectorTest, projectFlatMapNonExistentKey_RowValue) {
  auto valueRowType = ROW({{"a", INTEGER()}, {"b", VARCHAR()}});
  auto type =
      ROW({{"id", BIGINT()}, {"features", MAP(INTEGER(), valueRowType)}});

  // 2 rows × 2 entries (keys 1 and 2 — both real, so the source FlatMap has
  // a non-empty value template for `convertToVeloxType` / `childAt(0)`).
  const vector_size_t numRows = 2;
  const int entriesPerRow = 2;
  const int totalEntries = numRows * entriesPerRow;

  auto ids = makeIntVector<int64_t>({100, 200});
  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto* rawOffsets = mapOffsets->asMutable<vector_size_t>();
  auto* rawSizes = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }
  std::vector<int32_t> keys;
  std::vector<int32_t> aVals;
  std::vector<std::string> bVals;
  for (int i = 0; i < totalEntries; ++i) {
    keys.push_back((i % entriesPerRow) + 1); // keys 1, 2
    aVals.push_back(i);
    bVals.push_back(fmt::format("v{}", i));
  }
  auto keysVec = makeIntVector(keys);
  auto aVec = makeIntVector(aVals);
  auto bVec = makeStringVector(bVals);
  auto valueRows = std::make_shared<RowVector>(
      pool_.get(),
      valueRowType,
      nullptr,
      static_cast<vector_size_t>(totalEntries),
      std::vector<VectorPtr>{aVec, bVec});
  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), valueRowType),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      keysVec,
      valueRows);
  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Project a non-existent key (subscript 999 not in source). Should succeed:
  // the projected FlatMap holds "999" as a synthetic child whose value-subtree
  // is a clone of the source's value Row, with UINT32_MAX placeholders for
  // every stream in that subtree plus the inMap.
  auto subfields = makeSubfields({"features[\"999\"]"});
  Projector projector{
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection}};

  const auto& projectedSchema = projector.projectedSchema();
  ASSERT_EQ(Kind::Row, projectedSchema->kind());
  ASSERT_EQ(1, projectedSchema->asRow().childrenCount());
  const auto& projectedFlatMap =
      projectedSchema->asRow().childAt(0)->asFlatMap();
  ASSERT_EQ(1, projectedFlatMap.childrenCount());
  EXPECT_EQ("999", projectedFlatMap.nameAt(0));
  // Value subtree is a Row (cloned structurally from the source's value Row).
  ASSERT_EQ(Kind::Row, projectedFlatMap.childAt(0)->kind());
  EXPECT_EQ(2, projectedFlatMap.childAt(0)->asRow().childrenCount());

  // Round-trip: decoded map should be empty for every row (the inMap stream
  // is all-zero placeholder → gap-fill says "no rows have this key").
  auto projected = projectInput(projector, serialized, /*useIOBuf=*/false);
  DeserializerOptions deserOpts{.hasHeader = true};
  auto result = deserialize(toString(projected), projectedSchema, deserOpts);
  ASSERT_EQ(numRows, result->size());
  auto* features = result->as<RowVector>()->childAt(0)->as<MapVector>();
  ASSERT_NE(nullptr, features);
  for (vector_size_t i = 0; i < numRows; ++i) {
    EXPECT_EQ(0, features->sizeAt(i)) << "row " << i;
  }
}

// All-missing-keys projection on a FlatMap whose value subtree is an Array.
// Exercises the `emitPlaceholderOffsets` Array branch end-to-end (Array
// lengths + 1 element scalar = 2 UINT32_MAX value slots + 1 inMap per key).
TEST_F(ProjectorTest, projectFlatMapNonExistentKey_ArrayValue) {
  auto valueArrayType = ARRAY(INTEGER());
  auto type =
      ROW({{"id", BIGINT()}, {"features", MAP(INTEGER(), valueArrayType)}});

  // 2 rows × 2 entries; each value is a 2-element array.
  const vector_size_t numRows = 2;
  const int entriesPerRow = 2;
  const int totalEntries = numRows * entriesPerRow;
  const int elementsPerArray = 2;

  auto ids = makeIntVector<int64_t>({100, 200});
  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  auto* rawMapOff = mapOffsets->asMutable<vector_size_t>();
  auto* rawMapSz = mapSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawMapOff[i] = i * entriesPerRow;
    rawMapSz[i] = entriesPerRow;
  }
  std::vector<int32_t> keys;
  for (int i = 0; i < totalEntries; ++i) {
    keys.push_back((i % entriesPerRow) + 1); // keys 1, 2
  }
  auto keysVec = makeIntVector(keys);

  auto arrayOffsets = allocateOffsets(totalEntries, pool_.get());
  auto arraySizes = allocateSizes(totalEntries, pool_.get());
  auto* rawArrOff = arrayOffsets->asMutable<vector_size_t>();
  auto* rawArrSz = arraySizes->asMutable<vector_size_t>();
  for (int i = 0; i < totalEntries; ++i) {
    rawArrOff[i] = i * elementsPerArray;
    rawArrSz[i] = elementsPerArray;
  }
  std::vector<int32_t> elementVals;
  for (int i = 0; i < totalEntries * elementsPerArray; ++i) {
    elementVals.push_back(i);
  }
  auto elementsVec = makeIntVector(elementVals);
  auto arrayValues = std::make_shared<ArrayVector>(
      pool_.get(),
      valueArrayType,
      nullptr,
      static_cast<vector_size_t>(totalEntries),
      arrayOffsets,
      arraySizes,
      elementsVec);
  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), valueArrayType),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      keysVec,
      arrayValues);
  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  auto subfields = makeSubfields({"features[\"999\"]"});
  Projector projector{
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection}};

  const auto& projectedSchema = projector.projectedSchema();
  ASSERT_EQ(Kind::Row, projectedSchema->kind());
  const auto& projectedFlatMap =
      projectedSchema->asRow().childAt(0)->asFlatMap();
  ASSERT_EQ(1, projectedFlatMap.childrenCount());
  EXPECT_EQ("999", projectedFlatMap.nameAt(0));
  // Value subtree is an Array (cloned structurally from the source's value).
  ASSERT_EQ(Kind::Array, projectedFlatMap.childAt(0)->kind());

  auto projected = projectInput(projector, serialized, /*useIOBuf=*/false);
  DeserializerOptions deserOpts{.hasHeader = true};
  auto result = deserialize(toString(projected), projectedSchema, deserOpts);
  ASSERT_EQ(numRows, result->size());
  auto* features = result->as<RowVector>()->childAt(0)->as<MapVector>();
  ASSERT_NE(nullptr, features);
  for (vector_size_t i = 0; i < numRows; ++i) {
    EXPECT_EQ(0, features->sizeAt(i)) << "row " << i;
  }
}

// Verifies the placeholder-slot behavior for missing FlatMap keys in the
// production workflow:
//   1) Projector is constructed with the SOURCE schema (the schema the blob
//      was actually serialized with — keys "1" and "3" only).
//   2) Caller projects subfields ["1", "2", "3"], where "2" is missing in
//      the source.
//   3) Caller constructs an EXPANDED schema via the velox-based
//      buildProjectedNimbleType containing all three keys, and uses it to
//      deserialize the projected blob.
//
// The Projector's nimble-schema-based buildProjectedNimbleType emits key "2"
// as a synthetic child with UINT32_MAX input offsets, so the Projector
// writes 0-byte placeholder slots into the trailer at positions 4-5. The
// expanded schema's offsets line up with these positions, and the
// Deserializer's gap-fill produces a null/absent column for key "2" while
// keys "1" and "3" decode their real bytes.
//
// Asserted end-to-end semantics: data["1"]=10.0, data["2"] absent/null,
// data["3"]=30.0.
TEST_F(ProjectorTest, missingKeyInMiddleProducesPlaceholder) {
  auto type = ROW({{"data", MAP(INTEGER(), DOUBLE())}});

  // One row, two entries: keys 1 and 3 (key 2 is intentionally absent).
  const vector_size_t numRows = 1;
  const int entriesPerRow = 2;
  const int totalEntries = numRows * entriesPerRow;

  auto mapOffsets = allocateOffsets(numRows, pool_.get());
  auto mapSizes = allocateSizes(numRows, pool_.get());
  mapOffsets->asMutable<vector_size_t>()[0] = 0;
  mapSizes->asMutable<vector_size_t>()[0] = entriesPerRow;

  auto mapKeys = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), totalEntries, pool_.get());
  auto mapValues = BaseVector::create<FlatVector<double>>(
      DOUBLE(), totalEntries, pool_.get());
  mapKeys->set(0, 1);
  mapKeys->set(1, 3);
  mapValues->set(0, 10.0);
  mapValues->set(1, 30.0);

  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(INTEGER(), DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(), type, nullptr, numRows, std::vector<VectorPtr>{mapVector});

  // Serialize as FlatMap — blob will have streams only for keys "1" and "3".
  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"data", {}}},
  };
  auto [blob, sourceSchema] = serializeWithSchema(vec, type, serOpts);

  // Caller asks for all three keys (key "2" is missing in source).
  auto subfields = makeSubfields({"data[\"1\"]", "data[\"2\"]", "data[\"3\"]"});

  // Build the EXPANDED schema via the velox-based API for use at deserialize
  // time. It has children for all three keys with dense alphabetical offsets.
  nimble::ColumnEncodings encodings;
  encodings.flatMapColumns.insert("data");
  auto expandedSchema =
      nimble::buildProjectedNimbleType(type->asRow(), subfields, encodings);

  // Production flow: construct the Projector with the SOURCE schema (the
  // schema that actually matches the blob). The Projector currently silent-
  // drops key "2" because it's not in the source.
  Projector projector(
      sourceSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  auto projected = projectInput(projector, blob, /*useIOBuf=*/false);
  auto projectedStr = toString(projected);

  // Deserialize the projected blob using the EXPANDED schema (which expects
  // all three keys). The Projector emits a 0-byte placeholder slot for key 2
  // so the expanded schema's offsets line up with the projected blob and the
  // Deserializer's gap-fill produces a null column for the missing key.
  DeserializerOptions deserOpts{.hasHeader = true};
  auto result = deserialize(projectedStr, expandedSchema, deserOpts);

  ASSERT_EQ(result->size(), 1);
  auto resultRow = result->as<RowVector>();
  auto dataMap = resultRow->childAt(0)->as<MapVector>();
  ASSERT_NE(dataMap, nullptr);

  // Build (key -> optional<value>) map for the single row by walking the
  // MapVector. A key is "present" if it appears in the map; "absent" if it
  // doesn't. We expect: 1 -> 10.0, 2 -> absent, 3 -> 30.0.
  std::map<int32_t, std::optional<double>> got;
  auto* keys = dataMap->mapKeys()->as<FlatVector<int32_t>>();
  auto* values = dataMap->mapValues()->as<FlatVector<double>>();
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  const auto offset = dataMap->offsetAt(0);
  const auto size = dataMap->sizeAt(0);
  for (vector_size_t i = offset; i < offset + size; ++i) {
    const auto k = keys->valueAt(i);
    if (values->isNullAt(i)) {
      got[k] = std::nullopt;
    } else {
      got[k] = values->valueAt(i);
    }
  }

  const std::map<int32_t, std::optional<double>> expected{
      {1, 10.0},
      {3, 30.0},
  };
  EXPECT_EQ(got, expected);
}

// Test stream indices are correct.
TEST_F(ProjectorTest, streamIndicesCorrect) {
  auto type = ROW({
      {"a", INTEGER()}, // Stream 1
      {"b", BIGINT()}, // Stream 2
      {"c", VARCHAR()}, // Stream 3
  });
  // Stream 0 is root row nulls.

  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto inputSchema = getNimbleSchema(type, serOpts);

  // Project "b" only - should include streams 0 (root nulls) and 2 (b data).
  auto subfields = makeSubfields({"b"});
  Projector projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  const auto& indices = projector.testingInputStreamIndices();
  ASSERT_EQ(indices.size(), 2);
  EXPECT_EQ(indices[0], 0); // Root row nulls
  EXPECT_EQ(indices[1], 2); // Column b
}

TEST_F(ProjectorTest, nullBarrierFlagFollowsProjectedNullStreams) {
  constexpr vector_size_t kRows = 3;

  struct SerializedInput {
    std::string_view name;
    std::string serialized;
    std::shared_ptr<const nimble::Type> schema;
    bool expectedInputRequiresNullBarrier;
  };

  auto makeNestedRowInput = [&](std::string_view name,
                                bool profileHasNull,
                                bool rootHasNull) {
    auto type = ROW({
        {"id", BIGINT()},
        {"profile", ROW({{"score", INTEGER()}})},
    });

    auto ids = makeIntVector<int64_t>({1, 2, 3});
    auto scores = makeIntVector<int32_t>({10, 20, 30});
    auto profile = std::make_shared<RowVector>(
        pool_.get(),
        ROW({{"score", INTEGER()}}),
        nullptr,
        kRows,
        std::vector<VectorPtr>{scores});
    if (profileHasNull) {
      profile->setNull(1, true);
    }
    auto input = std::make_shared<RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRows,
        std::vector<VectorPtr>{ids, profile});
    if (rootHasNull) {
      input->setNull(2, true);
    }

    SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
    auto serialized = serialize(input, type, serOpts);
    auto inputSchema = getNimbleSchema(type, serOpts);
    return SerializedInput{
        .name = name,
        .serialized = std::move(serialized),
        .schema = std::move(inputSchema),
        .expectedInputRequiresNullBarrier = profileHasNull || rootHasNull,
    };
  };

  auto makeFlatMapRowInput = [&](std::string_view name, bool valueRowHasNull) {
    auto valueType = ROW({{"score", INTEGER()}});
    auto type = ROW({
        {"id", BIGINT()},
        {"features", MAP(VARCHAR(), valueType)},
    });

    auto ids = makeIntVector<int64_t>({1, 2, 3});
    auto offsets = allocateOffsets(kRows, pool_.get());
    auto sizes = allocateSizes(kRows, pool_.get());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < kRows; ++i) {
      rawOffsets[i] = i;
      rawSizes[i] = 1;
    }

    auto keys = makeStringVector({"a", "a", "a"});
    auto scores = makeIntVector<int32_t>({10, 20, 30});
    auto values = std::make_shared<RowVector>(
        pool_.get(), valueType, nullptr, kRows, std::vector<VectorPtr>{scores});
    if (valueRowHasNull) {
      values->setNull(1, true);
    }
    auto features = std::make_shared<MapVector>(
        pool_.get(),
        MAP(VARCHAR(), valueType),
        nullptr,
        kRows,
        offsets,
        sizes,
        keys,
        values);
    auto input = std::make_shared<RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRows,
        std::vector<VectorPtr>{ids, features});

    SerializerOptions serOpts{
        .version = SerializationVersion::kSerialization,
        .flatMapColumns = {{"features", {}}},
    };
    auto [serialized, inputSchema] = serializeWithSchema(input, type, serOpts);
    return SerializedInput{
        .name = name,
        .serialized = std::move(serialized),
        .schema = std::move(inputSchema),
        .expectedInputRequiresNullBarrier = valueRowHasNull,
    };
  };

  auto makeRegularDataNullInput = [&](std::string_view name) {
    auto type = ROW({
        {"id", BIGINT()},
        {"nullable_score", INTEGER()},
        {"items", ARRAY(INTEGER())},
        {"attrs", MAP(VARCHAR(), INTEGER())},
    });

    auto ids = makeIntVector<int64_t>({1, 2, 3});
    auto nullableScores = makeIntVector<int32_t>({10, 20, 30});
    nullableScores->setNull(0, true);

    auto arrayOffsets = allocateOffsets(kRows, pool_.get());
    auto arraySizes = allocateSizes(kRows, pool_.get());
    auto* rawArrayOffsets = arrayOffsets->asMutable<vector_size_t>();
    auto* rawArraySizes = arraySizes->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < kRows; ++i) {
      rawArrayOffsets[i] = i * 2;
      rawArraySizes[i] = 2;
    }
    auto items = std::make_shared<ArrayVector>(
        pool_.get(),
        ARRAY(INTEGER()),
        nullptr,
        kRows,
        arrayOffsets,
        arraySizes,
        makeIntVector<int32_t>({1, 2, 3, 4, 5, 6}));
    items->setNull(1, true);

    auto mapOffsets = allocateOffsets(kRows, pool_.get());
    auto mapSizes = allocateSizes(kRows, pool_.get());
    auto* rawMapOffsets = mapOffsets->asMutable<vector_size_t>();
    auto* rawMapSizes = mapSizes->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < kRows; ++i) {
      rawMapOffsets[i] = i;
      rawMapSizes[i] = 1;
    }
    auto attrs = std::make_shared<MapVector>(
        pool_.get(),
        MAP(VARCHAR(), INTEGER()),
        nullptr,
        kRows,
        mapOffsets,
        mapSizes,
        makeStringVector({"a", "b", "c"}),
        makeIntVector<int32_t>({100, 200, 300}));
    attrs->setNull(2, true);

    auto input = std::make_shared<RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRows,
        std::vector<VectorPtr>{ids, nullableScores, items, attrs});

    SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
    auto [serialized, inputSchema] = serializeWithSchema(input, type, serOpts);
    return SerializedInput{
        .name = name,
        .serialized = std::move(serialized),
        .schema = std::move(inputSchema),
        .expectedInputRequiresNullBarrier = false,
    };
  };

  auto inputRequiresNullBarrier = [](const SerializedInput& input) {
    return outputRequiresNullBarrier(
        folly::IOBuf::wrapBufferAsValue(
            input.serialized.data(), input.serialized.size()));
  };

  const auto nestedRowNoNulls =
      makeNestedRowInput("nestedRowNoNulls", false, false);
  const auto nestedRowHasNulls =
      makeNestedRowInput("nestedRowHasNulls", true, false);
  const auto topLevelRowHasNulls =
      makeNestedRowInput("topLevelRowHasNulls", false, true);
  const auto flatMapRowNoNulls =
      makeFlatMapRowInput("flatMapRowNoNulls", false);
  const auto flatMapRowHasNulls =
      makeFlatMapRowInput("flatMapRowHasNulls", true);
  const auto regularDataNulls = makeRegularDataNullInput("regularDataNulls");

  for (const auto* input :
       {&nestedRowNoNulls,
        &nestedRowHasNulls,
        &topLevelRowHasNulls,
        &flatMapRowNoNulls,
        &flatMapRowHasNulls,
        &regularDataNulls}) {
    SCOPED_TRACE(input->name);
    EXPECT_EQ(
        inputRequiresNullBarrier(*input),
        input->expectedInputRequiresNullBarrier);
  }

  auto rootNullResult = deserialize(
      topLevelRowHasNulls.serialized,
      topLevelRowHasNulls.schema,
      {.hasHeader = true});
  ASSERT_EQ(rootNullResult->size(), kRows);
  EXPECT_TRUE(rootNullResult->isNullAt(2));

  struct TestCase {
    std::string_view name;
    const SerializedInput* input;
    std::vector<common::Subfield> subfields;
    bool expectedRequiresNullBarrier;
  };
  std::vector<TestCase> testCases;
  testCases.reserve(10);
  testCases.push_back({
      .name = "nestedRowNoNulls",
      .input = &nestedRowNoNulls,
      .subfields = makeSubfields({"profile.score"}),
      .expectedRequiresNullBarrier = false,
  });
  testCases.push_back({
      .name = "nestedRowHasNullsScalarOnly",
      .input = &nestedRowHasNulls,
      .subfields = makeSubfields({"id"}),
      .expectedRequiresNullBarrier = false,
  });
  testCases.push_back({
      .name = "nestedRowHasNulls",
      .input = &nestedRowHasNulls,
      .subfields = makeSubfields({"profile.score"}),
      .expectedRequiresNullBarrier = true,
  });
  testCases.push_back({
      .name = "topLevelRowHasNulls",
      .input = &topLevelRowHasNulls,
      .subfields = makeSubfields({"id"}),
      .expectedRequiresNullBarrier = true,
  });
  testCases.push_back({
      .name = "flatMapRowNoNulls",
      .input = &flatMapRowNoNulls,
      .subfields = makeSubfields({"features[\"a\"].score"}),
      .expectedRequiresNullBarrier = false,
  });
  testCases.push_back({
      .name = "flatMapRowHasNullsScalarOnly",
      .input = &flatMapRowHasNulls,
      .subfields = makeSubfields({"id"}),
      .expectedRequiresNullBarrier = false,
  });
  testCases.push_back({
      .name = "flatMapRowHasNulls",
      .input = &flatMapRowHasNulls,
      .subfields = makeSubfields({"features[\"a\"].score"}),
      .expectedRequiresNullBarrier = true,
  });
  testCases.push_back({
      .name = "regularScalarNulls",
      .input = &regularDataNulls,
      .subfields = makeSubfields({"nullable_score"}),
      .expectedRequiresNullBarrier = false,
  });
  testCases.push_back({
      .name = "regularArrayNulls",
      .input = &regularDataNulls,
      .subfields = makeSubfields({"items"}),
      .expectedRequiresNullBarrier = false,
  });
  testCases.push_back({
      .name = "regularMapNulls",
      .input = &regularDataNulls,
      .subfields = makeSubfields({"attrs"}),
      .expectedRequiresNullBarrier = false,
  });

  for (const auto& testCase : testCases) {
    for (bool useChained : {false, true}) {
      SCOPED_TRACE(fmt::format("{} useChained={}", testCase.name, useChained));
      Projector projector(
          testCase.input->schema, testCase.subfields, pool_.get(), {});
      folly::IOBuf projected;
      if (useChained) {
        const auto mid = testCase.input->serialized.size() / 2;
        auto chainedBuf =
            folly::IOBuf::copyBuffer(testCase.input->serialized.data(), mid);
        chainedBuf->appendToChain(
            folly::IOBuf::copyBuffer(
                testCase.input->serialized.data() + mid,
                testCase.input->serialized.size() - mid));
        projected = projector.project(*chainedBuf);
      } else {
        projected =
            projector.project(std::string_view(testCase.input->serialized));
      }
      EXPECT_EQ(
          outputRequiresNullBarrier(projected),
          testCase.expectedRequiresNullBarrier);
      auto result = deserialize(
          toString(projected),
          projector.projectedSchema(),
          {.hasHeader = true});
      ASSERT_EQ(result->size(), kRows);
    }
  }
}

TEST_F(ProjectorTest, projectedRowNullMixedBatchPreserveNulls) {
  constexpr vector_size_t kRowsPerBatch = 3;

  auto type = ROW({
      {"id", BIGINT()},
      {"score", INTEGER()},
  });
  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto inputSchema = getNimbleSchema(type, serOpts);
  Projector projector(
      inputSchema,
      makeSubfields({"id"}),
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  auto makeInput = [&](const std::vector<int64_t>& ids,
                       const std::vector<int32_t>& scores,
                       std::optional<vector_size_t> nullRow) {
    auto input = std::make_shared<RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRowsPerBatch,
        std::vector<VectorPtr>{
            makeIntVector<int64_t>(ids), makeIntVector<int32_t>(scores)});
    if (nullRow.has_value()) {
      input->setNull(*nullRow, true);
    }
    return input;
  };

  const auto noNullBatch = serialize(
      makeInput({1, 2, 3}, {10, 20, 30}, std::nullopt), type, serOpts);
  const auto nullBatch =
      serialize(makeInput({4, 5, 6}, {40, 50, 60}, 1), type, serOpts);

  auto projectedNoNull =
      projectInput(projector, noNullBatch, /*useIOBuf=*/false);
  auto projectedWithNull =
      projectInput(projector, nullBatch, /*useIOBuf=*/false);
  EXPECT_FALSE(outputRequiresNullBarrier(projectedNoNull));
  EXPECT_TRUE(outputRequiresNullBarrier(projectedWithNull));

  const auto projectedNoNullString = toString(projectedNoNull);
  const auto projectedWithNullString = toString(projectedWithNull);
  std::vector<std::string_view> batches{
      projectedNoNullString, projectedWithNullString};
  Deserializer deserializer{
      projector.projectedSchema(), pool_.get(), {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(batches, output);

  ASSERT_EQ(output->size(), 2 * kRowsPerBatch);
  const std::vector<bool> expectedRowNulls{
      false, false, false, false, true, false};
  for (vector_size_t i = 0; i < output->size(); ++i) {
    EXPECT_EQ(output->isNullAt(i), expectedRowNulls[i]) << "row " << i;
  }

  auto* ids = output->as<RowVector>()->childAt(0)->as<FlatVector<int64_t>>();
  ASSERT_NE(ids, nullptr);
  EXPECT_EQ(ids->valueAt(0), 1);
  EXPECT_EQ(ids->valueAt(1), 2);
  EXPECT_EQ(ids->valueAt(2), 3);
  EXPECT_EQ(ids->valueAt(3), 4);
  EXPECT_EQ(ids->valueAt(5), 6);
}

TEST_F(ProjectorTest, projectedNestedRowNullsMixedBatchPreserveNulls) {
  constexpr vector_size_t kRowsPerBatch = 3;

  auto type = ROW({
      {"id", BIGINT()},
      {"profile", ROW({{"score", INTEGER()}})},
  });
  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto inputSchema = getNimbleSchema(type, serOpts);
  Projector projector(
      inputSchema,
      makeSubfields({"profile.score"}),
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  auto makeInput = [&](const std::vector<int64_t>& ids,
                       const std::vector<int32_t>& scores,
                       std::optional<vector_size_t> nullProfileRow) {
    auto profile = std::make_shared<RowVector>(
        pool_.get(),
        ROW({{"score", INTEGER()}}),
        nullptr,
        kRowsPerBatch,
        std::vector<VectorPtr>{makeIntVector<int32_t>(scores)});
    if (nullProfileRow.has_value()) {
      profile->setNull(*nullProfileRow, true);
    }
    return std::make_shared<RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRowsPerBatch,
        std::vector<VectorPtr>{makeIntVector<int64_t>(ids), profile});
  };

  const auto noNullBatch = serialize(
      makeInput({1, 2, 3}, {10, 20, 30}, std::nullopt), type, serOpts);
  const auto nullBatch =
      serialize(makeInput({4, 5, 6}, {40, 50, 60}, 1), type, serOpts);

  auto projectedNoNull =
      projectInput(projector, noNullBatch, /*useIOBuf=*/false);
  auto projectedWithNull =
      projectInput(projector, nullBatch, /*useIOBuf=*/false);
  EXPECT_FALSE(outputRequiresNullBarrier(projectedNoNull));
  EXPECT_TRUE(outputRequiresNullBarrier(projectedWithNull));

  const auto projectedNoNullString = toString(projectedNoNull);
  const auto projectedWithNullString = toString(projectedWithNull);
  std::vector<std::string_view> batches{
      projectedNoNullString, projectedWithNullString};
  Deserializer deserializer{
      projector.projectedSchema(), pool_.get(), {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(batches, output);

  ASSERT_EQ(output->size(), 2 * kRowsPerBatch);
  auto* profile = output->as<RowVector>()->childAt(0)->as<RowVector>();
  ASSERT_NE(profile, nullptr);
  const std::vector<bool> expectedProfileNulls{
      false, false, false, false, true, false};
  for (vector_size_t i = 0; i < output->size(); ++i) {
    EXPECT_EQ(profile->isNullAt(i), expectedProfileNulls[i]) << "row " << i;
  }

  auto* scores = profile->childAt(0)->as<FlatVector<int32_t>>();
  ASSERT_NE(scores, nullptr);
  EXPECT_EQ(scores->valueAt(0), 10);
  EXPECT_EQ(scores->valueAt(1), 20);
  EXPECT_EQ(scores->valueAt(2), 30);
  EXPECT_EQ(scores->valueAt(3), 40);
  EXPECT_EQ(scores->valueAt(5), 60);
}

TEST_F(ProjectorTest, projectedFlatMapNullsMixedBatchPreserveNulls) {
  constexpr vector_size_t kRowsPerBatch = 3;

  auto valueType = ROW({{"score", INTEGER()}});
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(VARCHAR(), valueType)},
  });
  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };

  auto makeInput = [&](const std::vector<int64_t>& ids,
                       const std::vector<int32_t>& scores,
                       std::optional<vector_size_t> nullValueRow) {
    auto offsets = allocateOffsets(kRowsPerBatch, pool_.get());
    auto sizes = allocateSizes(kRowsPerBatch, pool_.get());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < kRowsPerBatch; ++i) {
      rawOffsets[i] = i;
      rawSizes[i] = 1;
    }

    auto values = std::make_shared<RowVector>(
        pool_.get(),
        valueType,
        nullptr,
        kRowsPerBatch,
        std::vector<VectorPtr>{makeIntVector<int32_t>(scores)});
    if (nullValueRow.has_value()) {
      values->setNull(*nullValueRow, true);
    }
    auto features = std::make_shared<MapVector>(
        pool_.get(),
        MAP(VARCHAR(), valueType),
        nullptr,
        kRowsPerBatch,
        offsets,
        sizes,
        makeStringVector({"a", "a", "a"}),
        values);
    return std::make_shared<RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRowsPerBatch,
        std::vector<VectorPtr>{makeIntVector<int64_t>(ids), features});
  };

  const auto noNullInput = makeInput({1, 2, 3}, {10, 20, 30}, std::nullopt);
  const auto nullInput = makeInput({4, 5, 6}, {40, 50, 60}, 1);
  const auto [noNullBatch, inputSchema] =
      serializeWithSchema(noNullInput, type, serOpts);
  const auto [nullBatch, _] = serializeWithSchema(nullInput, type, serOpts);
  Projector projector(
      inputSchema,
      makeSubfields({"features[\"a\"].score"}),
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  auto projectedNoNull =
      projectInput(projector, noNullBatch, /*useIOBuf=*/false);
  auto projectedWithNull =
      projectInput(projector, nullBatch, /*useIOBuf=*/false);
  EXPECT_FALSE(outputRequiresNullBarrier(projectedNoNull));
  EXPECT_TRUE(outputRequiresNullBarrier(projectedWithNull));

  const auto projectedNoNullString = toString(projectedNoNull);
  const auto projectedWithNullString = toString(projectedWithNull);
  std::vector<std::string_view> batches{
      projectedNoNullString, projectedWithNullString};
  Deserializer deserializer{
      projector.projectedSchema(), pool_.get(), {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(batches, output);

  ASSERT_EQ(output->size(), 2 * kRowsPerBatch);
  auto* features = output->as<RowVector>()->childAt(0)->as<MapVector>();
  ASSERT_NE(features, nullptr);
  auto* values = features->mapValues()->as<RowVector>();
  ASSERT_NE(values, nullptr);
  const std::vector<bool> expectedValueNulls{
      false, false, false, false, true, false};
  for (vector_size_t i = 0; i < output->size(); ++i) {
    ASSERT_EQ(features->sizeAt(i), 1) << "row " << i;
    EXPECT_EQ(values->isNullAt(features->offsetAt(i)), expectedValueNulls[i])
        << "row " << i;
  }

  auto* scores = values->childAt(0)->as<FlatVector<int32_t>>();
  ASSERT_NE(scores, nullptr);
  EXPECT_EQ(scores->valueAt(features->offsetAt(0)), 10);
  EXPECT_EQ(scores->valueAt(features->offsetAt(1)), 20);
  EXPECT_EQ(scores->valueAt(features->offsetAt(2)), 30);
  EXPECT_EQ(scores->valueAt(features->offsetAt(3)), 40);
  EXPECT_EQ(scores->valueAt(features->offsetAt(5)), 60);
}

TEST_F(ProjectorTest, projectedUnselectedNullColumnDoesNotRequireNullBarrier) {
  constexpr vector_size_t kRowsPerBatch = 4;

  auto type = ROW({
      {"id", BIGINT()},
      {"kept", ROW({{"score", INTEGER()}})},
      {"skipped", ROW({{"score", INTEGER()}})},
  });
  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto inputSchema = getNimbleSchema(type, serOpts);
  Projector projector(
      inputSchema,
      makeSubfields({"kept.score"}),
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  auto kept = std::make_shared<RowVector>(
      pool_.get(),
      ROW({{"score", INTEGER()}}),
      nullptr,
      kRowsPerBatch,
      std::vector<VectorPtr>{makeIntVector<int32_t>({10, 20, 30, 40})});
  auto skipped = std::make_shared<RowVector>(
      pool_.get(),
      ROW({{"score", INTEGER()}}),
      nullptr,
      kRowsPerBatch,
      std::vector<VectorPtr>{makeIntVector<int32_t>({100, 200, 300, 400})});
  skipped->setNull(1, true);
  skipped->setNull(3, true);
  auto input = std::make_shared<RowVector>(
      pool_.get(),
      type,
      nullptr,
      kRowsPerBatch,
      std::vector<VectorPtr>{
          makeIntVector<int64_t>({1, 2, 3, 4}), kept, skipped});

  auto serialized = serialize(input, type, serOpts);
  auto projected = projectInput(projector, serialized, /*useIOBuf=*/false);
  EXPECT_FALSE(outputRequiresNullBarrier(projected));

  const auto projectedString = toString(projected);
  std::vector<std::string_view> batches{projectedString};
  Deserializer deserializer{
      projector.projectedSchema(), pool_.get(), {.hasHeader = true}};
  VectorPtr output;
  deserializer.deserialize(batches, output);

  ASSERT_EQ(output->size(), kRowsPerBatch);
  auto* outputKept = output->as<RowVector>()->childAt(0)->as<RowVector>();
  ASSERT_NE(outputKept, nullptr);
  for (vector_size_t i = 0; i < kRowsPerBatch; ++i) {
    EXPECT_FALSE(outputKept->isNullAt(i)) << "row " << i;
  }
  auto* scores = outputKept->childAt(0)->as<FlatVector<int32_t>>();
  ASSERT_NE(scores, nullptr);
  EXPECT_EQ(scores->valueAt(0), 10);
  EXPECT_EQ(scores->valueAt(1), 20);
  EXPECT_EQ(scores->valueAt(2), 30);
  EXPECT_EQ(scores->valueAt(3), 40);
}

TEST_F(ProjectorTest, projectedComplexNullFuzzerPreservesNulls) {
  constexpr vector_size_t kRowsPerBatch = 8;
  constexpr int kIterations = 8;
  constexpr int kBatches = 4;
  constexpr vector_size_t kEntriesPerMapRow = 2;

  auto mapValueType = ROW({
      {"weight", DOUBLE()},
      {"active", BOOLEAN()},
  });
  auto type = ROW({
      {"id", BIGINT()},
      {"profile",
       ROW({
           {"score", INTEGER()},
           {"details",
            ROW({
                {"rank", BIGINT()},
                {"quality", DOUBLE()},
            })},
       })},
      {"activity",
       ROW({
           {"clicks", INTEGER()},
           {"label", VARCHAR()},
           {"inner",
            ROW({
                {"flag", BOOLEAN()},
                {"weight", DOUBLE()},
            })},
       })},
      {"events",
       ARRAY(ROW({
           {"event_id", BIGINT()},
           {"payload",
            ROW({
                {"score", DOUBLE()},
                {"tag", VARCHAR()},
            })},
       }))},
      {"attrs", MAP(VARCHAR(), mapValueType)},
      {"flatAttrs", MAP(VARCHAR(), mapValueType)},
  });

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"flatAttrs", {}}},
  };
  const auto rowType = velox::checkedPointerCast<const velox::RowType>(type);

  struct RowLeafPath {
    std::string path;
    std::vector<std::string_view> names;
  };
  struct FlatMapLeafPath {
    std::string path;
    std::string_view key;
    std::string_view field;
  };

  const std::vector<RowLeafPath> rowLeafPaths = {
      {"id", {"id"}},
      {"profile.score", {"profile", "score"}},
      {"profile.details.rank", {"profile", "details", "rank"}},
      {"profile.details.quality", {"profile", "details", "quality"}},
      {"activity.clicks", {"activity", "clicks"}},
      {"activity.label", {"activity", "label"}},
      {"activity.inner.flag", {"activity", "inner", "flag"}},
      {"activity.inner.weight", {"activity", "inner", "weight"}},
  };
  const std::vector<FlatMapLeafPath> flatMapLeafPaths = {
      {"flatAttrs[\"a\"].weight", "a", "weight"},
      {"flatAttrs[\"a\"].active", "a", "active"},
      {"flatAttrs[\"b\"].weight", "b", "weight"},
      {"flatAttrs[\"b\"].active", "b", "active"},
  };

  auto makeFlatInputRow = [&](VectorFuzzer& fuzzer) {
    std::vector<VectorPtr> children;
    children.reserve(rowType->size());
    for (size_t child = 0; child < rowType->size(); ++child) {
      children.push_back(
          fuzzer.fuzzFlat(rowType->childAt(child), kRowsPerBatch));
    }
    return std::make_shared<RowVector>(
        pool_.get(), rowType, nullptr, kRowsPerBatch, std::move(children));
  };

  auto childIndex = [](const RowVector* row, std::string_view name) {
    const auto& rowType = row->type()->asRow();
    for (size_t i = 0; i < rowType.size(); ++i) {
      if (rowType.nameOf(i) == name) {
        return i;
      }
    }
    NIMBLE_FAIL("Missing child {}", name);
  };

  auto leafVector = [&](const RowVector* root,
                        const RowLeafPath& path) -> const BaseVector* {
    const RowVector* row = root;
    for (size_t i = 0; i + 1 < path.names.size(); ++i) {
      row = row->childAt(childIndex(row, path.names[i]))->as<RowVector>();
    }
    return row->childAt(childIndex(row, path.names.back())).get();
  };

  auto pathIsNull = [&](const RowVector* root,
                        const RowLeafPath& path,
                        vector_size_t rowIndex) {
    const RowVector* row = root;
    if (row->isNullAt(rowIndex)) {
      return true;
    }
    for (size_t i = 0; i + 1 < path.names.size(); ++i) {
      row = row->childAt(childIndex(row, path.names[i]))->as<RowVector>();
      if (row->isNullAt(rowIndex)) {
        return true;
      }
    }
    return row->childAt(childIndex(row, path.names.back()))->isNullAt(rowIndex);
  };

  auto expectLeafEqual = [&](const RowVector* expectedRoot,
                             vector_size_t expectedRow,
                             const RowVector* actualRoot,
                             vector_size_t actualRow,
                             const RowLeafPath& path) {
    const auto expectedNull = pathIsNull(expectedRoot, path, expectedRow);
    EXPECT_EQ(pathIsNull(actualRoot, path, actualRow), expectedNull)
        << "path=" << path.path << " expectedRow=" << expectedRow
        << " actualRow=" << actualRow;
    if (expectedNull) {
      return;
    }

    const auto* expected = leafVector(expectedRoot, path);
    const auto* actual = leafVector(actualRoot, path);
    switch (expected->type()->kind()) {
      case TypeKind::BOOLEAN:
        EXPECT_EQ(
            expected->as<FlatVector<bool>>()->valueAt(expectedRow),
            actual->as<FlatVector<bool>>()->valueAt(actualRow));
        break;
      case TypeKind::INTEGER:
        EXPECT_EQ(
            expected->as<FlatVector<int32_t>>()->valueAt(expectedRow),
            actual->as<FlatVector<int32_t>>()->valueAt(actualRow));
        break;
      case TypeKind::BIGINT:
        EXPECT_EQ(
            expected->as<FlatVector<int64_t>>()->valueAt(expectedRow),
            actual->as<FlatVector<int64_t>>()->valueAt(actualRow));
        break;
      case TypeKind::DOUBLE:
        EXPECT_EQ(
            expected->as<FlatVector<double>>()->valueAt(expectedRow),
            actual->as<FlatVector<double>>()->valueAt(actualRow));
        break;
      case TypeKind::VARCHAR:
        EXPECT_EQ(
            expected->as<FlatVector<StringView>>()->valueAt(expectedRow),
            actual->as<FlatVector<StringView>>()->valueAt(actualRow));
        break;
      default:
        NIMBLE_FAIL(
            "Unexpected projected type {}", expected->type()->toString());
    }
  };

  auto makeStringRowMap = [&](bool hasNulls, double valueBase) {
    const auto numEntries = kRowsPerBatch * kEntriesPerMapRow;
    auto offsets = allocateOffsets(kRowsPerBatch, pool_.get());
    auto sizes = allocateSizes(kRowsPerBatch, pool_.get());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    for (vector_size_t row = 0; row < kRowsPerBatch; ++row) {
      rawOffsets[row] = row * kEntriesPerMapRow;
      rawSizes[row] = kEntriesPerMapRow;
    }

    std::vector<std::string> keys;
    keys.reserve(numEntries);
    auto weights = BaseVector::create<FlatVector<double>>(
        DOUBLE(), numEntries, pool_.get());
    auto active = BaseVector::create<FlatVector<bool>>(
        BOOLEAN(), numEntries, pool_.get());
    for (vector_size_t row = 0; row < kRowsPerBatch; ++row) {
      for (vector_size_t mapIndex = 0; mapIndex < kEntriesPerMapRow;
           ++mapIndex) {
        const auto entry = row * kEntriesPerMapRow + mapIndex;
        keys.emplace_back(mapIndex == 0 ? "a" : "b");
        weights->set(entry, valueBase + row * 10 + mapIndex);
        active->set(entry, (row + mapIndex) % 2 == 0);
      }
    }

    auto values = std::make_shared<RowVector>(
        pool_.get(),
        mapValueType,
        nullptr,
        numEntries,
        std::vector<VectorPtr>{weights, active});
    if (hasNulls) {
      values->setNull(4 * kEntriesPerMapRow, true);
    }
    auto map = std::make_shared<MapVector>(
        pool_.get(),
        MAP(VARCHAR(), mapValueType),
        nullptr,
        kRowsPerBatch,
        offsets,
        sizes,
        makeStringVector(keys),
        values);
    if (hasNulls) {
      map->setNull(5, true);
    }
    return map;
  };

  auto findStringKeyEntry =
      [&](const MapVector* map,
          vector_size_t row,
          std::string_view key) -> std::optional<vector_size_t> {
    if (map->isNullAt(row)) {
      return std::nullopt;
    }
    const auto* keys = map->mapKeys()->as<FlatVector<StringView>>();
    const auto offset = map->offsetAt(row);
    const auto size = map->sizeAt(row);
    for (vector_size_t i = 0; i < size; ++i) {
      const auto entry = offset + i;
      if (keys->valueAt(entry).str() == key) {
        return entry;
      }
    }
    return std::nullopt;
  };

  auto flatMapFieldVector =
      [&](const RowVector* root,
          const FlatMapLeafPath& path,
          vector_size_t row) -> std::pair<const BaseVector*, vector_size_t> {
    const auto* map =
        root->childAt(childIndex(root, "flatAttrs"))->as<MapVector>();
    const auto entry = findStringKeyEntry(map, row, path.key);
    NIMBLE_CHECK(entry.has_value(), "Expected FlatMap key in test data");
    const auto* values = map->mapValues()->as<RowVector>();
    return {
        values->childAt(childIndex(values, path.field)).get(),
        *entry,
    };
  };

  auto flatMapFieldIsNull = [&](const RowVector* root,
                                const FlatMapLeafPath& path,
                                vector_size_t row) {
    if (root->isNullAt(row)) {
      return true;
    }
    const auto* map =
        root->childAt(childIndex(root, "flatAttrs"))->as<MapVector>();
    const auto entry = findStringKeyEntry(map, row, path.key);
    if (!entry.has_value()) {
      return true;
    }
    const auto* values = map->mapValues()->as<RowVector>();
    if (values->isNullAt(*entry)) {
      return true;
    }
    return values->childAt(childIndex(values, path.field))->isNullAt(*entry);
  };

  auto expectFlatMapLeafEqual = [&](const RowVector* expectedRoot,
                                    vector_size_t expectedRow,
                                    const RowVector* actualRoot,
                                    vector_size_t actualRow,
                                    const FlatMapLeafPath& path) {
    const auto expectedNull =
        flatMapFieldIsNull(expectedRoot, path, expectedRow);
    EXPECT_EQ(flatMapFieldIsNull(actualRoot, path, actualRow), expectedNull)
        << "path=" << path.path << " expectedRow=" << expectedRow
        << " actualRow=" << actualRow;
    if (expectedNull) {
      return;
    }

    const auto [expected, expectedEntry] =
        flatMapFieldVector(expectedRoot, path, expectedRow);
    const auto [actual, actualEntry] =
        flatMapFieldVector(actualRoot, path, actualRow);
    switch (expected->type()->kind()) {
      case TypeKind::BOOLEAN:
        EXPECT_EQ(
            expected->as<FlatVector<bool>>()->valueAt(expectedEntry),
            actual->as<FlatVector<bool>>()->valueAt(actualEntry));
        break;
      case TypeKind::DOUBLE:
        EXPECT_EQ(
            expected->as<FlatVector<double>>()->valueAt(expectedEntry),
            actual->as<FlatVector<double>>()->valueAt(actualEntry));
        break;
      default:
        NIMBLE_FAIL(
            "Unexpected FlatMap projected type {}",
            expected->type()->toString());
    }
  };

  auto expectRegularMapEqual = [&](const RowVector* expectedRoot,
                                   vector_size_t expectedRow,
                                   const RowVector* actualRoot,
                                   vector_size_t actualRow) {
    const auto* expectedMap =
        expectedRoot->childAt(childIndex(expectedRoot, "attrs"))
            ->as<MapVector>();
    const auto* actualMap =
        actualRoot->childAt(childIndex(actualRoot, "attrs"))->as<MapVector>();
    const auto expectedNull = expectedRoot->isNullAt(expectedRow) ||
        expectedMap->isNullAt(expectedRow);
    const auto actualNull =
        actualRoot->isNullAt(actualRow) || actualMap->isNullAt(actualRow);
    EXPECT_EQ(actualNull, expectedNull)
        << "expectedRow=" << expectedRow << " actualRow=" << actualRow;
    if (expectedNull) {
      return;
    }
    EXPECT_TRUE(actualMap->equalValueAt(expectedMap, actualRow, expectedRow))
        << "expectedRow=" << expectedRow << " actualRow=" << actualRow;
  };

  const auto seed = folly::Random::rand32();
  LOG(INFO) << "projectedComplexNullFuzzerPreservesNulls seed: " << seed;
  folly::detail::DefaultGenerator rng{seed};

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(fmt::format("iteration {}", iteration));

    std::vector<size_t> selectedRowIndices;
    auto containsRowIndex = [&](size_t index) {
      for (const auto selectedIndex : selectedRowIndices) {
        if (selectedIndex == index) {
          return true;
        }
      }
      return false;
    };
    auto addRowIndex = [&](size_t index) {
      if (!containsRowIndex(index)) {
        selectedRowIndices.push_back(index);
      }
    };

    for (size_t i = 0; i < rowLeafPaths.size(); ++i) {
      if (folly::Random::oneIn(2, rng)) {
        addRowIndex(i);
      }
    }
    if (selectedRowIndices.empty()) {
      addRowIndex(folly::Random::rand32(rng) % rowLeafPaths.size());
    }
    bool hasTwoLevelNestedProjection = false;
    for (const auto index : selectedRowIndices) {
      hasTwoLevelNestedProjection |= rowLeafPaths[index].names.size() > 2;
    }
    if (!hasTwoLevelNestedProjection) {
      addRowIndex(2 + folly::Random::rand32(rng) % (rowLeafPaths.size() - 2));
    }

    std::vector<std::string_view> selectedFlatMapKeys;
    auto addFlatMapKey = [&](std::string_view key) {
      for (const auto selectedKey : selectedFlatMapKeys) {
        if (selectedKey == key) {
          return;
        }
      }
      selectedFlatMapKeys.push_back(key);
    };
    addFlatMapKey("a");
    if (folly::Random::oneIn(2, rng)) {
      addFlatMapKey("b");
    }

    std::vector<common::Subfield> subfields;
    std::vector<const RowLeafPath*> selectedRowPaths;
    std::vector<const FlatMapLeafPath*> selectedFlatMapPaths;
    subfields.emplace_back("attrs");
    for (const auto index : selectedRowIndices) {
      subfields.emplace_back(rowLeafPaths[index].path);
      selectedRowPaths.push_back(&rowLeafPaths[index]);
    }
    for (const auto& path : flatMapLeafPaths) {
      for (const auto selectedKey : selectedFlatMapKeys) {
        if (path.key == selectedKey) {
          subfields.emplace_back(path.path);
          selectedFlatMapPaths.push_back(&path);
          break;
        }
      }
    }

    std::vector<RowVectorPtr> inputs;
    std::vector<std::string> serializedBatches;
    std::vector<std::string> projectedStrings;
    std::shared_ptr<const nimble::Type> inputSchema;
    inputs.reserve(kBatches);
    serializedBatches.reserve(kBatches);
    projectedStrings.reserve(kBatches);

    for (int batch = 0; batch < kBatches; ++batch) {
      const auto hasNulls = batch % 2 == 1;
      VectorFuzzer fuzzer(
          {
              .vectorSize = kRowsPerBatch,
              .nullRatio = 0,
              .useRandomNullPattern = true,
              .stringLength = 12,
              .stringVariableLength = true,
              .containerLength = 3,
              .containerVariableLength = true,
              .normalizeMapKeys = true,
          },
          pool_.get(),
          folly::Random::rand32(rng));
      auto fuzzed = makeFlatInputRow(fuzzer);
      std::vector<VectorPtr> children;
      children.reserve(rowType->size());
      for (size_t child = 0; child < rowType->size(); ++child) {
        children.push_back(fuzzed->childAt(child));
      }
      children[childIndex(fuzzed.get(), "attrs")] =
          makeStringRowMap(hasNulls, /*valueBase=*/1000);
      children[childIndex(fuzzed.get(), "flatAttrs")] =
          makeStringRowMap(hasNulls, /*valueBase=*/2000);
      auto input = std::make_shared<RowVector>(
          pool_.get(), type, nullptr, kRowsPerBatch, std::move(children));
      if (hasNulls) {
        auto* profile =
            input->childAt(childIndex(input.get(), "profile"))->as<RowVector>();
        auto* details =
            profile->childAt(childIndex(profile, "details"))->as<RowVector>();
        details->setNull(2, true);
        auto* activity = input->childAt(childIndex(input.get(), "activity"))
                             ->as<RowVector>();
        auto* inner =
            activity->childAt(childIndex(activity, "inner"))->as<RowVector>();
        inner->setNull(3, true);
      }

      auto [serialized, schema] = serializeWithSchema(input, type, serOpts);
      if (inputSchema == nullptr) {
        inputSchema = schema;
      }
      inputs.push_back(input);
      serializedBatches.push_back(std::move(serialized));
    }

    Projector projector(
        inputSchema,
        subfields,
        pool_.get(),
        {.projectVersion = SerializationVersion::kProjection});

    for (int batch = 0; batch < kBatches; ++batch) {
      const auto hasNulls = batch % 2 == 1;
      auto projected =
          projectInput(projector, serializedBatches[batch], /*useIOBuf=*/false);
      EXPECT_EQ(outputRequiresNullBarrier(projected), hasNulls);
      projectedStrings.push_back(toString(projected));
    }

    std::vector<std::string_view> batches;
    batches.reserve(projectedStrings.size());
    for (const auto& projectedString : projectedStrings) {
      batches.push_back(projectedString);
    }

    Deserializer deserializer{
        projector.projectedSchema(), pool_.get(), {.hasHeader = true}};
    VectorPtr output;
    deserializer.deserialize(batches, output);
    ASSERT_EQ(output->size(), kRowsPerBatch * kBatches);

    const auto* outputRow = output->as<RowVector>();
    for (int batch = 0; batch < kBatches; ++batch) {
      const auto* inputRow = inputs[batch].get();
      for (vector_size_t row = 0; row < kRowsPerBatch; ++row) {
        const auto outputRowIndex = batch * kRowsPerBatch + row;
        expectRegularMapEqual(inputRow, row, outputRow, outputRowIndex);
        for (const auto* path : selectedRowPaths) {
          expectLeafEqual(inputRow, row, outputRow, outputRowIndex, *path);
        }
        for (const auto* path : selectedFlatMapPaths) {
          expectFlatMapLeafEqual(
              inputRow, row, outputRow, outputRowIndex, *path);
        }
      }
    }
  }
}

// Test auto name mapping via projectType for schema evolution (column renames).
TEST_F(ProjectorTest, projectWithUpdatedRowType) {
  // Input schema uses old column names.
  auto inputType = ROW({
      {"old_id", INTEGER()},
      {"old_name", VARCHAR()},
      {"unchanged", BIGINT()},
  });

  // Project type uses new column names (current table schema).
  auto projectType = ROW({
      {"new_id", INTEGER()},
      {"new_name", VARCHAR()},
      {"unchanged", BIGINT()},
  });

  const int numRows = 5;
  std::vector<int32_t> idVals;
  std::vector<std::string> nameVals;
  std::vector<int64_t> valVals;
  for (int i = 0; i < numRows; ++i) {
    idVals.emplace_back(i * 10);
    nameVals.emplace_back("name" + std::to_string(i));
    valVals.emplace_back(i * 100);
  }
  auto ids = makeIntVector(idVals);
  auto names = makeStringVector(nameVals);
  auto values = makeIntVector(valVals);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      inputType,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, names, values});

  // Serialize with old column names.
  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto [serialized, inputSchema] = serializeWithSchema(vec, inputType, serOpts);

  // Query uses new column names from projectType.
  auto subfields = makeSubfields({"new_id", "unchanged"});

  // Without projectType, projection should fail.
  NIMBLE_ASSERT_THROW(
      Projector(
          inputSchema,
          subfields,
          pool_.get(),
          {.projectVersion = SerializationVersion::kProjection}),
      "Field 'new_id' not found in RowType");

  // With projectType, name mapping is auto-computed and projection succeeds.
  Projector::Options opts{
      .projectVersion = SerializationVersion::kProjection,
      .projectType = projectType,
  };
  Projector projector(inputSchema, subfields, pool_.get(), opts);

  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    ASSERT_FALSE(projected.empty());
  }

  // Verify projected schema uses new names from projectType.
  auto projectedSchema = projector.projectedSchema();
  ASSERT_TRUE(projectedSchema->isRow());
  const auto& projectedRow = projectedSchema->asRow();
  ASSERT_EQ(projectedRow.childrenCount(), 2);
  EXPECT_EQ(projectedRow.nameAt(0), "new_id");
  EXPECT_EQ(projectedRow.nameAt(1), "unchanged");

  // Verify stream indices include the mapped column.
  const auto& indices = projector.testingInputStreamIndices();
  // Should have: root nulls (0), old_id (1), unchanged (3).
  // old_name (2) is skipped.
  ASSERT_EQ(indices.size(), 3);
  EXPECT_EQ(indices[0], 0); // Root row nulls
  EXPECT_EQ(indices[1], 1); // old_id (maps to new_id)
  EXPECT_EQ(indices[2], 3); // unchanged
}

// Test auto name mapping with nested ROW inside FlatMap value.
TEST_F(ProjectorTest, projectWithUpdatedNestedRowType) {
  // Input schema has old nested field names.
  auto inputType = ROW({
      {"id", INTEGER()},
      {"features", MAP(VARCHAR(), ROW({{"old_value", INTEGER()}}))},
  });

  // Project type has new nested field names.
  auto projectType = ROW({
      {"id", INTEGER()},
      {"features", MAP(VARCHAR(), ROW({{"new_value", INTEGER()}}))},
  });

  const int numRows = 3;
  std::vector<int32_t> idVals = {0, 1, 2};
  auto ids = makeIntVector(idVals);

  // Create map with nested row values.
  auto offsets = allocateOffsets(numRows, pool_.get());
  auto sizes = allocateSizes(numRows, pool_.get());
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto* rawSizes = sizes->asMutable<vector_size_t>();

  std::vector<std::string> keys;
  std::vector<int32_t> vals;
  int offset = 0;
  for (int i = 0; i < numRows; ++i) {
    rawOffsets[i] = offset;
    rawSizes[i] = 2;
    keys.emplace_back("key_a");
    vals.emplace_back(i * 10);
    keys.emplace_back("key_b");
    vals.emplace_back(i * 20);
    offset += 2;
  }

  auto keysVec = makeStringVector(keys);
  auto valsVec = makeIntVector(vals);
  auto nestedRowType = ROW({{"old_value", INTEGER()}});
  auto nestedRows = std::make_shared<RowVector>(
      pool_.get(),
      nestedRowType,
      nullptr,
      static_cast<vector_size_t>(vals.size()),
      std::vector<VectorPtr>{valsVec});
  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(VARCHAR(), nestedRowType),
      nullptr,
      numRows,
      offsets,
      sizes,
      keysVec,
      nestedRows);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      inputType,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  // Serialize with FlatMap to discover keys.
  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, inputType, serOpts);

  // Query uses key from input (FlatMap keys are not renamed via projectType).
  auto subfields = makeSubfields({"features[\"key_a\"]"});

  // Without projectType, nested row keeps old field names.
  {
    Projector projectorNoType(
        inputSchema,
        subfields,
        pool_.get(),
        {.projectVersion = SerializationVersion::kProjection});
    const auto& nestedRow = projectorNoType.projectedSchema()
                                ->asRow()
                                .childAt(0)
                                ->asFlatMap()
                                .childAt(0)
                                ->asRow();
    EXPECT_EQ(nestedRow.nameAt(0), "old_value");
  }

  // With projectType, nested row field is renamed.
  Projector::Options opts{
      .projectVersion = SerializationVersion::kProjection,
      .projectType = projectType,
  };
  Projector projector(inputSchema, subfields, pool_.get(), opts);

  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    ASSERT_FALSE(projected.empty());
  }

  // Verify projected FlatMap has key_a with nested row using new field name.
  auto projectedSchema = projector.projectedSchema();
  ASSERT_TRUE(projectedSchema->isRow());
  const auto& projectedRow = projectedSchema->asRow();
  ASSERT_EQ(projectedRow.childrenCount(), 1);
  EXPECT_EQ(projectedRow.nameAt(0), "features");
  ASSERT_TRUE(projectedRow.childAt(0)->isFlatMap());
  const auto& projectedFlatMap = projectedRow.childAt(0)->asFlatMap();
  ASSERT_EQ(projectedFlatMap.childrenCount(), 1);
  EXPECT_EQ(projectedFlatMap.nameAt(0), "key_a");

  // Check nested row uses new field name from projectType.
  ASSERT_TRUE(projectedFlatMap.childAt(0)->isRow());
  const auto& nestedProjectedRow = projectedFlatMap.childAt(0)->asRow();
  ASSERT_EQ(nestedProjectedRow.childrenCount(), 1);
  EXPECT_EQ(nestedProjectedRow.nameAt(0), "new_value");
}

TEST_F(ProjectorTest, projectNestedFieldUnderFlatMapValue) {
  // Input schema has old nested field names.
  auto inputType = ROW({
      {"id", INTEGER()},
      {"features", MAP(VARCHAR(), ROW({{"old_value", INTEGER()}}))},
  });

  // Project type has new nested field names.
  auto projectType = ROW({
      {"id", INTEGER()},
      {"features", MAP(VARCHAR(), ROW({{"new_value", INTEGER()}}))},
  });

  const int numRows = 3;
  std::vector<int32_t> idVals = {0, 1, 2};
  auto ids = makeIntVector(idVals);

  // Create map with nested row values.
  auto offsets = allocateOffsets(numRows, pool_.get());
  auto sizes = allocateSizes(numRows, pool_.get());
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto* rawSizes = sizes->asMutable<vector_size_t>();

  std::vector<std::string> keys;
  std::vector<int32_t> vals;
  int offset = 0;
  for (int i = 0; i < numRows; ++i) {
    rawOffsets[i] = offset;
    rawSizes[i] = 2;
    keys.emplace_back("key_a");
    vals.emplace_back(i * 10);
    keys.emplace_back("key_b");
    vals.emplace_back(i * 20);
    offset += 2;
  }

  auto keysVec = makeStringVector(keys);
  auto valsVec = makeIntVector(vals);
  auto nestedRowType = ROW({{"old_value", INTEGER()}});
  auto nestedRows = std::make_shared<RowVector>(
      pool_.get(),
      nestedRowType,
      nullptr,
      static_cast<vector_size_t>(vals.size()),
      std::vector<VectorPtr>{valsVec});
  auto mapVector = std::make_shared<MapVector>(
      pool_.get(),
      MAP(VARCHAR(), nestedRowType),
      nullptr,
      numRows,
      offsets,
      sizes,
      keysVec,
      nestedRows);

  auto vec = std::make_shared<RowVector>(
      pool_.get(),
      inputType,
      nullptr,
      numRows,
      std::vector<VectorPtr>{ids, mapVector});

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, inputType, serOpts);

  // Project a nested field under FlatMap value using new name.
  auto subfields = makeSubfields({"features[\"key_a\"].new_value"});

  // Without projectType, should fail because 'new_value' doesn't exist.
  NIMBLE_ASSERT_THROW(
      Projector(
          inputSchema,
          subfields,
          pool_.get(),
          {.projectVersion = SerializationVersion::kProjection}),
      "Field 'new_value' not found in RowType");

  // With projectType, nested field is renamed and projection succeeds.
  Projector projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection,
       .projectType = projectType});

  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    ASSERT_FALSE(projected.empty());
  }

  // Verify projected schema: features -> key_a -> row with new_value.
  auto projectedSchema = projector.projectedSchema();
  const auto& projectedFlatMap =
      projectedSchema->asRow().childAt(0)->asFlatMap();
  ASSERT_EQ(projectedFlatMap.childrenCount(), 1);
  EXPECT_EQ(projectedFlatMap.nameAt(0), "key_a");
  const auto& nestedProjectedRow = projectedFlatMap.childAt(0)->asRow();
  ASSERT_EQ(nestedProjectedRow.childrenCount(), 1);
  EXPECT_EQ(nestedProjectedRow.nameAt(0), "new_value");
}

// Test projecting keys from multiple FlatMap columns at the same Row level.
// This validates that projected schema offsets match the data layout when
// input stream offsets from different FlatMap subtrees interleave numerically
// (e.g., map_b nulls offset falls between map_a nulls and map_a's first child).
TEST_F(ProjectorTest, projectMultipleFlatMapColumns) {
  // Row with two FlatMap columns.
  auto type = ROW({
      {"map_a", MAP(VARCHAR(), INTEGER())},
      {"map_b", MAP(VARCHAR(), BIGINT())},
  });

  const vector_size_t numRows = 3;

  // Build map_a: keys "x", "y" with int32 values.
  const int aEntriesPerRow = 2;
  const int aTotalEntries = numRows * aEntriesPerRow;

  auto aOffsets = allocateOffsets(numRows, pool_.get());
  auto aSizes = allocateSizes(numRows, pool_.get());
  auto* aRawOffsets = aOffsets->asMutable<vector_size_t>();
  auto* aRawSizes = aSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    aRawOffsets[i] = i * aEntriesPerRow;
    aRawSizes[i] = aEntriesPerRow;
  }
  auto aKeys = BaseVector::create<FlatVector<StringView>>(
      VARCHAR(), aTotalEntries, pool_.get());
  auto aValues = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), aTotalEntries, pool_.get());
  std::vector<std::string> aKeyNames = {"x", "y"};
  for (int i = 0; i < aTotalEntries; ++i) {
    aKeys->set(i, StringView(aKeyNames[i % aEntriesPerRow]));
    aValues->set(i, (i + 1) * 10);
  }
  auto mapA = std::make_shared<MapVector>(
      pool_.get(),
      MAP(VARCHAR(), INTEGER()),
      nullptr,
      numRows,
      aOffsets,
      aSizes,
      aKeys,
      aValues);

  // Build map_b: keys "p", "q" with int64 values.
  const int bEntriesPerRow = 2;
  const int bTotalEntries = numRows * bEntriesPerRow;

  auto bOffsets = allocateOffsets(numRows, pool_.get());
  auto bSizes = allocateSizes(numRows, pool_.get());
  auto* bRawOffsets = bOffsets->asMutable<vector_size_t>();
  auto* bRawSizes = bSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    bRawOffsets[i] = i * bEntriesPerRow;
    bRawSizes[i] = bEntriesPerRow;
  }
  auto bKeys = BaseVector::create<FlatVector<StringView>>(
      VARCHAR(), bTotalEntries, pool_.get());
  auto bValues = BaseVector::create<FlatVector<int64_t>>(
      BIGINT(), bTotalEntries, pool_.get());
  std::vector<std::string> bKeyNames = {"p", "q"};
  for (int i = 0; i < bTotalEntries; ++i) {
    bKeys->set(i, StringView(bKeyNames[i % bEntriesPerRow]));
    bValues->set(i, (i + 1) * 100L);
  }
  auto mapB = std::make_shared<MapVector>(
      pool_.get(),
      MAP(VARCHAR(), BIGINT()),
      nullptr,
      numRows,
      bOffsets,
      bSizes,
      bKeys,
      bValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(), type, nullptr, numRows, std::vector<VectorPtr>{mapA, mapB});

  // Serialize both maps as FlatMaps.
  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"map_a", {}}, {"map_b", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Verify stream offset interleaving: map_b nulls offset should fall between
  // map_a nulls and map_a's first child stream.
  const auto& inputRow = inputSchema->asRow();
  const auto& mapASchema = inputRow.childAt(0)->asFlatMap();
  const auto& mapBSchema = inputRow.childAt(1)->asFlatMap();
  ASSERT_LT(
      mapASchema.nullsDescriptor().offset(),
      mapASchema.childAt(0)->asScalar().scalarDescriptor().offset())
      << "map_a nulls should be before map_a children";
  ASSERT_LT(
      mapBSchema.nullsDescriptor().offset(),
      mapASchema.childAt(0)->asScalar().scalarDescriptor().offset())
      << "map_b nulls should interleave with map_a streams";

  // Project one key from each FlatMap.
  auto subfields = makeSubfields({"map_a[\"x\"]", "map_b[\"q\"]"});
  Projector projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  auto outputSchema = projector.projectedSchema();

  // Verify output schema structure.
  const auto& outRow = outputSchema->asRow();
  ASSERT_EQ(outRow.childrenCount(), 2);
  const auto& outMapA = outRow.childAt(0)->asFlatMap();
  ASSERT_EQ(outMapA.childrenCount(), 1);
  EXPECT_EQ(outMapA.nameAt(0), "x");
  const auto& outMapB = outRow.childAt(1)->asFlatMap();
  ASSERT_EQ(outMapB.childrenCount(), 1);
  EXPECT_EQ(outMapB.nameAt(0), "q");

  // Project and deserialize — this was crashing before the fix due to
  // misaligned stream offsets between schema and data.
  for (bool useIOBuf : {false, true}) {
    SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
    auto projected = projectInput(projector, serialized, useIOBuf);
    auto result =
        deserialize(toString(projected), outputSchema, {.hasHeader = true});

    ASSERT_EQ(result->size(), numRows);
    auto resultRow = result->as<RowVector>();

    // Verify map_a projected values (key "x").
    auto resultMapA = resultRow->childAt(0)->as<MapVector>();
    for (vector_size_t i = 0; i < numRows; ++i) {
      ASSERT_EQ(resultMapA->sizeAt(i), 1);
      auto keyIdx = resultMapA->offsetAt(i);
      auto keyVec = resultMapA->mapKeys()->as<FlatVector<StringView>>();
      EXPECT_EQ(keyVec->valueAt(keyIdx).str(), "x");
      auto valVec = resultMapA->mapValues()->as<FlatVector<int32_t>>();
      // key "x" is at even positions (0, 2, 4) in the input.
      EXPECT_EQ(valVec->valueAt(keyIdx), (i * aEntriesPerRow + 1) * 10);
    }

    // Verify map_b projected values (key "q").
    auto resultMapB = resultRow->childAt(1)->as<MapVector>();
    for (vector_size_t i = 0; i < numRows; ++i) {
      ASSERT_EQ(resultMapB->sizeAt(i), 1);
      auto keyIdx = resultMapB->offsetAt(i);
      auto keyVec = resultMapB->mapKeys()->as<FlatVector<StringView>>();
      EXPECT_EQ(keyVec->valueAt(keyIdx).str(), "q");
      auto valVec = resultMapB->mapValues()->as<FlatVector<int64_t>>();
      // key "q" is at odd positions (1, 3, 5) in the input.
      EXPECT_EQ(valVec->valueAt(keyIdx), (i * bEntriesPerRow + 2) * 100L);
    }
  }
}

// Verifies that the projector correctly handles serialized FlatMap data with
// omitted constant in-map streams (both all-false and all-true cases).
TEST_F(ProjectorTestBase, flatMapConstantInMapStreams) {
  auto type = ROW({
      {"id", BIGINT()},
      {"flat_map", MAP(VARCHAR(), DOUBLE())},
  });

  const vector_size_t batchSize = 10;

  auto generateBatch = [&](const std::vector<std::string>& keys) -> VectorPtr {
    const auto numKeys = static_cast<vector_size_t>(keys.size());
    const vector_size_t totalEntries = batchSize * numKeys;

    auto ids = BaseVector::create(BIGINT(), batchSize, pool_.get());
    auto mapKeys = BaseVector::create(VARCHAR(), totalEntries, pool_.get());
    auto mapValues = BaseVector::create(DOUBLE(), totalEntries, pool_.get());

    for (vector_size_t i = 0; i < batchSize; ++i) {
      ids->asFlatVector<int64_t>()->set(i, i);
    }

    vector_size_t idx = 0;
    for (vector_size_t row = 0; row < batchSize; ++row) {
      for (const auto& key : keys) {
        mapKeys->asFlatVector<StringView>()->set(idx, StringView(key));
        mapValues->asFlatVector<double>()->set(idx, row * 10.0 + idx);
        ++idx;
      }
    }

    auto mapVector = std::make_shared<MapVector>(
        pool_.get(),
        MAP(VARCHAR(), DOUBLE()),
        nullptr,
        batchSize,
        allocateOffsets(batchSize, pool_.get()),
        allocateSizes(batchSize, pool_.get()),
        mapKeys,
        mapValues);

    auto* rawOffsets =
        mapVector->mutableOffsets(batchSize)->asMutable<vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(batchSize)->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < batchSize; ++i) {
      rawOffsets[i] = i * numKeys;
      rawSizes[i] = numKeys;
    }

    return std::make_shared<RowVector>(
        pool_.get(),
        type,
        nullptr,
        batchSize,
        std::vector<VectorPtr>{ids, mapVector});
  };

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"flat_map", {}}},
  };

  // Serialize multiple batches with different key sets to exercise constant
  // in-map streams (all-false for absent keys, all-true for present keys).
  Serializer serializer{serOpts, type, pool_.get()};

  // Batch 1: keys "a" and "b"
  auto batch1 = generateBatch({"a", "b"});
  auto serialized1 = std::string(
      serializer.serialize(batch1, OrderedRanges::of(0, batch1->size())));

  // Batch 2: only key "a" (key "b" all-false in-map).
  auto batch2 = generateBatch({"a"});
  auto serialized2 = std::string(
      serializer.serialize(batch2, OrderedRanges::of(0, batch2->size())));

  // Batch 3: keys "a", "b", "c" (new key discovered)
  auto batch3 = generateBatch({"a", "b", "c"});
  auto serialized3 = std::string(
      serializer.serialize(batch3, OrderedRanges::of(0, batch3->size())));

  // Batch 4: keys "a", "b" (key "c" all-false, "a" and "b" all-true).
  auto batch4 = generateBatch({"a", "b"});
  auto serialized4 = std::string(
      serializer.serialize(batch4, OrderedRanges::of(0, batch4->size())));

  auto nimbleSchema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());

  // Helper to collect stream offsets present in serialized data.
  auto collectStreamOffsets =
      [&](std::string_view data) -> folly::F14FastSet<uint32_t> {
    DeserializerOptions desOpts{.hasHeader = true};
    serde::StreamDataParser reader{pool_.get(), desOpts};
    reader.initialize(data);
    folly::F14FastSet<uint32_t> offsets;
    reader.iterateStreams(
        [&](uint32_t offset, std::string_view) { offsets.insert(offset); });
    return offsets;
  };

  // Get in-map stream offsets from the schema.
  const auto& flatMap = nimbleSchema->asRow().childAt(1)->asFlatMap();
  std::vector<uint32_t> inMapOffsets;
  inMapOffsets.reserve(flatMap.childrenCount());
  for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
    inMapOffsets.push_back(flatMap.inMapDescriptorAt(i).offset());
  }
  ASSERT_EQ(inMapOffsets.size(), 3);

  // Verify constant in-map streams are omitted once their keys are discovered.
  {
    auto offsets1 = collectStreamOffsets(serialized1);
    EXPECT_FALSE(offsets1.contains(inMapOffsets[0]))
        << "batch1: in-map 'a' should be absent";
    EXPECT_FALSE(offsets1.contains(inMapOffsets[1]))
        << "batch1: in-map 'b' should be absent";
    EXPECT_FALSE(offsets1.contains(inMapOffsets[2]))
        << "batch1: in-map 'c' should be absent before discovery";

    auto offsets2 = collectStreamOffsets(serialized2);
    EXPECT_FALSE(offsets2.contains(inMapOffsets[0]))
        << "batch2: in-map 'a' should be absent";
    EXPECT_FALSE(offsets2.contains(inMapOffsets[1]))
        << "batch2: in-map 'b' should be absent";
    EXPECT_FALSE(offsets2.contains(inMapOffsets[2]))
        << "batch2: in-map 'c' should be absent before discovery";

    auto offsets3 = collectStreamOffsets(serialized3);
    EXPECT_FALSE(offsets3.contains(inMapOffsets[0]))
        << "batch3: in-map 'a' should be absent";
    EXPECT_FALSE(offsets3.contains(inMapOffsets[1]))
        << "batch3: in-map 'b' should be absent";
    EXPECT_FALSE(offsets3.contains(inMapOffsets[2]))
        << "batch3: in-map 'c' should be absent";

    auto offsets4 = collectStreamOffsets(serialized4);
    EXPECT_FALSE(offsets4.contains(inMapOffsets[0]))
        << "batch4: in-map 'a' should be absent";
    EXPECT_FALSE(offsets4.contains(inMapOffsets[1]))
        << "batch4: in-map 'b' should be absent";
    EXPECT_FALSE(offsets4.contains(inMapOffsets[2]))
        << "batch4: in-map 'c' should be absent";
  }

  // Project all columns through the projector. FlatMap requires explicit key
  // subscripts — full FlatMap projection is not supported.
  auto subfields = makeSubfields(
      {"id", "flat_map[\"a\"]", "flat_map[\"b\"]", "flat_map[\"c\"]"});
  Projector projector(
      nimbleSchema, subfields, pool_.get(), Projector::Options{});
  auto outputSchema = projector.projectedSchema();

  // Verify each projected batch deserializes correctly.
  std::vector<VectorPtr> inputs = {batch1, batch2, batch3, batch4};
  std::vector<std::string> serializedData = {
      serialized1, serialized2, serialized3, serialized4};

  for (size_t i = 0; i < inputs.size(); ++i) {
    SCOPED_TRACE(fmt::format("batch {}", i));
    for (bool useIOBuf : {false, true}) {
      SCOPED_TRACE(fmt::format("useIOBuf={}", useIOBuf));
      auto projected = projectInput(projector, serializedData[i], useIOBuf);
      auto result =
          deserialize(toString(projected), outputSchema, {.hasHeader = true});
      ASSERT_EQ(result->size(), inputs[i]->size());
      for (vector_size_t j = 0; j < inputs[i]->size(); ++j) {
        ASSERT_TRUE(result->equalValueAt(inputs[i].get(), j, j))
            << "index " << j << "\nExpected: " << inputs[i]->toString(j)
            << "\nActual: " << result->toString(j);
      }
    }
  }
}

// Test that chained IOBuf input works correctly.
TEST_F(ProjectorTest, chainedIOBufInput) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
      {"c", VARCHAR()},
  });

  auto vec = makeSimpleRowVector(
      {"a", "b", "c"},
      {
          makeIntVector<int32_t>({1, 2, 3}),
          makeIntVector<int64_t>({100, 200, 300}),
          makeStringVector({"x", "y", "z"}),
      });

  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto serialized = serialize(vec, type, serOpts);
  auto inputSchema = getNimbleSchema(type, serOpts);

  // Split the serialized buffer into two chained IOBuf segments at midpoint.
  ASSERT_GT(serialized.size(), 2);
  const auto mid = serialized.size() / 2;
  auto chainedBuf = folly::IOBuf::copyBuffer(serialized.data(), mid);
  chainedBuf->appendToChain(
      folly::IOBuf::copyBuffer(
          serialized.data() + mid, serialized.size() - mid));

  // Project column "b" from chained input.
  auto subfields = makeSubfields({"b"});
  Projector projector(inputSchema, subfields, pool_.get(), {});
  auto outputSchema = projector.projectedSchema();

  auto projected = projector.project(*chainedBuf);
  auto result =
      deserialize(toString(projected), outputSchema, {.hasHeader = true});
  ASSERT_EQ(result->size(), 3);

  auto resultRow = result->as<RowVector>();
  auto bCol = resultRow->childAt(0)->as<FlatVector<int64_t>>();
  EXPECT_EQ(bCol->valueAt(0), 100);
  EXPECT_EQ(bCol->valueAt(1), 200);
  EXPECT_EQ(bCol->valueAt(2), 300);

  // Verify projecting all columns from chained IOBuf still deserializes back.
  // Byte-level equality with `serialized` no longer holds because input is
  // kSerialization and output is kProjection (different version bytes).
  auto allSubfields = makeSubfields({"a", "b", "c"});
  Projector allColumnsProjector(inputSchema, allSubfields, pool_.get(), {});
  auto allColumnsResult = allColumnsProjector.project(*chainedBuf);
  auto allColumnsDeserialized = deserialize(
      toString(allColumnsResult), outputSchema, {.hasHeader = true});
  ASSERT_EQ(allColumnsDeserialized->size(), 3);
}

// Verifies inputStreamsSorted_ is true for non-FlatMap projections (stream
// indices are naturally sorted in DFS order) and that the sorted fast path
// produces correct results for both contiguous and chained IOBuf inputs.
TEST_F(ProjectorTest, sortedStreamIndicesFastPath) {
  auto type = ROW({
      {"a", INTEGER()},
      {"b", BIGINT()},
      {"c", VARCHAR()},
  });

  auto vec = makeSimpleRowVector(
      {"a", "b", "c"},
      {
          makeIntVector<int32_t>({1, 2, 3}),
          makeIntVector<int64_t>({100, 200, 300}),
          makeStringVector({"x", "y", "z"}),
      });

  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto serialized = serialize(vec, type, serOpts);
  auto inputSchema = getNimbleSchema(type, serOpts);

  // Project "a" and "c" (skipping "b") — indices are sorted since no FlatMap.
  auto subfields = makeSubfields({"a", "c"});
  Projector projector(inputSchema, subfields, pool_.get(), {});

  EXPECT_TRUE(projector.testingInputStreamsSorted());
  const auto& indices = projector.testingInputStreamIndices();
  EXPECT_TRUE(std::is_sorted(indices.begin(), indices.end()));

  auto outputSchema = projector.projectedSchema();

  DeserializerOptions deserOpts{.hasHeader = true};

  // Test contiguous path.
  {
    auto projected = projector.project(std::string_view(serialized));
    auto result = deserialize(toString(projected), outputSchema, deserOpts);
    ASSERT_EQ(result->size(), 3);
    auto resultRow = result->as<RowVector>();
    auto aCol = resultRow->childAt(0)->as<FlatVector<int32_t>>();
    auto cCol = resultRow->childAt(1)->as<FlatVector<StringView>>();
    for (vector_size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(aCol->valueAt(i), i + 1);
    }
    EXPECT_EQ(cCol->valueAt(0).str(), "x");
    EXPECT_EQ(cCol->valueAt(1).str(), "y");
    EXPECT_EQ(cCol->valueAt(2).str(), "z");
  }

  // Test chained IOBuf path.
  {
    const auto mid = serialized.size() / 2;
    auto chainedBuf = folly::IOBuf::copyBuffer(serialized.data(), mid);
    chainedBuf->appendToChain(
        folly::IOBuf::copyBuffer(
            serialized.data() + mid, serialized.size() - mid));
    auto projected = projector.project(*chainedBuf);
    auto result = deserialize(toString(projected), outputSchema, deserOpts);
    ASSERT_EQ(result->size(), 3);
    auto resultRow = result->as<RowVector>();
    auto aCol = resultRow->childAt(0)->as<FlatVector<int32_t>>();
    auto cCol = resultRow->childAt(1)->as<FlatVector<StringView>>();
    for (vector_size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(aCol->valueAt(i), i + 1);
    }
    EXPECT_EQ(cCol->valueAt(0).str(), "x");
    EXPECT_EQ(cCol->valueAt(1).str(), "y");
    EXPECT_EQ(cCol->valueAt(2).str(), "z");
  }
}

// Verifies inputStreamsSorted_ is false when multiple FlatMap columns cause
// interleaved stream indices, and that the reorder path produces correct
// results for both contiguous and chained IOBuf inputs.
TEST_F(ProjectorTest, unsortedStreamIndicesReorderPath) {
  auto type = ROW({
      {"map_a", MAP(VARCHAR(), INTEGER())},
      {"map_b", MAP(VARCHAR(), BIGINT())},
  });

  const vector_size_t numRows = 3;

  // Build map_a: keys "x", "y".
  const int aEntriesPerRow = 2;
  const int aTotalEntries = numRows * aEntriesPerRow;
  auto aOffsets = allocateOffsets(numRows, pool_.get());
  auto aSizes = allocateSizes(numRows, pool_.get());
  auto* aRawOffsets = aOffsets->asMutable<vector_size_t>();
  auto* aRawSizes = aSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    aRawOffsets[i] = i * aEntriesPerRow;
    aRawSizes[i] = aEntriesPerRow;
  }
  auto aKeys = BaseVector::create<FlatVector<StringView>>(
      VARCHAR(), aTotalEntries, pool_.get());
  auto aValues = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), aTotalEntries, pool_.get());
  for (int i = 0; i < aTotalEntries; ++i) {
    aKeys->set(i, StringView(i % 2 == 0 ? "x" : "y"));
    aValues->set(i, (i + 1) * 10);
  }
  auto mapA = std::make_shared<MapVector>(
      pool_.get(),
      MAP(VARCHAR(), INTEGER()),
      nullptr,
      numRows,
      aOffsets,
      aSizes,
      aKeys,
      aValues);

  // Build map_b: keys "p", "q".
  const int bEntriesPerRow = 2;
  const int bTotalEntries = numRows * bEntriesPerRow;
  auto bOffsets = allocateOffsets(numRows, pool_.get());
  auto bSizes = allocateSizes(numRows, pool_.get());
  auto* bRawOffsets = bOffsets->asMutable<vector_size_t>();
  auto* bRawSizes = bSizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    bRawOffsets[i] = i * bEntriesPerRow;
    bRawSizes[i] = bEntriesPerRow;
  }
  auto bKeys = BaseVector::create<FlatVector<StringView>>(
      VARCHAR(), bTotalEntries, pool_.get());
  auto bValues = BaseVector::create<FlatVector<int64_t>>(
      BIGINT(), bTotalEntries, pool_.get());
  for (int i = 0; i < bTotalEntries; ++i) {
    bKeys->set(i, StringView(i % 2 == 0 ? "p" : "q"));
    bValues->set(i, (i + 1) * 100L);
  }
  auto mapB = std::make_shared<MapVector>(
      pool_.get(),
      MAP(VARCHAR(), BIGINT()),
      nullptr,
      numRows,
      bOffsets,
      bSizes,
      bKeys,
      bValues);

  auto vec = std::make_shared<RowVector>(
      pool_.get(), type, nullptr, numRows, std::vector<VectorPtr>{mapA, mapB});

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"map_a", {}}, {"map_b", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Project one key from each FlatMap — interleaved streams cause unsorted
  // indices.
  auto subfields = makeSubfields({"map_a[\"x\"]", "map_b[\"q\"]"});
  Projector projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  EXPECT_FALSE(projector.testingInputStreamsSorted());
  const auto& indices = projector.testingInputStreamIndices();
  EXPECT_FALSE(std::is_sorted(indices.begin(), indices.end()));

  auto outputSchema = projector.projectedSchema();

  // Verify correct projection via both contiguous and chained paths.
  for (bool useChained : {false, true}) {
    SCOPED_TRACE(fmt::format("useChained={}", useChained));

    folly::IOBuf projected;
    if (useChained) {
      const auto mid = serialized.size() / 2;
      auto chainedBuf = folly::IOBuf::copyBuffer(serialized.data(), mid);
      chainedBuf->appendToChain(
          folly::IOBuf::copyBuffer(
              serialized.data() + mid, serialized.size() - mid));
      projected = projector.project(*chainedBuf);
    } else {
      projected = projector.project(std::string_view(serialized));
    }

    auto result =
        deserialize(toString(projected), outputSchema, {.hasHeader = true});
    ASSERT_EQ(result->size(), numRows);
    auto resultRow = result->as<RowVector>();

    // Verify map_a key "x".
    auto resultMapA = resultRow->childAt(0)->as<MapVector>();
    for (vector_size_t i = 0; i < numRows; ++i) {
      ASSERT_EQ(resultMapA->sizeAt(i), 1);
      auto keyIdx = resultMapA->offsetAt(i);
      EXPECT_EQ(
          resultMapA->mapKeys()
              ->as<FlatVector<StringView>>()
              ->valueAt(keyIdx)
              .str(),
          "x");
      EXPECT_EQ(
          resultMapA->mapValues()->as<FlatVector<int32_t>>()->valueAt(keyIdx),
          (i * aEntriesPerRow + 1) * 10);
    }

    // Verify map_b key "q".
    auto resultMapB = resultRow->childAt(1)->as<MapVector>();
    for (vector_size_t i = 0; i < numRows; ++i) {
      ASSERT_EQ(resultMapB->sizeAt(i), 1);
      auto keyIdx = resultMapB->offsetAt(i);
      EXPECT_EQ(
          resultMapB->mapKeys()
              ->as<FlatVector<StringView>>()
              ->valueAt(keyIdx)
              .str(),
          "q");
      EXPECT_EQ(
          resultMapB->mapValues()->as<FlatVector<int64_t>>()->valueAt(keyIdx),
          (i * bEntriesPerRow + 2) * 100L);
    }
  }
}

// Verifies that a single FlatMap column with sorted keys uses the sorted
// fast path (keys are alphabetically sorted in projected schema, and file
// order matches alphabetical order).
TEST_F(ProjectorTest, singleFlatMapSortedFastPath) {
  auto type = ROW({
      {"id", BIGINT()},
      {"features", MAP(VARCHAR(), DOUBLE())},
  });

  const vector_size_t numRows = 2;
  auto ids = makeIntVector<int64_t>({10, 20});

  // Keys "a", "b", "c" — already alphabetical, so projected schema won't
  // reorder them.
  const int entriesPerRow = 3;
  const int totalEntries = numRows * entriesPerRow;

  auto offsets = allocateOffsets(numRows, pool_.get());
  auto sizes = allocateSizes(numRows, pool_.get());
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto* rawSizes = sizes->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * entriesPerRow;
    rawSizes[i] = entriesPerRow;
  }

  auto keys = BaseVector::create<FlatVector<StringView>>(
      VARCHAR(), totalEntries, pool_.get());
  auto values = BaseVector::create<FlatVector<double>>(
      DOUBLE(), totalEntries, pool_.get());
  std::vector<std::string> keyNames = {"a", "b", "c"};
  for (int i = 0; i < totalEntries; ++i) {
    keys->set(i, StringView(keyNames[i % entriesPerRow]));
    values->set(i, i * 1.5);
  }

  auto mapVec = std::make_shared<MapVector>(
      pool_.get(),
      MAP(VARCHAR(), DOUBLE()),
      nullptr,
      numRows,
      offsets,
      sizes,
      keys,
      values);

  auto vec = std::make_shared<RowVector>(
      pool_.get(), type, nullptr, numRows, std::vector<VectorPtr>{ids, mapVec});

  SerializerOptions serOpts{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  auto [serialized, inputSchema] = serializeWithSchema(vec, type, serOpts);

  // Project "id" and two FlatMap keys "a", "c".
  auto subfields = makeSubfields({"id", "features[\"a\"]", "features[\"c\"]"});
  Projector projector(
      inputSchema,
      subfields,
      pool_.get(),
      {.projectVersion = SerializationVersion::kProjection});

  // Single FlatMap with alphabetical keys — indices should be sorted.
  EXPECT_TRUE(projector.testingInputStreamsSorted());

  auto outputSchema = projector.projectedSchema();
  auto projected = projector.project(std::string_view(serialized));
  auto result =
      deserialize(toString(projected), outputSchema, {.hasHeader = true});
  ASSERT_EQ(result->size(), numRows);

  auto resultRow = result->as<RowVector>();
  auto idCol = resultRow->childAt(0)->as<FlatVector<int64_t>>();
  EXPECT_EQ(idCol->valueAt(0), 10);
  EXPECT_EQ(idCol->valueAt(1), 20);

  auto resultMap = resultRow->childAt(1)->as<MapVector>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    ASSERT_EQ(resultMap->sizeAt(i), 2); // keys "a" and "c"
  }
}

// Verifies that projected output does not compress stream sizes in the trailer.
TEST_F(ProjectorTest, projectedTrailerNoCompression) {
  // Create a row with many columns to produce a large stream sizes trailer.
  std::vector<std::string> names;
  std::vector<VectorPtr> children;
  const int numColumns = 50;
  for (int i = 0; i < numColumns; ++i) {
    names.push_back(fmt::format("c{}", i));
    children.push_back(makeIntVector<int32_t>({i, i + 1, i + 2}));
  }
  auto type = ROW(std::vector<std::string>(names), extractTypes(children));
  auto vec = makeSimpleRowVector(names, children);

  // Serialize with kLegacyCompact.
  SerializerOptions serOpts{.version = SerializationVersion::kSerialization};
  auto serialized = serialize(vec, type, serOpts);
  auto inputSchema = getNimbleSchema(type, serOpts);

  // Project a subset of columns.
  auto subfields = makeSubfields({"c0", "c10", "c20", "c30", "c49"});
  Projector projector(inputSchema, subfields, pool_.get(), {});

  auto projected = projector.project(std::string_view(serialized));
  auto projectedStr = toString(projected);

  // Verify the projected output has a valid trailer.
  const auto* end = projectedStr.data() + projectedStr.size();
  auto [indices, sizes] = detail::readTrailerStreamMetadata(end);
  EXPECT_GT(indices.size(), 0);
  EXPECT_EQ(indices.size(), sizes.size());
}

// Fuzz test: serialize batches with different versions, project, and verify
// deserialization correctness.
TEST_F(ProjectorTest, fuzzMixedVersionProjection) {
  auto type = ROW({
      {"bool_val", BOOLEAN()},
      {"int_val", INTEGER()},
      {"long_val", BIGINT()},
      {"double_val", DOUBLE()},
      {"string_val", VARCHAR()},
  });

  const auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;

  const size_t batchSize = 20;
  VectorFuzzer fuzzer(
      {
          .vectorSize = batchSize,
          .nullRatio = 0,
          .stringLength = 20,
          .stringVariableLength = true,
      },
      pool_.get(),
      seed);

  // Versions to cycle through for each batch.
  const std::vector<SerializerOptions> inputVersions = {
      {.version = SerializationVersion::kSerialization},
      {.version = SerializationVersion::kSerialization},
      {.version = SerializationVersion::kSerialization,
       .streamIndicesEncodingType = EncodingType::Delta,
       .streamSizesEncodingType = EncodingType::Delta},
      {.version = SerializationVersion::kSerialization,
       .streamIndicesEncodingType = EncodingType::FixedBitWidth,
       .streamSizesEncodingType = EncodingType::Varint},
  };

  // Output projection versions.
  const std::vector<Projector::Options> outputVersions = {
      {.projectVersion = SerializationVersion::kProjection},
      {.projectVersion = SerializationVersion::kProjection},
      {.projectVersion = SerializationVersion::kProjection,
       .streamIndicesEncodingType = EncodingType::Delta,
       .streamSizesEncodingType = EncodingType::Delta},
      {.projectVersion = SerializationVersion::kProjection,
       .streamIndicesEncodingType = EncodingType::FixedBitWidth,
       .streamSizesEncodingType = EncodingType::Varint},
  };

  // Columns to project (subset of the full schema).
  auto subfields = makeSubfields({"int_val", "string_val"});

  const auto rowType = std::dynamic_pointer_cast<const velox::RowType>(type);

  const int iterations = 10;
  const int batchesPerIteration = 6;
  for (int iter = 0; iter < iterations; ++iter) {
    SCOPED_TRACE(fmt::format("iteration {}", iter));

    // Pick a random output version for this iteration.
    const auto& outputOpts = outputVersions[iter % outputVersions.size()];

    // Serialize batches with cycling input versions, project each, and
    // collect projected buffers + expected vectors.
    std::vector<std::string> projectedBuffers;
    VectorPtr expected;

    // Use the first input version to get schema (all versions produce the
    // same schema for the same type).
    auto inputSchema = getNimbleSchema(type, inputVersions[0]);
    Projector projector(inputSchema, subfields, pool_.get(), outputOpts);
    auto outputSchema = projector.projectedSchema();

    for (int i = 0; i < batchesPerIteration; ++i) {
      const auto& inputOpts = inputVersions[i % inputVersions.size()];

      auto input = fuzzer.fuzzInputRow(rowType);
      auto serialized = serialize(input, type, inputOpts);

      auto projected = projector.project(std::string_view(serialized));
      projectedBuffers.push_back(toString(projected));

      // Build expected output: extract projected columns from input.
      auto projectedInput = std::make_shared<RowVector>(
          pool_.get(),
          ROW({{"int_val", INTEGER()}, {"string_val", VARCHAR()}}),
          nullptr,
          input->size(),
          std::vector<VectorPtr>{
              input->as<RowVector>()->childAt(1),
              input->as<RowVector>()->childAt(4)});

      if (expected == nullptr) {
        expected = projectedInput;
      } else {
        const auto oldSize = expected->size();
        expected->resize(oldSize + projectedInput->size());
        expected->copy(
            projectedInput.get(), oldSize, 0, projectedInput->size());
      }
    }

    // Deserialize all projected buffers together.
    std::vector<std::string_view> projectedSVs;
    projectedSVs.reserve(projectedBuffers.size());
    for (const auto& buf : projectedBuffers) {
      projectedSVs.push_back(buf);
    }

    Deserializer deserializer(
        outputSchema, pool_.get(), DeserializerOptions{.hasHeader = true});
    VectorPtr output;
    deserializer.deserialize(projectedSVs, output);

    ASSERT_EQ(output->size(), expected->size());
    for (vector_size_t i = 0; i < expected->size(); ++i) {
      ASSERT_TRUE(output->equalValueAt(expected.get(), i, i))
          << "Mismatch at row " << i << "\nExpected: " << expected->toString(i)
          << "\nActual: " << output->toString(i);
    }
  }
}

} // namespace facebook::nimble::serde
