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

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/util/config.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/types/TimestampNTZRegistration.h"
#include "velox/functions/sparksql/types/TimestampNTZType.h"
#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

static void mockRelease(ArrowSchema*) {}

class SparkArrowBridgeSchemaExportTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    registerTimestampNTZType();
  }

  void testScalarType(
      const TypePtr& type,
      const char* arrowFormat,
      const ArrowOptions& options = ArrowOptions{}) {
    ArrowSchema arrowSchema;
    exportToArrow(type, arrowSchema, options);

    verifyScalarType(arrowSchema, arrowFormat);

    arrowSchema.release(&arrowSchema);
    EXPECT_EQ(nullptr, arrowSchema.release);
    EXPECT_EQ(nullptr, arrowSchema.private_data);
  }

  // Doesn't check the actual format string of the scalar leaf types (this is
  // tested by the function above), but tests that the types are nested in the
  // correct way.
  void testNestedType(const TypePtr& type) {
    ArrowSchema arrowSchema;
    exportToArrow(type, arrowSchema);

    verifyNestedType(type, &arrowSchema);

    arrowSchema.release(&arrowSchema);
    EXPECT_EQ(nullptr, arrowSchema.release);
    EXPECT_EQ(nullptr, arrowSchema.private_data);
  }

  void testConstant(
      const TypePtr& type,
      const char* arrowFormat,
      const ArrowOptions& options = ArrowOptions{}) {
    ArrowSchema arrowSchema;
    const bool isScalar = (type->size() == 0);
    const bool constantSize = 100;

    // If scalar, create the constant vector directly; if complex type, create a
    // complex vector first, then wrap it in a dictionary.
    auto constantVector = isScalar
        ? BaseVector::createConstant(
              type, variant(type->kind()), constantSize, pool_.get())
        : BaseVector::wrapInConstant(
              constantSize,
              3, // index to use for the constant
              BaseVector::create(type, 100, pool_.get()));

    velox::exportToArrow(constantVector, arrowSchema, options);

    EXPECT_STREQ("+r", arrowSchema.format);
    EXPECT_EQ(nullptr, arrowSchema.name);

    EXPECT_EQ(2, arrowSchema.n_children);
    EXPECT_NE(nullptr, arrowSchema.children);
    EXPECT_EQ(nullptr, arrowSchema.dictionary);

    // Validate run_ends.
    EXPECT_NE(nullptr, arrowSchema.children[0]);
    const auto& runEnds = *arrowSchema.children[0];

    EXPECT_STREQ("i", runEnds.format);
    EXPECT_STREQ("run_ends", runEnds.name);
    EXPECT_EQ(0, runEnds.n_children);
    EXPECT_EQ(nullptr, runEnds.children);
    EXPECT_EQ(nullptr, runEnds.dictionary);

    // Validate values.
    EXPECT_NE(nullptr, arrowSchema.children[1]);

    if (isScalar) {
      verifyScalarType(*arrowSchema.children[1], arrowFormat, "values");
    } else {
      EXPECT_STREQ(arrowFormat, arrowSchema.children[1]->format);
      verifyNestedType(type, arrowSchema.children[1]);
    }

    arrowSchema.release(&arrowSchema);
    EXPECT_EQ(nullptr, arrowSchema.release);
    EXPECT_EQ(nullptr, arrowSchema.private_data);
  }

  void exportToArrow(
      const TypePtr& type,
      ArrowSchema& out,
      const ArrowOptions& options = ArrowOptions{}) {
    velox::exportToArrow(
        BaseVector::create(type, 0, pool_.get()), out, options);
  }

  ArrowSchema makeArrowSchema(const char* format) {
    return ArrowSchema{
        .format = format,
        .name = nullptr,
        .metadata = nullptr,
        .flags = 0,
        .n_children = 0,
        .children = nullptr,
        .dictionary = nullptr,
        .release = mockRelease,
        .private_data = nullptr,
    };
  }

 private:
  void verifyScalarType(
      const ArrowSchema& arrowSchema,
      const char* arrowFormat,
      const char* name = nullptr) {
    EXPECT_STREQ(arrowFormat, arrowSchema.format);
    if (name == nullptr) {
      EXPECT_EQ(nullptr, arrowSchema.name);
    } else {
      EXPECT_STREQ(name, arrowSchema.name);
    }
    EXPECT_EQ(nullptr, arrowSchema.metadata);
    EXPECT_EQ(arrowSchema.flags | ARROW_FLAG_NULLABLE, ARROW_FLAG_NULLABLE);

    EXPECT_EQ(0, arrowSchema.n_children);
    EXPECT_EQ(nullptr, arrowSchema.children);
    EXPECT_EQ(nullptr, arrowSchema.dictionary);
    EXPECT_NE(nullptr, arrowSchema.release);
  }

  void verifyNestedType(const TypePtr& type, ArrowSchema* schema) {
    if (type->kind() == TypeKind::ARRAY) {
      EXPECT_STREQ("+l", schema->format);
    } else if (type->kind() == TypeKind::MAP) {
      EXPECT_STREQ("+m", schema->format);
      ASSERT_EQ(schema->n_children, 1);
      schema = schema->children[0];
      // Map data should be a non-nullable struct type
      ASSERT_EQ(schema->flags & ARROW_FLAG_NULLABLE, 0);
      ASSERT_EQ(schema->n_children, 2);
      // Map data key type should be a non-nullable
      ASSERT_EQ(schema->children[0]->flags & ARROW_FLAG_NULLABLE, 0);
    } else if (type->kind() == TypeKind::ROW) {
      EXPECT_STREQ("+s", schema->format);
    }
    // Scalar type.
    else {
      EXPECT_EQ(nullptr, schema->children);
    }
    ASSERT_EQ(type->size(), schema->n_children);

    // Recurse down the children.
    for (size_t i = 0; i < type->size(); ++i) {
      verifyNestedType(type->childAt(i), schema->children[i]);

      // If this is a rowType, assert that the children returned with the
      // correct name set.
      if (auto rowType = std::dynamic_pointer_cast<const RowType>(type)) {
        EXPECT_EQ(rowType->nameOf(i), std::string(schema->children[i]->name));
      }
    }
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
};

TEST_F(SparkArrowBridgeSchemaExportTest, timestampNTZ) {
  testScalarType(TIMESTAMP_NTZ(), "ts_ntz");
  testNestedType(ARRAY(TIMESTAMP_NTZ()));
  testNestedType(MAP(VARCHAR(), TIMESTAMP_NTZ()));
  testNestedType(
      ROW({TIMESTAMP_NTZ(), ARRAY(BIGINT()), MAP(VARCHAR(), BIGINT())}));

  testConstant(TIMESTAMP_NTZ(), "ts_ntz");
  testConstant(ARRAY(TIMESTAMP_NTZ()), "+l");
  testConstant(MAP(VARCHAR(), TIMESTAMP_NTZ()), "+m");
  testConstant(
      ROW({TIMESTAMP_NTZ(), ARRAY(BIGINT()), MAP(VARCHAR(), BIGINT())}), "+s");
}

class SparkArrowBridgeSchemaImportTest
    : public SparkArrowBridgeSchemaExportTest {
 protected:
  TypePtr testSchemaImport(const char* format) {
    auto arrowSchema = makeArrowSchema(format);
    auto type = importFromArrow(arrowSchema);
    arrowSchema.release(&arrowSchema);
    return type;
  }

  TypePtr testSchemaDictionaryImport(const char* indexFmt, ArrowSchema schema) {
    auto dictionarySchema = makeArrowSchema(indexFmt);
    dictionarySchema.dictionary = &schema;

    auto type = importFromArrow(dictionarySchema);
    dictionarySchema.release(&dictionarySchema);
    return type;
  }

  TypePtr testSchemaReeImport(const char* valuesFmt) {
    auto reeSchema = makeArrowSchema("+r");
    auto runsSchema = makeArrowSchema("i");
    auto valuesSchema = makeArrowSchema(valuesFmt);

    std::vector<ArrowSchema*> schemas{&runsSchema, &valuesSchema};
    reeSchema.n_children = 2;
    reeSchema.children = schemas.data();

    auto type = importFromArrow(reeSchema);
    reeSchema.release(&reeSchema);
    return type;
  }
};

TEST_F(SparkArrowBridgeSchemaImportTest, timestampNTZ) {
  EXPECT_EQ(*TIMESTAMP_NTZ(), *testSchemaImport("ts_ntz"));
  EXPECT_EQ(
      TIMESTAMP_NTZ(),
      testSchemaDictionaryImport("i", makeArrowSchema("ts_ntz")));
  EXPECT_EQ(TIMESTAMP_NTZ(), testSchemaReeImport("ts_ntz"));
}

class SparkArrowBridgeSchemaTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    registerTimestampNTZType();
  }

  void roundtripTest(
      const TypePtr& inputType,
      const ArrowOptions& options = ArrowOptions{}) {
    ArrowSchema arrowSchema;
    exportToArrow(inputType, arrowSchema, options);
    auto outputType = importFromArrow(arrowSchema);
    arrowSchema.release(&arrowSchema);
    EXPECT_EQ(*inputType, *outputType);
  }

 private:
  void exportToArrow(
      const TypePtr& type,
      ArrowSchema& out,
      const ArrowOptions& options = ArrowOptions{}) {
    velox::exportToArrow(
        BaseVector::create(type, 0, pool_.get()), out, options);
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
};

TEST_F(SparkArrowBridgeSchemaTest, roundtrip) {
  roundtripTest(TIMESTAMP_NTZ());
  roundtripTest(ARRAY(ARRAY(ARRAY(ARRAY(TIMESTAMP_NTZ())))));
  roundtripTest(MAP(VARCHAR(), ARRAY(MAP(ARRAY(BIGINT()), TIMESTAMP_NTZ()))));
  roundtripTest(
      ROW({TIMESTAMP_NTZ(), ARRAY(BIGINT()), MAP(VARCHAR(), BIGINT())}));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
