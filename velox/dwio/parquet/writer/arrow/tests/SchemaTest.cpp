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

// Adapted from Apache Arrow.

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "arrow/util/checked_cast.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/SchemaInternal.h"
#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

using ::arrow::internal::checked_cast;

namespace facebook::velox::parquet::arrow {

using facebook::velox::parquet::thrift::FieldRepetitionType;
using facebook::velox::parquet::thrift::SchemaElement;

namespace schema {

static inline SchemaElement newPrimitive(
    const std::string& name,
    FieldRepetitionType::type repetition,
    Type::type type,
    int fieldId = -1) {
  SchemaElement result;
  result.__set_name(name);
  result.__set_repetition_type(repetition);
  result.__set_type(
      static_cast<facebook::velox::parquet::thrift::Type::type>(type));
  if (fieldId >= 0) {
    result.__set_field_id(fieldId);
  }
  return result;
}

static inline SchemaElement newGroup(
    const std::string& name,
    FieldRepetitionType::type repetition,
    int numChildren,
    int fieldId = -1) {
  SchemaElement result;
  result.__set_name(name);
  result.__set_repetition_type(repetition);
  result.__set_num_children(numChildren);

  if (fieldId >= 0) {
    result.__set_field_id(fieldId);
  }

  return result;
}

template <typename NodeType>
static void checkNodeRoundtrip(const Node& node) {
  facebook::velox::parquet::thrift::SchemaElement serialized;
  node.toParquet(&serialized);
  std::unique_ptr<Node> recovered = NodeType::fromParquet(&serialized);
  ASSERT_TRUE(node.equals(recovered.get()))
      << "Recovered node not equivalent to original node constructed "
      << "with logical type " << node.logicalType()->toString() << " got "
      << recovered->logicalType()->toString();
}

static void confirmPrimitiveNodeRoundtrip(
    const std::shared_ptr<const LogicalType>& logicalType,
    Type::type physicalType,
    int physicalLength,
    int fieldId = -1) {
  auto Node = PrimitiveNode::make(
      "something",
      Repetition::kRequired,
      logicalType,
      physicalType,
      physicalLength,
      fieldId);
  checkNodeRoundtrip<PrimitiveNode>(*Node);
}

static void confirmGroupNodeRoundtrip(
    std::string name,
    const std::shared_ptr<const LogicalType>& logicalType,
    int fieldId = -1) {
  auto Node =
      GroupNode::make(name, Repetition::kRequired, {}, logicalType, fieldId);
  checkNodeRoundtrip<GroupNode>(*Node);
}

// ----------------------------------------------------------------------.
// ColumnPath.

TEST(TestColumnPath, TestAttrs) {
  ColumnPath path(std::vector<std::string>({"toplevel", "leaf"}));

  ASSERT_EQ(path.toDotString(), "toplevel.leaf");

  std::shared_ptr<ColumnPath> pathPtr =
      ColumnPath::fromDotString("toplevel.leaf");
  ASSERT_EQ(pathPtr->toDotString(), "toplevel.leaf");

  std::shared_ptr<ColumnPath> extended = pathPtr->extend("anotherlevel");
  ASSERT_EQ(extended->toDotString(), "toplevel.leaf.anotherlevel");
}

// ----------------------------------------------------------------------.
// Primitive node.

class TestPrimitiveNode : public ::testing::Test {
 public:
  void SetUp() {
    name_ = "name";
    fieldId_ = 5;
  }

  void convert(const facebook::velox::parquet::thrift::SchemaElement* element) {
    node_ = PrimitiveNode::fromParquet(element);
    ASSERT_TRUE(node_->isPrimitive());
    primNode_ = static_cast<const PrimitiveNode*>(node_.get());
  }

 protected:
  std::string name_;
  const PrimitiveNode* primNode_;

  int fieldId_;
  std::unique_ptr<Node> node_;
};

TEST_F(TestPrimitiveNode, Attrs) {
  PrimitiveNode node1("foo", Repetition::kRepeated, Type::kInt32);

  PrimitiveNode node2(
      "bar", Repetition::kOptional, Type::kByteArray, ConvertedType::kUtf8);

  ASSERT_EQ("foo", node1.name());

  ASSERT_TRUE(node1.isPrimitive());
  ASSERT_FALSE(node1.isGroup());

  ASSERT_EQ(Repetition::kRepeated, node1.repetition());
  ASSERT_EQ(Repetition::kOptional, node2.repetition());

  ASSERT_EQ(Node::kPrimitive, node1.nodeType());

  ASSERT_EQ(Type::kInt32, node1.physicalType());
  ASSERT_EQ(Type::kByteArray, node2.physicalType());

  // Logical types.
  ASSERT_EQ(ConvertedType::kNone, node1.convertedType());
  ASSERT_EQ(ConvertedType::kUtf8, node2.convertedType());

  // Repetition.
  PrimitiveNode node3("foo", Repetition::kRepeated, Type::kInt32);
  PrimitiveNode node4("foo", Repetition::kRequired, Type::kInt32);
  PrimitiveNode node5("foo", Repetition::kOptional, Type::kInt32);

  ASSERT_TRUE(node3.isRepeated());
  ASSERT_FALSE(node3.isOptional());

  ASSERT_TRUE(node4.isRequired());

  ASSERT_TRUE(node5.isOptional());
  ASSERT_FALSE(node5.isRequired());
}

TEST_F(TestPrimitiveNode, fromParquet) {
  SchemaElement elt = newPrimitive(
      name_, FieldRepetitionType::OPTIONAL, Type::kInt32, fieldId_);
  ASSERT_NO_FATAL_FAILURE(convert(&elt));
  ASSERT_EQ(name_, primNode_->name());
  ASSERT_EQ(fieldId_, primNode_->fieldId());
  ASSERT_EQ(Repetition::kOptional, primNode_->repetition());
  ASSERT_EQ(Type::kInt32, primNode_->physicalType());
  ASSERT_EQ(ConvertedType::kNone, primNode_->convertedType());

  // Test a logical type.
  elt = newPrimitive(
      name_, FieldRepetitionType::REQUIRED, Type::kByteArray, fieldId_);
  elt.__set_converted_type(
      facebook::velox::parquet::thrift::ConvertedType::UTF8);

  ASSERT_NO_FATAL_FAILURE(convert(&elt));
  ASSERT_EQ(Repetition::kRequired, primNode_->repetition());
  ASSERT_EQ(Type::kByteArray, primNode_->physicalType());
  ASSERT_EQ(ConvertedType::kUtf8, primNode_->convertedType());

  // FIXED_LEN_BYTE_ARRAY.
  elt = newPrimitive(
      name_, FieldRepetitionType::OPTIONAL, Type::kFixedLenByteArray, fieldId_);
  elt.__set_type_length(16);

  ASSERT_NO_FATAL_FAILURE(convert(&elt));
  ASSERT_EQ(name_, primNode_->name());
  ASSERT_EQ(fieldId_, primNode_->fieldId());
  ASSERT_EQ(Repetition::kOptional, primNode_->repetition());
  ASSERT_EQ(Type::kFixedLenByteArray, primNode_->physicalType());
  ASSERT_EQ(16, primNode_->typeLength());

  // Facebook::velox::parquet::thrift::ConvertedType::Decimal.
  elt = newPrimitive(
      name_, FieldRepetitionType::OPTIONAL, Type::kFixedLenByteArray, fieldId_);
  elt.__set_converted_type(
      facebook::velox::parquet::thrift::ConvertedType::DECIMAL);
  elt.__set_type_length(6);
  elt.__set_scale(2);
  elt.__set_precision(12);

  ASSERT_NO_FATAL_FAILURE(convert(&elt));
  ASSERT_EQ(Type::kFixedLenByteArray, primNode_->physicalType());
  ASSERT_EQ(ConvertedType::kDecimal, primNode_->convertedType());
  ASSERT_EQ(6, primNode_->typeLength());
  ASSERT_EQ(2, primNode_->decimalMetadata().scale);
  ASSERT_EQ(12, primNode_->decimalMetadata().precision);
}

TEST_F(TestPrimitiveNode, equals) {
  PrimitiveNode node1("foo", Repetition::kRequired, Type::kInt32);
  PrimitiveNode node2("foo", Repetition::kRequired, Type::kInt64);
  PrimitiveNode node3("bar", Repetition::kRequired, Type::kInt32);
  PrimitiveNode node4("foo", Repetition::kOptional, Type::kInt32);
  PrimitiveNode node5("foo", Repetition::kRequired, Type::kInt32);

  ASSERT_TRUE(node1.equals(&node1));
  ASSERT_FALSE(node1.equals(&node2));
  ASSERT_FALSE(node1.equals(&node3));
  ASSERT_FALSE(node1.equals(&node4));
  ASSERT_TRUE(node1.equals(&node5));

  PrimitiveNode flba1(
      "foo",
      Repetition::kRequired,
      Type::kFixedLenByteArray,
      ConvertedType::kDecimal,
      12,
      4,
      2);

  PrimitiveNode flba2(
      "foo",
      Repetition::kRequired,
      Type::kFixedLenByteArray,
      ConvertedType::kDecimal,
      1,
      4,
      2);
  flba2.setTypeLength(12);

  PrimitiveNode flba3(
      "foo",
      Repetition::kRequired,
      Type::kFixedLenByteArray,
      ConvertedType::kDecimal,
      1,
      4,
      2);
  flba3.setTypeLength(16);

  PrimitiveNode flba4(
      "foo",
      Repetition::kRequired,
      Type::kFixedLenByteArray,
      ConvertedType::kDecimal,
      12,
      4,
      0);

  PrimitiveNode flba5(
      "foo",
      Repetition::kRequired,
      Type::kFixedLenByteArray,
      ConvertedType::kNone,
      12,
      4,
      0);

  ASSERT_TRUE(flba1.equals(&flba2));
  ASSERT_FALSE(flba1.equals(&flba3));
  ASSERT_FALSE(flba1.equals(&flba4));
  ASSERT_FALSE(flba1.equals(&flba5));
}

TEST_F(TestPrimitiveNode, PhysicalLogicalMapping) {
  ASSERT_NO_THROW(
      PrimitiveNode::make(
          "foo", Repetition::kRequired, Type::kInt32, ConvertedType::kInt32));
  ASSERT_NO_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kByteArray,
          ConvertedType::kJson));
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo", Repetition::kRequired, Type::kInt32, ConvertedType::kJson),
      ParquetException);
  ASSERT_NO_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kInt64,
          ConvertedType::kTimestampMillis));
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo", Repetition::kRequired, Type::kInt32, ConvertedType::kInt64),
      ParquetException);
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo", Repetition::kRequired, Type::kByteArray, ConvertedType::kInt8),
      ParquetException);
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kByteArray,
          ConvertedType::kInterval),
      ParquetException);
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kEnum),
      ParquetException);
  ASSERT_NO_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kByteArray,
          ConvertedType::kEnum));
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kDecimal,
          0,
          2,
          4),
      ParquetException);
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kFloat,
          ConvertedType::kDecimal,
          0,
          2,
          4),
      ParquetException);
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kDecimal,
          0,
          4,
          0),
      ParquetException);
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kDecimal,
          10,
          0,
          4),
      ParquetException);
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kDecimal,
          10,
          4,
          -1),
      ParquetException);
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kDecimal,
          10,
          2,
          4),
      ParquetException);
  ASSERT_NO_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kDecimal,
          10,
          6,
          4));
  ASSERT_NO_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kInterval,
          12));
  ASSERT_THROW(
      PrimitiveNode::make(
          "foo",
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kInterval,
          10),
      ParquetException);
}

// ----------------------------------------------------------------------.
// Group node.

class TestGroupNode : public ::testing::Test {
 public:
  NodeVector fields1() {
    NodeVector fields;

    fields.push_back(int32("one", Repetition::kRequired));
    fields.push_back(int64("two"));
    fields.push_back(doubleType("three"));

    return fields;
  }

  NodeVector fields2() {
    // Fields with a duplicate name.
    NodeVector fields;

    fields.push_back(int32("duplicate", Repetition::kRequired));
    fields.push_back(int64("unique"));
    fields.push_back(doubleType("duplicate"));

    return fields;
  }
};

TEST_F(TestGroupNode, Attrs) {
  NodeVector fields = fields1();

  GroupNode node1("foo", Repetition::kRepeated, fields);
  GroupNode node2("bar", Repetition::kOptional, fields, ConvertedType::kList);

  ASSERT_EQ("foo", node1.name());

  ASSERT_TRUE(node1.isGroup());
  ASSERT_FALSE(node1.isPrimitive());

  ASSERT_EQ(fields.size(), node1.fieldCount());

  ASSERT_TRUE(node1.isRepeated());
  ASSERT_TRUE(node2.isOptional());

  ASSERT_EQ(Repetition::kRepeated, node1.repetition());
  ASSERT_EQ(Repetition::kOptional, node2.repetition());

  ASSERT_EQ(Node::kGroup, node1.nodeType());

  // Logical types.
  ASSERT_EQ(ConvertedType::kNone, node1.convertedType());
  ASSERT_EQ(ConvertedType::kList, node2.convertedType());
}

TEST_F(TestGroupNode, equals) {
  NodeVector f1 = fields1();
  NodeVector f2 = fields1();

  GroupNode group1("group", Repetition::kRepeated, f1);
  GroupNode group2("group", Repetition::kRepeated, f2);
  GroupNode group3("group2", Repetition::kRepeated, f2);

  // This is copied in the GroupNode ctor, so this is okay.
  f2.push_back(floatType("four", Repetition::kOptional));
  GroupNode group4("group", Repetition::kRepeated, f2);
  GroupNode group5("group", Repetition::kRepeated, fields1());

  ASSERT_TRUE(group1.equals(&group1));
  ASSERT_TRUE(group1.equals(&group2));
  ASSERT_FALSE(group1.equals(&group3));

  ASSERT_FALSE(group1.equals(&group4));
  ASSERT_FALSE(group5.equals(&group4));
}

TEST_F(TestGroupNode, fieldIndex) {
  NodeVector fields = fields1();
  GroupNode group("group", Repetition::kRequired, fields);
  for (size_t i = 0; i < fields.size(); i++) {
    auto field = group.field(static_cast<int>(i));
    ASSERT_EQ(i, group.fieldIndex(*field));
  }

  // Test a non field node.
  auto nonFieldAlien = int32("alien", Repetition::kRequired); // other name
  auto nonFieldFamiliar = int32("one", Repetition::kRepeated); // other node
  ASSERT_LT(group.fieldIndex(*nonFieldAlien), 0);
  ASSERT_LT(group.fieldIndex(*nonFieldFamiliar), 0);
}

TEST_F(TestGroupNode, FieldIndexDuplicateName) {
  NodeVector fields = fields2();
  GroupNode group("group", Repetition::kRequired, fields);
  for (size_t i = 0; i < fields.size(); i++) {
    auto field = group.field(static_cast<int>(i));
    ASSERT_EQ(i, group.fieldIndex(*field));
  }
}

// ----------------------------------------------------------------------.
// Test convert group.

class TestSchemaConverter : public ::testing::Test {
 public:
  void SetUp() {
    name_ = "parquet_schema";
  }

  void convert(
      const facebook::velox::parquet::thrift::SchemaElement* elements,
      int length) {
    node_ = unflatten(elements, length);
    ASSERT_TRUE(node_->isGroup());
    group_ = static_cast<const GroupNode*>(node_.get());
  }

 protected:
  std::string name_;
  const GroupNode* group_;
  std::unique_ptr<Node> node_;
};

bool checkForParentConsistency(const GroupNode* Node) {
  // Each node should have the group as parent.
  for (int i = 0; i < Node->fieldCount(); i++) {
    const NodePtr& field = Node->field(i);
    if (field->parent() != Node) {
      return false;
    }
    if (field->isGroup()) {
      const GroupNode* group = static_cast<GroupNode*>(field.get());
      if (!checkForParentConsistency(group)) {
        return false;
      }
    }
  }
  return true;
}

TEST_F(TestSchemaConverter, NestedExample) {
  SchemaElement elt;
  std::vector<SchemaElement> elements;
  elements.push_back(newGroup(name_, FieldRepetitionType::REPEATED, 2, 0));

  // A primitive one.
  elements.push_back(
      newPrimitive("a", FieldRepetitionType::REQUIRED, Type::kInt32, 1));

  // A group.
  elements.push_back(newGroup("bag", FieldRepetitionType::OPTIONAL, 1, 2));

  // 3-Level list encoding, by hand.
  elt = newGroup("b", FieldRepetitionType::REPEATED, 1, 3);
  elt.__set_converted_type(
      facebook::velox::parquet::thrift::ConvertedType::LIST);
  elements.push_back(elt);
  elements.push_back(
      newPrimitive("item", FieldRepetitionType::OPTIONAL, Type::kInt64, 4));

  ASSERT_NO_FATAL_FAILURE(
      convert(&elements[0], static_cast<int>(elements.size())));

  // Construct the expected schema.
  NodeVector fields;
  fields.push_back(int32("a", Repetition::kRequired, 1));

  // 3-Level list encoding.
  NodePtr item = int64("item", Repetition::kOptional, 4);
  NodePtr List(
      GroupNode::make(
          "b", Repetition::kRepeated, {item}, ConvertedType::kList, 3));
  NodePtr bag(
      GroupNode::make("bag", Repetition::kOptional, {List}, nullptr, 2));
  fields.push_back(bag);

  NodePtr schema =
      GroupNode::make(name_, Repetition::kRepeated, fields, nullptr, 0);

  ASSERT_TRUE(schema->equals(group_));

  // Check that the parent relationship in each node is consistent.
  ASSERT_EQ(group_->parent(), nullptr);
  ASSERT_TRUE(checkForParentConsistency(group_));
}

TEST_F(TestSchemaConverter, ZeroColumns) {
  // ARROW-3843.
  SchemaElement elements[1];
  elements[0] = newGroup("schema", FieldRepetitionType::REPEATED, 0, 0);
  ASSERT_NO_THROW(convert(elements, 1));
}

TEST_F(TestSchemaConverter, InvalidRoot) {
  // According to the Parquet specification, the first element in the.
  // list<SchemaElement> is a group whose children (and their descendants)
  // Contain all of the rest of the flattened schema elements. If the first.
  // Element is not a group, it is a malformed Parquet file.

  SchemaElement elements[2];
  elements[0] = newPrimitive(
      "not-a-group", FieldRepetitionType::REQUIRED, Type::kInt32, 0);
  ASSERT_THROW(convert(elements, 2), ParquetException);

  // While the Parquet spec indicates that the root group should have REPEATED.
  // Repetition type, some implementations may return REQUIRED or OPTIONAL.
  // Groups as the first element. These tests check that this is okay as a.
  // Practicality matter.
  elements[0] = newGroup("not-repeated", FieldRepetitionType::REQUIRED, 1, 0);
  elements[1] =
      newPrimitive("a", FieldRepetitionType::REQUIRED, Type::kInt32, 1);
  ASSERT_NO_FATAL_FAILURE(convert(elements, 2));

  elements[0] = newGroup("not-repeated", FieldRepetitionType::OPTIONAL, 1, 0);
  ASSERT_NO_FATAL_FAILURE(convert(elements, 2));
}

TEST_F(TestSchemaConverter, NotEnoughChildren) {
  // Throw a ParquetException, but don't core dump or anything.
  SchemaElement elt;
  std::vector<SchemaElement> elements;
  elements.push_back(newGroup(name_, FieldRepetitionType::REPEATED, 2, 0));
  ASSERT_THROW(convert(&elements[0], 1), ParquetException);
}

// ----------------------------------------------------------------------.
// Schema tree flatten / unflatten.

class TestSchemaFlatten : public ::testing::Test {
 public:
  void SetUp() {
    name_ = "parquet_schema";
  }

  void flatten(const GroupNode* schema) {
    toParquet(schema, &elements_);
  }

 protected:
  std::string name_;
  std::vector<facebook::velox::parquet::thrift::SchemaElement> elements_;
};

TEST_F(TestSchemaFlatten, DecimalMetadata) {
  // Checks that DecimalMetadata is only set for DecimalTypes.
  NodePtr Node = PrimitiveNode::make(
      "decimal",
      Repetition::kRequired,
      Type::kInt64,
      ConvertedType::kDecimal,
      -1,
      8,
      4);
  NodePtr group = GroupNode::make(
      "group", Repetition::kRepeated, {Node}, ConvertedType::kList);
  flatten(reinterpret_cast<GroupNode*>(group.get()));
  ASSERT_EQ("decimal", elements_[1].name);
  ASSERT_TRUE(elements_[1].__isset.precision);
  ASSERT_TRUE(elements_[1].__isset.scale);

  elements_.clear();
  // ... Including those created with new logical types.
  Node = PrimitiveNode::make(
      "decimal",
      Repetition::kRequired,
      DecimalLogicalType::make(10, 5),
      Type::kInt64,
      -1);
  group = GroupNode::make(
      "group", Repetition::kRepeated, {Node}, ListLogicalType::make());
  flatten(reinterpret_cast<GroupNode*>(group.get()));
  ASSERT_EQ("decimal", elements_[1].name);
  ASSERT_TRUE(elements_[1].__isset.precision);
  ASSERT_TRUE(elements_[1].__isset.scale);

  elements_.clear();
  // Not for integers with no logical type.
  group = GroupNode::make(
      "group", Repetition::kRepeated, {int64("int64")}, ConvertedType::kList);
  flatten(reinterpret_cast<GroupNode*>(group.get()));
  ASSERT_EQ("int64", elements_[1].name);
  ASSERT_FALSE(elements_[0].__isset.precision);
  ASSERT_FALSE(elements_[0].__isset.scale);
}

TEST_F(TestSchemaFlatten, NestedExample) {
  SchemaElement elt;
  std::vector<SchemaElement> elements;
  elements.push_back(newGroup(name_, FieldRepetitionType::REPEATED, 2, 0));

  // A primitive one.
  elements.push_back(
      newPrimitive("a", FieldRepetitionType::REQUIRED, Type::kInt32, 1));

  // A group.
  elements.push_back(newGroup("bag", FieldRepetitionType::OPTIONAL, 1, 2));

  // 3-Level list encoding, by hand.
  elt = newGroup("b", FieldRepetitionType::REPEATED, 1, 3);
  elt.__set_converted_type(
      facebook::velox::parquet::thrift::ConvertedType::LIST);
  facebook::velox::parquet::thrift::ListType ls;
  facebook::velox::parquet::thrift::LogicalType lt;
  lt.__set_LIST(ls);
  elt.__set_logicalType(lt);
  elements.push_back(elt);
  elements.push_back(
      newPrimitive("item", FieldRepetitionType::OPTIONAL, Type::kInt64, 4));

  // Construct the schema.
  NodeVector fields;
  fields.push_back(int32("a", Repetition::kRequired, 1));

  // 3-Level list encoding.
  NodePtr item = int64("item", Repetition::kOptional, 4);
  NodePtr List(
      GroupNode::make(
          "b", Repetition::kRepeated, {item}, ConvertedType::kList, 3));
  NodePtr bag(
      GroupNode::make("bag", Repetition::kOptional, {List}, nullptr, 2));
  fields.push_back(bag);

  NodePtr schema =
      GroupNode::make(name_, Repetition::kRepeated, fields, nullptr, 0);

  flatten(static_cast<GroupNode*>(schema.get()));
  ASSERT_EQ(elements_.size(), elements.size());
  for (size_t i = 0; i < elements_.size(); i++) {
    ASSERT_EQ(elements_[i], elements[i]);
  }
}

TEST(TestColumnDescriptor, TestAttrs) {
  NodePtr Node = PrimitiveNode::make(
      "name", Repetition::kOptional, Type::kByteArray, ConvertedType::kUtf8);
  ColumnDescriptor descr(Node, 4, 1);

  ASSERT_EQ("name", descr.name());
  ASSERT_EQ(4, descr.maxDefinitionLevel());
  ASSERT_EQ(1, descr.maxRepetitionLevel());

  ASSERT_EQ(Type::kByteArray, descr.physicalType());

  ASSERT_EQ(-1, descr.typeLength());
  const char* expectedDescr = R"(column descriptor = {
  name: name,
  path: ,
  physical_type: BYTE_ARRAY,
  converted_type: UTF8,
  logical_type: String,
  max_definition_level: 4,
  max_repetition_level: 1,
})";
  ASSERT_EQ(expectedDescr, descr.toString());

  // Test FIXED_LEN_BYTE_ARRAY.
  Node = PrimitiveNode::make(
      "name",
      Repetition::kOptional,
      Type::kFixedLenByteArray,
      ConvertedType::kDecimal,
      12,
      10,
      4);
  ColumnDescriptor descr2(Node, 4, 1);

  ASSERT_EQ(Type::kFixedLenByteArray, descr2.physicalType());
  ASSERT_EQ(12, descr2.typeLength());

  expectedDescr = R"(column descriptor = {
  name: name,
  path: ,
  physical_type: FIXED_LEN_BYTE_ARRAY,
  converted_type: DECIMAL,
  logical_type: Decimal(precision=10, scale=4),
  max_definition_level: 4,
  max_repetition_level: 1,
  length: 12,
  precision: 10,
  scale: 4,
})";
  ASSERT_EQ(expectedDescr, descr2.toString());
}

class TestSchemaDescriptor : public ::testing::Test {
 public:
  void SetUp() {}

 protected:
  SchemaDescriptor descr_;
};

TEST_F(TestSchemaDescriptor, InitNonGroup) {
  NodePtr Node =
      PrimitiveNode::make("field", Repetition::kOptional, Type::kInt32);

  ASSERT_THROW(descr_.init(Node), ParquetException);
}

TEST_F(TestSchemaDescriptor, equals) {
  NodePtr schema;

  NodePtr inta = int32("a", Repetition::kRequired);
  NodePtr intb = int64("b", Repetition::kOptional);
  NodePtr intb2 = int64("b2", Repetition::kOptional);
  NodePtr intc = byteArray("c", Repetition::kRepeated);

  NodePtr item1 = int64("item1", Repetition::kRequired);
  NodePtr item2 = boolean("item2", Repetition::kOptional);
  NodePtr item3 = int32("item3", Repetition::kRepeated);
  NodePtr List(
      GroupNode::make(
          "records",
          Repetition::kRepeated,
          {item1, item2, item3},
          ConvertedType::kList));

  NodePtr bag(GroupNode::make("bag", Repetition::kOptional, {List}));
  NodePtr bag2(GroupNode::make("bag", Repetition::kRequired, {List}));

  SchemaDescriptor descr1;
  descr1.init(
      GroupNode::make(
          "schema", Repetition::kRepeated, {inta, intb, intc, bag}));

  ASSERT_TRUE(descr1.equals(descr1));

  SchemaDescriptor descr2;
  descr2.init(
      GroupNode::make(
          "schema", Repetition::kRepeated, {inta, intb, intc, bag2}));
  ASSERT_FALSE(descr1.equals(descr2));

  SchemaDescriptor descr3;
  descr3.init(
      GroupNode::make(
          "schema", Repetition::kRepeated, {inta, intb2, intc, bag}));
  ASSERT_FALSE(descr1.equals(descr3));

  // Robust to name of parent node.
  SchemaDescriptor descr4;
  descr4.init(
      GroupNode::make(
          "SCHEMA", Repetition::kRepeated, {inta, intb, intc, bag}));
  ASSERT_TRUE(descr1.equals(descr4));

  SchemaDescriptor descr5;
  descr5.init(
      GroupNode::make(
          "schema", Repetition::kRepeated, {inta, intb, intc, bag, intb2}));
  ASSERT_FALSE(descr1.equals(descr5));

  // Different max repetition / definition levels.
  ColumnDescriptor col1(inta, 5, 1);
  ColumnDescriptor col2(inta, 6, 1);
  ColumnDescriptor col3(inta, 5, 2);

  ASSERT_TRUE(col1.equals(col1));
  ASSERT_FALSE(col1.equals(col2));
  ASSERT_FALSE(col1.equals(col3));
}

TEST_F(TestSchemaDescriptor, buildTree) {
  NodeVector fields;
  NodePtr schema;

  NodePtr inta = int32("a", Repetition::kRequired);
  fields.push_back(inta);
  fields.push_back(int64("b", Repetition::kOptional));
  fields.push_back(byteArray("c", Repetition::kRepeated));

  // 3-Level list encoding.
  NodePtr item1 = int64("item1", Repetition::kRequired);
  NodePtr item2 = boolean("item2", Repetition::kOptional);
  NodePtr item3 = int32("item3", Repetition::kRepeated);
  NodePtr List(
      GroupNode::make(
          "records",
          Repetition::kRepeated,
          {item1, item2, item3},
          ConvertedType::kList));
  NodePtr bag(GroupNode::make("bag", Repetition::kOptional, {List}));
  fields.push_back(bag);

  schema = GroupNode::make("schema", Repetition::kRepeated, fields);

  descr_.init(schema);

  int nleaves = 6;

  // 6 Leaves.
  ASSERT_EQ(nleaves, descr_.numColumns());

  //                             Mdef mrep.
  // Required int32 a            0    0.
  // Optional int64 b            1    0.
  // Repeated byte_array c       1    1.
  // Optional group bag          1    0.
  //   Repeated group records    2    1.
  //     Required int64 item1    2    1.
  //     Optional boolean item2  3    1.
  //     Repeated int32 item3    3    2.
  int16_t exMaxDefLevels[6] = {0, 1, 1, 2, 3, 3};
  int16_t exMaxRepLevels[6] = {0, 0, 1, 1, 1, 2};

  for (int i = 0; i < nleaves; ++i) {
    const ColumnDescriptor* col = descr_.column(i);
    EXPECT_EQ(exMaxDefLevels[i], col->maxDefinitionLevel()) << i;
    EXPECT_EQ(exMaxRepLevels[i], col->maxRepetitionLevel()) << i;
  }

  ASSERT_EQ(descr_.column(0)->path()->toDotString(), "a");
  ASSERT_EQ(descr_.column(1)->path()->toDotString(), "b");
  ASSERT_EQ(descr_.column(2)->path()->toDotString(), "c");
  ASSERT_EQ(descr_.column(3)->path()->toDotString(), "bag.records.item1");
  ASSERT_EQ(descr_.column(4)->path()->toDotString(), "bag.records.item2");
  ASSERT_EQ(descr_.column(5)->path()->toDotString(), "bag.records.item3");

  for (int i = 0; i < nleaves; ++i) {
    auto col = descr_.column(i);
    ASSERT_EQ(i, descr_.columnIndex(*col->schemaNode()));
  }

  // Test non-column nodes find.
  NodePtr nonColumnAlien = int32("alien", Repetition::kRequired); // other path
  NodePtr nonColumnFamiliar = int32("a", Repetition::kRepeated); // other node
  ASSERT_LT(descr_.columnIndex(*nonColumnAlien), 0);
  ASSERT_LT(descr_.columnIndex(*nonColumnFamiliar), 0);

  ASSERT_EQ(inta.get(), descr_.getColumnRoot(0));
  ASSERT_EQ(bag.get(), descr_.getColumnRoot(3));
  ASSERT_EQ(bag.get(), descr_.getColumnRoot(4));
  ASSERT_EQ(bag.get(), descr_.getColumnRoot(5));

  ASSERT_EQ(schema.get(), descr_.groupNode());

  // Init clears the leaves.
  descr_.init(schema);
  ASSERT_EQ(nleaves, descr_.numColumns());
}

TEST_F(TestSchemaDescriptor, hasRepeatedFields) {
  NodeVector fields;
  NodePtr schema;

  NodePtr inta = int32("a", Repetition::kRequired);
  fields.push_back(inta);
  fields.push_back(int64("b", Repetition::kOptional));
  fields.push_back(byteArray("c", Repetition::kRepeated));

  schema = GroupNode::make("schema", Repetition::kRepeated, fields);
  descr_.init(schema);
  ASSERT_EQ(true, descr_.hasRepeatedFields());

  // 3-Level list encoding.
  NodePtr item1 = int64("item1", Repetition::kRequired);
  NodePtr item2 = boolean("item2", Repetition::kOptional);
  NodePtr item3 = int32("item3", Repetition::kRepeated);
  NodePtr List(
      GroupNode::make(
          "records",
          Repetition::kRepeated,
          {item1, item2, item3},
          ConvertedType::kList));
  NodePtr bag(GroupNode::make("bag", Repetition::kOptional, {List}));
  fields.push_back(bag);

  schema = GroupNode::make("schema", Repetition::kRepeated, fields);
  descr_.init(schema);
  ASSERT_EQ(true, descr_.hasRepeatedFields());

  // 3-Level list encoding.
  NodePtr itemKey = int64("key", Repetition::kRequired);
  NodePtr itemValue = boolean("value", Repetition::kOptional);
  NodePtr Map(
      GroupNode::make(
          "map",
          Repetition::kRepeated,
          {itemKey, itemValue},
          ConvertedType::kMap));
  NodePtr myMap(GroupNode::make("my_map", Repetition::kOptional, {Map}));
  fields.push_back(myMap);

  schema = GroupNode::make("schema", Repetition::kRepeated, fields);
  descr_.init(schema);
  ASSERT_EQ(true, descr_.hasRepeatedFields());
  ASSERT_EQ(true, descr_.hasRepeatedFields());
}

static std::string print(const NodePtr& Node) {
  std::stringstream ss;
  printSchema(Node.get(), ss);
  return ss.str();
}

TEST(TestSchemaPrinter, Examples) {
  // Test schema 1.
  NodeVector fields;
  fields.push_back(int32("a", Repetition::kRequired, 1));

  // 3-Level list encoding.
  NodePtr item1 = int64("item1", Repetition::kOptional, 4);
  NodePtr item2 = boolean("item2", Repetition::kRequired, 5);
  NodePtr List(
      GroupNode::make(
          "b", Repetition::kRepeated, {item1, item2}, ConvertedType::kList, 3));
  NodePtr bag(
      GroupNode::make("bag", Repetition::kOptional, {List}, nullptr, 2));
  fields.push_back(bag);

  fields.push_back(
      PrimitiveNode::make(
          "c",
          Repetition::kRequired,
          Type::kInt32,
          ConvertedType::kDecimal,
          -1,
          3,
          2,
          6));

  fields.push_back(
      PrimitiveNode::make(
          "d",
          Repetition::kRequired,
          DecimalLogicalType::make(10, 5),
          Type::kInt64,
          -1,
          7));

  NodePtr schema =
      GroupNode::make("schema", Repetition::kRepeated, fields, nullptr, 0);

  std::string result = print(schema);

  std::string expected = R"(repeated group field_id=0 schema {
  required int32 field_id=1 a;
  optional group field_id=2 bag {
    repeated group field_id=3 b (List) {
      optional int64 field_id=4 item1;
      required boolean field_id=5 item2;
    }
  }
  required int32 field_id=6 c (Decimal(precision=3, scale=2));
  required int64 field_id=7 d (Decimal(precision=10, scale=5));
}
)";
  ASSERT_EQ(expected, result);
}

static void confirmFactoryEquivalence(
    ConvertedType::type convertedType,
    const std::shared_ptr<const LogicalType>& fromMake,
    std::function<bool(const std::shared_ptr<const LogicalType>&)>
        checkIsType) {
  std::shared_ptr<const LogicalType> fromConvertedType =
      LogicalType::fromConvertedType(convertedType);
  ASSERT_EQ(fromConvertedType->type(), fromMake->type())
      << fromMake->toString()
      << " logical types unexpectedly do not match on type";
  ASSERT_TRUE(fromConvertedType->equals(*fromMake))
      << fromMake->toString() << " logical types unexpectedly not equivalent";
  ASSERT_TRUE(checkIsType(fromConvertedType))
      << fromConvertedType->toString()
      << " logical type (from converted type) does not have expected type property";
  ASSERT_TRUE(checkIsType(fromMake))
      << fromMake->toString()
      << " logical type (from Make()) does not have expected type property";
  return;
}

TEST(TestLogicalTypeConstruction, FactoryEquivalence) {
  // For each legacy converted type, ensure that the equivalent logical type.
  // object can be obtained from either the base class's FromConvertedType()
  // Factory method or the logical type type class's Make() method (accessed
  // via. Convenience methods on the base class) and that these logical type
  // objects. Are equivalent.

  struct ConfirmFactoryEquivalenceArguments {
    ConvertedType::type convertedType;
    std::shared_ptr<const LogicalType> logicalType;
    std::function<bool(const std::shared_ptr<const LogicalType>&)> checkIsType;
  };

  auto checkIsString =
      [](const std::shared_ptr<const LogicalType>& logicalType) {
        return logicalType->isString();
      };
  auto checkIsMap = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isMap();
  };
  auto checkIsList = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isList();
  };
  auto checkIsenum = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isEnum();
  };
  auto checkIsDate = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isDate();
  };
  auto checkIsTime = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isTime();
  };
  auto checkIsTimestamp =
      [](const std::shared_ptr<const LogicalType>& logicalType) {
        return logicalType->isTimestamp();
      };
  auto checkIsInt = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isInt();
  };
  auto checkIsJson = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isJson();
  };
  auto checkIsBson = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isBson();
  };
  auto checkIsInterval =
      [](const std::shared_ptr<const LogicalType>& logicalType) {
        return logicalType->isInterval();
      };
  auto checkIsNone = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isNone();
  };

  std::vector<ConfirmFactoryEquivalenceArguments> cases = {
      {ConvertedType::kUtf8, LogicalType::string(), checkIsString},
      {ConvertedType::kMap, LogicalType::map(), checkIsMap},
      {ConvertedType::kMapKeyValue, LogicalType::map(), checkIsMap},
      {ConvertedType::kList, LogicalType::list(), checkIsList},
      {ConvertedType::kEnum, LogicalType::enumType(), checkIsenum},
      {ConvertedType::kDate, LogicalType::date(), checkIsDate},
      {ConvertedType::kTimeMillis,
       LogicalType::time(true, LogicalType::TimeUnit::kMillis),
       checkIsTime},
      {ConvertedType::kTimeMicros,
       LogicalType::time(true, LogicalType::TimeUnit::kMicros),
       checkIsTime},
      {ConvertedType::kTimestampMillis,
       LogicalType::timestamp(true, LogicalType::TimeUnit::kMillis),
       checkIsTimestamp},
      {ConvertedType::kTimestampMicros,
       LogicalType::timestamp(true, LogicalType::TimeUnit::kMicros),
       checkIsTimestamp},
      {ConvertedType::kUint8, LogicalType::intType(8, false), checkIsInt},
      {ConvertedType::kUint16, LogicalType::intType(16, false), checkIsInt},
      {ConvertedType::kUint32, LogicalType::intType(32, false), checkIsInt},
      {ConvertedType::kUint64, LogicalType::intType(64, false), checkIsInt},
      {ConvertedType::kInt8, LogicalType::intType(8, true), checkIsInt},
      {ConvertedType::kInt16, LogicalType::intType(16, true), checkIsInt},
      {ConvertedType::kInt32, LogicalType::intType(32, true), checkIsInt},
      {ConvertedType::kInt64, LogicalType::intType(64, true), checkIsInt},
      {ConvertedType::kJson, LogicalType::json(), checkIsJson},
      {ConvertedType::kBson, LogicalType::bson(), checkIsBson},
      {ConvertedType::kInterval, LogicalType::interval(), checkIsInterval},
      {ConvertedType::kNone, LogicalType::none(), checkIsNone}};

  for (const ConfirmFactoryEquivalenceArguments& c : cases) {
    confirmFactoryEquivalence(c.convertedType, c.logicalType, c.checkIsType);
  }

  // ConvertedType::kDecimal, LogicalType::decimal, is_decimal.
  schema::DecimalMetadata convertedDecimalMetadata;
  convertedDecimalMetadata.isset = true;
  convertedDecimalMetadata.precision = 10;
  convertedDecimalMetadata.scale = 4;
  std::shared_ptr<const LogicalType> fromConvertedType =
      LogicalType::fromConvertedType(
          ConvertedType::kDecimal, convertedDecimalMetadata);
  std::shared_ptr<const LogicalType> fromMake = LogicalType::decimal(10, 4);
  ASSERT_EQ(fromConvertedType->type(), fromMake->type());
  ASSERT_TRUE(fromConvertedType->equals(*fromMake));
  ASSERT_TRUE(fromConvertedType->isDecimal());
  ASSERT_TRUE(fromMake->isDecimal());
  ASSERT_TRUE(LogicalType::decimal(16)->equals(*LogicalType::decimal(16, 0)));
}

static void confirmConvertedTypeCompatibility(
    const std::shared_ptr<const LogicalType>& original,
    ConvertedType::type expectedConvertedType) {
  ASSERT_TRUE(original->isValid())
      << original->toString() << " logical type unexpectedly is not valid";
  schema::DecimalMetadata convertedDecimalMetadata;
  ConvertedType::type convertedType =
      original->toConvertedType(&convertedDecimalMetadata);
  ASSERT_EQ(convertedType, expectedConvertedType)
      << original->toString()
      << " logical type unexpectedly returns incorrect converted type";
  ASSERT_FALSE(convertedDecimalMetadata.isset)
      << original->toString()
      << " logical type unexpectedly returns converted decimal metadata that is set";
  ASSERT_TRUE(original->isCompatible(convertedType, convertedDecimalMetadata))
      << original->toString()
      << " logical type unexpectedly is incompatible with converted type and decimal "
         "metadata it returned";
  ASSERT_FALSE(original->isCompatible(convertedType, {true, 1, 1}))
      << original->toString()
      << " logical type unexpectedly is compatible with converted decimal metadata that "
         "is "
         "set";
  ASSERT_TRUE(original->isCompatible(convertedType))
      << original->toString()
      << " logical type unexpectedly is incompatible with converted type it returned";
  std::shared_ptr<const LogicalType> reconstructed =
      LogicalType::fromConvertedType(convertedType, convertedDecimalMetadata);
  ASSERT_TRUE(reconstructed->isValid())
      << "Reconstructed " << reconstructed->toString()
      << " logical type unexpectedly is not valid";
  ASSERT_TRUE(reconstructed->equals(*original))
      << "Reconstructed logical type (" << reconstructed->toString()
      << ") unexpectedly not equivalent to original logical type ("
      << original->toString() << ")";
  return;
}

TEST(TestLogicalTypeConstruction, ConvertedTypeCompatibility) {
  // For each legacy converted type, ensure that the equivalent logical type.
  // Emits correct, compatible converted type information and that the emitted.
  // Information can be used to reconstruct another equivalent logical type.

  struct ExpectedConvertedType {
    std::shared_ptr<const LogicalType> logicalType;
    ConvertedType::type convertedType;
  };

  std::vector<ExpectedConvertedType> cases = {
      {LogicalType::string(), ConvertedType::kUtf8},
      {LogicalType::map(), ConvertedType::kMap},
      {LogicalType::list(), ConvertedType::kList},
      {LogicalType::enumType(), ConvertedType::kEnum},
      {LogicalType::date(), ConvertedType::kDate},
      {LogicalType::time(true, LogicalType::TimeUnit::kMillis),
       ConvertedType::kTimeMillis},
      {LogicalType::time(true, LogicalType::TimeUnit::kMicros),
       ConvertedType::kTimeMicros},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMillis),
       ConvertedType::kTimestampMillis},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMicros),
       ConvertedType::kTimestampMicros},
      {LogicalType::intType(8, false), ConvertedType::kUint8},
      {LogicalType::intType(16, false), ConvertedType::kUint16},
      {LogicalType::intType(32, false), ConvertedType::kUint32},
      {LogicalType::intType(64, false), ConvertedType::kUint64},
      {LogicalType::intType(8, true), ConvertedType::kInt8},
      {LogicalType::intType(16, true), ConvertedType::kInt16},
      {LogicalType::intType(32, true), ConvertedType::kInt32},
      {LogicalType::intType(64, true), ConvertedType::kInt64},
      {LogicalType::json(), ConvertedType::kJson},
      {LogicalType::bson(), ConvertedType::kBson},
      {LogicalType::interval(), ConvertedType::kInterval},
      {LogicalType::none(), ConvertedType::kNone}};

  for (const ExpectedConvertedType& c : cases) {
    confirmConvertedTypeCompatibility(c.logicalType, c.convertedType);
  }

  // Special cases ...

  std::shared_ptr<const LogicalType> original;
  ConvertedType::type convertedType;
  schema::DecimalMetadata convertedDecimalMetadata;
  std::shared_ptr<const LogicalType> reconstructed;

  // DECIMAL.
  std::memset(
      &convertedDecimalMetadata, 0x00, sizeof(convertedDecimalMetadata));
  original = LogicalType::decimal(6, 2);
  ASSERT_TRUE(original->isValid());
  convertedType = original->toConvertedType(&convertedDecimalMetadata);
  ASSERT_EQ(convertedType, ConvertedType::kDecimal);
  ASSERT_TRUE(convertedDecimalMetadata.isset);
  ASSERT_EQ(convertedDecimalMetadata.precision, 6);
  ASSERT_EQ(convertedDecimalMetadata.scale, 2);
  ASSERT_TRUE(original->isCompatible(convertedType, convertedDecimalMetadata));
  reconstructed =
      LogicalType::fromConvertedType(convertedType, convertedDecimalMetadata);
  ASSERT_TRUE(reconstructed->isValid());
  ASSERT_TRUE(reconstructed->equals(*original));

  // Undefined.
  original = UndefinedLogicalType::make();
  ASSERT_TRUE(original->isInvalid());
  ASSERT_FALSE(original->isValid());
  convertedType = original->toConvertedType(&convertedDecimalMetadata);
  ASSERT_EQ(convertedType, ConvertedType::kUndefined);
  ASSERT_FALSE(convertedDecimalMetadata.isset);
  ASSERT_TRUE(original->isCompatible(convertedType, convertedDecimalMetadata));
  ASSERT_TRUE(original->isCompatible(convertedType));
  reconstructed =
      LogicalType::fromConvertedType(convertedType, convertedDecimalMetadata);
  ASSERT_TRUE(reconstructed->isInvalid());
  ASSERT_TRUE(reconstructed->equals(*original));
}

static void confirmNewTypeIncompatibility(
    const std::shared_ptr<const LogicalType>& logicalType,
    std::function<bool(const std::shared_ptr<const LogicalType>&)>
        checkIsType) {
  ASSERT_TRUE(logicalType->isValid())
      << logicalType->toString() << " logical type unexpectedly is not valid";
  ASSERT_TRUE(checkIsType(logicalType))
      << logicalType->toString()
      << " logical type is not expected logical type";
  schema::DecimalMetadata convertedDecimalMetadata;
  ConvertedType::type convertedType =
      logicalType->toConvertedType(&convertedDecimalMetadata);
  ASSERT_EQ(convertedType, ConvertedType::kNone)
      << logicalType->toString()
      << " logical type converted type unexpectedly is not NONE";
  ASSERT_FALSE(convertedDecimalMetadata.isset)
      << logicalType->toString()
      << " logical type converted decimal metadata unexpectedly is set";
  return;
}

TEST(TestLogicalTypeConstruction, NewTypeIncompatibility) {
  // For each new logical type, ensure that the type.
  // Correctly reports that it has no legacy equivalent.

  struct ConfirmNewTypeIncompatibilityArguments {
    std::shared_ptr<const LogicalType> logicalType;
    std::function<bool(const std::shared_ptr<const LogicalType>&)> checkIsType;
  };

  auto checkIsUuid = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isUuid();
  };
  auto checkIsNull = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isNull();
  };
  auto checkIsTime = [](const std::shared_ptr<const LogicalType>& logicalType) {
    return logicalType->isTime();
  };
  auto checkIsTimestamp =
      [](const std::shared_ptr<const LogicalType>& logicalType) {
        return logicalType->isTimestamp();
      };

  std::vector<ConfirmNewTypeIncompatibilityArguments> cases = {
      {LogicalType::uuid(), checkIsUuid},
      {LogicalType::nullType(), checkIsNull},
      {LogicalType::time(false, LogicalType::TimeUnit::kMillis), checkIsTime},
      {LogicalType::time(false, LogicalType::TimeUnit::kMicros), checkIsTime},
      {LogicalType::time(false, LogicalType::TimeUnit::kNanos), checkIsTime},
      {LogicalType::time(true, LogicalType::TimeUnit::kNanos), checkIsTime},
      {LogicalType::timestamp(false, LogicalType::TimeUnit::kNanos),
       checkIsTimestamp},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kNanos),
       checkIsTimestamp},
  };

  for (const ConfirmNewTypeIncompatibilityArguments& c : cases) {
    confirmNewTypeIncompatibility(c.logicalType, c.checkIsType);
  }
}

TEST(TestLogicalTypeConstruction, FactoryExceptions) {
  // Ensure that logical type construction catches invalid arguments.

  std::vector<std::function<void()>> cases = {
      []() {
        TimeLogicalType::make(true, LogicalType::TimeUnit::kUnknown);
      }, // Invalid TimeUnit
      []() {
        TimestampLogicalType::make(true, LogicalType::TimeUnit::kUnknown);
      }, // Invalid TimeUnit
      []() { IntLogicalType::make(-1, false); }, // Invalid bit width
      []() { IntLogicalType::make(0, false); }, // Invalid bit width
      []() { IntLogicalType::make(1, false); }, // Invalid bit width
      []() { IntLogicalType::make(65, false); }, // Invalid bit width
      []() { DecimalLogicalType::make(-1); }, // Invalid precision
      []() { DecimalLogicalType::make(0); }, // Invalid precision
      []() { DecimalLogicalType::make(0, 0); }, // Invalid precision
      []() { DecimalLogicalType::make(10, -1); }, // Invalid scale
      []() { DecimalLogicalType::make(10, 11); } // Invalid scale
  };

  for (auto f : cases) {
    ASSERT_ANY_THROW(f());
  }
}

static void confirmLogicalTypeProperties(
    const std::shared_ptr<const LogicalType>& logicalType,
    bool nested,
    bool serialized,
    bool valid) {
  ASSERT_TRUE(logicalType->isNested() == nested)
      << logicalType->toString()
      << " logical type has incorrect nested() property";
  ASSERT_TRUE(logicalType->isSerialized() == serialized)
      << logicalType->toString()
      << " logical type has incorrect serialized() property";
  ASSERT_TRUE(logicalType->isValid() == valid)
      << logicalType->toString()
      << " logical type has incorrect valid() property";
  ASSERT_TRUE(logicalType->isNonnested() != nested)
      << logicalType->toString()
      << " logical type has incorrect nonnested() property";
  ASSERT_TRUE(logicalType->isInvalid() != valid)
      << logicalType->toString()
      << " logical type has incorrect invalid() property";
  return;
}

TEST(TestLogicalTypeOperation, LogicalTypeProperties) {
  // For each logical type, ensure that the correct general properties are.
  // Reported.

  struct ExpectedProperties {
    std::shared_ptr<const LogicalType> logicalType;
    bool nested;
    bool serialized;
    bool valid;
  };

  std::vector<ExpectedProperties> cases = {
      {StringLogicalType::make(), false, true, true},
      {MapLogicalType::make(), true, true, true},
      {ListLogicalType::make(), true, true, true},
      {EnumLogicalType::make(), false, true, true},
      {DecimalLogicalType::make(16, 6), false, true, true},
      {DateLogicalType::make(), false, true, true},
      {TimeLogicalType::make(true, LogicalType::TimeUnit::kMicros),
       false,
       true,
       true},
      {TimestampLogicalType::make(true, LogicalType::TimeUnit::kMicros),
       false,
       true,
       true},
      {IntervalLogicalType::make(), false, true, true},
      {IntLogicalType::make(8, false), false, true, true},
      {IntLogicalType::make(64, true), false, true, true},
      {NullLogicalType::make(), false, true, true},
      {JsonLogicalType::make(), false, true, true},
      {BsonLogicalType::make(), false, true, true},
      {UuidLogicalType::make(), false, true, true},
      {NoLogicalType::make(), false, false, true},
  };

  for (const ExpectedProperties& c : cases) {
    confirmLogicalTypeProperties(
        c.logicalType, c.nested, c.serialized, c.valid);
  }
}

static constexpr int PHYSICAL_TYPE_COUNT = 8;

static Type::type physicalType[PHYSICAL_TYPE_COUNT] = {
    Type::kBoolean,
    Type::kInt32,
    Type::kInt64,
    Type::kInt96,
    Type::kFloat,
    Type::kDouble,
    Type::kByteArray,
    Type::kFixedLenByteArray};

static void confirmSinglePrimitiveTypeApplicability(
    const std::shared_ptr<const LogicalType>& logicalType,
    Type::type applicableType) {
  for (int i = 0; i < PHYSICAL_TYPE_COUNT; ++i) {
    if (physicalType[i] == applicableType) {
      ASSERT_TRUE(logicalType->isApplicable(physicalType[i]))
          << logicalType->toString()
          << " logical type unexpectedly inapplicable to physical type "
          << typeToString(physicalType[i]);
    } else {
      ASSERT_FALSE(logicalType->isApplicable(physicalType[i]))
          << logicalType->toString()
          << " logical type unexpectedly applicable to physical type "
          << typeToString(physicalType[i]);
    }
  }
  return;
}

static void confirmAnyPrimitiveTypeApplicability(
    const std::shared_ptr<const LogicalType>& logicalType) {
  for (int i = 0; i < PHYSICAL_TYPE_COUNT; ++i) {
    ASSERT_TRUE(logicalType->isApplicable(physicalType[i]))
        << logicalType->toString()
        << " logical type unexpectedly inapplicable to physical type "
        << typeToString(physicalType[i]);
  }
  return;
}

static void confirmNoPrimitiveTypeApplicability(
    const std::shared_ptr<const LogicalType>& logicalType) {
  for (int i = 0; i < PHYSICAL_TYPE_COUNT; ++i) {
    ASSERT_FALSE(logicalType->isApplicable(physicalType[i]))
        << logicalType->toString()
        << " logical type unexpectedly applicable to physical type "
        << typeToString(physicalType[i]);
  }
  return;
}

TEST(TestLogicalTypeOperation, LogicalTypeApplicability) {
  // Check that each logical type correctly reports which.
  // Underlying primitive type(s) it can be applied to.

  struct ExpectedApplicability {
    std::shared_ptr<const LogicalType> logicalType;
    Type::type applicableType;
  };

  std::vector<ExpectedApplicability> singleTypeCases = {
      {LogicalType::string(), Type::kByteArray},
      {LogicalType::enumType(), Type::kByteArray},
      {LogicalType::date(), Type::kInt32},
      {LogicalType::time(true, LogicalType::TimeUnit::kMillis), Type::kInt32},
      {LogicalType::time(true, LogicalType::TimeUnit::kMicros), Type::kInt64},
      {LogicalType::time(true, LogicalType::TimeUnit::kNanos), Type::kInt64},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMillis),
       Type::kInt64},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMicros),
       Type::kInt64},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kNanos),
       Type::kInt64},
      {LogicalType::intType(8, false), Type::kInt32},
      {LogicalType::intType(16, false), Type::kInt32},
      {LogicalType::intType(32, false), Type::kInt32},
      {LogicalType::intType(64, false), Type::kInt64},
      {LogicalType::intType(8, true), Type::kInt32},
      {LogicalType::intType(16, true), Type::kInt32},
      {LogicalType::intType(32, true), Type::kInt32},
      {LogicalType::intType(64, true), Type::kInt64},
      {LogicalType::json(), Type::kByteArray},
      {LogicalType::bson(), Type::kByteArray}};

  for (const ExpectedApplicability& c : singleTypeCases) {
    confirmSinglePrimitiveTypeApplicability(c.logicalType, c.applicableType);
  }

  std::vector<std::shared_ptr<const LogicalType>> noTypeCases = {
      LogicalType::map(), LogicalType::list()};

  for (auto c : noTypeCases) {
    confirmNoPrimitiveTypeApplicability(c);
  }

  std::vector<std::shared_ptr<const LogicalType>> anyTypeCases = {
      LogicalType::nullType(),
      LogicalType::none(),
      UndefinedLogicalType::make()};

  for (auto c : anyTypeCases) {
    confirmAnyPrimitiveTypeApplicability(c);
  }

  // Fixed binary, exact length cases ...

  struct InapplicableType {
    Type::type physicalType;
    int physicalLength;
  };

  std::vector<InapplicableType> inapplicableTypes = {
      {Type::kFixedLenByteArray, 8},
      {Type::kFixedLenByteArray, 20},
      {Type::kBoolean, -1},
      {Type::kInt32, -1},
      {Type::kInt64, -1},
      {Type::kInt96, -1},
      {Type::kFloat, -1},
      {Type::kDouble, -1},
      {Type::kByteArray, -1}};

  std::shared_ptr<const LogicalType> logicalType;

  logicalType = LogicalType::interval();
  ASSERT_TRUE(logicalType->isApplicable(Type::kFixedLenByteArray, 12));
  for (const InapplicableType& t : inapplicableTypes) {
    ASSERT_FALSE(logicalType->isApplicable(t.physicalType, t.physicalLength));
  }

  logicalType = LogicalType::uuid();
  ASSERT_TRUE(logicalType->isApplicable(Type::kFixedLenByteArray, 16));
  for (const InapplicableType& t : inapplicableTypes) {
    ASSERT_FALSE(logicalType->isApplicable(t.physicalType, t.physicalLength));
  }
}

TEST(TestLogicalTypeOperation, DecimalLogicalTypeApplicability) {
  // Check that the decimal logical type correctly reports which.
  // Underlying primitive type(s) it can be applied to.

  std::shared_ptr<const LogicalType> logicalType;

  for (int32_t precision = 1; precision <= 9; ++precision) {
    logicalType = DecimalLogicalType::make(precision, 0);
    ASSERT_TRUE(logicalType->isApplicable(Type::kInt32))
        << logicalType->toString()
        << " unexpectedly inapplicable to physical type INT32";
  }
  logicalType = DecimalLogicalType::make(10, 0);
  ASSERT_FALSE(logicalType->isApplicable(Type::kInt32))
      << logicalType->toString()
      << " unexpectedly applicable to physical type INT32";

  for (int32_t precision = 1; precision <= 18; ++precision) {
    logicalType = DecimalLogicalType::make(precision, 0);
    ASSERT_TRUE(logicalType->isApplicable(Type::kInt64))
        << logicalType->toString()
        << " unexpectedly inapplicable to physical type INT64";
  }
  logicalType = DecimalLogicalType::make(19, 0);
  ASSERT_FALSE(logicalType->isApplicable(Type::kInt64))
      << logicalType->toString()
      << " unexpectedly applicable to physical type INT64";

  for (int32_t precision = 1; precision <= 36; ++precision) {
    logicalType = DecimalLogicalType::make(precision, 0);
    ASSERT_TRUE(logicalType->isApplicable(Type::kByteArray))
        << logicalType->toString()
        << " unexpectedly inapplicable to physical type BYTE_ARRAY";
  }

  struct PrecisionLimits {
    int32_t physicalLength;
    int32_t precisionLimit;
  };

  std::vector<PrecisionLimits> cases = {
      {1, 2},
      {2, 4},
      {3, 6},
      {4, 9},
      {8, 18},
      {10, 23},
      {16, 38},
      {20, 47},
      {32, 76}};

  for (const PrecisionLimits& c : cases) {
    int32_t precision;
    for (precision = 1; precision <= c.precisionLimit; ++precision) {
      logicalType = DecimalLogicalType::make(precision, 0);
      ASSERT_TRUE(
          logicalType->isApplicable(Type::kFixedLenByteArray, c.physicalLength))
          << logicalType->toString()
          << " unexpectedly inapplicable to physical type FIXED_LEN_BYTE_ARRAY with "
             "length "
          << c.physicalLength;
    }
    logicalType = DecimalLogicalType::make(precision, 0);
    ASSERT_FALSE(
        logicalType->isApplicable(Type::kFixedLenByteArray, c.physicalLength))
        << logicalType->toString()
        << " unexpectedly applicable to physical type FIXED_LEN_BYTE_ARRAY with length "
        << c.physicalLength;
  }

  ASSERT_FALSE((DecimalLogicalType::make(16, 6))->isApplicable(Type::kBoolean));
  ASSERT_FALSE((DecimalLogicalType::make(16, 6))->isApplicable(Type::kFloat));
  ASSERT_FALSE((DecimalLogicalType::make(16, 6))->isApplicable(Type::kDouble));
}

TEST(TestLogicalTypeOperation, LogicalTypeRepresentation) {
  // Ensure that each logical type prints a correct string and.
  // JSON representation.

  struct ExpectedRepresentation {
    std::shared_ptr<const LogicalType> logicalType;
    const char* stringRepresentation;
    const char* jsonRepresentation;
  };

  std::vector<ExpectedRepresentation> cases = {
      {UndefinedLogicalType::make(), "Undefined", R"({"Type": "Undefined"})"},
      {LogicalType::string(), "String", R"({"Type": "String"})"},
      {LogicalType::map(), "Map", R"({"Type": "Map"})"},
      {LogicalType::list(), "List", R"({"Type": "List"})"},
      {LogicalType::enumType(), "Enum", R"({"Type": "Enum"})"},
      {LogicalType::decimal(10, 4),
       "Decimal(precision=10, scale=4)",
       R"({"Type": "Decimal", "precision": 10, "scale": 4})"},
      {LogicalType::decimal(10),
       "Decimal(precision=10, scale=0)",
       R"({"Type": "Decimal", "precision": 10, "scale": 0})"},
      {LogicalType::date(), "Date", R"({"Type": "Date"})"},
      {LogicalType::time(true, LogicalType::TimeUnit::kMillis),
       "Time(isAdjustedToUTC=true, timeUnit=milliseconds)",
       R"({"Type": "Time", "isAdjustedToUTC": true, "timeUnit": "milliseconds"})"},
      {LogicalType::time(true, LogicalType::TimeUnit::kMicros),
       "Time(isAdjustedToUTC=true, timeUnit=microseconds)",
       R"({"Type": "Time", "isAdjustedToUTC": true, "timeUnit": "microseconds"})"},
      {LogicalType::time(true, LogicalType::TimeUnit::kNanos),
       "Time(isAdjustedToUTC=true, timeUnit=nanoseconds)",
       R"({"Type": "Time", "isAdjustedToUTC": true, "timeUnit": "nanoseconds"})"},
      {LogicalType::time(false, LogicalType::TimeUnit::kMillis),
       "Time(isAdjustedToUTC=false, timeUnit=milliseconds)",
       R"({"Type": "Time", "isAdjustedToUTC": false, "timeUnit": "milliseconds"})"},
      {LogicalType::time(false, LogicalType::TimeUnit::kMicros),
       "Time(isAdjustedToUTC=false, timeUnit=microseconds)",
       R"({"Type": "Time", "isAdjustedToUTC": false, "timeUnit": "microseconds"})"},
      {LogicalType::time(false, LogicalType::TimeUnit::kNanos),
       "Time(isAdjustedToUTC=false, timeUnit=nanoseconds)",
       R"({"Type": "Time", "isAdjustedToUTC": false, "timeUnit": "nanoseconds"})"},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMillis),
       "Timestamp(isAdjustedToUTC=true, timeUnit=milliseconds, "
       "is_from_converted_type=false, force_set_converted_type=false)",
       R"({"Type": "Timestamp", "isAdjustedToUTC": true, "timeUnit": "milliseconds", )"
       R"("isFromConvertedType": false, "forceSetConvertedType": false})"},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMicros),
       "Timestamp(isAdjustedToUTC=true, timeUnit=microseconds, "
       "is_from_converted_type=false, force_set_converted_type=false)",
       R"({"Type": "Timestamp", "isAdjustedToUTC": true, "timeUnit": "microseconds", )"
       R"("isFromConvertedType": false, "forceSetConvertedType": false})"},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kNanos),
       "Timestamp(isAdjustedToUTC=true, timeUnit=nanoseconds, "
       "is_from_converted_type=false, force_set_converted_type=false)",
       R"({"Type": "Timestamp", "isAdjustedToUTC": true, "timeUnit": "nanoseconds", )"
       R"("isFromConvertedType": false, "forceSetConvertedType": false})"},
      {LogicalType::timestamp(
           false, LogicalType::TimeUnit::kMillis, true, true),
       "Timestamp(isAdjustedToUTC=false, timeUnit=milliseconds, "
       "is_from_converted_type=true, force_set_converted_type=true)",
       R"({"Type": "Timestamp", "isAdjustedToUTC": false, "timeUnit": "milliseconds", )"
       R"("isFromConvertedType": true, "forceSetConvertedType": true})"},
      {LogicalType::timestamp(false, LogicalType::TimeUnit::kMicros),
       "Timestamp(isAdjustedToUTC=false, timeUnit=microseconds, "
       "is_from_converted_type=false, force_set_converted_type=false)",
       R"({"Type": "Timestamp", "isAdjustedToUTC": false, "timeUnit": "microseconds", )"
       R"("isFromConvertedType": false, "forceSetConvertedType": false})"},
      {LogicalType::timestamp(false, LogicalType::TimeUnit::kNanos),
       "Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, "
       "is_from_converted_type=false, force_set_converted_type=false)",
       R"({"Type": "Timestamp", "isAdjustedToUTC": false, "timeUnit": "nanoseconds", )"
       R"("isFromConvertedType": false, "forceSetConvertedType": false})"},
      {LogicalType::interval(), "Interval", R"({"Type": "Interval"})"},
      {LogicalType::intType(8, false),
       "Int(bitWidth=8, isSigned=false)",
       R"({"Type": "int", "bitWidth": 8, "isSigned": false})"},
      {LogicalType::intType(16, false),
       "Int(bitWidth=16, isSigned=false)",
       R"({"Type": "int", "bitWidth": 16, "isSigned": false})"},
      {LogicalType::intType(32, false),
       "Int(bitWidth=32, isSigned=false)",
       R"({"Type": "int", "bitWidth": 32, "isSigned": false})"},
      {LogicalType::intType(64, false),
       "Int(bitWidth=64, isSigned=false)",
       R"({"Type": "int", "bitWidth": 64, "isSigned": false})"},
      {LogicalType::intType(8, true),
       "Int(bitWidth=8, isSigned=true)",
       R"({"Type": "int", "bitWidth": 8, "isSigned": true})"},
      {LogicalType::intType(16, true),
       "Int(bitWidth=16, isSigned=true)",
       R"({"Type": "int", "bitWidth": 16, "isSigned": true})"},
      {LogicalType::intType(32, true),
       "Int(bitWidth=32, isSigned=true)",
       R"({"Type": "int", "bitWidth": 32, "isSigned": true})"},
      {LogicalType::intType(64, true),
       "Int(bitWidth=64, isSigned=true)",
       R"({"Type": "int", "bitWidth": 64, "isSigned": true})"},
      {LogicalType::nullType(), "Null", R"({"Type": "Null"})"},
      {LogicalType::json(), "JSON", R"({"Type": "JSON"})"},
      {LogicalType::bson(), "BSON", R"({"Type": "BSON"})"},
      {LogicalType::uuid(), "UUID", R"({"Type": "UUID"})"},
      {LogicalType::none(), "None", R"({"Type": "None"})"},
  };

  for (const ExpectedRepresentation& c : cases) {
    ASSERT_STREQ(c.logicalType->toString().c_str(), c.stringRepresentation);
    ASSERT_STREQ(c.logicalType->toJson().c_str(), c.jsonRepresentation);
  }
}

TEST(TestLogicalTypeOperation, LogicalTypeSortOrder) {
  // Ensure that each logical type reports the correct sort order.

  struct ExpectedSortOrder {
    std::shared_ptr<const LogicalType> logicalType;
    SortOrder::type sortOrder;
  };

  std::vector<ExpectedSortOrder> cases = {
      {LogicalType::string(), SortOrder::kUnsigned},
      {LogicalType::map(), SortOrder::kUnknown},
      {LogicalType::list(), SortOrder::kUnknown},
      {LogicalType::enumType(), SortOrder::kUnsigned},
      {LogicalType::decimal(8, 2), SortOrder::kSigned},
      {LogicalType::date(), SortOrder::kSigned},
      {LogicalType::time(true, LogicalType::TimeUnit::kMillis),
       SortOrder::kSigned},
      {LogicalType::time(true, LogicalType::TimeUnit::kMicros),
       SortOrder::kSigned},
      {LogicalType::time(true, LogicalType::TimeUnit::kNanos),
       SortOrder::kSigned},
      {LogicalType::time(false, LogicalType::TimeUnit::kMillis),
       SortOrder::kSigned},
      {LogicalType::time(false, LogicalType::TimeUnit::kMicros),
       SortOrder::kSigned},
      {LogicalType::time(false, LogicalType::TimeUnit::kNanos),
       SortOrder::kSigned},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMillis),
       SortOrder::kSigned},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMicros),
       SortOrder::kSigned},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kNanos),
       SortOrder::kSigned},
      {LogicalType::timestamp(false, LogicalType::TimeUnit::kMillis),
       SortOrder::kSigned},
      {LogicalType::timestamp(false, LogicalType::TimeUnit::kMicros),
       SortOrder::kSigned},
      {LogicalType::timestamp(false, LogicalType::TimeUnit::kNanos),
       SortOrder::kSigned},
      {LogicalType::interval(), SortOrder::kUnknown},
      {LogicalType::intType(8, false), SortOrder::kUnsigned},
      {LogicalType::intType(16, false), SortOrder::kUnsigned},
      {LogicalType::intType(32, false), SortOrder::kUnsigned},
      {LogicalType::intType(64, false), SortOrder::kUnsigned},
      {LogicalType::intType(8, true), SortOrder::kSigned},
      {LogicalType::intType(16, true), SortOrder::kSigned},
      {LogicalType::intType(32, true), SortOrder::kSigned},
      {LogicalType::intType(64, true), SortOrder::kSigned},
      {LogicalType::nullType(), SortOrder::kUnknown},
      {LogicalType::json(), SortOrder::kUnsigned},
      {LogicalType::bson(), SortOrder::kUnsigned},
      {LogicalType::uuid(), SortOrder::kUnsigned},
      {LogicalType::none(), SortOrder::kUnknown}};

  for (const ExpectedSortOrder& c : cases) {
    ASSERT_EQ(c.logicalType->sortOrder(), c.sortOrder)
        << c.logicalType->toString()
        << " logical type has incorrect sort order";
  }
}

static void confirmPrimitiveNodeFactoryEquivalence(
    const std::shared_ptr<const LogicalType>& logicalType,
    ConvertedType::type convertedType,
    Type::type physicalType,
    int physicalLength,
    int precision,
    int scale) {
  std::string name = "something";
  Repetition::type repetition = Repetition::kRequired;
  NodePtr fromConvertedType = PrimitiveNode::make(
      name,
      repetition,
      physicalType,
      convertedType,
      physicalLength,
      precision,
      scale);
  NodePtr fromLogicalType = PrimitiveNode::make(
      name, repetition, logicalType, physicalType, physicalLength);
  ASSERT_TRUE(fromConvertedType->equals(fromLogicalType.get()))
      << "Primitive node constructed with converted type "
      << convertedTypeToString(convertedType)
      << " unexpectedly not equivalent to primitive node constructed with logical "
         "type "
      << logicalType->toString();
  return;
}

static void confirmGroupNodeFactoryEquivalence(
    std::string name,
    const std::shared_ptr<const LogicalType>& logicalType,
    ConvertedType::type convertedType) {
  Repetition::type repetition = Repetition::kOptional;
  NodePtr fromConvertedType =
      GroupNode::make(name, repetition, {}, convertedType);
  NodePtr fromLogicalType = GroupNode::make(name, repetition, {}, logicalType);
  ASSERT_TRUE(fromConvertedType->equals(fromLogicalType.get()))
      << "Group node constructed with converted type "
      << convertedTypeToString(convertedType)
      << " unexpectedly not equivalent to group node constructed with logical type "
      << logicalType->toString();
  return;
}

TEST(TestSchemaNodeCreation, FactoryEquivalence) {
  // Ensure that the Node factory methods produce equivalent results regardless.
  // Of whether they are given a converted type or a logical type.

  // Primitive nodes ...

  struct PrimitiveNodeFactoryArguments {
    std::shared_ptr<const LogicalType> logicalType;
    ConvertedType::type convertedType;
    Type::type physicalType;
    int physicalLength;
    int precision;
    int scale;
  };

  std::vector<PrimitiveNodeFactoryArguments> cases = {
      {LogicalType::string(),
       ConvertedType::kUtf8,
       Type::kByteArray,
       -1,
       -1,
       -1},
      {LogicalType::enumType(),
       ConvertedType::kEnum,
       Type::kByteArray,
       -1,
       -1,
       -1},
      {LogicalType::decimal(16, 6),
       ConvertedType::kDecimal,
       Type::kInt64,
       -1,
       16,
       6},
      {LogicalType::date(), ConvertedType::kDate, Type::kInt32, -1, -1, -1},
      {LogicalType::time(true, LogicalType::TimeUnit::kMillis),
       ConvertedType::kTimeMillis,
       Type::kInt32,
       -1,
       -1,
       -1},
      {LogicalType::time(true, LogicalType::TimeUnit::kMicros),
       ConvertedType::kTimeMicros,
       Type::kInt64,
       -1,
       -1,
       -1},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMillis),
       ConvertedType::kTimestampMillis,
       Type::kInt64,
       -1,
       -1,
       -1},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMicros),
       ConvertedType::kTimestampMicros,
       Type::kInt64,
       -1,
       -1,
       -1},
      {LogicalType::interval(),
       ConvertedType::kInterval,
       Type::kFixedLenByteArray,
       12,
       -1,
       -1},
      {LogicalType::intType(8, false),
       ConvertedType::kUint8,
       Type::kInt32,
       -1,
       -1,
       -1},
      {LogicalType::intType(8, true),
       ConvertedType::kInt8,
       Type::kInt32,
       -1,
       -1,
       -1},
      {LogicalType::intType(16, false),
       ConvertedType::kUint16,
       Type::kInt32,
       -1,
       -1,
       -1},
      {LogicalType::intType(16, true),
       ConvertedType::kInt16,
       Type::kInt32,
       -1,
       -1,
       -1},
      {LogicalType::intType(32, false),
       ConvertedType::kUint32,
       Type::kInt32,
       -1,
       -1,
       -1},
      {LogicalType::intType(32, true),
       ConvertedType::kInt32,
       Type::kInt32,
       -1,
       -1,
       -1},
      {LogicalType::intType(64, false),
       ConvertedType::kUint64,
       Type::kInt64,
       -1,
       -1,
       -1},
      {LogicalType::intType(64, true),
       ConvertedType::kInt64,
       Type::kInt64,
       -1,
       -1,
       -1},
      {LogicalType::json(), ConvertedType::kJson, Type::kByteArray, -1, -1, -1},
      {LogicalType::bson(), ConvertedType::kBson, Type::kByteArray, -1, -1, -1},
      {LogicalType::none(), ConvertedType::kNone, Type::kInt64, -1, -1, -1}};

  for (const PrimitiveNodeFactoryArguments& c : cases) {
    confirmPrimitiveNodeFactoryEquivalence(
        c.logicalType,
        c.convertedType,
        c.physicalType,
        c.physicalLength,
        c.precision,
        c.scale);
  }

  // Group nodes ...
  confirmGroupNodeFactoryEquivalence(
      "map", LogicalType::map(), ConvertedType::kMap);
  confirmGroupNodeFactoryEquivalence(
      "list", LogicalType::list(), ConvertedType::kList);
}

TEST(TestSchemaNodeCreation, FactoryExceptions) {
  // Ensure that the Node factory method that accepts a logical type refuses to.
  // Create an object if compatibility conditions are not met.

  // Nested logical type on non-group node ...
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "map", Repetition::kRequired, MapLogicalType::make(), Type::kInt64));
  // Incompatible primitive type ...
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "string",
          Repetition::kRequired,
          StringLogicalType::make(),
          Type::kBoolean));
  // Incompatible primitive length ...
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "interval",
          Repetition::kRequired,
          IntervalLogicalType::make(),
          Type::kFixedLenByteArray,
          11));
  // Scale is greater than precision.
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "decimal",
          Repetition::kRequired,
          DecimalLogicalType::make(10, 11),
          Type::kInt64));
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "decimal",
          Repetition::kRequired,
          DecimalLogicalType::make(17, 18),
          Type::kInt64));
  // Primitive too small for given precision ...
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "decimal",
          Repetition::kRequired,
          DecimalLogicalType::make(16, 6),
          Type::kInt32));
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "decimal",
          Repetition::kRequired,
          DecimalLogicalType::make(10, 9),
          Type::kInt32));
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "decimal",
          Repetition::kRequired,
          DecimalLogicalType::make(19, 17),
          Type::kInt64));
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "decimal",
          Repetition::kRequired,
          DecimalLogicalType::make(308, 6),
          Type::kFixedLenByteArray,
          128));
  // Length is too long.
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "decimal",
          Repetition::kRequired,
          DecimalLogicalType::make(10, 6),
          Type::kFixedLenByteArray,
          891723283));

  // Incompatible primitive length ...
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "uuid",
          Repetition::kRequired,
          UuidLogicalType::make(),
          Type::kFixedLenByteArray,
          64));
  // Non-positive length argument for fixed length binary ...
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "negative_length",
          Repetition::kRequired,
          NoLogicalType::make(),
          Type::kFixedLenByteArray,
          -16));
  // Non-positive length argument for fixed length binary ...
  ASSERT_ANY_THROW(
      PrimitiveNode::make(
          "zero_length",
          Repetition::kRequired,
          NoLogicalType::make(),
          Type::kFixedLenByteArray,
          0));
  // Non-nested logical type on group node ...
  ASSERT_ANY_THROW(
      GroupNode::make(
          "list", Repetition::kRepeated, {}, JsonLogicalType::make()));

  // Nullptr logical type arguments convert to
  // NoLogicalType/ConvertedType::kNone.
  std::shared_ptr<const LogicalType> empty;
  NodePtr Node;
  ASSERT_NO_THROW(
      Node = PrimitiveNode::make(
          "value", Repetition::kRequired, empty, Type::kDouble));
  ASSERT_TRUE(Node->logicalType()->isNone());
  ASSERT_EQ(Node->convertedType(), ConvertedType::kNone);
  ASSERT_NO_THROW(
      Node = GroupNode::make("items", Repetition::kRepeated, {}, empty));
  ASSERT_TRUE(Node->logicalType()->isNone());
  ASSERT_EQ(Node->convertedType(), ConvertedType::kNone);

  // Invalid ConvertedType in deserialized element ...
  Node = PrimitiveNode::make(
      "string",
      Repetition::kRequired,
      StringLogicalType::make(),
      Type::kByteArray);
  ASSERT_EQ(Node->logicalType()->type(), LogicalType::Type::kString);
  ASSERT_TRUE(Node->logicalType()->isValid());
  ASSERT_TRUE(Node->logicalType()->isSerialized());
  facebook::velox::parquet::thrift::SchemaElement stringIntermediary;
  Node->toParquet(&stringIntermediary);
  // ... Corrupt the Thrift intermediary ....
  stringIntermediary.logicalType.__isset.STRING = false;
  ASSERT_ANY_THROW(Node = PrimitiveNode::fromParquet(&stringIntermediary));

  // Invalid TimeUnit in deserialized TimeLogicalType ...
  Node = PrimitiveNode::make(
      "time",
      Repetition::kRequired,
      TimeLogicalType::make(true, LogicalType::TimeUnit::kNanos),
      Type::kInt64);
  facebook::velox::parquet::thrift::SchemaElement timeIntermediary;
  Node->toParquet(&timeIntermediary);
  // ... Corrupt the Thrift intermediary ....
  timeIntermediary.logicalType.TIME.unit.__isset.NANOS = false;
  ASSERT_ANY_THROW(PrimitiveNode::fromParquet(&timeIntermediary));

  // Invalid TimeUnit in deserialized TimestampLogicalType ...
  Node = PrimitiveNode::make(
      "timestamp",
      Repetition::kRequired,
      TimestampLogicalType::make(true, LogicalType::TimeUnit::kNanos),
      Type::kInt64);
  facebook::velox::parquet::thrift::SchemaElement timestampIntermediary;
  Node->toParquet(&timestampIntermediary);
  // ... Corrupt the Thrift intermediary ....
  timestampIntermediary.logicalType.TIMESTAMP.unit.__isset.NANOS = false;
  ASSERT_ANY_THROW(PrimitiveNode::fromParquet(&timestampIntermediary));
}

struct SchemaElementConstructionArguments {
  SchemaElementConstructionArguments(
      std::string name,
      std::shared_ptr<const LogicalType> logicalType,
      Type::type physicalType,
      int physicalLength,
      bool expectConvertedType,
      ConvertedType::type convertedType,
      bool expectLogicaltype,
      std::function<bool()> checkLogicaltype)
      : name(std::move(name)),
        logicalType(std::move(logicalType)),
        physicalType(physicalType),
        physicalLength(physicalLength),
        expectConvertedType(expectConvertedType),
        convertedType(convertedType),
        expectLogicaltype(expectLogicaltype),
        checkLogicaltype(std::move(checkLogicaltype)) {}

  std::string name;
  std::shared_ptr<const LogicalType> logicalType;
  Type::type physicalType;
  int physicalLength;
  bool expectConvertedType;
  ConvertedType::type convertedType;
  bool expectLogicaltype;
  std::function<bool()> checkLogicaltype;
};

struct LegacySchemaElementConstructionArguments {
  std::string name;
  Type::type physicalType;
  int physicalLength;
  bool expectConvertedType;
  ConvertedType::type convertedType;
  bool expectLogicaltype;
  std::function<bool()> checkLogicaltype;
};

class TestSchemaElementConstruction : public ::testing::Test {
 public:
  TestSchemaElementConstruction* reconstruct(
      const SchemaElementConstructionArguments& c) {
    // Make node, create serializable Thrift object from it ...
    node_ = PrimitiveNode::make(
        c.name,
        Repetition::kRequired,
        c.logicalType,
        c.physicalType,
        c.physicalLength);
    element_.reset(new facebook::velox::parquet::thrift::SchemaElement);
    node_->toParquet(element_.get());

    // ... Then set aside some values for later inspection.
    name_ = c.name;
    expectConvertedType_ = c.expectConvertedType;
    convertedType_ = c.convertedType;
    expectLogicaltype_ = c.expectLogicaltype;
    checkLogicaltype_ = c.checkLogicaltype;
    return this;
  }

  TestSchemaElementConstruction* legacyReconstruct(
      const LegacySchemaElementConstructionArguments& c) {
    // Make node, create serializable Thrift object from it ...
    node_ = PrimitiveNode::make(
        c.name,
        Repetition::kRequired,
        c.physicalType,
        c.convertedType,
        c.physicalLength);
    element_.reset(new facebook::velox::parquet::thrift::SchemaElement);
    node_->toParquet(element_.get());

    // ... Then set aside some values for later inspection.
    name_ = c.name;
    expectConvertedType_ = c.expectConvertedType;
    convertedType_ = c.convertedType;
    expectLogicaltype_ = c.expectLogicaltype;
    checkLogicaltype_ = c.checkLogicaltype;
    return this;
  }

  void inspect() {
    ASSERT_EQ(element_->name, name_);
    if (expectConvertedType_) {
      ASSERT_TRUE(element_->__isset.converted_type)
          << node_->logicalType()->toString()
          << " logical type unexpectedly failed to generate a converted type in the "
             "Thrift "
             "intermediate object";
      ASSERT_EQ(element_->converted_type, toThrift(convertedType_))
          << node_->logicalType()->toString()
          << " logical type unexpectedly failed to generate correct converted type in "
             "the "
             "Thrift intermediate object";
    } else {
      ASSERT_FALSE(element_->__isset.converted_type)
          << node_->logicalType()->toString()
          << " logical type unexpectedly generated a converted type in the Thrift "
             "intermediate object";
    }
    if (expectLogicaltype_) {
      ASSERT_TRUE(element_->__isset.logicalType)
          << node_->logicalType()->toString()
          << " logical type unexpectedly failed to genverate a logicalType in the Thrift "
             "intermediate object";
      ASSERT_TRUE(checkLogicaltype_())
          << node_->logicalType()->toString()
          << " logical type generated incorrect logicalType "
             "settings in the Thrift intermediate object";
    } else {
      ASSERT_FALSE(element_->__isset.logicalType)
          << node_->logicalType()->toString()
          << " logical type unexpectedly generated a logicalType in the Thrift "
             "intermediate object";
    }
    return;
  }

 protected:
  NodePtr node_;
  std::unique_ptr<facebook::velox::parquet::thrift::SchemaElement> element_;
  std::string name_;
  bool expectConvertedType_;
  ConvertedType::type
      convertedType_; // expected converted type in Thrift object
  bool expectLogicaltype_;
  std::function<bool()>
      checkLogicaltype_; // specialized (by logical type)
                         // LogicalType check for Thrift object.
};

/*
 * The Test*SchemaElementConstruction suites confirm that the logical type
 * and converted type members of the Thrift intermediate message object
 * (facebook::velox::parquet::thrift::SchemaElement) that is created upon
 * serialization of an annotated schema node are correctly populated.
 */

TEST_F(TestSchemaElementConstruction, SimpleCases) {
  auto checkNothing = []() {
    return true;
  }; // used for logical types that don't expect a logicalType to be set

  std::vector<SchemaElementConstructionArguments> cases = {
      {"string",
       LogicalType::string(),
       Type::kByteArray,
       -1,
       true,
       ConvertedType::kUtf8,
       true,
       [this]() { return element_->logicalType.__isset.STRING; }},
      {"enum",
       LogicalType::enumType(),
       Type::kByteArray,
       -1,
       true,
       ConvertedType::kEnum,
       true,
       [this]() { return element_->logicalType.__isset.ENUM; }},
      {"date",
       LogicalType::date(),
       Type::kInt32,
       -1,
       true,
       ConvertedType::kDate,
       true,
       [this]() { return element_->logicalType.__isset.DATE; }},
      {"interval",
       LogicalType::interval(),
       Type::kFixedLenByteArray,
       12,
       true,
       ConvertedType::kInterval,
       false,
       checkNothing},
      {"null",
       LogicalType::nullType(),
       Type::kDouble,
       -1,
       false,
       ConvertedType::kNa,
       true,
       [this]() { return element_->logicalType.__isset.UNKNOWN; }},
      {"json",
       LogicalType::json(),
       Type::kByteArray,
       -1,
       true,
       ConvertedType::kJson,
       true,
       [this]() { return element_->logicalType.__isset.JSON; }},
      {"bson",
       LogicalType::bson(),
       Type::kByteArray,
       -1,
       true,
       ConvertedType::kBson,
       true,
       [this]() { return element_->logicalType.__isset.BSON; }},
      {"uuid",
       LogicalType::uuid(),
       Type::kFixedLenByteArray,
       16,
       false,
       ConvertedType::kNa,
       true,
       [this]() { return element_->logicalType.__isset.UUID; }},
      {"none",
       LogicalType::none(),
       Type::kInt64,
       -1,
       false,
       ConvertedType::kNa,
       false,
       checkNothing}};

  for (const SchemaElementConstructionArguments& c : cases) {
    this->reconstruct(c)->inspect();
  }

  std::vector<LegacySchemaElementConstructionArguments> legacyCases = {
      {"timestamp_ms",
       Type::kInt64,
       -1,
       true,
       ConvertedType::kTimestampMillis,
       false,
       checkNothing},
      {"timestamp_us",
       Type::kInt64,
       -1,
       true,
       ConvertedType::kTimestampMicros,
       false,
       checkNothing},
  };

  for (const LegacySchemaElementConstructionArguments& c : legacyCases) {
    this->legacyReconstruct(c)->inspect();
  }
}

class TestDecimalSchemaElementConstruction
    : public TestSchemaElementConstruction {
 public:
  TestDecimalSchemaElementConstruction* reconstruct(
      const SchemaElementConstructionArguments& c) {
    TestSchemaElementConstruction::reconstruct(c);
    const auto& decimalLogicalType =
        checked_cast<const DecimalLogicalType&>(*c.logicalType);
    precision_ = decimalLogicalType.precision();
    scale_ = decimalLogicalType.scale();
    return this;
  }

  void inspect() {
    TestSchemaElementConstruction::inspect();
    ASSERT_EQ(element_->precision, precision_);
    ASSERT_EQ(element_->scale, scale_);
    ASSERT_EQ(element_->logicalType.DECIMAL.precision, precision_);
    ASSERT_EQ(element_->logicalType.DECIMAL.scale, scale_);
    return;
  }

 protected:
  int32_t precision_;
  int32_t scale_;
};

TEST_F(TestDecimalSchemaElementConstruction, DecimalCases) {
  auto checkDecimal = [this]() {
    return element_->logicalType.__isset.DECIMAL;
  };

  std::vector<SchemaElementConstructionArguments> cases = {
      {"decimal",
       LogicalType::decimal(16, 6),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kDecimal,
       true,
       checkDecimal},
      {"decimal",
       LogicalType::decimal(1, 0),
       Type::kInt32,
       -1,
       true,
       ConvertedType::kDecimal,
       true,
       checkDecimal},
      {"decimal",
       LogicalType::decimal(10),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kDecimal,
       true,
       checkDecimal},
      {"decimal",
       LogicalType::decimal(11, 11),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kDecimal,
       true,
       checkDecimal},
      {"decimal",
       LogicalType::decimal(9, 9),
       Type::kInt32,
       -1,
       true,
       ConvertedType::kDecimal,
       true,
       checkDecimal},
      {"decimal",
       LogicalType::decimal(18, 18),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kDecimal,
       true,
       checkDecimal},
      {"decimal",
       LogicalType::decimal(307, 7),
       Type::kFixedLenByteArray,
       128,
       true,
       ConvertedType::kDecimal,
       true,
       checkDecimal},
      {"decimal",
       LogicalType::decimal(310, 32),
       Type::kFixedLenByteArray,
       129,
       true,
       ConvertedType::kDecimal,
       true,
       checkDecimal},
      {"decimal",
       LogicalType::decimal(2147483645, 2147483645),
       Type::kFixedLenByteArray,
       891723282,
       true,
       ConvertedType::kDecimal,
       true,
       checkDecimal},
  };

  for (const SchemaElementConstructionArguments& c : cases) {
    this->reconstruct(c)->inspect();
  }
}

class TestTemporalSchemaElementConstruction
    : public TestSchemaElementConstruction {
 public:
  template <typename T>
  TestTemporalSchemaElementConstruction* reconstruct(
      const SchemaElementConstructionArguments& c) {
    TestSchemaElementConstruction::reconstruct(c);
    const auto& t = checked_cast<const T&>(*c.logicalType);
    adjusted_ = t.isAdjustedToUtc();
    unit_ = t.timeUnit();
    return this;
  }

  template <typename T>
  void inspect() {
    FAIL() << "Invalid typename specified in test suite";
    return;
  }

 protected:
  bool adjusted_;
  LogicalType::TimeUnit::Unit unit_;
};

template <>
void TestTemporalSchemaElementConstruction::inspect<
    facebook::velox::parquet::thrift::TimeType>() {
  TestSchemaElementConstruction::inspect();
  ASSERT_EQ(element_->logicalType.TIME.isAdjustedToUTC, adjusted_);
  switch (unit_) {
    case LogicalType::TimeUnit::kMillis:
      ASSERT_TRUE(element_->logicalType.TIME.unit.__isset.MILLIS);
      break;
    case LogicalType::TimeUnit::kMicros:
      ASSERT_TRUE(element_->logicalType.TIME.unit.__isset.MICROS);
      break;
    case LogicalType::TimeUnit::kNanos:
      ASSERT_TRUE(element_->logicalType.TIME.unit.__isset.NANOS);
      break;
    case LogicalType::TimeUnit::kUnknown:
    default:
      FAIL() << "Invalid time unit in test case";
  }
  return;
}

template <>
void TestTemporalSchemaElementConstruction::inspect<
    facebook::velox::parquet::thrift::TimestampType>() {
  TestSchemaElementConstruction::inspect();
  ASSERT_EQ(element_->logicalType.TIMESTAMP.isAdjustedToUTC, adjusted_);
  switch (unit_) {
    case LogicalType::TimeUnit::kMillis:
      ASSERT_TRUE(element_->logicalType.TIMESTAMP.unit.__isset.MILLIS);
      break;
    case LogicalType::TimeUnit::kMicros:
      ASSERT_TRUE(element_->logicalType.TIMESTAMP.unit.__isset.MICROS);
      break;
    case LogicalType::TimeUnit::kNanos:
      ASSERT_TRUE(element_->logicalType.TIMESTAMP.unit.__isset.NANOS);
      break;
    case LogicalType::TimeUnit::kUnknown:
    default:
      FAIL() << "Invalid time unit in test case";
  }
  return;
}

TEST_F(TestTemporalSchemaElementConstruction, TemporalCases) {
  auto checkTime = [this]() { return element_->logicalType.__isset.TIME; };

  std::vector<SchemaElementConstructionArguments> timeCases = {
      {"time_T_ms",
       LogicalType::time(true, LogicalType::TimeUnit::kMillis),
       Type::kInt32,
       -1,
       true,
       ConvertedType::kTimeMillis,
       true,
       checkTime},
      {"time_F_ms",
       LogicalType::time(false, LogicalType::TimeUnit::kMillis),
       Type::kInt32,
       -1,
       false,
       ConvertedType::kNa,
       true,
       checkTime},
      {"time_T_us",
       LogicalType::time(true, LogicalType::TimeUnit::kMicros),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kTimeMicros,
       true,
       checkTime},
      {"time_F_us",
       LogicalType::time(false, LogicalType::TimeUnit::kMicros),
       Type::kInt64,
       -1,
       false,
       ConvertedType::kNa,
       true,
       checkTime},
      {"time_T_ns",
       LogicalType::time(true, LogicalType::TimeUnit::kNanos),
       Type::kInt64,
       -1,
       false,
       ConvertedType::kNa,
       true,
       checkTime},
      {"time_F_ns",
       LogicalType::time(false, LogicalType::TimeUnit::kNanos),
       Type::kInt64,
       -1,
       false,
       ConvertedType::kNa,
       true,
       checkTime},
  };

  for (const SchemaElementConstructionArguments& c : timeCases) {
    this->reconstruct<TimeLogicalType>(c)
        ->inspect<facebook::velox::parquet::thrift::TimeType>();
  }

  auto checkTimestamp = [this]() {
    return element_->logicalType.__isset.TIMESTAMP;
  };

  std::vector<SchemaElementConstructionArguments> timestampCases = {
      {"timestamp_T_ms",
       LogicalType::timestamp(true, LogicalType::TimeUnit::kMillis),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kTimestampMillis,
       true,
       checkTimestamp},
      {"timestamp_F_ms",
       LogicalType::timestamp(false, LogicalType::TimeUnit::kMillis),
       Type::kInt64,
       -1,
       false,
       ConvertedType::kNa,
       true,
       checkTimestamp},
      {"timestamp_F_ms_force",
       LogicalType::timestamp(
           false, LogicalType::TimeUnit::kMillis, false, true),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kTimestampMillis,
       true,
       checkTimestamp},
      {"timestamp_T_us",
       LogicalType::timestamp(true, LogicalType::TimeUnit::kMicros),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kTimestampMicros,
       true,
       checkTimestamp},
      {"timestamp_F_us",
       LogicalType::timestamp(false, LogicalType::TimeUnit::kMicros),
       Type::kInt64,
       -1,
       false,
       ConvertedType::kNa,
       true,
       checkTimestamp},
      {"timestamp_F_us_force",
       LogicalType::timestamp(
           false, LogicalType::TimeUnit::kMillis, false, true),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kTimestampMillis,
       true,
       checkTimestamp},
      {"timestamp_T_ns",
       LogicalType::timestamp(true, LogicalType::TimeUnit::kNanos),
       Type::kInt64,
       -1,
       false,
       ConvertedType::kNa,
       true,
       checkTimestamp},
      {"timestamp_F_ns",
       LogicalType::timestamp(false, LogicalType::TimeUnit::kNanos),
       Type::kInt64,
       -1,
       false,
       ConvertedType::kNa,
       true,
       checkTimestamp},
  };

  for (const SchemaElementConstructionArguments& c : timestampCases) {
    this->reconstruct<TimestampLogicalType>(c)
        ->inspect<facebook::velox::parquet::thrift::TimestampType>();
  }
}

class TestIntegerSchemaElementConstruction
    : public TestSchemaElementConstruction {
 public:
  TestIntegerSchemaElementConstruction* reconstruct(
      const SchemaElementConstructionArguments& c) {
    TestSchemaElementConstruction::reconstruct(c);
    const auto& intLogicalType =
        checked_cast<const IntLogicalType&>(*c.logicalType);
    width_ = intLogicalType.bitWidth();
    signed_ = intLogicalType.isSigned();
    return this;
  }

  void inspect() {
    TestSchemaElementConstruction::inspect();
    ASSERT_EQ(element_->logicalType.INTEGER.bitWidth, width_);
    ASSERT_EQ(element_->logicalType.INTEGER.isSigned, signed_);
    return;
  }

 protected:
  int width_;
  bool signed_;
};

TEST_F(TestIntegerSchemaElementConstruction, IntegerCases) {
  auto checkInteger = [this]() {
    return element_->logicalType.__isset.INTEGER;
  };

  std::vector<SchemaElementConstructionArguments> cases = {
      {"uint8",
       LogicalType::intType(8, false),
       Type::kInt32,
       -1,
       true,
       ConvertedType::kUint8,
       true,
       checkInteger},
      {"uint16",
       LogicalType::intType(16, false),
       Type::kInt32,
       -1,
       true,
       ConvertedType::kUint16,
       true,
       checkInteger},
      {"uint32",
       LogicalType::intType(32, false),
       Type::kInt32,
       -1,
       true,
       ConvertedType::kUint32,
       true,
       checkInteger},
      {"uint64",
       LogicalType::intType(64, false),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kUint64,
       true,
       checkInteger},
      {"int8",
       LogicalType::intType(8, true),
       Type::kInt32,
       -1,
       true,
       ConvertedType::kInt8,
       true,
       checkInteger},
      {"int16",
       LogicalType::intType(16, true),
       Type::kInt32,
       -1,
       true,
       ConvertedType::kInt16,
       true,
       checkInteger},
      {"int32",
       LogicalType::intType(32, true),
       Type::kInt32,
       -1,
       true,
       ConvertedType::kInt32,
       true,
       checkInteger},
      {"int64",
       LogicalType::intType(64, true),
       Type::kInt64,
       -1,
       true,
       ConvertedType::kInt64,
       true,
       checkInteger},
  };

  for (const SchemaElementConstructionArguments& c : cases) {
    this->reconstruct(c)->inspect();
  }
}

TEST(TestLogicalTypeSerialization, SchemaElementNestedCases) {
  // Confirm that the intermediate Thrift objects created during node.
  // Serialization contain correct ConvertedType and ConvertedType information.

  NodePtr stringNode = PrimitiveNode::make(
      "string",
      Repetition::kRequired,
      StringLogicalType::make(),
      Type::kByteArray);
  NodePtr dateNode = PrimitiveNode::make(
      "date", Repetition::kRequired, DateLogicalType::make(), Type::kInt32);
  NodePtr jsonNode = PrimitiveNode::make(
      "json", Repetition::kRequired, JsonLogicalType::make(), Type::kByteArray);
  NodePtr uuidNode = PrimitiveNode::make(
      "uuid",
      Repetition::kRequired,
      UuidLogicalType::make(),
      Type::kFixedLenByteArray,
      16);
  NodePtr timestampNode = PrimitiveNode::make(
      "timestamp",
      Repetition::kRequired,
      TimestampLogicalType::make(false, LogicalType::TimeUnit::kNanos),
      Type::kInt64);
  NodePtr intNode = PrimitiveNode::make(
      "int",
      Repetition::kRequired,
      IntLogicalType::make(64, false),
      Type::kInt64);
  NodePtr decimalNode = PrimitiveNode::make(
      "decimal",
      Repetition::kRequired,
      DecimalLogicalType::make(16, 6),
      Type::kInt64);

  NodePtr listNode = GroupNode::make(
      "list",
      Repetition::kRepeated,
      {stringNode,
       dateNode,
       jsonNode,
       uuidNode,
       timestampNode,
       intNode,
       decimalNode},
      ListLogicalType::make());
  std::vector<facebook::velox::parquet::thrift::SchemaElement> listElements;
  toParquet(reinterpret_cast<GroupNode*>(listNode.get()), &listElements);
  ASSERT_EQ(listElements[0].name, "list");
  ASSERT_TRUE(listElements[0].__isset.converted_type);
  ASSERT_TRUE(listElements[0].__isset.logicalType);
  ASSERT_EQ(listElements[0].converted_type, toThrift(ConvertedType::kList));
  ASSERT_TRUE(listElements[0].logicalType.__isset.LIST);
  ASSERT_TRUE(listElements[1].logicalType.__isset.STRING);
  ASSERT_TRUE(listElements[2].logicalType.__isset.DATE);
  ASSERT_TRUE(listElements[3].logicalType.__isset.JSON);
  ASSERT_TRUE(listElements[4].logicalType.__isset.UUID);
  ASSERT_TRUE(listElements[5].logicalType.__isset.TIMESTAMP);
  ASSERT_TRUE(listElements[6].logicalType.__isset.INTEGER);
  ASSERT_TRUE(listElements[7].logicalType.__isset.DECIMAL);

  NodePtr mapNode =
      GroupNode::make("map", Repetition::kRequired, {}, MapLogicalType::make());
  std::vector<facebook::velox::parquet::thrift::SchemaElement> mapElements;
  toParquet(reinterpret_cast<GroupNode*>(mapNode.get()), &mapElements);
  ASSERT_EQ(mapElements[0].name, "map");
  ASSERT_TRUE(mapElements[0].__isset.converted_type);
  ASSERT_TRUE(mapElements[0].__isset.logicalType);
  ASSERT_EQ(mapElements[0].converted_type, toThrift(ConvertedType::kMap));
  ASSERT_TRUE(mapElements[0].logicalType.__isset.MAP);
}

TEST(TestLogicalTypeSerialization, Roundtrips) {
  // Confirm that Thrift serialization-deserialization of nodes with logical.
  // Types produces equivalent reconstituted nodes.

  // Primitive nodes ...
  struct AnnotatedPrimitiveNodeFactoryArguments {
    std::shared_ptr<const LogicalType> logicalType;
    Type::type physicalType;
    int physicalLength;
  };

  std::vector<AnnotatedPrimitiveNodeFactoryArguments> cases = {
      {LogicalType::string(), Type::kByteArray, -1},
      {LogicalType::enumType(), Type::kByteArray, -1},
      {LogicalType::decimal(16, 6), Type::kInt64, -1},
      {LogicalType::date(), Type::kInt32, -1},
      {LogicalType::time(true, LogicalType::TimeUnit::kMillis),
       Type::kInt32,
       -1},
      {LogicalType::time(true, LogicalType::TimeUnit::kMicros),
       Type::kInt64,
       -1},
      {LogicalType::time(true, LogicalType::TimeUnit::kNanos),
       Type::kInt64,
       -1},
      {LogicalType::time(false, LogicalType::TimeUnit::kMillis),
       Type::kInt32,
       -1},
      {LogicalType::time(false, LogicalType::TimeUnit::kMicros),
       Type::kInt64,
       -1},
      {LogicalType::time(false, LogicalType::TimeUnit::kNanos),
       Type::kInt64,
       -1},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMillis),
       Type::kInt64,
       -1},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kMicros),
       Type::kInt64,
       -1},
      {LogicalType::timestamp(true, LogicalType::TimeUnit::kNanos),
       Type::kInt64,
       -1},
      {LogicalType::timestamp(false, LogicalType::TimeUnit::kMillis),
       Type::kInt64,
       -1},
      {LogicalType::timestamp(false, LogicalType::TimeUnit::kMicros),
       Type::kInt64,
       -1},
      {LogicalType::timestamp(false, LogicalType::TimeUnit::kNanos),
       Type::kInt64,
       -1},
      {LogicalType::interval(), Type::kFixedLenByteArray, 12},
      {LogicalType::intType(8, false), Type::kInt32, -1},
      {LogicalType::intType(16, false), Type::kInt32, -1},
      {LogicalType::intType(32, false), Type::kInt32, -1},
      {LogicalType::intType(64, false), Type::kInt64, -1},
      {LogicalType::intType(8, true), Type::kInt32, -1},
      {LogicalType::intType(16, true), Type::kInt32, -1},
      {LogicalType::intType(32, true), Type::kInt32, -1},
      {LogicalType::intType(64, true), Type::kInt64, -1},
      {LogicalType::nullType(), Type::kBoolean, -1},
      {LogicalType::json(), Type::kByteArray, -1},
      {LogicalType::bson(), Type::kByteArray, -1},
      {LogicalType::uuid(), Type::kFixedLenByteArray, 16},
      {LogicalType::none(), Type::kBoolean, -1}};

  for (const AnnotatedPrimitiveNodeFactoryArguments& c : cases) {
    confirmPrimitiveNodeRoundtrip(
        c.logicalType, c.physicalType, c.physicalLength);
  }

  // Group nodes ...
  confirmGroupNodeRoundtrip("map", LogicalType::map());
  confirmGroupNodeRoundtrip("list", LogicalType::list());
}

} // namespace schema

} // namespace facebook::velox::parquet::arrow
