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

// This module contains the logical Parquet-cpp types (independent of Thrift
// structures), schema nodes, and related type tools.

#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

namespace facebook::velox::parquet::arrow {

class SchemaDescriptor;

namespace schema {

class Node;

// List encodings: using the terminology from Impala to define different styles
// of representing logical lists (a.k.a. ARRAY types) in Parquet schemas. Since
// the converted type named in the Parquet metadata is ConvertedType::kList we
// use that terminology here. It also helps distinguish from the *_ARRAY
// primitive types.
//
// One-level encoding: Only allows required lists with required cells.
//   Repeated value_type name.
//
// Two-level encoding: Enables optional lists with only required cells.
//   <Required/optional> group list.
//     Repeated value_type item.
//
// Three-level encoding: Enables optional lists with optional cells.
//   <Required/optional> group bag.
//     Repeated group list.
//       <Required/optional> value_type item.
//
// 2- And 1-level encoding are respectively equivalent to 3-level encoding with
// the non-repeated nodes set to required.
//
// The "official" encoding recommended in the Parquet spec is the 3-level, and
// we use that as the default when creating list types. For semantic
// completeness we allow the other two. Since all types of encodings will occur
// "in the wild" we need to be able to interpret the associated definition
// levels in the context of the actual encoding used in the file.
//
// NB: Some Parquet writers may not set ConvertedType::kList on the repeated
// SchemaElement, which could make things challenging if we are trying to infer
// that a sequence of nodes semantically represents an array according to one
// of these encodings (versus a struct containing an array). We should refuse
// the temptation to guess, as they say.
struct ListEncoding {
  enum type { kOneLevel, kTwoLevel, kThreeLevel };
};

class PARQUET_EXPORT ColumnPath {
 public:
  ColumnPath() : path_() {}
  explicit ColumnPath(const std::vector<std::string>& path) : path_(path) {}
  explicit ColumnPath(std::vector<std::string>&& path)
      : path_(std::move(path)) {}

  static std::shared_ptr<ColumnPath> fromDotString(
      const std::string& dotstring);
  static std::shared_ptr<ColumnPath> fromNode(const Node& Node);

  std::shared_ptr<ColumnPath> extend(const std::string& nodeName) const;
  std::string toDotString() const;
  const std::vector<std::string>& toDotVector() const;

 protected:
  std::vector<std::string> path_;
};

// Base class for logical schema types. A type has a name, repetition level,
// and optionally a logical type (ConvertedType in Parquet metadata parlance)..
class PARQUET_EXPORT Node {
 public:
  enum type { kPrimitive, kGroup };

  virtual ~Node() {}

  bool isPrimitive() const {
    return type_ == Node::kPrimitive;
  }

  bool isGroup() const {
    return type_ == Node::kGroup;
  }

  bool isOptional() const {
    return repetition_ == Repetition::kOptional;
  }

  bool isRepeated() const {
    return repetition_ == Repetition::kRepeated;
  }

  bool isRequired() const {
    return repetition_ == Repetition::kRequired;
  }

  virtual bool equals(const Node* other) const = 0;

  const std::string& name() const {
    return name_;
  }

  Node::type nodeType() const {
    return type_;
  }

  Repetition::type repetition() const {
    return repetition_;
  }

  ConvertedType::type convertedType() const {
    return convertedType_;
  }

  const std::shared_ptr<const LogicalType>& logicalType() const {
    return logicalType_;
  }

  /// \brief The fieldId value for the serialized SchemaElement. If the
  /// fieldId is less than 0 (e.g. -1), it will not be set when serialized to
  /// Thrift.
  int fieldId() const {
    return fieldId_;
  }

  const Node* parent() const {
    return parent_;
  }

  const std::shared_ptr<ColumnPath> path() const;

  virtual void toParquet(void* element) const = 0;

  // Node::Visitor abstract class for walking schemas with the visitor pattern.
  class Visitor {
   public:
    virtual ~Visitor() {}

    virtual void visit(Node* Node) = 0;
  };
  class ConstVisitor {
   public:
    virtual ~ConstVisitor() {}

    virtual void visit(const Node* Node) = 0;
  };

  virtual void visit(Visitor* visitor) = 0;
  virtual void visitConst(ConstVisitor* visitor) const = 0;

 protected:
  friend class GroupNode;

  Node(
      Node::type type,
      const std::string& name,
      Repetition::type repetition,
      ConvertedType::type convertedType = ConvertedType::kNone,
      int fieldId = -1)
      : type_(type),
        name_(name),
        repetition_(repetition),
        convertedType_(convertedType),
        fieldId_(fieldId),
        parent_(NULLPTR) {}

  Node(
      Node::type type,
      const std::string& name,
      Repetition::type repetition,
      std::shared_ptr<const LogicalType> logicalType,
      int fieldId = -1)
      : type_(type),
        name_(name),
        repetition_(repetition),
        logicalType_(std::move(logicalType)),
        fieldId_(fieldId),
        parent_(NULLPTR) {}

  Node::type type_;
  std::string name_;
  Repetition::type repetition_;
  ConvertedType::type convertedType_;
  std::shared_ptr<const LogicalType> logicalType_;
  int fieldId_;
  // Nodes should not be shared, they have a single parent.
  const Node* parent_;

  bool equalsInternal(const Node* other) const;
  void setParent(const Node* pParent);

 private:
  PARQUET_DISALLOW_COPY_AND_ASSIGN(Node);
};

// Save our breath all over the place with these typedefs.
using NodePtr = std::shared_ptr<Node>;
using NodeVector = std::vector<NodePtr>;

// A type that is one of the primitive Parquet storage types. In addition to
// the other type metadata (name, repetition level, logical type), also has the
// physical storage type and their type-specific metadata (byte width, decimal
// parameters).
class PARQUET_EXPORT PrimitiveNode : public Node {
 public:
  static std::unique_ptr<Node> fromParquet(const void* opaqueElement);

  // A field_id -1 (or any negative value) will be serialized as null in Thrift.
  static inline NodePtr make(
      const std::string& name,
      Repetition::type repetition,
      Type::type type,
      ConvertedType::type convertedType = ConvertedType::kNone,
      int length = -1,
      int precision = -1,
      int scale = -1,
      int fieldId = -1) {
    return NodePtr(new PrimitiveNode(
        name,
        repetition,
        type,
        convertedType,
        length,
        precision,
        scale,
        fieldId));
  }

  // If no logical type, pass LogicalType::None() or nullptr.
  // A field_id -1 (or any negative value) will be serialized as null in Thrift.
  static inline NodePtr make(
      const std::string& name,
      Repetition::type repetition,
      std::shared_ptr<const LogicalType> logicalType,
      Type::type primitiveType,
      int primitiveLength = -1,
      int fieldId = -1) {
    return NodePtr(new PrimitiveNode(
        name,
        repetition,
        std::move(logicalType),
        primitiveType,
        primitiveLength,
        fieldId));
  }

  bool equals(const Node* other) const override;

  Type::type physicalType() const {
    return physicalType_;
  }

  ColumnOrder columnOrder() const {
    return columnOrder_;
  }

  void setColumnOrder(ColumnOrder columnOrder) {
    columnOrder_ = columnOrder;
  }

  int32_t typeLength() const {
    return typeLength_;
  }

  const DecimalMetadata& decimalMetadata() const {
    return decimalMetadata_;
  }

  void toParquet(void* element) const override;
  void visit(Visitor* visitor) override;
  void visitConst(ConstVisitor* visitor) const override;

 private:
  PrimitiveNode(
      const std::string& name,
      Repetition::type repetition,
      Type::type type,
      ConvertedType::type convertedType = ConvertedType::kNone,
      int length = -1,
      int precision = -1,
      int scale = -1,
      int fieldId = -1);

  PrimitiveNode(
      const std::string& name,
      Repetition::type repetition,
      std::shared_ptr<const LogicalType> logicalType,
      Type::type primitiveType,
      int primitiveLength = -1,
      int fieldId = -1);

  Type::type physicalType_;
  int32_t typeLength_;
  DecimalMetadata decimalMetadata_;
  ColumnOrder columnOrder_;

  // For FIXED_LEN_BYTE_ARRAY.
  void setTypeLength(int32_t length) {
    typeLength_ = length;
  }

  bool equalsInternal(const PrimitiveNode* other) const;

  FRIEND_TEST(TestPrimitiveNode, Attrs);
  FRIEND_TEST(TestPrimitiveNode, equals);
  FRIEND_TEST(TestPrimitiveNode, PhysicalLogicalMapping);
  FRIEND_TEST(TestPrimitiveNode, fromParquet);
};

class PARQUET_EXPORT GroupNode : public Node {
 public:
  static std::unique_ptr<Node> fromParquet(
      const void* opaqueElement,
      NodeVector fields = {});

  // A field_id -1 (or any negative value) will be serialized as null in Thrift.
  static inline NodePtr make(
      const std::string& name,
      Repetition::type repetition,
      const NodeVector& fields,
      ConvertedType::type convertedType = ConvertedType::kNone,
      int fieldId = -1) {
    return NodePtr(
        new GroupNode(name, repetition, fields, convertedType, fieldId));
  }

  // If no logical type, pass nullptr.
  // A field_id -1 (or any negative value) will be serialized as null in Thrift.
  static inline NodePtr make(
      const std::string& name,
      Repetition::type repetition,
      const NodeVector& fields,
      std::shared_ptr<const LogicalType> logicalType,
      int fieldId = -1) {
    return NodePtr(
        new GroupNode(name, repetition, fields, logicalType, fieldId));
  }

  bool equals(const Node* other) const override;

  const NodePtr& field(int i) const {
    return fields_[i];
  }
  // Get the index of a field by its name, or negative value if not found.
  // If several fields share the same name, it is unspecified which one
  // is returned.
  int fieldIndex(const std::string& name) const;
  // Get the index of a field by its node, or negative value if not found.
  int fieldIndex(const Node& node) const;

  int fieldCount() const {
    return static_cast<int>(fields_.size());
  }

  void toParquet(void* element) const override;
  void visit(Visitor* visitor) override;
  void visitConst(ConstVisitor* visitor) const override;

  /// \brief Return true if this node or any child node has REPEATED repetition
  /// type.
  bool hasRepeatedFields() const;

 private:
  GroupNode(
      const std::string& name,
      Repetition::type repetition,
      const NodeVector& fields,
      ConvertedType::type convertedType = ConvertedType::kNone,
      int fieldId = -1);

  GroupNode(
      const std::string& name,
      Repetition::type repetition,
      const NodeVector& fields,
      std::shared_ptr<const LogicalType> logicalType,
      int fieldId = -1);

  NodeVector fields_;
  bool equalsInternal(const GroupNode* other) const;

  // Mapping between field name to the field index.
  std::unordered_multimap<std::string, int> fieldNameToIdx_;

  FRIEND_TEST(TestGroupNode, Attrs);
  FRIEND_TEST(TestGroupNode, equals);
  FRIEND_TEST(TestGroupNode, fieldIndex);
  FRIEND_TEST(TestGroupNode, FieldIndexDuplicateName);
};

// ----------------------------------------------------------------------.
// Convenience primitive type factory functions.

#define PRIMITIVE_FACTORY(funcName, TYPE)                  \
  static inline NodePtr funcName(                          \
      const std::string& name,                             \
      Repetition::type repetition = Repetition::kOptional, \
      int fieldId = -1) {                                  \
    return PrimitiveNode::make(                            \
        name,                                              \
        repetition,                                        \
        Type::TYPE,                                        \
        ConvertedType::kNone,                              \
        /*length=*/-1,                                     \
        /*precision=*/-1,                                  \
        /*scale=*/-1,                                      \
        fieldId);                                          \
  }

PRIMITIVE_FACTORY(boolean, kBoolean)
PRIMITIVE_FACTORY(int32, kInt32)
PRIMITIVE_FACTORY(int64, kInt64)
PRIMITIVE_FACTORY(int96, kInt96)
PRIMITIVE_FACTORY(floatType, kFloat)
PRIMITIVE_FACTORY(doubleType, kDouble)
PRIMITIVE_FACTORY(byteArray, kByteArray)

void PARQUET_EXPORT printSchema(
    const schema::Node* schema,
    std::ostream& stream,
    int indentWidth = 2);

} // namespace schema

// The ColumnDescriptor encapsulates information necessary to interpret
// primitive column data in the context of a particular schema. We have to
// examine the node structure of a column's path to the root in the schema tree
// to be able to reassemble the nested structure from the repetition and
// definition levels.
class PARQUET_EXPORT ColumnDescriptor {
 public:
  ColumnDescriptor(
      schema::NodePtr Node,
      int16_t maxDefinitionLevel,
      int16_t maxRepetitionLevel,
      const SchemaDescriptor* schemaDescr = NULLPTR);

  bool equals(const ColumnDescriptor& other) const;

  int16_t maxDefinitionLevel() const {
    return maxDefinitionLevel_;
  }

  int16_t maxRepetitionLevel() const {
    return maxRepetitionLevel_;
  }

  Type::type physicalType() const {
    return primitiveNode_->physicalType();
  }

  ConvertedType::type convertedType() const {
    return primitiveNode_->convertedType();
  }

  const std::shared_ptr<const LogicalType>& logicalType() const {
    return primitiveNode_->logicalType();
  }

  ColumnOrder columnOrder() const {
    return primitiveNode_->columnOrder();
  }

  SortOrder::type sortOrder() const {
    auto la = logicalType();
    auto pt = physicalType();
    return la ? getSortOrder(la, pt) : getSortOrder(convertedType(), pt);
  }

  const std::string& name() const {
    return primitiveNode_->name();
  }

  const std::shared_ptr<schema::ColumnPath> path() const;

  const schema::NodePtr& schemaNode() const {
    return node_;
  }

  std::string toString() const;

  int typeLength() const;

  int typePrecision() const;

  int typeScale() const;

 private:
  schema::NodePtr node_;
  const schema::PrimitiveNode* primitiveNode_;

  int16_t maxDefinitionLevel_;
  int16_t maxRepetitionLevel_;
};

// Container for the converted Parquet schema with a computed information from
// the schema analysis needed for file reading.
//
// * Column index to Node.
// * Max repetition / definition levels for each primitive node.
//
// The ColumnDescriptor objects produced by this class can be used to assist in
// the reconstruction of fully materialized data structures from the
// repetition-definition level encoding of nested data.
//
// TODO(wesm): This object can be recomputed from a Schema.
class PARQUET_EXPORT SchemaDescriptor {
 public:
  SchemaDescriptor() {}
  ~SchemaDescriptor() {}

  // Analyze the schema.
  void init(std::unique_ptr<schema::Node> schema);
  void init(schema::NodePtr schema);

  const ColumnDescriptor* column(int i) const;

  // Get the index of a column by its dotstring path, or negative value if not
  // found. If several columns share the same dotstring path, it is unspecified
  // which one is returned.
  int columnIndex(const std::string& nodePath) const;
  // Get the index of a column by its node, or negative value if not found.
  int columnIndex(const schema::Node& node) const;

  bool equals(const SchemaDescriptor& other, std::ostream* diffOutput = NULLPTR)
      const;

  // The number of physical columns appearing in the file.
  int numColumns() const {
    return static_cast<int>(leaves_.size());
  }

  const schema::NodePtr& schemaRoot() const {
    return schema_;
  }

  const schema::GroupNode* groupNode() const {
    return groupNode_;
  }

  // Returns the root (child of the schema root) node of the leaf (column) node.
  const schema::Node* getColumnRoot(int i) const;

  const std::string& name() const {
    return groupNode_->name();
  }

  std::string toString() const;

  void updateColumnOrders(const std::vector<ColumnOrder>& columnOrders);

  /// \brief Return column index corresponding to a particular
  /// PrimitiveNode. Returns -1 if not found.
  int getColumnIndex(const schema::PrimitiveNode& node) const;

  /// \brief Return true if any field or their children have REPEATED
  /// repetition type.
  bool hasRepeatedFields() const;

 private:
  friend class ColumnDescriptor;

  // Root Node.
  schema::NodePtr schema_;
  // Root Node.
  const schema::GroupNode* groupNode_;

  void buildTree(
      const schema::NodePtr& Node,
      int16_t maxDefLevel,
      int16_t maxRepLevel,
      const schema::NodePtr& base);

  // Result of leaf node / tree analysis.
  std::vector<ColumnDescriptor> leaves_;

  std::unordered_map<const schema::PrimitiveNode*, int> nodeToLeafIndex_;

  // Mapping between leaf nodes and root group of leaf (first node
  // below the schema's root group).
  //
  // For example, the leaf `a.b.c.d` would have a link back to `a`.
  //
  // -- A  <------.
  // -- -- B     |.
  // -- -- -- C  |.
  // -- -- -- -- D.
  std::unordered_map<int, schema::NodePtr> leafToBase_;

  // Mapping between ColumnPath DotString to the leaf index.
  std::unordered_multimap<std::string, int> leafToIdx_;
};

} // namespace facebook::velox::parquet::arrow
