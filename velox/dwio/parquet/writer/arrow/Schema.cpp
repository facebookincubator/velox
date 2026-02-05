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

#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/SchemaInternal.h"
#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "velox/dwio/parquet/writer/arrow/Exception.h"

using facebook::velox::parquet::thrift::SchemaElement;

namespace facebook::velox::parquet::arrow {
namespace {

void throwInvalidLogicalType(const LogicalType& logicalType) {
  std::stringstream ss;
  ss << "Invalid logical type: " << logicalType.toString();
  throw ParquetException(ss.str());
}

void checkColumnBounds(int columnIndex, size_t maxColumns) {
  if (ARROW_PREDICT_FALSE(
          columnIndex < 0 || static_cast<size_t>(columnIndex) >= maxColumns)) {
    std::stringstream ss;
    ss << "Invalid Column Index: " << columnIndex
       << " Num columns: " << maxColumns;
    throw ParquetException(ss.str());
  }
}

} // namespace

namespace schema {

// ----------------------------------------------------------------------.
// ColumnPath.

std::shared_ptr<ColumnPath> ColumnPath::fromDotString(
    const std::string& dotstring) {
  std::stringstream ss(dotstring);
  std::string item;
  std::vector<std::string> path;
  while (std::getline(ss, item, '.')) {
    path.push_back(item);
  }
  return std::make_shared<ColumnPath>(std::move(path));
}

std::shared_ptr<ColumnPath> ColumnPath::fromNode(const Node& node) {
  // Build the path in reverse order as we traverse the nodes to the top.
  std::vector<std::string> rpath_;
  const Node* cursor = &node;
  // The schema node is not part of the ColumnPath.
  while (cursor->parent()) {
    rpath_.push_back(cursor->name());
    cursor = cursor->parent();
  }

  // Build ColumnPath in correct order.
  std::vector<std::string> path(rpath_.crbegin(), rpath_.crend());
  return std::make_shared<ColumnPath>(std::move(path));
}

std::shared_ptr<ColumnPath> ColumnPath::extend(
    const std::string& nodeName) const {
  std::vector<std::string> path;
  path.reserve(path_.size() + 1);
  path.resize(path_.size() + 1);
  std::copy(path_.cbegin(), path_.cend(), path.begin());
  path[path_.size()] = nodeName;

  return std::make_shared<ColumnPath>(std::move(path));
}

std::string ColumnPath::toDotString() const {
  std::stringstream ss;
  for (auto it = path_.cbegin(); it != path_.cend(); ++it) {
    if (it != path_.cbegin()) {
      ss << ".";
    }
    ss << *it;
  }
  return ss.str();
}

const std::vector<std::string>& ColumnPath::toDotVector() const {
  return path_;
}

// ----------------------------------------------------------------------.
// Base node.

const std::shared_ptr<ColumnPath> Node::path() const {
  // TODO(itaiin): Cache the result, or more precisely, cache ->ToDotString()
  //    Since it is being used to access the leaf nodes.
  return ColumnPath::fromNode(*this);
}

bool Node::equalsInternal(const Node* other) const {
  return type_ == other->type_ && name_ == other->name_ &&
      repetition_ == other->repetition_ &&
      convertedType_ == other->convertedType_ && fieldId_ == other->fieldId() &&
      logicalType_->equals(*(other->logicalType()));
}

void Node::setParent(const Node* parent) {
  parent_ = parent;
}

// ----------------------------------------------------------------------.
// Primitive node.

PrimitiveNode::PrimitiveNode(
    const std::string& name,
    Repetition::type repetition,
    Type::type type,
    ConvertedType::type convertedType,
    int length,
    int precision,
    int scale,
    int id)
    : Node(Node::PRIMITIVE, name, repetition, convertedType, id),
      physicalType_(type),
      typeLength_(length) {
  std::stringstream ss;

  // PARQUET-842: In an earlier revision, decimal_metadata_.isset was being.
  // Set to true, but Impala will raise an incompatible metadata in such cases.
  memset(&decimalMetadata_, 0, sizeof(decimalMetadata_));

  // Check if the physical and logical types match.
  // Mapping referred from Apache parquet-mr as on 2016-02-22.
  switch (convertedType) {
    case ConvertedType::kNone:
      // Logical type not set.
      break;
    case ConvertedType::kUtf8:
    case ConvertedType::kJson:
    case ConvertedType::kBson:
      if (type != Type::kByteArray) {
        ss << convertedTypeToString(convertedType);
        ss << " can only annotate BYTE_ARRAY fields";
        throw ParquetException(ss.str());
      }
      break;
    case ConvertedType::kDecimal:
      if ((type != Type::kInt32) && (type != Type::kInt64) &&
          (type != Type::kByteArray) && (type != Type::kFixedLenByteArray)) {
        ss << "DECIMAL can only annotate INT32, INT64, BYTE_ARRAY, and FIXED";
        throw ParquetException(ss.str());
      }
      if (precision <= 0) {
        ss << "Invalid DECIMAL precision: " << precision
           << ". Precision must be a number between 1 and 38 inclusive";
        throw ParquetException(ss.str());
      }
      if (scale < 0) {
        ss << "Invalid DECIMAL scale: " << scale
           << ". Scale must be a number between 0 and precision inclusive";
        throw ParquetException(ss.str());
      }
      if (scale > precision) {
        ss << "Invalid DECIMAL scale " << scale;
        ss << " cannot be greater than precision " << precision;
        throw ParquetException(ss.str());
      }
      decimalMetadata_.isset = true;
      decimalMetadata_.precision = precision;
      decimalMetadata_.scale = scale;
      break;
    case ConvertedType::kDate:
    case ConvertedType::kTimeMillis:
    case ConvertedType::kUint8:
    case ConvertedType::kUint16:
    case ConvertedType::kUint32:
    case ConvertedType::kInt8:
    case ConvertedType::kInt16:
    case ConvertedType::kInt32:
      if (type != Type::kInt32) {
        ss << convertedTypeToString(convertedType);
        ss << " can only annotate INT32";
        throw ParquetException(ss.str());
      }
      break;
    case ConvertedType::kTimeMicros:
    case ConvertedType::kTimestampMillis:
    case ConvertedType::kTimestampMicros:
    case ConvertedType::kUint64:
    case ConvertedType::kInt64:
      if (type != Type::kInt64) {
        ss << convertedTypeToString(convertedType);
        ss << " can only annotate INT64";
        throw ParquetException(ss.str());
      }
      break;
    case ConvertedType::kInterval:
      if ((type != Type::kFixedLenByteArray) || (length != 12)) {
        ss << "INTERVAL can only annotate FIXED_LEN_BYTE_ARRAY(12)";
        throw ParquetException(ss.str());
      }
      break;
    case ConvertedType::kEnum:
      if (type != Type::kByteArray) {
        ss << "ENUM can only annotate BYTE_ARRAY fields";
        throw ParquetException(ss.str());
      }
      break;
    case ConvertedType::kNa:
      // NA can annotate any type.
      break;
    default:
      ss << convertedTypeToString(convertedType);
      ss << " cannot be applied to a primitive type";
      throw ParquetException(ss.str());
  }
  // For forward compatibility, create an equivalent logical type.
  logicalType_ =
      LogicalType::fromConvertedType(convertedType_, decimalMetadata_);
  if (!(logicalType_ && !logicalType_->isNested() &&
        logicalType_->isCompatible(convertedType_, decimalMetadata_))) {
    throwInvalidLogicalType(*logicalType_);
  }

  if (type == Type::kFixedLenByteArray) {
    if (length <= 0) {
      ss << "Invalid FIXED_LEN_BYTE_ARRAY length: " << length;
      throw ParquetException(ss.str());
    }
    typeLength_ = length;
  }
}

PrimitiveNode::PrimitiveNode(
    const std::string& name,
    Repetition::type repetition,
    std::shared_ptr<const LogicalType> logicalType,
    Type::type physicalType,
    int physicalLength,
    int id)
    : Node(Node::PRIMITIVE, name, repetition, std::move(logicalType), id),
      physicalType_(physicalType),
      typeLength_(physicalLength) {
  std::stringstream error;
  if (logicalType_) {
    // Check for logical type <=> node type consistency.
    if (!logicalType_->isNested()) {
      // Check for logical type <=> physical type consistency.
      if (logicalType_->isApplicable(physicalType, physicalLength)) {
        // For backward compatibility, assign equivalent legacy.
        // converted type (if possible)
        convertedType_ = logicalType_->toConvertedType(&decimalMetadata_);
      } else {
        error << logicalType_->toString();
        error << " can not be applied to primitive type ";
        error << typeToString(physicalType);
        throw ParquetException(error.str());
      }
    } else {
      error << "Nested logical type ";
      error << logicalType_->toString();
      error << " can not be applied to non-group node";
      throw ParquetException(error.str());
    }
  } else {
    logicalType_ = NoLogicalType::make();
    convertedType_ = logicalType_->toConvertedType(&decimalMetadata_);
  }
  if (!(logicalType_ && !logicalType_->isNested() &&
        logicalType_->isCompatible(convertedType_, decimalMetadata_))) {
    throwInvalidLogicalType(*logicalType_);
  }

  if (physicalType == Type::kFixedLenByteArray) {
    if (physicalLength <= 0) {
      error << "Invalid FIXED_LEN_BYTE_ARRAY length: " << physicalLength;
      throw ParquetException(error.str());
    }
  }
}

bool PrimitiveNode::equalsInternal(const PrimitiveNode* other) const {
  bool isEqual = true;
  if (physicalType_ != other->physicalType_) {
    return false;
  }
  if (convertedType_ == ConvertedType::kDecimal) {
    isEqual &=
        (decimalMetadata_.precision == other->decimalMetadata_.precision) &&
        (decimalMetadata_.scale == other->decimalMetadata_.scale);
  }
  if (physicalType_ == Type::kFixedLenByteArray) {
    isEqual &= (typeLength_ == other->typeLength_);
  }
  return isEqual;
}

bool PrimitiveNode::equals(const Node* other) const {
  if (!Node::equalsInternal(other)) {
    return false;
  }
  return equalsInternal(static_cast<const PrimitiveNode*>(other));
}

void PrimitiveNode::visit(Node::Visitor* visitor) {
  visitor->visit(this);
}

void PrimitiveNode::visitConst(Node::ConstVisitor* visitor) const {
  visitor->visit(this);
}

// ----------------------------------------------------------------------.
// Group node.

GroupNode::GroupNode(
    const std::string& name,
    Repetition::type repetition,
    const NodeVector& fields,
    ConvertedType::type convertedType,
    int id)
    : Node(Node::GROUP, name, repetition, convertedType, id), fields_(fields) {
  // For forward compatibility, create an equivalent logical type.
  logicalType_ = LogicalType::fromConvertedType(convertedType_);
  if (!(logicalType_ && (logicalType_->isNested() || logicalType_->isNone()) &&
        logicalType_->isCompatible(convertedType_))) {
    throwInvalidLogicalType(*logicalType_);
  }

  fieldNameToIdx_.clear();
  auto fieldIdx = 0;
  for (NodePtr& field : fields_) {
    field->setParent(this);
    fieldNameToIdx_.emplace(field->name(), fieldIdx++);
  }
}

GroupNode::GroupNode(
    const std::string& name,
    Repetition::type repetition,
    const NodeVector& fields,
    std::shared_ptr<const LogicalType> logicalType,
    int id)
    : Node(Node::GROUP, name, repetition, std::move(logicalType), id),
      fields_(fields) {
  if (logicalType_) {
    // Check for logical type <=> node type consistency.
    if (logicalType_->isNested()) {
      // For backward compatibility, assign equivalent legacy converted type
      // (if. possible)
      convertedType_ = logicalType_->toConvertedType(nullptr);
    } else {
      std::stringstream error;
      error << "Logical type ";
      error << logicalType_->toString();
      error << " can not be applied to group node";
      throw ParquetException(error.str());
    }
  } else {
    logicalType_ = NoLogicalType::make();
    convertedType_ = logicalType_->toConvertedType(nullptr);
  }
  if (!(logicalType_ && (logicalType_->isNested() || logicalType_->isNone()) &&
        logicalType_->isCompatible(convertedType_))) {
    throwInvalidLogicalType(*logicalType_);
  }

  fieldNameToIdx_.clear();
  auto fieldIdx = 0;
  for (NodePtr& field : fields_) {
    field->setParent(this);
    fieldNameToIdx_.emplace(field->name(), fieldIdx++);
  }
}

bool GroupNode::equalsInternal(const GroupNode* other) const {
  if (this == other) {
    return true;
  }
  if (this->fieldCount() != other->fieldCount()) {
    return false;
  }
  for (int i = 0; i < this->fieldCount(); ++i) {
    if (!this->field(i)->equals(other->field(i).get())) {
      return false;
    }
  }
  return true;
}

bool GroupNode::equals(const Node* other) const {
  if (!Node::equalsInternal(other)) {
    return false;
  }
  return equalsInternal(static_cast<const GroupNode*>(other));
}

int GroupNode::fieldIndex(const std::string& name) const {
  auto search = fieldNameToIdx_.find(name);
  if (search == fieldNameToIdx_.end()) {
    // Not found.
    return -1;
  }
  return search->second;
}

int GroupNode::fieldIndex(const Node& node) const {
  auto search = fieldNameToIdx_.equal_range(node.name());
  for (auto it = search.first; it != search.second; ++it) {
    const int idx = it->second;
    if (&node == field(idx).get()) {
      return idx;
    }
  }
  return -1;
}

void GroupNode::visit(Node::Visitor* visitor) {
  visitor->visit(this);
}

void GroupNode::visitConst(Node::ConstVisitor* visitor) const {
  visitor->visit(this);
}

// ----------------------------------------------------------------------.
// Node construction from Parquet metadata.

std::unique_ptr<Node> GroupNode::fromParquet(
    const void* opaqueElement,
    NodeVector fields) {
  const facebook::velox::parquet::thrift::SchemaElement* element =
      static_cast<const facebook::velox::parquet::thrift::SchemaElement*>(
          opaqueElement);

  int fieldId = -1;
  if (element->__isset.field_id) {
    fieldId = element->field_id;
  }

  std::unique_ptr<GroupNode> groupNode;
  if (element->__isset.logicalType) {
    // Updated writer with logical type present.
    groupNode = std::unique_ptr<GroupNode>(new GroupNode(
        element->name,
        loadenumSafe(&element->repetition_type),
        fields,
        LogicalType::fromThrift(element->logicalType),
        fieldId));
  } else {
    groupNode = std::unique_ptr<GroupNode>(new GroupNode(
        element->name,
        loadenumSafe(&element->repetition_type),
        fields,
        (element->__isset.converted_type
             ? loadenumSafe(&element->converted_type)
             : ConvertedType::kNone),
        fieldId));
  }

  return std::unique_ptr<Node>(groupNode.release());
}

std::unique_ptr<Node> PrimitiveNode::fromParquet(const void* opaqueElement) {
  const facebook::velox::parquet::thrift::SchemaElement* element =
      static_cast<const facebook::velox::parquet::thrift::SchemaElement*>(
          opaqueElement);

  int fieldId = -1;
  if (element->__isset.field_id) {
    fieldId = element->field_id;
  }

  std::unique_ptr<PrimitiveNode> primitiveNode;
  if (element->__isset.logicalType) {
    // Updated writer with logical type present.
    primitiveNode = std::unique_ptr<PrimitiveNode>(new PrimitiveNode(
        element->name,
        loadenumSafe(&element->repetition_type),
        LogicalType::fromThrift(element->logicalType),
        loadenumSafe(&element->type),
        element->type_length,
        fieldId));
  } else if (element->__isset.converted_type) {
    // Legacy writer with converted type present.
    primitiveNode = std::unique_ptr<PrimitiveNode>(new PrimitiveNode(
        element->name,
        loadenumSafe(&element->repetition_type),
        loadenumSafe(&element->type),
        loadenumSafe(&element->converted_type),
        element->type_length,
        element->precision,
        element->scale,
        fieldId));
  } else {
    // Logical type not present.
    primitiveNode = std::unique_ptr<PrimitiveNode>(new PrimitiveNode(
        element->name,
        loadenumSafe(&element->repetition_type),
        NoLogicalType::make(),
        loadenumSafe(&element->type),
        element->type_length,
        fieldId));
  }

  // Return as unique_ptr to the base type.
  return std::unique_ptr<Node>(primitiveNode.release());
}

bool GroupNode::hasRepeatedFields() const {
  for (int i = 0; i < this->fieldCount(); ++i) {
    auto field = this->field(i);
    if (field->repetition() == Repetition::kRepeated) {
      return true;
    }
    if (field->isGroup()) {
      const auto& group = static_cast<const GroupNode&>(*field);
      return group.hasRepeatedFields();
    }
  }
  return false;
}

void GroupNode::toParquet(void* opaqueElement) const {
  facebook::velox::parquet::thrift::SchemaElement* element =
      static_cast<facebook::velox::parquet::thrift::SchemaElement*>(
          opaqueElement);
  element->__set_name(name_);
  element->__set_num_children(fieldCount());
  element->__set_repetition_type(toThrift(repetition_));
  if (convertedType_ != ConvertedType::kNone) {
    element->__set_converted_type(toThrift(convertedType_));
  }
  if (fieldId_ >= 0) {
    element->__set_field_id(fieldId_);
  }
  if (logicalType_ && logicalType_->isSerialized()) {
    element->__set_logicalType(logicalType_->toThrift());
  }
  return;
}

void PrimitiveNode::toParquet(void* opaqueElement) const {
  facebook::velox::parquet::thrift::SchemaElement* element =
      static_cast<facebook::velox::parquet::thrift::SchemaElement*>(
          opaqueElement);
  element->__set_name(name_);
  element->__set_repetition_type(toThrift(repetition_));
  if (convertedType_ != ConvertedType::kNone) {
    if (convertedType_ != ConvertedType::kNa) {
      element->__set_converted_type(toThrift(convertedType_));
    } else {
      // ConvertedType::kNa is an unreleased, obsolete synonym for.
      // LogicalType::nullType. Never emit it (see PARQUET-1990 for discussion).
      if (!logicalType_ || !logicalType_->isNull()) {
        throw ParquetException(
            "ConvertedType::kNa is obsolete, please use LogicalType::nullType instead");
      }
    }
  }
  if (fieldId_ >= 0) {
    element->__set_field_id(fieldId_);
  }
  if (logicalType_ && logicalType_->isSerialized() &&
      // TODO(tpboudreau): remove the following conjunct to enable
      // serialization. Of IntervalTypes after parquet.thrift recognizes them.
      !logicalType_->isInterval()) {
    element->__set_logicalType(logicalType_->toThrift());
  }
  element->__set_type(toThrift(physicalType_));
  if (physicalType_ == Type::kFixedLenByteArray) {
    element->__set_type_length(typeLength_);
  }
  if (decimalMetadata_.isset) {
    element->__set_precision(decimalMetadata_.precision);
    element->__set_scale(decimalMetadata_.scale);
  }
  return;
}

// ----------------------------------------------------------------------.
// Schema converters.

std::unique_ptr<Node> unflatten(
    const facebook::velox::parquet::thrift::SchemaElement* elements,
    int length) {
  if (elements[0].num_children == 0) {
    if (length == 1) {
      // Degenerate case of Parquet file with no columns.
      return GroupNode::fromParquet(elements, {});
    } else {
      throw ParquetException(
          "Parquet schema had multiple nodes but root had no children");
    }
  }

  // We don't check that the root node is repeated since this is not.
  // Consistently set by implementations.

  int pos = 0;

  std::function<std::unique_ptr<Node>()> nextNode = [&]() {
    if (pos == length) {
      throw ParquetException("Malformed schema: not enough elements");
    }
    const SchemaElement& element = elements[pos++];
    const void* opaqueElement = static_cast<const void*>(&element);

    if (element.num_children == 0 && element.__isset.type) {
      // Leaf (primitive) node: always has a type.
      return PrimitiveNode::fromParquet(opaqueElement);
    } else {
      // Group node (may have 0 children, but cannot have a type)
      NodeVector fields;
      for (int i = 0; i < element.num_children; ++i) {
        std::unique_ptr<Node> field = nextNode();
        fields.push_back(NodePtr(field.release()));
      }
      return GroupNode::fromParquet(opaqueElement, std::move(fields));
    }
  };
  return nextNode();
}

std::shared_ptr<SchemaDescriptor> fromParquet(
    const std::vector<SchemaElement>& schema) {
  if (schema.empty()) {
    throw ParquetException("Empty file schema (no root)");
  }
  std::unique_ptr<Node> root =
      unflatten(&schema[0], static_cast<int>(schema.size()));
  std::shared_ptr<SchemaDescriptor> descr =
      std::make_shared<SchemaDescriptor>();
  descr->init(
      std::shared_ptr<GroupNode>(static_cast<GroupNode*>(root.release())));
  return descr;
}

class SchemaVisitor : public Node::ConstVisitor {
 public:
  explicit SchemaVisitor(
      std::vector<facebook::velox::parquet::thrift::SchemaElement>* elements)
      : elements_(elements) {}

  void visit(const Node* node) override {
    facebook::velox::parquet::thrift::SchemaElement element;
    node->toParquet(&element);
    elements_->push_back(element);

    if (node->isGroup()) {
      const GroupNode* groupNode = static_cast<const GroupNode*>(node);
      for (int i = 0; i < groupNode->fieldCount(); ++i) {
        groupNode->field(i)->visitConst(this);
      }
    }
  }

 private:
  std::vector<facebook::velox::parquet::thrift::SchemaElement>* elements_;
};

void toParquet(
    const GroupNode* schema,
    std::vector<facebook::velox::parquet::thrift::SchemaElement>* out) {
  SchemaVisitor visitor(out);
  schema->visitConst(&visitor);
}

// ----------------------------------------------------------------------.
// Schema printing.

static void printRepLevel(Repetition::type repetition, std::ostream& stream) {
  switch (repetition) {
    case Repetition::kRequired:
      stream << "required";
      break;
    case Repetition::kOptional:
      stream << "optional";
      break;
    case Repetition::kRepeated:
      stream << "repeated";
      break;
    default:
      break;
  }
}

static void printType(const PrimitiveNode* Node, std::ostream& stream) {
  switch (Node->physicalType()) {
    case Type::kBoolean:
      stream << "boolean";
      break;
    case Type::kInt32:
      stream << "int32";
      break;
    case Type::kInt64:
      stream << "int64";
      break;
    case Type::kInt96:
      stream << "int96";
      break;
    case Type::kFloat:
      stream << "float";
      break;
    case Type::kDouble:
      stream << "double";
      break;
    case Type::kByteArray:
      stream << "binary";
      break;
    case Type::kFixedLenByteArray:
      stream << "fixed_len_byte_array(" << Node->typeLength() << ")";
      break;
    default:
      break;
  }
}

static void printConvertedType(
    const PrimitiveNode* Node,
    std::ostream& stream) {
  auto lt = Node->convertedType();
  auto la = Node->logicalType();
  if (la && la->isValid() && !la->isNone()) {
    stream << " (" << la->toString() << ")";
  } else if (lt == ConvertedType::kDecimal) {
    stream << " (" << convertedTypeToString(lt) << "("
           << Node->decimalMetadata().precision << ","
           << Node->decimalMetadata().scale << "))";
  } else if (lt != ConvertedType::kNone) {
    stream << " (" << convertedTypeToString(lt) << ")";
  }
}

struct SchemaPrinter : public Node::ConstVisitor {
  explicit SchemaPrinter(std::ostream& stream, int indentWidth)
      : stream_(stream), indent_(0), indentWidth_(2) {}

  void indent() {
    if (indent_ > 0) {
      std::string spaces(indent_, ' ');
      stream_ << spaces;
    }
  }

  void visit(const Node* Node) {
    indent();
    if (Node->isGroup()) {
      visit(static_cast<const GroupNode*>(Node));
    } else {
      // Primitive.
      visit(static_cast<const PrimitiveNode*>(Node));
    }
  }

  void visit(const PrimitiveNode* Node) {
    printRepLevel(Node->repetition(), stream_);
    stream_ << " ";
    printType(Node, stream_);
    stream_ << " field_id=" << Node->fieldId() << " " << Node->name();
    printConvertedType(Node, stream_);
    stream_ << ";" << std::endl;
  }

  void visit(const GroupNode* Node) {
    printRepLevel(Node->repetition(), stream_);
    stream_ << " group " << "field_id=" << Node->fieldId() << " "
            << Node->name();
    auto lt = Node->convertedType();
    auto la = Node->logicalType();
    if (la && la->isValid() && !la->isNone()) {
      stream_ << " (" << la->toString() << ")";
    } else if (lt != ConvertedType::kNone) {
      stream_ << " (" << convertedTypeToString(lt) << ")";
    }
    stream_ << " {" << std::endl;

    indent_ += indentWidth_;
    for (int i = 0; i < Node->fieldCount(); ++i) {
      Node->field(i)->visitConst(this);
    }
    indent_ -= indentWidth_;
    indent();
    stream_ << "}" << std::endl;
  }

  std::ostream& stream_;
  int indent_;
  int indentWidth_;
};

void printSchema(const Node* schema, std::ostream& stream, int indentWidth) {
  SchemaPrinter printer(stream, indentWidth);
  printer.visit(schema);
}

} // namespace schema

using schema::ColumnPath;
using schema::GroupNode;
using schema::Node;
using schema::NodePtr;
using schema::PrimitiveNode;

void SchemaDescriptor::init(std::unique_ptr<schema::Node> schema) {
  init(NodePtr(schema.release()));
}

class SchemaUpdater : public Node::Visitor {
 public:
  explicit SchemaUpdater(const std::vector<ColumnOrder>& columnOrders)
      : columnOrders_(columnOrders), leafCount_(0) {}

  void visit(Node* node) override {
    if (node->isGroup()) {
      GroupNode* groupNode = static_cast<GroupNode*>(node);
      for (int i = 0; i < groupNode->fieldCount(); ++i) {
        groupNode->field(i)->visit(this);
      }
    } else { // leaf node
      PrimitiveNode* leafNode = static_cast<PrimitiveNode*>(node);
      leafNode->setColumnOrder(columnOrders_[leafCount_++]);
    }
  }

 private:
  const std::vector<ColumnOrder>& columnOrders_;
  int leafCount_;
};

void SchemaDescriptor::updateColumnOrders(
    const std::vector<ColumnOrder>& columnOrders) {
  if (static_cast<int>(columnOrders.size()) != numColumns()) {
    throw ParquetException("Malformed schema: not enough ColumnOrder values");
  }
  SchemaUpdater visitor(columnOrders);
  const_cast<GroupNode*>(groupNode_)->visit(&visitor);
}

void SchemaDescriptor::init(NodePtr schema) {
  schema_ = std::move(schema);

  if (!schema_->isGroup()) {
    throw ParquetException("Must initialize with a schema group");
  }

  groupNode_ = static_cast<const GroupNode*>(schema_.get());
  leaves_.clear();

  for (int i = 0; i < groupNode_->fieldCount(); ++i) {
    buildTree(groupNode_->field(i), 0, 0, groupNode_->field(i));
  }
}

bool SchemaDescriptor::equals(
    const SchemaDescriptor& other,
    std::ostream* diffOutput) const {
  if (this->numColumns() != other.numColumns()) {
    if (diffOutput != nullptr) {
      *diffOutput << "This schema has " << this->numColumns()
                  << " columns, other has " << other.numColumns();
    }
    return false;
  }

  for (int i = 0; i < this->numColumns(); ++i) {
    if (!this->column(i)->equals(*other.column(i))) {
      if (diffOutput != nullptr) {
        *diffOutput << "The two columns with index " << i << " differ."
                    << std::endl
                    << this->column(i)->toString() << std::endl
                    << other.column(i)->toString() << std::endl;
      }
      return false;
    }
  }

  return true;
}

void SchemaDescriptor::buildTree(
    const NodePtr& Node,
    int16_t maxDefLevel,
    int16_t maxRepLevel,
    const NodePtr& base) {
  if (Node->isOptional()) {
    ++maxDefLevel;
  } else if (Node->isRepeated()) {
    // Repeated fields add a definition level. This is used to distinguish.
    // Between an empty list and a list with an item in it.
    ++maxRepLevel;
    ++maxDefLevel;
  }

  // Now, walk the schema and create a ColumnDescriptor for each leaf node.
  if (Node->isGroup()) {
    const GroupNode* group = static_cast<const GroupNode*>(Node.get());
    for (int i = 0; i < group->fieldCount(); ++i) {
      buildTree(group->field(i), maxDefLevel, maxRepLevel, base);
    }
  } else {
    nodeToLeafIndex_[static_cast<const PrimitiveNode*>(Node.get())] =
        static_cast<int>(leaves_.size());

    // Primitive node, append to leaves.
    leaves_.push_back(ColumnDescriptor(Node, maxDefLevel, maxRepLevel, this));
    leafToBase_.emplace(static_cast<int>(leaves_.size()) - 1, base);
    leafToIdx_.emplace(
        Node->path()->toDotString(), static_cast<int>(leaves_.size()) - 1);
  }
}

int SchemaDescriptor::getColumnIndex(const PrimitiveNode& Node) const {
  auto it = nodeToLeafIndex_.find(&Node);
  if (it == nodeToLeafIndex_.end()) {
    return -1;
  }
  return it->second;
}

ColumnDescriptor::ColumnDescriptor(
    schema::NodePtr Node,
    int16_t maxDefinitionLevel,
    int16_t maxRepetitionLevel,
    const SchemaDescriptor* schemaDescr)
    : node_(std::move(Node)),
      maxDefinitionLevel_(maxDefinitionLevel),
      maxRepetitionLevel_(maxRepetitionLevel) {
  if (!node_->isPrimitive()) {
    throw ParquetException("Must be a primitive type");
  }
  primitiveNode_ = static_cast<const PrimitiveNode*>(node_.get());
}

bool ColumnDescriptor::equals(const ColumnDescriptor& other) const {
  return primitiveNode_->equals(other.primitiveNode_) &&
      maxRepetitionLevel() == other.maxRepetitionLevel() &&
      maxDefinitionLevel() == other.maxDefinitionLevel();
}

const ColumnDescriptor* SchemaDescriptor::column(int i) const {
  checkColumnBounds(i, leaves_.size());
  return &leaves_[i];
}

int SchemaDescriptor::columnIndex(const std::string& nodePath) const {
  auto search = leafToIdx_.find(nodePath);
  if (search == leafToIdx_.end()) {
    // Not found.
    return -1;
  }
  return search->second;
}

int SchemaDescriptor::columnIndex(const Node& node) const {
  auto search = leafToIdx_.equal_range(node.path()->toDotString());
  for (auto it = search.first; it != search.second; ++it) {
    const int idx = it->second;
    if (&node == column(idx)->schemaNode().get()) {
      return idx;
    }
  }
  return -1;
}

const schema::Node* SchemaDescriptor::getColumnRoot(int i) const {
  checkColumnBounds(i, leaves_.size());
  return leafToBase_.find(i)->second.get();
}

bool SchemaDescriptor::hasRepeatedFields() const {
  return groupNode_->hasRepeatedFields();
}

std::string SchemaDescriptor::toString() const {
  std::ostringstream ss;
  printSchema(schema_.get(), ss);
  return ss.str();
}

std::string ColumnDescriptor::toString() const {
  std::ostringstream ss;
  ss << "column descriptor = {" << std::endl
     << "  name: " << name() << "," << std::endl
     << "  path: " << path()->toDotString() << "," << std::endl
     << "  physical_type: " << typeToString(physicalType()) << "," << std::endl
     << "  converted_type: " << convertedTypeToString(convertedType()) << ","
     << std::endl
     << "  logical_type: " << logicalType()->toString() << "," << std::endl
     << "  max_definition_level: " << maxDefinitionLevel() << "," << std::endl
     << "  max_repetition_level: " << maxRepetitionLevel() << "," << std::endl;

  if (physicalType() ==
      ::facebook::velox::parquet::arrow::Type::kFixedLenByteArray) {
    ss << "  length: " << typeLength() << "," << std::endl;
  }

  if (convertedType() ==
      ::facebook::velox::parquet::arrow::ConvertedType::kDecimal) {
    ss << "  precision: " << typePrecision() << "," << std::endl
       << "  scale: " << typeScale() << "," << std::endl;
  }

  ss << "}";
  return ss.str();
}

int ColumnDescriptor::typeScale() const {
  return primitiveNode_->decimalMetadata().scale;
}

int ColumnDescriptor::typePrecision() const {
  return primitiveNode_->decimalMetadata().precision;
}

int ColumnDescriptor::typeLength() const {
  return primitiveNode_->typeLength();
}

const std::shared_ptr<ColumnPath> ColumnDescriptor::path() const {
  return primitiveNode_->path();
}

} // namespace facebook::velox::parquet::arrow
