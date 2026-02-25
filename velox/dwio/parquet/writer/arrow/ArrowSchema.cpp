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

#include "velox/dwio/parquet/writer/arrow/ArrowSchema.h"

#include <charconv>
#include <functional>
#include <string>
#include <vector>

#include "arrow/extension_type.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/api.h"
#include "arrow/type.h"
#include "arrow/util/base64.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/string.h"
#include "arrow/util/value_parsing.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/ArrowSchemaInternal.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

using arrow::DecimalType;
using arrow::Field;
using arrow::FieldVector;
using arrow::KeyValueMetadata;
using arrow::Status;
using arrow::internal::checked_cast;

using ArrowType = arrow::DataType;
using ArrowTypeId = arrow::Type;

namespace facebook::velox::parquet::arrow::arrow {

using schema::GroupNode;
using schema::Node;
using schema::NodePtr;
using schema::PrimitiveNode;

using ParquetType = Type;

// ----------------------------------------------------------------------.
// Parquet to Arrow schema conversion.

namespace {

/// Increments levels according to the cardinality of node.
void incrementLevels(LevelInfo& currentLevels, const schema::Node& node) {
  if (node.isRepeated()) {
    currentLevels.IncrementRepeated();
    return;
  }
  if (node.isOptional()) {
    currentLevels.IncrementOptional();
    return;
  }
}

/// Like std::string_view::ends_with in C++20.
inline bool endsWith(std::string_view s, std::string_view suffix) {
  return s.length() >= suffix.length() &&
      (s.empty() || s.substr(s.length() - suffix.length()) == suffix);
}

namespace detail {
template <typename T, typename = void>
struct CanToChars : public std::false_type {};

template <typename T>
struct CanToChars<
    T,
    std::void_t<decltype(std::to_chars(
        std::declval<char*>(),
        std::declval<char*>(),
        std::declval<std::remove_reference_t<T>>()))>> : public std::true_type {
};
} // namespace detail

/// \brief Whether std::to_chars exists for the current value type.
///
/// This is useful as some C++ libraries do not implement all specified
/// overloads for std::to_chars.
template <typename T>
inline constexpr bool haveToChars = detail::CanToChars<T>::value;

/// \brief An ergonomic wrapper around std::to_chars, returning a std::string.
///
/// For most inputs, the std::string result will not incur any heap allocation
/// thanks to small string optimization.
///
/// Compared to std::to_string, this function gives locale-agnostic results
/// and might also be faster.
template <typename T, typename... Args>
std::string toChars(T value, Args&&... args) {
  if constexpr (!haveToChars<T>) {
    // Some C++ standard libraries do not yet implement std::to_chars for all.
    // Types, in which case we have to fallback to std::string.
    return std::to_string(value);
  } else {
    // According to various sources, the GNU libstdc++ and Microsoft's C++ STL
    // allow up to 15 bytes of small string optimization, while clang's libc++
    // goes up to 22 bytes. Choose the pessimistic value.
    std::string out(15, 0);
    auto res = std::to_chars(&out.front(), &out.back(), value, args...);
    while (res.ec != std::errc{}) {
      assert(res.ec == std::errc::value_too_large);
      out.resize(out.capacity() * 2);
      res = std::to_chars(&out.front(), &out.back(), value, args...);
    }
    const auto length = res.ptr - out.data();
    assert(length <= static_cast<int64_t>(out.length()));
    out.resize(length);
    return out;
  }
}

Repetition::type repetitionFromNullable(bool isNullable) {
  return isNullable ? Repetition::kOptional : Repetition::kRequired;
}

Status fieldToNode(
    const std::string& name,
    const std::shared_ptr<Field>& field,
    const WriterProperties& properties,
    const ArrowWriterProperties& arrowProperties,
    NodePtr* out);

Status listToNode(
    const std::shared_ptr<::arrow::BaseListType>& type,
    const std::string& name,
    bool nullable,
    int fieldId,
    const WriterProperties& properties,
    const ArrowWriterProperties& arrowProperties,
    NodePtr* out) {
  NodePtr element;
  std::string valueName = arrowProperties.compliantNestedTypes()
      ? "element"
      : type->value_field()->name();
  RETURN_NOT_OK(fieldToNode(
      valueName, type->value_field(), properties, arrowProperties, &element));

  NodePtr List = GroupNode::make("list", Repetition::kRepeated, {element});
  *out = GroupNode::make(
      name,
      repetitionFromNullable(nullable),
      {List},
      LogicalType::list(),
      fieldId);
  return Status::OK();
}

Status mapToNode(
    const std::shared_ptr<::arrow::MapType>& type,
    const std::string& name,
    bool nullable,
    int fieldId,
    const WriterProperties& properties,
    const ArrowWriterProperties& arrowProperties,
    NodePtr* out) {
  // TODO: Should we offer a non-compliant mode that forwards the type names?
  NodePtr keyNode;
  RETURN_NOT_OK(fieldToNode(
      "key", type->key_field(), properties, arrowProperties, &keyNode));

  NodePtr valueNode;
  RETURN_NOT_OK(fieldToNode(
      "value", type->item_field(), properties, arrowProperties, &valueNode));

  NodePtr keyValue =
      GroupNode::make("key_value", Repetition::kRepeated, {keyNode, valueNode});
  *out = GroupNode::make(
      name,
      repetitionFromNullable(nullable),
      {keyValue},
      LogicalType::map(),
      fieldId);
  return Status::OK();
}

Status structToNode(
    const std::shared_ptr<::arrow::StructType>& type,
    const std::string& name,
    bool nullable,
    int fieldId,
    const WriterProperties& properties,
    const ArrowWriterProperties& arrowProperties,
    NodePtr* out) {
  std::vector<NodePtr> children(type->num_fields());
  if (type->num_fields() != 0) {
    for (int i = 0; i < type->num_fields(); i++) {
      RETURN_NOT_OK(fieldToNode(
          type->field(i)->name(),
          type->field(i),
          properties,
          arrowProperties,
          &children[i]));
    }
  } else {
    // XXX (ARROW-10928) We could add a dummy primitive node but that would
    // require special handling when writing and reading, to avoid column index
    // mismatches.
    return Status::NotImplemented(
        "Cannot write struct type '",
        name,
        "' with no child field to Parquet. "
        "Consider adding a dummy child field.");
  }

  *out = GroupNode::make(
      name, repetitionFromNullable(nullable), children, nullptr, fieldId);
  return Status::OK();
}

static std::shared_ptr<const LogicalType>
timestampLogicalTypeFromArrowTimestamp(
    const ::arrow::TimestampType& timestampType,
    ::arrow::TimeUnit::type timeUnit) {
  const bool utc = !(timestampType.timezone().empty());
  // ARROW-5878(wesm): for forward compatibility reasons, and because
  // there's no other way to signal to old readers that values are
  // timestamps, we force the ConvertedType field to be set to the
  // corresponding TIMESTAMP_* value. This does cause some ambiguity
  // as Parquet readers have not been consistent about the
  // interpretation of TIMESTAMP_* values as being UTC-normalized.
  switch (timeUnit) {
    case ::arrow::TimeUnit::MILLI:
      return LogicalType::timestamp(
          utc, LogicalType::TimeUnit::kMillis, false, true);
    case ::arrow::TimeUnit::MICRO:
      return LogicalType::timestamp(
          utc, LogicalType::TimeUnit::kMicros, false, true);
    case ::arrow::TimeUnit::NANO:
      return LogicalType::timestamp(utc, LogicalType::TimeUnit::kNanos);
    case ::arrow::TimeUnit::SECOND:
      // No equivalent parquet logical type.
      break;
  }
  return LogicalType::none();
}

static Status getTimestampMetadata(
    const ::arrow::TimestampType& type,
    const WriterProperties& properties,
    const ArrowWriterProperties& arrowProperties,
    ParquetType::type* physicalType,
    std::shared_ptr<const LogicalType>* logicalType) {
  const bool coerce = arrowProperties.coerceTimestampsEnabled();
  const auto targetUnit =
      coerce ? arrowProperties.coerceTimestampsUnit() : type.unit();
  const auto version = properties.version();

  // The user is explicitly asking for Impala int96 encoding, there is no
  // logical type.
  if (arrowProperties.supportDeprecatedInt96Timestamps()) {
    *physicalType = ParquetType::kInt96;
    return Status::OK();
  }

  *physicalType = ParquetType::kInt64;
  *logicalType = timestampLogicalTypeFromArrowTimestamp(type, targetUnit);

  // The user is explicitly asking for timestamp data to be converted to the
  // specified units (target_unit).
  if (coerce) {
    if (version == ParquetVersion::PARQUET_1_0 ||
        version == ParquetVersion::PARQUET_2_4) {
      switch (targetUnit) {
        case ::arrow::TimeUnit::MILLI:
        case ::arrow::TimeUnit::MICRO:
          break;
        case ::arrow::TimeUnit::NANO:
        case ::arrow::TimeUnit::SECOND:
          return Status::NotImplemented(
              "For Parquet version ",
              parquetVersionToString(version),
              ", can only coerce Arrow timestamps to "
              "milliseconds or microseconds");
      }
    } else {
      switch (targetUnit) {
        case ::arrow::TimeUnit::MILLI:
        case ::arrow::TimeUnit::MICRO:
        case ::arrow::TimeUnit::NANO:
          break;
        case ::arrow::TimeUnit::SECOND:
          return Status::NotImplemented(
              "For Parquet version ",
              parquetVersionToString(version),
              ", can only coerce Arrow timestamps to "
              "milliseconds, microseconds, or nanoseconds");
      }
    }
    return Status::OK();
  }

  // The user implicitly wants timestamp data to retain its original time
  // units. However, the ConvertedType field used to indicate logical types for
  // Parquet version <= 2.4 fields does not allow for nanosecond time units and
  // so nanoseconds must be coerced to microseconds.
  if ((version == ParquetVersion::PARQUET_1_0 ||
       version == ParquetVersion::PARQUET_2_4) &&
      type.unit() == ::arrow::TimeUnit::NANO) {
    *logicalType =
        timestampLogicalTypeFromArrowTimestamp(type, ::arrow::TimeUnit::MICRO);
    return Status::OK();
  }

  // The user implicitly wants timestamp data to retain its original time
  // units. However, the Arrow seconds time unit can not be represented
  // (annotated) in any version of Parquet and so must be coerced to
  // milliseconds.
  if (type.unit() == ::arrow::TimeUnit::SECOND) {
    *logicalType =
        timestampLogicalTypeFromArrowTimestamp(type, ::arrow::TimeUnit::MILLI);
    return Status::OK();
  }

  return Status::OK();
}

static constexpr char FIELD_ID_KEY[] = "PARQUET:field_id";

int fieldIdFromMetadata(
    const std::shared_ptr<const ::arrow::KeyValueMetadata>& metadata) {
  if (!metadata) {
    return -1;
  }
  int key = metadata->FindKey(FIELD_ID_KEY);
  if (key < 0) {
    return -1;
  }
  std::string fieldIdStr = metadata->value(key);
  int fieldId;
  if (::arrow::internal::ParseValue<::arrow::Int32Type>(
          fieldIdStr.c_str(), fieldIdStr.length(), &fieldId)) {
    if (fieldId < 0) {
      // Thrift should convert any negative value to null but normalize to -1.
      // Here in case we later check this in logic.
      return -1;
    }
    return fieldId;
  } else {
    return -1;
  }
}

Status fieldToNode(
    const std::string& name,
    const std::shared_ptr<Field>& field,
    const WriterProperties& properties,
    const ArrowWriterProperties& arrowProperties,
    NodePtr* out) {
  std::shared_ptr<const LogicalType> logicalType = LogicalType::none();
  ParquetType::type type;
  Repetition::type repetition = repetitionFromNullable(field->nullable());
  int fieldId = fieldIdFromMetadata(field->metadata());

  int length = -1;
  int precision = -1;
  int scale = -1;

  switch (field->type()->id()) {
    case ArrowTypeId::NA: {
      type = ParquetType::kInt32;
      logicalType = LogicalType::nullType();
      if (repetition != Repetition::kOptional) {
        return Status::Invalid("NullType Arrow field must be nullable");
      }
    } break;
    case ArrowTypeId::BOOL:
      type = ParquetType::kBoolean;
      break;
    case ArrowTypeId::UINT8:
      type = ParquetType::kInt32;
      logicalType = LogicalType::intType(8, false);
      break;
    case ArrowTypeId::INT8:
      type = ParquetType::kInt32;
      logicalType = LogicalType::intType(8, true);
      break;
    case ArrowTypeId::UINT16:
      type = ParquetType::kInt32;
      logicalType = LogicalType::intType(16, false);
      break;
    case ArrowTypeId::INT16:
      type = ParquetType::kInt32;
      logicalType = LogicalType::intType(16, true);
      break;
    case ArrowTypeId::UINT32:
      if (properties.version() == ParquetVersion::PARQUET_1_0) {
        type = ParquetType::kInt64;
      } else {
        type = ParquetType::kInt32;
        logicalType = LogicalType::intType(32, false);
      }
      break;
    case ArrowTypeId::INT32:
      type = ParquetType::kInt32;
      break;
    case ArrowTypeId::UINT64:
      type = ParquetType::kInt64;
      logicalType = LogicalType::intType(64, false);
      break;
    case ArrowTypeId::INT64:
      type = ParquetType::kInt64;
      break;
    case ArrowTypeId::FLOAT:
      type = ParquetType::kFloat;
      break;
    case ArrowTypeId::DOUBLE:
      type = ParquetType::kDouble;
      break;
    case ArrowTypeId::LARGE_STRING:
    case ArrowTypeId::STRING:
      type = ParquetType::kByteArray;
      logicalType = LogicalType::string();
      break;
    case ArrowTypeId::LARGE_BINARY:
    case ArrowTypeId::BINARY:
      type = ParquetType::kByteArray;
      break;
    case ArrowTypeId::FIXED_SIZE_BINARY: {
      type = ParquetType::kFixedLenByteArray;
      const auto& fixedSizeBinaryType =
          static_cast<const ::arrow::FixedSizeBinaryType&>(*field->type());
      length = fixedSizeBinaryType.byte_width();
    } break;
    case ArrowTypeId::DECIMAL128:
    case ArrowTypeId::DECIMAL256: {
      const auto& decimalType =
          static_cast<const ::arrow::DecimalType&>(*field->type());
      precision = decimalType.precision();
      scale = decimalType.scale();
      if (properties.storeDecimalAsInteger() && 1 <= precision &&
          precision <= 18) {
        type = precision <= 9 ? ParquetType::kInt32 : ParquetType::kInt64;
      } else {
        type = ParquetType::kFixedLenByteArray;
        length = DecimalType::DecimalSize(precision);
      }
      PARQUET_CATCH_NOT_OK(
          logicalType = LogicalType::decimal(precision, scale));
    } break;
    case ArrowTypeId::DATE32:
      type = ParquetType::kInt32;
      logicalType = LogicalType::date();
      break;
    case ArrowTypeId::DATE64:
      type = ParquetType::kInt32;
      logicalType = LogicalType::date();
      break;
    case ArrowTypeId::TIMESTAMP:
      RETURN_NOT_OK(getTimestampMetadata(
          static_cast<::arrow::TimestampType&>(*field->type()),
          properties,
          arrowProperties,
          &type,
          &logicalType));
      break;
    case ArrowTypeId::TIME32:
      type = ParquetType::kInt32;
      logicalType = LogicalType::time(true, LogicalType::TimeUnit::kMillis);
      break;
    case ArrowTypeId::TIME64: {
      type = ParquetType::kInt64;
      auto timeType = static_cast<::arrow::Time64Type*>(field->type().get());
      if (timeType->unit() == ::arrow::TimeUnit::NANO) {
        logicalType = LogicalType::time(true, LogicalType::TimeUnit::kNanos);
      } else {
        logicalType = LogicalType::time(true, LogicalType::TimeUnit::kMicros);
      }
    } break;
    case ArrowTypeId::DURATION:
      type = ParquetType::kInt64;
      break;
    case ArrowTypeId::STRUCT: {
      auto structType =
          std::static_pointer_cast<::arrow::StructType>(field->type());
      return structToNode(
          structType,
          name,
          field->nullable(),
          fieldId,
          properties,
          arrowProperties,
          out);
    }
    case ArrowTypeId::FIXED_SIZE_LIST:
    case ArrowTypeId::LARGE_LIST:
    case ArrowTypeId::LIST: {
      auto listType =
          std::static_pointer_cast<::arrow::BaseListType>(field->type());
      return listToNode(
          listType,
          name,
          field->nullable(),
          fieldId,
          properties,
          arrowProperties,
          out);
    }
    case ArrowTypeId::DICTIONARY: {
      // Parquet has no Dictionary type, dictionary-encoded is handled on
      // the encoding, not the schema level.
      const ::arrow::DictionaryType& dictType =
          static_cast<const ::arrow::DictionaryType&>(*field->type());
      std::shared_ptr<::arrow::Field> unpackedField = ::arrow::field(
          name, dictType.value_type(), field->nullable(), field->metadata());
      return fieldToNode(name, unpackedField, properties, arrowProperties, out);
    }
    case ArrowTypeId::EXTENSION: {
      auto extType =
          std::static_pointer_cast<::arrow::ExtensionType>(field->type());
      std::shared_ptr<::arrow::Field> storageField = ::arrow::field(
          name, extType->storage_type(), field->nullable(), field->metadata());
      return fieldToNode(name, storageField, properties, arrowProperties, out);
    }
    case ArrowTypeId::MAP: {
      auto mapType = std::static_pointer_cast<::arrow::MapType>(field->type());
      return mapToNode(
          mapType,
          name,
          field->nullable(),
          fieldId,
          properties,
          arrowProperties,
          out);
    }

    default: {
      // TODO: DENSE_UNION, SPARE_UNION, JSON_SCALAR, DECIMAL_TEXT, VARCHAR.
      return Status::NotImplemented(
          "Unhandled type for Arrow to Parquet schema conversion: ",
          field->type()->ToString());
    }
  }

  PARQUET_CATCH_NOT_OK(*out = PrimitiveNode::make(name, repetition, logicalType, type, length, fieldId));

  return Status::OK();
}

struct SchemaTreeContext {
  SchemaManifest* manifest;
  ArrowReaderProperties properties;
  const SchemaDescriptor* schema;

  void linkParent(const SchemaField* child, const SchemaField* parent) {
    manifest->childToParent[child] = parent;
  }

  void recordLeaf(const SchemaField* leaf) {
    manifest->columnIndexToField[leaf->columnIndex] = leaf;
  }
};

bool isDictionaryReadSupported(const ArrowType& type) {
  // Only supported currently for BYTE_ARRAY types.
  return type.id() == ::arrow::Type::BINARY ||
      type.id() == ::arrow::Type::STRING;
}

// ----------------------------------------------------------------------.
// Schema logic.

::arrow::Result<std::shared_ptr<ArrowType>> getTypeForNode(
    int columnIndex,
    const schema::PrimitiveNode& primitiveNode,
    SchemaTreeContext* ctx) {
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<ArrowType> storageType,
      getArrowType(primitiveNode, ctx->properties.coerceInt96TimestampUnit()));
  if (ctx->properties.readDictionary(columnIndex) &&
      isDictionaryReadSupported(*storageType)) {
    return ::arrow::dictionary(::arrow::int32(), storageType);
  }
  return storageType;
}

Status nodeToSchemaField(
    const Node& node,
    LevelInfo currentLevels,
    SchemaTreeContext* ctx,
    const SchemaField* parent,
    SchemaField* out);

Status groupToSchemaField(
    const GroupNode& groupNode,
    LevelInfo currentLevels,
    SchemaTreeContext* ctx,
    const SchemaField* parent,
    SchemaField* out);

Status populateLeaf(
    int columnIndex,
    const std::shared_ptr<Field>& field,
    LevelInfo currentLevels,
    SchemaTreeContext* ctx,
    const SchemaField* parent,
    SchemaField* out) {
  out->field = field;
  out->columnIndex = columnIndex;
  out->levelInfo = currentLevels;
  ctx->recordLeaf(out);
  ctx->linkParent(out, parent);
  return Status::OK();
}

// Special case mentioned in the format spec:
//   If the name is array or ends in _tuple, this should be a list of struct,
//   even for single child elements.
bool hasStructListName(const GroupNode& groupNode) {
  ::std::string_view name{groupNode.name()};
  return name == "array" || endsWith(name, "_tuple");
}

Status groupToStruct(
    const GroupNode& groupNode,
    LevelInfo currentLevels,
    SchemaTreeContext* ctx,
    const SchemaField* parent,
    SchemaField* out) {
  std::vector<std::shared_ptr<Field>> arrowFields;
  out->children.resize(groupNode.fieldCount());
  // All level increments for the node are expected to happen by callers.
  // This is required because repeated elements need to have their own
  // SchemaField.

  for (int i = 0; i < groupNode.fieldCount(); i++) {
    RETURN_NOT_OK(nodeToSchemaField(
        *groupNode.field(i), currentLevels, ctx, out, &out->children[i]));
    arrowFields.push_back(out->children[i].field);
  }
  auto structType = ::arrow::struct_(arrowFields);
  out->field = ::arrow::field(
      groupNode.name(),
      structType,
      groupNode.isOptional(),
      fieldIdMetadata(groupNode.fieldId()));
  out->levelInfo = currentLevels;
  return Status::OK();
}

Status listToSchemaField(
    const GroupNode& group,
    LevelInfo currentLevels,
    SchemaTreeContext* ctx,
    const SchemaField* parent,
    SchemaField* out);

Status mapToSchemaField(
    const GroupNode& group,
    LevelInfo currentLevels,
    SchemaTreeContext* ctx,
    const SchemaField* parent,
    SchemaField* out) {
  if (group.fieldCount() != 1) {
    return Status::Invalid("MAP-annotated groups must have a single child.");
  }
  if (group.isRepeated()) {
    return Status::Invalid("MAP-annotated groups must not be repeated.");
  }

  const Node& keyValueNode = *group.field(0);

  if (!keyValueNode.isRepeated()) {
    return Status::Invalid(
        "Non-repeated key value in a MAP-annotated group are not supported.");
  }

  if (!keyValueNode.isGroup()) {
    return Status::Invalid("Key-value node must be a group.");
  }

  const GroupNode& keyValue = checked_cast<const GroupNode&>(keyValueNode);
  if (keyValue.fieldCount() != 1 && keyValue.fieldCount() != 2) {
    return Status::Invalid(
        "Key-value map node must have 1 or 2 child elements. Found: ",
        keyValue.fieldCount());
  }
  const Node& keyNode = *keyValue.field(0);
  if (!keyNode.isRequired()) {
    return Status::Invalid("Map keys must be annotated as required.");
  }
  // Arrow doesn't support 1 column maps (i.e. Sets).  The options are to
  // either make the values column nullable, or process the map as a list.  We
  // choose the latter as it is simpler.
  if (keyValue.fieldCount() == 1) {
    return listToSchemaField(group, currentLevels, ctx, parent, out);
  }

  incrementLevels(currentLevels, group);
  int16_t repeatedAncestorDefLevel = currentLevels.IncrementRepeated();

  out->children.resize(1);
  SchemaField* keyValueField = &out->children[0];

  keyValueField->children.resize(2);
  SchemaField* keyField = &keyValueField->children[0];
  SchemaField* valueField = &keyValueField->children[1];

  ctx->linkParent(out, parent);
  ctx->linkParent(keyValueField, out);
  ctx->linkParent(keyField, keyValueField);
  ctx->linkParent(valueField, keyValueField);

  // required/optional group name=whatever {
  //   repeated group name=key_values{
  //     required TYPE key;
  // required/optional TYPE value;
  //   }
  // }
  //

  RETURN_NOT_OK(nodeToSchemaField(
      *keyValue.field(0), currentLevels, ctx, keyValueField, keyField));
  RETURN_NOT_OK(nodeToSchemaField(
      *keyValue.field(1), currentLevels, ctx, keyValueField, valueField));

  keyValueField->field = ::arrow::field(
      group.name(),
      ::arrow::struct_({keyField->field, valueField->field}),
      false,
      fieldIdMetadata(keyValue.fieldId()));
  keyValueField->levelInfo = currentLevels;

  out->field = ::arrow::field(
      group.name(),
      std::make_shared<::arrow::MapType>(keyValueField->field),
      group.isOptional(),
      fieldIdMetadata(group.fieldId()));
  out->levelInfo = currentLevels;
  // At this point current levels contains the def level for this list.
  // We need to reset to the prior parent.
  out->levelInfo.repeatedAncestorDefLevel = repeatedAncestorDefLevel;
  return Status::OK();
}

Status listToSchemaField(
    const GroupNode& group,
    LevelInfo currentLevels,
    SchemaTreeContext* ctx,
    const SchemaField* parent,
    SchemaField* out) {
  if (group.fieldCount() != 1) {
    return Status::Invalid("LIST-annotated groups must have a single child.");
  }
  if (group.isRepeated()) {
    return Status::Invalid("LIST-annotated groups must not be repeated.");
  }
  incrementLevels(currentLevels, group);

  out->children.resize(group.fieldCount());
  SchemaField* childField = &out->children[0];

  ctx->linkParent(out, parent);
  ctx->linkParent(childField, out);

  const Node& listNode = *group.field(0);

  if (!listNode.isRepeated()) {
    return Status::Invalid(
        "Non-repeated nodes in a LIST-annotated group are not supported.");
  }

  int16_t repeatedAncestorDefLevel = currentLevels.IncrementRepeated();
  if (listNode.isGroup()) {
    // Resolve 3-level encoding.
    //
    // required/optional group name=whatever {
    //   repeated group name=list {
    //     required/optional TYPE item;
    //   }
    // }
    //
    // Yields list<item: TYPE ?nullable> ?nullable.
    //
    // We distinguish the special case that we have.
    //
    // required/optional group name=whatever {
    //   repeated group name=array or $SOMETHING_tuple {
    //     required/optional TYPE item;
    //   }
    // }
    //
    // In this latter case, the inner type of the list should be a struct
    // rather than a primitive value.
    //
    // Yields list<item: struct<item: TYPE ?nullable> not null> ?nullable.
    const auto& listGroup = static_cast<const GroupNode&>(listNode);
    // Special case mentioned in the format spec:
    //   If the name is array or ends in _tuple, this should be a list of
    //   struct, even for single child elements.
    if (listGroup.fieldCount() == 1 && !hasStructListName(listGroup)) {
      // List of primitive type.
      RETURN_NOT_OK(nodeToSchemaField(
          *listGroup.field(0), currentLevels, ctx, out, childField));
    } else {
      RETURN_NOT_OK(
          groupToStruct(listGroup, currentLevels, ctx, out, childField));
    }
  } else {
    // Two-level list encoding.
    //
    // required/optional group LIST {
    //   repeated TYPE;
    // }
    const auto& primitiveNode = static_cast<const PrimitiveNode&>(listNode);
    int columnIndex = ctx->schema->getColumnIndex(primitiveNode);
    ARROW_ASSIGN_OR_RAISE(
        std::shared_ptr<ArrowType> type,
        getTypeForNode(columnIndex, primitiveNode, ctx));
    auto itemField = ::arrow::field(
        listNode.name(), type, false, fieldIdMetadata(listNode.fieldId()));
    RETURN_NOT_OK(populateLeaf(
        columnIndex, itemField, currentLevels, ctx, out, childField));
  }
  out->field = ::arrow::field(
      group.name(),
      ::arrow::list(childField->field),
      group.isOptional(),
      fieldIdMetadata(group.fieldId()));
  out->levelInfo = currentLevels;
  // At this point current levels contains the def level for this list.
  // We need to reset to the prior parent.
  out->levelInfo.repeatedAncestorDefLevel = repeatedAncestorDefLevel;
  return Status::OK();
}

Status groupToSchemaField(
    const GroupNode& groupNode,
    LevelInfo currentLevels,
    SchemaTreeContext* ctx,
    const SchemaField* parent,
    SchemaField* out) {
  if (groupNode.logicalType()->isList()) {
    return listToSchemaField(groupNode, currentLevels, ctx, parent, out);
  } else if (groupNode.logicalType()->isMap()) {
    return mapToSchemaField(groupNode, currentLevels, ctx, parent, out);
  }
  std::shared_ptr<ArrowType> type;
  if (groupNode.isRepeated()) {
    // Simple repeated struct.
    //
    // repeated group $NAME {
    //   R/o TYPE[0] f0.
    //   R/o TYPE[1] f1.
    // }
    out->children.resize(1);

    int16_t repeatedAncestorDefLevel = currentLevels.IncrementRepeated();
    RETURN_NOT_OK(
        groupToStruct(groupNode, currentLevels, ctx, out, &out->children[0]));
    out->field = ::arrow::field(
        groupNode.name(),
        ::arrow::list(out->children[0].field),
        false,
        fieldIdMetadata(groupNode.fieldId()));

    ctx->linkParent(&out->children[0], out);
    out->levelInfo = currentLevels;
    // At this point current_levels contains this list as the def level, we
    // need to use the previous ancestor of this list.
    out->levelInfo.repeatedAncestorDefLevel = repeatedAncestorDefLevel;
    return Status::OK();
  } else {
    incrementLevels(currentLevels, groupNode);
    return groupToStruct(groupNode, currentLevels, ctx, parent, out);
  }
}

Status nodeToSchemaField(
    const Node& node,
    LevelInfo currentLevels,
    SchemaTreeContext* ctx,
    const SchemaField* parent,
    SchemaField* out) {
  // Workhorse function for converting a Parquet schema node to an Arrow
  // type. Handles different conventions for nested data.

  ctx->linkParent(out, parent);

  // Now, walk the schema and create a ColumnDescriptor for each leaf node.
  if (node.isGroup()) {
    // A nested field, but we don't know what kind yet.
    return groupToSchemaField(
        static_cast<const GroupNode&>(node), currentLevels, ctx, parent, out);
  } else {
    // Either a normal flat primitive type, or a list type encoded with 1-level.
    // List encoding. Note that the 3-level encoding is the form recommended by
    // the Parquet specification, but technically we can have either.
    //
    // Required/optional $TYPE $FIELD_NAME.
    //
    // Or.
    //
    // Repeated $TYPE $FIELD_NAME.
    const auto& primitiveNode = static_cast<const PrimitiveNode&>(node);
    int columnIndex = ctx->schema->getColumnIndex(primitiveNode);
    ARROW_ASSIGN_OR_RAISE(
        std::shared_ptr<ArrowType> type,
        getTypeForNode(columnIndex, primitiveNode, ctx));
    if (node.isRepeated()) {
      // One-level list encoding, e.g.
      // a: repeated int32;
      int16_t repeatedAncestorDefLevel = currentLevels.IncrementRepeated();
      out->children.resize(1);
      auto childField = ::arrow::field(node.name(), type, false);
      RETURN_NOT_OK(populateLeaf(
          columnIndex, childField, currentLevels, ctx, out, &out->children[0]));

      out->field = ::arrow::field(
          node.name(),
          ::arrow::list(childField),
          false,
          fieldIdMetadata(node.fieldId()));
      out->levelInfo = currentLevels;
      // At this point current_levels has consider this list the ancestor so
      // restore the actual ancestor.
      out->levelInfo.repeatedAncestorDefLevel = repeatedAncestorDefLevel;
      return Status::OK();
    } else {
      incrementLevels(currentLevels, node);
      // A normal (required/optional) primitive node.
      return populateLeaf(
          columnIndex,
          ::arrow::field(
              node.name(),
              type,
              node.isOptional(),
              fieldIdMetadata(node.fieldId())),
          currentLevels,
          ctx,
          parent,
          out);
    }
  }
}

// Get the original Arrow schema, as serialized in the Parquet metadata.
Status getOriginSchema(
    const std::shared_ptr<const KeyValueMetadata>& metadata,
    std::shared_ptr<const KeyValueMetadata>* cleanMetadata,
    std::shared_ptr<::arrow::Schema>* out) {
  if (metadata == nullptr) {
    *out = nullptr;
    *cleanMetadata = nullptr;
    return Status::OK();
  }

  static const std::string kArrowSchemaKey = "ARROW:schema";
  int schemaIndex = metadata->FindKey(kArrowSchemaKey);
  if (schemaIndex == -1) {
    *out = nullptr;
    *cleanMetadata = metadata;
    return Status::OK();
  }

  // The original Arrow schema was serialized using the store_schema option.
  // We deserialize it here and use it to inform read options such as
  // dictionary-encoded fields.
  auto decoded = ::arrow::util::base64_decode(metadata->value(schemaIndex));
  auto schemaBuf = std::make_shared<Buffer>(decoded);

  ::arrow::ipc::DictionaryMemo dictMemo;
  ::arrow::io::BufferReader input(schemaBuf);

  ARROW_ASSIGN_OR_RAISE(*out, ::arrow::ipc::ReadSchema(&input, &dictMemo));

  if (metadata->size() > 1) {
    // Copy the metadata without the schema key.
    auto newMetadata = ::arrow::key_value_metadata({}, {});
    newMetadata->reserve(metadata->size() - 1);
    for (int64_t i = 0; i < metadata->size(); ++i) {
      if (i == schemaIndex)
        continue;
      newMetadata->Append(metadata->key(i), metadata->value(i));
    }
    *cleanMetadata = newMetadata;
  } else {
    // No other keys, let metadata be null.
    *cleanMetadata = nullptr;
  }
  return Status::OK();
}

// Restore original Arrow field information that was serialized as Parquet.
// metadata but that is not necessarily present in the field reconstituted from
// Parquet data (for example, Parquet timestamp types doesn't carry timezone
// information).

Result<bool> applyOriginalMetadata(
    const Field& originField,
    SchemaField* inferred);

std::function<std::shared_ptr<::arrow::DataType>(FieldVector)> getNestedFactory(
    const ArrowType& originType,
    const ArrowType& inferredType) {
  switch (inferredType.id()) {
    case ::arrow::Type::STRUCT:
      if (originType.id() == ::arrow::Type::STRUCT) {
        return [](FieldVector fields) {
          return ::arrow::struct_(std::move(fields));
        };
      }
      break;
    case ::arrow::Type::LIST:
      if (originType.id() == ::arrow::Type::LIST) {
        return [](FieldVector fields) {
          VELOX_DCHECK_EQ(fields.size(), 1);
          return ::arrow::list(std::move(fields[0]));
        };
      }
      if (originType.id() == ::arrow::Type::LARGE_LIST) {
        return [](FieldVector fields) {
          VELOX_DCHECK_EQ(fields.size(), 1);
          return ::arrow::large_list(std::move(fields[0]));
        };
      }
      if (originType.id() == ::arrow::Type::FIXED_SIZE_LIST) {
        const auto listSize =
            checked_cast<const ::arrow::FixedSizeListType&>(originType)
                .list_size();
        return [listSize](FieldVector fields) {
          VELOX_DCHECK_EQ(fields.size(), 1);
          return ::arrow::fixed_size_list(std::move(fields[0]), listSize);
        };
      }
      break;
    default:
      break;
  }
  return {};
}

Result<bool> applyOriginalStorageMetadata(
    const Field& originField,
    SchemaField* inferred) {
  bool modified = false;

  auto& originType = originField.type();
  auto& inferredType = inferred->field->type();

  const int numChildren = inferredType->num_fields();

  if (numChildren > 0 && originType->num_fields() == numChildren) {
    VELOX_DCHECK_EQ(static_cast<int>(inferred->children.size()), numChildren);
    const auto factory = getNestedFactory(*originType, *inferredType);
    if (factory) {
      // The type may be modified (e.g. LargeList) while the children stay the
      // same.
      modified |= originType->id() != inferredType->id();

      // Apply original metadata recursively to children.
      for (int i = 0; i < inferredType->num_fields(); ++i) {
        ARROW_ASSIGN_OR_RAISE(
            const bool childModified,
            applyOriginalMetadata(
                *originType->field(i), &inferred->children[i]));
        modified |= childModified;
      }
      if (modified) {
        // Recreate this field using the modified child fields.
        ::arrow::FieldVector modifiedChildren(inferredType->num_fields());
        for (int i = 0; i < inferredType->num_fields(); ++i) {
          modifiedChildren[i] = inferred->children[i].field;
        }
        inferred->field =
            inferred->field->WithType(factory(std::move(modifiedChildren)));
      }
    }
  }

  if (originType->id() == ::arrow::Type::TIMESTAMP &&
      inferredType->id() == ::arrow::Type::TIMESTAMP) {
    // Restore time zone, if any.
    const auto& tsType =
        checked_cast<const ::arrow::TimestampType&>(*inferredType);
    const auto& tsOriginType =
        checked_cast<const ::arrow::TimestampType&>(*originType);

    // If the data is tz-aware, then set the original time zone, since Parquet
    // has no native storage for timezones.
    if (tsType.timezone() == "UTC" && !tsOriginType.timezone().empty()) {
      if (tsType.unit() == tsOriginType.unit()) {
        inferred->field = inferred->field->WithType(originType);
      } else {
        auto tsTypeNew =
            ::arrow::timestamp(tsType.unit(), tsOriginType.timezone());
        inferred->field = inferred->field->WithType(tsTypeNew);
      }
    }
    modified = true;
  }

  if (originType->id() == ::arrow::Type::DURATION &&
      inferredType->id() == ::arrow::Type::INT64) {
    // Read back int64 arrays as duration.
    inferred->field = inferred->field->WithType(originType);
    modified = true;
  }

  if (originType->id() == ::arrow::Type::DICTIONARY &&
      inferredType->id() != ::arrow::Type::DICTIONARY &&
      isDictionaryReadSupported(*inferredType)) {
    // Direct dictionary reads are only supported for a couple primitive types,
    // so no need to recurse on value types.
    const auto& dictOriginType =
        checked_cast<const ::arrow::DictionaryType&>(*originType);
    inferred->field = inferred->field->WithType(
        ::arrow::dictionary(
            ::arrow::int32(), inferredType, dictOriginType.ordered()));
    modified = true;
  }

  if ((originType->id() == ::arrow::Type::LARGE_BINARY &&
       inferredType->id() == ::arrow::Type::BINARY) ||
      (originType->id() == ::arrow::Type::LARGE_STRING &&
       inferredType->id() == ::arrow::Type::STRING)) {
    // Read back binary-like arrays with the intended offset width.
    inferred->field = inferred->field->WithType(originType);
    modified = true;
  }

  if (originType->id() == ::arrow::Type::DECIMAL256 &&
      inferredType->id() == ::arrow::Type::DECIMAL128) {
    inferred->field = inferred->field->WithType(originType);
    modified = true;
  }

  // Restore field metadata.
  std::shared_ptr<const KeyValueMetadata> fieldMetadata =
      originField.metadata();
  if (fieldMetadata != nullptr) {
    if (inferred->field->metadata()) {
      // Prefer the metadata keys (like field_id) from the current metadata.
      fieldMetadata = fieldMetadata->Merge(*inferred->field->metadata());
    }
    inferred->field = inferred->field->WithMetadata(fieldMetadata);
    modified = true;
  }

  return modified;
}

Result<bool> applyOriginalMetadata(
    const Field& originField,
    SchemaField* inferred) {
  bool modified = false;

  auto& originType = originField.type();

  if (originType->id() == ::arrow::Type::EXTENSION) {
    const auto& exType =
        checked_cast<const ::arrow::ExtensionType&>(*originType);
    auto originStorageField = originField.WithType(exType.storage_type());

    // Apply metadata recursively to storage type.
    RETURN_NOT_OK(applyOriginalStorageMetadata(*originStorageField, inferred));

    // Restore extension type, if the storage type is the same as inferred
    // from the Parquet type.
    if (exType.storage_type()->Equals(*inferred->field->type())) {
      inferred->field = inferred->field->WithType(originType);
    }
    modified = true;
  } else {
    ARROW_ASSIGN_OR_RAISE(
        modified, applyOriginalStorageMetadata(originField, inferred));
  }

  return modified;
}

} // namespace

std::shared_ptr<::arrow::KeyValueMetadata> fieldIdMetadata(int fieldId) {
  if (fieldId >= 0) {
    return ::arrow::key_value_metadata({FIELD_ID_KEY}, {toChars(fieldId)});
  } else {
    return nullptr;
  }
}

Status fieldToNode(
    const std::shared_ptr<Field>& field,
    const WriterProperties& properties,
    const ArrowWriterProperties& arrowProperties,
    NodePtr* out) {
  return fieldToNode(field->name(), field, properties, arrowProperties, out);
}

Status toParquetSchema(
    const ::arrow::Schema* arrowSchema,
    const WriterProperties& properties,
    const ArrowWriterProperties& arrowProperties,
    std::shared_ptr<SchemaDescriptor>* out) {
  std::vector<NodePtr> nodes(arrowSchema->num_fields());
  for (int i = 0; i < arrowSchema->num_fields(); i++) {
    RETURN_NOT_OK(fieldToNode(
        arrowSchema->field(i), properties, arrowProperties, &nodes[i]));
  }

  NodePtr schema = GroupNode::make("schema", Repetition::kRequired, nodes);
  *out = std::make_shared<SchemaDescriptor>();
  PARQUET_CATCH_NOT_OK((*out)->init(schema));

  return Status::OK();
}

Status toParquetSchema(
    const ::arrow::Schema* arrowSchema,
    const WriterProperties& properties,
    std::shared_ptr<SchemaDescriptor>* out) {
  return toParquetSchema(
      arrowSchema, properties, *defaultArrowWriterProperties(), out);
}

Status fromParquetSchema(
    const SchemaDescriptor* schema,
    const ArrowReaderProperties& properties,
    const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata,
    std::shared_ptr<::arrow::Schema>* out) {
  SchemaManifest manifest;
  RETURN_NOT_OK(
      SchemaManifest::make(schema, keyValueMetadata, properties, &manifest));
  std::vector<std::shared_ptr<Field>> fields(manifest.schemaFields.size());

  for (int i = 0; i < static_cast<int>(fields.size()); i++) {
    const auto& schemaField = manifest.schemaFields[i];
    fields[i] = schemaField.field;
  }
  if (manifest.originSchema) {
    // ARROW-8980: If the ARROW:schema was in the input metadata, then
    // manifest.originSchema will have it scrubbed out.
    *out = ::arrow::schema(fields, manifest.originSchema->metadata());
  } else {
    *out = ::arrow::schema(fields, keyValueMetadata);
  }
  return Status::OK();
}

Status fromParquetSchema(
    const SchemaDescriptor* parquetSchema,
    const ArrowReaderProperties& properties,
    std::shared_ptr<::arrow::Schema>* out) {
  return fromParquetSchema(parquetSchema, properties, nullptr, out);
}

Status fromParquetSchema(
    const SchemaDescriptor* parquetSchema,
    std::shared_ptr<::arrow::Schema>* out) {
  ArrowReaderProperties properties;
  return fromParquetSchema(parquetSchema, properties, nullptr, out);
}

Status SchemaManifest::make(
    const SchemaDescriptor* schema,
    const std::shared_ptr<const KeyValueMetadata>& metadata,
    const ArrowReaderProperties& properties,
    SchemaManifest* manifest) {
  SchemaTreeContext ctx;
  ctx.manifest = manifest;
  ctx.properties = properties;
  ctx.schema = schema;
  const GroupNode& schemaNode = *schema->groupNode();
  manifest->descr = schema;
  manifest->schemaFields.resize(schemaNode.fieldCount());

  // Try to deserialize original Arrow schema.
  RETURN_NOT_OK(getOriginSchema(
      metadata, &manifest->schemaMetadata, &manifest->originSchema));
  // Ignore original schema if it's not compatible with the Parquet schema.
  if (manifest->originSchema != nullptr &&
      manifest->originSchema->num_fields() != schemaNode.fieldCount()) {
    manifest->originSchema = nullptr;
  }

  for (int i = 0; i < static_cast<int>(schemaNode.fieldCount()); ++i) {
    SchemaField* outField = &manifest->schemaFields[i];
    RETURN_NOT_OK(nodeToSchemaField(
        *schemaNode.field(i), LevelInfo(), &ctx, nullptr, outField));

    // TODO(wesm): As follow up to ARROW-3246, we should really pass the origin
    // schema (if any) through all functions in the schema reconstruction, but
    // I'm being lazy and just setting dictionary fields at the top level for
    // now.
    if (manifest->originSchema == nullptr) {
      continue;
    }

    auto& originField = manifest->originSchema->field(i);
    RETURN_NOT_OK(applyOriginalMetadata(*originField, outField));
  }
  return Status::OK();
}

} // namespace facebook::velox::parquet::arrow::arrow
