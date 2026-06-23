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

#include "velox/dwio/dwrf/utils/ProtoUtils.h"
#include "velox/dwio/common/exception/Exception.h"

namespace facebook::velox::dwrf {

namespace {

template <TypeKind T>
class SchemaType {};

#define CREATE_TYPE_TRAIT(Kind, SchemaKind)       \
  template <>                                     \
  struct SchemaType<TypeKind::Kind> {             \
    static constexpr proto::Type_Kind kind =      \
        proto::Type_Kind::Type_Kind_##SchemaKind; \
  };

CREATE_TYPE_TRAIT(BOOLEAN, BOOLEAN)
CREATE_TYPE_TRAIT(TINYINT, BYTE)
CREATE_TYPE_TRAIT(SMALLINT, SHORT)
CREATE_TYPE_TRAIT(INTEGER, INT)
CREATE_TYPE_TRAIT(BIGINT, LONG)
CREATE_TYPE_TRAIT(REAL, FLOAT)
CREATE_TYPE_TRAIT(DOUBLE, DOUBLE)
CREATE_TYPE_TRAIT(VARCHAR, STRING)
CREATE_TYPE_TRAIT(VARBINARY, BINARY)
CREATE_TYPE_TRAIT(TIMESTAMP, TIMESTAMP)
CREATE_TYPE_TRAIT(ARRAY, LIST)
CREATE_TYPE_TRAIT(MAP, MAP)
CREATE_TYPE_TRAIT(ROW, STRUCT)

#undef CREATE_TYPE_TRAIT

} // namespace

void ProtoUtils::writeType(
    const Type& type,
    FooterWriteWrapper& footer,
    TypeWriteWrapper* parent,
    const AttributeProvider& attributeProvider) {
  auto self = footer.addTypes();
  const uint32_t typeId = footer.typesSize() - 1;
  if (parent) {
    parent->addSubtypes(static_cast<int>(typeId));
  }

  auto kind =
      VELOX_STATIC_FIELD_DYNAMIC_DISPATCH(SchemaType, kind, type.kind());
  auto typeKindWrapper = TypeKindWrapper(&kind);
  self.setKind(typeKindWrapper);

  // Stamp per-type attributes (e.g. Iceberg field ids) before recursing into
  // children. An empty result keeps the wire format byte-identical to the
  // pre-attributes serialization for callers that do not provide attributes.
  if (attributeProvider) {
    for (const auto& [key, value] : attributeProvider(typeId)) {
      self.addAttribute(key, value);
    }
  }

  switch (type.kind()) {
    case TypeKind::ROW: {
      auto& row = type.asRow();
      for (size_t i = 0; i < row.size(); ++i) {
        self.addFieldnames(row.nameOf(i));
        writeType(*row.childAt(i), footer, &self, attributeProvider);
      }
      break;
    }
    case TypeKind::ARRAY:
      writeType(
          *type.asArray().elementType(), footer, &self, attributeProvider);
      break;
    case TypeKind::MAP: {
      auto& map = type.asMap();
      writeType(*map.keyType(), footer, &self, attributeProvider);
      writeType(*map.valueType(), footer, &self, attributeProvider);
      break;
    }
    default:
      DWIO_ENSURE(type.isPrimitiveType());
      break;
  }
}

std::unordered_map<uint32_t, std::vector<std::pair<std::string, std::string>>>
ProtoUtils::readAttributes(const FooterWrapper& footer) {
  std::unordered_map<uint32_t, std::vector<std::pair<std::string, std::string>>>
      result;
  for (int32_t i = 0; i < footer.typesSize(); ++i) {
    const auto type = footer.types(i);
    const auto numAttributes = type.attributesSize();
    if (numAttributes == 0) {
      continue;
    }
    std::vector<std::pair<std::string, std::string>> attributes;
    attributes.reserve(numAttributes);
    for (int32_t j = 0; j < numAttributes; ++j) {
      attributes.emplace_back(type.attribute(j));
    }
    result.emplace(static_cast<uint32_t>(i), std::move(attributes));
  }
  return result;
}

std::shared_ptr<const Type> ProtoUtils::fromFooter(
    const proto::Footer& footer,
    std::function<bool(uint32_t)> selector,
    uint32_t index) {
  const auto& type = footer.types(index);
  switch (type.kind()) {
    case proto::Type_Kind_BOOLEAN:
    case proto::Type_Kind_BYTE:
    case proto::Type_Kind_SHORT:
    case proto::Type_Kind_INT:
    case proto::Type_Kind_LONG:
    case proto::Type_Kind_FLOAT:
    case proto::Type_Kind_DOUBLE:
    case proto::Type_Kind_STRING:
    case proto::Type_Kind_BINARY:
    case proto::Type_Kind_TIMESTAMP:
      return createScalarType(static_cast<TypeKind>(type.kind()));
    case proto::Type_Kind_LIST:
      return ARRAY(fromFooter(footer, selector, type.subtypes(0)));
    case proto::Type_Kind_MAP:
      return MAP(
          fromFooter(footer, selector, type.subtypes(0)),
          fromFooter(footer, selector, type.subtypes(1)));
    case proto::Type_Kind_UNION: {
      DWIO_RAISE("union type is deprecated");
    }
    case proto::Type_Kind_STRUCT: {
      std::vector<std::shared_ptr<const Type>> tl;
      std::vector<std::string> names;
      for (int32_t i = 0; i < type.subtypes_size(); ++i) {
        auto typeId = type.subtypes(i);
        if (selector(typeId)) {
          auto child = fromFooter(footer, selector, typeId);
          names.push_back(type.fieldnames(i));
          tl.push_back(std::move(child));
        }
      }
      // NOTE: There are empty dwrf files in data warehouse that has empty
      // struct as the root type. So the assumption that struct has at least one
      // child doesn't hold.
      return ROW(std::move(names), std::move(tl));
    }
    default:
      DWIO_RAISE("unknown type");
  }
}

} // namespace facebook::velox::dwrf
