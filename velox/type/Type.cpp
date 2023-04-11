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

#include "velox/type/Type.h"
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <folly/Demangle.h>
#include <sstream>
#include <typeindex>
#include "velox/common/base/Exceptions.h"

namespace std {
template <>
struct hash<facebook::velox::TypeKind> {
  size_t operator()(const facebook::velox::TypeKind& typeKind) const {
    return std::hash<int32_t>()((int32_t)typeKind);
  }
};
} // namespace std

namespace {
bool isColumnNameRequiringEscaping(const std::string& name) {
  static const boost::regex re("^[a-zA-Z_][a-zA-Z0-9_]*$");
  return !regex_match(name, re);
}
} // namespace

namespace facebook::velox {

bool isDecimalName(const std::string& typeName) {
  auto typeNameUpper = boost::algorithm::to_upper_copy(typeName);
  return (
      typeNameUpper == TypeTraits<TypeKind::SHORT_DECIMAL>::name ||
      typeNameUpper == TypeTraits<TypeKind::LONG_DECIMAL>::name);
}

bool isDecimalTypeSignature(const std::string& arg) {
  auto upper = boost::algorithm::to_upper_copy(arg);
  return (
      upper.find(TypeTraits<TypeKind::SHORT_DECIMAL>::name) !=
          std::string::npos ||
      upper.find(TypeTraits<TypeKind::LONG_DECIMAL>::name) !=
          std::string::npos);
}

// Static variable intialization is not thread safe for non
// constant-initialization, but scoped static initialization is thread safe.
const std::unordered_map<std::string, TypeKind>& getTypeStringMap() {
  static const std::unordered_map<std::string, TypeKind> kTypeStringMap{
      {"BOOLEAN", TypeKind::BOOLEAN},
      {"TINYINT", TypeKind::TINYINT},
      {"SMALLINT", TypeKind::SMALLINT},
      {"INTEGER", TypeKind::INTEGER},
      {"BIGINT", TypeKind::BIGINT},
      {"REAL", TypeKind::REAL},
      {"DOUBLE", TypeKind::DOUBLE},
      {"VARCHAR", TypeKind::VARCHAR},
      {"VARBINARY", TypeKind::VARBINARY},
      {"TIMESTAMP", TypeKind::TIMESTAMP},
      {"DATE", TypeKind::DATE},
      {"SHORT_DECIMAL", TypeKind::SHORT_DECIMAL},
      {"LONG_DECIMAL", TypeKind::LONG_DECIMAL},
      {"ARRAY", TypeKind::ARRAY},
      {"MAP", TypeKind::MAP},
      {"ROW", TypeKind::ROW},
      {"FUNCTION", TypeKind::FUNCTION},
      {"UNKNOWN", TypeKind::UNKNOWN},
      {"OPAQUE", TypeKind::OPAQUE},
      {"INVALID", TypeKind::INVALID}};
  return kTypeStringMap;
}

std::optional<TypeKind> tryMapNameToTypeKind(const std::string& name) {
  auto found = getTypeStringMap().find(name);

  if (found == getTypeStringMap().end()) {
    return std::nullopt;
  }

  return found->second;
}

TypeKind mapNameToTypeKind(const std::string& name) {
  auto found = getTypeStringMap().find(name);

  if (found == getTypeStringMap().end()) {
    VELOX_USER_FAIL("Specified element is not found : {}", name);
  }

  return found->second;
}

std::string mapTypeKindToName(const TypeKind& typeKind) {
  static std::unordered_map<TypeKind, std::string> typeEnumMap{
      {TypeKind::BOOLEAN, "BOOLEAN"},
      {TypeKind::TINYINT, "TINYINT"},
      {TypeKind::SMALLINT, "SMALLINT"},
      {TypeKind::INTEGER, "INTEGER"},
      {TypeKind::BIGINT, "BIGINT"},
      {TypeKind::REAL, "REAL"},
      {TypeKind::DOUBLE, "DOUBLE"},
      {TypeKind::VARCHAR, "VARCHAR"},
      {TypeKind::VARBINARY, "VARBINARY"},
      {TypeKind::TIMESTAMP, "TIMESTAMP"},
      {TypeKind::DATE, "DATE"},
      {TypeKind::SHORT_DECIMAL, "SHORT_DECIMAL"},
      {TypeKind::LONG_DECIMAL, "LONG_DECIMAL"},
      {TypeKind::ARRAY, "ARRAY"},
      {TypeKind::MAP, "MAP"},
      {TypeKind::ROW, "ROW"},
      {TypeKind::FUNCTION, "FUNCTION"},
      {TypeKind::UNKNOWN, "UNKNOWN"},
      {TypeKind::OPAQUE, "OPAQUE"},
      {TypeKind::INVALID, "INVALID"}};

  auto found = typeEnumMap.find(typeKind);

  if (found == typeEnumMap.end()) {
    VELOX_USER_FAIL("Specified element is not found : {}", (int32_t)typeKind);
  }

  return found->second;
}

std::pair<int, int> getDecimalPrecisionScale(const Type& type) {
  VELOX_CHECK(type.isShortDecimal() || type.isLongDecimal());
  if (type.isShortDecimal()) {
    const auto& decimalType = type.asShortDecimal();
    return {decimalType.precision(), decimalType.scale()};
  } else {
    const auto& decimalType = type.asLongDecimal();
    return {decimalType.precision(), decimalType.scale()};
  }
}

namespace {
struct OpaqueSerdeRegistry {
  struct Entry {
    std::string persistentName;
    // to avoid creating new shared_ptr's every time
    OpaqueType::SerializeFunc<void> serialize;
    OpaqueType::DeserializeFunc<void> deserialize;
  };
  std::unordered_map<std::type_index, Entry> mapping;
  std::unordered_map<std::string, std::shared_ptr<const OpaqueType>> reverse;

  static OpaqueSerdeRegistry& get() {
    static OpaqueSerdeRegistry instance;
    return instance;
  }
};
} // namespace

std::ostream& operator<<(std::ostream& os, const TypeKind& kind) {
  os << mapTypeKindToName(kind);
  return os;
}

namespace {
std::vector<TypePtr> deserializeChildTypes(const folly::dynamic& obj) {
  return velox::ISerializable::deserialize<std::vector<Type>>(obj["cTypes"]);
}
} // namespace

TypePtr Type::create(const folly::dynamic& obj) {
  std::vector<TypePtr> childTypes;
  if (obj.find("cTypes") != obj.items().end()) {
    childTypes = deserializeChildTypes(obj);
  }

  // Checks if 'typeName' specifies a custom type.
  auto typeName = obj["type"].asString();
  if (customTypeExists(typeName)) {
    return getCustomType(typeName);
  }

  // 'typeName' must be a built-in type.
  TypeKind typeKind = mapNameToTypeKind(typeName);
  switch (typeKind) {
    case TypeKind::SHORT_DECIMAL: {
      VELOX_USER_CHECK(
          childTypes.empty(), "Short decimal type should not have child types");
      return SHORT_DECIMAL(obj["precision"].asInt(), obj["scale"].asInt());
    }
    case TypeKind::LONG_DECIMAL: {
      VELOX_USER_CHECK(
          childTypes.empty(), "Long decimal type should not have child types");
      return LONG_DECIMAL(obj["precision"].asInt(), obj["scale"].asInt());
    }
    case TypeKind::ROW: {
      VELOX_USER_CHECK(obj["names"].isArray());
      std::vector<std::string> names;
      for (const auto& name : obj["names"]) {
        names.push_back(name.asString());
      }

      return std::make_shared<const RowType>(
          std::move(names), std::move(childTypes));
    }

    case TypeKind::OPAQUE: {
      const auto& persistentName = obj["opaque"].asString();
      const auto& registry = OpaqueSerdeRegistry::get();
      auto it = registry.reverse.find(persistentName);
      VELOX_USER_CHECK(
          it != registry.reverse.end(),
          "Opaque type with persistent name '{}' is not registered",
          persistentName);
      if (auto withExtra = it->second->deserializeExtra(obj)) {
        return withExtra;
      }
      return it->second;
    }
    default: {
      return createType(typeKind, std::move(childTypes));
    }
  }
}

// static
void Type::registerSerDe() {
  auto& registry = velox::DeserializationRegistryForSharedPtr();
  registry.Register(
      Type::getClassName(),
      static_cast<std::shared_ptr<const Type> (*)(const folly::dynamic&)>(
          Type::create));

  registry.Register("IntervalDayTimeType", IntervalDayTimeType::deserialize);
}

std::string ArrayType::toString() const {
  return "ARRAY<" + child_->toString() + ">";
}

const TypePtr& ArrayType::childAt(uint32_t idx) const {
  VELOX_USER_CHECK_EQ(idx, 0, "List type should have only one child");
  return elementType();
}

ArrayType::ArrayType(TypePtr child)
    : child_{std::move(child)}, parameters_{{TypeParameter(child_)}} {}

bool ArrayType::equivalent(const Type& other) const {
  if (&other == this) {
    return true;
  }
  if (!Type::hasSameTypeId(other)) {
    return false;
  }
  auto& otherArray = other.asArray();
  return child_->equivalent(*otherArray.child_);
}

folly::dynamic ArrayType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = TypeTraits<TypeKind::ARRAY>::name;

  folly::dynamic children = folly::dynamic::array;
  children.push_back(child_->serialize());
  obj["cTypes"] = children;

  return obj;
}

const TypePtr& MapType::childAt(uint32_t idx) const {
  if (idx == 0) {
    return keyType();
  } else if (idx == 1) {
    return valueType();
  }
  VELOX_USER_FAIL(
      "Map type should have only two children. Tried to access child '{}'",
      idx);
}

MapType::MapType(TypePtr keyType, TypePtr valueType)
    : keyType_{std::move(keyType)},
      valueType_{std::move(valueType)},
      parameters_{{TypeParameter(keyType_), TypeParameter(valueType_)}} {}

std::string MapType::toString() const {
  return "MAP<" + keyType()->toString() + "," + valueType()->toString() + ">";
}

folly::dynamic MapType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = TypeTraits<TypeKind::MAP>::name;

  folly::dynamic children = folly::dynamic::array;
  children.push_back(keyType()->serialize());
  children.push_back(valueType()->serialize());
  obj["cTypes"] = children;

  return obj;
}

namespace {
std::vector<TypeParameter> createTypeParameters(
    const std::vector<TypePtr>& children) {
  std::vector<TypeParameter> parameters;
  parameters.reserve(children.size());
  for (const auto& child : children) {
    parameters.push_back(TypeParameter(child));
  }
  return parameters;
}
} // namespace

RowType::RowType(std::vector<std::string>&& names, std::vector<TypePtr>&& types)
    : names_{std::move(names)},
      children_{std::move(types)},
      parameters_{createTypeParameters(children_)} {
  VELOX_USER_CHECK_EQ(
      names_.size(), children_.size(), "Mismatch names/types sizes");
}

uint32_t RowType::size() const {
  return children_.size();
}

const TypePtr& RowType::childAt(uint32_t idx) const {
  return children_.at(idx);
}

namespace {
template <typename T>
std::string makeFieldNotFoundErrorMessage(
    const T& name,
    const std::vector<std::string>& availableNames) {
  std::stringstream errorMessage;
  errorMessage << "Field not found: " << name << ". Available fields are: ";
  for (auto i = 0; i < availableNames.size(); ++i) {
    if (i > 0) {
      errorMessage << ", ";
    }
    errorMessage << availableNames[i];
  }
  errorMessage << ".";
  return errorMessage.str();
}
} // namespace

const TypePtr& RowType::findChild(folly::StringPiece name) const {
  for (uint32_t i = 0; i < names_.size(); ++i) {
    if (names_.at(i) == name) {
      return children_.at(i);
    }
  }
  VELOX_USER_FAIL(makeFieldNotFoundErrorMessage(name, names_));
}

bool RowType::containsChild(std::string_view name) const {
  return std::find(names_.begin(), names_.end(), name) != names_.end();
}

uint32_t RowType::getChildIdx(const std::string& name) const {
  auto index = getChildIdxIfExists(name);
  if (!index.has_value()) {
    VELOX_USER_FAIL(makeFieldNotFoundErrorMessage(name, names_));
  }
  return index.value();
}

std::optional<uint32_t> RowType::getChildIdxIfExists(
    const std::string& name) const {
  for (uint32_t i = 0; i < names_.size(); i++) {
    if (names_.at(i) == name) {
      return i;
    }
  }
  return std::nullopt;
}

bool RowType::equivalent(const Type& other) const {
  if (&other == this) {
    return true;
  }
  if (!Type::hasSameTypeId(other)) {
    return false;
  }
  auto& otherTyped = other.asRow();
  if (otherTyped.size() != size()) {
    return false;
  }
  for (size_t i = 0; i < size(); ++i) {
    if (*childAt(i) != *otherTyped.childAt(i)) {
      return false;
    }
  }
  return true;
}

bool RowType::equals(const Type& other) const {
  if (!this->equivalent(other)) {
    return false;
  }
  auto& otherTyped = other.asRow();
  for (size_t i = 0; i < size(); ++i) {
    // todo: case sensitivity
    if (nameOf(i) != otherTyped.nameOf(i)) {
      return false;
    }
  }
  return true;
}

bool RowType::operator==(const Type& other) const {
  return this->equals(other);
}

bool RowType::operator==(const RowType& other) const {
  return this->equals(other);
}

void RowType::printChildren(std::stringstream& ss, std::string_view delimiter)
    const {
  bool any = false;
  for (size_t i = 0; i < children_.size(); ++i) {
    if (any) {
      ss << delimiter;
    }
    const auto& name = names_.at(i);
    if (isColumnNameRequiringEscaping(name)) {
      ss << std::quoted(name, '"', '"');
    } else {
      ss << name;
    }
    ss << ':' << children_.at(i)->toString();
    any = true;
  }
}

std::string RowType::toString() const {
  std::stringstream ss;
  ss << (TypeTraits<TypeKind::ROW>::name) << "<";
  printChildren(ss);
  ss << ">";
  return ss.str();
}

folly::dynamic RowType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = TypeTraits<TypeKind::ROW>::name;
  obj["names"] = velox::ISerializable::serialize(names_);
  obj["cTypes"] = velox::ISerializable::serialize(children_);
  return obj;
}

size_t Type::hashKind() const {
  size_t hash = (int32_t)kind();
  for (auto& child : *this) {
    hash = hash * 31 + child->hashKind();
  }
  return hash;
}

bool Type::kindEquals(const TypePtr& other) const {
  // recursive kind match (ignores names)
  if (this->kind() != other->kind()) {
    return false;
  }
  if (this->size() != other->size()) {
    return false;
  }
  for (size_t i = 0; i < this->size(); ++i) {
    if (!this->childAt(i)->kindEquals(other->childAt(i))) {
      return false;
    }
  }
  return true;
}

bool MapType::equivalent(const Type& other) const {
  if (&other == this) {
    return true;
  }
  if (!Type::hasSameTypeId(other)) {
    return false;
  }
  auto& otherMap = other.asMap();
  return keyType_->equivalent(*otherMap.keyType_) &&
      valueType_->equivalent(*otherMap.valueType_);
}

FunctionType::FunctionType(
    std::vector<std::shared_ptr<const Type>>&& argumentTypes,
    std::shared_ptr<const Type> returnType)
    : children_(allChildren(std::move(argumentTypes), returnType)),
      parameters_{createTypeParameters(children_)} {}

bool FunctionType::equivalent(const Type& other) const {
  if (&other == this) {
    return true;
  }
  if (!Type::hasSameTypeId(other)) {
    return false;
  }
  auto& otherTyped = *reinterpret_cast<const FunctionType*>(&other);
  return children_ == otherTyped.children_;
}

std::string FunctionType::toString() const {
  std::stringstream out;
  out << "FUNCTION<";
  for (auto i = 0; i < children_.size(); ++i) {
    out << children_[i]->toString() << (i == children_.size() - 1 ? "" : ", ");
  }
  out << ">";
  return out.str();
}

folly::dynamic FunctionType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = TypeTraits<TypeKind::FUNCTION>::name;
  obj["cTypes"] = velox::ISerializable::serialize(children_);
  return obj;
}

OpaqueType::OpaqueType(const std::type_index& typeIndex)
    : typeIndex_(typeIndex) {}

bool OpaqueType::equivalent(const Type& other) const {
  if (&other == this) {
    return true;
  }
  if (!Type::hasSameTypeId(other)) {
    return false;
  }
  return true;
}

bool OpaqueType::operator==(const Type& other) const {
  if (&other == this) {
    return true;
  }
  if (!this->equivalent(other)) {
    return false;
  }
  auto& otherTyped = *reinterpret_cast<const OpaqueType*>(&other);
  return typeIndex_ == otherTyped.typeIndex_;
}

std::string OpaqueType::toString() const {
  std::stringstream out;
  out << "OPAQUE<" << folly::demangle(typeIndex_.name()) << ">";
  return out.str();
}

folly::dynamic OpaqueType::serialize() const {
  const auto& registry = OpaqueSerdeRegistry::get();
  auto it = registry.mapping.find(typeIndex_);
  VELOX_CHECK(
      it != registry.mapping.end(),
      "No serialization persistent name registered for {}",
      toString());

  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = TypeTraits<TypeKind::OPAQUE>::name;
  obj["opaque"] = it->second.persistentName;
  return obj;
}

OpaqueType::SerializeFunc<void> OpaqueType::getSerializeFunc() const {
  const auto& registry = OpaqueSerdeRegistry::get();
  auto it = registry.mapping.find(typeIndex_);
  VELOX_CHECK(
      it != registry.mapping.end() && it->second.serialize,
      "No serialization function registered for {}",
      toString());
  return it->second.serialize;
}

OpaqueType::DeserializeFunc<void> OpaqueType::getDeserializeFunc() const {
  const auto& registry = OpaqueSerdeRegistry::get();
  auto it = registry.mapping.find(typeIndex_);
  VELOX_CHECK(
      it != registry.mapping.end() && it->second.deserialize,
      "No deserialization function registered for {}",
      toString());
  return it->second.deserialize;
}

std::shared_ptr<const OpaqueType> OpaqueType::deserializeExtra(
    const folly::dynamic&) const {
  return nullptr;
}

void OpaqueType::registerSerializationTypeErased(
    const std::shared_ptr<const OpaqueType>& type,
    const std::string& persistentName,
    SerializeFunc<void> serialize,
    DeserializeFunc<void> deserialize) {
  auto& registry = OpaqueSerdeRegistry::get();
  VELOX_CHECK(
      !registry.mapping.count(type->typeIndex_),
      "Trying to register duplicated serialization information for type {}",
      type->toString());
  VELOX_CHECK(
      !registry.reverse.count(persistentName),
      "Trying to register duplicated persistent type name '{}' for type {}, "
      "it's already taken by type {}",
      persistentName,
      type->toString(),
      registry.reverse.at(persistentName)->toString());
  registry.mapping[type->typeIndex_] = {
      .persistentName = persistentName,
      .serialize = serialize,
      .deserialize = deserialize};
  registry.reverse[persistentName] = type;
}

std::shared_ptr<const ArrayType> ARRAY(TypePtr elementType) {
  return std::make_shared<const ArrayType>(std::move(elementType));
}

std::shared_ptr<const RowType> ROW(
    std::vector<std::string>&& names,
    std::vector<TypePtr>&& types) {
  return TypeFactory<TypeKind::ROW>::create(std::move(names), std::move(types));
}

std::shared_ptr<const RowType> ROW(
    std::initializer_list<std::pair<const std::string, TypePtr>>&& pairs) {
  std::vector<TypePtr> types;
  std::vector<std::string> names;
  types.reserve(pairs.size());
  names.reserve(pairs.size());
  for (auto& p : pairs) {
    types.push_back(p.second);
    names.push_back(p.first);
  }
  return TypeFactory<TypeKind::ROW>::create(std::move(names), std::move(types));
}

std::shared_ptr<const RowType> ROW(std::vector<TypePtr>&& types) {
  std::vector<std::string> names;
  names.reserve(types.size());
  for (auto& p : types) {
    names.push_back("");
  }
  return TypeFactory<TypeKind::ROW>::create(std::move(names), std::move(types));
}

std::shared_ptr<const MapType> MAP(TypePtr keyType, TypePtr valType) {
  return std::make_shared<const MapType>(
      std::move(keyType), std::move(valType));
};

std::shared_ptr<const FunctionType> FUNCTION(
    std::vector<TypePtr>&& argumentTypes,
    TypePtr returnType) {
  return std::make_shared<const FunctionType>(
      std::move(argumentTypes), std::move(returnType));
};

#define KOSKI_DEFINE_SCALAR_ACCESSOR(KIND)                   \
  std::shared_ptr<const ScalarType<TypeKind::KIND>> KIND() { \
    return ScalarType<TypeKind::KIND>::create();             \
  }

KOSKI_DEFINE_SCALAR_ACCESSOR(INTEGER);
KOSKI_DEFINE_SCALAR_ACCESSOR(BOOLEAN);
KOSKI_DEFINE_SCALAR_ACCESSOR(TINYINT);
KOSKI_DEFINE_SCALAR_ACCESSOR(SMALLINT);
KOSKI_DEFINE_SCALAR_ACCESSOR(BIGINT);
KOSKI_DEFINE_SCALAR_ACCESSOR(REAL);
KOSKI_DEFINE_SCALAR_ACCESSOR(DOUBLE);
KOSKI_DEFINE_SCALAR_ACCESSOR(TIMESTAMP);
KOSKI_DEFINE_SCALAR_ACCESSOR(VARCHAR);
KOSKI_DEFINE_SCALAR_ACCESSOR(VARBINARY);
KOSKI_DEFINE_SCALAR_ACCESSOR(DATE);
KOSKI_DEFINE_SCALAR_ACCESSOR(UNKNOWN);

#undef KOSKI_DEFINE_SCALAR_ACCESSOR

std::shared_ptr<const ShortDecimalType> SHORT_DECIMAL(
    const uint8_t precision,
    const uint8_t scale) {
  return std::make_shared<ShortDecimalType>(precision, scale);
}

std::shared_ptr<const LongDecimalType> LONG_DECIMAL(
    const uint8_t precision,
    const uint8_t scale) {
  return std::make_shared<LongDecimalType>(precision, scale);
}

TypePtr DECIMAL(const uint8_t precision, const uint8_t scale) {
  if (precision <= DecimalType<TypeKind::SHORT_DECIMAL>::kMaxPrecision) {
    return SHORT_DECIMAL(precision, scale);
  }
  return LONG_DECIMAL(precision, scale);
}

TypePtr createScalarType(TypeKind kind) {
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(createScalarType, kind);
}

TypePtr createType(TypeKind kind, std::vector<TypePtr>&& children) {
  if (kind == TypeKind::FUNCTION) {
    VELOX_USER_CHECK_GE(
        children.size(),
        1,
        "FUNCTION type should have at least one child type");
    std::vector<TypePtr> argTypes(
        children.begin(), children.begin() + children.size() - 1);
    return std::make_shared<FunctionType>(std::move(argTypes), children.back());
  }

  if (kind == TypeKind::UNKNOWN) {
    VELOX_USER_CHECK_EQ(
        children.size(), 0, "UNKNOWN type should not have child types");
    return UNKNOWN();
  }
  return VELOX_DYNAMIC_TYPE_DISPATCH(createType, kind, std::move(children));
}

template <>
TypePtr createType<TypeKind::SHORT_DECIMAL>(
    std::vector<TypePtr>&& /*children*/) {
  std::string name{TypeTraits<TypeKind::SHORT_DECIMAL>::name};
  VELOX_USER_FAIL("Not supported for kind: {}", name);
}

template <>
TypePtr createType<TypeKind::LONG_DECIMAL>(
    std::vector<TypePtr>&& /*children*/) {
  std::string name{TypeTraits<TypeKind::LONG_DECIMAL>::name};
  VELOX_USER_FAIL("Not supported for kind: {}", name);
}

template <>
TypePtr createType<TypeKind::ROW>(std::vector<TypePtr>&& /*children*/) {
  std::string name{TypeTraits<TypeKind::ROW>::name};
  VELOX_USER_FAIL("Not supported for kind: {}", name);
}

template <>
TypePtr createType<TypeKind::ARRAY>(std::vector<TypePtr>&& children) {
  VELOX_USER_CHECK_EQ(children.size(), 1, "ARRAY should have only one child");
  return ARRAY(children.at(0));
}

template <>
TypePtr createType<TypeKind::MAP>(std::vector<TypePtr>&& children) {
  VELOX_USER_CHECK_EQ(children.size(), 2, "MAP should have only two children");
  return MAP(children.at(0), children.at(1));
}

template <>
TypePtr createType<TypeKind::OPAQUE>(std::vector<TypePtr>&& /*children*/) {
  std::string name{TypeTraits<TypeKind::OPAQUE>::name};
  VELOX_USER_FAIL("Not supported for kind: {}", name);
}

bool Type::containsUnknown() const {
  if (kind_ == TypeKind::UNKNOWN) {
    return true;
  }
  for (auto i = 0; i < size(); ++i) {
    if (childAt(i)->containsUnknown()) {
      return true;
    }
  }
  return false;
}

namespace {

std::unordered_map<std::string, std::unique_ptr<const CustomTypeFactories>>&
typeFactories() {
  static std::
      unordered_map<std::string, std::unique_ptr<const CustomTypeFactories>>
          factories;
  return factories;
}

} // namespace

bool registerCustomType(
    const std::string& name,
    std::unique_ptr<const CustomTypeFactories> factories) {
  auto uppercaseName = boost::algorithm::to_upper_copy(name);
  return typeFactories().emplace(uppercaseName, std::move(factories)).second;
}

bool customTypeExists(const std::string& name) {
  auto uppercaseName = boost::algorithm::to_upper_copy(name);
  return typeFactories().count(uppercaseName) > 0;
}

std::unordered_set<std::string> getCustomTypeNames() {
  std::unordered_set<std::string> typeNames;
  for (const auto& [name, unused] : typeFactories()) {
    typeNames.insert(name);
  }
  return typeNames;
}

bool unregisterCustomType(const std::string& name) {
  auto uppercaseName = boost::algorithm::to_upper_copy(name);
  return typeFactories().erase(uppercaseName) == 1;
}

const CustomTypeFactories* FOLLY_NULLABLE
getTypeFactories(const std::string& name) {
  auto uppercaseName = boost::algorithm::to_upper_copy(name);
  auto it = typeFactories().find(uppercaseName);

  if (it != typeFactories().end()) {
    return it->second.get();
  }

  return nullptr;
}

TypePtr getCustomType(const std::string& name) {
  auto factories = getTypeFactories(name);
  if (factories) {
    return factories->getType();
  }

  return nullptr;
}

exec::CastOperatorPtr getCustomTypeCastOperator(const std::string& name) {
  auto factories = getTypeFactories(name);
  if (factories) {
    return factories->getCastOperator();
  }

  return nullptr;
}

TypePtr fromKindToScalerType(TypeKind kind) {
  switch (kind) {
    case TypeKind::TINYINT:
      return TINYINT();
    case TypeKind::BOOLEAN:
      return BOOLEAN();
    case TypeKind::SMALLINT:
      return SMALLINT();
    case TypeKind::BIGINT:
      return BIGINT();
    case TypeKind::INTEGER:
      return INTEGER();
    case TypeKind::REAL:
      return REAL();
    case TypeKind::VARCHAR:
      return VARCHAR();
    case TypeKind::VARBINARY:
      return VARBINARY();
    case TypeKind::TIMESTAMP:
      return TIMESTAMP();
    case TypeKind::DOUBLE:
      return DOUBLE();
    case TypeKind::DATE:
      return DATE();
    case TypeKind::UNKNOWN:
      return UNKNOWN();
    default:
      VELOX_UNSUPPORTED(
          "Kind is not a scalar type: {}", mapTypeKindToName(kind));
      return nullptr;
  }
}

void toTypeSql(const TypePtr& type, std::ostream& out) {
  switch (type->kind()) {
    case TypeKind::SHORT_DECIMAL: {
      const auto& decimal = type->asShortDecimal();
      out << "DECIMAL(" << std::to_string(decimal.precision()) << ", "
          << std::to_string(decimal.scale()) << ")";
      break;
    }
    case TypeKind::LONG_DECIMAL: {
      const auto& decimal = type->asLongDecimal();
      out << "DECIMAL(" << std::to_string(decimal.precision()) << ", "
          << std::to_string(decimal.scale()) << ")";
      break;
    }
    case TypeKind::ARRAY:
      // Append <type>[], e.g. bigint[].
      toTypeSql(type->childAt(0), out);
      out << "[]";
      break;
    case TypeKind::MAP:
      // Append map(<key>, <value>), e.g. map(varchar, bigint).
      out << "map(";
      toTypeSql(type->childAt(0), out);
      out << ", ";
      toTypeSql(type->childAt(1), out);
      out << ")";
      break;
    case TypeKind::ROW: {
      // Append struct(name1 type1, name2 type2,..), e.g.
      // struct(a bigint, b real);
      const auto& rowType = type->asRow();
      out << "struct(";
      for (auto i = 0; i < type->size(); ++i) {
        if (i > 0) {
          out << ", ";
        }
        out << rowType.nameOf(i) << " ";
        toTypeSql(type->childAt(i), out);
      }
      out << ")";
      break;
    }
    default:
      if (type->isPrimitiveType()) {
        out << type->toString();
        return;
      }
      VELOX_UNSUPPORTED("Type is not supported: {}", type->toString());
  }
}

std::string IntervalDayTimeType::valueToString(int64_t value) const {
  static const char* kIntervalFormat = "%d %02d:%02d:%02d.%03d";

  int64_t remainMillis = value;
  const int64_t days = remainMillis / kMillisInDay;
  remainMillis -= days * kMillisInDay;
  const int64_t hours = remainMillis / kMillisInHour;
  remainMillis -= hours * kMillisInHour;
  const int64_t minutes = remainMillis / kMillisInMinute;
  remainMillis -= minutes * kMillisInMinute;
  const int64_t seconds = remainMillis / kMillisInSecond;
  remainMillis -= seconds * kMillisInSecond;
  char buf[64];
  snprintf(
      buf,
      sizeof(buf),
      kIntervalFormat,
      days,
      hours,
      minutes,
      seconds,
      remainMillis);

  return buf;
}

namespace {
using SingletonTypeMap = std::unordered_map<std::string, TypePtr>;

const SingletonTypeMap& singletonBuiltInTypes() {
  static const SingletonTypeMap kTypes = {
      {"BOOLEAN", BOOLEAN()},
      {"TINYINT", TINYINT()},
      {"SMALLINT", SMALLINT()},
      {"INTEGER", INTEGER()},
      {"BIGINT", BIGINT()},
      {"REAL", REAL()},
      {"DOUBLE", DOUBLE()},
      {"VARCHAR", VARCHAR()},
      {"VARBINARY", VARBINARY()},
      {"TIMESTAMP", TIMESTAMP()},
      {"DATE", DATE()},
      {"INTERVAL DAY TO SECOND", INTERVAL_DAY_TIME()},
      {"UNKNOWN", UNKNOWN()},
  };
  return kTypes;
};

class DecimalParametricType {
 public:
  static TypePtr create(const std::vector<TypeParameter>& parameters) {
    VELOX_USER_CHECK_EQ(2, parameters.size());
    VELOX_USER_CHECK(parameters[0].kind == TypeParameterKind::kLongLiteral);
    VELOX_USER_CHECK(parameters[0].longLiteral.has_value());
    VELOX_USER_CHECK(parameters[1].kind == TypeParameterKind::kLongLiteral);
    VELOX_USER_CHECK(parameters[1].longLiteral.has_value());

    return DECIMAL(
        parameters[0].longLiteral.value(), parameters[1].longLiteral.value());
  }
};

class ArrayParametricType {
 public:
  static TypePtr create(const std::vector<TypeParameter>& parameters) {
    VELOX_USER_CHECK_EQ(1, parameters.size());
    VELOX_USER_CHECK(parameters[0].kind == TypeParameterKind::kType);
    VELOX_USER_CHECK_NOT_NULL(parameters[0].type);

    return ARRAY(parameters[0].type);
  }
};

class MapParametricType {
 public:
  static TypePtr create(const std::vector<TypeParameter>& parameters) {
    VELOX_USER_CHECK_EQ(2, parameters.size());
    VELOX_USER_CHECK(parameters[0].kind == TypeParameterKind::kType);
    VELOX_USER_CHECK_NOT_NULL(parameters[0].type);

    VELOX_USER_CHECK(parameters[1].kind == TypeParameterKind::kType);
    VELOX_USER_CHECK_NOT_NULL(parameters[1].type);

    return MAP(parameters[0].type, parameters[1].type);
  }
};

class RowParametricType {
 public:
  static TypePtr create(const std::vector<TypeParameter>& parameters) {
    for (const auto& parameter : parameters) {
      VELOX_USER_CHECK(parameter.kind == TypeParameterKind::kType);
      VELOX_USER_CHECK_NOT_NULL(parameter.type);
    }

    std::vector<TypePtr> argumentTypes;
    argumentTypes.reserve(parameters.size());
    for (const auto& parameter : parameters) {
      argumentTypes.push_back(parameter.type);
    }

    return ROW(std::move(argumentTypes));
  }
};

class FunctionParametricType {
 public:
  static TypePtr create(const std::vector<TypeParameter>& parameters) {
    VELOX_USER_CHECK_GE(parameters.size(), 1);
    for (const auto& parameter : parameters) {
      VELOX_USER_CHECK(parameter.kind == TypeParameterKind::kType);
      VELOX_USER_CHECK_NOT_NULL(parameter.type);
    }

    std::vector<TypePtr> argumentTypes;
    argumentTypes.reserve(parameters.size() - 1);
    for (auto i = 0; i < parameters.size() - 1; ++i) {
      argumentTypes.push_back(parameters[i].type);
    }

    return FUNCTION(std::move(argumentTypes), parameters.back().type);
  }
};

using ParametricTypeMap = std::unordered_map<
    std::string,
    std::function<TypePtr(const std::vector<TypeParameter>& parameters)>>;

const ParametricTypeMap& parametricBuiltinTypes() {
  static const ParametricTypeMap kTypes = {
      {"DECIMAL", DecimalParametricType::create},
      {"ARRAY", ArrayParametricType::create},
      {"MAP", MapParametricType::create},
      {"ROW", RowParametricType::create},
      {"FUNCTION", FunctionParametricType::create},
  };
  return kTypes;
}

} // namespace

bool hasType(const std::string& name) {
  if (singletonBuiltInTypes().count(name)) {
    return true;
  }

  if (parametricBuiltinTypes().count(name)) {
    return true;
  }

  if (customTypeExists(name)) {
    return true;
  }

  return false;
}

TypePtr getType(
    const std::string& name,
    const std::vector<TypeParameter>& parameters) {
  if (singletonBuiltInTypes().count(name)) {
    return singletonBuiltInTypes().at(name);
  }

  if (parametricBuiltinTypes().count(name)) {
    return parametricBuiltinTypes().at(name)(parameters);
  }

  return getCustomType(name);
}

} // namespace facebook::velox
