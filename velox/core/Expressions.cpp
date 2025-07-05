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
#include "velox/core/Expressions.h"
#include "velox/common/encode/Base64.h"
#include "velox/vector/SimpleVector.h"
#include "velox/vector/VectorSaver.h"

namespace facebook::velox::core {

namespace {
TypePtr deserializeType(const folly::dynamic& obj, void* context) {
  return ISerializable::deserialize<Type>(obj["type"]);
}

std::vector<TypedExprPtr> deserializeInputs(
    const folly::dynamic& obj,
    void* context) {
  if (obj.count("inputs")) {
    return ISerializable::deserialize<std::vector<ITypedExpr>>(
        obj["inputs"], context);
  }

  return {};
}
} // namespace

folly::dynamic ITypedExpr::serializeBase(std::string_view name) const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = name;
  obj["type"] = type_->serialize();

  if (!inputs_.empty()) {
    folly::dynamic serializedInputs = folly::dynamic::array;
    for (const auto& input : inputs_) {
      serializedInputs.push_back(input->serialize());
    }

    obj["inputs"] = serializedInputs;
  }

  return obj;
}

// static
void ITypedExpr::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();

  registry.Register("CallTypedExpr", core::CallTypedExpr::create);
  registry.Register("CastTypedExpr", core::CastTypedExpr::create);
  registry.Register("ConcatTypedExpr", core::ConcatTypedExpr::create);
  registry.Register("ConstantTypedExpr", core::ConstantTypedExpr::create);
  registry.Register("DereferenceTypedExpr", core::DereferenceTypedExpr::create);
  registry.Register("FieldAccessTypedExpr", core::FieldAccessTypedExpr::create);
  registry.Register("InputTypedExpr", core::InputTypedExpr::create);
  registry.Register("LambdaTypedExpr", core::LambdaTypedExpr::create);
}

void InputTypedExpr::accept(
    const ITypedExprVisitor& visitor,
    ITypedExprVisitorContext& context) const {
  visitor.visit(*this, context);
}

folly::dynamic InputTypedExpr::serialize() const {
  return ITypedExpr::serializeBase("InputTypedExpr");
}

// static
TypedExprPtr InputTypedExpr::create(const folly::dynamic& obj, void* context) {
  auto type = core::deserializeType(obj, context);

  return std::make_shared<InputTypedExpr>(std::move(type));
}

void ConstantTypedExpr::accept(
    const ITypedExprVisitor& visitor,
    ITypedExprVisitorContext& context) const {
  visitor.visit(*this, context);
}

folly::dynamic ConstantTypedExpr::serialize() const {
  auto obj = ITypedExpr::serializeBase("ConstantTypedExpr");
  if (valueVector_) {
    std::ostringstream out;
    saveVector(*valueVector_, out);
    auto serializedValue = out.str();
    obj["valueVector"] = encoding::Base64::encode(
        serializedValue.data(), serializedValue.size());
  } else {
    obj["value"] = value_.serialize();
  }

  return obj;
}

// static
TypedExprPtr ConstantTypedExpr::create(
    const folly::dynamic& obj,
    void* context) {
  auto type = core::deserializeType(obj, context);

  if (obj.count("value")) {
    auto value = Variant::create(obj["value"]);
    return std::make_shared<ConstantTypedExpr>(std::move(type), value);
  }

  auto encodedData = obj["valueVector"].asString();
  auto serializedData = encoding::Base64::decode(encodedData);
  std::istringstream dataStream(serializedData);

  auto* pool = static_cast<memory::MemoryPool*>(context);

  return std::make_shared<ConstantTypedExpr>(restoreVector(dataStream, pool));
}

namespace {
template <TypeKind Kind>
std::string toStringImpl(const TypePtr& type, const Variant& value) {
  using T = typename TypeTraits<Kind>::NativeType;

  return SimpleVector<T>::valueToString(type, T(value.value<T>()));
}

} // namespace

std::string ConstantTypedExpr::toString() const {
  if (hasValueVector()) {
    return valueVector_->toString(0);
  }

  if (value_.isNull()) {
    return std::string(BaseVector::kNullValueString);
  }

  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      toStringImpl, type()->kind(), type(), value_);
}

namespace {

template <TypeKind KIND>
bool equalsNoNulls(const VectorPtr& vector, const Variant& value) {
  using T = typename TypeTraits<KIND>::NativeType;

  const auto thisValue = vector->as<SimpleVector<T>>()->valueAt(0);
  const auto otherValue = T(value.value<T>());

  const auto& type = vector->type();

  auto result = type->providesCustomComparison()
      ? SimpleVector<T>::comparePrimitiveAscWithCustomComparison(
            type.get(), thisValue, otherValue)
      : SimpleVector<T>::comparePrimitiveAsc(thisValue, otherValue);
  return result == 0;
}

bool equalsImpl(const VectorPtr& vector, const Variant& value) {
  static constexpr CompareFlags kEqualValueAtFlags =
      CompareFlags::equality(CompareFlags::NullHandlingMode::kNullAsValue);

  bool thisNull = vector->isNullAt(0);
  bool otherNull = value.isNull();

  if (otherNull || thisNull) {
    return BaseVector::compareNulls(thisNull, otherNull, kEqualValueAtFlags)
        .value();
  }

  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      equalsNoNulls, vector->typeKind(), vector, value);
}
} // namespace

bool ConstantTypedExpr::equals(const ITypedExpr& other) const {
  const auto* casted = dynamic_cast<const ConstantTypedExpr*>(&other);
  if (!casted) {
    return false;
  }

  if (*this->type() != *casted->type()) {
    return false;
  }

  if (this->hasValueVector() != casted->hasValueVector()) {
    if (this->hasValueVector()) {
      return equalsImpl(this->valueVector_, casted->value_);
    } else {
      return equalsImpl(casted->valueVector_, this->value_);
    }
  }

  if (this->hasValueVector()) {
    return this->valueVector_->equalValueAt(casted->valueVector_.get(), 0, 0);
  }

  return this->value_ == casted->value_;
}

namespace {
template <TypeKind KIND>
uint64_t hashImpl(const TypePtr& type, const Variant& value) {
  using T = typename TypeTraits<KIND>::NativeType;

  const auto& v = value.value<KIND>();

  if (type->providesCustomComparison()) {
    return SimpleVector<T>::hashValueAtWithCustomType(type, T(v));
  }

  if constexpr (std::is_floating_point_v<T>) {
    return util::floating_point::NaNAwareHash<T>{}(T(v));
  } else {
    return folly::hasher<T>{}(T(v));
  }
}
} // namespace

size_t ConstantTypedExpr::localHash() const {
  static const size_t kBaseHash = std::hash<const char*>()("ConstantTypedExpr");

  uint64_t hash;

  if (hasValueVector()) {
    hash = valueVector_->hashValueAt(0);
  } else if (value_.isNull()) {
    hash = BaseVector::kNullHash;
  } else {
    hash = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        hashImpl, type()->kind(), type(), value_);
  }

  return bits::hashMix(kBaseHash, hash);
}

void CallTypedExpr::accept(
    const ITypedExprVisitor& visitor,
    ITypedExprVisitorContext& context) const {
  visitor.visit(*this, context);
}

folly::dynamic CallTypedExpr::serialize() const {
  auto obj = ITypedExpr::serializeBase("CallTypedExpr");
  obj["functionName"] = name_;
  return obj;
}

// static
TypedExprPtr CallTypedExpr::create(const folly::dynamic& obj, void* context) {
  auto type = core::deserializeType(obj, context);
  auto inputs = deserializeInputs(obj, context);

  return std::make_shared<CallTypedExpr>(
      std::move(type), std::move(inputs), obj["functionName"].asString());
}

void FieldAccessTypedExpr::accept(
    const ITypedExprVisitor& visitor,
    ITypedExprVisitorContext& context) const {
  visitor.visit(*this, context);
}

folly::dynamic FieldAccessTypedExpr::serialize() const {
  auto obj = ITypedExpr::serializeBase("FieldAccessTypedExpr");
  obj["fieldName"] = name_;
  return obj;
}

// static
TypedExprPtr FieldAccessTypedExpr::create(
    const folly::dynamic& obj,
    void* context) {
  auto type = core::deserializeType(obj, context);
  auto inputs = deserializeInputs(obj, context);
  VELOX_CHECK_LE(inputs.size(), 1);

  auto name = obj["fieldName"].asString();

  if (inputs.empty()) {
    return std::make_shared<FieldAccessTypedExpr>(std::move(type), name);
  } else {
    return std::make_shared<FieldAccessTypedExpr>(
        std::move(type), std::move(inputs[0]), name);
  }
}

void DereferenceTypedExpr::accept(
    const ITypedExprVisitor& visitor,
    ITypedExprVisitorContext& context) const {
  visitor.visit(*this, context);
}

folly::dynamic DereferenceTypedExpr::serialize() const {
  auto obj = ITypedExpr::serializeBase("DereferenceTypedExpr");
  obj["fieldIndex"] = index_;
  return obj;
}

// static
TypedExprPtr DereferenceTypedExpr::create(
    const folly::dynamic& obj,
    void* context) {
  auto type = core::deserializeType(obj, context);
  auto inputs = deserializeInputs(obj, context);
  VELOX_CHECK_EQ(inputs.size(), 1);

  uint32_t index = obj["fieldIndex"].asInt();

  return std::make_shared<DereferenceTypedExpr>(
      std::move(type), std::move(inputs[0]), index);
}

void ConcatTypedExpr::accept(
    const ITypedExprVisitor& visitor,
    ITypedExprVisitorContext& context) const {
  visitor.visit(*this, context);
}

folly::dynamic ConcatTypedExpr::serialize() const {
  return ITypedExpr::serializeBase("ConcatTypedExpr");
}

// static
TypedExprPtr ConcatTypedExpr::create(const folly::dynamic& obj, void* context) {
  auto type = core::deserializeType(obj, context);
  auto inputs = deserializeInputs(obj, context);

  return std::make_shared<ConcatTypedExpr>(
      type->asRow().names(), std::move(inputs));
}

void LambdaTypedExpr::accept(
    const ITypedExprVisitor& visitor,
    ITypedExprVisitorContext& context) const {
  visitor.visit(*this, context);
}

folly::dynamic LambdaTypedExpr::serialize() const {
  auto obj = ITypedExpr::serializeBase("LambdaTypedExpr");
  obj["signature"] = signature_->serialize();
  obj["body"] = body_->serialize();
  return obj;
}

// static
TypedExprPtr LambdaTypedExpr::create(const folly::dynamic& obj, void* context) {
  auto signature = ISerializable::deserialize<Type>(obj["signature"]);
  auto body = ISerializable::deserialize<ITypedExpr>(obj["body"], context);

  return std::make_shared<LambdaTypedExpr>(
      asRowType(signature), std::move(body));
}

void CastTypedExpr::accept(
    const ITypedExprVisitor& visitor,
    ITypedExprVisitorContext& context) const {
  visitor.visit(*this, context);
}

folly::dynamic CastTypedExpr::serialize() const {
  auto obj = ITypedExpr::serializeBase("CastTypedExpr");
  obj["isTryCast"] = isTryCast_;
  return obj;
}

// static
TypedExprPtr CastTypedExpr::create(const folly::dynamic& obj, void* context) {
  auto type = core::deserializeType(obj, context);
  auto inputs = deserializeInputs(obj, context);

  return std::make_shared<CastTypedExpr>(
      std::move(type), std::move(inputs), obj["isTryCast"].asBool());
}

} // namespace facebook::velox::core
