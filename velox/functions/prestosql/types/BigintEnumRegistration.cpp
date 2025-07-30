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

#include "velox/functions/prestosql/types/BigintEnumRegistration.h"
#include "velox/expression/CastExpr.h"
#include "velox/functions/prestosql/types/BigintEnumType.h"

namespace facebook::velox {
namespace {
class BigintEnumCastOperator : public exec::CastOperator {
 public:
  static const std::shared_ptr<const CastOperator>& get() {
    static const std::shared_ptr<const CastOperator> instance{
        new BigintEnumCastOperator()};

    return instance;
  }

  bool isSupportedFromType(const TypePtr& other) const override {
    return isCompatibleWith(other);
  }

  bool isSupportedToType(const TypePtr& other) const override {
    return isCompatibleWith(other);
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    switch (input.typeKind()) {
      case TypeKind::TINYINT:
        castFromType<int8_t>(input, context, rows, resultType, result);
        break;
      case TypeKind::SMALLINT:
        castFromType<int16_t>(input, context, rows, resultType, result);
        break;
      case TypeKind::INTEGER:
        castFromType<int32_t>(input, context, rows, resultType, result);
        break;
      case TypeKind::BIGINT:
        castFromType<int64_t>(input, context, rows, resultType, result);
        break;
      default:
        VELOX_UNREACHABLE("Unsupported type: {}", input.type()->toString());
    }
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    switch (resultType->kind()) {
      case TypeKind::TINYINT:
        castToType<int8_t>(input, context, rows, resultType, result);
        break;
      case TypeKind::SMALLINT:
        castToType<int16_t>(input, context, rows, resultType, result);
        break;
      case TypeKind::INTEGER:
        castToType<int32_t>(input, context, rows, resultType, result);
        break;
      case TypeKind::BIGINT:
        castToType<int64_t>(input, context, rows, resultType, result);
        break;
      default:
        VELOX_UNREACHABLE("Unsupported type: {}", input.type()->toString());
    }
  }

 private:
  // Cast is supported for all integer types.
  // Casting to and from a different BigintEnumType is not supported
  bool isCompatibleWith(const TypePtr& other) const {
    return BIGINT()->equivalent(*other) || TINYINT()->equivalent(*other) ||
        SMALLINT()->equivalent(*other) || INTEGER()->equivalent(*other);
  }

  template <typename T>
  void castToType(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const {
    context.ensureWritable(rows, resultType, result);
    auto* flatResult = result->asChecked<FlatVector<T>>();
    flatResult->clearNulls(rows);

    const auto* enumInts = input.as<SimpleVector<int64_t>>();
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      int64_t value = enumInts->valueAt(row);
      if (value < std::numeric_limits<T>::min() ||
          value > std::numeric_limits<T>::max()) {
        context.setStatus(
            row,
            Status::UserError(
                "Value '{}' out of range for {}",
                value,
                resultType->toString()));
        return;
      }
      flatResult->set(row, static_cast<T>(value));
    });
  }

  template <typename T>
  void castFromType(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const {
    context.ensureWritable(rows, resultType, result);
    auto* flatResult = result->asChecked<FlatVector<int64_t>>();
    flatResult->clearNulls(rows);

    VELOX_CHECK(
        isBigintEnumType(resultType),
        "Invalid CAST TO type for BigintEnumType");
    auto toType = std::dynamic_pointer_cast<const BigintEnumType>(resultType);
    const auto* intVector = input.as<SimpleVector<T>>();
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const int64_t intToCast = intVector->valueAt(row);
      if (!toType->containsValue(intToCast)) {
        context.setStatus(
            row,
            Status::UserError(
                "No value '{}' in {}", intToCast, toType->enumName()));
        return;
      }
      flatResult->set(row, intToCast);
    });
  }
};

class BigintEnumTypeFactory : public CustomTypeFactory {
 public:
  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK_EQ(parameters.size(), 2);
    return BIGINT_ENUM(parameters);
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return BigintEnumCastOperator::get();
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& /*config*/) const override {
    return nullptr;
  }
};
} // namespace

void registerBigintEnumType() {
  registerCustomType(
      "BIGINT_ENUM", std::make_unique<const BigintEnumTypeFactory>());
}
} // namespace facebook::velox
