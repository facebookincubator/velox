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

#include "velox/expression/CastExpr.h"
#include "velox/functions/prestosql/types/BigintEnumType.h"

namespace facebook::velox {
namespace {
class BigintEnumCastOperator : public exec::CastOperator {
  std::shared_ptr<BigintEnumType> bigintEnum_;

 public:
  explicit BigintEnumCastOperator(std::shared_ptr<BigintEnumType> bigintEnum)
      : bigintEnum_(bigintEnum) {}

  bool isSupportedFromType(const TypePtr& other) const override {
    return other->kind() == TypeKind::BIGINT &&
        (strcmp(other->name(), "BIGINT") == 0 ||
         other->name() == bigintEnum_->name());
  }

  bool isSupportedToType(const TypePtr& other) const override {
    return other->kind() == TypeKind::BIGINT &&
        (strcmp(other->name(), "BIGINT") == 0 ||
         other->name() == bigintEnum_->name());
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);
    castFromInt(input, context, rows, *result);
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);
    castToInt(input, context, rows, *result);
  }

 private:
  std::string typeInfoString;
  static void castToInt(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<int64_t>>();
    const auto* enumInts = input.as<SimpleVector<int64_t>>();
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const int64_t enumIntValue = enumInts->valueAt(row);
      flatResult->set(row, enumIntValue);
    });
  }

  void castFromInt(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) const {
    auto* flatResult = result.as<FlatVector<int64_t>>();
    const auto* intVector = input.as<SimpleVector<int64_t>>();
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const int64_t intToCast = intVector->valueAt(row);
      if (!enumContainsValue(intToCast)) {
        context.setStatus(
            row,
            Status::UserError(
                "No value '{}' in {}",
                intToCast,
                bigintEnum_->getTypeInfoString()));
        return;
      }
      flatResult->set(row, intToCast);
    });
  }

  bool enumContainsValue(int64_t value) const {
    auto enumMap = bigintEnum_->getEnumMap();
    return std::any_of(
        enumMap.begin(), enumMap.end(), [value](const auto& pair) {
          return pair.second == value;
        });
  }
};

class BigintEnumTypeFactories : public CustomTypeFactories {
 private:
  std::shared_ptr<BigintEnumType> bigintEnum_;

 public:
  explicit BigintEnumTypeFactories(BigintEnumType* bigintEnum)
      : bigintEnum_(bigintEnum) {}

  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return BigintEnumType::get(bigintEnum_->getTypeInfoString());
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return std::make_shared<BigintEnumCastOperator>(bigintEnum_);
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& /*config*/) const override {
    return nullptr;
  }
};
} // namespace

void registerBigintEnumType(const std::string& typeInfoString) {
  auto enumType = new BigintEnumType(typeInfoString);
  registerCustomType(
      enumType->name(),
      std::make_unique<const BigintEnumTypeFactories>(enumType));
}
} // namespace facebook::velox
