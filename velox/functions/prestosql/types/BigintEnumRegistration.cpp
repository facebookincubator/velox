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
 public:
  explicit BigintEnumCastOperator(
      std::shared_ptr<const BigintEnumType> bigintEnum)
      : bigintEnum_(std::move(bigintEnum)) {}

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
    auto* flatResult = result->as<FlatVector<int64_t>>();
    const auto* intVector = input.as<SimpleVector<int64_t>>();
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const int64_t intToCast = intVector->valueAt(row);
      if (!bigintEnum_->containsValue(intToCast)) {
        context.setStatus(
            row,
            Status::UserError(
                "No value '{}' in {}", intToCast, bigintEnum_->name()));
        return;
      }
      flatResult->set(row, intToCast);
    });
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    auto* flatResult = result->as<FlatVector<int64_t>>();
    const auto* enumInts = input.as<SimpleVector<int64_t>>();
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const int64_t enumIntValue = enumInts->valueAt(row);
      flatResult->set(row, enumIntValue);
    });
  }

 private:
  std::shared_ptr<const BigintEnumType> bigintEnum_;
};

class BigintEnumTypeFactories : public CustomTypeFactories {
 private:
  std::shared_ptr<const BigintEnumType> bigintEnum_;

 public:
  explicit BigintEnumTypeFactories(
      std::shared_ptr<const BigintEnumType> bigintEnum)
      : bigintEnum_(bigintEnum) {}

  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return bigintEnum_;
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

void registerBigintEnumType(const std::string& enumTypeString) {
  try {
    auto [enumName, enumMap] = BigintEnumType::parseTypeInfo(enumTypeString);
    auto enumType = BigintEnumType::create(enumName, enumMap);
    registerCustomType(
        enumName, std::make_unique<BigintEnumTypeFactories>(enumType));
  } catch (std::invalid_argument& e) {
    VELOX_USER_FAIL("Failed to parse type {}, {}", enumTypeString, e.what());
  }
}
} // namespace facebook::velox
