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

#include "velox/functions/sparksql/types/TimestampNTZRegistration.h"
#include "velox/common/fuzzer/ConstrainedGenerators.h"
#include "velox/expression/CastExpr.h"
#include "velox/functions/sparksql/types/TimestampNTZCastUtil.h"
#include "velox/functions/sparksql/types/TimestampNTZType.h"

namespace facebook::velox::functions::sparksql {
namespace {

class TimestampNTZCastOperator final : public exec::CastOperator {
  TimestampNTZCastOperator() = default;

 public:
  static std::shared_ptr<const CastOperator> get() {
    VELOX_CONSTEXPR_SINGLETON TimestampNTZCastOperator kInstance;
    return {std::shared_ptr<const CastOperator>{}, &kInstance};
  }

  // Returns true if casting from other type to TIMESTAMP_NTZ type is supported.
  bool isSupportedFromType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::VARCHAR:
        return true;
      default:
        return false;
    }
  }

  // Return true if casting from TIMESTAMP_NTZ type to other type is supported.
  bool isSupportedToType(const TypePtr& other) const override {
    return false;
  }

  // Casts the input vector to the TIMESTAMP_NTZ type.
  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);
    auto* timestampNTZResult = result->asFlatVector<int64_t>();
    timestampNTZResult->clearNulls(rows);

    auto* rawResults = timestampNTZResult->mutableRawValues();
    if (input.typeKind() == TypeKind::VARCHAR) {
      const auto inputVector = input.as<SimpleVector<StringView>>();
      castFromString(*inputVector, context, rows, rawResults);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from {} to TIMESTAMP WITHOUT TIME ZONE not yet supported",
          resultType->toString());
    }
  }

  // Casts the input vector of the TIMESTAMP_NTZ type to result type.
  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    VELOX_UNSUPPORTED(
        "Cast from TIMESTAMP WITHOUT TIME ZONE to {} not yet supported",
        resultType->toString());
  }
};

class TimestampNTZTypeFactory : public CustomTypeFactory {
 public:
  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return TIMESTAMP_NTZ();
  }

  // Type casting from and to TimestampNTZ is not supported yet.
  exec::CastOperatorPtr getCastOperator() const override {
    return TimestampNTZCastOperator::get();
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& config) const override {
    return std::make_shared<fuzzer::RandomInputGenerator<int64_t>>(
        config.seed_, TIMESTAMP_NTZ(), config.nullRatio_);
  }
};

} // namespace

void registerTimestampNTZType() {
  registerCustomType(
      "timestamp_ntz", std::make_unique<const TimestampNTZTypeFactory>());
}

} // namespace facebook::velox::functions::sparksql
