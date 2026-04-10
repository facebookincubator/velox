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

#include "velox/functions/prestosql/types/UuidRegistration.h"

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "velox/expression/CastExpr.h"
#include "velox/functions/prestosql/types/UuidType.h"
#include "velox/type/CastRegistry.h"

namespace facebook::velox {
namespace {
template <TypeKind>
struct UuidParser {
  static int128_t parse(const StringView& uuidValue) = delete;
};

template <>
struct UuidParser<TypeKind::VARCHAR> {
  static int128_t parse(const StringView& uuidValue) {
    auto uuid = boost::lexical_cast<boost::uuids::uuid>(uuidValue);
    int128_t u;
    memcpy(&u, &uuid, 16);
    return u;
  }
};

template <>
struct UuidParser<TypeKind::VARBINARY> {
  static int128_t parse(const StringView& uuidValue) {
    int128_t u;
    memcpy(&u, uuidValue.data(), 16);
    return u;
  }
};

class UuidCastOperator : public exec::CastOperator {
 public:
  bool isSupportedFromType(const TypePtr& other) const override {
    return VARCHAR()->equivalent(*other) || VARBINARY()->equivalent(*other);
  }

  bool isSupportedToType(const TypePtr& other) const override {
    return VARCHAR()->equivalent(*other) || VARBINARY()->equivalent(*other);
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (input.typeKind() == TypeKind::VARCHAR) {
      castFromTypeKind<TypeKind::VARCHAR>(input, context, rows, *result);
    } else if (input.typeKind() == TypeKind::VARBINARY) {
      castFromTypeKind<TypeKind::VARBINARY>(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from {} to UUID not yet supported", input.type()->toString());
    }
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (resultType->kind() == TypeKind::VARCHAR) {
      castToTypeKind<TypeKind::VARCHAR>(input, context, rows, *result);
    } else if (resultType->kind() == TypeKind::VARBINARY) {
      castToTypeKind<TypeKind::VARBINARY>(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from UUID to {} not yet supported", resultType->toString());
    }
  }

 private:
  template <TypeKind KIND>
  static void castToTypeKind(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* uuids = input.as<SimpleVector<int128_t>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto value = uuids->valueAt(row);
      exec::StringWriter writer(flatResult, row);
      if constexpr (KIND == TypeKind::VARCHAR) {
        writer.resize(UuidType::kStringSize);
        UUID()->valueToString(value, writer.data());
      } else {
        auto bigEndian = DecimalUtil::bigEndian(value);
        writer.resize(16);
        memcpy(writer.data(), &bigEndian, 16);
      }
      writer.finalize();
    });
  }

  template <TypeKind KIND>
  static void castFromTypeKind(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<int128_t>>();
    const auto* uuidInput = input.as<SimpleVector<StringView>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto uuidValue = uuidInput->valueAt(row);
      auto u = UuidParser<KIND>::parse(uuidValue);
      // Convert a big endian value to native byte-order.
      u = DecimalUtil::bigEndian(u);
      flatResult->set(row, u);
    });
  }
};

class UuidTypeFactory : public CustomTypeFactory {
 public:
  UuidTypeFactory() = default;

  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return UUID();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return std::make_shared<UuidCastOperator>();
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& /*config*/) const override {
    return nullptr;
  }
};
} // namespace

void registerUuidType() {
  registerCustomType("UUID", std::make_unique<const UuidTypeFactory>());
  registerCastRules({
      {.fromType = "VARCHAR", .toType = "UUID"},
      {.fromType = "VARBINARY", .toType = "UUID"},
      {.fromType = "UUID", .toType = "VARCHAR"},
      {.fromType = "UUID", .toType = "VARBINARY"},
  });
}
} // namespace facebook::velox
