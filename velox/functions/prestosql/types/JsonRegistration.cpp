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

#include "velox/functions/prestosql/types/JsonRegistration.h"

#include "velox/common/fuzzer/ConstrainedGenerators.h"
#include "velox/functions/prestosql/types/JsonCastOperator.h"
#include "velox/functions/prestosql/types/JsonType.h"
#include "velox/type/CastRegistry.h"
#include "velox/type/Type.h"

namespace facebook::velox {
namespace {
class JsonTypeFactory : public CustomTypeFactory {
 public:
  JsonTypeFactory() = default;

  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return JSON();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return std::make_shared<JsonCastOperator>();
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& config) const override {
    static const std::vector<TypePtr> kScalarTypes{
        BOOLEAN(),
        TINYINT(),
        SMALLINT(),
        INTEGER(),
        BIGINT(),
        REAL(),
        DOUBLE(),
        VARCHAR(),
    };
    fuzzer::FuzzerGenerator rng(config.seed_);
    return std::make_shared<fuzzer::JsonInputGenerator>(
        config.seed_,
        JSON(),
        config.nullRatio_,
        fuzzer::getRandomInputGenerator(
            config.seed_,
            fuzzer::randType(rng, kScalarTypes, 3),
            config.nullRatio_),
        false);
  }
};

// Returns true if 'type' is a primitive scalar type usable as a MAP key in
// JSON casts. JSON itself has VARCHAR kind so it passes here; callers that
// need to reject JSON keys (e.g. JSON → MAP) check isJsonType separately.
bool isValidJsonMapKey(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::BIGINT:
    case TypeKind::INTEGER:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT:
    case TypeKind::DOUBLE:
    case TypeKind::REAL:
    case TypeKind::VARCHAR:
      return true;
    default:
      return false;
  }
}

// Validator for DECIMAL → JSON: only short decimals (precision ≤ 18) are
// supported. Short decimals use TypeKind::BIGINT; long decimals use HUGEINT.
bool canCastDecimalToJson(const TypePtr& from, const TypePtr& /*to*/) {
  return from->kind() == TypeKind::BIGINT;
}

// Validator for ARRAY → JSON: element type must be castable to JSON.
bool canCastArrayToJson(const TypePtr& from, const TypePtr& to) {
  return CastRulesRegistry::instance().canCast(from->childAt(0), to);
}

// Validator for ROW → JSON: all child types must be castable to JSON.
bool canCastRowToJson(const TypePtr& from, const TypePtr& to) {
  for (auto i = 0; i < from->size(); ++i) {
    if (!CastRulesRegistry::instance().canCast(from->childAt(i), to)) {
      return false;
    }
  }
  return true;
}

// Validator for MAP → JSON: key must be a valid JSON map key, value must be
// castable to JSON. Special case: MAP<UNKNOWN, UNKNOWN> is allowed.
// Note: JSON keys are allowed here (JSON has VARCHAR kind and passes
// isValidJsonMapKey). The old isSupportedFromType had an inconsistency where
// JSON keys were rejected only in the UNKNOWN-value branch but allowed
// otherwise; we use the more permissive behavior consistently.
bool canCastMapToJson(const TypePtr& from, const TypePtr& to) {
  const auto& keyType = from->childAt(0);
  const auto& valueType = from->childAt(1);
  if (keyType->isUnknown() && valueType->isUnknown()) {
    return true;
  }
  if (!isValidJsonMapKey(keyType)) {
    return false;
  }
  return CastRulesRegistry::instance().canCast(valueType, to);
}

// Validator for JSON → ARRAY: element type must be castable from JSON.
bool canCastJsonToArray(const TypePtr& from, const TypePtr& to) {
  return CastRulesRegistry::instance().canCast(from, to->childAt(0));
}

// Validator for JSON → ROW: all child types must be castable from JSON.
bool canCastJsonToRow(const TypePtr& from, const TypePtr& to) {
  for (auto i = 0; i < to->size(); ++i) {
    if (!CastRulesRegistry::instance().canCast(from, to->childAt(i))) {
      return false;
    }
  }
  return true;
}

// Validator for JSON → MAP: key must be a valid JSON map key and not JSON
// itself, value must be castable from JSON.
bool canCastJsonToMap(const TypePtr& from, const TypePtr& to) {
  const auto& keyType = to->childAt(0);
  const auto& valueType = to->childAt(1);
  if (!isValidJsonMapKey(keyType) || isJsonType(keyType)) {
    return false;
  }
  return CastRulesRegistry::instance().canCast(from, valueType);
}

} // namespace

void registerJsonType() {
  registerCustomType("JSON", std::make_unique<const JsonTypeFactory>());
  registerCastRules({
      // TO JSON (from primitive types).
      {.fromType = "UNKNOWN", .toType = "JSON"},
      {.fromType = "BOOLEAN", .toType = "JSON"},
      {.fromType = "TINYINT", .toType = "JSON"},
      {.fromType = "SMALLINT", .toType = "JSON"},
      {.fromType = "INTEGER", .toType = "JSON"},
      {.fromType = "BIGINT", .toType = "JSON"},
      {.fromType = "REAL", .toType = "JSON"},
      {.fromType = "DOUBLE", .toType = "JSON"},
      {.fromType = "VARCHAR", .toType = "JSON"},
      {.fromType = "TIMESTAMP", .toType = "JSON"},
      {.fromType = "DATE",
       .toType = "JSON",
       .implicitAllowed = false,
       .validator = {}},
      {.fromType = "DECIMAL",
       .toType = "JSON",
       .implicitAllowed = false,
       .validator = canCastDecimalToJson},
      // TO JSON (from container types with recursive validation).
      {.fromType = "ARRAY",
       .toType = "JSON",
       .implicitAllowed = false,
       .validator = canCastArrayToJson},
      {.fromType = "ROW",
       .toType = "JSON",
       .implicitAllowed = false,
       .validator = canCastRowToJson},
      {.fromType = "MAP",
       .toType = "JSON",
       .implicitAllowed = false,
       .validator = canCastMapToJson},
      // FROM JSON (to primitive types).
      // Note: JSON -> TIMESTAMP is not supported in Presto.
      {.fromType = "JSON", .toType = "BOOLEAN"},
      {.fromType = "JSON", .toType = "TINYINT"},
      {.fromType = "JSON", .toType = "SMALLINT"},
      {.fromType = "JSON", .toType = "INTEGER"},
      {.fromType = "JSON", .toType = "BIGINT"},
      {.fromType = "JSON", .toType = "REAL"},
      {.fromType = "JSON", .toType = "DOUBLE"},
      {.fromType = "JSON", .toType = "VARCHAR"},
      // FROM JSON (to container types with recursive validation).
      {.fromType = "JSON",
       .toType = "ARRAY",
       .implicitAllowed = false,
       .validator = canCastJsonToArray},
      {.fromType = "JSON",
       .toType = "ROW",
       .implicitAllowed = false,
       .validator = canCastJsonToRow},
      {.fromType = "JSON",
       .toType = "MAP",
       .implicitAllowed = false,
       .validator = canCastJsonToMap},
  });
}
} // namespace facebook::velox
