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
} // namespace

void registerJsonType() {
  registerCustomType("JSON", std::make_unique<const JsonTypeFactory>());
  // Register primitive cast rules only. Container types (ARRAY, MAP, ROW)
  // are cross-type casts (e.g. ARRAY -> JSON), which the registry cannot
  // resolve via its same-base-type recursive logic. These are handled at
  // runtime by JsonCastOperator::isSupportedFromType/isSupportedToType.
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
  });
}
} // namespace facebook::velox
