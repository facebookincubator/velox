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

#include "velox/functions/prestosql/types/BingTileRegistration.h"

#include "velox/functions/prestosql/types/BingTileType.h"

namespace facebook::velox {

namespace {

class BingTileTypeFactories : public CustomTypeFactories {
 public:
  velox::TypePtr getType(
      const std::vector<velox::TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return BINGTILE();
  }

  // TODO: Provide casting to/from BIGINT
  exec::CastOperatorPtr getCastOperator() const override {
    return nullptr;
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& /*config*/) const override {
    return nullptr;
  }
};

} // namespace

void registerBingTileType() {
  registerCustomType(
      "bingtile", std::make_unique<const BingTileTypeFactories>());
}

} // namespace facebook::velox
