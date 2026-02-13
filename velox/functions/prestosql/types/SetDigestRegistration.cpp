/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/functions/prestosql/types/SetDigestRegistration.h"

#include "velox/common/fuzzer/ConstrainedGenerators.h"
#include "velox/functions/prestosql/types/SetDigestType.h"
#include "velox/type/Type.h"

namespace facebook::velox {
namespace {
class SetDigestTypeFactory : public CustomTypeFactory {
 public:
  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return SETDIGEST();
  }

  // SetDigest should be treated as Varbinary during type castings.
  exec::CastOperatorPtr getCastOperator() const override {
    return nullptr;
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& config) const override {
    return std::make_shared<fuzzer::SetDigestInputGenerator>(
        config.seed_, SETDIGEST(), config.nullRatio_);
  }
};
} // namespace
void registerSetDigestType() {
  registerCustomType(
      "setdigest", std::make_unique<const SetDigestTypeFactory>());
}
} // namespace facebook::velox
