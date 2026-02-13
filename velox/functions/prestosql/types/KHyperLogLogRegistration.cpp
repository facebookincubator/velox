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

#include "velox/functions/prestosql/types/KHyperLogLogRegistration.h"

#include "velox/functions/prestosql/types/KHyperLogLogType.h"
#include "velox/functions/prestosql/types/fuzzer_utils/KHyperLogLogInputGenerator.h"
#include "velox/type/Type.h"

namespace facebook::velox {
namespace {
class KHyperLogLogTypeFactory : public CustomTypeFactory {
 public:
  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return KHYPERLOGLOG();
  }

  // KHyperLogLog should be treated as Varbinary during type castings.
  exec::CastOperatorPtr getCastOperator() const override {
    return nullptr;
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& config) const override {
    return std::static_pointer_cast<AbstractInputGenerator>(
        std::make_shared<fuzzer::KHyperLogLogInputGenerator>(
            config.seed_, config.nullRatio_, config.pool_));
  }
};
} // namespace
void registerKHyperLogLogType() {
  registerCustomType(
      "khyperloglog", std::make_unique<const KHyperLogLogTypeFactory>());
}
} // namespace facebook::velox
