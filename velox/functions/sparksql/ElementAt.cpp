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
#include "velox/functions/sparksql/ElementAt.h"

#include "velox/functions/lib/SubscriptUtil.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"

namespace facebook::velox::functions::sparksql {
namespace {

// 'allowOutOfBound' only affects arrays. A missing map key returns NULL either
// way, as in Spark.
template <bool allowOutOfBound>
class ElementAtFunction : public SubscriptImpl<
                              /*allowNegativeIndices=*/true,
                              /*nullOnNegativeIndices=*/false,
                              allowOutOfBound,
                              /*indexStartsAtOne=*/true> {
 public:
  explicit ElementAtFunction(bool allowCaching)
      : SubscriptImpl<true, false, allowOutOfBound, true>(allowCaching) {}
};

} // namespace

void registerElementAtFunction(const std::string& name) {
  exec::registerStatefulVectorFunction(
      name,
      ElementAtFunction<true>::signatures(),
      [](const std::string& /*name*/,
         const std::vector<exec::VectorFunctionArg>& inputArgs,
         const core::QueryConfig& config)
          -> std::shared_ptr<exec::VectorFunction> {
        const bool ansiEnabled = SparkQueryConfig{config}.ansiEnabled();
        if (inputArgs[0].type->isArray()) {
          static const auto kNullOnOutOfBound =
              std::make_shared<ElementAtFunction<true>>(false);
          static const auto kThrowOnOutOfBound =
              std::make_shared<ElementAtFunction<false>>(false);
          if (ansiEnabled) {
            return kThrowOnOutOfBound;
          }
          return kNullOnOutOfBound;
        }
        return std::make_shared<ElementAtFunction<true>>(
            config.isExpressionEvaluationCacheEnabled());
      });
}

} // namespace facebook::velox::functions::sparksql
