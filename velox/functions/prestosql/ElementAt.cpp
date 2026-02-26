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

#include "velox/functions/lib/SubscriptUtil.h"

namespace facebook::velox::functions {

namespace {

/// element_at(array, index) -> array[index]
/// element_at(map, key) -> array[map_values]
///
/// - allows negative indices for arrays (if index < 0, accesses elements from
///    the last to the first).
/// - actually return elements from last to first on negative indices.
/// - allows out of bounds accesses for arrays and maps (returns NULL if out of
///    bounds).
/// - index starts at 1 for arrays.
class ElementAtFunction : public SubscriptImpl<
                              /* allowNegativeIndices */ true,
                              /* nullOnNegativeIndices */ false,
                              /* allowOutOfBound */ true,
                              /* indexStartsAtOne */ true> {
 public:
  explicit ElementAtFunction(bool allowcaching) : SubscriptImpl(allowcaching) {}
};
/// checked_element_at for Spark ANSI mode:
/// - throws on out-of-bounds array access (instead of returning NULL).
/// - throws on missing map key (instead of returning NULL).
class CheckedElementAtFunction : public SubscriptImpl<
                                     /* allowNegativeIndices */ true,
                                     /* nullOnNegativeIndices */ false,
                                     /* allowOutOfBound */ false,
                                     /* indexStartsAtOne */ true,
                                     /* throwOnMissingMapKey */ true> {
 public:
  explicit CheckedElementAtFunction(bool allowcaching)
      : SubscriptImpl(allowcaching) {}
};
} // namespace

void registerElementAtFunction(
    const std::string& name,
    bool enableCaching = true) {
  exec::registerStatefulVectorFunction(
      name,
      ElementAtFunction::signatures(),
      [enableCaching](
          const std::string&,
          const std::vector<exec::VectorFunctionArg>& inputArgs,
          const velox::core::QueryConfig& config) {
        static const auto kSubscriptStateLess =
            std::make_shared<ElementAtFunction>(false);
        if (inputArgs[0].type->isArray()) {
          return kSubscriptStateLess;
        } else {
          return std::make_shared<ElementAtFunction>(
              enableCaching && config.isExpressionEvaluationCacheEnabled());
        }
      });
}

void registerCheckedElementAtFunction(
    const std::string& name,
    bool enableCaching = true) {
  exec::registerStatefulVectorFunction(
      name,
      CheckedElementAtFunction::signatures(),
      [enableCaching](
          const std::string&,
          const std::vector<exec::VectorFunctionArg>& inputArgs,
          const velox::core::QueryConfig& config) {
        static const auto kSubscriptStateLess =
            std::make_shared<CheckedElementAtFunction>(false);
        if (inputArgs[0].type->isArray()) {
          return kSubscriptStateLess;
        } else {
          return std::make_shared<CheckedElementAtFunction>(
              enableCaching && config.isExpressionEvaluationCacheEnabled());
        }
      });
}

} // namespace facebook::velox::functions
