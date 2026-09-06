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
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "velox/exec/Aggregate.h"
#include "velox/type/Type.h"

namespace facebook::velox::aggregate::prestosql {

/// An additional mergeable sketch type a caller injects into merge() (e.g. a
/// proprietary type). The signature's argument and return types must both
/// match 'type'.
struct MergeSketchType {
  /// Signature for merge(<type>) -> <type>. The intermediate type is not
  /// validated; the caller must ensure it matches the type's serialized form.
  std::shared_ptr<exec::AggregateFunctionSignature> signature;

  /// Concrete type matched against argTypes[0] at factory dispatch.
  TypePtr type;

  /// Builds the aggregate for this type given the resolved result type.
  std::function<std::unique_ptr<exec::Aggregate>(const TypePtr& resultType)>
      factory;
};

/// Registers merge() for the built-in sketch types plus any caller-supplied
/// additionalSketchTypes (matched before the built-ins).
void registerMergeAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite,
    const std::vector<MergeSketchType>& additionalSketchTypes);

} // namespace facebook::velox::aggregate::prestosql
