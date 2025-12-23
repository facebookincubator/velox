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

#include <string>
#include <vector>

#include "velox/type/Type.h"

namespace facebook::velox::exec {

/// Given a name of aggregate function and argument types, returns the result
/// type if the function exists. Throws if function doesn't exist or doesn't
/// support specified argument types. Since aggregate functions can be
/// integrated into internal steps of an aggregate operator — rather than
/// always being used as standalone functions at the SQL level — their result
/// types may not always be inferable from the intermediate types. As a
/// result, an exception might be thrown during the type resolution process. In
/// such cases, the caller should explicitly specify the result type. More
/// details can be found in
/// https://github.com/facebookincubator/velox/pull/11999#issuecomment-3274577979
/// and https://github.com/facebookincubator/velox/issues/12830.
TypePtr resolveResultType(
    const std::string& name,
    const std::vector<TypePtr>& argTypes);

/// Like 'resolveResultType', but with support for applying type conversions if
/// a function signature doesn't match 'argTypes' exactly.
///
/// @param coercions A list of optional type coercions that were applied to
/// resolve a function successfully. Contains one entry per argument. The entry
/// is null if no coercion is required for that argument. The entry is not null
/// if coercion is necessary.
TypePtr resolveResultTypeWithCoercions(
    const std::string& name,
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions);

/// Given a name of aggregate function and argument types, returns the
/// intermediate type if the function exists. Throws if function doesn't exist
/// or doesn't support specified argument types.
TypePtr resolveIntermediateType(
    const std::string& name,
    const std::vector<TypePtr>& argTypes);

/// Returns all the registered aggregation function names.
std::vector<std::string> getAggregateFunctionNames();

} // namespace facebook::velox::exec
