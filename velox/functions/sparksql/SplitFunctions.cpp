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

#include <utility>

#include "velox/expression/VectorFunction.h"
#include "velox/expression/VectorWriters.h"

namespace facebook::velox::functions::sparksql {
namespace {



/// The function returns specialized version of split based on the constant
/// inputs.
/// \param inputArgs the inputs types (VARCHAR, VARCHAR, int64) and constant
///     values (if provided).
std::shared_ptr<exec::VectorFunction> createSplit(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  return makeRe2SplitAll(name, inputArgs);
}

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  // varchar, varchar -> array(varchar)
  return {exec::FunctionSignatureBuilder()
              .returnType("array(varchar)")
              .argumentType("varchar")
              .constantArgumentType("varchar")
              .build()};
}

} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_regexp_split,
    signatures(),
    createSplit);
} // namespace facebook::velox::functions::sparksql
