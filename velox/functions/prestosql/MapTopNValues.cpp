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
#include "velox/functions/prestosql/MapTopNLambdaImpl.h"

namespace facebook::velox::functions {
namespace {

/// See documentation at https://prestodb.io/docs/current/functions/map.html
/// Returns the top N values of the given map, ranked by the provided transform
/// lambda, in descending order.
class MapTopNValuesLambdaFunction
    : public MapTopNLambdaFunction<MapTopNMode::Values> {
 public:
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // map_top_n_values(map(K,V), bigint, function(K,V,U)) -> array(V)
    return {exec::FunctionSignatureBuilder()
                .typeVariable("K")
                .typeVariable("V")
                .orderableTypeVariable("U")
                .returnType("array(V)")
                .argumentType("map(K,V)")
                .argumentType("bigint")
                .argumentType("function(K, V, U)")
                .build()};
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_map_top_n_values,
    MapTopNValuesLambdaFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<MapTopNValuesLambdaFunction>());

} // namespace facebook::velox::functions
