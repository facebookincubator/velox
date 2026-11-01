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

namespace facebook::velox::aggregate::prestosql {

/// Registers differential_entropy, with three signatures:
///   differential_entropy(size, x) -> double
///     Unweighted reservoir sampling + Vasicek order-statistics estimator.
///     Does not require known bounds on x.
///   differential_entropy(size, x, weight) -> double
///     Weighted reservoir sampling (Efraimidis-Spirakis A-ExpJ) + the same
///     Vasicek estimator applied to the drawn sample.
///   differential_entropy(bucket_count, x, weight, method, min, max) -> double
///     Fixed-histogram estimator over known bounds [min, max]. `method` is
///     'fixed_histogram_mle' or 'fixed_histogram_jacknife'.
void registerDifferentialEntropyAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

} // namespace facebook::velox::aggregate::prestosql
