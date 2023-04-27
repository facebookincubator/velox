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

#include "boost/math/distributions/beta.hpp"

namespace facebook::velox::functions {

constexpr double kInf = std::numeric_limits<double>::infinity();

namespace {

template <typename T>
struct BetaCDFFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void
  call(double& result, double a, double b, double value) {
    VELOX_USER_CHECK((a > 0) && (b > 0), "alpha and beta must be > 0");
    VELOX_USER_CHECK(
        (value >= 0) && (value <= 1), "value must be in the interval [0, 1]");
    VELOX_USER_CHECK(
        (a != kInf) && (b != kInf),
        "alpha, beta values can't accept infinity value");

    beta_distribution<> dist(a, b);
    result = boost::math::cdf(dist, value);
  }
};

} // namespace
} // namespace facebook::velox::functions
