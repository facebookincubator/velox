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
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Probability.h"

// Register Presto Probability, Trigonometric, and Statistical functions.

namespace facebook::velox::functions {

namespace {
void registerProbTrigFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<CosFunction, double, double>(
      {prefix + "cos"}, {}, true, defaultOwner);
  registerFunction<CoshFunction, double, double>(
      {prefix + "cosh"}, {}, true, defaultOwner);
  registerFunction<AcosFunction, double, double>(
      {prefix + "acos"}, {}, true, defaultOwner);
  registerFunction<SinFunction, double, double>(
      {prefix + "sin"}, {}, true, defaultOwner);
  registerFunction<AsinFunction, double, double>(
      {prefix + "asin"}, {}, true, defaultOwner);
  registerFunction<TanFunction, double, double>(
      {prefix + "tan"}, {}, true, defaultOwner);
  registerFunction<TanhFunction, double, double>(
      {prefix + "tanh"}, {}, true, defaultOwner);
  registerFunction<AtanFunction, double, double>(
      {prefix + "atan"}, {}, true, defaultOwner);
  registerFunction<Atan2Function, double, double, double>(
      {prefix + "atan2"}, {}, true, defaultOwner);

  registerFunction<BetaCDFFunction, double, double, double, double>(
      {prefix + "beta_cdf"}, {}, true, defaultOwner);
  registerFunction<NormalCDFFunction, double, double, double, double>(
      {prefix + "normal_cdf"}, {}, true, defaultOwner);
  registerFunction<BinomialCDFFunction, double, int64_t, double, int64_t>(
      {prefix + "binomial_cdf"}, {}, true, defaultOwner);
  registerFunction<BinomialCDFFunction, double, int32_t, double, int32_t>(
      {prefix + "binomial_cdf"}, {}, true, defaultOwner);
  registerFunction<CauchyCDFFunction, double, double, double, double>(
      {prefix + "cauchy_cdf"}, {}, true, defaultOwner);
  registerFunction<ChiSquaredCDFFunction, double, double, double>(
      {prefix + "chi_squared_cdf"}, {}, true, defaultOwner);
  registerFunction<FCDFFunction, double, double, double, double>(
      {prefix + "f_cdf"}, {}, true, defaultOwner);
  registerFunction<InverseBetaCDFFunction, double, double, double, double>(
      {prefix + "inverse_beta_cdf"}, {}, true, defaultOwner);
  registerFunction<InverseNormalCDFFunction, double, double, double, double>(
      {prefix + "inverse_normal_cdf"}, {}, true, defaultOwner);
  registerFunction<PoissonCDFFunction, double, double, int32_t>(
      {prefix + "poisson_cdf"}, {}, true, defaultOwner);
  registerFunction<GammaCDFFunction, double, double, double, double>(
      {prefix + "gamma_cdf"}, {}, true, defaultOwner);
  registerFunction<LaplaceCDFFunction, double, double, double, double>(
      {prefix + "laplace_cdf"}, {}, true, defaultOwner);
  registerFunction<
      WilsonIntervalUpperFunction,
      double,
      int64_t,
      int64_t,
      double>({prefix + "wilson_interval_upper"}, {}, true, defaultOwner);
  registerFunction<
      WilsonIntervalLowerFunction,
      double,
      int64_t,
      int64_t,
      double>({prefix + "wilson_interval_lower"}, {}, true, defaultOwner);

  registerFunction<WeibullCDFFunction, double, double, double, double>(
      {prefix + "weibull_cdf"}, {}, true, defaultOwner);
  registerFunction<InverseWeibullCDFFunction, double, double, double, double>(
      {prefix + "inverse_weibull_cdf"}, {}, true, defaultOwner);
  registerFunction<InverseCauchyCDFFunction, double, double, double, double>(
      {prefix + "inverse_cauchy_cdf"}, {}, true, defaultOwner);
  registerFunction<InverseLaplaceCDFFunction, double, double, double, double>(
      {prefix + "inverse_laplace_cdf"}, {}, true, defaultOwner);
  registerFunction<InverseGammaCDFFunction, double, double, double, double>(
      {prefix + "inverse_gamma_cdf"}, {}, true, defaultOwner);
  registerFunction<
      InverseBinomialCDFFunction,
      int32_t,
      int32_t,
      double,
      double>({prefix + "inverse_binomial_cdf"}, {}, true, defaultOwner);
  registerFunction<InversePoissonCDFFunction, int32_t, double, double>(
      {prefix + "inverse_poisson_cdf"}, {}, true, defaultOwner);
  registerFunction<InverseFCDFFunction, double, double, double, double>(
      {prefix + "inverse_f_cdf"}, {}, true, defaultOwner);
  registerFunction<InverseChiSquaredCdf, double, double, double>(
      {prefix + "inverse_chi_squared_cdf"}, {}, true, defaultOwner);
  registerFunction<TCDFFunction, double, double, double>(
      {prefix + "t_cdf"}, {}, true, defaultOwner);
  registerFunction<InverseTCDFFunction, double, double, double>(
      {prefix + "inverse_t_cdf"}, {}, true, defaultOwner);
}

} // namespace

void registerProbabilityTrigonometryFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {}) {
  registerProbTrigFunctions(prefix, defaultOwner);
}

} // namespace facebook::velox::functions
