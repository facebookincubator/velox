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
#include "velox/functions/prestosql/StringFunctions.h"
#include "velox/functions/prestosql/URLFunctions.h"

namespace facebook::velox::functions {

void registerURLFunctions(const std::string& prefix) {
  registerFunction<UrlExtractHostFunction, Varchar, Varchar>(
      {prefix + "url_extract_host"});
  registerFunction<UrlExtractHostFunction, VarcharN<L1>, VarcharN<L1>>(
      {prefix + "url_extract_host"});
  registerFunction<UrlExtractFragmentFunction, Varchar, Varchar>(
      {prefix + "url_extract_fragment"});
  registerFunction<UrlExtractFragmentFunction, VarcharN<L1>, VarcharN<L1>>(
      {prefix + "url_extract_fragment"});
  registerFunction<UrlExtractPathFunction, Varchar, Varchar>(
      {prefix + "url_extract_path"});
  registerFunction<UrlExtractPathFunction, VarcharN<L1>, VarcharN<L1>>(
      {prefix + "url_extract_path"});
  registerFunction<UrlExtractParameterFunction, Varchar, Varchar, Varchar>(
      {prefix + "url_extract_parameter"});
  registerFunction<
      UrlExtractParameterFunction,
      VarcharN<L1>,
      VarcharN<L1>,
      VarcharN<L2>>({prefix + "url_extract_parameter"});
  registerFunction<UrlExtractProtocolFunction, Varchar, Varchar>(
      {prefix + "url_extract_protocol"});
  registerFunction<UrlExtractProtocolFunction, VarcharN<L1>, VarcharN<L1>>(
      {prefix + "url_extract_protocol"});
  registerFunction<UrlExtractPortFunction, int64_t, Varchar>(
      {prefix + "url_extract_port"});
  registerFunction<UrlExtractPortFunction, int64_t, VarcharN<L1>>(
      {prefix + "url_extract_port"});
  registerFunction<UrlExtractQueryFunction, Varchar, Varchar>(
      {prefix + "url_extract_query"});
  registerFunction<UrlExtractQueryFunction, VarcharN<L1>, VarcharN<L1>>(
      {prefix + "url_extract_query"});
  registerFunction<UrlEncodeFunction, Varchar, Varchar>(
      {prefix + "url_encode"});
  exec::SignatureVariable urlEncodeConstraint = exec::SignatureVariable(
      L2::name(),
      fmt::format("min(2147483647, {x} * 12)", fmt::arg("x", L1::name())),
      exec::ParameterType::kIntegerParameter);
  registerFunction<UrlEncodeFunction, VarcharN<L2>, VarcharN<L1>>(
      {prefix + "url_encode"}, {urlEncodeConstraint});
  registerFunction<UrlDecodeFunction, Varchar, Varchar>(
      {prefix + "url_decode"});
  registerFunction<UrlDecodeFunction, VarcharN<L1>, VarcharN<L1>>(
      {prefix + "url_decode"});
}
} // namespace facebook::velox::functions
