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
#include "velox/functions/lib/Re2Functions.h"
#include "velox/functions/prestosql/RegexpReplace.h"
#include "velox/functions/prestosql/SplitPart.h"
#include "velox/functions/prestosql/StringFunctions.h"

namespace facebook::velox::functions {

namespace {
std::shared_ptr<exec::VectorFunction> makeRegexExtract(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  return makeRe2Extract(name, inputArgs, /*emptyNoMatch=*/false);
}

void registerSimpleFunctions(const std::string& prefix) {
  using namespace stringImpl;

  // Register string functions.
  registerFunction<ChrFunction, Varchar, int64_t>({prefix + "chr"});
  registerFunction<CodePointFunction, int32_t, Varchar>({prefix + "codepoint"});
  registerFunction<LengthFunction, int64_t, Varchar>({prefix + "length"});

  registerFunction<SubstrFunction, Varchar, Varchar, int64_t>(
      {prefix + "substr"});
  registerFunction<SubstrFunction, Varchar, Varchar, int64_t, int64_t>(
      {prefix + "substr"});
  registerFunction<SubstrFunction, Varchar, Varchar, int32_t>(
      {prefix + "substr"});
  registerFunction<SubstrFunction, Varchar, Varchar, int32_t, int32_t>(
      {prefix + "substr"});

  registerFunction<SplitPart, Varchar, Varchar, Varchar, int64_t>(
      {prefix + "split_part"});

  registerFunction<TrimFunction, Varchar, Varchar>({prefix + "trim"});
  registerFunction<LTrimFunction, Varchar, Varchar>({prefix + "ltrim"});
  registerFunction<RTrimFunction, Varchar, Varchar>({prefix + "rtrim"});

  registerFunction<LPadFunction, Varchar, Varchar, int64_t, Varchar>(
      {prefix + "lpad"});
  registerFunction<RPadFunction, Varchar, Varchar, int64_t, Varchar>(
      {prefix + "rpad"});

  exec::registerStatefulVectorFunction(
      prefix + "like", likeSignatures(), makeLike);

  registerFunction<SplitPart, Varchar, Varchar, Varchar, int64_t>(
      {prefix + "split_part"});
  registerFunction<Re2RegexpReplacePresto, Varchar, Varchar, Varchar>(
      {prefix + "regexp_replace"});
  registerFunction<Re2RegexpReplacePresto, Varchar, Varchar, Varchar, Varchar>(
      {prefix + "regexp_replace"});
}
} // namespace

void registerStringFunctions(const std::string& prefix) {
  registerSimpleFunctions(prefix);

  VELOX_REGISTER_VECTOR_FUNCTION(udf_lower, prefix + "lower");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_upper, prefix + "upper");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_split, prefix + "split");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_concat, prefix + "concat");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_replace, prefix + "replace");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_reverse, prefix + "reverse");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_to_utf8, prefix + "to_utf8");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_from_utf8, prefix + "from_utf8");

  // Regex functions
  exec::registerStatefulVectorFunction(
      prefix + "regexp_extract", re2ExtractSignatures(), makeRegexExtract);
  exec::registerStatefulVectorFunction(
      prefix + "regexp_extract_all",
      re2ExtractAllSignatures(),
      makeRe2ExtractAll);
  exec::registerStatefulVectorFunction(
      prefix + "regexp_like", re2SearchSignatures(), makeRe2Search);

  registerFunction<StrLPosFunction, int64_t, Varchar, Varchar>(
      {prefix + "strpos"});
  registerFunction<StrLPosFunction, int64_t, Varchar, Varchar, int64_t>(
      {prefix + "strpos"});
  registerFunction<StrRPosFunction, int64_t, Varchar, Varchar>(
      {prefix + "strrpos"});
  registerFunction<StrRPosFunction, int64_t, Varchar, Varchar, int64_t>(
      {prefix + "strrpos"});
}
} // namespace facebook::velox::functions
