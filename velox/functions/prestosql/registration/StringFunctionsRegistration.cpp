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
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/Re2Functions.h"
#include "velox/functions/prestosql/RegexpReplace.h"
#include "velox/functions/prestosql/RegexpSplit.h"
#include "velox/functions/prestosql/SplitPart.h"
#include "velox/functions/prestosql/SplitToMap.h"
#include "velox/functions/prestosql/SplitToMultiMap.h"
#include "velox/functions/prestosql/StringFunctions.h"
#include "velox/functions/prestosql/WordStem.h"

namespace facebook::velox::functions {

namespace {
std::shared_ptr<exec::VectorFunction> makeRegexExtract(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  return makeRe2Extract(name, inputArgs, config, /*emptyNoMatch=*/false);
}

void registerSimpleFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  using namespace stringImpl;

  // Register string functions.
  registerFunction<ChrFunction, Varchar, int64_t>(
      {prefix + "chr"}, {}, true, defaultOwner);
  registerFunction<CodePointFunction, int32_t, Varchar>(
      {prefix + "codepoint"}, {}, true, defaultOwner);
  registerFunction<HammingDistanceFunction, int64_t, Varchar, Varchar>(
      {prefix + "hamming_distance"}, {}, true, defaultOwner);
  registerFunction<LevenshteinDistanceFunction, int64_t, Varchar, Varchar>(
      {prefix + "levenshtein_distance"}, {}, true, defaultOwner);
  registerFunction<JaroWinklerSimilarityFunction, double, Varchar, Varchar>(
      {prefix + "jarowinkler_similarity"}, {}, true, defaultOwner);
  registerFunction<LongestCommonPrefixFunction, Varchar, Varchar, Varchar>(
      {prefix + "longest_common_prefix"}, {}, true, defaultOwner);
  registerFunction<LengthFunction, int64_t, Varchar>(
      {prefix + "length"}, {}, true, defaultOwner);
  registerFunction<BitLengthFunction, int64_t, Varchar>(
      {prefix + "bit_length"}, {}, true, defaultOwner);
  registerFunction<XxHash64StringFunction, int64_t, Varchar>(
      {prefix + "xxhash64_internal"}, {}, true, defaultOwner);

  // Length for varbinary have different semantics.
  registerFunction<LengthVarbinaryFunction, int64_t, Varbinary>(
      {prefix + "length"}, {}, true, defaultOwner);

  registerFunction<StartsWithFunction, bool, Varchar, Varchar>(
      {prefix + "starts_with"}, {}, true, defaultOwner);
  registerFunction<EndsWithFunction, bool, Varchar, Varchar>(
      {prefix + "ends_with"}, {}, true, defaultOwner);
  registerFunction<EndsWithFunction, bool, Varchar, UnknownValue>(
      {prefix + "ends_with"}, {}, true, defaultOwner);

  registerFunction<TrailFunction, Varchar, Varchar, int32_t>(
      {prefix + "trail"}, {}, true, defaultOwner);

  registerFunction<SubstrFunction, Varchar, Varchar, int64_t>(
      {prefix + "substr", prefix + "substring"}, {}, true, defaultOwner);
  registerFunction<SubstrFunction, Varchar, Varchar, int64_t, int64_t>(
      {prefix + "substr", prefix + "substring"}, {}, true, defaultOwner);

  // TODO Presto doesn't allow INTEGER types for 2nd and 3rd arguments. Remove
  // these signatures.
  registerFunction<SubstrFunction, Varchar, Varchar, int32_t>(
      {prefix + "substr", prefix + "substring"}, {}, true, defaultOwner);
  registerFunction<SubstrFunction, Varchar, Varchar, int32_t, int32_t>(
      {prefix + "substr", prefix + "substring"}, {}, true, defaultOwner);

  registerFunction<SubstrVarbinaryFunction, Varbinary, Varbinary, int64_t>(
      {prefix + "substr"}, {}, true, defaultOwner);
  registerFunction<
      SubstrVarbinaryFunction,
      Varbinary,
      Varbinary,
      int64_t,
      int64_t>({prefix + "substr"}, {}, true, defaultOwner);

  registerFunction<SplitPart, Varchar, Varchar, Varchar, int64_t>(
      {prefix + "split_part"}, {}, true, defaultOwner);

  registerFunction<TrimFunction, Varchar, Varchar>(
      {prefix + "trim"}, {}, true, defaultOwner);
  registerFunction<TrimFunction, Varchar, Varchar, Varchar>(
      {prefix + "trim"}, {}, true, defaultOwner);
  registerFunction<LTrimFunction, Varchar, Varchar>(
      {prefix + "ltrim"}, {}, true, defaultOwner);
  registerFunction<LTrimFunction, Varchar, Varchar, Varchar>(
      {prefix + "ltrim"}, {}, true, defaultOwner);
  registerFunction<RTrimFunction, Varchar, Varchar>(
      {prefix + "rtrim"}, {}, true, defaultOwner);
  registerFunction<RTrimFunction, Varchar, Varchar, Varchar>(
      {prefix + "rtrim"}, {}, true, defaultOwner);

  registerFunction<LPadFunction, Varchar, Varchar, int64_t, Varchar>(
      {prefix + "lpad"}, {}, true, defaultOwner);
  registerFunction<RPadFunction, Varchar, Varchar, int64_t, Varchar>(
      {prefix + "rpad"}, {}, true, defaultOwner);

  exec::registerStatefulVectorFunction(
      prefix + "like",
      likeSignatures(),
      makeLike,
      {},
      /*overwrite=*/true,
      defaultOwner);

  registerFunction<Re2RegexpReplacePresto, Varchar, Varchar, Varchar>(
      {prefix + "regexp_replace"}, {}, true, defaultOwner);
  registerFunction<Re2RegexpReplacePresto, Varchar, Varchar, Varchar, Varchar>(
      {prefix + "regexp_replace"}, {}, true, defaultOwner);
  exec::registerStatefulVectorFunction(
      prefix + "regexp_replace",
      regexpReplaceWithLambdaSignatures(),
      makeRegexpReplaceWithLambda,
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
      /*overwrite=*/true,
      defaultOwner);

  registerFunction<Re2RegexpSplit, Array<Varchar>, Varchar, Varchar>(
      {prefix + "regexp_split"}, {}, true, defaultOwner);
}

void registerSplitToMultiMap(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      SplitToMultiMapFunction,
      Map<Varchar, Array<Varchar>>,
      Varchar,
      Varchar,
      Varchar>({prefix + "split_to_multimap"}, {}, true, defaultOwner);
  registerFunction<
      SplitToMultiMapFunction,
      Map<Varchar, Array<Varchar>>,
      Varchar,
      UnknownValue,
      Varchar>({prefix + "split_to_multimap"}, {}, true, defaultOwner);
  registerFunction<
      SplitToMultiMapFunction,
      Map<Varchar, Array<Varchar>>,
      Varchar,
      Varchar,
      UnknownValue>({prefix + "split_to_multimap"}, {}, true, defaultOwner);
  registerFunction<
      SplitToMultiMapFunction,
      Map<Varchar, Array<Varchar>>,
      Varchar,
      UnknownValue,
      UnknownValue>({prefix + "split_to_multimap"}, {}, true, defaultOwner);
}

void registerSplitToMap(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      SplitToMapFunction,
      Map<Varchar, Varchar>,
      Varchar,
      Varchar,
      Varchar>({prefix + "split_to_map"}, {}, true, defaultOwner);

  registerFunction<
      SplitToMapFunction,
      Map<Varchar, Array<Varchar>>,
      Varchar,
      UnknownValue,
      Varchar>({prefix + "split_to_map"}, {}, true, defaultOwner);
  registerFunction<
      SplitToMapFunction,
      Map<Varchar, Array<Varchar>>,
      Varchar,
      Varchar,
      UnknownValue>({prefix + "split_to_map"}, {}, true, defaultOwner);
  registerFunction<
      SplitToMapFunction,
      Map<Varchar, Array<Varchar>>,
      Varchar,
      UnknownValue,
      UnknownValue>({prefix + "split_to_map"}, {}, true, defaultOwner);

  exec::registerVectorFunction(
      prefix + "split_to_map",
      {
          exec::FunctionSignatureBuilder()
              .returnType("map(varchar,varchar)")
              .argumentType("varchar")
              .argumentType("varchar")
              .argumentType("varchar")
              .argumentType("function(varchar,varchar,varchar,varchar)")
              .build(),
      },
      std::make_unique<exec::ApplyNeverCalled>(),
      {},
      /*overwrite=*/true,
      defaultOwner);
  registerFunction<
      SplitToMapFunction,
      Map<Varchar, Varchar>,
      Varchar,
      Varchar,
      Varchar,
      bool>({"$internal$split_to_map"}, {}, true, defaultOwner);
  expression::ExprRewriteRegistry::instance().registerRewrite(
      [prefix](const auto& expr) {
        return rewriteSplitToMapCall(prefix, expr);
      });
}
} // namespace

void registerStringFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerSimpleFunctions(prefix, defaultOwner);

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_lower, prefix + "lower", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_upper, prefix + "upper", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_split, prefix + "split", defaultOwner);

  registerSplitToMap(prefix, defaultOwner);
  registerSplitToMultiMap(prefix, defaultOwner);

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_concat, prefix + "concat", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_replaceFirst, prefix + "replace_first", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_replace, prefix + "replace", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_reverse, prefix + "reverse", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_to_utf8, prefix + "to_utf8", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_from_utf8, prefix + "from_utf8", defaultOwner);

  // Regex functions
  exec::registerStatefulVectorFunction(
      prefix + "regexp_extract",
      re2ExtractSignatures(),
      makeRegexExtract,
      {},
      /*overwrite=*/true,
      defaultOwner);
  exec::registerStatefulVectorFunction(
      prefix + "regexp_extract_all",
      re2ExtractAllSignatures(),
      makeRe2ExtractAll,
      {},
      /*overwrite=*/true,
      defaultOwner);
  exec::registerStatefulVectorFunction(
      prefix + "regexp_like",
      re2SearchSignatures(),
      makeRe2Search,
      {},
      /*overwrite=*/true,
      defaultOwner);

  registerFunction<StrLPosFunction, int64_t, Varchar, Varchar>(
      {prefix + "strpos"}, {}, true, defaultOwner);
  registerFunction<StrLPosFunction, int64_t, Varchar, Varchar, int64_t>(
      {prefix + "strpos"}, {}, true, defaultOwner);
  registerFunction<StrRPosFunction, int64_t, Varchar, Varchar>(
      {prefix + "strrpos"}, {}, true, defaultOwner);
  registerFunction<StrRPosFunction, int64_t, Varchar, Varchar, int64_t>(
      {prefix + "strrpos"}, {}, true, defaultOwner);

  registerFunction<NormalizeFunction, Varchar, Varchar>(
      {prefix + "normalize"}, {}, true, defaultOwner);
  registerFunction<NormalizeFunction, Varchar, Varchar, Varchar>(
      {prefix + "normalize"}, {}, true, defaultOwner);

  // word_stem function
  registerFunction<WordStemFunction, Varchar, Varchar>(
      {prefix + "word_stem"}, {}, true, defaultOwner);
  registerFunction<WordStemFunction, Varchar, Varchar, Varchar>(
      {prefix + "word_stem"}, {}, true, defaultOwner);

  registerFunction<KeySamplingPercentFunction, double, Varchar>(
      {prefix + "key_sampling_percent"}, {}, true, defaultOwner);
}
} // namespace facebook::velox::functions
