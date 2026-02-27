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
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/functions/prestosql/aggregates/ApproxDistinctAggregates.h"
#include "velox/functions/prestosql/aggregates/ApproxMostFrequentAggregate.h"
#include "velox/functions/prestosql/aggregates/ApproxPercentileAggregate.h"
#include "velox/functions/prestosql/aggregates/ArbitraryAggregate.h"
#include "velox/functions/prestosql/aggregates/ArrayAggAggregate.h"
#include "velox/functions/prestosql/aggregates/ArrayUnionSumAggregate.h"
#include "velox/functions/prestosql/aggregates/AverageAggregate.h"
#include "velox/functions/prestosql/aggregates/BitwiseAggregates.h"
#include "velox/functions/prestosql/aggregates/BitwiseXorAggregate.h"
#include "velox/functions/prestosql/aggregates/BoolAggregates.h"
#include "velox/functions/prestosql/aggregates/CentralMomentsAggregates.h"
#include "velox/functions/prestosql/aggregates/ChecksumAggregate.h"
#include "velox/functions/prestosql/aggregates/ClassificationAggregation.h"
#include "velox/functions/prestosql/aggregates/CountAggregate.h"
#include "velox/functions/prestosql/aggregates/CountIfAggregate.h"
#include "velox/functions/prestosql/aggregates/CovarianceAggregates.h"
#include "velox/functions/prestosql/aggregates/EntropyAggregates.h"
#include "velox/functions/prestosql/aggregates/GeometricMeanAggregate.h"
#include "velox/functions/prestosql/aggregates/HistogramAggregate.h"
#include "velox/functions/prestosql/aggregates/KHyperLogLogAggregate.h"
#include "velox/functions/prestosql/aggregates/MapAggAggregate.h"
#include "velox/functions/prestosql/aggregates/MapUnionAggregate.h"
#include "velox/functions/prestosql/aggregates/MapUnionSumAggregate.h"
#include "velox/functions/prestosql/aggregates/MaxByAggregate.h"
#include "velox/functions/prestosql/aggregates/MaxSizeForStatsAggregate.h"
#include "velox/functions/prestosql/aggregates/MergeAggregate.h"
#include "velox/functions/prestosql/aggregates/MinByAggregate.h"
#include "velox/functions/prestosql/aggregates/MinMaxAggregates.h"
#include "velox/functions/prestosql/aggregates/MultiMapAggAggregate.h"
#include "velox/functions/prestosql/aggregates/NoisyApproxSfmAggregate.h"
#include "velox/functions/prestosql/aggregates/NoisyAvgGaussianAggregate.h"
#include "velox/functions/prestosql/aggregates/NoisyCountGaussianAggregate.h"
#include "velox/functions/prestosql/aggregates/NoisyCountIfGaussianAggregate.h"
#include "velox/functions/prestosql/aggregates/NoisySumGaussianAggregate.h"
#include "velox/functions/prestosql/aggregates/NumericHistogramAggregate.h"
#include "velox/functions/prestosql/aggregates/QDigestAggAggregate.h"
#include "velox/functions/prestosql/aggregates/ReduceAgg.h"
#include "velox/functions/prestosql/aggregates/ReservoirSampleAggregate.h"
#include "velox/functions/prestosql/aggregates/SetAggregates.h"
#include "velox/functions/prestosql/aggregates/SetDigestAggregate.h"
#include "velox/functions/prestosql/aggregates/SumAggregate.h"
#include "velox/functions/prestosql/aggregates/SumDataSizeForStatsAggregate.h"
#include "velox/functions/prestosql/aggregates/VarianceAggregates.h"
#include "velox/functions/prestosql/types/JsonRegistration.h"
#include "velox/functions/prestosql/types/KHyperLogLogRegistration.h"
#include "velox/functions/prestosql/types/SetDigestRegistration.h"
#include "velox/functions/prestosql/types/TDigestRegistration.h"
#ifdef VELOX_ENABLE_GEO
#include "velox/functions/prestosql/aggregates/GeometryAggregate.h"
#endif

namespace facebook::velox::aggregate::prestosql {

extern void registerApproxMostFrequentAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerApproxPercentileAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerArbitraryAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerArrayAggAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerAverageAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerBitwiseXorAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool onlyPrestoSignatures,
    bool overwrite);
extern void registerChecksumAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerFalloutAggregation(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerPrecisionAggregation(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerRecallAggregation(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMissRateAggregation(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerThresholdAggregation(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerCountAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerCountIfAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerEntropyAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerGeometricMeanAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
#ifdef VELOX_ENABLE_GEO
extern void registerGeometryAggregate(
    const std::vector<std::string>& convexHullNames,
    const std::vector<std::string>& geometryUnionNames,
    bool overwrite);
#endif
extern void registerHistogramAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMapAggAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMapUnionAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMapUnionSumAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMakeSetDigestAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMergeSetDigestAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMaxDataSizeForStatsAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMultiMapAggAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerSumDataSizeForStatsAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerReduceAgg(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerSetAggAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerSetUnionAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);

extern void registerApproxDistinctAggregates(
    const std::vector<std::string>& approxDistinctNames,
    const std::vector<std::string>& approxSetNames,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerBitwiseAndAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool onlyPrestoSignatures,
    bool overwrite);
extern void registerBitwiseOrAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool onlyPrestoSignatures,
    bool overwrite);
extern void registerBoolAndAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerBoolOrAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerKurtosisAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerSkewnessAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerCovarPopAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerCovarSampAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerCorrAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerRegrInterceptAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerRegrSlopeAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerRegrCountAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerRegrAvgyAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerRegrAvgxAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerRegrSxyAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerRegrSxxAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerRegrSyyAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerRegrR2Aggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMinAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMaxAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMaxByAggregates(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerMinByAggregates(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerSumAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerStdDevSampAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerStdDevPopAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerVarSampAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerVarPopAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerTDigestAggregate(
    const std::vector<std::string>& names,
    bool overwrite);
extern void registerKHyperLogLogAggregates(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite);
extern void registerQDigestAggAggregate(
    const std::vector<std::string>& names,
    bool overwrite);

void registerAllAggregateFunctions(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool onlyPrestoSignatures,
    bool overwrite) {
  registerJsonType();
  registerSetDigestType();
  registerTDigestType();
  registerKHyperLogLogType();
  registerApproxDistinctAggregates(
      {prefix + kApproxDistinct},
      {prefix + kApproxSet},
      withCompanionFunctions,
      overwrite);
  registerApproxMostFrequentAggregate(
      {prefix + kApproxMostFrequent}, withCompanionFunctions, overwrite);
  registerApproxPercentileAggregate(
      {prefix + kApproxPercentile}, withCompanionFunctions, overwrite);
  registerQDigestAggAggregate({prefix + kQDigestAgg}, overwrite);
  registerArbitraryAggregate(
      {prefix + kArbitrary, prefix + kAnyValue},
      withCompanionFunctions,
      overwrite);
  registerArrayAggAggregate(
      {prefix + kArrayAgg}, withCompanionFunctions, overwrite);
  // array_union_sum is a Velox-only function, not available in Presto.
  if (!onlyPrestoSignatures) {
    registerArrayUnionSumAggregate(prefix, withCompanionFunctions, overwrite);
  }
  registerAverageAggregate({prefix + kAvg}, withCompanionFunctions, overwrite);
  registerBitwiseAndAggregate(
      {prefix + kBitwiseAnd},
      withCompanionFunctions,
      onlyPrestoSignatures,
      overwrite);
  registerBitwiseOrAggregate(
      {prefix + kBitwiseOr},
      withCompanionFunctions,
      onlyPrestoSignatures,
      overwrite);
  registerBitwiseXorAggregate(
      {prefix + kBitwiseXor},
      withCompanionFunctions,
      onlyPrestoSignatures,
      overwrite);
  registerBoolAndAggregate(
      {prefix + kEvery, prefix + kBoolAnd}, withCompanionFunctions, overwrite);
  registerBoolOrAggregate(
      {prefix + kBoolOr}, withCompanionFunctions, overwrite);
  registerKurtosisAggregate(
      {prefix + kKurtosis}, withCompanionFunctions, overwrite);
  registerSkewnessAggregate(
      {prefix + kSkewness}, withCompanionFunctions, overwrite);
  registerChecksumAggregate(
      {prefix + kChecksum}, withCompanionFunctions, overwrite);
  registerFalloutAggregation(
      {prefix + kClassificationFallout}, withCompanionFunctions, overwrite);
  registerPrecisionAggregation(
      {prefix + kClassificationPrecision}, withCompanionFunctions, overwrite);
  registerRecallAggregation(
      {prefix + kClassificationRecall}, withCompanionFunctions, overwrite);
  registerMissRateAggregation(
      {prefix + kClassificationMissRate}, withCompanionFunctions, overwrite);
  registerThresholdAggregation(
      {prefix + kClassificationThreshold}, withCompanionFunctions, overwrite);
  registerCountAggregate({prefix + kCount}, withCompanionFunctions, overwrite);
  registerCountIfAggregate(
      {prefix + kCountIf}, withCompanionFunctions, overwrite);
  registerCovarPopAggregate(
      {prefix + kCovarPop}, withCompanionFunctions, overwrite);
  registerCovarSampAggregate(
      {prefix + kCovarSamp}, withCompanionFunctions, overwrite);
  registerCorrAggregate({prefix + kCorr}, withCompanionFunctions, overwrite);
  registerRegrInterceptAggregate(
      {prefix + kRegrIntercept}, withCompanionFunctions, overwrite);
  registerRegrSlopeAggregate(
      {prefix + kRegrSlop}, withCompanionFunctions, overwrite);
  registerRegrCountAggregate(
      {prefix + kRegrCount}, withCompanionFunctions, overwrite);
  registerRegrAvgyAggregate(
      {prefix + kRegrAvgy}, withCompanionFunctions, overwrite);
  registerRegrAvgxAggregate(
      {prefix + kRegrAvgx}, withCompanionFunctions, overwrite);
  registerRegrSxyAggregate(
      {prefix + kRegrSxy}, withCompanionFunctions, overwrite);
  registerRegrSxxAggregate(
      {prefix + kRegrSxx}, withCompanionFunctions, overwrite);
  registerRegrSyyAggregate(
      {prefix + kRegrSyy}, withCompanionFunctions, overwrite);
  registerRegrR2Aggregate(
      {prefix + kRegrR2}, withCompanionFunctions, overwrite);
  registerEntropyAggregate(
      {prefix + kEntropy}, withCompanionFunctions, overwrite);
  registerGeometricMeanAggregate(
      {prefix + kGeometricMean}, withCompanionFunctions, overwrite);
#ifdef VELOX_ENABLE_GEO
  registerGeometryAggregate(
      {prefix + kConvexHull}, {prefix + kGeometryUnion}, overwrite);
#endif
  registerHistogramAggregate(
      {prefix + kHistogram}, withCompanionFunctions, overwrite);
  registerMapAggAggregate(
      {prefix + kMapAgg}, withCompanionFunctions, overwrite);
  registerMapUnionAggregate(
      {prefix + kMapUnion}, withCompanionFunctions, overwrite);
  registerMapUnionSumAggregate(
      {prefix + kMapUnionSum}, withCompanionFunctions, overwrite);
  registerMakeSetDigestAggregate(
      {prefix + kMakeSetDigest}, withCompanionFunctions, overwrite);
  registerMergeSetDigestAggregate(
      {prefix + kMergeSetDigest}, withCompanionFunctions, overwrite);
  registerMaxDataSizeForStatsAggregate(
      {prefix + kMaxSizeForStats}, withCompanionFunctions, overwrite);
  registerMultiMapAggAggregate(
      {prefix + kMultiMapAgg}, withCompanionFunctions, overwrite);
  registerSumDataSizeForStatsAggregate(
      {prefix + kSumDataSizeForStats}, withCompanionFunctions, overwrite);
  registerMergeAggregate({prefix + kMerge}, withCompanionFunctions, overwrite);
  registerMinAggregate({prefix + kMin}, withCompanionFunctions, overwrite);
  registerMaxAggregate({prefix + kMax}, withCompanionFunctions, overwrite);
  registerMaxByAggregates({prefix + kMaxBy}, withCompanionFunctions, overwrite);
  registerMinByAggregates({prefix + kMinBy}, withCompanionFunctions, overwrite);
  registerNoisyAvgGaussianAggregate(
      {prefix + kNoisyAvgGaussian}, withCompanionFunctions, overwrite);
  registerNoisyCountIfGaussianAggregate(
      {prefix + kNoisyCountIfGaussian}, withCompanionFunctions, overwrite);
  registerNoisyCountGaussianAggregate(
      {prefix + kNoisyCountGaussian}, withCompanionFunctions, overwrite);
  registerNoisySumGaussianAggregate(
      {prefix + kNoisySumGaussian}, withCompanionFunctions, overwrite);
  registerReduceAgg({prefix + kReduceAgg}, withCompanionFunctions, overwrite);
  registerReservoirSampleAggregate(
      {prefix + kReservoirSample}, withCompanionFunctions, overwrite);
  registerSetAggAggregate(
      {prefix + kSetAgg}, withCompanionFunctions, overwrite);
  registerSetUnionAggregate(
      {prefix + kSetUnion}, withCompanionFunctions, overwrite);
  registerSumAggregate({prefix + kSum}, withCompanionFunctions, overwrite);
  registerStdDevSampAggregate(
      {prefix + kStdDevSamp, prefix + kStdDev},
      withCompanionFunctions,
      overwrite);
  registerStdDevPopAggregate(
      {prefix + kStdDevPop}, withCompanionFunctions, overwrite);
  registerVarSampAggregate(
      {prefix + kVarSamp, prefix + kVariance},
      withCompanionFunctions,
      overwrite);
  registerVarPopAggregate(
      {prefix + kVarPop}, withCompanionFunctions, overwrite);
  registerTDigestAggregate({prefix + kTDigestAgg}, overwrite);
  registerNoisyApproxSfmAggregate(
      {prefix + kNoisyApproxSetSfm},
      {prefix + kNoisyApproxDistinctSfm},
      {prefix + kNoisyApproxSetSfmFromIndexAndZeros},
      withCompanionFunctions,
      overwrite);
  registerNumericHistogramAggregate(
      {prefix + kNumericHistogram}, withCompanionFunctions, overwrite);
  registerKHyperLogLogAggregates(
      {prefix + kKHyperLogLogAgg}, withCompanionFunctions, overwrite);
}

void registerInternalAggregateFunctions(const std::string& prefix) {
  bool withCompanionFunctions = false;
  bool overwrite = false;
  registerCountDistinctAggregate(
      {prefix + "$internal$count_distinct"}, withCompanionFunctions, overwrite);
  registerInternalArrayAggAggregate(
      {prefix + "$internal$array_agg"}, withCompanionFunctions, overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
