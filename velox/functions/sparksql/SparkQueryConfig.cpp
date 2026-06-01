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
#include "velox/functions/sparksql/SparkQueryConfig.h"

namespace facebook::velox::functions::sparksql {

const std::vector<config::ConfigProperty>&
SparkQueryConfig::registeredProperties() {
  static const std::vector<config::ConfigProperty> kProperties = [] {
    std::vector<config::ConfigProperty> properties;
#define VELOX_REGISTER_SPARK_CONFIG(constName)                           \
  config::registerConfigProperty<SparkQueryConfig::constName##Property>( \
      properties)

    VELOX_REGISTER_SPARK_CONFIG(kAnsiEnabled);
    VELOX_REGISTER_SPARK_CONFIG(kBloomFilterExpectedNumItems);
    VELOX_REGISTER_SPARK_CONFIG(kBloomFilterNumBits);
    VELOX_REGISTER_SPARK_CONFIG(kBloomFilterMaxNumBits);
    VELOX_REGISTER_SPARK_CONFIG(kBloomFilterMaxNumItems);
    VELOX_REGISTER_SPARK_CONFIG(kPartitionId);
    VELOX_REGISTER_SPARK_CONFIG(kLegacyDateFormatter);
    VELOX_REGISTER_SPARK_CONFIG(kLegacyStatisticalAggregate);
    VELOX_REGISTER_SPARK_CONFIG(kJsonIgnoreNullFields);
    VELOX_REGISTER_SPARK_CONFIG(kCollectListIgnoreNulls);
    VELOX_REGISTER_SPARK_CONFIG(kLegacySplitEmptyPattern);

#undef VELOX_REGISTER_SPARK_CONFIG

    return properties;
  }();
  return kProperties;
}

} // namespace facebook::velox::functions::sparksql
