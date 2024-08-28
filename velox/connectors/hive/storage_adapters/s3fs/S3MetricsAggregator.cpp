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

#include "velox/connectors/hive/storage_adapters/s3fs/S3MetricsAggregator.h"
#include <folly/Singleton.h>
#include "velox/common/base/StatsReporter.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Metrics.h"
namespace facebook::velox::filesystems {

namespace {
folly::Singleton<S3MetricsAggregator> s3MetricsAggregatorSingleton;
} // namespace

std::shared_ptr<S3MetricsAggregator> S3MetricsAggregator::getInstance() {
  return s3MetricsAggregatorSingleton.try_get();
}

S3MetricsAggregator::S3MetricsAggregator() {
  // Define and initialize the metrics
  DEFINE_METRIC(kMetricS3ActiveConnections, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3StartedUploads, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3FailedUploads, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3SuccessfulUploads, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3MetadataCalls, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3ListStatusCalls, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3ListLocatedStatusCalls, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3ListObjectsCalls, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3OtherReadErrors, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3AwsAbortedExceptions, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3SocketExceptions, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3GetObjectErrors, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3GetMetadataErrors, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3GetObjectRetries, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3GetMetadataRetries, velox::StatType::COUNT);
  DEFINE_METRIC(kMetricS3ReadRetries, velox::StatType::COUNT);

  // Initialize the metrics map with default values
  auto metrics = metrics_.wlock(); // Write lock for initialization
  std::vector<std::string> metricNames = {
      kMetricS3ActiveConnections,
      kMetricS3StartedUploads,
      kMetricS3FailedUploads,
      kMetricS3SuccessfulUploads,
      kMetricS3MetadataCalls,
      kMetricS3ListStatusCalls,
      kMetricS3ListLocatedStatusCalls,
      kMetricS3ListObjectsCalls,
      kMetricS3OtherReadErrors,
      kMetricS3AwsAbortedExceptions,
      kMetricS3SocketExceptions,
      kMetricS3GetObjectErrors,
      kMetricS3GetMetadataErrors,
      kMetricS3GetObjectRetries,
      kMetricS3GetMetadataRetries,
      kMetricS3ReadRetries};

  for (const auto& name : metricNames) {
    (*metrics)[name] = 0;
  }
}

// Increment the specified metric
void S3MetricsAggregator::incrementMetric(const std::string& metricName) {
  auto metrics = metrics_.wlock();
  (*metrics)[metricName]++;
}

// Retrieve the current value of the specified metric
uint64_t S3MetricsAggregator::getMetric(const std::string& metricName) {
  auto metrics = metrics_.rlock();
  auto it = metrics->find(metricName);
  if (it != metrics->end()) {
    return it->second;
  }
  return 0; // If the metric is not found, return 0
}

void S3MetricsAggregator::resetMetric(const std::string& metricName) {
  auto metrics = metrics_.wlock();
  (*metrics)[metricName] = 0;
}

} // namespace facebook::velox::filesystems
