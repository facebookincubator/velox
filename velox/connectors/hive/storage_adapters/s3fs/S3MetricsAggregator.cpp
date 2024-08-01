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
#include <glog/logging.h>
#include <folly/Singleton.h>

namespace facebook::velox::filesystems {

namespace {
folly::Singleton<S3MetricsAggregator> s3MetricsAggregatorSingleton;
} // namespace

std::shared_ptr<S3MetricsAggregator> S3MetricsAggregator::getInstance() {
  return s3MetricsAggregatorSingleton.try_get();
}

S3MetricsAggregator::S3MetricsAggregator() {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<std::string> metricNames = {
      "presto_hive_s3_presto_s3_file_system_active_connections_total_count",
      "presto_hive_s3_presto_s3_file_system_started_uploads_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_failed_uploads_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_successful_uploads_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_metadata_calls_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_list_status_calls_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_list_located_status_calls_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_list_objects_calls_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_other_read_errors_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_aws_aborted_exceptions_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_socket_exceptions_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_get_object_errors_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_get_metadata_errors_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_get_object_retries_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_get_metadata_retries_one_minute_count",
      "presto_hive_s3_presto_s3_file_system_read_retries_one_minute_count"};

  for (const auto& name : metricNames) {
    metrics_[name] = 0;
  }
}

// Increment the specified metric
void S3MetricsAggregator::incrementMetric(const std::string& metricName) {
  std::lock_guard<std::mutex> guard(mutex_);
  metrics_[metricName]++;
  LOG(INFO) << "Incremented metric " << metricName << " to "
            << metrics_[metricName];
}

// Retrieve the current value of the specified metric
uint64_t S3MetricsAggregator::getMetric(const std::string& metricName) {
  std::lock_guard<std::mutex> guard(mutex_);
  return metrics_[metricName];
}

// Reset the specified metric
void S3MetricsAggregator::resetMetric(const std::string& metricName) {
  std::lock_guard<std::mutex> guard(mutex_);
  metrics_[metricName] = 0;
  LOG(INFO) << "Reset metric " << metricName << " to 0";
}

} // namespace facebook::velox::filesystems
