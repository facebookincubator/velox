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

#ifndef VELOX_CONNECTORS_HIVE_STORAGE_ADAPTERS_S3FS_S3METRICSAGGREGATOR_H_
#define VELOX_CONNECTORS_HIVE_STORAGE_ADAPTERS_S3FS_S3METRICSAGGREGATOR_H_

#include <folly/Singleton.h>
#include <mutex>
#include <unordered_map>

namespace facebook::velox::filesystems {

class S3MetricsAggregator {
 public:
  static std::shared_ptr<S3MetricsAggregator> getInstance();

  void incrementMetric(const std::string& metricName);
  uint64_t getMetric(const std::string& metricName);
  void resetMetric(const std::string& metricName);

 private:
  S3MetricsAggregator();
  std::unordered_map<std::string, uint64_t> metrics_;
  std::mutex mutex_;
  friend class folly::Singleton<S3MetricsAggregator>;
};

} // namespace facebook::velox::filesystems

#endif // VELOX_CONNECTORS_HIVE_STORAGE_ADAPTERS_S3FS_S3METRICSAGGREGATOR_H_
