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

#include "velox/common/base/StatsReporter.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Counters.h"

namespace facebook::velox::filesystems {

void registerS3Metrics() {
  DEFINE_METRIC(kCounterS3ActiveConnections, velox::StatType::SUM);
  DEFINE_METRIC(kCounterS3StartedUploads, velox::StatType::COUNT);
  DEFINE_METRIC(kCounterS3FailedUploads, velox::StatType::COUNT);
  DEFINE_METRIC(kCounterS3SuccessfulUploads, velox::StatType::COUNT);
  DEFINE_METRIC(kCounterS3MetadataCalls, velox::StatType::COUNT);
  DEFINE_METRIC(kCounterS3ListStatusCalls, velox::StatType::COUNT);
  DEFINE_METRIC(kCounterS3ListLocatedStatusCalls, velox::StatType::COUNT);
  DEFINE_METRIC(kCounterS3GetObjectErrors, velox::StatType::COUNT);
  DEFINE_METRIC(kCounterS3GetMetadataErrors, velox::StatType::COUNT);
  DEFINE_METRIC(kCounterS3GetObjectRetries, velox::StatType::COUNT);
  DEFINE_METRIC(kCounterS3GetMetadataRetries, velox::StatType::COUNT);
}
} // namespace facebook::velox::filesystems