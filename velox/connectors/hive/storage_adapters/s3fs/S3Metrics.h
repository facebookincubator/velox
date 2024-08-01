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

#pragma once

namespace facebook::velox::filesystems {

/// Metric names for S3 FileSystem.
/// These metrics are used for monitoring and reporting various S3 operations.
constexpr auto kMetricS3ActiveConnections =
    "presto_hive_s3_presto_s3_file_system_active_connections_total_count";
constexpr auto kMetricS3StartedUploads =
    "presto_hive_s3_presto_s3_file_system_started_uploads_one_minute_count";
constexpr auto kMetricS3FailedUploads =
    "presto_hive_s3_presto_s3_file_system_failed_uploads_one_minute_count";
constexpr auto kMetricS3SuccessfulUploads =
    "presto_hive_s3_presto_s3_file_system_successful_uploads_one_minute_count";
constexpr auto kMetricS3MetadataCalls =
    "presto_hive_s3_presto_s3_file_system_metadata_calls_one_minute_count";
constexpr auto kMetricS3ListStatusCalls =
    "presto_hive_s3_presto_s3_file_system_list_status_calls_one_minute_count";
constexpr auto kMetricS3ListLocatedStatusCalls =
    "presto_hive_s3_presto_s3_file_system_list_located_status_calls_one_minute_count";
constexpr auto kMetricS3ListObjectsCalls =
    "presto_hive_s3_presto_s3_file_system_list_objects_calls_one_minute_count";
constexpr auto kMetricS3OtherReadErrors =
    "presto_hive_s3_presto_s3_file_system_other_read_errors_one_minute_count";
constexpr auto kMetricS3AwsAbortedExceptions =
    "presto_hive_s3_presto_s3_file_system_aws_aborted_exceptions_one_minute_count";
constexpr auto kMetricS3SocketExceptions =
    "presto_hive_s3_presto_s3_file_system_socket_exceptions_one_minute_count";
constexpr auto kMetricS3GetObjectErrors =
    "presto_hive_s3_presto_s3_file_system_get_object_errors_one_minute_count";
constexpr auto kMetricS3GetMetadataErrors =
    "presto_hive_s3_presto_s3_file_system_get_metadata_errors_one_minute_count";
constexpr auto kMetricS3GetObjectRetries =
    "presto_hive_s3_presto_s3_file_system_get_object_retries_one_minute_count";
constexpr auto kMetricS3GetMetadataRetries =
    "presto_hive_s3_presto_s3_file_system_get_metadata_retries_one_minute_count";
constexpr auto kMetricS3ReadRetries =
    "presto_hive_s3_presto_s3_file_system_read_retries_one_minute_count";

} // namespace facebook::velox::filesystems
