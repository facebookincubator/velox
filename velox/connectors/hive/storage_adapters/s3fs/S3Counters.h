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

constexpr std::string_view kMetricS3ActiveConnections{
    "presto_cpp.hive.s3.active_connections"};
constexpr std::string_view kMetricS3StartedUploads{
    "presto_cpp.hive.s3.started_uploads"};
constexpr std::string_view kMetricS3SuccessfulUploads{
    "presto_cpp.hive.s3.successful_uploads"};
constexpr std::string_view kMetricS3FailedUploads{
    "presto_cpp.hive.s3.failed_uploads"};
constexpr std::string_view kMetricS3MetadataCalls{
    "presto_cpp.hive.s3.metadata_calls"};
constexpr std::string_view kMetricS3GetObjectErrors{
    "presto_cpp.hive.s3.get_object_errors"};
constexpr std::string_view kMetricS3GetMetadataErrors{
    "presto_cpp.hive.s3.get_metadata_errors"};
constexpr std::string_view kMetricS3GetObjectRetries{
    "presto_cpp.hive.s3.get_object_retries"};
constexpr std::string_view kMetricS3GetMetadataRetries{
    "presto_cpp.hive.s3.get_metadata_retries"};

} // namespace facebook::velox::filesystems
