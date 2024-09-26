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
    constexpr auto kMetricS3ActiveConnections = "S3ActiveConnections";
    constexpr auto kMetricS3MetadataCalls = "S3MetadataCalls";
    constexpr auto kMetricS3ListStatusCalls = "S3ListStatusCalls";
    constexpr auto kMetricS3ListLocatedStatusCalls = "S3ListLocatedStatusCalls";
    constexpr auto kMetricS3ListObjectsCalls = "S3ListObjectsCalls";
    constexpr auto kMetricS3OtherReadErrors = "S3OtherReadErrors";
    constexpr auto kMetricS3AwsAbortedExceptions = "S3AwsAbortedExceptions";
    constexpr auto kMetricS3SocketExceptions = "S3SocketExceptions";
    constexpr auto kMetricS3GetObjectErrors = "S3GetObjectErrors";
    constexpr auto kMetricS3GetMetadataErrors = "S3GetMetadataErrors";
    constexpr auto kMetricS3GetObjectRetries = "S3GetObjectRetries";
    constexpr auto kMetricS3GetMetadataRetries = "S3GetMetadataRetries";
    constexpr auto kMetricS3ReadRetries = "S3ReadRetries";
    constexpr auto kMetricS3StartedUploads = "S3StartedUploads";
    constexpr auto kMetricS3FailedUploads = "S3FailedUploads";
    constexpr auto kMetricS3SuccessfulUploads = "S3SuccessfulUploads";
}
