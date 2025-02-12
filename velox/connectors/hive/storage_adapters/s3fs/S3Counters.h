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

#include <folly/Range.h>

namespace facebook::velox::filesystems {

void registerS3Metrics();

constexpr folly::StringPiece kCounterS3ActiveConnections{
  "presto_cpp.hive.s3.active_connections"};
constexpr folly::StringPiece kCounterS3StartedUploads{
  "presto_cpp.hive.s3.started_uploads"};
constexpr folly::StringPiece kCounterS3SuccessfulUploads{
  "presto_cpp.hive.s3.successful_uploads"};
constexpr folly::StringPiece kCounterS3FailedUploads{
  "presto_cpp.hive.s3.failed_uploads"};
constexpr folly::StringPiece kCounterS3MetadataCalls{
  "presto_cpp.hive.s3.metadata_calls"};
constexpr folly::StringPiece kCounterS3ListStatusCalls{
  "presto_cpp.hive.s3.list_status_calls"};
constexpr folly::StringPiece kCounterS3ListLocatedStatusCalls{
  "presto_cpp.hive.s3.list_located_status_calls"};
constexpr folly::StringPiece kCounterS3ListObjectsCalls{
  "presto_cpp.hive.s3.list_objects_calls"};
constexpr folly::StringPiece kCounterS3GetObjectErrors{
  "presto_cpp.hive.s3.get_object_errors"};
constexpr folly::StringPiece kCounterS3GetMetadataErrors{
  "presto_cpp.hive.s3.get_metadata_errors"};
constexpr folly::StringPiece kCounterS3GetObjectRetries{
  "presto_cpp.hive.s3.get_object_retries"};
constexpr folly::StringPiece kCounterS3GetMetadataRetries{
  "presto_cpp.hive.s3.get_metadata_retries"};
} // namespace facebook::velox