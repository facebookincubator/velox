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

#include "velox/connectors/hive/storage_adapters/gcs/GcsUtil.h"

#include <string>
#include <utility>
#include <vector>

#include "velox/common/base/tests/GTestUtils.h"

#include "gtest/gtest.h"

using namespace facebook::velox;

TEST(GcsUtilTest, isGcsFile) {
  EXPECT_FALSE(isGcsFile("gs:"));
  EXPECT_FALSE(isGcsFile("gs::/bucket"));
  EXPECT_FALSE(isGcsFile("gs:/bucket"));
  EXPECT_TRUE(isGcsFile("gs://bucket/file.txt"));
}

TEST(GcsUtilTest, setBucketAndKeyFromGcsPath) {
  std::string bucket, key;
  auto path = "bucket/file.txt";
  setBucketAndKeyFromGcsPath(path, bucket, key);
  EXPECT_EQ(bucket, "bucket");
  EXPECT_EQ(key, "file.txt");
}

TEST(GcsUtilTest, errorString) {
  using google::cloud::StatusCode;

  const std::vector<std::pair<StatusCode, std::string>> testCases{
      {StatusCode::kOk, "No error"},
      {StatusCode::kCancelled, "Request cancelled"},
      {StatusCode::kUnknown, "Unknown error"},
      {StatusCode::kInvalidArgument, "Invalid argument"},
      {StatusCode::kDeadlineExceeded, "Deadline exceeded"},
      {StatusCode::kNotFound, "Resource not found"},
      {StatusCode::kAlreadyExists, "Resource already exists"},
      {StatusCode::kPermissionDenied, "Access denied"},
      {StatusCode::kResourceExhausted, "Resource exhausted"},
      {StatusCode::kFailedPrecondition, "Failed precondition"},
      {StatusCode::kAborted, "Request aborted"},
      {StatusCode::kOutOfRange, "Out of range"},
      {StatusCode::kUnimplemented, "Operation not implemented"},
      {StatusCode::kInternal, "Internal error"},
      {StatusCode::kUnavailable, "Service unavailable"},
      {StatusCode::kDataLoss, "Data loss"},
      {StatusCode::kUnauthenticated, "Unauthenticated"},
  };

  for (const auto& [statusCode, expectedMessage] : testCases) {
    EXPECT_EQ(getErrorStringFromGcsError(statusCode), expectedMessage);
  }

  EXPECT_EQ(
      getErrorStringFromGcsError(static_cast<StatusCode>(123)),
      "UNEXPECTED_STATUS_CODE=123");
}

TEST(GcsUtilTest, statusError) {
  const google::cloud::Status status{
      google::cloud::StatusCode::kResourceExhausted,
      "Quota exceeded",
      google::cloud::ErrorInfo{
          "RATE_LIMIT_EXCEEDED",
          "storage.googleapis.com",
          {},
      }};

  VELOX_ASSERT_THROW(
      checkGcsStatus(status, "Failed to read GCS object", "bucket", "key"),
      "Failed to read GCS object due to: Path:'gs://bucket/key', GCS Status Code:Resource exhausted, Error Domain:'storage.googleapis.com', Error Reason:'RATE_LIMIT_EXCEEDED', Message:'Quota exceeded' This GCS error is transient. Consider increasing 'hive.gcs.max-retry-count' or 'hive.gcs.max-retry-time'.");
}
