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

namespace facebook::velox {

namespace {

// Returns true for status codes treated as transient by the GCS retry policy.
bool isRetryableGcsStatus(google::cloud::StatusCode code) {
  using ::google::cloud::StatusCode;

  switch (code) {
    case StatusCode::kDeadlineExceeded:
    case StatusCode::kInternal:
    case StatusCode::kResourceExhausted:
    case StatusCode::kUnavailable:
      return true;
    default:
      return false;
  }
}

} // namespace

std::string getErrorStringFromGcsError(const google::cloud::StatusCode& code) {
  return google::cloud::StatusCodeToString(code);
}

void checkGcsStatus(
    const google::cloud::Status outcome,
    const std::string_view& errorMsgPrefix,
    const std::string& bucket,
    const std::string& key) {
  if (!outcome.ok()) {
    auto errorMessage = fmt::format(
        "{} due to: Path:'{}', GCS Status Code:{}, Error Domain:'{}', Error Reason:'{}', Message:'{}'",
        errorMsgPrefix,
        gcsURI(bucket, key),
        getErrorStringFromGcsError(outcome.code()),
        outcome.error_info().domain(),
        outcome.error_info().reason(),
        outcome.message());
    if (isRetryableGcsStatus(outcome.code())) {
      errorMessage.append(
          " This GCS error is transient. Consider increasing "
          "'hive.gcs.max-retry-count' or 'hive.gcs.max-retry-time'.");
    }
    if (outcome.code() == google::cloud::StatusCode::kNotFound) {
      VELOX_FILE_NOT_FOUND_ERROR(errorMessage);
    }
    VELOX_FAIL(errorMessage);
  }
}

} // namespace facebook::velox
