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

#include <google/cloud/credentials.h>

namespace facebook::velox::filesystems {

namespace gcs = ::google::cloud::storage;

/// Interface for providing credentials for Google Cloud Storage (GCS).
/// Implementations should return a `google::cloud::Credentials` used for
/// creating the GCS client for a specific bucket via `getCredentials`.
/// Build the return value with one of the factory functions in
/// `<google/cloud/credentials.h>` such as `MakeAccessTokenCredentials` for
/// bearer-token flows or `MakeServiceAccountCredentials` for service-account
/// JSON. The deprecated `storage::oauth2::Credentials` type was removed in
/// google-cloud-cpp 3.x.
class GcsOAuthCredentialsProvider {
 public:
  virtual ~GcsOAuthCredentialsProvider() = default;

  virtual std::shared_ptr<::google::cloud::Credentials> getCredentials(
      const std::string& bucket) = 0;
};

} // namespace facebook::velox::filesystems
