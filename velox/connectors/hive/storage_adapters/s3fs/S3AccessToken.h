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

#include <functional>
#include <string>

#include "velox/common/file/TokenProvider.h"

namespace facebook::velox::filesystems {

/// Key the S3 file system presents to a query's TokenProvider when opening a
/// file: the scheme-stripped path ("bucket/key") being accessed. A provider
/// resolves the credential set covering that path from it, which is what
/// prefix-scoped credentials such as Iceberg REST-catalog vended credentials
/// need.
class S3AccessTokenKey : public AccessTokenKey {
 public:
  explicit S3AccessTokenKey(std::string path) : path_(std::move(path)) {}

  const std::string& path() const {
    return path_;
  }

 private:
  const std::string path_;
};

/// A (possibly STS-temporary) S3 credential set resolved by a TokenProvider.
class S3AccessToken : public AccessToken {
 public:
  S3AccessToken(
      std::string accessKeyId,
      std::string secretAccessKey,
      std::string sessionToken = "")
      : accessKeyId_(std::move(accessKeyId)),
        secretAccessKey_(std::move(secretAccessKey)),
        sessionToken_(std::move(sessionToken)) {}

  const std::string& accessKeyId() const {
    return accessKeyId_;
  }

  const std::string& secretAccessKey() const {
    return secretAccessKey_;
  }

  /// Empty when the credentials are not STS-temporary.
  const std::string& sessionToken() const {
    return sessionToken_;
  }

  /// Stable identity for client caching that does not use the raw secret as a
  /// map key. Access key ids are unique per issued credential set; the hash
  /// suffix keeps sets apart if a deployment reuses an access key id with a
  /// rotated secret.
  std::string fingerprint() const {
    return accessKeyId_ + "-" +
        std::to_string(std::hash<std::string>()(
            secretAccessKey_ + ":" + sessionToken_));
  }

 private:
  const std::string accessKeyId_;
  const std::string secretAccessKey_;
  const std::string sessionToken_;
};

} // namespace facebook::velox::filesystems
