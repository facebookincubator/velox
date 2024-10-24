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
#include <google/cloud/storage/client.h>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox {

namespace {
constexpr const char* kSep{"/"};
constexpr std::string_view kGCSScheme{"gs://"};

} // namespace

static std::string_view kLoremIpsum =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor"
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis "
    "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu"
    "fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in"
    "culpa qui officia deserunt mollit anim id est laborum.";

std::string getErrorStringFromGCSError(const google::cloud::StatusCode& error);

inline bool isGCSFile(const std::string_view filename) {
  return (filename.substr(0, kGCSScheme.size()) == kGCSScheme);
}

inline void setBucketAndKeyFromGCSPath(
    const std::string& path,
    std::string& bucket,
    std::string& key) {
  auto firstSep = path.find_first_of(kSep);
  bucket = path.substr(0, firstSep);
  key = path.substr(firstSep + 1);
}

inline std::string gcsURI(std::string_view bucket) {
  std::stringstream ss;
  ss << kGCSScheme << bucket;
  return ss.str();
}

inline std::string gcsURI(std::string_view bucket, std::string_view key) {
  std::stringstream ss;
  ss << kGCSScheme << bucket << kSep << key;
  return ss.str();
}

inline std::string gcsPath(const std::string_view& path) {
  // Remove the prefix gcs:// from the given path
  return std::string(path.substr(kGCSScheme.length()));
}

} // namespace facebook::velox
