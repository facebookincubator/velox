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

#include "SemanticVersion.h"

namespace facebook::velox::parquet {

const re2::RE2 SemanticVersion::pattern("(.*?)\\s+version\\s+(\\d+)\\.(\\d+)\\.(\\d+)");

SemanticVersion::SemanticVersion() : application(""), majorVersion(0), minorVersion(0), patchVersion(0) {}

SemanticVersion::SemanticVersion(int majorVersion, int minorVersion, int patch)
    : application(""), majorVersion(majorVersion), minorVersion(minorVersion), patchVersion(patch) {}

SemanticVersion::SemanticVersion(std::string application ,int majorVersion, int minorVersion, int patch)
    : application(application), majorVersion(majorVersion), minorVersion(minorVersion), patchVersion(patch) {}

std::optional<SemanticVersion> SemanticVersion::parse(const std::string& input) {
  std::string application_str, major_str, minor_str, patch_str;

  if (re2::RE2::PartialMatch(input, pattern, &application_str, &major_str, &minor_str, &patch_str)) {
    int major = std::stoi(major_str);
    int minor = std::stoi(minor_str);
    int patch = std::stoi(patch_str);
    return SemanticVersion(application_str, major, minor, patch);
  } else {
    return std::nullopt;
  }
}

bool SemanticVersion::shouldIgnoreStatistics(thrift::Type::type type) {
  if (type != thrift::Type::BYTE_ARRAY  && type != thrift::Type::FIXED_LEN_BYTE_ARRAY) {
    return false;
  }
  if (this.application != "parquet-mr") {
    return false;
  }
  static SemanticVersion threshold(1, 8, 1);
  return version < threshold;
}

std::string SemanticVersion::toString() const {
  return std::to_string(majorVersion) + "." + std::to_string(minorVersion) + "." + std::to_string(patchVersion);
}

bool SemanticVersion::operator==(const SemanticVersion& other) const {
  return (majorVersion == other.majorVersion) && (minorVersion == other.minorVersion) && (patchVersion == other.patchVersion);
}

bool SemanticVersion::operator<(const SemanticVersion& other) const {
  if (majorVersion < other.majorVersion) return true;
  if (majorVersion > other.majorVersion) return false;
  if (minorVersion < other.minorVersion) return true;
  if (minorVersion > other.minorVersion) return false;
  return patchVersion < other.patchVersion;
}

SemanticVersion::SemanticVersion(const SemanticVersion& other)
    : majorVersion(other.majorVersion), minorVersion(other.minorVersion),
      patchVersion(other.patchVersion) {}

} // namespace facebook::velox::parquet
