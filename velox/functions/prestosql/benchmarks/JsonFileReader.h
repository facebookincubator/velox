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

#include <boost/algorithm/string/replace.hpp>
#include <boost/filesystem.hpp>
#include <google/protobuf/util/json_util.h>
#include <fstream>
#include <sstream>
#include "velox/common/base/Exceptions.h"

namespace fs = boost::filesystem;

namespace {

std::string getDataPath(const std::string& fileSize) {
  std::string currentPath = fs::current_path().c_str();

  // CLion runs the tests from cmake-build-release/ or cmake-build-debug/
  // directory. Hard-coded json files are not copied there and test fails with
  // file not found. Fixing the path so that we can trigger these tests from
  // CLion.
  boost::algorithm::replace_all(currentPath, "cmake-build-release/", "");
  boost::algorithm::replace_all(currentPath, "cmake-build-debug/", "");

  return currentPath + "/velox/functions/prestosql/json/tests/data/twitter_" +
      fileSize + ".json";
}

} // namespace

namespace facebook::velox::functions::prestosql {

class JsonFileReader {
 public:
  static std::string readJsonStringFromFile(const std::string& fileSize) {
    // Read json file.
    std::string jsonPath = getDataPath(fileSize);
    std::ifstream jsonIfstream(jsonPath);
    VELOX_CHECK(
        !jsonIfstream.fail(),
        "Failed to open file: {}. {}",
        jsonPath,
        strerror(errno));
    std::stringstream buffer;
    buffer << jsonIfstream.rdbuf();
    return buffer.str();
  }
};

} // namespace facebook::velox::functions::prestosql
