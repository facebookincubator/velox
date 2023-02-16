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

  return currentPath + "/data/twitter_" + fileSize + ".json";
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
