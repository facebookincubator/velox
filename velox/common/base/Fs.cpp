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

#include "velox/common/base/Fs.h"

#include <fmt/format.h>
#include <glog/logging.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <direct.h>
#include <random>

// Define permission constants for Windows
#ifndef S_IRUSR
#define S_IRUSR _S_IREAD
#endif
#ifndef S_IWUSR
#define S_IWUSR _S_IWRITE
#endif

// Windows implementation of mkstemp
inline int mkstemp(char* tmpl) {
  // Find the XXXXXX pattern
  char* xes = strstr(tmpl, "XXXXXX");
  if (!xes || strlen(xes) != 6) {
    return -1;
  }
  
  // Generate random suffix
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 35);
  
  const char* chars = "0123456789abcdefghijklmnopqrstuvwxyz";
  for (int attempt = 0; attempt < 100; ++attempt) {
    for (int i = 0; i < 6; ++i) {
      xes[i] = chars[dis(gen)];
    }
    
    int fd = _open(tmpl, _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
    if (fd != -1) {
      return fd;
    }
  }
  return -1;
}

// Windows implementation of mkdtemp
inline char* mkdtemp(char* tmpl) {
  // Find the XXXXXX pattern
  char* xes = strstr(tmpl, "XXXXXX");
  if (!xes || strlen(xes) != 6) {
    return nullptr;
  }
  
  // Generate random suffix
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 35);
  
  const char* chars = "0123456789abcdefghijklmnopqrstuvwxyz";
  for (int attempt = 0; attempt < 100; ++attempt) {
    for (int i = 0; i < 6; ++i) {
      xes[i] = chars[dis(gen)];
    }
    
    if (_mkdir(tmpl) == 0) {
      return tmpl;
    }
  }
  return nullptr;
}
#endif // _WIN32

namespace facebook::velox::common {

bool generateFileDirectory(const char* dirPath) {
  std::error_code errorCode;
  const auto success = fs::create_directories(dirPath, errorCode);
  fs::permissions(dirPath, fs::perms::all, fs::perm_options::replace);
  if (!success && errorCode.value() != 0) {
    LOG(ERROR) << "Failed to create file directory '" << dirPath
               << "'. Error: " << errorCode.message() << " errno "
               << errorCode.value();
    return false;
  }
  return true;
}

std::optional<std::string> generateTempFilePath(
    const char* basePath,
    const char* prefix) {
  auto path = fmt::format("{}/velox_{}_XXXXXX", basePath, prefix);
  auto fd = ::mkstemp(path.data());
  if (fd == -1) {
    return std::nullopt;
  }
  return path;
}

std::optional<std::string> generateTempFolderPath(
    const char* basePath,
    const char* prefix) {
  auto path = fmt::format("{}/velox_{}_XXXXXX", basePath, prefix);
  auto createdPath = ::mkdtemp(path.data());
  if (createdPath == nullptr) {
    return std::nullopt;
  }
  return path;
}

} // namespace facebook::velox::common
