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
#include <glog/logging.h>

namespace facebook::velox::common {

bool generateFileDirectory(const char* dirPath) {
  std::error_code errorCode;
  auto success = fs::create_directories(dirPath, errorCode);
  if (!success) {
    LOG(ERROR) << "Failed to create file directory '" << dirPath
               << "'. Error: " << errorCode.message() << " errno "
               << errorCode.value();
    return false;
  }
  return true;
}

} // namespace facebook::velox::common
