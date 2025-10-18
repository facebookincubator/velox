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

#include "velox/common/base/ExceptionHelper.h"
#include "velox/experimental/cudf/CudfConfig.h"

namespace facebook::velox::cudf_velox {

namespace detail {
inline const char* extractFileName(const char* file) {
  return strrchr(file, '/') ? strrchr(file, '/') + 1 : file;
}
} // namespace detail

#define LOG_VALIDATION_MSG(...)                                             \
  do {                                                                      \
    if (CudfConfig::getInstance().logValidationFailure) {                          \
      auto message = ::facebook::velox::errorMessage(__VA_ARGS__);          \
      LOG(WARNING) << fmt::format(                                          \
          "Validation failed at file:{}, line:{}, function:{}, reason:{}",  \
          ::facebook::velox::cudf_velox::detail::extractFileName(__FILE__), \
          __LINE__,                                                         \
          __FUNCTION__,                                                     \
          message);                                                         \
    }                                                                       \
  } while (0)
} // namespace facebook::velox::cudf_velox
