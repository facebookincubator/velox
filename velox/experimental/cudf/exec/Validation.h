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

#include "velox/experimental/cudf/CudfConfig.h"

#include "velox/common/base/ExceptionHelper.h"

namespace facebook::velox::cudf_velox {

/// Log a validation failure message if
/// CudfConfig::getInstance().logValidationFailure is enabled. It captures the
/// file name, line number, function name, and reason for the failure. The
/// validation failure introduces extra data format conversion which can
/// negatively impact the performance, so log the messgage to notify the
/// important reason.
///
/// Usage:
///
/// LOG_VALIDATION_MSG("The Sort merge join is not supported");
/// LOG_VALIDATION_MSG("The expression {} is not supported", name);
#define LOG_VALIDATION_MSG(...)                                                                                              \
  do {                                                                                                                       \
    if (CudfConfig::getInstance().logValidationFailure) {                                                                    \
      auto message = ::facebook::velox::errorMessage(__VA_ARGS__);                                                           \
      LOG(WARNING) << fmt::format(                                                                                           \
          "Validation failed at function: {}, reason: Operation is not supported in cuDF execution: {}", \
          __FUNCTION__,                                                                                                      \
          message);                                                                                                          \
    }                                                                                                                        \
  } while (0)
} // namespace facebook::velox::cudf_velox
