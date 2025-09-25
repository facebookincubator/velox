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

#include <string>

namespace facebook::velox::cudf_velox {

struct CudfConfig {
  /// Singleton CudfConfig instance.
  /// Clients must set the configs below before invoking registerCudf().
  static CudfConfig& getInstance();

  /// Enable debug printing.
  bool debugEnabled{false};
  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};
  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;
  /// The initial percent of GPU memory to allocate for memory resource for one
  /// thread.
  int memoryPercent{50};
  /// Force replacement of operators. Throws an error if a replacement fails.
  bool forceReplace{false};
};

} // namespace facebook::velox::cudf_velox
