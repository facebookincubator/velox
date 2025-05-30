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

#include "velox4j/init/Config.h"
#include "velox4j/init/Init.h"

namespace facebook::velox4j {
inline void testingEnsureInitializedForSpark() {
  static std::once_flag flag;
  auto conf = std::make_shared<ConfigArray>(
      std::vector<std::pair<std::string, std::string>>{
          {VELOX4J_INIT_PRESET.key, folly::to<std::string>(Preset::SPARK)}});
  std::call_once(flag, [&]() { initialize(conf); });
}
} // namespace facebook::velox4j
