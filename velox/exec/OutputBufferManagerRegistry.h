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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <folly/Synchronized.h>

#include "velox/exec/IOutputBufferManager.h"

namespace facebook::velox::exec {

class OutputBufferManagerRegistry {
 public:
  static void registerManager(
      const std::string& name,
      std::shared_ptr<IOutputBufferManager> manager);

  static bool unregisterManager(const std::string& name);

  static std::shared_ptr<IOutputBufferManager> getManager(
      const std::string& name);

  template <typename T>
  static std::shared_ptr<T> getManagerAs(const std::string& name) {
    return std::dynamic_pointer_cast<T>(getManager(name));
  }

  static std::vector<std::shared_ptr<IOutputBufferManager>> getAllManagers();

  static bool hasManager(const std::string& name);

  static void clear();

 private:
  static folly::Synchronized<
      std::unordered_map<std::string, std::shared_ptr<IOutputBufferManager>>,
      std::mutex>&
  managers();
};

} // namespace facebook::velox::exec
