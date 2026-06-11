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
#include "velox/exec/OutputBufferManagerRegistry.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {

folly::Synchronized<
    std::unordered_map<std::string, std::shared_ptr<IOutputBufferManager>>,
    std::mutex>&
OutputBufferManagerRegistry::managers() {
  static folly::Synchronized<
      std::unordered_map<std::string, std::shared_ptr<IOutputBufferManager>>,
      std::mutex>
      instance;
  return instance;
}

void OutputBufferManagerRegistry::registerManager(
    const std::string& name,
    std::shared_ptr<IOutputBufferManager> manager) {
  managers().withLock([&](auto& map) {
    auto result = map.emplace(name, std::move(manager));
    VELOX_CHECK(
        result.second,
        "OutputBufferManager already registered with name: {}",
        name);
  });
}

bool OutputBufferManagerRegistry::unregisterManager(const std::string& name) {
  return managers().withLock([&](auto& map) { return map.erase(name) > 0; });
}

std::shared_ptr<IOutputBufferManager> OutputBufferManagerRegistry::getManager(
    const std::string& name) {
  return managers().withLock(
      [&](auto& map) -> std::shared_ptr<IOutputBufferManager> {
        auto it = map.find(name);
        return it == map.end() ? nullptr : it->second;
      });
}

std::vector<std::shared_ptr<IOutputBufferManager>>
OutputBufferManagerRegistry::getAllManagers() {
  return managers().withLock([](auto& map) {
    std::vector<std::shared_ptr<IOutputBufferManager>> result;
    result.reserve(map.size());
    for (auto& [name, mgr] : map) {
      result.push_back(mgr);
    }
    return result;
  });
}

bool OutputBufferManagerRegistry::hasManager(const std::string& name) {
  return managers().withLock([&](auto& map) { return map.count(name) > 0; });
}

void OutputBufferManagerRegistry::clear() {
  managers().withLock([](auto& map) { map.clear(); });
}

} // namespace facebook::velox::exec
