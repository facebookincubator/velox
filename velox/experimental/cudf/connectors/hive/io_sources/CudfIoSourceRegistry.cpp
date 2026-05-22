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

#include "velox/experimental/cudf/connectors/hive/io_sources/CudfIoSourceRegistry.h"

#include <mutex>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::io_sources {

namespace {

struct Entry {
  CudfIoSourceMatcher matcher;
  CudfIoSourceFactory factory;
};

// Guards mutation of the singleton entry vector and the default-slot
// factory. Reads happen on the hot path but the lock is brief and the
// registration list is small, so a single mutex is sufficient.
std::mutex& registryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<Entry>& registry() {
  static std::vector<Entry> entries;
  return entries;
}

CudfIoSourceFactory& defaultFactory() {
  static CudfIoSourceFactory factory;
  return factory;
}

} // namespace

void registerCudfIoSource(
    CudfIoSourceMatcher matcher,
    CudfIoSourceFactory factory) {
  std::lock_guard<std::mutex> lock(registryMutex());
  registry().push_back(Entry{std::move(matcher), std::move(factory)});
}

void registerCudfDefaultIoSource(CudfIoSourceFactory factory) {
  std::lock_guard<std::mutex> lock(registryMutex());
  defaultFactory() = std::move(factory);
}

std::shared_ptr<cudf::io::datasource> getCudfIoSource(
    std::string_view path,
    const std::shared_ptr<const config::ConfigBase>& properties) {
  std::lock_guard<std::mutex> lock(registryMutex());
  for (const auto& entry : registry()) {
    if (entry.matcher(path)) {
      return entry.factory(path, properties);
    }
  }
  if (defaultFactory()) {
    return defaultFactory()(path, properties);
  }
  return nullptr;
}

void unregisterCudfIoSources() {
  std::lock_guard<std::mutex> lock(registryMutex());
  registry().clear();
  defaultFactory() = nullptr;
}

} // namespace facebook::velox::cudf_velox::connector::hive::io_sources
