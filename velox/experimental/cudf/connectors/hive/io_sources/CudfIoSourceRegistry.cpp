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
  std::string name;
  SourceMatcher matcher;
  SourceGenerator generator;
};

// Guards mutation of the singleton entry vector. Reads happen on the
// hot path but the lock is brief and the registration list is small,
// so a single mutex is sufficient.
std::mutex& registryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<Entry>& registry() {
  static std::vector<Entry> entries;
  return entries;
}

} // namespace

void registerCudfIoSource(
    std::string name,
    SourceMatcher matcher,
    SourceGenerator generator) {
  std::lock_guard<std::mutex> lock(registryMutex());
  auto& entries = registry();
  // Linear scan: the registry holds one entry per backend so N is tiny
  // and a vector keeps lookup order deterministic (first-match wins).
  // Re-registration under the same name is a no-op so per-test-fixture
  // `SetUp` calls are safe, while `unregisterCudfIoSources()` followed
  // by re-registration still works (unlike a `folly::call_once` guard
  // at the call site, whose flag cannot be reset).
  for (const auto& entry : entries) {
    if (entry.name == name) {
      return;
    }
  }
  entries.push_back(
      Entry{std::move(name), std::move(matcher), std::move(generator)});
}

std::shared_ptr<cudf::io::datasource> getCudfIoSource(
    std::string_view path,
    const std::shared_ptr<const config::ConfigBase>& properties) {
  std::lock_guard<std::mutex> lock(registryMutex());
  for (const auto& entry : registry()) {
    if (entry.matcher(path)) {
      return entry.generator(path, properties);
    }
  }
  return nullptr;
}

void unregisterCudfIoSources() {
  std::lock_guard<std::mutex> lock(registryMutex());
  registry().clear();
}

} // namespace facebook::velox::cudf_velox::connector::hive::io_sources
