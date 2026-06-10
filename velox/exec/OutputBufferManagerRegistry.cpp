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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "velox/core/QueryCtx.h"
#include "velox/exec/IOutputBufferManager.h"
#include "velox/exec/OutputBufferManagerRegistryInternal.h"

namespace facebook::velox::exec {

ScopedRegistry<std::string, IOutputBufferManager>& outputBufferManagers() {
  static ScopedRegistry<std::string, IOutputBufferManager> instance;
  return instance;
}

namespace {

OutputBufferManagerRegistry::Registry& registryFor(
    const core::QueryCtx& queryCtx) {
  auto registry = queryCtx.registry<OutputBufferManagerRegistry::Registry>(
      OutputBufferManagerRegistry::kRegistryKey);
  return registry ? *registry : OutputBufferManagerRegistry::global();
}

} // namespace

// static
OutputBufferManagerRegistry::Registry& OutputBufferManagerRegistry::global() {
  return outputBufferManagers();
}

// static
std::shared_ptr<OutputBufferManagerRegistry::Registry>
OutputBufferManagerRegistry::create(const Registry* parent) {
  return std::make_shared<Registry>(parent);
}

// static
std::shared_ptr<IOutputBufferManager> OutputBufferManagerRegistry::tryGet(
    const core::QueryCtx& queryCtx,
    const std::string& id) {
  return registryFor(queryCtx).find(id);
}

// static
std::shared_ptr<IOutputBufferManager> OutputBufferManagerRegistry::tryGet(
    const std::string& id) {
  return global().find(id);
}

// static
std::vector<std::pair<std::string, std::shared_ptr<IOutputBufferManager>>>
OutputBufferManagerRegistry::getAll(const core::QueryCtx& queryCtx) {
  return snapshot(queryCtx);
}

// static
std::vector<std::pair<std::string, std::shared_ptr<IOutputBufferManager>>>
OutputBufferManagerRegistry::getAll() {
  return global().snapshot();
}

// static
void OutputBufferManagerRegistry::unregisterAll(
    const core::QueryCtx& queryCtx) {
  auto registry = queryCtx.registry<OutputBufferManagerRegistry::Registry>(
      OutputBufferManagerRegistry::kRegistryKey);
  if (registry) {
    registry->clear();
  }
}

// static
void OutputBufferManagerRegistry::unregisterAll() {
  global().clear();
}

// static
std::vector<std::pair<std::string, std::shared_ptr<IOutputBufferManager>>>
OutputBufferManagerRegistry::snapshot(const core::QueryCtx& queryCtx) {
  return registryFor(queryCtx).snapshot();
}

} // namespace facebook::velox::exec
